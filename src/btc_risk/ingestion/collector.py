"""Live collection with atomic bar/signal commits and REST reconciliation."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta
import logging
from queue import Empty
import random
import signal
from threading import Event
import time

import psycopg

from btc_risk.config import CollectorConfig, DetectorConfig, IngestionConfig
from btc_risk.database.connection import connect
from btc_risk.database.repository import insert_bar, insert_signal, inspect_range
from btc_risk.ingestion.binance_rest import BinanceREST, DataQualityError, EPOCH, parse_kline
from btc_risk.ingestion.binance_ws import closed_stream
from btc_risk.online.shadow import LiveIFShadow
from btc_risk.realtime.robust_zscore import Observation, RobustZScore

logger = logging.getLogger(__name__)


class CollectorEngine:
    def __init__(self, symbol, interval, rest, config=None, connection_factory=connect, stop=None):
        self.symbol, self.interval, self.rest = symbol, interval, rest
        self.config = config or DetectorConfig.from_env()
        self.connect = connection_factory
        self.stop = stop or Event()
        self.detector = RobustZScore(symbol, interval, self.config)
        self.step = self.detector.step
        self.live_count = self.recovery_count = 0
        self.shadow = None
        self._shadow_loaded = False
        self.check_lease = lambda: None

    def refresh_active_shadow(self):
        """Load the last fully published IF artifact, never train in the collector."""
        try:
            with self.connect() as conn:
                shadow = LiveIFShadow.load_active(conn, self.symbol, self.interval)
        except ValueError:
            # Metadata without a valid, complete artifact must never block
            # Z-score collection or lead to inference from a partial model.
            logger.exception("Active IF artifact invalid; continuing with Z-score only")
            shadow = None
        prior = self.shadow.publication_id if self.shadow is not None else None
        self.shadow = shadow
        current = shadow.publication_id if shadow is not None else None
        if current != prior or not self._shadow_loaded:
            if shadow is None:
                logger.warning("No active IF publication; continuing with Z-score only")
            else:
                logger.info("Active IF publication loaded publication=%s available_at=%s",
                            shadow.publication_id, shadow.model_available_at)
        self._shadow_loaded = True

    @contextmanager
    def lease(self):
        with self.connect() as guard:
            guard.autocommit = True
            locked = guard.execute("SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked",
                                   (f"collector:{self.symbol}:{self.interval}",)).fetchone()["locked"]
            if not locked:
                raise RuntimeError("Another collector already owns this symbol/interval")
            self.check_lease = lambda: guard.execute("SELECT 1")
            try:
                yield
            finally:
                self.check_lease = lambda: None

    def restore_before(self, timestamp):
        with self.connect() as conn:
            rows = conn.execute("""
                SELECT timestamp, symbol, interval, close FROM market_bars
                WHERE symbol=%s AND interval=%s AND source='binance' AND timestamp < %s
                ORDER BY timestamp DESC LIMIT %s
            """, (self.symbol, self.interval, timestamp, self.config.window + 1)).fetchall()
        detector = RobustZScore(self.symbol, self.interval, self.config)
        for row in reversed(rows):
            detector.process(Observation(**row))
        self.detector = detector
        logger.info("Detector state restored historical_bars=%s returns=%s latest=%s",
                    len(rows), len(detector.returns), detector.previous.timestamp if detector.previous else None)

    def apply(self, bar, mode):
        self.check_lease()
        previous = self.detector.previous
        duplicate = previous is not None and bar.timestamp <= previous.timestamp
        if previous and not duplicate and bar.timestamp != previous.timestamp + self.step:
            raise DataQualityError(f"Unresolved gap: {previous.timestamp} -> {bar.timestamp}")
        # Work on a candidate. Only replace authoritative state AFTER COMMIT.
        candidate = deepcopy(self.detector)
        result = shadow_result = None
        with self.connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"realtime:{self.symbol}:{self.interval}",))
            inserted = insert_bar(conn, bar)  # validates duplicates without overwriting
            if duplicate:
                stored = conn.execute("""
                    SELECT 1 FROM realtime_signals WHERE symbol=%s AND interval=%s AND timestamp=%s
                """, (self.symbol, self.interval, bar.timestamp)).fetchone()
                if inserted or stored is None:
                    raise DataQualityError("Old timestamp lacks bar/signal; full reconciliation required")
            else:
                result = candidate.process(Observation(bar.timestamp, bar.symbol, bar.interval, bar.close),
                                           save=lambda value: insert_signal(conn, value))
                shadow_result = self.shadow.score(conn, bar) if self.shadow is not None else None
        if duplicate:
            logger.info("Duplicate bar skipped timestamp=%s", bar.timestamp)
            return None
        self.detector = candidate
        if mode == "live":
            self.live_count += 1
        else:
            self.recovery_count += 1
        logger.info("Bar+signal committed mode=%s timestamp=%s close=%s return=%s z=%s status=%s alert=%s",
                    mode, bar.timestamp, bar.close, result.log_return, result.robust_zscore, result.status, result.alert_flag)
        if result.alert_flag:
            logger.warning("Z-score alert triggered mode=%s timestamp=%s close=%s return=%s robust_zscore=%s",
                           mode, bar.timestamp, bar.close, result.log_return, result.robust_zscore)
        if shadow_result is not None:
            logger.info("IF shadow score mode=%s timestamp=%s publication=%s score=%s anomaly=%s",
                        mode, bar.timestamp, self.shadow.publication_id, shadow_result["anomaly_score"],
                        shadow_result["anomaly_flag"])
        return result

    def recover(self, start, end, as_of):
        if start >= end:
            return
        logger.info("REST recovery started start=%s end=%s", start, end)
        expected = start
        for page_start, page_end, rows in self.rest.pages(self.symbol, self.interval, start, end):
            bars = [parse_kline(row, self.symbol, self.interval, as_of) for row in rows]
            for bar in sorted((bar for bar in bars if bar is not None), key=lambda item: item.timestamp):
                if self.stop.is_set():
                    raise InterruptedError("Collector stopping during REST recovery")
                if not page_start <= bar.timestamp < page_end or bar.timestamp != expected:
                    raise DataQualityError(f"REST recovery incomplete/duplicate: expected {expected}, got {bar.timestamp}")
                self.apply(bar, "recovery")
                expected += self.step
        if expected != end:
            raise DataQualityError(f"REST recovery incomplete: expected end {end}, reached {expected}")
        logger.info("REST recovery completed bars=%s through=%s", (end - start) // self.step, end - self.step)

    def reconcile(self, bootstrap_days=2):
        as_of = self.rest.server_time()
        cutoff = EPOCH + ((as_of - EPOCH) // self.step) * self.step
        with self.connect() as conn:
            bounds = conn.execute("""
                SELECT min(timestamp) AS first, max(timestamp) AS latest FROM market_bars
                WHERE symbol=%s AND interval=%s AND source='binance'
            """, (self.symbol, self.interval)).fetchone()
            logger.info("Database connected latest Binance bar=%s", bounds["latest"])
            if bounds["latest"] is not None and bounds["latest"] >= cutoff:
                raise DataQualityError("Database contains Binance candles not yet closed")
            if bounds["first"] is None:
                start = cutoff - timedelta(days=bootstrap_days)
            else:
                coverage = inspect_range(conn, self.symbol, self.interval, bounds["first"], cutoff, self.step)
                missing_signal = conn.execute("""
                    SELECT min(b.timestamp) AS timestamp FROM market_bars b
                    LEFT JOIN realtime_signals s USING (symbol, interval, timestamp)
                    WHERE b.symbol=%s AND b.interval=%s AND b.source='binance' AND s.timestamp IS NULL
                """, (self.symbol, self.interval)).fetchone()["timestamp"]
                start = min([cutoff] + coverage["missing_timestamps"] + ([missing_signal] if missing_signal else []))
                if coverage["gap_count"]:
                    logger.warning("Gap detected missing_bars=%s first=%s", coverage["gap_count"], start)
                if missing_signal:
                    logger.warning("Missing signal detected first=%s", missing_signal)
        self.restore_before(start)
        self.recover(start, cutoff, as_of)
        logger.info("Reconciliation complete latest=%s rolling_returns=%s",
                    self.detector.previous.timestamp if self.detector.previous else None, len(self.detector.returns))

    def on_live_bar(self, bar):
        # A scheduler may activate a new artifact while the websocket stays up.
        # Refresh before the bar is committed/scored; recovery starts with the
        # equivalent refresh in run_loop.
        self.refresh_active_shadow()
        previous = self.detector.previous
        if previous and bar.timestamp > previous.timestamp + self.step:
            logger.warning("Gap detected before live bar previous=%s received=%s", previous.timestamp, bar.timestamp)
            self.recover(previous.timestamp + self.step, bar.timestamp, self.rest.server_time())
        return self.apply(bar, "live")


def run_loop(engine, config, stop, stream_factory=closed_stream):
    failures = 0
    while not stop.is_set():
        started = time.monotonic()
        try:
            # Reader starts BEFORE REST so closed candles during reconciliation queue up.
            with engine.lease(), stream_factory(engine.symbol, engine.interval, config, stop) as buffer:
                engine.refresh_active_shadow()
                engine.reconcile(config.bootstrap_days)
                for bar in buffer.drain():
                    if stop.is_set():
                        break
                    engine.on_live_bar(bar)
                logger.info("REST/WebSocket handoff complete; live streaming")
                while not stop.is_set():
                    engine.check_lease()
                    try:
                        bar = buffer.get(timeout=1)
                    except Empty:
                        continue
                    engine.on_live_bar(bar)
        except psycopg.Error:
            if stop.is_set():
                break
            logger.exception("Database error; reconnect and rebuild detector from committed rows")
        except Exception:
            if stop.is_set():
                break
            logger.exception("Collector connection/reconciliation failed; discarding memory and reconnecting")
        if stop.is_set():
            break
        if time.monotonic() - started >= 60:
            failures = 0
        delay = min(config.backoff_max, 2 ** min(failures, 16))
        delay = min(config.backoff_max, delay * random.uniform(1, 1.2))
        failures += 1
        logger.warning("Reconnect attempt=%s delay=%.2fs; will reconcile again", failures, delay)
        stop.wait(delay)


def main():
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    ingestion = IngestionConfig.from_env()
    config = CollectorConfig.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=ingestion.symbol)
    parser.add_argument("--interval", default=ingestion.interval, choices=("5m", "1h"))
    args = parser.parse_args()
    stop = Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    engine = CollectorEngine(args.symbol, args.interval, BinanceREST(ingestion), stop=stop)
    logger.info("Collector started symbol=%s interval=%s", args.symbol, args.interval)
    try:
        run_loop(engine, config, stop)
        return 0
    except psycopg.Error:
        logger.exception("Database error")
        return 1
    except Exception:
        logger.exception("Collector failed")
        return 1
    finally:
        logger.info("Collector shutdown live_bars=%s recovery_bars=%s", engine.live_count, engine.recovery_count)


if __name__ == "__main__":
    raise SystemExit(main())
