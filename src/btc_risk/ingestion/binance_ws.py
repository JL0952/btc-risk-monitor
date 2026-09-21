"""Closed-kline adapter and a bounded handoff queue; calculations live elsewhere."""

from contextlib import contextmanager
import json
import logging
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

from websockets.sync.client import connect

from btc_risk.ingestion.binance_rest import DataQualityError, from_milliseconds, parse_kline

logger = logging.getLogger(__name__)


def parse_message(message, symbol, interval):
    try:
        payload = json.loads(message)
        if not isinstance(payload, dict):
            raise DataQualityError("WebSocket payload must be an object")
        if payload.get("e") == "serverShutdown":
            raise ConnectionError("Binance announced server shutdown")
        if payload.get("e") != "kline" or payload.get("s") != symbol:
            raise DataQualityError("Unexpected event/symbol")
        kline = payload["k"]
        if kline["s"] != symbol or kline["i"] != interval or type(kline["x"]) is not bool:
            raise DataQualityError("Unexpected kline symbol/interval/closed flag")
        if not kline["x"]:
            return None
        # Reuse the REST adapter's normalization and strict OHLCV/time checks.
        row = [kline[key] for key in ("t", "o", "h", "l", "c", "v", "T", "q", "n", "V", "Q", "B")]
        result = parse_kline(row, symbol, interval, from_milliseconds(payload["E"]))
        if result is None:
            raise DataQualityError("Closed flag precedes candle end")
        return result
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, DataQualityError):
            raise
        raise DataQualityError(f"Malformed WebSocket message: {exc}") from exc


class ClosedBarBuffer:
    def __init__(self, size):
        self.queue = Queue(maxsize=size)
        self.error = None

    def add(self, bar):
        try:
            self.queue.put_nowait(bar)
        except Full as exc:
            self.error = ConnectionError("Closed-bar buffer overflow; reconnect/reconcile required")
            raise self.error from exc

    def get(self, timeout=1):
        if self.error:
            raise self.error
        return self.queue.get(timeout=timeout)

    def drain(self):
        if self.error:
            raise self.error
        bars = []
        while True:
            try:
                bars.append(self.queue.get_nowait())
            except Empty:
                return sorted(bars, key=lambda bar: bar.timestamp)


@contextmanager
def closed_stream(symbol, interval, config, stop):
    url = f"{config.ws_url}/{symbol.lower()}@kline_{interval}"
    buffer = ClosedBarBuffer(config.queue_size)
    reader_stop = Event()
    with connect(url, open_timeout=15, close_timeout=3, ping_interval=20, ping_timeout=20,
                 max_size=65536, max_queue=16) as websocket:
        logger.info("WebSocket connected endpoint=%s", url)

        def reader():
            last_message = time.monotonic()
            try:
                while not stop.is_set() and not reader_stop.is_set():
                    try:
                        message = websocket.recv(timeout=1)
                    except TimeoutError:
                        if time.monotonic() - last_message > config.idle_timeout:
                            raise TimeoutError("WebSocket application-message timeout")
                        continue
                    last_message = time.monotonic()
                    try:
                        bar = parse_message(message, symbol, interval)
                    except DataQualityError:
                        # A bad message is not written. A subsequent timestamp jump
                        # invokes recovery; reconnect also runs a full reconciliation.
                        logger.exception("Malformed message; reconnect for reconciliation")
                        raise
                    if bar is not None:
                        logger.info("Closed bar received timestamp=%s close=%s", bar.timestamp, bar.close)
                        buffer.add(bar)
            except Exception as exc:
                buffer.error = exc
            finally:
                if not stop.is_set() and not reader_stop.is_set():
                    logger.warning("WebSocket disconnected: %s", buffer.error)

        thread = Thread(target=reader, name="binance-closed-bars", daemon=True)
        thread.start()
        try:
            yield buffer
        finally:
            reader_stop.set()
            websocket.close()
            thread.join(timeout=5)
