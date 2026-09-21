from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import math
from statistics import median

import pytest

from btc_risk.config import DetectorConfig
from btc_risk.database.repository import insert_bar
from btc_risk.realtime.replay import replay
from btc_risk.realtime.robust_zscore import Observation, RobustZScore


def observations(bar, returns):
    close = Decimal("100")
    result = [Observation(bar.timestamp, bar.symbol, "5m", close)]
    for i, value in enumerate(returns, 1):
        close *= Decimal(str(math.exp(value)))
        result.append(Observation(bar.timestamp + timedelta(minutes=5 * i), bar.symbol, "5m", close))
    return result


def run(items, config=None):
    detector = RobustZScore(items[0].symbol, "5m", config)
    return [detector.process(item) for item in items]


def test_log_return_and_first_bar(bar):
    items = observations(bar, [0.02])
    signals = run(items)
    assert signals[0].status == "no_previous_close"
    assert signals[0].log_return is None
    assert signals[0].alert_flag is None
    assert signals[1].log_return == pytest.approx(math.log(float(items[1].close / items[0].close)))


def test_exact_288_prior_returns_and_bounded_window(bar):
    items = observations(bar, [0.0001 * math.sin(i) for i in range(288)] + [0.1, -0.1])
    detector = RobustZScore(bar.symbol, "5m")
    signals = [detector.process(item) for item in items[:289]]
    assert len(signals) == 289
    assert all(s.status == "warmup" and s.robust_zscore is None and s.alert_flag is None for s in signals[1:])
    prior = [s.log_return for s in signals[1:]]
    expected_median = median(prior)
    expected_mad = median(abs(r - expected_median) for r in prior)
    saved = []

    def save(signal):
        assert list(detector.returns) == prior  # before current return enters state
        saved.append(signal)

    first = detector.process(items[289], save=save)
    assert first.rolling_median == expected_median
    assert first.rolling_mad == expected_mad
    assert first.robust_zscore == (first.log_return - expected_median) / (1.4826 * expected_mad)
    assert first.timestamp == bar.timestamp + timedelta(minutes=5 * 289)
    assert saved == [first]
    detector.process(items[290])
    assert detector.returns.maxlen == len(detector.returns) == 288


@pytest.mark.parametrize("baseline", [0, 1e-12])
def test_zero_and_near_zero_mad(bar, baseline):
    signals = run(observations(bar, [baseline, -baseline, baseline, 0.01]), DetectorConfig(window=3))
    last = signals[-1]
    assert last.status == "scale_floored"
    assert math.isfinite(last.robust_zscore)
    assert last.robust_zscore == (last.log_return - last.rolling_median) / 1e-8


@pytest.mark.parametrize("shock", [0.02, -0.02])
def test_both_signs_and_inclusive_threshold(bar, shock):
    items = observations(bar, [-0.001, 0, 0.001, shock])
    config = DetectorConfig(window=3)
    last = run(items, config)[-1]
    assert last.alert_flag
    assert last.robust_zscore * shock > 0
    boundary = abs(last.robust_zscore)
    assert run(items, replace(config, threshold=boundary))[-1].alert_flag is True
    assert run(items, replace(config, threshold=boundary + 0.001))[-1].alert_flag is False


def test_small_normal_return_not_alert(bar):
    assert run(observations(bar, [-0.001, 0, 0.001, 0.0001]), DetectorConfig(window=3))[-1].alert_flag is False


def test_gap_resets_without_multiperiod_return(bar):
    items = observations(bar, [0.001] * 8)
    detector = RobustZScore(bar.symbol, "5m", DetectorConfig(window=2))
    for item in items[:4]:
        detector.process(item)
    gap = detector.process(items[5])  # missing index 4
    assert gap.status == "gap"
    assert gap.log_return is gap.robust_zscore is gap.alert_flag is None
    assert len(detector.returns) == 0
    after = detector.process(items[6])
    assert after.status == "warmup"
    assert after.log_return == pytest.approx(math.log(float(items[6].close / items[5].close)))
    assert detector.process(items[7]).status == "warmup"
    assert detector.process(items[8]).robust_zscore is not None


def test_reject_duplicate_out_of_order_and_invalid_without_mutation(bar):
    items = observations(bar, [0.001, 0.002])
    detector = RobustZScore(bar.symbol, "5m")
    detector.process(items[0])
    detector.process(items[1])
    for invalid in [items[0], items[1], replace(items[2], close=Decimal("NaN")),
                    replace(items[2], timestamp=items[2].timestamp + timedelta(seconds=1))]:
        with pytest.raises(ValueError):
            detector.process(invalid)
        assert detector.previous == items[1]
        assert len(detector.returns) == 1


def test_save_failure_does_not_advance_state(bar):
    items = observations(bar, [0.01])
    detector = RobustZScore(bar.symbol, "5m")
    detector.process(items[0])

    def failing_save(_):
        raise RuntimeError("simulated database failure")

    with pytest.raises(RuntimeError):
        detector.process(items[1], save=failing_save)
    assert detector.previous == items[0]
    assert not detector.returns
    assert detector.process(items[1]) == run(items)[1]


def test_future_changes_cannot_change_past_scores(bar):
    items = observations(bar, [0.001 * math.sin(i * 0.7) for i in range(400)])
    original = run(items)
    changed = [replace(item, close=item.close * 3) if i >= 330 else item for i, item in enumerate(items)]
    modified = run(changed)
    assert original[:330] == modified[:330]  # compares returns, median, MAD, z, flag, status
    assert original[330].robust_zscore != modified[330].robust_zscore


@pytest.fixture
def stored_sequence(conn, bar):
    # Deliberately insert market rows out of time order and include a synthetic row.
    for i in [5, 2, 0, 4, 1, 3]:
        price = Decimal("90000") + i * 10
        insert_bar(conn, replace(bar, timestamp=bar.timestamp + timedelta(minutes=5 * i),
                               open=price, high=price, low=price, close=price, source="binance"))
    insert_bar(conn, replace(bar, timestamp=bar.timestamp - timedelta(minutes=5)))
    return bar.timestamp - timedelta(minutes=5), bar.timestamp + timedelta(minutes=30)


def test_sql_replay_order_source_filter_and_idempotency(conn, bar, stored_sequence):
    start, end = stored_sequence
    config = DetectorConfig(window=2)
    first = replay(conn, bar.symbol, "5m", start, end, config, fetch_size=2)
    original = conn.execute("SELECT * FROM realtime_signals WHERE symbol = %s ORDER BY timestamp", (bar.symbol,)).fetchall()
    second = replay(conn, bar.symbol, "5m", start, end, config, fetch_size=1)
    assert first["inserted"] == 6
    assert first["valid_returns"] == 5
    assert first["scored_observations"] == 3
    assert first["missing_bars"] == 1  # excluded synthetic observation
    assert second["inserted"] == 0
    assert second["duplicates_skipped"] == 6
    assert original == conn.execute("SELECT * FROM realtime_signals WHERE symbol = %s ORDER BY timestamp", (bar.symbol,)).fetchall()
    assert original[0]["log_return"] is None
    assert original[1]["log_return"] == math.log(90010 / 90000)


def test_changed_config_conflicts_and_preserves_existing(conn, bar, stored_sequence):
    start, end = stored_sequence
    replay(conn, bar.symbol, "5m", start, end, DetectorConfig(window=2))
    original = conn.execute("SELECT * FROM realtime_signals WHERE symbol = %s ORDER BY timestamp", (bar.symbol,)).fetchall()
    with pytest.raises(ValueError, match="Signal inconsistency"):
        with conn.transaction():
            replay(conn, bar.symbol, "5m", start, end, DetectorConfig(window=3))
    assert original == conn.execute("SELECT * FROM realtime_signals WHERE symbol = %s ORDER BY timestamp", (bar.symbol,)).fetchall()


def test_late_inconsistency_rolls_back_earlier_new_signals(conn, bar, stored_sequence):
    start, end = stored_sequence
    conflict_time = bar.timestamp + timedelta(minutes=15)
    conn.execute("""
        INSERT INTO realtime_signals (timestamp, symbol, interval, status)
        VALUES (%s, %s, %s, %s)
    """, (conflict_time, bar.symbol, "5m", "deliberately_inconsistent_test"))
    with pytest.raises(ValueError, match="Signal inconsistency"):
        with conn.transaction():
            replay(conn, bar.symbol, "5m", start, end, DetectorConfig(window=2))
    rows = conn.execute("SELECT timestamp, status FROM realtime_signals WHERE symbol = %s", (bar.symbol,)).fetchall()
    assert rows == [{"timestamp": conflict_time, "status": "deliberately_inconsistent_test"}]
