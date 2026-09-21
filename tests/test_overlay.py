from pathlib import Path

import numpy as np
import pandas as pd

from btc_risk.research.overlay import CONFIG, STRATEGIES, metrics, overlay_path


def frame(length=30, alerts=(), if_alerts=(), unavailable_if=()):
    index = pd.date_range("2026-08-15", periods=length, freq="5min", tz="UTC")
    close = 100 * np.cumprod(1 + np.full(length, .001))
    result = pd.DataFrame({"timestamp": index, "close": close, "z_alert": False,
                           "if_alert": False, "if_available": True})
    result.loc[list(alerts), "z_alert"] = True
    result.loc[list(if_alerts), "if_alert"] = True
    result.loc[list(unavailable_if), "if_available"] = False
    return result


def test_signal_bar_uses_old_exposure_and_next_twelve_are_reduced():
    path = overlay_path(frame(alerts=(2,)), "z")
    assert path.exposure.iloc[2] == 1.0  # signal at close 2 cannot alter 2->3 return
    np.testing.assert_array_equal(path.exposure.iloc[3:15], np.full(12, .5))
    assert path.exposure.iloc[15] == 1.0


def test_later_alert_extends_not_stacks_risk_window():
    path = overlay_path(frame(alerts=(2, 5)), "z")
    np.testing.assert_array_equal(path.exposure.iloc[3:18], np.full(15, .5))
    assert path.exposure.iloc[18] == 1.0
    assert path.exposure.between(.5, 1).all()


def test_turnover_cost_and_gross_net_relationship():
    path = overlay_path(frame(alerts=(2,)), "z")
    changes = path.loc[path.turnover > 0]
    assert len(changes) == 2
    assert changes.turnover.tolist() == [.5, .5]
    assert changes.cost.tolist() == [.00025, .00025]
    np.testing.assert_allclose(path.net_return, path.gross_return - path.cost)
    value = metrics(path)
    assert value["turnover"] == 1.0 and value["transaction_cost"] == .0005
    assert value["total_return_net"] < value["total_return_gross"]


def test_future_alert_cannot_change_past_exposure():
    first = overlay_path(frame(alerts=(12,)), "z")
    second = overlay_path(frame(alerts=(12, 20)), "z")
    pd.testing.assert_series_equal(first.exposure.iloc[:21], second.exposure.iloc[:21])


def test_unavailable_actionable_if_never_triggers_overlay():
    path = overlay_path(frame(if_alerts=(2,), unavailable_if=(2,)), "if")
    assert (path.exposure == 1).all()


def test_overlay_module_rejects_retrospective_if_source():
    source = Path("src/btc_risk/research/overlay.py").read_text()
    assert "isolation_forest_scores" not in source.replace("online_isolation_forest_scores", "")
    assert STRATEGIES == {"buy_hold":"none", "z_overlay":"z", "if_overlay":"if", "union_overlay":"union"}
    assert CONFIG["reduced_exposure"] == .5 and CONFIG["hold_bars"] == 12
