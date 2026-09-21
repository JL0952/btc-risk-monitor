"""Forward labels only; production modules must not import this module."""

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=5)
LABEL_DEFINITIONS = {
    "horizons_minutes": [30, 60],
    "rv": "sqrt(sum_{k=1..h} log(close[t+k]/close[t+k-1])^2); excludes current return",
    "maximum_adverse_move": "max(0, 1 - min(low[t+1..t+h])/close[t]); positive long-position loss fraction",
    "absolute_return": "abs(log(close[t+h]/close[t]))",
    "complete_horizon": "all h consecutive future 5m candles required; no filling",
    "timestamps": "bar opening UTC; outcomes begin after this bar's close",
}


def future_labels(bars):
    if bars.empty or bars.index.tz is None:
        raise ValueError("Need nonempty timezone-aware market data")
    data = bars.sort_index().copy()
    data.index = data.index.tz_convert("UTC")
    if data.index.has_duplicates or (data.index.asi8 % STEP.value != 0).any():
        raise ValueError("Duplicate or unaligned timestamps")
    if not np.isfinite(data[["close", "low"]].astype(float)).all().all() or (data[["close", "low"]].astype(float) <= 0).any().any():
        raise ValueError("Invalid close/low")
    original_index = data.index
    # Reindex the grid WITHOUT filling. shift(k) now means exactly k*5 minutes.
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq=STEP))
    close, low = data.close.astype(float), data.low.astype(float)
    frames = []
    for minutes in LABEL_DEFINITIONS["horizons_minutes"]:
        h = minutes // 5
        valid = close.notna()
        rv_squared = pd.Series(0.0, index=data.index)
        future_low = pd.Series(np.inf, index=data.index)
        for k in range(1, h + 1):
            next_close, next_low = close.shift(-k), low.shift(-k)
            valid &= next_close.notna() & next_low.notna()
            rv_squared += np.log(next_close / close.shift(-(k - 1))).pow(2)
            future_low = np.minimum(future_low, next_low)
        status = pd.Series("available", index=data.index)
        status.loc[~valid] = "gap"
        status.loc[data.index + h * STEP > data.index.max()] = "incomplete_horizon"
        result = pd.DataFrame({
            "horizon_minutes": minutes, "label_status": status,
            "future_realized_volatility": np.sqrt(rv_squared).where(valid),
            "future_maximum_adverse_move": (1 - future_low / close).clip(lower=0).where(valid),
            "future_absolute_return": np.log(close.shift(-h) / close).abs().where(valid),
        })
        result.index.name = "timestamp"
        frames.append(result.loc[original_index].rename_axis("timestamp").reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "horizon_minutes"]).reset_index(drop=True)


def classify(z_alert, if_alert):
    if pd.isna(z_alert) or pd.isna(if_alert):
        raise ValueError("Unavailable detector result is not normal")
    return {(False, False): "A", (True, False): "B", (False, True): "C", (True, True): "D"}[(bool(z_alert), bool(if_alert))]
