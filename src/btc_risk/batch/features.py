"""Five causal features; rolling state uses only completed current/past bars."""

import numpy as np
import pandas as pd

from btc_risk.config import BatchConfig

FEATURES = ["log_return", "rolling_volatility", "volume_zscore", "high_low_range", "volume_change"]


def build_features(bars: pd.DataFrame, config: BatchConfig, step: pd.Timedelta) -> pd.DataFrame:
    data = bars.copy()
    if data.index.tz is None:
        raise ValueError("Feature index must have a timezone")
    data.index = data.index.tz_convert("UTC")
    data = data.sort_index()
    if data.index.has_duplicates or (len(data) > 1 and not (data.index.to_series().diff().iloc[1:] == step).all()):
        raise ValueError("Duplicate or missing market timestamps")
    if (data.index.asi8 % step.value != 0).any():
        raise ValueError("Unaligned market timestamp")
    values = data[["open", "high", "low", "close", "volume"]].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values.iloc[:, :4] <= 0).any().any() or (values.volume < 0).any():
        raise ValueError("Invalid OHLCV")
    if not ((values.low <= values.open) & (values.open <= values.high) &
            (values.low <= values.close) & (values.close <= values.high)).all():
        raise ValueError("Invalid OHLC ordering")
    returns = np.log(values.close / values.close.shift(1))
    # RV = sqrt(sum r^2), trailing 12 returns INCLUDING the current closed bar.
    volatility = np.sqrt(returns.pow(2).rolling(config.volatility_window,
                         min_periods=config.volatility_window, center=False).sum())
    # Baseline explicitly EXCLUDES current volume; population std, no global fit.
    historical_volume = values.volume.shift(1).rolling(config.volume_window,
                         min_periods=config.volume_window, center=False)
    volume_z = (values.volume - historical_volume.mean()) / historical_volume.std(ddof=0).clip(lower=config.volume_scale_floor)
    result = pd.DataFrame({
        "log_return": returns,
        "rolling_volatility": volatility,
        "volume_zscore": volume_z,
        "high_low_range": np.log(values.high / values.low),
        "volume_change": np.log1p(values.volume).diff(),
    }, index=data.index)
    # NaN is only allowed in the INITIAL warmup prefix, never silently drop later rows.
    warmup = max(config.volatility_window, config.volume_window)
    if not np.isfinite(result.iloc[warmup:].to_numpy()).all():
        raise ValueError("Nonfinite features after warmup")
    return result.loc[:, FEATURES]
