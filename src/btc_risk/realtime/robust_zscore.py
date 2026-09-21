"""Source-independent stateful detector: score, save, then advance rolling state."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import logging
import math
from statistics import median
from typing import Callable

from btc_risk.config import DetectorConfig

logger = logging.getLogger(__name__)
MAD_NORMALIZATION = 1.4826
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Observation:
    timestamp: datetime
    symbol: str
    interval: str
    close: Decimal


@dataclass(frozen=True)
class Signal:
    timestamp: datetime
    symbol: str
    interval: str
    log_return: float | None
    rolling_median: float | None
    rolling_mad: float | None
    robust_zscore: float | None
    alert_flag: bool | None
    status: str


class RobustZScore:
    def __init__(self, symbol: str, interval: str, config: DetectorConfig | None = None):
        self.symbol = symbol
        self.interval = interval
        self.step = {"5m": timedelta(minutes=5), "1h": timedelta(hours=1)}[interval]
        self.config = config or DetectorConfig()
        self.returns: deque[float] = deque(maxlen=self.config.window)
        self.previous: Observation | None = None

    def process(self, bar: Observation, save: Callable[[Signal], object] | None = None) -> Signal:
        """No state change until validation, calculation, and save have succeeded.

        The caller owns durability/transactions. If a surrounding DB transaction
        later fails, discard this instance and replay from the same origin.
        """
        if bar.symbol != self.symbol or bar.interval != self.interval:
            raise ValueError("Detector cannot mix symbols or intervals")
        if bar.timestamp.tzinfo is None or bar.timestamp.utcoffset() is None:
            raise ValueError("Observation must have a timezone")
        timestamp = bar.timestamp.astimezone(timezone.utc)
        if (timestamp - EPOCH) % self.step:
            raise ValueError("Observation timestamp is not interval aligned")
        if not bar.close.is_finite() or bar.close <= 0:
            raise ValueError("Close must be finite and positive")
        bar = Observation(timestamp, bar.symbol, bar.interval, bar.close)
        previous = self.previous
        if previous and timestamp <= previous.timestamp:
            raise ValueError("Duplicate or out-of-order timestamp; detector state unchanged")
        gap = previous is not None and timestamp - previous.timestamp != self.step
        value = location = mad = zscore = alert = None
        if previous is None:
            status = "no_previous_close"
        elif gap:
            status = "gap"
            logger.warning("Gap symbol=%s interval=%s previous=%s current=%s; resetting baseline",
                           self.symbol, self.interval, previous.timestamp, timestamp)
        else:
            value = math.log(float(bar.close / previous.close))
            if not math.isfinite(value):
                raise ValueError("Nonfinite log return")
            status = "warmup"
            if len(self.returns) == self.config.window:
                location = median(self.returns)
                mad = median(abs(item - location) for item in self.returns)
                raw_scale = MAD_NORMALIZATION * mad
                scale = max(raw_scale, self.config.scale_floor)
                zscore = (value - location) / scale
                if not math.isfinite(zscore):
                    raise ValueError("Nonfinite Robust Z-score")
                alert = abs(zscore) >= self.config.threshold
                status = "scale_floored" if raw_scale < self.config.scale_floor else "scored"
        signal = Signal(timestamp, self.symbol, self.interval, value, location, mad, zscore, alert, status)
        if save is not None:
            save(signal)
        # Current return can only affect the NEXT observation's baseline.
        if gap:
            self.returns.clear()
        if value is not None:
            self.returns.append(value)
        self.previous = bar
        return signal
