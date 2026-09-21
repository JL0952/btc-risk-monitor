"""Load local database settings; process environment takes precedence over .env."""

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class BatchConfig:
    training_days: int = 60
    volatility_window: int = 12
    volume_window: int = 288
    volume_scale_floor: float = 1e-8
    n_estimators: int = 200
    contamination: float = 0.01
    random_state: int = 42
    models_dir: Path = Path("models")

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.training_days, self.n_estimators, self.volatility_window)):
            raise ValueError("Batch day/window/tree counts must be positive integers")
        if self.volume_window < 2 or not 0 < self.contamination <= 0.5:
            raise ValueError("Require volume window >= 2 and contamination in (0, 0.5]")
        if not math.isfinite(self.volume_scale_floor) or self.volume_scale_floor <= 0:
            raise ValueError("Volume scale floor must be finite and positive")

    @classmethod
    def from_env(cls):
        load_dotenv(".env", override=False)
        return cls(training_days=int(os.getenv("IF_TRAINING_DAYS", "60")),
                   volatility_window=int(os.getenv("IF_VOLATILITY_WINDOW", "12")),
                   volume_window=int(os.getenv("IF_VOLUME_WINDOW", "288")),
                   volume_scale_floor=float(os.getenv("IF_VOLUME_SCALE_FLOOR", "1e-8")),
                   n_estimators=int(os.getenv("IF_N_ESTIMATORS", "200")),
                   contamination=float(os.getenv("IF_CONTAMINATION", "0.01")),
                   random_state=int(os.getenv("IF_RANDOM_STATE", "42")),
                   models_dir=Path(os.getenv("MODELS_DIR", "models")))


@dataclass(frozen=True)
class CollectorConfig:
    ws_url: str = "wss://data-stream.binance.vision/ws"
    queue_size: int = 64
    idle_timeout: float = 30
    backoff_max: float = 60
    bootstrap_days: int = 2

    @classmethod
    def from_env(cls):
        load_dotenv(".env", override=False)
        result = cls(ws_url=os.getenv("BINANCE_WS_URL", cls.ws_url).rstrip("/"),
                     queue_size=int(os.getenv("COLLECTOR_QUEUE_SIZE", "64")),
                     idle_timeout=float(os.getenv("COLLECTOR_IDLE_TIMEOUT", "30")),
                     backoff_max=float(os.getenv("COLLECTOR_BACKOFF_MAX", "60")),
                     bootstrap_days=int(os.getenv("COLLECTOR_BOOTSTRAP_DAYS", "2")))
        if result.queue_size < 1 or result.bootstrap_days < 1 or any(
            not math.isfinite(v) or v <= 0 for v in (result.idle_timeout, result.backoff_max)
        ):
            raise ValueError("Invalid collector configuration")
        return result


@dataclass(frozen=True)
class DetectorConfig:
    window: int = 288
    threshold: float = 3.5
    scale_floor: float = 1e-8

    def __post_init__(self):
        if type(self.window) is not int or self.window < 2:
            raise ValueError("Z_WINDOW must be an integer >= 2")
        if any(not math.isfinite(value) or value <= 0 for value in (self.threshold, self.scale_floor)):
            raise ValueError("Z_THRESHOLD and Z_SCALE_FLOOR must be finite and positive")

    @classmethod
    def from_env(cls):
        load_dotenv(".env", override=False)
        return cls(window=int(os.getenv("Z_WINDOW", "288")),
                   threshold=float(os.getenv("Z_THRESHOLD", "3.5")),
                   scale_floor=float(os.getenv("Z_SCALE_FLOOR", "1e-8")))


@dataclass(frozen=True)
class IngestionConfig:
    symbol: str = "BTCUSDT"
    interval: str = "5m"
    base_url: str = "https://data-api.binance.vision"
    page_size: int = 1000
    timeout: float = 20
    attempts: int = 3

    @classmethod
    def from_env(cls):
        load_dotenv(".env", override=False)
        result = cls(
            symbol=os.getenv("MARKET_SYMBOL", "BTCUSDT"),
            interval=os.getenv("BAR_INTERVAL", "5m"),
            base_url=os.getenv("BINANCE_REST_URL", "https://data-api.binance.vision").rstrip("/"),
            page_size=int(os.getenv("BINANCE_PAGE_SIZE", "1000")),
            timeout=float(os.getenv("BINANCE_TIMEOUT", "20")),
            attempts=int(os.getenv("BINANCE_ATTEMPTS", "3")),
        )
        if not 1 <= result.page_size <= 1000 or result.timeout <= 0 or not 1 <= result.attempts <= 5:
            raise ValueError("Invalid Binance pagination/timeout/retry configuration")
        return result


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str = field(repr=False)

    @classmethod
    def from_env(cls, env_file: Path | str = ".env") -> "DatabaseConfig":
        load_dotenv(env_file, override=False)
        required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
        missing = [key for key in required if not os.getenv(key)]
        if missing:
            raise ValueError(f"Missing environment settings: {', '.join(missing)}")
        password = os.environ["POSTGRES_PASSWORD"]
        if password == "replace_with_a_random_local_password":
            raise ValueError("Replace the example POSTGRES_PASSWORD in .env")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        if not 1 <= port <= 65535:
            raise ValueError("POSTGRES_PORT must be between 1 and 65535")
        return cls(
            host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
            port=port,
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=password,
        )


@dataclass(frozen=True)
class ApiConfig:
    """Explicit freshness limits for the read-only monitoring API, in seconds."""
    market_stale_seconds: int = 900
    zscore_stale_seconds: int = 900
    if_score_stale_seconds: int = 900
    if_model_stale_seconds: int = 129600

    @classmethod
    def from_env(cls):
        load_dotenv(".env", override=False)
        values = cls(
            market_stale_seconds=int(os.getenv("API_MARKET_STALE_SECONDS", "900")),
            zscore_stale_seconds=int(os.getenv("API_ZSCORE_STALE_SECONDS", "900")),
            if_score_stale_seconds=int(os.getenv("API_IF_SCORE_STALE_SECONDS", "900")),
            if_model_stale_seconds=int(os.getenv("API_IF_MODEL_STALE_SECONDS", "129600")),
        )
        if any(value < 1 for value in values.__dict__.values()):
            raise ValueError("API stale intervals must be positive seconds")
        return values
