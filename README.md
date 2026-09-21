# BTC Market Risk Monitoring System

This system monitors short-horizon market risk in BTCUSDT spot using five-minute
Binance candles. It stores bars and signals in PostgreSQL, calculates a causal
Robust Z-score, runs a daily frozen Isolation Forest, and exposes the current
state through FastAPI. Docker Compose runs the database, collector, scheduler,
and API together.

## Results

Across a 29-day walk-forward evaluation, IF-only anomaly observations had mean
future one-hour realized volatility of 0.7798%, compared with 0.3357% for
normal observations. Their high-risk-event rate was 32.56%, compared with
4.47% for normal observations. That is about 7.3x enrichment.

Each daily model uses only data from earlier complete UTC days, then scores the
next day with fixed parameters. The results are therefore out of sample for
each frozen model. The research design and the exposure rule were developed on
the same historical sample, so the project does not have an untouched
end-to-end holdout. Final operational validation ran 104 passing tests,
including persistence and restart coverage.

The portfolio overlay did not improve the outcome. Buy & Hold returned 22.65%
with a maximum drawdown of -6.79%. The IF overlay returned 12.51% after costs
with a maximum drawdown of -7.61%. The anomaly states were followed by higher
short-horizon volatility, but the simple exposure rule still performed worse
than Buy & Hold.

## Architecture

```text
Binance REST / WebSocket
           |
           v
Collector --> Robust Z-score + frozen IF shadow inference
     |                         |
     +--------> PostgreSQL <---+---- Scheduler: daily IF training and activation
                     |
                     v
             Read-only FastAPI
```

PostgreSQL holds market bars, detector outputs, model metadata, and
scheduler records. The collector writes closed bars and signals. The scheduler
trains and activates the daily model. FastAPI reads the stored state and never
starts ingestion or model training.

## Data Pipeline

REST backfill downloads closed Binance candles in pages and writes them
idempotently. The WebSocket collector accepts a candle only after it closes. On
restart or disconnect, the collector checks for missing bars and fills them with
REST before resuming.

If bars are missing, the collector records the gap and does not calculate a
return across it. Replaying an already stored bar is safe. A conflicting bar
with the same key is rejected instead of silently overwritten.

## Robust Z-Score

For each closed bar, the detector calculates the log return:

```text
r_t = log(C_t / C_(t-1))
```

It compares that return with the median and MAD of the previous 288 returns,
which is one day of five-minute data. The score is:

```text
z_t = (r_t - median(previous 288 returns)) / max(1.4826 * MAD, 1e-8)
```

An alert is raised when `abs(z_t) >= 3.5`. The current return is added to the
rolling state after scoring, so it cannot affect its own baseline. A gap resets
the rolling state.

## Isolation Forest

The daily Isolation Forest uses these five features from completed bars:

- `log_return`
- `rolling_volatility`
- `volume_zscore`
- `high_low_range`
- `volume_change`

For scoring day `D`, the model trains on the 60 complete UTC days in
`[D - 60 days, D)`. Day `D` is excluded from training. The fitted model stays
fixed for the full scoring day. Missing feature warmup, training, or scoring
data stops the run instead of producing a partial model.

The default model uses 200 estimators, contamination 0.01, and random state 42.
The system stores the feature order, parameters, input hash, library versions,
and artifact checksum with each run.

## Historical Replay and Live Scoring

For historical replay, the model is treated as available at 00:10 UTC. The
00:00 and 00:05 five-minute bars are skipped because their close times are not
strictly later than that point. The first eligible bar starts at 00:10 UTC and
closes at 00:15 UTC.

In the live system, the scheduler trains from prior data and activates the
model when training completes. The collector loads the active frozen artifact,
checks its checksum, and scores only later closed bars. These scores are stored
for monitoring in shadow mode. They do not change positions.

## Risk Validation

The validation measures realized volatility and adverse moves over the next 30
and 60 minutes. It excludes the return from the signal bar. Each observation is
classified as normal, Z-only, IF-only, or both.

There were 43 actionable IF-only observations and 7,947 normal observations.
The IF-only group had a 0.4442 percentage-point higher mean future one-hour
realized volatility. A 29-day block bootstrap with 2,000 resamples and seed 42
gave a 95% interval of [0.3078, 0.5487]. The sample is short and the IF events
cluster by day. See [risk validation](docs/risk_validation.md) for the full
setup and overlay results.

## Portfolio Overlay

The overlay test asks whether the risk signal helps when converted into a simple
exposure rule. It starts at exposure 1.0 and cuts exposure to 0.5 on the bar
after an actionable signal. The reduced exposure lasts 12 five-minute bars, or
one hour. Each unit of one-way turnover costs 5 basis points when exposure is
reduced and when it returns to normal.

The IF overlay reduced some volatility and tail-loss measures. It still had a
lower return and a deeper maximum drawdown than Buy & Hold over this period.

## Operations

Docker Compose runs four services:

- `database` stores PostgreSQL data.
- `collector` ingests bars, recovers gaps, and calculates signals.
- `scheduler` trains and activates the daily frozen IF model.
- `api` serves read-only monitoring endpoints.

Model artifacts and their SHA-256 checksums are stored with publication data.
The collector restores its rolling state and active model after a restart. See
[operations](docs/operations_validation.md) for the tested lifecycle.

## API

```text
GET /health
GET /market/latest
GET /signals/zscore/latest
GET /signals/if/latest
GET /risk-status
GET /signals/history
```

Latest-value endpoints report `fresh`, `stale`, or `unavailable`. History
accepts timezone-aware `start` and `end` parameters, plus `limit` and `cursor`
for pagination.

## Testing

Tests cover causal feature construction, future-data invariance, batch and
online parity, idempotent writes, gap recovery, migrations, model timing,
checksums, API behavior, and persistence.

```bash
python -m pytest -q -m 'not persistence'
```

The persistence test stops and recreates the local database container, so it
runs separately:

```bash
python -m pytest -q --run-persistence
```

## Reproduction

Python 3.11 or newer is required. The Docker image and CI use Python 3.12.

### Docker

```bash
cp .env.example .env
# Replace the placeholder POSTGRES_PASSWORD in .env.
docker compose up -d --build --wait
docker compose ps
curl -s http://127.0.0.1:8000/health
```

Compose persists database and model volumes. Run `docker compose down -v` only
when you want to remove local state.

### Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock -e '.[test]'
cp .env.example .env
# Replace the placeholder POSTGRES_PASSWORD in .env.
docker compose up -d --wait database
python -m btc_risk.database.migrate
python -m pytest -q -m 'not persistence'
```

## Repository Structure

```text
src/btc_risk/ingestion/   Binance REST, WebSocket, collector, and recovery
src/btc_risk/realtime/    Causal Robust Z-score calculation and replay
src/btc_risk/batch/       IF features and walk-forward training
src/btc_risk/online/      Frozen-model simulation and shadow scoring
src/btc_risk/operations/  Scheduler and read-only FastAPI
src/btc_risk/research/    Risk validation and fixed overlay experiment
sql/                      Immutable PostgreSQL migrations and verification SQL
tests/                    Unit, integration, and persistence tests
docs/                     Methodology, validation, and operations notes
```
