# Operations

## Services

Docker Compose runs four services. PostgreSQL stores bars, signals, model
records, and scheduler history. The collector ingests closed Binance candles,
recovers gaps with REST, calculates the Robust Z-score, and runs shadow IF
scoring. The scheduler trains and activates the daily frozen model. FastAPI
serves the stored monitoring state.

## Model lifecycle

The scheduler records each daily training attempt. It trains from prior data,
saves the model artifact, then points the active model record at that artifact.
The artifact checksum is saved with the model record. The collector validates
the checksum before loading the active model.

After a database or collector restart, the collector restores its rolling
Robust Z-score state and reloads the active IF artifact. A scheduler restart
keeps the already published model active instead of publishing another one for
the same day.

## API

The API opens database requests in read-only transactions and exposes:

```text
GET /health
GET /market/latest
GET /signals/zscore/latest
GET /signals/if/latest
GET /risk-status
GET /signals/history
```

Latest-value responses report `fresh`, `stale`, or `unavailable`. The history
endpoint requires timezone-aware `start` and `end` parameters and supports
`limit` and `cursor` pagination.

## Tests

Final operational validation ran 104 passing tests. The suite covered schema
migrations, idempotent ingestion, gap recovery, causal replay, frozen-model
parity, historical scoring times, artifact checksums, scheduler restarts, API
responses, and database container persistence.

The default CI job runs the normal non-persistence suite. The local persistence
test stops, starts, and recreates the database container, so it remains opt-in:

```bash
python -m pytest -q --run-persistence
```

## Deployment limits

This is a local Docker Compose deployment. It has no cloud deployment,
external scheduler redundancy, live capital, order routing, or automatic
exposure changes. Database and model volumes persist across restarts.
