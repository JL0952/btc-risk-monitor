# Methodology

## Data and state

The system works with closed BTCUSDT five-minute candles from Binance. REST
backfill and WebSocket collection write to the same PostgreSQL tables. Inserts
are idempotent. When the collector sees a gap, it resets the return state and
does not create a return across the missing interval.

## Robust Z-score

For a closed bar at time `t`, the detector calculates:

```text
r_t = log(C_t / C_(t-1))
z_t = (r_t - median(previous 288 returns)) / max(1.4826 * MAD, 1e-8)
```

The median and MAD come from the previous 288 returns. The current return is
scored before it is appended to that rolling window. An alert is raised when
`abs(z_t) >= 3.5`. The first bar has no prior close. The next 288 returns warm
up the window. A timestamp gap clears the window before processing continues.

## Daily Isolation Forest

The Isolation Forest uses five features from completed bars:

- `log_return`
- `rolling_volatility`
- `volume_zscore`
- `high_low_range`
- `volume_change`

For scoring day `D`, training uses the 60 complete UTC days from `[D - 60 days,
D)`. The scoring day never enters the training data. The fitted model stays
fixed for that day.

Feature construction requires a complete warmup window, complete training data,
and a complete scoring day. Missing bars cause the job to fail. The default
model uses 200 estimators, contamination 0.01, and random state 42.

## Model records

Each model run records the feature order, parameters, training and scoring
periods, input hash, code version, library versions, artifact name, and
SHA-256 checksum. The artifact is saved before the database row is committed.
On a repeat run, the code checks the stored metadata, artifact checksum, and
scores before reusing the result.

## Historical replay and live scoring

Historical replay treats the model as available at 00:10 UTC on each scoring
day. Bars that close at or before that time are skipped. The 00:00 and 00:05
bars are therefore excluded. The first eligible bar starts at 00:10 UTC and
closes at 00:15 UTC.

The live scheduler trains from data before the current UTC day. It saves the
artifact and activates it when training finishes. The collector verifies the
active artifact checksum and scores only closed bars after activation. It stores
the score, `bar_closed_at`, and `score_available_at` in shadow mode.

## Reproducibility checks

The test suite checks that earlier Z-scores and features cannot change when
future data changes. It also checks batch and online feature parity, score
parity, strict scoring times, artifact checksums, and rerun consistency. These
checks make sure earlier signals cannot use future data.
