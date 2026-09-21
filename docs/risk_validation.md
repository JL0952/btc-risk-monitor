# Risk Validation

## Setup

The validation asks whether anomaly states are followed by higher short-horizon
risk. For every scored bar, it calculates realized volatility, maximum adverse
move, and absolute return over the next 30 and 60 minutes. Those labels use
later returns and exclude the return from the signal bar.

Observations fall into four groups: normal, Robust-Z-only, actionable-IF-only,
and both. The evaluation covers 29 UTC scoring days from 2026-08-15 through
2026-09-12. Historical replay applies the 00:10 UTC availability time described
in [Methodology](methodology.md).

## Results

The IF-only group contained 43 observations. The normal group contained 7,947.
Mean future one-hour realized volatility was 0.7798% for IF-only observations
and 0.3357% for normal observations.

The high-risk-event rate was 32.56% for IF-only observations, or 14 of 43. It
was 4.47% for normal observations, or 355 of 7,947. That is about 7.3x
enrichment. The high-risk cutoff is the full-sample 95th percentile of future
one-hour realized volatility and is used only for this descriptive comparison.

## Bootstrap

The bootstrap resamples whole UTC days so observations from the same day stay
together. It uses 29 daily blocks, 2,000 resamples, and seed 42. The IF-only
minus normal difference in mean future one-hour realized volatility was 0.4442
percentage points. The 95% interval was [0.3078, 0.5487].

The anomaly periods had higher short-horizon volatility in this sample. The
sample is short, and the IF events cluster across days, so the result needs more
data before it can support a broader claim.

## Exposure Test

The exposure test checks whether the risk signal helped when used in a simple
long-only rule. Buy & Hold keeps exposure at 1.0. The IF overlay cuts exposure
to 0.5 on the next bar after an actionable signal and holds that level for 12
five-minute bars. Each unit of one-way turnover costs 5 basis points when
exposure changes.

| Metric | Buy & Hold | IF overlay |
| --- | ---: | ---: |
| Net total return | 22.65% | 12.51% |
| Maximum drawdown | -6.79% | -7.61% |
| Annualized volatility | 40.06% | 35.33% |
| Cost drag | 0.00% | 2.45% |

The overlay lowered volatility and some tail-loss measures. It also gave up
return and had a deeper maximum drawdown than Buy & Hold.

## Limitations

The data covers BTCUSDT spot on one venue and a short period. The full research
design and overlay rule were developed on the same historical sample, so there
is no untouched end-to-end holdout. Daily IF scores are fit independently and
are not a probability scale for ranking across days. The overlay has no live
execution, slippage, capital, or position-management model.
