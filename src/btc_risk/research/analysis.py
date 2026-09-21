"""Descriptive group comparisons and fixed UTC-day block bootstrap."""

import numpy as np
import pandas as pd

GROUPS = {"A": "Neither", "B": "Z only", "C": "IF only", "D": "Both"}
METRICS = ["future_realized_volatility", "future_maximum_adverse_move", "future_absolute_return"]
ANALYSIS_CONFIG = {"high_risk_quantile": 0.95, "bootstrap_replicates": 2000,
                   "bootstrap_seed": 42, "bootstrap_block": "whole UTC day", "ci_percentiles": [2.5, 97.5],
                   "plot_price_days": 3}


def analyze(observations, labels):
    group_counts = {g: int((observations.signal_group == g).sum()) for g in GROUPS}
    counts_by_day = observations.assign(day=observations.timestamp.dt.floor("D"))
    distinct_days = {g: int(counts_by_day.loc[counts_by_day.signal_group == g, "day"].nunique()) for g in GROUPS}
    available = labels.loc[labels.label_status == "available"].copy()
    distributions = []
    for minutes in (30, 60):
        for group in GROUPS:
            subset = available.loc[(available.horizon_minutes == minutes) & (available.signal_group == group)]
            for metric in METRICS:
                series = subset[metric]
                distributions.append(dict(group=group, horizon_minutes=minutes, metric=metric,
                    count=len(series), mean=float(series.mean()) if len(series) else None,
                    median=float(series.median()) if len(series) else None,
                    p75=float(series.quantile(.75)) if len(series) else None,
                    p90=float(series.quantile(.90)) if len(series) else None,
                    p95=float(series.quantile(.95)) if len(series) else None))
    hourly = available.loc[available.horizon_minutes == 60].copy()
    if hourly.empty:
        raise ValueError("No complete 1h horizons for evaluation")
    threshold = float(hourly.future_realized_volatility.quantile(ANALYSIS_CONFIG["high_risk_quantile"]))
    hourly["event"] = hourly.future_realized_volatility >= threshold
    events = []
    baseline = hourly.loc[hourly.signal_group == "A", "event"]
    p_normal = float(baseline.mean()) if len(baseline) else None
    for group in GROUPS:
        values = hourly.loc[hourly.signal_group == group, "event"]
        probability = float(values.mean()) if len(values) else None
        events.append(dict(group=group, count=len(values), high_risk_count=int(values.sum()),
                           probability=probability, enrichment_ratio=probability / p_normal if probability is not None and p_normal else None))
    # Resample whole days jointly across all four groups. Daily sums/counts are
    # equivalent to concatenating all observations in sampled day blocks.
    days = pd.DatetimeIndex(observations.timestamp.dt.floor("D").unique()).sort_values()
    grouped = hourly.assign(day=hourly.timestamp.dt.floor("D")).groupby(["day", "signal_group"]).future_realized_volatility.agg(["sum", "count"])
    sums = grouped["sum"].unstack().reindex(index=days, columns=GROUPS).fillna(0).to_numpy()
    counts = grouped["count"].unstack().reindex(index=days, columns=GROUPS).fillna(0).to_numpy()
    rng = np.random.default_rng(ANALYSIS_CONFIG["bootstrap_seed"])
    sampled = rng.integers(0, len(days), size=(ANALYSIS_CONFIG["bootstrap_replicates"], len(days)))
    bootstrap_count = counts[sampled].sum(axis=1)
    means = np.divide(sums[sampled].sum(axis=1), bootstrap_count,
                      out=np.full(bootstrap_count.shape, np.nan), where=bootstrap_count > 0)
    original_count = counts.sum(axis=0)
    original_mean = np.divide(sums.sum(axis=0), original_count, out=np.full(4,np.nan), where=original_count>0)
    differences = []
    for i, group in enumerate(list(GROUPS)[1:], 1):
        differences_sample = means[:, i] - means[:, 0]
        differences_sample = differences_sample[np.isfinite(differences_sample)]
        ci = np.percentile(differences_sample, ANALYSIS_CONFIG["ci_percentiles"]) if len(differences_sample) and len(days) >= 2 else (None, None)
        difference = original_mean[i] - original_mean[0]
        differences.append(dict(group=group, mean_1h_rv_difference=float(difference) if np.isfinite(difference) else None,
                                ci_low=float(ci[0]) if ci[0] is not None else None,
                                ci_high=float(ci[1]) if ci[1] is not None else None,
                                valid_replicates=len(differences_sample)))
    inclusive = {}
    for name, groups in {"Z_alert": ["B", "D"], "IF_anomaly": ["C", "D"]}.items():
        values = hourly.loc[hourly.signal_group.isin(groups), "future_realized_volatility"]
        inclusive[name] = dict(count=len(values), mean_1h_rv=float(values.mean()) if len(values) else None)
    return dict(evaluation_observations=len(observations), group_counts=group_counts, group_days=distinct_days,
                horizon_status_counts={f"{h}:{s}": int(n) for (h,s),n in labels.groupby(["horizon_minutes","label_status"]).size().items()},
                distributions=distributions, high_risk_threshold_1h_rv=threshold, event_enrichment=events,
                block_bootstrap=differences, bootstrap_days=len(days), inclusive_detector_means=inclusive)


def plots(observations, labels, summary, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    colors = {"A":"#64748b", "B":"#2563eb", "C":"#d97706", "D":"#9333ea"}
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False})
    hourly = labels.loc[(labels.horizon_minutes == 60) & (labels.label_status == "available")]
    fig, ax = plt.subplots(figsize=(8,4.8), layout="constrained")
    for group, name in GROUPS.items():
        values = np.sort(hourly.loc[hourly.signal_group == group,"future_realized_volatility"].to_numpy() * 100)
        if len(values):
            ax.step(values, np.arange(1,len(values)+1)/len(values), where="post", color=colors[group], label=f"{name} (n={len(values)})")
    ax.set(xlabel="Future 1h realized volatility (%)", ylabel="Cumulative fraction", title="Forward risk distributions by signal group")
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(output_dir / "01_future_1h_distribution.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4.5),layout="constrained")
    for ax, minutes in zip(axes,(30,60)):
        rows = [r for r in summary["distributions"] if r["metric"] == "future_realized_volatility" and r["horizon_minutes"] == minutes]
        x = np.arange(4)
        ax.bar(x-.18,[100*r["mean"] if r["mean"] is not None else np.nan for r in rows],.36,label="Mean",color="#2563eb")
        ax.bar(x+.18,[100*r["median"] if r["median"] is not None else np.nan for r in rows],.36,label="Median",color="#94a3b8")
        ax.set(xticks=x,xticklabels=list(GROUPS.values()),title=f"Future {minutes}m RV",ylabel="Realized volatility (%)")
        ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.savefig(output_dir / "02_mean_median_rv.png",dpi=160); plt.close(fig)
    start = observations.timestamp.min()
    limited = observations.loc[observations.timestamp < start + pd.Timedelta(days=ANALYSIS_CONFIG["plot_price_days"])]
    fig,ax=plt.subplots(figsize=(11,4.8),layout="constrained")
    ax.plot(limited.timestamp,limited.close.astype(float),color="#334155",linewidth=1,label="BTCUSDT close")
    z=limited.loc[limited.zscore_alert]; iso=limited.loc[limited.isolation_alert]
    ax.scatter(z.timestamp,z.close.astype(float),s=28,marker="x",color=colors["B"],label="Z alert")
    ax.scatter(iso.timestamp,iso.close.astype(float),s=65,facecolors="none",edgecolors=colors["C"],label="IF anomaly (retrospective)")
    ax.set(title="First 3 evaluation days: detector locations on price",ylabel="BTCUSDT",xlabel="Bar opening time (UTC); IF results only available after daily batch")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M",tz="UTC"))
    ax.legend(); ax.grid(alpha=.2)
    fig.savefig(output_dir / "03_price_detectors.png",dpi=160); plt.close(fig)
