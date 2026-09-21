"""Fixed-rule long-only risk-overlay backtest using actionable Stage 7 signals."""
import argparse
from dataclasses import dataclass
from datetime import timedelta
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from btc_risk.database.connection import connect

STEP = pd.Timedelta(minutes=5)
PER_YEAR = 365 * 24 * 12
CONFIG = dict(normal_exposure=1.0, reduced_exposure=0.5, hold_bars=12,
              one_way_turnover_cost=0.0005, annualization_periods=PER_YEAR,
              if_source="online_isolation_forest_scores/historical_simulation only")
STRATEGIES = {"buy_hold": "none", "z_overlay": "z", "if_overlay": "if", "union_overlay": "union"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def signal_mask(frame, kind):
    if kind == "none": return np.zeros(len(frame), dtype=bool)
    if kind == "z": return frame.z_alert.to_numpy(dtype=bool)
    usable_if = frame.if_alert.to_numpy(dtype=bool) & frame.if_available.to_numpy(dtype=bool)
    if kind == "if": return usable_if
    return frame.z_alert.to_numpy(dtype=bool) | usable_if


def overlay_path(frame, kind, config=CONFIG):
    """Exposure[i] applies only to close[i] -> close[i+1].

    A signal at i updates expiry after return i is assigned, so it can only
    reduce i+1 through i+hold_bars.  This is the anti-lookahead invariant.
    """
    # Slice off the terminal close before Decimal division; pandas otherwise
    # evaluates its None shift sentinel even though that row has no return.
    btc = (frame.close.iloc[1:].to_numpy(dtype=float) / frame.close.iloc[:-1].to_numpy(dtype=float) - 1)
    alerts = signal_mask(frame.iloc[:-1], kind)
    n = len(btc); exposure = np.ones(n); expiry = -1
    for i in range(n):
        exposure[i] = config["reduced_exposure"] if i <= expiry else config["normal_exposure"]
        if alerts[i]: expiry = max(expiry, i + config["hold_bars"])
    prior = np.r_[config["normal_exposure"], exposure[:-1]]
    turnover = np.abs(exposure - prior)
    cost = turnover * config["one_way_turnover_cost"]
    gross = exposure * btc
    net = gross - cost
    result = frame.iloc[:-1][["timestamp", "close", "z_alert", "if_alert"]].copy()
    result["btc_return"] = btc; result["exposure"] = exposure; result["turnover"] = turnover
    result["cost"] = cost; result["gross_return"] = gross; result["net_return"] = net
    return result


def _total(returns): return float(np.prod(1 + returns) - 1)


def metrics(path, config=CONFIG):
    gross, net = path.gross_return.to_numpy(), path.net_return.to_numpy()
    n = len(net); factor = config["annualization_periods"]
    equity = np.r_[1.0, np.cumprod(1 + net)]
    dd = equity / np.maximum.accumulate(equity) - 1
    std = np.std(net, ddof=1)
    ann_vol = float(std * np.sqrt(factor))
    ann_return = float((1 + _total(net)) ** (factor / n) - 1)
    downside = float(np.sqrt(np.mean(np.minimum(net, 0) ** 2)) * np.sqrt(factor))
    one_hour = pd.Series(net).rolling(12).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    one_day = pd.Series(net).rolling(288).apply(lambda x: np.prod(1 + x) - 1, raw=True)
    losses = -net
    reduced = path.exposure.to_numpy() < config["normal_exposure"]
    withheld = (config["normal_exposure"] - path.exposure.to_numpy()) * path.btc_return.to_numpy()
    return dict(total_return_gross=_total(gross), total_return_net=_total(net), annualized_return_net=ann_return,
                annualized_volatility=ann_vol, sharpe_ratio=float(np.mean(net) / std * np.sqrt(factor)) if std else None,
                downside_deviation=downside, sortino_ratio=float(ann_return / downside) if downside else None,
                maximum_drawdown=float(dd.min()), turnover=float(path.turnover.sum()), exposure_changes=int((path.turnover > 0).sum()),
                reduced_exposure_fraction=float(reduced.mean()), transaction_cost=float(path.cost.sum()),
                worst_1h_return=float(one_hour.min()), worst_daily_return=float(one_day.min()),
                loss_quantile_95=float(np.quantile(losses, .95)), loss_quantile_99=float(np.quantile(losses, .99)),
                downside_losses_avoided=float(np.maximum(-withheld, 0).sum()), upside_gains_missed=float(np.maximum(withheld, 0).sum()),
                return_observations=n)


def load_inputs(conn, symbol="BTCUSDT"):
    bounds = conn.execute("""SELECT min(o.timestamp) AS start,max(o.timestamp) AS last,
        array_agg(DISTINCT o.publication_id ORDER BY o.publication_id) AS publication_ids
        FROM online_isolation_forest_scores o JOIN online_if_model_publications p USING(publication_id)
        JOIN realtime_signals z ON (z.symbol,z.interval,z.timestamp)=(o.symbol,o.interval,o.timestamp)
        WHERE o.symbol=%s AND o.interval='5m' AND p.publication_mode='historical_simulation'
          AND z.status IN ('scored','scale_floored') AND o.score_available_at=o.bar_closed_at
          AND o.bar_closed_at > p.model_available_at""", (symbol,)).fetchone()
    if not bounds["start"]: raise ValueError("No legal actionable IF / Z overlap")
    # Start at the first legal actionable signal. The resulting portfolio path is
    # continuous; daily pre-publication bars later in the path are no-IF-decision
    # bars, never classified as IF normal.
    start, end = bounds["start"], bounds["last"] + timedelta(minutes=10)
    rows = conn.execute("""SELECT b.timestamp,b.close,
        coalesce(z.alert_flag,false) AS z_alert,z.status AS z_status,
        coalesce(o.anomaly_flag,false) AS if_alert,o.timestamp IS NOT NULL AS if_available,o.anomaly_score
        FROM market_bars b
        LEFT JOIN realtime_signals z USING(symbol,interval,timestamp)
        LEFT JOIN online_isolation_forest_scores o ON (o.symbol,o.interval,o.timestamp)=(b.symbol,b.interval,b.timestamp)
        LEFT JOIN online_if_model_publications p ON p.publication_id=o.publication_id
        WHERE b.symbol=%s AND b.interval='5m' AND b.source='binance' AND b.timestamp >= %s AND b.timestamp < %s
          AND (o.publication_id IS NULL OR p.publication_mode='historical_simulation')
        ORDER BY b.timestamp""", (symbol, start, end)).fetchall()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    expected = pd.date_range(frame.timestamp.iloc[0], periods=len(frame), freq=STEP)
    if not frame.timestamp.equals(pd.Series(expected)) or frame.close.isna().any():
        raise ValueError("Market gap in overlay evaluation interval")
    if frame.z_status.isna().any() or ~frame.z_status.isin(["scored","scale_floored"]).all():
        raise ValueError("Unavailable Z score inside overlay evaluation interval")
    return frame, bounds["publication_ids"], start, end


def event_rows(frame, paths):
    # Predefined selection: top five IF-only and top five both, ordered only by
    # the contemporaneous actionable IF anomaly score, never by future outcome.
    source = frame.loc[frame.if_available & frame.if_alert, ["timestamp","if_alert","z_alert","anomaly_score"]]
    ranked = pd.concat([
        source.loc[~source.z_alert].sort_values(["anomaly_score","timestamp"], ascending=[False,True]).head(5),
        source.loc[source.z_alert].sort_values(["anomaly_score","timestamp"], ascending=[False,True]).head(5),
    ]).sort_values("timestamp")
    rows=[]
    for _, event in ranked.iterrows():
        i = int(frame.index[frame.timestamp == event.timestamp][0]); stop = min(i + 12, len(frame)-1)
        btc = frame.close.iloc[stop] / frame.close.iloc[i] - 1
        result = dict(signal_timestamp=str(event.timestamp), signal_type="both" if event.z_alert else "if_only",
                      actionable_if_anomaly_score=float(event.anomaly_score),
                      subsequent_1h_btc_return=float(btc))
        for name, path in paths.items():
            segment = path.iloc[i:min(i+12,len(path))]
            result[name + "_net_return"] = _total(segment.net_return.to_numpy()) if len(segment) else None
        rows.append(result)
    return rows


def run(conn, symbol="BTCUSDT"):
    frame, publication_ids, start, end = load_inputs(conn, symbol)
    paths = {name: overlay_path(frame, kind) for name, kind in STRATEGIES.items()}
    result = {name: metrics(path) for name, path in paths.items()}
    events = event_rows(frame, paths)
    code_version = digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    data_hash = digest(frame.astype(object).where(pd.notna(frame), None).to_dict("records"))
    libraries = {name:version(name) for name in ("numpy","pandas","psycopg","matplotlib")}
    identity = digest(dict(symbol=symbol,start=start,end=end,publication_ids=[str(x) for x in publication_ids],
                           config=CONFIG,code=code_version,data=data_hash,libraries=libraries))
    summary = dict(evaluation_start=str(start),evaluation_end=str(end),bars=len(frame),days=(end-start).total_seconds()/86400,
                   actionable_if_score_count=int(frame.if_available.sum()), if_unavailable_decision_bars=int((~frame.if_available).sum()),
                   strategies=result, events=events)
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",(identity,))
    old=conn.execute("SELECT overlay_run_id,summary FROM research.overlay_runs WHERE overlay_identity=%s",(identity,)).fetchone()
    if old:
        if old["summary"] != summary: raise RuntimeError("Identical overlay input produced different output")
        run_id=old["overlay_run_id"]; reused=True
    else:
        run_id=uuid4(); reused=False
        conn.execute("""INSERT INTO research.overlay_runs
            (overlay_run_id,overlay_identity,symbol,interval,evaluation_start,evaluation_end,return_observations,
             actionable_publication_ids,strategy_definitions,code_version,data_hash,library_versions,summary)
            VALUES (%s,%s,%s,'5m',%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (run_id,identity,symbol,start,end,len(paths["buy_hold"]),publication_ids,Jsonb(dict(config=CONFIG,strategies=STRATEGIES)),code_version,data_hash,Jsonb(libraries),Jsonb(summary)))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO research.overlay_results (overlay_run_id,strategy_name,metrics) VALUES (%s,%s,%s)",
                            [(run_id,name,Jsonb(value)) for name,value in result.items()])
    return str(run_id),reused,summary,paths


def plots(paths, frame, output):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    for name,path in paths.items(): ax.plot(path.timestamp,np.cumprod(1+path.net_return),label=name)
    ax.set(title="Risk overlay equity curves (net)",ylabel="Equity",xlabel="UTC"); ax.legend(); ax.grid(alpha=.2); fig.savefig(output/"01_equity.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained")
    for name,path in paths.items():
        equity=np.cumprod(1+path.net_return); peak=np.maximum.accumulate(np.r_[1,equity])[1:]
        ax.plot(path.timestamp,equity/peak-1,label=name)
    ax.set(title="Drawdown curves",ylabel="Drawdown",xlabel="UTC"); ax.legend(); ax.grid(alpha=.2); fig.savefig(output/"02_drawdown.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.5),layout="constrained")
    for name in ("z_overlay","if_overlay","union_overlay"): ax.step(paths[name].timestamp,paths[name].exposure,where="post",label=name)
    ax.set(title="Exposure after closed-bar decisions",ylim=(.45,1.05),ylabel="Exposure",xlabel="UTC"); ax.legend(); ax.grid(alpha=.2); fig.savefig(output/"03_exposure.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained"); ax.plot(frame.timestamp,frame.close,color="#334155",lw=.8)
    reduced=paths["union_overlay"].exposure.to_numpy()<1
    ax.fill_between(paths["union_overlay"].timestamp,frame.close.iloc[:-1].min(),frame.close.iloc[:-1].max(),where=reduced,color="#f59e0b",alpha=.2,label="Union reduced")
    ax.set(title="BTC close and union risk-reduction periods",ylabel="BTCUSDT",xlabel="UTC"); ax.legend(); fig.savefig(output/"04_price_reduced.png",dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--symbol",default="BTCUSDT"); parser.add_argument("--output-dir",type=Path,default=Path("outputs")); args=parser.parse_args()
    with connect() as conn: run_id,reused,summary,paths=run(conn,args.symbol)
    output=args.output_dir/("overlay_"+run_id); plots(paths, pd.concat([paths["buy_hold"][["timestamp","close"]],pd.DataFrame()],axis=1), output)
    pd.DataFrame(summary["strategies"]).T.to_csv(output/"metrics.csv"); pd.DataFrame(summary["events"]).to_csv(output/"events.csv",index=False)
    (output/"summary.json").write_text(json.dumps(dict(overlay_run_id=run_id,reused=reused,**summary),indent=2)+"\n")
    print(json.dumps(dict(overlay_run_id=run_id,reused=reused,**summary),indent=2))


if __name__=="__main__": main()
