"""Offline SQL-backed detector information study. No operational writes."""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
from psycopg.types.json import Jsonb
from btc_risk.database.connection import connect
from .analysis import ANALYSIS_CONFIG, analyze, plots
from .labels import LABEL_DEFINITIONS, classify, future_labels

LOG = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def validate(conn, start, end, symbol="BTCUSDT", run_ids=None):
    """End is exclusive. Caller owns the transaction; only research schema is written."""
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Require increasing timezone-aware boundaries")
    if start.utcoffset() != timedelta(0) or end.utcoffset() != timedelta(0):
        raise ValueError("Evaluation boundaries must be UTC")
    rows = conn.execute("""
        SELECT i.timestamp, i.symbol, i.interval, i.run_id AS if_run_id,
               i.anomaly_flag AS isolation_alert, i.anomaly_score,
               i.calculated_at AS if_calculated_at, m.completed_at AS if_available_at,
               z.alert_flag AS zscore_alert, z.status AS z_status, z.robust_zscore,
               z.calculated_at AS z_calculated_at, b.close
        FROM isolation_forest_scores i JOIN model_runs m ON m.run_id=i.run_id
        JOIN market_bars b ON (b.symbol,b.interval,b.timestamp)=(i.symbol,i.interval,i.timestamp)
        LEFT JOIN realtime_signals z ON (z.symbol,z.interval,z.timestamp)=(i.symbol,i.interval,i.timestamp)
        WHERE i.symbol=%s AND i.interval='5m' AND i.timestamp >= %s AND i.timestamp < %s
          AND m.model_version='1.0' AND (%s::uuid[] IS NULL OR i.run_id=ANY(%s::uuid[]))
        ORDER BY i.timestamp, i.run_id
    """, (symbol, start, end, run_ids, run_ids)).fetchall()
    if len({r['timestamp'] for r in rows}) != len(rows):
        raise ValueError("Multiple IF runs score the same timestamp; select --if-run-id explicitly")
    usable = [r for r in rows if r['z_status'] in ('scored','scale_floored') and r['zscore_alert'] is not None]
    if not usable:
        raise ValueError("No common scored Z / IF observations")
    market = conn.execute("""SELECT timestamp, close, low FROM market_bars
        WHERE symbol=%s AND interval='5m' AND timestamp >= %s AND timestamp < %s
          AND timestamp + INTERVAL '5 minutes' <= %s ORDER BY timestamp""",
        (symbol, start, end + timedelta(hours=1), datetime.now(timezone.utc))).fetchall()
    observations = pd.DataFrame(usable)
    observations['timestamp'] = pd.to_datetime(observations.timestamp, utc=True)
    observations['signal_group'] = [classify(r['zscore_alert'],r['isolation_alert']) for r in usable]
    bars = pd.DataFrame(market).set_index('timestamp')
    bars.index = pd.to_datetime(bars.index, utc=True)
    labels = observations[['timestamp','signal_group','zscore_alert','isolation_alert','if_run_id']].merge(
        future_labels(bars), on='timestamp', how='left', validate='one_to_many')
    if labels.label_status.isna().any():
        raise ValueError("A scored observation has no closed market bar")
    summary = analyze(observations, labels)
    summary['input_if_scores'] = len(rows)
    summary['excluded_unscored_z'] = len(rows)-len(usable)
    summary['market_last_timestamp'] = str(bars.index.max())
    summary['if_available_at_range'] = [str(min(r['if_available_at'] for r in usable)), str(max(r['if_available_at'] for r in usable))]
    selected_ids = sorted({str(r['if_run_id']) for r in usable})
    code_version = digest({p.name:p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))})
    libraries = {name:version(name) for name in ('numpy','pandas','psycopg','matplotlib')}
    data_hash = digest({'signals':rows,'market':market})
    identity = digest(dict(start=start,end=end,symbol=symbol,run_ids=selected_ids,labels=LABEL_DEFINITIONS,
                           config=ANALYSIS_CONFIG,code=code_version,libraries=libraries,data=data_hash))
    conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))',(identity,))
    previous = conn.execute('SELECT validation_run_id, summary FROM research.validation_runs WHERE validation_identity=%s',(identity,)).fetchone()
    if previous:
        if previous['summary'] != summary:
            raise RuntimeError('Identical inputs produced inconsistent research results')
        run_id = previous['validation_run_id']
        count = conn.execute('SELECT count(*) AS n FROM research.risk_validation WHERE validation_run_id=%s',(run_id,)).fetchone()['n']
        if count != len(labels):
            raise RuntimeError('Existing validation run has incomplete records')
    else:
        run_id = uuid4()
        conn.execute('''INSERT INTO research.validation_runs
            (validation_run_id,validation_identity,symbol,interval,evaluation_start,evaluation_end,
             if_run_ids,label_definitions,analysis_config,code_version,data_hash,library_versions,summary)
             VALUES (%s,%s,%s,'5m',%s,%s,%s,%s,%s,%s,%s,%s,%s)''',
             (run_id,identity,symbol,start,end,selected_ids,Jsonb(LABEL_DEFINITIONS),Jsonb(ANALYSIS_CONFIG),code_version,data_hash,Jsonb(libraries),Jsonb(summary)))
        records=[]
        for r in labels.to_dict('records'):
            valid = r['label_status']=='available'
            values = [float(r[k]) if valid else None for k in ('future_realized_volatility','future_maximum_adverse_move','future_absolute_return')]
            event = values[0]>=summary['high_risk_threshold_1h_rv'] if valid and r['horizon_minutes']==60 else None
            records.append((run_id,r['timestamp'],symbol,'5m',r['if_run_id'],r['zscore_alert'],r['isolation_alert'],r['signal_group'],r['horizon_minutes'],r['label_status'],*values,event))
        with conn.cursor() as cur:
            cur.executemany('''INSERT INTO research.risk_validation
             (validation_run_id,timestamp,symbol,interval,if_run_id,zscore_alert,isolation_alert,signal_group,
              horizon_minutes,label_status,future_realized_volatility,future_maximum_adverse_move,future_absolute_return,high_risk_flag)
              VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''',records)
    return str(run_id), bool(previous), summary, observations, labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',required=True,type=date.fromisoformat)
    parser.add_argument('--end',required=True,type=date.fromisoformat,help='Inclusive UTC date')
    parser.add_argument('--symbol',default='BTCUSDT')
    parser.add_argument('--if-run-id',action='append')
    parser.add_argument('--output-dir',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    logging.Formatter.converter=time.gmtime
    logging.basicConfig(level=logging.INFO,format='%(asctime)sZ %(name)s %(levelname)s %(message)s')
    start=datetime.combine(args.start,datetime.min.time(),tzinfo=timezone.utc)
    end=datetime.combine(args.end+timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc)
    LOG.info('Validation start %s to %s (exclusive)',start,end)
    with connect() as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        run_id,reused,summary,observations,labels=validate(conn,start,end,args.symbol,args.if_run_id)
    output=args.output_dir / ('risk_validation_'+run_id)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(dict(validation_run_id=run_id,**summary),indent=2,allow_nan=False)+'\n')
    pd.DataFrame(summary['distributions']).to_csv(output/'metrics.csv',index=False)
    plots(observations,labels,summary,output)
    LOG.info('Validation complete run_id=%s reused=%s observations=%s records=%s groups=%s output=%s',run_id,reused,len(observations),len(labels),summary['group_counts'],output)


if __name__=='__main__':
    main()
