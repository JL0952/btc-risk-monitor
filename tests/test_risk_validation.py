from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from btc_risk.research.labels import future_labels, classify
from btc_risk.research.analysis import analyze
from btc_risk.research.validate_risk import validate


def bars():
    returns=np.arange(1,25)*.001
    close=100*np.exp(np.r_[0,np.cumsum(returns)])
    return pd.DataFrame({'close':close,'low':close*.99},index=pd.date_range('2026-01-01',periods=25,freq='5min',tz='UTC'))


@pytest.mark.parametrize('minutes,h',[(30,6),(60,12)])
def test_future_rv_exact_horizon(minutes,h):
    data=bars()
    row=future_labels(data).query('horizon_minutes==@minutes').iloc[0]
    assert row.future_realized_volatility == pytest.approx(np.sqrt(np.sum((np.arange(1,h+1)*.001)**2)))
    assert row.future_absolute_return == pytest.approx(np.sum(np.arange(1,h+1)*.001))
    assert row.future_maximum_adverse_move == pytest.approx(max(0,1-data.low.iloc[1:h+1].min()/data.close.iloc[0]))


def test_current_return_and_past_are_excluded():
    data=bars()
    original=future_labels(data).query('timestamp==@data.index[5]')
    data.loc[data.index[:5],['close','low']]*=7
    pd.testing.assert_frame_equal(original,future_labels(data).query('timestamp==@data.index[5]'))


def test_incomplete_tail_not_filled():
    result=future_labels(bars())
    tail=result.groupby('horizon_minutes').tail(1)
    assert set(tail.label_status)=={'incomplete_horizon'}
    assert tail.future_realized_volatility.isna().all()


def test_gap_not_crossed():
    data=bars().drop(bars().index[3])
    result=future_labels(data)
    assert set(result.iloc[:2].label_status)=={'gap'}
    assert result.iloc[:2].future_realized_volatility.isna().all()
    assert result.loc[result.timestamp==data.index[4],'label_status'].eq('available').all()


def test_groups_and_missing_signals():
    assert [classify(z,i) for z,i in [(False,False),(True,False),(False,True),(True,True)]]==list('ABCD')
    with pytest.raises(ValueError): classify(None,False)


def test_sort_and_utc():
    data=bars()
    reverse=data.iloc[::-1].copy()
    reverse.index=reverse.index.tz_convert('America/Toronto')
    pd.testing.assert_frame_equal(future_labels(data),future_labels(reverse))


def test_block_bootstrap_deterministic():
    observations=pd.DataFrame({'timestamp':pd.date_range('2026-01-01',periods=96,freq='1h',tz='UTC'), 'signal_group':list('ABCD')*24})
    labels=observations.assign(horizon_minutes=60,label_status='available',future_realized_volatility=np.tile([1.,2.,3.,4.],24),future_maximum_adverse_move=.1,future_absolute_return=.1)
    first=analyze(observations,labels)
    assert first==analyze(observations,labels)
    assert first['bootstrap_days']==4
    assert [(r['ci_low'],r['ci_high']) for r in first['block_bootstrap']]==[(1,1),(2,2),(3,3)]


def test_production_does_not_import_research():
    root=Path('src/btc_risk')
    for path in root.rglob('*.py'):
        if 'research' not in path.parts:
            assert 'btc_risk.research' not in path.read_text()


def test_database_alignment_idempotence_and_no_operational_writes(conn):
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    end=start+timedelta(hours=1)
    symbol='RESEARCH_TEST_'+uuid4().hex
    model_id=uuid4()
    data=bars()
    for i,(stamp,r) in enumerate(data.iterrows()):
        conn.execute("INSERT INTO market_bars(timestamp,symbol,interval,open,high,low,close,volume,source) VALUES (%s,%s,'5m',%s,%s,%s,%s,1,'synthetic-test')",(stamp,symbol,r.close,r.close,r.low,r.close))
        conn.execute("INSERT INTO realtime_signals(timestamp,symbol,interval,log_return,rolling_median,rolling_mad,robust_zscore,alert_flag,status) VALUES (%s,%s,'5m',0,0,1,0,%s,'scored')",(stamp,symbol,i%4 in (1,3)))
    conn.execute("""INSERT INTO model_runs(run_id,run_identity,model_name,model_version,symbol,interval,
      training_start,training_end,scoring_start,scoring_end,feature_set,model_parameters,training_rows,scoring_rows,
      data_hash,code_version,library_versions,artifact_path,artifact_sha256)
      VALUES (%s,%s,'test-fixture','1.0',%s,'5m',%s,%s,%s,%s,'[]','{}',1,12,'test','test','{}','test','test')""",
      (model_id,str(model_id),symbol,start-timedelta(days=1),start,start,end))
    # Insert IF rows in reverse order to verify joins use timestamps, not positions.
    for i in reversed(range(12)):
        conn.execute("INSERT INTO isolation_forest_scores(timestamp,symbol,interval,run_id,anomaly_score,anomaly_flag,feature_values) VALUES (%s,%s,'5m',%s,0,%s,'{}')",(data.index[i],symbol,model_id,i%4 in (2,3)))
    before={table:conn.execute('SELECT count(*) AS n FROM '+table).fetchone()['n'] for table in ('market_bars','realtime_signals','model_runs','isolation_forest_scores')}
    first=validate(conn,start,end,symbol)
    second=validate(conn,start,end,symbol)
    assert first[0]==second[0] and second[1] and first[2]==second[2]
    observations=first[3]
    assert observations.timestamp.is_unique
    assert observations.timestamp.is_monotonic_increasing
    for r in observations.to_dict('records'):
        expected=conn.execute('SELECT z.alert_flag,i.anomaly_flag FROM realtime_signals z JOIN isolation_forest_scores i USING(symbol,interval,timestamp) WHERE z.timestamp=%s AND i.run_id=%s',(r['timestamp'],r['if_run_id'])).fetchone()
        assert r['signal_group']==classify(expected['alert_flag'],expected['anomaly_flag'])
    for table,n in before.items():
        assert conn.execute('SELECT count(*) AS n FROM '+table).fetchone()['n']==n
    assert conn.execute('SELECT count(*) AS n FROM research.risk_validation WHERE validation_run_id=%s',(first[0],)).fetchone()['n']==len(observations)*2
