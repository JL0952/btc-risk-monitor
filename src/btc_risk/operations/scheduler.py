"""Small UTC polling scheduler for daily frozen IF publication."""
import logging
from datetime import datetime, time as day_time, timezone
from time import sleep
from uuid import uuid4

from btc_risk.config import BatchConfig, IngestionConfig
from btc_risk.database.connection import connect
from btc_risk.database.migrate import migrate
from btc_risk.online.shadow import LiveIFShadow
from btc_risk.online.simulate import MODEL_AVAILABLE_OFFSET

LOG = logging.getLogger(__name__)

def publish_today(now=None, symbol="BTCUSDT", interval="5m", config=None):
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc); config = config or BatchConfig.from_env()
    with connect() as conn:
        prior = conn.execute("SELECT status FROM if_training_attempts WHERE symbol=%s AND interval=%s AND scoring_day=%s",(symbol,interval,now.date())).fetchone()
        if prior and prior["status"] == "published": return "already_published"
        conn.execute("""INSERT INTO if_training_attempts (attempt_id,symbol,interval,scoring_day,training_started_at,status)
            VALUES (%s,%s,%s,%s,%s,'running') ON CONFLICT (symbol,interval,scoring_day)
            DO UPDATE SET training_started_at=excluded.training_started_at,status='running',error_message=NULL""",(uuid4(),symbol,interval,now.date(),now))
    try:
        # Publication metadata commits first. A failure here can leave an
        # unreferenced artifact/metadata row, but never changes the active
        # pointer that collector processes read.
        with connect() as conn:
            shadow, _ = LiveIFShadow.publish(conn, symbol, interval, now=now, config=config)
        # Activation is intentionally a separate transaction after the
        # artifact and publication metadata are durable. Availability is set
        # at activation, never at training start.
        activated_at = datetime.now(timezone.utc)
        floor = datetime.combine(now.date(), day_time(), timezone.utc) + MODEL_AVAILABLE_OFFSET
        available_at = max(activated_at, floor)
        with connect() as conn:
            conn.execute("UPDATE online_if_model_publications SET model_available_at=%s WHERE publication_id=%s",
                         (available_at, shadow.publication_id))
            conn.execute("""INSERT INTO active_if_publications (symbol,interval,publication_id,activated_at)
                VALUES (%s,%s,%s,%s) ON CONFLICT (symbol,interval)
                DO UPDATE SET publication_id=excluded.publication_id,activated_at=excluded.activated_at""",
                         (symbol,interval,shadow.publication_id,available_at))
            conn.execute("""UPDATE if_training_attempts SET training_completed_at=%s,status='published',publication_id=%s,error_message=NULL
                WHERE symbol=%s AND interval=%s AND scoring_day=%s""",
                         (available_at,shadow.publication_id,symbol,interval,now.date()))
        shadow.model_available_at = available_at
        LOG.info("IF publication active publication_id=%s available_at=%s",shadow.publication_id,shadow.model_available_at)
        return "published"
    except Exception as exc:
        with connect() as conn:
            conn.execute("""UPDATE if_training_attempts SET training_completed_at=%s,status='failed',error_message=%s
                WHERE symbol=%s AND interval=%s AND scoring_day=%s""",(datetime.now(timezone.utc),str(exc)[:2000],symbol,interval,now.date()))
        LOG.exception("IF publication failed; active predecessor retained")
        return "failed"

def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)sZ %(name)s %(levelname)s %(message)s")
    migrate(); ingestion=IngestionConfig.from_env()
    while True:
        publish_today(symbol=ingestion.symbol, interval=ingestion.interval)
        sleep(60)

if __name__ == "__main__": main()
