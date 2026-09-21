FROM python:3.12.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.lock .
COPY sql ./sql
RUN mkdir -p /app/models && chown -R 10001:10001 /app
USER 10001:10001
CMD ["python", "-m", "btc_risk.ingestion.collector"]
