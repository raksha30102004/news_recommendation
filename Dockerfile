# ---- base: Python deps shared by the training and runtime stages
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY recsys recsys

# ---- trainer: download MIND-small, evaluate, train ALS (~1.3 GB RAM, ~2-3 min). Only artifacts/ is kept.
FROM base AS trainer
COPY scripts scripts
COPY train.py .
ARG TRAIN_ARGS=""
RUN python scripts/download_data.py && python train.py ${TRAIN_ARGS} && rm -rf data

# ---- runtime: API + static frontend + trained artifacts (~32 MB)
FROM base
COPY --from=trainer /srv/artifacts artifacts
COPY app app
# STATE_DIR holds SQLite (users, clicks, impressions) + retrained models: mount a persistent volume here.
# (world-writable because Hugging Face Spaces runs the container as uid 1000, not root)
ENV STATE_DIR=/data PORT=8000
RUN mkdir -p /data && chmod 777 /data
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
