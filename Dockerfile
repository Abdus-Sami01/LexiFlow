FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEXIFLOW_HOME=/data \
    LEXIFLOW_MODELS=/data/models

RUN apt-get update \
 && apt-get install -y --no-install-recommends libsndfile1 libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY lexiflow ./lexiflow

RUN pip install --upgrade pip && pip install ".[ui,asr,nlp]"

VOLUME ["/data"]
EXPOSE 8501

ENTRYPOINT ["lexiflow"]
CMD ["dashboard", "--port", "8501", "--address", "0.0.0.0"]
