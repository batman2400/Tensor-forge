# Serving image. The encoder files are the final fit, checked by the manifest hashes.
FROM python:3.12-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.12-slim

# tokenizers 0.23.2 depends on huggingface-hub. These offline flags, and a
# --network none run, keep that library from fetching weights. transformers
# is not installed.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data

WORKDIR /opt/tensorforge

COPY --from=build /opt/venv /opt/venv
COPY app ./app
COPY ui ./ui
COPY ml/__init__.py ml/truncation.py ./ml/
COPY artifacts/classical.joblib artifacts/fusion.json artifacts/manifest.json artifacts/encoder.int8.onnx artifacts/tokenizer.json artifacts/encoder_meta.json ./artifacts/

RUN python -c "from app.inference import Engine; engine = Engine.load(); assert engine.encoder is not None; assert str(engine.model_version).startswith('v1.0.0-'); pred = engine.predict({'channel': 'chat', 'subject': 'refund', 'text': 'my payment was charged twice'}); required = {'category', 'secondary_category', 'team', 'is_urgent', 'confidence', 'model_version'}; missing = required - set(pred); assert not missing, missing; assert pred['model_version'] == engine.model_version"

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "app.main"]
