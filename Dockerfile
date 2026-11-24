# API + chat UI. CPU-only PyTorch keeps the image small; bge-m3 (2.2 GB) is downloaded on
# first start into the Hugging Face cache volume, not baked into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    DATA_DIR=/app/data

WORKDIR /app

# CPU wheel first, so sentence-transformers does not pull the CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Data the app reads at run time (the PDFs and the interim lines are not needed).
COPY data/processed ./data/processed
COPY data/amendments.json ./data/amendments.json

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /models/hf /app/data/cache \
    && chown -R app:app /models /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "vnlaw_rag.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
