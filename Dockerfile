FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-bge-chroma.txt ./

ARG INSTALL_BGE_CHROMA=false
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ "$INSTALL_BGE_CHROMA" = "true" ]; then pip install -r requirements-bge-chroma.txt; fi

COPY src ./src
COPY scripts ./scripts
COPY prompts ./prompts
COPY standards ./standards
COPY web ./web
COPY data/jd.txt ./data/jd.txt
COPY data/knowledge ./data/knowledge
COPY data/eval ./data/eval
COPY examples ./examples
COPY README.md ./

RUN mkdir -p data/resumes data/uploads data/vector_index data/chroma_db results logs \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()" || exit 1

CMD ["python", "scripts/web_app.py", "--host", "0.0.0.0", "--port", "8000"]
