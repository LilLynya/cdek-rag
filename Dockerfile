# syntax=docker/dockerfile:1.7
# Используем готовый образ Astral uv — быстрый и современный Python-package manager.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

WORKDIR /app

# Сначала зависимости — для лучшего кэширования слоёв.
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Затем — исходники и база знаний.
COPY src/ ./src/
COPY data/ ./data/

# Запуск под непривилегированным пользователем.
RUN useradd --create-home --shell /usr/sbin/nologin app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).getcode()==200 else 1)" \
    || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
