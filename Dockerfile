FROM python:3.12-slim

WORKDIR /app

# curl do healthchecka
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user
RUN adduser --disabled-password --gecos "" provenuance \
    && chown -R provenuance:provenuance /app
USER provenuance

EXPOSE 8000

# workers=4 bezpieczne z asyncpg (connection pool per worker)
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info"]
