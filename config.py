"""
config.py — Konfiguracja aplikacji przez zmienne środowiskowe.
Wszystkie zmienne mają prefiks PROVE_NUANCE_.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Postgres (domyślnie lokalne dev; w Dockerze nadpisywane przez env)
    db_url: str = "postgresql+asyncpg://prove_nuance:prove_nuance_secret@localhost:5432/prove_nuance"

    # Reasoner
    max_reasoner_depth: int = 30
    reasoner_timeout_ms: int = 5000

    # Logging
    log_level: str = "INFO"

    # NER backend service
    ner_backend_url: str = "https://n8n.49nn.eu/webhook/ner"
    ner_timeout_ms: int = 10_000

    # App
    app_title: str = "ProveNuance"
    app_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_prefix="PROVE_NUANCE_", env_file=".env", extra="ignore")
