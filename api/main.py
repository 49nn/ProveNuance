"""
api/main.py — punkt wejścia FastAPI.

Lifespan:
  - Tworzy pule asyncpg dla DocumentStore i KnowledgeStore
  - Inicjalizuje adaptery (FrameExtractor, FrameMapper, EntityLinker, Validator)
  - Przy zamknięciu zamyka połączenia do bazy

DSN: config.db_url może mieć prefiks 'postgresql+asyncpg://' (SQLAlchemy-style);
asyncpg oczekuje 'postgresql://'. Prefiks jest tu konwertowany.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from adapters.document_store.postgres_document_store import PostgresDocumentStore
from adapters.entity_linker.dict_entity_linker import DictEntityLinker
from adapters.entity_linker.stanza_linker import StanzaEntityLinker, StanzaConfig
from adapters.frame_extractor.llm_extractor import LLMFrameExtractor
from adapters.frame_extractor.stanza_aware_extractor import StanzaAwareExtractor
from adapters.frame_mapper.math_grade1to3_mapper import MathGrade1to3Mapper
from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore
from adapters.math_problem_parser.llm_parser import LLMMathParser
from adapters.math_problem_parser.regex_parser import RegexMathParser
from adapters.validator.simple_validator import SimpleValidator
from api.routers import extract, ingest, knowledge, ner, promote, proof, solve
from api.schemas import HealthResponse
from config import Settings

logger = logging.getLogger("prove_nuance")


def _asyncpg_dsn(url: str) -> str:
    """Konwertuje 'postgresql+asyncpg://...' → 'postgresql://...'."""
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    dsn = _asyncpg_dsn(settings.db_url)

    logger.info("Connecting to PostgreSQL...")
    app.state.document_store = await PostgresDocumentStore.create(dsn)
    app.state.knowledge_store = await PostgresKnowledgeStore.create(dsn)

    # Adaptery bezstanowe — tworzone raz
    # Jeden linker stanza obsługuje NER + analizę NLP dla frame extractora
    stanza_linker = StanzaEntityLinker(
        config=StanzaConfig(lang="pl", processors="tokenize,pos,lemma,depparse,ner")
    )
    app.state.frame_extractor = StanzaAwareExtractor(linker=stanza_linker)
    app.state.frame_mapper = MathGrade1to3Mapper()
    app.state.entity_linker = DictEntityLinker()
    app.state.validator = SimpleValidator()
    app.state.llm_extractor = LLMFrameExtractor(
        ner_backend_url=settings.ner_backend_url,
        timeout_ms=settings.ner_timeout_ms,
    )
    app.state.math_problem_parser = LLMMathParser(
        backend_url=settings.math_parser_backend_url,
        timeout_ms=settings.math_parser_timeout_ms,
        fallback_parser=RegexMathParser(),
    )

    logger.info("ProveNuance API ready.")
    yield

    logger.info("Shutting down — closing DB pools.")
    await app.state.document_store.close()
    await app.state.knowledge_store.close()


def create_app() -> FastAPI:
    settings = Settings()
    logging.basicConfig(level=settings.log_level.upper())

    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = settings

    # Routers
    app.include_router(ingest.router)
    app.include_router(extract.router)
    app.include_router(ner.router)
    app.include_router(promote.router)
    app.include_router(solve.router)
    app.include_router(proof.router)
    app.include_router(knowledge.router)

    # Health
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health(request: Request):
        db_status = "ok"
        try:
            pool = request.app.state.document_store._pool
            await pool.fetchval("SELECT 1")
        except Exception as e:
            db_status = f"error: {e}"

        return HealthResponse(
            status="ok" if db_status == "ok" else "degraded",
            db=db_status,
            version=settings.app_version,
        )

    # Globalny handler błędów
    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    return app


app = create_app()
