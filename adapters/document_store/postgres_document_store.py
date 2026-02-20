"""
PostgresDocumentStore — implementacja portu DocumentStore na PostgreSQL.

Schema jest tworzona automatycznie przy pierwszym połączeniu (apply_schema).
Używa asyncpg bezpośrednio (bez ORM) dla przejrzystości i wydajności.

Full-text search: PostgreSQL tsvector + plainto_tsquery (język 'simple'
żeby działało i po polsku, i po angielsku bez stemming issues).
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import asyncpg

from adapters.text_cleaner import clean_markup
from contracts import DocSpan, DocSpanHit, DocumentIn, DocumentRef

# Wyrażenie do segmentacji na zdania (MVP: split po ". " / "! " / "? ")
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

# Nagłówki Markdown: # / ## / ### / itd.
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

# DDL — tworzone przy starcie jeśli nie istnieją
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    source_type TEXT NOT NULL DEFAULT 'text',
    raw_text    TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}',
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS doc_spans (
    span_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id       UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    version      INTEGER NOT NULL,
    surface_text TEXT NOT NULL,
    start_char   INTEGER NOT NULL,
    end_char     INTEGER NOT NULL,
    span_type    TEXT NOT NULL DEFAULT 'sentence',
    metadata     JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_doc_spans_doc_id
    ON doc_spans(doc_id);

CREATE INDEX IF NOT EXISTS idx_doc_spans_fts
    ON doc_spans USING gin(to_tsvector('simple', surface_text));
"""


class PostgresDocumentStore:
    """Implementacja DocumentStore na PostgreSQL + asyncpg."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    # ──────────────────────── Lifecycle ──────────────────────────────────

    @classmethod
    async def create(cls, dsn: str) -> "PostgresDocumentStore":
        """Factory: tworzy pool połączeń i aplikuje schemat."""
        pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        store = cls(pool)
        await store._apply_schema()
        return store

    async def _apply_schema(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(_SCHEMA_SQL)

    async def close(self) -> None:
        await self._pool.close()

    # ──────────────────────── Ingest ─────────────────────────────────────

    async def ingest_document(self, doc: DocumentIn) -> DocumentRef:
        """
        Zapisuje dokument i segmentuje raw_text na spany (zdania).
        Zwraca DocumentRef z przydzielonym doc_id.
        """
        doc_id = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc)

        if doc.source_type == "markdown":
            spans = _segment_markdown(doc.raw_text, doc_id, version=1)
        else:
            spans = _segment_sentences(doc.raw_text, doc_id, version=1)

        for span in spans:
            span.surface_text = clean_markup(span.surface_text)
            if span.span_type == "section" and span.metadata.get("heading"):
                span.metadata["heading"] = clean_markup(span.metadata["heading"])

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO documents
                        (doc_id, title, version, source_type, raw_text, metadata, ingested_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    doc_id,
                    doc.title,
                    1,
                    doc.source_type,
                    doc.raw_text,
                    json.dumps(doc.metadata),
                    now,
                )
                if spans:
                    await conn.executemany(
                        """
                        INSERT INTO doc_spans
                            (span_id, doc_id, version, surface_text,
                             start_char, end_char, span_type, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        [
                            (
                                s.span_id,
                                s.doc_id,
                                s.version,
                                s.surface_text,
                                s.start_char,
                                s.end_char,
                                s.span_type,
                                json.dumps(s.metadata),
                            )
                            for s in spans
                        ],
                    )

        return DocumentRef(doc_id=doc_id, title=doc.title, version=1, ingested_at=now)

    # ──────────────────────── Read ────────────────────────────────────────

    async def get_document(self, doc_id: str) -> DocumentRef:
        """Zwraca metadata dokumentu. Rzuca KeyError jeśli nie znaleziono."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT doc_id, title, version, ingested_at FROM documents WHERE doc_id = $1",
                doc_id,
            )
        if row is None:
            raise KeyError(f"Document not found: {doc_id}")
        return DocumentRef(
            doc_id=str(row["doc_id"]),
            title=row["title"],
            version=row["version"],
            ingested_at=row["ingested_at"],
        )

    async def list_spans(self, doc_id: str) -> list[DocSpan]:
        """Zwraca wszystkie spany dokumentu posortowane po start_char."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT span_id, doc_id, version, surface_text,
                       start_char, end_char, span_type, metadata
                FROM doc_spans
                WHERE doc_id = $1
                ORDER BY start_char
                """,
                doc_id,
            )
        return [_row_to_span(r) for r in rows]

    async def get_span(self, span_id: str) -> DocSpan:
        """Zwraca pojedynczy DocSpan. Rzuca KeyError jeśli nie znaleziono."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT span_id, doc_id, version, surface_text,
                       start_char, end_char, span_type, metadata
                FROM doc_spans
                WHERE span_id = $1
                """,
                span_id,
            )
        if row is None:
            raise KeyError(f"Span not found: {span_id}")
        return _row_to_span(row)

    # ──────────────────────── Search ─────────────────────────────────────

    async def search_spans(
        self,
        query: str,
        doc_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[DocSpanHit]:
        """
        Full-text search przez tsvector/plainto_tsquery.
        Wyniki posortowane malejąco po ts_rank.
        """
        if not query.strip():
            return []

        base_sql = """
            SELECT
                span_id, doc_id, version, surface_text,
                start_char, end_char, span_type, metadata,
                ts_rank(to_tsvector('simple', surface_text),
                        plainto_tsquery('simple', $1)) AS rank
            FROM doc_spans
            WHERE to_tsvector('simple', surface_text)
                  @@ plainto_tsquery('simple', $1)
        """
        params: list = [query]

        if doc_id:
            base_sql += " AND doc_id = $2 ORDER BY rank DESC LIMIT $3"
            params += [doc_id, limit]
        else:
            base_sql += " ORDER BY rank DESC LIMIT $2"
            params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(base_sql, *params)

        return [
            DocSpanHit(
                span=_row_to_span(r),
                score=min(1.0, float(r["rank"])),
            )
            for r in rows
        ]

    # ──────────────────────── Delete ─────────────────────────────────────

    async def delete_document(self, doc_id: str) -> None:
        """Usuwa dokument i wszystkie jego spany (CASCADE). Rzuca KeyError jeśli nie znaleziono."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM documents WHERE doc_id = $1", doc_id
            )
        if result == "DELETE 0":
            raise KeyError(f"Document not found: {doc_id}")


# ──────────────────────── Helpers ────────────────────────────────────────

def _segment_markdown(
    text: str, doc_id: str, version: int
) -> list[DocSpan]:
    """
    Dzieli tekst Markdown na sekcje wg nagłówków (#…######).
    Każda sekcja (nagłówek + treść aż do następnego nagłówka) staje się
    osobnym DocSpan z span_type='section' i metadata:
      {"heading": "Tytuł sekcji", "level": 2}

    Tekst przed pierwszym nagłówkiem (preambuła) trafia jako sekcja
    z heading=None, level=0.
    Jeśli brak jakichkolwiek nagłówków — cały tekst jako jedna sekcja.
    """
    spans: list[DocSpan] = []
    matches = list(_HEADING_RE.finditer(text))

    def _make_span(surface: str, start: int, end: int,
                   heading: str | None, level: int) -> DocSpan | None:
        surface = surface.strip()
        if not surface:
            return None
        return DocSpan(
            span_id=str(uuid.uuid4()),
            doc_id=doc_id,
            version=version,
            surface_text=surface,
            start_char=start,
            end_char=end,
            span_type="section",
            metadata={"heading": heading, "level": level},
        )

    if not matches:
        # Brak nagłówków — cały dokument jako jedna sekcja
        span = _make_span(text, 0, len(text), heading=None, level=0)
        return [span] if span else []

    # Preambuła przed pierwszym nagłówkiem
    preamble = _make_span(
        text[:matches[0].start()], 0, matches[0].start(),
        heading=None, level=0,
    )
    if preamble:
        spans.append(preamble)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = _make_span(text[start:end], start, end, heading, level)
        if span:
            spans.append(span)

    return spans


def _segment_sentences(
    text: str, doc_id: str, version: int
) -> list[DocSpan]:
    """
    Dzieli tekst na zdania regexem (?<=[.!?])\\s+.
    Każde zdanie staje się osobnym DocSpan.
    """
    spans: list[DocSpan] = []
    cursor = 0
    for sentence in _SENTENCE_RE.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            start = text.index(sentence, cursor)
        except ValueError:
            continue
        end = start + len(sentence)
        spans.append(
            DocSpan(
                span_id=str(uuid.uuid4()),
                doc_id=doc_id,
                version=version,
                surface_text=sentence,
                start_char=start,
                end_char=end,
                span_type="sentence",
            )
        )
        cursor = end
    return spans


def _row_to_span(row: asyncpg.Record) -> DocSpan:
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return DocSpan(
        span_id=str(row["span_id"]),
        doc_id=str(row["doc_id"]),
        version=row["version"],
        surface_text=row["surface_text"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        span_type=row["span_type"],
        metadata=metadata or {},
    )
