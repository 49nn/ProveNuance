"""
Port: DocumentStore
Odpowiedzialność: przechowywanie dokumentów i fragmentów (DocSpan), wersjonowanie.
"""
from typing import Optional, Protocol, runtime_checkable

from contracts import DocSpan, DocSpanHit, DocumentIn, DocumentRef


@runtime_checkable
class DocumentStore(Protocol):
    async def ingest_document(self, doc: DocumentIn) -> DocumentRef:
        """
        Ingests a document: persists it, segments into DocSpans and stores them.
        Returns a DocumentRef with assigned doc_id.
        """
        ...

    async def get_document(self, doc_id: str) -> DocumentRef:
        """Returns DocumentRef metadata. Raises KeyError if not found."""
        ...

    async def list_spans(self, doc_id: str) -> list[DocSpan]:
        """Returns all DocSpans for a given document, ordered by start_char."""
        ...

    async def get_span(self, span_id: str) -> DocSpan:
        """Returns a single DocSpan by ID. Raises KeyError if not found."""
        ...

    async def search_spans(
        self,
        query: str,
        doc_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[DocSpanHit]:
        """
        Full-text search over surface_text using BM25-style ranking.
        Optionally scoped to a specific document.
        Returns list of DocSpanHit sorted by descending relevance score.
        """
        ...

    async def delete_document(self, doc_id: str) -> None:
        """Deletes a document and all its spans (cascade). Raises KeyError if not found."""
        ...
