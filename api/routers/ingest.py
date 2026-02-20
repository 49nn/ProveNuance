"""
Router: POST /ingest
Przyjmuje dokument tekstowy, segmentuje na spany i zapisuje do DocumentStore.
"""
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_document_store
from api.schemas import IngestRequest, IngestResponse
from contracts import DocumentIn

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: str,
    store=Depends(get_document_store),
) -> None:
    try:
        await store.delete_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")


@router.post("", response_model=IngestResponse)
async def ingest_document(
    body: IngestRequest,
    store=Depends(get_document_store),
) -> IngestResponse:
    doc_in = DocumentIn(
        title=body.title,
        raw_text=body.raw_text,
        source_type=body.source_type,
        metadata=body.metadata,
    )
    doc_ref = await store.ingest_document(doc_in)
    spans = await store.list_spans(doc_ref.doc_id)
    return IngestResponse(
        doc_id=doc_ref.doc_id,
        title=doc_ref.title,
        version=doc_ref.version,
        span_count=len(spans),
    )
