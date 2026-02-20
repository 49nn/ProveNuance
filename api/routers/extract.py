"""
Router: POST /extract
Dla każdego spanu dokumentu:
  1. Wyciąga ramki (FrameExtractor)
  2. Waliduje ramki
  3. Mapuje na fakty/reguły (FrameMapper)
  4. Zapisuje do KnowledgeStore (jako hypothesis)
  5. Opcjonalnie promuje od razu (auto_promote=True)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from adapters.frame_extractor._printer import print_frames
from api.dependencies import (
    get_document_store,
    get_frame_extractor,
    get_frame_mapper,
    get_knowledge_store,
)
from api.schemas import ExtractRequest, ExtractResponse, FrameResult
from contracts import Provenance

router = APIRouter(prefix="/extract", tags=["extract"])


@router.post("", response_model=ExtractResponse)
async def extract_frames(
    body: ExtractRequest,
    doc_store=Depends(get_document_store),
    knowledge_store=Depends(get_knowledge_store),
    extractor=Depends(get_frame_extractor),
    mapper=Depends(get_frame_mapper),
) -> ExtractResponse:
    # Pobierz spany dokumentu
    try:
        spans = await doc_store.list_spans(body.doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Document not found: {body.doc_id}")

    if not spans:
        raise HTTPException(status_code=404, detail=f"No spans found for doc_id: {body.doc_id}")

    results: list[FrameResult] = []
    total_frames = 0
    total_facts = 0
    total_rules = 0

    for span in spans:
        frames = extractor.extract_frames(span)
        if body.verbose:
            print_frames(span.surface_text, frames)
        all_issues = []
        mapping_combined = None

        facts_this_span = 0
        rules_this_span = 0

        for frame in frames:
            issues = extractor.validate_frame(frame)
            all_issues.extend(issues)

            # Nie mapuj jeśli są błędy krytyczne
            has_errors = any(i.severity == "error" for i in issues)
            if has_errors:
                continue

            provenance = Provenance(
                span_ids=[span.span_id],
                doc_ref=body.doc_id,
                extractor_version="1.0.0",
            )
            mapping = mapper.map(frame, provenance)

            # Zapisz fakty
            for fact in mapping.facts:
                fact_ref = await knowledge_store.upsert_fact(fact)
                facts_this_span += 1
                if body.auto_promote:
                    await knowledge_store.promote_fact(fact_ref.fact_id, reason="auto_promote")

            # Zapisz reguły
            for rule in mapping.rules:
                rule_ref = await knowledge_store.upsert_rule(rule)
                rules_this_span += 1
                if body.auto_promote:
                    await knowledge_store.promote_rule(rule_ref.rule_id, reason="auto_promote")

            # Zbierz mapowania do odpowiedzi (ostatnie jest widoczne — wystarczy dla MVP)
            mapping_combined = mapping

        if frames or all_issues:
            results.append(FrameResult(
                span_id=span.span_id,
                surface_text=span.surface_text,
                frames=frames,
                issues=all_issues,
                mapping=mapping_combined,
            ))

        total_frames += len(frames)
        total_facts += facts_this_span
        total_rules += rules_this_span

    return ExtractResponse(
        doc_id=body.doc_id,
        spans_processed=len(spans),
        frames_found=total_frames,
        facts_created=total_facts,
        rules_created=total_rules,
        results=results,
    )
