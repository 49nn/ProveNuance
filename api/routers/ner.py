"""
Router: POST /ner

Wysyła tekst do zewnętrznego serwisu NER i zwraca encje oraz ramki.
Backend NER jest wskazany przez config.ner_backend_url.
"""
from __future__ import annotations

import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from adapters.frame_extractor.llm_extractor import LLMFrameExtractor, LLMNEROutput
from api.dependencies import get_llm_extractor
from api.schemas import NEREntity, NERRequest, NERResponse

router = APIRouter(prefix="/ner", tags=["ner"])


def _to_schema_entities(output: LLMNEROutput) -> list[NEREntity]:
    result = []
    for e in output.entities:
        try:
            result.append(NEREntity(
                text=e.text,
                type=e.type,
                start=e.start,
                end=e.end,
                canonical=e.canonical,
                confidence=e.confidence,
            ))
        except Exception as exc:
            # Pomiń encje z nieprawidłowymi danymi
            pass
    return result


@router.post("", response_model=NERResponse)
async def ner_extract(
    body: NERRequest,
    llm_extractor: LLMFrameExtractor = Depends(get_llm_extractor),
) -> NERResponse:
    t0 = time.monotonic()

    try:
        output = llm_extractor.call_backend(body.text)
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail=f"Backend NER niedostępny: {exc}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Backend NER przekroczył limit czasu")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Backend NER zwrócił błąd: HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Błąd ekstrakcji NER: {exc}")

    span_id = body.span_id or "ner-ephemeral"
    frames = llm_extractor.map_frames(output.frames, span_id)
    entities = _to_schema_entities(output)
    processing_time_ms = int((time.monotonic() - t0) * 1000)

    return NERResponse(
        span_id=body.span_id,
        text=body.text,
        entities=entities,
        frames=frames,
        processing_time_ms=processing_time_ms,
    )
