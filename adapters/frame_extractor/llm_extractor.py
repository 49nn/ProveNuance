"""
LLMFrameExtractor — adapter FrameExtractor delegujący do zewnętrznego serwisu NER.

Kontrakt backendu:
    POST {ner_backend_url}
    Content-Type: text/plain; charset=utf-8
    Body: surowy tekst

    ← 200 OK, Content-Type: application/json
    Body: {"entities": [...], "frames": [...], "unmapped": [...]}

Adapter implementuje port FrameExtractor (ports/frame_extractor.py).
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from adapters.frame_extractor._printer import print_frames
from contracts import (
    ArithExampleFrame,
    DefinitionFrame,
    DocSpan,
    Frame,
    ProcedureFrame,
    PropertyFrame,
    ValidationIssue,
)

logger = logging.getLogger("prove_nuance.llm_extractor")


# ─── Wewnętrzne modele odpowiedzi backendu ────────────────────────────────────

class _NEREntity(BaseModel):
    text: str
    type: str
    start: int
    end: int
    canonical: str
    confidence: float = 1.0


class LLMNEROutput(BaseModel):
    entities: list[_NEREntity]
    frames: list[dict[str, Any]]
    unmapped: list[str] = []


# ─── Adapter ──────────────────────────────────────────────────────────────────

class LLMFrameExtractor:
    """
    Wysyła tekst do zewnętrznego serwisu NER przez HTTP i mapuje wynik
    na typy Frame z contracts.py.
    """

    def __init__(
        self,
        ner_backend_url: str,
        timeout_ms: int = 10_000,
        verbose: bool = False,
    ) -> None:
        self._url = ner_backend_url
        self._timeout = timeout_ms / 1000.0
        self._verbose = verbose

    # ── Wywołanie backendu ────────────────────────────────────────────────────

    def call_backend(self, text: str) -> LLMNEROutput:
        """POST surowy tekst do backendu NER, zwraca sparsowany wynik."""
        response = httpx.post(
            self._url,
            content=text.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return LLMNEROutput.model_validate(response.json())

    # ── Normalizacja pól ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(raw: dict[str, Any], frame_type: str) -> dict[str, Any]:
        """Mapuje aliasy pól z LLM na nazwy wymagane przez contracts.py."""
        d = dict(raw)
        if frame_type == "PROPERTY":
            # subject: source_entity → subject
            if "subject" not in d and "source_entity" in d:
                d["subject"] = str(d.pop("source_entity"))
            # property_name: property_type / name / type → property_name
            for alias in ("property_type", "name", "type"):
                if "property_name" not in d and alias in d:
                    d["property_name"] = d.pop(alias)
                    break
            # value: domyślnie True jeśli brak
            d.setdefault("value", True)
        elif frame_type == "ARITH_EXAMPLE":
            # operands: numbers / args → operands
            for alias in ("numbers", "args", "arguments"):
                if "operands" not in d and alias in d:
                    d["operands"] = d.pop(alias)
                    break
            # result: answer / output / value → result
            for alias in ("answer", "output", "value"):
                if "result" not in d and alias in d:
                    d["result"] = d.pop(alias)
                    break
            # operation: op / action → operation
            for alias in ("op", "action"):
                if "operation" not in d and alias in d:
                    d["operation"] = d.pop(alias)
                    break
        elif frame_type == "PROCEDURE":
            # steps: instructions / items → steps
            for alias in ("instructions", "items"):
                if "steps" not in d and alias in d:
                    d["steps"] = d.pop(alias)
                    break
        elif frame_type == "DEFINITION":
            # term: concept / word / name / entity → term
            for alias in ("concept", "word", "name", "entity"):
                if "term" not in d and alias in d:
                    d["term"] = d.pop(alias)
                    break
            # definition: description / meaning / text / body → definition
            for alias in ("description", "meaning", "text", "body"):
                if "definition" not in d and alias in d:
                    d["definition"] = d.pop(alias)
                    break
        return d

    # ── Mapowanie ramek ───────────────────────────────────────────────────────

    def map_frames(self, raw_frames: list[dict[str, Any]], span_id: str) -> list[Frame]:
        """Konwertuje surowe dicts z backendu na obiekty Frame z contracts.py."""
        result: list[Frame] = []
        for raw in raw_frames:
            ft = raw.get("frame_type")
            payload = {**self._normalize(raw, ft or ""), "source_span_id": span_id}
            try:
                if ft == "ARITH_EXAMPLE":
                    result.append(ArithExampleFrame.model_validate(payload))
                elif ft == "PROPERTY":
                    result.append(PropertyFrame.model_validate(payload))
                elif ft == "PROCEDURE":
                    result.append(ProcedureFrame.model_validate(payload))
                elif ft == "DEFINITION":
                    result.append(DefinitionFrame.model_validate(payload))
                else:
                    logger.warning("Nieznany frame_type z backendu NER: %r", ft)
            except ValidationError as exc:
                logger.warning("Błąd walidacji ramki: %s | raw=%r", exc, raw)
        return result

    # ── Port FrameExtractor ───────────────────────────────────────────────────

    def extract_frames(self, span: DocSpan) -> list[Frame]:
        output = self.call_backend(span.surface_text)
        frames = self.map_frames(output.frames, span.span_id)
        if self._verbose:
            print_frames(span.surface_text, frames)
        return frames

    def validate_frame(self, frame: Frame) -> list[ValidationIssue]:
        # Ramki przeszły już walidację Pydantic przy parsowaniu odpowiedzi backendu.
        return []
