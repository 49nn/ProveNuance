"""
schemas.py — Request/Response modele FastAPI.
Oddzielone od contracts.py żeby API mogło ewoluować niezależnie.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from contracts import DocumentRef, DocSpan, Fact, Frame, MappingResult, Rule, ValidationIssue


# ─────────────────────────── /ingest ─────────────────────────────

class IngestRequest(BaseModel):
    title: str
    raw_text: str
    source_type: str = "text"
    metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    doc_id: str
    title: str
    version: int
    span_count: int


# ─────────────────────────── /extract ────────────────────────────

class ExtractRequest(BaseModel):
    doc_id: str
    auto_promote: bool = False  # od razu promuje fakty/reguły do 'asserted'
    verbose: bool = False       # wyświetla znalezione ramki na stdout serwera


class FrameResult(BaseModel):
    span_id: str
    surface_text: str
    frames: list[Frame]
    issues: list[ValidationIssue]
    mapping: Optional[MappingResult] = None


class ExtractResponse(BaseModel):
    doc_id: str
    spans_processed: int
    frames_found: int
    facts_created: int
    rules_created: int
    results: list[FrameResult]


# ─────────────────────────── /promote ────────────────────────────

class PromoteRequest(BaseModel):
    fact_ids: list[str] = []
    rule_ids: list[str] = []
    reason: str = "manual promotion"


class PromoteResponse(BaseModel):
    promoted_facts: int
    promoted_rules: int
    errors: list[str]


# ─────────────────────────── /solve ──────────────────────────────

class SolveRequest(BaseModel):
    text: str


class SolveResponse(BaseModel):
    message: str = "Solve endpoint not yet implemented (requires Etap 3 & 4)"


# ─────────────────────────── /proof ──────────────────────────────

class ProofResponse(BaseModel):
    message: str = "Proof endpoint not yet implemented (requires Etap 4)"


# ─────────────────────────── /ner ────────────────────────────────

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    doc_id: str | None = None
    span_id: str | None = None


class NEREntity(BaseModel):
    text: str
    type: str  # OPERATION | PROPERTY | NUMBER | VARIABLE | RELATION | CONCEPT
    start: int
    end: int
    canonical: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class NERResponse(BaseModel):
    span_id: str | None
    text: str
    entities: list[NEREntity]
    frames: list[Frame]
    processing_time_ms: int


# ─────────────────────────── /health ─────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: str
    version: str
