"""
NerLink — wewnętrzne modele danych (izolowane od contracts.py).
Odpowiadają sekcjom 3.1, 3.2 i 5.x specyfikacji.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


# ── Wejście ──────────────────────────────────────────────────────────────────


class Mention(BaseModel):
    mention_id: str
    start: int          # offset znakowy, inclusive
    end: int            # offset znakowy, exclusive
    text: str
    label: str          # PERSON / ORG / LAW / TERM / ...


# ── Wyniki pośrednie ──────────────────────────────────────────────────────────


class FlexionAnalysis(BaseModel):
    surface: str
    candidates: list[str]           # zawsze co najmniej [surface]
    features: dict[str, str] = {}   # np. {"case": "GEN", "number": "SG"}


class DictMatch(BaseModel):
    matched: bool
    entity_id: Optional[str] = None
    matched_alias: Optional[str] = None
    confidence: float = 1.0
    meta: dict = {}


# ── Wyjście ───────────────────────────────────────────────────────────────────


class Span(BaseModel):
    start: int
    end: int
    text: str


class ResolvedEntity(BaseModel):
    mention_id: str
    span: Span
    label: str
    status: Literal["linked", "new", "ambiguous"]
    entity_id: Optional[str] = None
    canonical_name: Optional[str] = None
    confidence: float               # 0..1
    method: str                     # dict / fuzzy / emb / dict+fuzzy / fuzzy+emb / new
    alias_to_add: Optional[str] = None


class AliasUpdate(BaseModel):
    entity_id: str
    alias: str
    mention_id: str
    alias_type: str = "surface"


class NewEntitySpec(BaseModel):
    entity_id: str
    label: str
    canonical_name: str
    aliases: list[str]


class LightResult(BaseModel):
    doc_id: str
    entities: list[ResolvedEntity]
    aliases_to_add: list[AliasUpdate] = []
    entities_to_create: list[NewEntitySpec] = []
