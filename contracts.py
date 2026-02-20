"""
contracts.py — Jedyne źródło prawdy dla wszystkich typów danych w ProveNuance.
Wszystkie moduły importują WYŁĄCZNIE stąd. Nie modyfikować bez versioning.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
from typing import Any, Iterable, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

CONTRACTS_VERSION = "1.0.0"


# ─────────────────────────── Helpers ─────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


# ─────────────────────────── EntityLinker ────────────────────────────────

class EntityRef(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str  # "number", "operation", "concept", "object"


# ─────────────────────────── Provenance ──────────────────────────────────

class Provenance(BaseModel):
    span_ids: list[str]
    doc_ref: str
    extraction_timestamp: datetime = Field(default_factory=_now)
    extractor_version: str = "1.0.0"


# ─────────────────────────── DocumentStore ───────────────────────────────

class DocumentIn(BaseModel):
    title: str
    source_type: Literal["pdf", "text", "markdown"] = "text"
    raw_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRef(BaseModel):
    doc_id: str = Field(default_factory=_new_id)
    title: str
    version: int = 1
    ingested_at: datetime = Field(default_factory=_now)


class DocSpan(BaseModel):
    span_id: str = Field(default_factory=_new_id)
    doc_id: str
    version: int
    surface_text: str
    start_char: int
    end_char: int
    span_type: Literal["sentence", "paragraph", "section"] = "sentence"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocSpanHit(BaseModel):
    span: DocSpan
    score: float  # 0.0–1.0 BM25-style relevance


# ─────────────────────────── Frame DSL ───────────────────────────────────

class FrameType(str, Enum):
    ARITH_EXAMPLE = "ARITH_EXAMPLE"  # np. "3 + 4 = 7"
    PROPERTY = "PROPERTY"            # np. "dodawanie jest przemienne"
    PROCEDURE = "PROCEDURE"          # np. "aby dodać: najpierw..."
    DEFINITION = "DEFINITION"        # np. "dodawanie to łączenie dwóch liczb"
    TASK = "TASK"                    # np. "Oblicz: 2 + 3"
    CONDITION = "CONDITION"          # np. "Jeśli a > 0, to a jest dodatnie"


class ArithExampleFrame(BaseModel):
    frame_type: Literal[FrameType.ARITH_EXAMPLE] = FrameType.ARITH_EXAMPLE
    operation: Literal["add", "sub", "mul", "div"]
    operands: list[Union[int, str]]  # int = literał, str = zmienna
    result: Union[int, str]
    source_span_id: str


class PropertyFrame(BaseModel):
    frame_type: Literal[FrameType.PROPERTY] = FrameType.PROPERTY
    subject: str          # "addition", "subtraction"
    property_name: str    # "commutative", "associative", "zero_neutral"
    value: Any = True
    source_span_id: str


class ProcedureFrame(BaseModel):
    frame_type: Literal[FrameType.PROCEDURE] = FrameType.PROCEDURE
    operation: str
    steps: list[str]
    source_span_id: str


class DefinitionFrame(BaseModel):
    frame_type: Literal[FrameType.DEFINITION] = FrameType.DEFINITION
    term: str          # pojęcie definiowane, po angielsku: "add", "sum", "subtraction"
    definition: str    # treść definicji (dowolny język)
    source_span_id: str


class TaskFrame(BaseModel):
    frame_type: Literal[FrameType.TASK] = FrameType.TASK
    verb: str                          # czasownik polecenia, np. "oblicz"
    target: str                        # co obliczyć / uzupełnić
    expr_ast: Optional[ExprAST] = None # AST wyrażenia (jeśli udało się sparsować)
    source_span_id: str


class ConditionFrame(BaseModel):
    frame_type: Literal[FrameType.CONDITION] = FrameType.CONDITION
    condition: str    # człon warunkowy ("a > 0")
    conclusion: str   # człon wynikowy ("a jest dodatnie")
    source_span_id: str


Frame = Union[
    ArithExampleFrame, PropertyFrame, ProcedureFrame,
    DefinitionFrame, TaskFrame, ConditionFrame,
]


class HypothesisFrame(BaseModel):
    """Wrapper dla frame'u, który nie przeszedł walidacji — wymaga ręcznej korekty."""
    frame: Frame
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    issues: list[ValidationIssue] = Field(default_factory=list)
    needs_review: bool = True


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning", "info"]
    code: str      # np. "MISSING_RESULT", "TYPE_MISMATCH", "UNKNOWN_OP"
    message: str
    field_path: Optional[str] = None


# ─────────────────────────── KnowledgeStore ──────────────────────────────

class FactStatus(str, Enum):
    HYPOTHESIS = "hypothesis"
    ASSERTED = "asserted"
    RETRACTED = "retracted"


class Fact(BaseModel):
    fact_id: str = Field(default_factory=_new_id)
    h: str         # head / subject (entity_id lub literał)
    r: str         # relacja
    t: str         # tail / object (entity_id lub literał)
    status: FactStatus = FactStatus.HYPOTHESIS
    provenance: list[str] = Field(default_factory=list)  # span_ids
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class FactRef(BaseModel):
    fact_id: str
    created: bool  # True = nowy, False = zaktualizowany


class Rule(BaseModel):
    rule_id: str = Field(default_factory=_new_id)
    head: str              # atom w stylu Prologu: "add(X,Y,Z)"
    body: list[str]        # lista atomów; pusta = fakt-reguła
    status: FactStatus = FactStatus.HYPOTHESIS
    provenance: list[str] = Field(default_factory=list)
    priority: int = 0      # wyższy = stosowany pierwszy
    created_at: datetime = Field(default_factory=_now)


class RuleRef(BaseModel):
    rule_id: str
    created: bool


class SnapshotRef(BaseModel):
    snapshot_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    label: Optional[str] = None


class FactDiff(BaseModel):
    added: list[Fact]
    removed: list[Fact]
    changed: list[tuple[Fact, Fact]]  # (przed, po)


class RuleDiff(BaseModel):
    added: list[Rule]
    removed: list[Rule]


class SnapshotDiff(BaseModel):
    snapshot_a: str
    snapshot_b: str
    fact_diff: FactDiff
    rule_diff: RuleDiff


# ─────────────────────────── FrameMapper ─────────────────────────────────

class MappingResult(BaseModel):
    facts: list[Fact]
    rules: list[Rule]
    unmapped_slots: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ─────────────────────────── Validator ───────────────────────────────────

class ConflictSet(BaseModel):
    fact_ids: list[str]
    conflict_type: str    # np. "CONTRADICTORY_VALUES", "FUNCTIONAL_VIOLATION"
    description: str


# ─────────────────────────── Reasoner ────────────────────────────────────

class LogicQuery(BaseModel):
    goal: str              # np. "add(3,4,?Z)" lub "add(3,4,Z)"
    variables: list[str] = Field(default_factory=list)  # np. ["Z"]
    max_depth: int = 20
    timeout_ms: int = 5000


class ProofStep(BaseModel):
    step_id: int
    rule_or_fact_id: str
    substitution: dict[str, str]  # {"X": "3", "Y": "4", "Z": "7"}
    conclusion: str               # np. "add(3,4,7)"
    span_citations: list[str]     # provenance span_ids użyte w tym kroku


class ReasonerResult(BaseModel):
    query_id: str = Field(default_factory=_new_id)
    goal: str
    success: bool
    answers: list[dict[str, str]]   # lista podstawień zmiennych
    proof: list[ProofStep]
    used_fact_ids: list[str]
    used_rule_ids: list[str]
    depth_reached: int
    duration_ms: float


# ─────────────────────────── MathProblemParser ───────────────────────────

class ProblemType(str, Enum):
    EXPR = "expr"                   # "3 + 4" — bezpośrednie wyrażenie
    WORD_PROBLEM = "word_problem"   # "Mary ma 5 jabłek..."
    UNKNOWN = "unknown"


# AST dla wyrażeń arytmetycznych

class NumberNode(BaseModel):
    node_type: Literal["number"] = "number"
    value: int


class VariableNode(BaseModel):
    node_type: Literal["variable"] = "variable"
    name: str


class BinOpNode(BaseModel):
    node_type: Literal["binop"] = "binop"
    op: Literal["+", "-", "*", "/", "×", "÷"]
    left: "ExprAST"
    right: "ExprAST"


class UnaryOpNode(BaseModel):
    node_type: Literal["unary"] = "unary"
    op: Literal["-"]
    operand: "ExprAST"


ExprAST = Union[NumberNode, VariableNode, BinOpNode, UnaryOpNode]
BinOpNode.model_rebuild()
UnaryOpNode.model_rebuild()


class ParsedProblem(BaseModel):
    problem_id: str = Field(default_factory=_new_id)
    original_text: str
    problem_type: ProblemType
    expr_ast: Optional[ExprAST] = None
    logic_query: Optional[LogicQuery] = None
    slots: dict[str, Any] = Field(default_factory=dict)
    entities_mentioned: list[str] = Field(default_factory=list)
    extracted_numbers: list[int] = Field(default_factory=list)
    operation_hint: Optional[Literal["add", "sub", "mul", "div"]] = None


# ─────────────────────────── Evaluator ───────────────────────────────────

class EvalResult(BaseModel):
    value: Union[int, float, str]   # str dla wyników symbolicznych
    is_exact: bool = True
    steps: list[str] = Field(default_factory=list)  # czytelne kroki


# ─────────────────────────── API response ────────────────────────────────

class SolveResponse(BaseModel):
    problem_id: str
    original_text: str
    result: Union[int, float, str]
    short_proof: str           # 1–3 zdania narracji
    proof_steps: list[ProofStep]
    citations: list[DocSpan]   # dosłowny tekst każdego cytowanego spanu
    used_fact_ids: list[str]
    used_rule_ids: list[str]
    snapshot_id: str
    duration_ms: float
