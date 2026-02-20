"""
Port: RuleBootstrap
Odpowiedzialność: bootstrap kandydatów reguł Horn z asserted facts
oraz opcjonalna promocja do asserted_rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from contracts import Fact
from ports.knowledge_store import KnowledgeStore

DEFAULT_RELATION_WHITELIST = frozenset(
    {
        "eq",
        "instance_of",
        "is_a",
        "has_property",
        "defined_as",
        "add",
        "sub",
        "mul",
        "div",
    }
)


@dataclass
class BootstrapConfig:
    relation_whitelist: set[str] = field(
        default_factory=lambda: set(DEFAULT_RELATION_WHITELIST)
    )
    max_body_literals: int = 2
    min_coverage: int = 3
    min_support: int = 3
    min_confidence: float = 0.95
    min_lift: float = 1.0
    max_violations: int = 0
    max_corruption_hits: int = 0
    max_local_cwa_negatives: int = 0
    corruption_samples: int = 2
    use_local_cwa: bool = True
    functional_relations: set[str] = field(default_factory=lambda: {"eq"})
    max_groundings_per_pattern: int = 5000
    top_k: int = 100
    priority: int = 20
    store_candidates: bool = True
    promote: bool = True
    promotion_reason: str = "rule_bootstrap"
    dry_run: bool = False


@dataclass
class CandidateRule:
    head_relation: str
    body_relations: tuple[str, ...]
    head: str
    body: tuple[str, ...]
    coverage: int
    support: int
    confidence: float
    lift: float
    base_rate: float
    corruption_hits: int
    local_cwa_negatives: int
    violations: int
    sample_grounding: str | None
    promotable: bool
    rejection_reasons: list[str]
    provenance: list[str] = field(default_factory=list)
    rule_id: str | None = None
    promoted: bool = False


@dataclass
class BootstrapSummary:
    asserted_fact_count: int
    relation_count: int
    pattern_count: int
    candidate_count: int
    stored_created: int
    stored_updated: int
    promoted: int
    candidates: list[CandidateRule]


@runtime_checkable
class RuleBootstrap(Protocol):
    def mine_candidates(
        self, facts: list[Fact]
    ) -> tuple[list[CandidateRule], int, int, int]:
        """
        Learns candidate rules from asserted facts.
        Returns (candidates, asserted_fact_count, relation_count, pattern_count).
        """
        ...

    async def bootstrap(self, store: KnowledgeStore) -> BootstrapSummary:
        """
        Runs full bootstrap pipeline on KnowledgeStore:
        read asserted facts -> mine/score -> optionally upsert/promote rules.
        """
        ...
