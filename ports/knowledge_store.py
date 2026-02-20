"""
Port: KnowledgeStore
Odpowiedzialność: przechowywanie trójek i reguł, status hypothesis/asserted,
provenance, snapshoty i diff.
"""
from typing import AsyncIterable, Optional, Protocol, runtime_checkable

from contracts import (
    Fact,
    FactRef,
    Rule,
    RuleRef,
    SnapshotDiff,
    SnapshotRef,
)


@runtime_checkable
class KnowledgeStore(Protocol):
    async def upsert_fact(self, f: Fact) -> FactRef:
        """
        Insert or update a fact by (h, r, t) uniqueness.
        Returns FactRef with created=True if new, False if updated.
        """
        ...

    async def upsert_rule(self, r: Rule) -> RuleRef:
        """
        Insert or update a rule by (head, body) uniqueness.
        Returns RuleRef with created=True if new, False if updated.
        """
        ...

    async def get_fact(self, fact_id: str) -> Fact:
        """Returns a Fact by ID. Raises KeyError if not found."""
        ...

    async def get_rule(self, rule_id: str) -> Rule:
        """Returns a Rule by ID. Raises KeyError if not found."""
        ...

    async def get_asserted_facts(
        self, snapshot: Optional[SnapshotRef] = None
    ) -> AsyncIterable[Fact]:
        """
        Iterates over all currently-asserted facts.
        If snapshot is given, returns facts asserted at that snapshot point.
        """
        ...

    async def get_asserted_rules(
        self, snapshot: Optional[SnapshotRef] = None
    ) -> AsyncIterable[Rule]:
        """
        Iterates over all currently-asserted rules.
        If snapshot is given, returns rules asserted at that snapshot point.
        """
        ...

    async def promote_fact(self, fact_id: str, reason: str) -> None:
        """
        Promotes a hypothesis fact to asserted status.
        Records promotion in fact_history audit log.
        Raises KeyError if fact not found.
        """
        ...

    async def promote_rule(self, rule_id: str, reason: str) -> None:
        """
        Promotes a hypothesis rule to asserted status.
        Records promotion in rule_history audit log.
        Raises KeyError if rule not found.
        """
        ...

    async def retract_fact(self, fact_id: str, reason: str) -> None:
        """
        Retracts an asserted fact (sets status to retracted).
        Records retraction in fact_history audit log.
        """
        ...

    async def create_snapshot(self, label: Optional[str] = None) -> SnapshotRef:
        """
        Creates a snapshot of all currently-asserted facts and rules.
        Returns SnapshotRef identifying the snapshot point.
        """
        ...

    async def diff_snapshots(self, a_id: str, b_id: str) -> SnapshotDiff:
        """
        Computes diff between two snapshots: added, removed, changed facts/rules.
        Raises KeyError if either snapshot is not found.
        """
        ...
