"""
Port: Reasoner
Odpowiedzialność: deterministyczne wnioskowanie na faktach/regułach, proof trace.
"""
from typing import Optional, Protocol, runtime_checkable

from contracts import LogicQuery, ReasonerResult, SnapshotRef


@runtime_checkable
class Reasoner(Protocol):
    async def query(
        self,
        q: LogicQuery,
        snapshot: Optional[SnapshotRef] = None,
    ) -> ReasonerResult:
        """
        Resolves a logic query against the asserted facts and rules.
        If snapshot is given, uses that snapshot's fact/rule set.
        Returns ReasonerResult with:
          - answers: list of variable bindings
          - proof: step-by-step ProofTrace
          - used_fact_ids / used_rule_ids: for provenance tracking
        Raises TimeoutError if timeout_ms is exceeded.
        """
        ...
