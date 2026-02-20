"""
Port: Validator
Odpowiedzialność: wykrywanie błędów mapowania i konfliktów między faktami.
"""
from typing import Any, Protocol, runtime_checkable

from contracts import ConflictSet, Fact, Rule, ValidationIssue


@runtime_checkable
class Validator(Protocol):
    def validate_fact(
        self, f: Fact, schema: dict[str, Any]
    ) -> list[ValidationIssue]:
        """
        Validates a Fact against a schema (type checks, required fields, ranges).
        Returns list of ValidationIssue; empty = fact is valid.
        """
        ...

    def validate_rule(
        self, r: Rule, schema: dict[str, Any]
    ) -> list[ValidationIssue]:
        """
        Validates a Rule (head/body syntax, arity consistency).
        Returns list of ValidationIssue; empty = rule is valid.
        """
        ...

    def find_conflicts(
        self,
        facts: list[Fact],
        constraints: list[dict[str, Any]],
    ) -> list[ConflictSet]:
        """
        Finds contradictions among a set of facts given constraints.
        Example: two facts with same (h,r) but different t where r is functional.
        Returns list of ConflictSet grouping conflicting fact_ids.
        """
        ...
