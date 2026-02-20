"""
Adapter: SimpleValidator
Implementuje port Validator — strukturalna walidacja faktów i reguł
oraz wykrywanie konfliktów (sprzeczności funkcjonalnych).

Obsługiwane schematy faktów (field "relation_type"):
  "functional"  — dana para (h, r) może mieć TYLKO JEDEN t
  "positive"    — t musi być parsowalne jako liczba >= 0
  "known_op"    — r ∈ {add, sub, mul, div, eq, instance_of, …}

Obsługiwane schematy reguł:
  "require_body"  — body nie może być puste
  "check_arity"   — head i każdy atom w body muszą mieć taką samą arność
"""
from __future__ import annotations

import re
from typing import Any

from contracts import (
    ConflictSet,
    Fact,
    Rule,
    ValidationIssue,
)

_ATOM_RE = re.compile(r"^(\w+)\(([^)]*)\)$")

_KNOWN_RELATIONS = {
    "eq", "instance_of",
    "add", "sub", "mul", "div",
    "has_property", "is_a",
}


def _parse_arity(atom: str) -> int | None:
    """Zwraca arność atomu w stylu Prologu lub None jeśli nie pasuje."""
    m = _ATOM_RE.match(atom.strip())
    if not m:
        return None
    args = m.group(2)
    return len(args.split(",")) if args.strip() else 0


class SimpleValidator:
    """Prosta walidacja strukturalna + wykrywanie konfliktów funkcjonalnych."""

    # -- Validator protocol ------------------------------------

    def validate_fact(
        self, f: Fact, schema: dict[str, Any]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        # Podstawowe pola
        if not f.h.strip():
            issues.append(ValidationIssue(
                severity="error", code="EMPTY_HEAD",
                message="Pole 'h' (head) faktu nie może być puste.",
                field_path="h",
            ))
        if not f.r.strip():
            issues.append(ValidationIssue(
                severity="error", code="EMPTY_RELATION",
                message="Pole 'r' (relation) faktu nie może być puste.",
                field_path="r",
            ))
        if not f.t.strip():
            issues.append(ValidationIssue(
                severity="error", code="EMPTY_TAIL",
                message="Pole 't' (tail) faktu nie może być puste.",
                field_path="t",
            ))

        # Opcjonalne reguły schematu
        relation_type = schema.get("relation_type", "")

        if relation_type == "positive":
            try:
                val = float(f.t)
                if val < 0:
                    raise ValueError
            except (ValueError, TypeError):
                issues.append(ValidationIssue(
                    severity="warning", code="NOT_POSITIVE",
                    message=f"Wartość {f.t!r} nie jest liczbą nieujemną.",
                    field_path="t",
                ))

        if schema.get("known_relations_only", False):
            if f.r not in _KNOWN_RELATIONS:
                issues.append(ValidationIssue(
                    severity="warning", code="UNKNOWN_RELATION",
                    message=f"Relacja {f.r!r} nie należy do zbioru znanych relacji.",
                    field_path="r",
                ))

        return issues

    def validate_rule(
        self, r: Rule, schema: dict[str, Any]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if not r.head.strip():
            issues.append(ValidationIssue(
                severity="error", code="EMPTY_HEAD",
                message="head reguły nie może być pusty.",
                field_path="head",
            ))

        if schema.get("require_body", False) and not r.body:
            issues.append(ValidationIssue(
                severity="warning", code="EMPTY_BODY",
                message="Reguła ma pustą listę body (oczekiwano co najmniej jednego atomu).",
                field_path="body",
            ))

        if schema.get("check_arity", False):
            head_arity = _parse_arity(r.head)
            for atom in r.body:
                body_arity = _parse_arity(atom)
                if head_arity is not None and body_arity is not None:
                    functor_head = _ATOM_RE.match(r.head.strip())
                    functor_body = _ATOM_RE.match(atom.strip())
                    if (functor_head and functor_body and
                            functor_head.group(1) == functor_body.group(1) and
                            head_arity != body_arity):
                        issues.append(ValidationIssue(
                            severity="error", code="ARITY_MISMATCH",
                            message=(
                                f"Niezgodność arności: head ma {head_arity} arg(i), "
                                f"body atom {atom!r} ma {body_arity}."
                            ),
                            field_path="body",
                        ))

        return issues

    def find_conflicts(
        self,
        facts: list[Fact],
        constraints: list[dict[str, Any]],
    ) -> list[ConflictSet]:
        """
        Wykrywa konflikty funkcjonalne: dwa fakty z tym samym (h, r)
        ale różnym t, gdy constraint["type"] == "functional".
        """
        conflicts: list[ConflictSet] = []

        for constraint in constraints:
            if constraint.get("type") != "functional":
                continue

            # Grupuj fakty po (h, r)
            groups: dict[tuple[str, str], list[Fact]] = {}
            for f in facts:
                key = (f.h, f.r)
                groups.setdefault(key, []).append(f)

            for (h, r), group in groups.items():
                # Zbierz unikalne wartości t
                unique_tails = {f.t for f in group}
                if len(unique_tails) > 1:
                    conflicts.append(ConflictSet(
                        fact_ids=[f.fact_id for f in group],
                        conflict_type="FUNCTIONAL_VIOLATION",
                        description=(
                            f"Relacja {r!r} jest funkcjonalna, ale "
                            f"{h!r} → {sorted(unique_tails)} (sprzeczne wartości)."
                        ),
                    ))

        return conflicts
