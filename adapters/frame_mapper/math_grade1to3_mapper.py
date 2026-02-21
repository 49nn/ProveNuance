"""
Adapter: MathGrade1to3Mapper
Implementuje port FrameMapper dla ramek matematyki klas 1–3.

Mapowania:
  ArithExampleFrame  →  Fact (atom gruntowy)
      add(3,4,7)  /  sub(10,3,7)  /  mul(2,3,6)  /  div(6,2,3)

  PropertyFrame(commutative)  →  Rule Horn
      add(X,Y,Z) :- add(Y,X,Z)

  PropertyFrame(associative)  →  Rule Horn
      add(X,Y,W) :- add(Y,Z,W), add(X,Z,_)   (uproszczony)

  PropertyFrame(zero_neutral)  →  Rule Horn
      add(X,0,X) :- true
"""
from __future__ import annotations

from typing import Any

from contracts import (
    ConditionFrame,
    DefinitionFrame,
    Fact,
    FactStatus,
    Frame,
    FrameType,
    MappingResult,
    Provenance,
    Rule,
    TaskFrame,
)

# ──────────────────────────────────────────────────────────────────
# Szablony reguł dla właściwości
# ──────────────────────────────────────────────────────────────────

_COMMUTATIVE_RULES: dict[str, tuple[str, list[str]]] = {
    "addition": (
        "add(X,Y,Z)",
        ["add(Y,X,Z)"],
    ),
    "multiplication": (
        "mul(X,Y,Z)",
        ["mul(Y,X,Z)"],
    ),
}

_ASSOCIATIVE_RULES: dict[str, tuple[str, list[str]]] = {
    "addition": (
        "add(X,Y,W)",
        ["add(X,Mid,W)", "add(Y,Z,Mid)"],  # uproszczony wariant
    ),
    "multiplication": (
        "mul(X,Y,W)",
        ["mul(X,Mid,W)", "mul(Y,Z,Mid)"],
    ),
}

_ZERO_NEUTRAL_RULES: dict[str, tuple[str, list[str]]] = {
    "addition": ("add(X,0,X)", []),
    "subtraction": ("sub(X,0,X)", []),
    "multiplication": ("mul(X,1,X)", []),   # jedynka neutralna dla mnożenia
}

_DISTRIBUTIVE_RULES: dict[str, tuple[str, list[str]]] = {
    # mul(A, add(B,C), Z) :- add(B,C,BC), mul(A,BC,Z)
    "multiplication": (
        "mul(A,BC,Z)",
        ["add(B,C,BC)", "mul(A,B,AB)", "mul(A,C,AC)", "add(AB,AC,Z)"],
    ),
}


def _op_functor(operation: str) -> str:
    """Mapuje 'add'/'sub'/'mul'/'div' na funktor Prolog-style."""
    return operation  # już jest krótką formą: add, sub, mul, div


def _compute_intermediate(functor: str, left: Any, right: Any) -> int | None:
    if not isinstance(left, int) or not isinstance(right, int):
        return None
    if functor == "add":
        return left + right
    if functor == "sub":
        return left - right
    if functor == "mul":
        return left * right
    if functor == "div":
        if right == 0:
            return None
        if left % right != 0:
            return None
        return left // right
    return None


class MathGrade1to3Mapper:
    """Deterministyczny mapper ramek na fakty i reguły KnowledgeStore."""

    # -- FrameMapper protocol ----------------------------------

    def map(self, frame: Frame, provenance: Provenance) -> MappingResult:
        span_ids = provenance.span_ids

        if frame.frame_type == FrameType.ARITH_EXAMPLE:
            return self._map_arith_example(frame, span_ids)  # type: ignore[arg-type]

        if frame.frame_type == FrameType.PROPERTY:
            return self._map_property(frame, span_ids)  # type: ignore[arg-type]

        if frame.frame_type == FrameType.DEFINITION:
            return self._map_definition(frame, span_ids)  # type: ignore[arg-type]

        if frame.frame_type == FrameType.CONDITION:
            return self._map_condition(frame, span_ids)  # type: ignore[arg-type]

        if frame.frame_type == FrameType.TASK:
            return self._map_task(frame, span_ids)  # type: ignore[arg-type]

        # PROCEDURE i inne — nie mapowane w tej wersji
        return MappingResult(
            facts=[],
            rules=[],
            warnings=[f"Nieobsługiwany typ ramki: {frame.frame_type}"],
        )

    # -- Prywatne metody ---------------------------------------

    def _map_arith_example(self, frame, span_ids: list[str]) -> MappingResult:
        functor = _op_functor(frame.operation)
        operands = list(frame.operands)
        if len(operands) < 2:
            return MappingResult(
                facts=[],
                rules=[],
                warnings=["Brak wystarczajacej liczby operandow w ARITH_EXAMPLE."],
            )

        facts: list[Fact] = []
        current: Any = operands[0]
        final_result: Any = frame.result

        for idx, right in enumerate(operands[1:], start=1):
            is_last = idx == len(operands) - 1
            if is_last:
                step_result: Any = final_result
            else:
                step_result = _compute_intermediate(functor, current, right)
                if step_result is None:
                    return MappingResult(
                        facts=[],
                        rules=[],
                        warnings=[
                            "Nie mozna wyznaczyc wartosci posredniej dla lancucha arytmetycznego."
                        ],
                    )

            atom = f"{functor}({current},{right},{step_result})"
            facts.append(
                Fact(
                    h=f"{functor}({current},{right})",
                    r="eq",
                    t=str(step_result),
                    status=FactStatus.HYPOTHESIS,
                    provenance=span_ids,
                    confidence=1.0,
                )
            )
            facts.append(
                Fact(
                    h=atom,
                    r="instance_of",
                    t=functor,
                    status=FactStatus.HYPOTHESIS,
                    provenance=span_ids,
                    confidence=1.0,
                )
            )
            current = step_result

        return MappingResult(facts=facts, rules=[])

    def _map_property(self, frame, span_ids: list[str]) -> MappingResult:
        subject = frame.subject        # "addition", "multiplication", …
        prop = frame.property_name     # "commutative", "associative", …

        template_map: dict[str, dict[str, tuple[str, list[str]]]] = {
            "commutative": _COMMUTATIVE_RULES,
            "associative": _ASSOCIATIVE_RULES,
            "zero_neutral": _ZERO_NEUTRAL_RULES,
            "distributive": _DISTRIBUTIVE_RULES,
        }

        templates = template_map.get(prop, {})
        entry = templates.get(subject)

        if entry is None:
            return MappingResult(
                facts=[],
                rules=[],
                warnings=[
                    f"Brak szablonu reguły dla {prop!r} × {subject!r}"
                ],
            )

        head, body = entry
        rule = Rule(
            head=head,
            body=body,
            status=FactStatus.HYPOTHESIS,
            provenance=span_ids,
            priority=10,
        )
        return MappingResult(facts=[], rules=[rule])

    def _map_definition(self, frame: DefinitionFrame, span_ids: list[str]) -> MappingResult:
        fact = Fact(
            h=frame.term,
            r="defined_as",
            t=frame.definition,
            status=FactStatus.HYPOTHESIS,
            provenance=span_ids,
            confidence=1.0,
        )
        return MappingResult(facts=[fact], rules=[])

    def _map_condition(self, frame: ConditionFrame, span_ids: list[str]) -> MappingResult:
        """
        ConditionFrame → reguła Horn.
        "Jeśli <warunek> to <wniosek>"  ≡  wniosek :- warunek.

        Słoty condition/conclusion są tekstem naturalnym — przechowywane
        jako atom quoted, żeby reasoner mógł je ewaluować lub pokazać użytkownikowi.
        """
        rule = Rule(
            head=frame.conclusion,
            body=[frame.condition],
            status=FactStatus.HYPOTHESIS,
            provenance=span_ids,
            priority=5,
        )
        return MappingResult(facts=[], rules=[rule])

    def _map_task(self, frame: TaskFrame, span_ids: list[str]) -> MappingResult:
        """
        TaskFrame → fakt pytający (do rozwiązania przez Reasoner).
        "Oblicz 2 + 3"  →  h="compute(2 + 3)", r="requires_answer", t="?"
        """
        fact = Fact(
            h=f"{frame.verb}({frame.target})",
            r="requires_answer",
            t="?",
            status=FactStatus.HYPOTHESIS,
            provenance=span_ids,
            confidence=0.9,
        )
        return MappingResult(facts=[fact], rules=[])
