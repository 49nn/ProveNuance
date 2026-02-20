"""
Adapter: RuleBasedExtractor
Implementuje port FrameExtractor za pomocą regexów (MVP).

Obsługiwane wzorce:
  ARITH_EXAMPLE  — "3 + 4 = 7", "12 minus 5 equals 7", "2 × 3 = 6" itd.
  PROPERTY       — "addition is commutative", "dodawanie jest przemienne" itd.
"""
from __future__ import annotations

import re
from typing import Any

from adapters.frame_extractor._printer import print_frames
from contracts import (
    ArithExampleFrame,
    DocSpan,
    Frame,
    FrameType,
    PropertyFrame,
    ValidationIssue,
)

# ──────────────────────────────────────────────────────────────────
# Pomocnicze słowniki do normalizacji
# ──────────────────────────────────────────────────────────────────

_OP_WORDS: dict[str, str] = {
    # English
    "plus": "add", "added to": "add", "and": "add", "add": "add",
    "minus": "sub", "less": "sub", "subtract": "sub", "subtracted": "sub",
    "times": "mul", "multiplied by": "mul", "multiply": "mul",
    "divided by": "div", "divide": "div",
    # Symbols
    "+": "add", "-": "sub", "×": "mul", "·": "mul", "*": "mul",
    "/": "div", "÷": "div",
    # Polish
    "dodać": "add", "plus": "add", "odjąć": "sub", "minus": "sub",
    "razy": "mul", "podzielić": "div",
}

_OP_SYMBOLS = r"(?:\+|-|×|·|\*|/|÷)"

_OP_WORDS_PATTERN = (
    r"(?:added\s+to|subtracted\s+from|multiplied\s+by|divided\s+by"
    r"|plus|minus|times|and|add|subtract|multiply|divide"
    r"|dodać|odjąć|razy|podzielić)"
)

_EQ_WORDS = r"(?:equals?|is|=|wynosi)"

_NUM = r"(?:\d+(?:[.,]\d+)?)"          # liczba całkowita lub dziesiętna

# ──────────────────────────────────────────────────────────────────
# Regex ARITH_EXAMPLE
# Formaty: "3 + 4 = 7"  /  "three plus four equals seven"
#          "2 × 3 = 6"  /  "12 minus 5 equals 7"
# ──────────────────────────────────────────────────────────────────

_ARITH_PATTERN = re.compile(
    rf"({_NUM})\s*"
    rf"({_OP_SYMBOLS}|{_OP_WORDS_PATTERN})\s*"
    rf"({_NUM})\s*"
    rf"{_EQ_WORDS}\s*"
    rf"({_NUM})",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────
# Regex PROPERTY
# Przykłady:
#   "Addition is commutative"
#   "Multiplication is associative"
#   "Adding zero leaves the number unchanged" (zero_neutral)
# ──────────────────────────────────────────────────────────────────

_SUBJECTS: dict[str, str] = {
    "addition": "addition", "adding": "addition", "dodawanie": "addition",
    "subtraction": "subtraction", "subtracting": "subtraction", "odejmowanie": "subtraction",
    "multiplication": "multiplication", "multiplying": "multiplication", "mnożenie": "multiplication",
    "division": "division", "dividing": "division", "dzielenie": "division",
}

_PROPERTIES: dict[str, str] = {
    "commutative": "commutative", "przemienne": "commutative",
    "associative": "associative", "łączne": "associative",
    "zero": "zero_neutral", "identity": "zero_neutral",
    "distributive": "distributive", "rozdzielne": "distributive",
}

_SUBJ_PATTERN = "|".join(re.escape(k) for k in _SUBJECTS)
_PROP_PATTERN = "|".join(re.escape(k) for k in _PROPERTIES)

_PROPERTY_PATTERN = re.compile(
    rf"({_SUBJ_PATTERN})\s+(?:is\s+|jest\s+)?({_PROP_PATTERN})",
    re.IGNORECASE,
)

# Alternatywny wzorzec dla "zero" — "adding 0 does not change", itp.
_ZERO_NEUTRAL_PATTERN = re.compile(
    rf"({_SUBJ_PATTERN}).*?\bzero\b",
    re.IGNORECASE,
)


def _norm_op(raw: str) -> str:
    """Normalizuje surowy operator/słowo do 'add'|'sub'|'mul'|'div'."""
    key = raw.strip().lower()
    return _OP_WORDS.get(key, "add")


def _parse_num(raw: str) -> int | str:
    """Zwraca int jeśli możliwe, w przeciwnym razie surowy string."""
    clean = raw.replace(",", ".")
    try:
        f = float(clean)
        return int(f) if f == int(f) else raw
    except ValueError:
        return raw


# ──────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────

class RuleBasedExtractor:
    """Wyciąga ArithExampleFrame i PropertyFrame z tekstu span-u."""

    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

    # -- FrameExtractor protocol --------------------------------

    def extract_frames(self, span: DocSpan) -> list[Frame]:
        text = span.surface_text
        frames: list[Frame] = []
        frames.extend(self._extract_arith(text, span.span_id))
        frames.extend(self._extract_properties(text, span.span_id))
        if self._verbose:
            print_frames(span.surface_text, frames)
        return frames

    def validate_frame(self, frame: Frame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if frame.frame_type == FrameType.ARITH_EXAMPLE:
            issues.extend(self._validate_arith(frame))  # type: ignore[arg-type]
        elif frame.frame_type == FrameType.PROPERTY:
            issues.extend(self._validate_property(frame))  # type: ignore[arg-type]
        return issues

    # -- Prywatne metody ekstrakcji ----------------------------

    def _extract_arith(self, text: str, span_id: str) -> list[ArithExampleFrame]:
        results = []
        for m in _ARITH_PATTERN.finditer(text):
            op_raw = m.group(2)
            frame = ArithExampleFrame(
                operation=_norm_op(op_raw),
                operands=[_parse_num(m.group(1)), _parse_num(m.group(3))],
                result=_parse_num(m.group(4)),
                source_span_id=span_id,
            )
            results.append(frame)
        return results

    def _extract_properties(self, text: str, span_id: str) -> list[PropertyFrame]:
        results: list[PropertyFrame] = []
        seen: set[tuple[str, str]] = set()

        for m in _PROPERTY_PATTERN.finditer(text):
            subj = _SUBJECTS.get(m.group(1).lower(), m.group(1).lower())
            prop = _PROPERTIES.get(m.group(2).lower(), m.group(2).lower())
            key = (subj, prop)
            if key in seen:
                continue
            seen.add(key)
            results.append(PropertyFrame(
                subject=subj,
                property_name=prop,
                source_span_id=span_id,
            ))

        # Fallback: "adding zero…"
        for m in _ZERO_NEUTRAL_PATTERN.finditer(text):
            subj = _SUBJECTS.get(m.group(1).lower(), m.group(1).lower())
            key = (subj, "zero_neutral")
            if key in seen:
                continue
            seen.add(key)
            results.append(PropertyFrame(
                subject=subj,
                property_name="zero_neutral",
                source_span_id=span_id,
            ))

        return results

    # -- Prywatna walidacja ------------------------------------

    def _validate_arith(self, frame: ArithExampleFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if frame.operation not in {"add", "sub", "mul", "div"}:
            issues.append(ValidationIssue(
                severity="error",
                code="UNKNOWN_OP",
                message=f"Nieznana operacja: {frame.operation!r}",
                field_path="operation",
            ))
        if len(frame.operands) != 2:
            issues.append(ValidationIssue(
                severity="error",
                code="WRONG_ARITY",
                message=f"Oczekiwano 2 operandów, got {len(frame.operands)}",
                field_path="operands",
            ))
        # Sprawdź spójność liczbową
        if all(isinstance(x, int) for x in [*frame.operands, frame.result]):
            a, b, r = frame.operands[0], frame.operands[1], frame.result
            expected: dict[str, Any] = {
                "add": a + b,  # type: ignore[operator]
                "sub": a - b,  # type: ignore[operator]
                "mul": a * b,  # type: ignore[operator]
                "div": a / b if b != 0 else None,  # type: ignore[operator]
            }
            exp = expected.get(frame.operation)
            if exp is not None and exp != r:
                issues.append(ValidationIssue(
                    severity="warning",
                    code="ARITHMETIC_MISMATCH",
                    message=f"Wynik {r} ≠ oczekiwany {exp}",
                    field_path="result",
                ))
        return issues

    def _validate_property(self, frame: PropertyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        known_properties = {
            "commutative", "associative", "zero_neutral",
            "distributive", "identity",
        }
        if frame.property_name not in known_properties:
            issues.append(ValidationIssue(
                severity="warning",
                code="UNKNOWN_PROPERTY",
                message=f"Nieznana właściwość: {frame.property_name!r}",
                field_path="property_name",
            ))
        return issues
