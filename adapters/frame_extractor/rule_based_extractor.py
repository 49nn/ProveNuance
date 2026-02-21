"""
Adapter: RuleBasedExtractor
Simple regex-based frame extractor.

Supported patterns:
- ARITH_EXAMPLE: "3 + 4 = 7", "12 minus 5 equals 7", "2 * 3 = 6"
- PROPERTY: "addition is commutative", "dodawanie jest przemienne"
"""
from __future__ import annotations

import re

from adapters.frame_extractor._printer import print_frames
from contracts import (
    ArithExampleFrame,
    DocSpan,
    Frame,
    FrameType,
    PropertyFrame,
    ValidationIssue,
)

_OP_WORDS: dict[str, str] = {
    # English
    "plus": "add",
    "added to": "add",
    "and": "add",
    "add": "add",
    "minus": "sub",
    "less": "sub",
    "subtract": "sub",
    "subtracted": "sub",
    "times": "mul",
    "multiplied by": "mul",
    "multiply": "mul",
    "divided by": "div",
    "divide": "div",
    # Symbols
    "+": "add",
    "-": "sub",
    "×": "mul",
    "·": "mul",
    "*": "mul",
    "/": "div",
    "÷": "div",
    # Polish
    "dodać": "add",
    "odjąć": "sub",
    "razy": "mul",
    "podzielić": "div",
}

_OP_SYMBOLS = r"(?:\+|-|×|·|\*|/|÷)"
_OP_WORDS_PATTERN = (
    r"(?:added\s+to|subtracted\s+from|multiplied\s+by|divided\s+by"
    r"|plus|minus|times|and|add|subtract|multiply|divide"
    r"|dodać|odjąć|razy|podzielić)"
)
_EQ_WORDS = r"(?:equals?|is|=|wynosi)"
_NUM = r"(?:\d+(?:[.,]\d+)?)"

_ARITH_PATTERN = re.compile(
    rf"({_NUM})\s*"
    rf"({_OP_SYMBOLS}|{_OP_WORDS_PATTERN})\s*"
    rf"({_NUM})\s*"
    rf"{_EQ_WORDS}\s*"
    rf"({_NUM})",
    re.IGNORECASE,
)
_CHAIN_EXPR_PATTERN = re.compile(
    rf"^\s*({_NUM}(?:\s*{_OP_SYMBOLS}\s*{_NUM})+)\s*{_EQ_WORDS}\s*({_NUM})\s*$",
    re.IGNORECASE,
)
_CHAIN_EXPR_REVERSED_PATTERN = re.compile(
    rf"^\s*({_NUM})\s*{_EQ_WORDS}\s*({_NUM}(?:\s*{_OP_SYMBOLS}\s*{_NUM})+)\s*$",
    re.IGNORECASE,
)

_SUBJECTS: dict[str, str] = {
    "addition": "addition",
    "adding": "addition",
    "dodawanie": "addition",
    "subtraction": "subtraction",
    "subtracting": "subtraction",
    "odejmowanie": "subtraction",
    "multiplication": "multiplication",
    "multiplying": "multiplication",
    "mnożenie": "multiplication",
    "division": "division",
    "dividing": "division",
    "dzielenie": "division",
}
_PROPERTIES: dict[str, str] = {
    "commutative": "commutative",
    "przemienne": "commutative",
    "associative": "associative",
    "łączne": "associative",
    "zero": "zero_neutral",
    "identity": "zero_neutral",
    "distributive": "distributive",
    "rozdzielne": "distributive",
}
_SUBJ_PATTERN = "|".join(re.escape(k) for k in _SUBJECTS)
_PROP_PATTERN = "|".join(re.escape(k) for k in _PROPERTIES)
_PROPERTY_PATTERN = re.compile(
    rf"({_SUBJ_PATTERN})\s+(?:is\s+|jest\s+)?({_PROP_PATTERN})",
    re.IGNORECASE,
)
_ZERO_NEUTRAL_PATTERN = re.compile(
    rf"({_SUBJ_PATTERN}).*?\bzero\b",
    re.IGNORECASE,
)


def _norm_op(raw: str) -> str:
    return _OP_WORDS.get(raw.strip().lower(), "add")


def _parse_num(raw: str) -> int | str:
    clean = raw.replace(",", ".")
    try:
        f = float(clean)
        return int(f) if f == int(f) else raw
    except ValueError:
        return raw


def _eval_chain(operation: str, operands: list[int]) -> int | float | None:
    if len(operands) < 2:
        return None
    value: int | float = operands[0]
    for rhs in operands[1:]:
        if operation == "add":
            value = value + rhs
        elif operation == "sub":
            value = value - rhs
        elif operation == "mul":
            value = value * rhs
        elif operation == "div":
            if rhs == 0:
                return None
            value = value / rhs
        else:
            return None
    return value


def _parse_chain_equation(line: str) -> tuple[str, list[int | str], int | str] | None:
    m = _CHAIN_EXPR_PATTERN.match(line)
    if m:
        lhs = m.group(1)
        rhs = m.group(2)
    else:
        m_rev = _CHAIN_EXPR_REVERSED_PATTERN.match(line)
        if not m_rev:
            return None
        lhs = m_rev.group(2)
        rhs = m_rev.group(1)

    nums = re.findall(_NUM, lhs)
    ops = re.findall(_OP_SYMBOLS, lhs)
    if len(nums) < 2 or len(ops) != len(nums) - 1:
        return None
    normalized_ops = [_norm_op(op) for op in ops]
    if any(op != normalized_ops[0] for op in normalized_ops[1:]):
        return None
    return normalized_ops[0], [_parse_num(n) for n in nums], _parse_num(rhs)


class RuleBasedExtractor:
    def __init__(self, verbose: bool = False) -> None:
        self._verbose = verbose

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

    def _extract_arith(self, text: str, span_id: str) -> list[ArithExampleFrame]:
        results: list[ArithExampleFrame] = []
        seen: set[tuple[str, tuple[int | str, ...], int | str]] = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed = _parse_chain_equation(line)
            if parsed is None:
                continue
            operation, operands, result = parsed
            sig = (operation, tuple(operands), result)
            if sig in seen:
                continue
            seen.add(sig)
            results.append(
                ArithExampleFrame(
                    operation=operation,
                    operands=operands,
                    result=result,
                    source_span_id=span_id,
                )
            )

        for m in _ARITH_PATTERN.finditer(text):
            prefix = text[: m.start()].rstrip()
            if prefix and prefix[-1] in "+-*/×÷·":
                continue
            suffix = text[m.end() :].lstrip()
            if suffix and suffix[0] in "+-*/×÷·":
                continue

            operation = _norm_op(m.group(2))
            operands = [_parse_num(m.group(1)), _parse_num(m.group(3))]
            result = _parse_num(m.group(4))
            sig = (operation, tuple(operands), result)
            if sig in seen:
                continue
            seen.add(sig)
            results.append(
                ArithExampleFrame(
                    operation=operation,
                    operands=operands,
                    result=result,
                    source_span_id=span_id,
                )
            )
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
            results.append(
                PropertyFrame(
                    subject=subj,
                    property_name=prop,
                    source_span_id=span_id,
                )
            )

        for m in _ZERO_NEUTRAL_PATTERN.finditer(text):
            subj = _SUBJECTS.get(m.group(1).lower(), m.group(1).lower())
            key = (subj, "zero_neutral")
            if key in seen:
                continue
            seen.add(key)
            results.append(
                PropertyFrame(
                    subject=subj,
                    property_name="zero_neutral",
                    source_span_id=span_id,
                )
            )

        return results

    def _validate_arith(self, frame: ArithExampleFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if frame.operation not in {"add", "sub", "mul", "div"}:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="UNKNOWN_OP",
                    message=f"Nieznana operacja: {frame.operation!r}",
                    field_path="operation",
                )
            )
        if len(frame.operands) < 2:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="WRONG_ARITY",
                    message=f"Oczekiwano >=2 operandow, got {len(frame.operands)}",
                    field_path="operands",
                )
            )
        if len(frame.operands) >= 2 and all(
            isinstance(x, int) for x in [*frame.operands, frame.result]
        ):
            operands = [int(x) for x in frame.operands]
            result = int(frame.result)
            expected = _eval_chain(frame.operation, operands)
            if expected is not None and expected != result:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="ARITHMETIC_MISMATCH",
                        message=f"Wynik {result} != oczekiwany {expected}",
                        field_path="result",
                    )
                )
        return issues

    def _validate_property(self, frame: PropertyFrame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        known_properties = {
            "commutative",
            "associative",
            "zero_neutral",
            "distributive",
            "identity",
        }
        if frame.property_name not in known_properties:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="UNKNOWN_PROPERTY",
                    message=f"Nieznana wlasciwosc: {frame.property_name!r}",
                    field_path="property_name",
                )
            )
        return issues
