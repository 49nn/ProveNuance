"""
Adapter: RegexMathParser
Implementuje port MathProblemParser.

Klasyfikacja tekstu:
  EXPR         — wyrażenie arytmetyczne (tylko cyfry, operatory, nawiasy)
  WORD_PROBLEM — zadanie tekstowe z liczbami i słowami operacji
  UNKNOWN      — brak pasującego wzorca

Parsowanie EXPR — precedence climbing parser:
  expr   = term (('+'|'-') term)*
  term   = unary (('*'|'/') unary)*
  unary  = '-'? atom
  atom   = NUMBER | '(' expr ')'

Parsowanie WORD_PROBLEM:
  - Wyciąga liczby (cyfry) w kolejności wystąpienia
  - Wykrywa wskazówkę operacji (add/sub/mul/div) ze słów kluczowych
  - Buduje LogicQuery: add(5,3,?Z)
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from contracts import (
    BinOpNode,
    LogicQuery,
    NumberNode,
    ParsedProblem,
    ProblemType,
    UnaryOpNode,
    VariableNode,
)

# ──────────────────────────────────────────────────────────────────────────────
# Klasyfikacja tekstu
# ──────────────────────────────────────────────────────────────────────────────

# Wyrażenie arytmetyczne: tylko cyfry, operatory, nawiasy, białe znaki, "= ?"
_EXPR_CHARS = re.compile(r'^[\d\s\+\-\*\/\(\)×÷=\?\.,]+$')

# Słowa kluczowe operacji (operacja → lista wzorców regex, case-insensitive)
_OP_KEYWORDS: dict[str, list[str]] = {
    "add": [
        r"\bmore\b", r"\bin all\b", r"\btotal\b", r"\bsum\b", r"\baltogether\b",
        r"\badded\b", r"\bplus\b", r"\bgive[ns]?\b", r"\breceive[sd]?\b",
        r"\bgets?\b", r"\bears?\b", r"\bbought\b", r"\bbuys?\b",
        r"\bdodać\b", r"\brazem\b", r"\bsuma\b", r"\bnabył\b",
    ],
    "sub": [
        r"\bleft\b", r"\bremain\b", r"\bfewer\b", r"\bless\b",
        r"\bdifference\b", r"\blost\b", r"\bspent\b", r"\bgave away\b",
        r"\bused\b", r"\bate\b", r"\beat\b", r"\bodjąć\b", r"\bpozostało\b",
        r"\bwydał\b", r"\bzjadł\b",
    ],
    "mul": [
        r"\btimes\b", r"\beach\b", r"\bevery\b", r"\bgroups? of\b",
        r"\brows? of\b", r"\bper\b", r"\brazy\b", r"\bkażdy\b",
    ],
    "div": [
        r"\bshare[sd]?\b", r"\bdivide[sd]?\b", r"\bsplit\b", r"\bequally\b",
        r"\bpodzielić\b", r"\bpodzielono\b",
    ],
}

_NUMBER_RE = re.compile(r'\b(\d+)\b')


def _detect_operation(text: str) -> str | None:
    """Zwraca operację ('add'/'sub'/'mul'/'div') lub None."""
    text_lower = text.lower()
    for op, patterns in _OP_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return op
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r'(\d+(?:\.\d+)?)'      # liczba (całkowita lub dziesiętna)
    r'|([+\-*/×÷()])'       # operator lub nawias
    r'|\s+'                  # białe znaki (pominięte)
)

_OP_MAP = {"×": "*", "÷": "/"}


def _tokenize(text: str) -> list[str]:
    """Tokenizuje wyrażenie arytmetyczne. Whitespace ignorowany."""
    tokens = []
    # Wyczyść "= ?" na końcu (pytanie o wynik)
    text = re.sub(r'\s*[=?]+\s*$', '', text).strip()
    for m in _TOKEN_RE.finditer(text):
        num, op = m.group(1), m.group(2)
        if num:
            tokens.append(num)
        elif op:
            tokens.append(_OP_MAP.get(op, op))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# Precedence climbing parser
# ──────────────────────────────────────────────────────────────────────────────

# Lewy binding power operatorów binarnych
_LEFT_BP: dict[str, int] = {"+": 10, "-": 10, "*": 20, "/": 20}


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> str | None:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _consume(self) -> str:
        t = self._tokens[self._pos]
        self._pos += 1
        return t

    def _expect(self, tok: str) -> None:
        got = self._consume()
        if got != tok:
            raise SyntaxError(f"Oczekiwano {tok!r}, got {got!r}")

    def parse(self):
        node = self._expr(0)
        if self._pos < len(self._tokens):
            raise SyntaxError(f"Nieoczekiwany token: {self._tokens[self._pos]!r}")
        return node

    def _expr(self, min_bp: int):
        left = self._unary()
        while True:
            op = self._peek()
            if op is None or op not in _LEFT_BP:
                break
            bp = _LEFT_BP[op]
            if bp <= min_bp:
                break
            self._consume()
            # Lewostronne wiązanie: right_bp = bp (nie bp+1) dla left-assoc
            right = self._expr(bp)
            left = BinOpNode(op=op, left=left, right=right)  # type: ignore[arg-type]
        return left

    def _unary(self):
        if self._peek() == "-":
            self._consume()
            operand = self._unary()
            return UnaryOpNode(op="-", operand=operand)  # type: ignore[arg-type]
        return self._primary()

    def _primary(self):
        tok = self._peek()
        if tok is None:
            raise SyntaxError("Nieoczekiwany koniec wyrażenia")
        if tok == "(":
            self._consume()
            node = self._expr(0)
            self._expect(")")
            return node
        # Liczba
        tok = self._consume()
        try:
            val = float(tok)
            return NumberNode(value=int(val) if val == int(val) else int(val))
        except ValueError:
            # Zmienna (litera)
            return VariableNode(name=tok)


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class RegexMathParser:
    """
    Parsuje tekst ucznia do ParsedProblem.
    Nigdy nie rzuca wyjątku — błędy enkodowane są w problem_type=UNKNOWN.
    """

    # -- MathProblemParser protocol -----------------------------------------

    def parse(self, text: str) -> ParsedProblem:
        text_stripped = text.strip()

        # 1. Spróbuj EXPR
        if self._looks_like_expr(text_stripped):
            return self._parse_expr_problem(text_stripped)

        # 2. Spróbuj WORD_PROBLEM
        wp = self._parse_word_problem(text_stripped)
        if wp is not None:
            return wp

        return ParsedProblem(
            original_text=text_stripped,
            problem_type=ProblemType.UNKNOWN,
        )

    # -- Prywatne -----------------------------------------------------------

    def _looks_like_expr(self, text: str) -> bool:
        """True jeśli tekst wygląda jak wyrażenie arytmetyczne."""
        # Musi zawierać cyfry + przynajmniej jeden operator
        if not _NUMBER_RE.search(text):
            return False
        clean = _EXPR_CHARS.match(text)
        return bool(clean)

    def _parse_expr_problem(self, text: str) -> ParsedProblem:
        try:
            tokens = _tokenize(text)
            if not tokens:
                raise ValueError("Puste wyrażenie")
            ast = _Parser(tokens).parse()
            nums = [int(n) for n in _NUMBER_RE.findall(text)]
            op_hint = self._infer_op_from_ast(ast)
            return ParsedProblem(
                original_text=text,
                problem_type=ProblemType.EXPR,
                expr_ast=ast,
                extracted_numbers=nums,
                operation_hint=op_hint,
            )
        except Exception:
            return ParsedProblem(
                original_text=text,
                problem_type=ProblemType.UNKNOWN,
            )

    def _infer_op_from_ast(self, ast) -> str | None:
        """Wyciąga wskazówkę operacji z korzenia AST."""
        if isinstance(ast, BinOpNode):
            return {"+": "add", "-": "sub", "*": "mul", "/": "div"}.get(ast.op)
        return None

    def _parse_word_problem(self, text: str) -> ParsedProblem | None:
        """Parsuje zadanie tekstowe. Zwraca None jeśli nie można."""
        nums = [int(n) for n in _NUMBER_RE.findall(text)]
        if len(nums) < 2:
            return None

        op = _detect_operation(text)
        if op is None:
            return None

        # Buduj LogicQuery z dwóch pierwszych liczb
        a, b = nums[0], nums[1]
        goal = f"{op}({a},{b},?Z)"
        logic_query = LogicQuery(goal=goal, variables=["Z"])

        # Encje (słowa nie będące liczbami, długości 3+)
        entities = list({
            w for w in re.findall(r'\b([A-ZÁĘÓŚŻŹĆŃ][a-záęóśżźćń]+)\b', text)
        })

        return ParsedProblem(
            original_text=text,
            problem_type=ProblemType.WORD_PROBLEM,
            logic_query=logic_query,
            slots={"a": a, "b": b, "op": op},
            entities_mentioned=entities,
            extracted_numbers=nums,
            operation_hint=op,  # type: ignore[arg-type]
        )
