"""
Adapter: ASTEvaluator
Implementuje port Evaluator — rekurencyjne przejście ExprAST z Fraction.

Fractions zapewniają dokładną arytmetykę dla operacji na liczbach całkowitych
i ułamkach (unikamy błędów zmiennoprzecinkowych przy dzieleniu).

eval_expr() — oblicza wartość; zwraca int jeśli wynik jest całkowity
simplify() — constant-folding: BinOp(Num, Num) → Num
"""
from __future__ import annotations

from fractions import Fraction
from typing import Union

from contracts import (
    BinOpNode,
    EvalResult,
    ExprAST,
    NumberNode,
    UnaryOpNode,
    VariableNode,
)

# Mapowanie symboli operatorów na operacje Fraction
_OP_FUNCS = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "×":  lambda a, b: a * b,
    "·":  lambda a, b: a * b,
    "/":  lambda a, b: _safe_div(a, b),
    "÷":  lambda a, b: _safe_div(a, b),
}


def _safe_div(a: Fraction, b: Fraction) -> Fraction:
    if b == 0:
        raise ZeroDivisionError("Dzielenie przez zero")
    return a / b


def _op_symbol(op: str) -> str:
    return {"×": "*", "·": "*", "÷": "/"}.get(op, op)


class ASTEvaluator:
    """Dokładny ewaluator wyrażeń arytmetycznych oparty na AST."""

    # -- Evaluator protocol ------------------------------------------------

    def eval_expr(
        self,
        ast: ExprAST,
        env: dict[str, int] | None = None,
    ) -> EvalResult:
        """
        Rekurencyjnie oblicza wartość AST.
        env: opcjonalne podstawienia zmiennych (np. {"x": 5}).
        Zwraca EvalResult z wartością i krokami proof.
        """
        frac_env = {k: Fraction(v) for k, v in (env or {}).items()}
        value, steps = self._eval(ast, frac_env)

        # Konwersja Fraction → int lub float
        if value.denominator == 1:
            return EvalResult(value=int(value), is_exact=True, steps=steps)
        else:
            return EvalResult(
                value=float(value),
                is_exact=False,
                steps=steps,
            )

    def simplify(self, ast: ExprAST) -> ExprAST:
        """
        Constant-folding: BinOpNode(NumberNode, NumberNode) → NumberNode.
        Nie oblicza węzłów zawierających zmienne.
        """
        if isinstance(ast, NumberNode):
            return ast
        if isinstance(ast, VariableNode):
            return ast
        if isinstance(ast, UnaryOpNode):
            operand = self.simplify(ast.operand)
            if isinstance(operand, NumberNode):
                return NumberNode(value=-operand.value)
            return UnaryOpNode(op=ast.op, operand=operand)  # type: ignore[arg-type]
        if isinstance(ast, BinOpNode):
            left = self.simplify(ast.left)   # type: ignore[arg-type]
            right = self.simplify(ast.right)  # type: ignore[arg-type]
            if isinstance(left, NumberNode) and isinstance(right, NumberNode):
                fn = _OP_FUNCS.get(ast.op)
                if fn:
                    try:
                        result = fn(Fraction(left.value), Fraction(right.value))
                        if result.denominator == 1:
                            return NumberNode(value=int(result))
                    except ZeroDivisionError:
                        pass
            return BinOpNode(op=ast.op, left=left, right=right)  # type: ignore[arg-type]
        return ast  # fallthrough

    # -- Prywatne ----------------------------------------------------------

    def _eval(
        self,
        node: ExprAST,
        env: dict[str, Fraction],
    ) -> tuple[Fraction, list[str]]:
        """Zwraca (wartość, lista kroków)."""

        if isinstance(node, NumberNode):
            return Fraction(node.value), []

        if isinstance(node, VariableNode):
            if node.name not in env:
                raise ValueError(f"Niezwiązana zmienna: {node.name!r}")
            val = env[node.name]
            return val, [f"{node.name} = {_fmt(val)}"]

        if isinstance(node, UnaryOpNode):
            val, steps = self._eval(node.operand, env)  # type: ignore[arg-type]
            result = -val
            steps = steps + [f"-({_fmt(val)}) = {_fmt(result)}"]
            return result, steps

        if isinstance(node, BinOpNode):
            left_val, left_steps = self._eval(node.left, env)   # type: ignore[arg-type]
            right_val, right_steps = self._eval(node.right, env)  # type: ignore[arg-type]

            fn = _OP_FUNCS.get(node.op)
            if fn is None:
                raise ValueError(f"Nieznany operator: {node.op!r}")

            result = fn(left_val, right_val)
            op_sym = _op_symbol(node.op)
            step = f"{_fmt(left_val)} {op_sym} {_fmt(right_val)} = {_fmt(result)}"
            all_steps = left_steps + right_steps + [step]
            return result, all_steps

        raise TypeError(f"Nieznany typ węzła AST: {type(node)}")


def _fmt(v: Fraction) -> str:
    """Czytelna reprezentacja Fraction."""
    if v.denominator == 1:
        return str(int(v))
    return f"{v.numerator}/{v.denominator}"
