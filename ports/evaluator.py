"""
Port: Evaluator
Odpowiedzialność: deterministyczne liczenie wyrażeń AST bez LLM.
"""
from typing import Protocol, runtime_checkable

from contracts import EvalResult, ExprAST


@runtime_checkable
class Evaluator(Protocol):
    def eval_expr(
        self,
        ast: ExprAST,
        env: dict[str, int] | None = None,
    ) -> EvalResult:
        """
        Evaluates an arithmetic AST to a numeric result.
        env: optional variable bindings for VariableNode resolution.
        Returns EvalResult with:
          - value: int (or Fraction for non-integer division)
          - steps: list of human-readable computation steps for proof
        Raises ZeroDivisionError on division by zero.
        Raises ValueError for unbound variables.
        """
        ...

    def simplify(self, ast: ExprAST) -> ExprAST:
        """
        Constant-folds an AST: BinOpNode(NumberNode, NumberNode) → NumberNode.
        Does not evaluate variables. Returns a potentially simplified AST.
        """
        ...
