"""
Port: MathProblemParser
Odpowiedzialność: parsowanie pytań ucznia do AST lub LogicQuery.
"""
from typing import Protocol, runtime_checkable

from contracts import ParsedProblem


@runtime_checkable
class MathProblemParser(Protocol):
    def parse(self, text: str) -> ParsedProblem:
        """
        Parses a student's math question into a structured ParsedProblem.

        For arithmetic expressions (EXPR type):
          - Builds expr_ast (BinOpNode tree)
          - Sets operation_hint and extracted_numbers

        For word problems (WORD_PROBLEM type):
          - Extracts slots (numbers, entities, operation hint)
          - Constructs logic_query (e.g., LogicQuery(goal="add(5,3,?Z)"))

        Returns ParsedProblem(problem_type=UNKNOWN) if text cannot be parsed.
        Never raises; errors are encoded in the returned object.
        """
        ...
