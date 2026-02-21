from adapters.math_problem_parser.regex_parser import RegexMathParser
from contracts import ProblemType


def test_regex_math_parser_parses_reversed_equation_expression():
    parser = RegexMathParser()

    parsed = parser.parse("6 = 1 + 2 + 3")

    assert parsed.problem_type == ProblemType.EXPR
    assert parsed.operation_hint == "add"
    assert parsed.extracted_numbers == [1, 2, 3]
