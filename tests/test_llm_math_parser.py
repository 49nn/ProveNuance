from __future__ import annotations

import httpx

from adapters.math_problem_parser.llm_parser import LLMMathParser
from adapters.math_problem_parser.regex_parser import RegexMathParser
from contracts import ProblemType


class _StubResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def test_llm_math_parser_parses_word_problem(monkeypatch):
    payload = {
        "problem_type": "word_problem",
        "operation_hint": "add",
        "extracted_numbers": [5, 3],
        "entities_mentioned": ["Mary"],
        "slots": {"a": 5, "b": 3, "op": "add"},
        "logic_query_goal": "add(5,3,?Z)",
        "logic_query_variables": ["Z"],
    }

    def _fake_post(url, json, timeout):
        assert url == "https://example.test/parse"
        assert json == {"text": "Mary has 5 apples and gets 3 more."}
        assert timeout == 10.0
        return _StubResponse(payload)

    monkeypatch.setattr("adapters.math_problem_parser.llm_parser.httpx.post", _fake_post)

    parser = LLMMathParser(backend_url="https://example.test/parse")
    parsed = parser.parse("Mary has 5 apples and gets 3 more.")

    assert parsed.problem_type == ProblemType.WORD_PROBLEM
    assert parsed.operation_hint == "add"
    assert parsed.extracted_numbers == [5, 3]
    assert parsed.logic_query is not None
    assert parsed.logic_query.goal == "add(5,3,?Z)"
    assert parsed.logic_query.variables == ["Z"]


def test_llm_math_parser_uses_regex_fallback_on_backend_error(monkeypatch):
    def _fake_post(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr("adapters.math_problem_parser.llm_parser.httpx.post", _fake_post)

    parser = LLMMathParser(
        backend_url="https://example.test/parse",
        fallback_parser=RegexMathParser(),
    )
    parsed = parser.parse("2 + 3")

    assert parsed.problem_type == ProblemType.EXPR
    assert parsed.operation_hint == "add"
    assert parsed.extracted_numbers == [2, 3]
