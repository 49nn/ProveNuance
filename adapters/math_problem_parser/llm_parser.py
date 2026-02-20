"""
Adapter: LLMMathParser
Uses an external LLM backend to parse a student math task into ParsedProblem.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from contracts import LogicQuery, ParsedProblem, ProblemType
from ports.math_problem_parser import MathProblemParser

logger = logging.getLogger("prove_nuance.llm_math_parser")


class _LLMParseOutput(BaseModel):
    problem_type: Literal["expr", "word_problem", "unknown"] = "unknown"
    operation_hint: Literal["add", "sub", "mul", "div"] | None = None
    extracted_numbers: list[int] = Field(default_factory=list)
    entities_mentioned: list[str] = Field(default_factory=list)
    slots: dict[str, Any] = Field(default_factory=dict)
    logic_query_goal: str | None = None
    logic_query_variables: list[str] = Field(default_factory=list)


class LLMMathParser:
    """
    Parses text by calling an external LLM endpoint.
    Falls back to another parser if configured and backend call fails.
    """

    def __init__(
        self,
        backend_url: str,
        timeout_ms: int = 10_000,
        fallback_parser: MathProblemParser | None = None,
    ) -> None:
        self._url = backend_url.strip()
        self._timeout = timeout_ms / 1000.0
        self._fallback = fallback_parser

    def parse(self, text: str) -> ParsedProblem:
        clean_text = text.strip()
        if not clean_text:
            return ParsedProblem(
                original_text="",
                problem_type=ProblemType.UNKNOWN,
            )

        if not self._url:
            return self._fallback_or_unknown(clean_text, "Missing LLM parser backend URL.")

        try:
            llm_output = self._call_backend(clean_text)
            return self._to_parsed_problem(clean_text, llm_output)
        except Exception as exc:
            logger.warning("LLM parser failed: %s", exc)
            return self._fallback_or_unknown(clean_text, str(exc))

    def _call_backend(self, text: str) -> _LLMParseOutput:
        response = httpx.post(
            self._url,
            json={"text": text},
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload: Any = response.json()

        # Some backends wrap the payload under a "result" key.
        if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
            payload = payload["result"]
        return _LLMParseOutput.model_validate(payload)

    @staticmethod
    def _to_parsed_problem(text: str, llm_output: _LLMParseOutput) -> ParsedProblem:
        problem_type = ProblemType(llm_output.problem_type)
        logic_query = None
        if llm_output.logic_query_goal:
            logic_query = LogicQuery(
                goal=llm_output.logic_query_goal,
                variables=llm_output.logic_query_variables,
            )

        return ParsedProblem(
            original_text=text,
            problem_type=problem_type,
            logic_query=logic_query,
            slots=llm_output.slots,
            entities_mentioned=llm_output.entities_mentioned,
            extracted_numbers=llm_output.extracted_numbers,
            operation_hint=llm_output.operation_hint,
        )

    def _fallback_or_unknown(self, text: str, reason: str) -> ParsedProblem:
        if self._fallback is not None:
            logger.info("Using fallback parser (%s).", reason)
            return self._fallback.parse(text)
        return ParsedProblem(
            original_text=text,
            problem_type=ProblemType.UNKNOWN,
        )
