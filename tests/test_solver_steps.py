from __future__ import annotations

from contracts import BinOpNode, NumberNode
from provenuance import (
    _extract_expr_chain_plan,
    _extract_flat_expr_chain_plan,
    _extract_timeline_plan,
    _first_binding,
)


def test_extract_timeline_plan_reads_initial_and_events():
    slots = {
        "initial": 2,
        "events": [
            {"op": "add", "value": 1},
            {"operation": "+", "value": 3},
            {"op": "sub", "value": 0},
        ],
    }

    plan = _extract_timeline_plan(slots, default_op=None)

    assert plan == (2, [("add", 1), ("add", 3), ("sub", 0)])


def test_extract_timeline_plan_returns_none_for_invalid_payload():
    slots = {
        "initial": 2,
        "events": [{"op": "add"}],
    }

    assert _extract_timeline_plan(slots, default_op=None) is None


def test_first_binding_prefers_named_variable():
    answer = {"X": "10", "Z": "7"}

    assert _first_binding(answer, preferred_var="Z") == "7"


def test_extract_expr_chain_plan_for_left_associative_sum():
    ast = BinOpNode(
        op="+",
        left=BinOpNode(op="+", left=NumberNode(value=1), right=NumberNode(value=2)),
        right=NumberNode(value=3),
    )

    plan = _extract_expr_chain_plan(ast)

    assert plan == (1, [("add", 2), ("add", 3)])


def test_extract_expr_chain_plan_returns_none_for_non_chain_ast():
    ast = BinOpNode(
        op="+",
        left=NumberNode(value=3),
        right=BinOpNode(op="*", left=NumberNode(value=4), right=NumberNode(value=2)),
    )

    assert _extract_expr_chain_plan(ast) is None


def test_extract_flat_expr_chain_plan_for_three_operands():
    plan = _extract_flat_expr_chain_plan("add", [2, 1, 3])

    assert plan == (2, [("add", 1), ("add", 3)])


def test_extract_flat_expr_chain_plan_returns_none_for_short_sequence():
    assert _extract_flat_expr_chain_plan("add", [2, 1]) is None
