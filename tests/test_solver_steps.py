from __future__ import annotations

from provenuance import _extract_timeline_plan, _first_binding


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
