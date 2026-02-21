from __future__ import annotations

from adapters.rule_bootstrap import RuleBootstrapper
from contracts import Fact, FactStatus
from ports.rule_bootstrap import BootstrapConfig


def _asserted_fact(h: str, r: str, t: str, provenance: list[str] | None = None) -> Fact:
    return Fact(
        h=h,
        r=r,
        t=t,
        status=FactStatus.ASSERTED,
        provenance=provenance or [],
        confidence=1.0,
    )


def test_rule_bootstrap_mines_promotable_path_rule():
    facts = [
        _asserted_fact("a", "sibling", "b"),
        _asserted_fact("c", "sibling", "d"),
        _asserted_fact("e", "sibling", "f"),
        _asserted_fact("b", "owns", "ball"),
        _asserted_fact("d", "owns", "book"),
        _asserted_fact("f", "owns", "car"),
        _asserted_fact("a", "owns", "ball"),
        _asserted_fact("c", "owns", "book"),
        _asserted_fact("e", "owns", "car"),
    ]

    cfg = BootstrapConfig(
        relation_whitelist={"sibling", "owns"},
        max_body_literals=2,
        min_coverage=2,
        min_support=2,
        min_confidence=0.9,
        min_lift=1.0,
        max_violations=0,
        max_corruption_hits=5,
        max_local_cwa_negatives=5,
        use_local_cwa=False,
        functional_relations=set(),
        top_k=50,
    )
    bootstrapper = RuleBootstrapper(cfg)

    candidates, asserted_count, relation_count, pattern_count = bootstrapper.mine_candidates(facts)

    assert asserted_count == len(facts)
    assert relation_count == 2
    assert pattern_count > 0
    candidate = next(
        c
        for c in candidates
        if c.head == "owns(X,Z)" and c.body_relations == ("sibling", "owns")
    )
    assert candidate.coverage == 3
    assert candidate.support == 3
    assert candidate.promotable is True


def test_rule_bootstrap_respects_relation_whitelist():
    facts = [
        _asserted_fact("a", "sibling", "b"),
        _asserted_fact("b", "owns", "ball"),
        _asserted_fact("a", "owns", "ball"),
        _asserted_fact("a", "secret", "x"),
    ]
    cfg = BootstrapConfig(
        relation_whitelist={"sibling", "owns"},
        min_coverage=1,
        min_support=1,
        min_confidence=0.0,
        min_lift=0.0,
        use_local_cwa=False,
        functional_relations=set(),
        top_k=50,
    )
    bootstrapper = RuleBootstrapper(cfg)

    candidates, _, _, _ = bootstrapper.mine_candidates(facts)

    assert candidates
    assert all(c.head_relation in {"sibling", "owns"} for c in candidates)
    assert all("secret(" not in c.head for c in candidates)


def test_rule_bootstrap_blocks_promotion_on_functional_violations():
    facts = [
        _asserted_fact("a", "p", "b1"),
        _asserted_fact("c", "p", "d1"),
        _asserted_fact("a", "p", "b2"),
        _asserted_fact("b1", "q", "1"),
        _asserted_fact("d1", "q", "2"),
        _asserted_fact("b2", "q", "2"),
        _asserted_fact("a", "eq", "1"),
        _asserted_fact("c", "eq", "2"),
    ]

    cfg = BootstrapConfig(
        relation_whitelist={"p", "q", "eq"},
        max_body_literals=2,
        min_coverage=3,
        min_support=2,
        min_confidence=0.6,
        min_lift=0.0,
        max_violations=0,
        max_corruption_hits=10,
        max_local_cwa_negatives=10,
        use_local_cwa=False,
        functional_relations={"eq"},
        top_k=50,
    )
    bootstrapper = RuleBootstrapper(cfg)

    candidates, _, _, _ = bootstrapper.mine_candidates(facts)
    candidate = next(c for c in candidates if c.head == "eq(X,Z)" and c.body_relations == ("p", "q"))

    assert candidate.coverage == 3
    assert candidate.support == 2
    assert candidate.violations > 0
    assert candidate.promotable is False
    assert "constraint_violations" in candidate.rejection_reasons


def test_rule_bootstrap_mines_eq_instance_of_bridge_without_paths():
    facts = [
        _asserted_fact("add(2,3)", "eq", "5"),
        _asserted_fact("add(2,3,5)", "instance_of", "add"),
        _asserted_fact("add(1,1)", "eq", "2"),
        _asserted_fact("add(1,1,2)", "instance_of", "add"),
    ]

    cfg = BootstrapConfig(
        relation_whitelist={"eq", "instance_of"},
        max_body_literals=2,
        min_coverage=1,
        min_support=1,
        min_confidence=0.9,
        min_lift=0.0,
        max_violations=0,
        max_corruption_hits=0,
        max_local_cwa_negatives=0,
        use_local_cwa=False,
        functional_relations={"eq"},
        top_k=20,
    )
    bootstrapper = RuleBootstrapper(cfg)

    candidates, asserted_count, relation_count, pattern_count = bootstrapper.mine_candidates(facts)

    assert asserted_count == 4
    assert relation_count == 2
    assert pattern_count > 0

    bridge = next(
        c for c in candidates
        if c.head == "eq(add(X,Y),Z)" and c.body == ("instance_of(add(X,Y,Z),add)",)
    )
    assert bridge.coverage == 2
    assert bridge.support == 2
    assert bridge.promotable is True


def test_rule_bootstrap_mines_add_commutativity_pattern():
    facts = [
        _asserted_fact("add(2,1)", "eq", "3"),
        _asserted_fact("add(1,2)", "eq", "3"),
    ]

    cfg = BootstrapConfig(
        relation_whitelist={"eq"},
        max_body_literals=2,
        min_coverage=1,
        min_support=1,
        min_confidence=0.9,
        min_lift=0.0,
        max_violations=0,
        max_corruption_hits=0,
        max_local_cwa_negatives=0,
        use_local_cwa=False,
        functional_relations={"eq"},
        top_k=20,
    )
    bootstrapper = RuleBootstrapper(cfg)

    candidates, asserted_count, relation_count, pattern_count = bootstrapper.mine_candidates(facts)

    assert asserted_count == 2
    assert relation_count == 1
    assert pattern_count > 0

    comm = next(
        c for c in candidates
        if c.head == "eq(add(X,Y),Z)" and c.body == ("eq(add(Y,X),Z)",)
    )
    assert comm.coverage == 2
    assert comm.support == 2
    assert comm.promotable is True
