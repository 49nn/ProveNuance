"""
adapters/pattern_miner/spec.py — HypothesisLanguageSpec i powiązane modele.

Definiuje język hipotez dla PatternMiner: które predykaty, jakie szablony,
jakie progi jakościowe. Oddziela konfigurację domeny od logiki miningu.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ports.rule_bootstrap import BootstrapConfig

TemplateKind = Literal[
    "ARG_PERMUTATION",
    "CROSS_PREDICATE_REWRITE",
    "CONSTANT_SCHEMA",
    "FUNCTIONALITY_CONSTRAINT",
    "COMPOSITION_EQUIVALENCE",
]


class PredicateSignature(BaseModel):
    arity: int
    arg_types: List[str]


class MiningThresholds(BaseModel):
    min_support: int = 3
    min_confidence: float = 0.95
    max_counterexamples: int = 0
    min_distinct_bindings: int = 3


class ValidationSplitSpec(BaseModel):
    enabled: bool = False
    strategy: str = "holdout_by_hash"
    ratio: float = 0.2


class HypothesisLanguageSpec(BaseModel):
    version: str
    domain: str
    predicate_signatures: Dict[str, PredicateSignature]
    enabled_predicates: List[str]
    allow_negation: bool = False
    allow_function_terms: bool = False
    allow_constants_in_rules: bool = True
    allowed_constants: List[str] = Field(default_factory=list)
    max_body_atoms: int = 1
    max_head_atoms: int = 1
    max_variables: int = 4
    require_safe_rules: bool = True
    allow_recursive_rules: bool = False
    allowed_template_kinds: List[TemplateKind]
    template_limits: Dict[str, Any] = Field(default_factory=dict)
    mining_thresholds: MiningThresholds = Field(default_factory=MiningThresholds)
    validation_split: ValidationSplitSpec = Field(default_factory=ValidationSplitSpec)


def default_math_grade1to3_spec() -> HypothesisLanguageSpec:
    """Domyślna specyfikacja dla PoC matematyki kl. 1–3 (poc-math-v1)."""
    return HypothesisLanguageSpec(
        version="poc-math-v1",
        domain="math_grade_1_3",
        predicate_signatures={
            "add": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
            "sub": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
            "eq": PredicateSignature(arity=2, arg_types=["ExprOrNumber", "ExprOrNumber"]),
        },
        enabled_predicates=["add", "sub"],
        allow_negation=False,
        allow_function_terms=False,
        allow_constants_in_rules=True,
        allowed_constants=["0", "1"],
        max_body_atoms=1,
        max_head_atoms=1,
        max_variables=4,
        require_safe_rules=True,
        allow_recursive_rules=False,
        allowed_template_kinds=[
            "ARG_PERMUTATION",
            "CROSS_PREDICATE_REWRITE",
            "CONSTANT_SCHEMA",
            "FUNCTIONALITY_CONSTRAINT",
            "COMPOSITION_EQUIVALENCE",
        ],
        template_limits={
            "arg_permutation": {
                "allowed_predicates": ["add", "sub"],
                "max_permutations_per_predicate": 6,
            },
            "cross_predicate_rewrite": {
                "allowed_pairs": [["add", "sub"], ["sub", "add"]],
            },
            "constant_schema": {
                "allowed_predicates": ["add", "sub"],
                "max_constants_per_rule": 1,
            },
            "composition_equivalence": {
                # Każdy wpis definiuje dwa łańcuchy 2-hop na tym samym predykacie.
                # left:  pred(A,B,S1), pred(S1,C,R)   → liczymy R przez lewą stronę
                # right: pred(B,C,S2), pred(A,S2,R)   → liczymy R przez prawą stronę
                # Sprawdzamy czy R_left == R_right dla każdego (A,B,C).
                # Odkrywa łączność (associativity).
                "chains": [
                    {
                        "predicate": "add",
                        "left_chain":  [("A", "B", "S1"), ("S1", "C", "R")],
                        "right_chain": [("B", "C", "S2"), ("A",  "S2", "R")],
                        "label": "associativity(add)",
                    }
                ],
            },
        },
        mining_thresholds=MiningThresholds(
            min_support=3,
            min_confidence=0.95,
            max_counterexamples=0,
            min_distinct_bindings=3,
        ),
        validation_split=ValidationSplitSpec(enabled=False),
    )


def spec_from_bootstrap_config(config: BootstrapConfig) -> HypothesisLanguageSpec:
    """
    Buduje HypothesisLanguageSpec z istniejącego BootstrapConfig.
    Umożliwia reużycie CLI bez zmian w argumentach.
    Enabled predicates = relation_whitelist ∩ znane predykaty relacyjne.
    """
    known_relational = {"add", "sub", "mul", "div"}
    enabled = sorted(known_relational & config.relation_whitelist) or ["add", "sub"]

    all_signatures = {
        "add": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
        "sub": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
        "mul": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
        "div": PredicateSignature(arity=3, arg_types=["Number", "Number", "Number"]),
    }
    signatures = {p: all_signatures[p] for p in enabled if p in all_signatures}

    # Buduj allowed_pairs dla CROSS_PREDICATE_REWRITE
    cross_pairs: list[list[str]] = []
    for a in enabled:
        for b in enabled:
            if a != b:
                cross_pairs.append([a, b])

    return HypothesisLanguageSpec(
        version="bootstrap-config-derived",
        domain="math_grade_1_3",
        predicate_signatures=signatures,
        enabled_predicates=enabled,
        allow_constants_in_rules=True,
        allowed_constants=["0", "1"],
        allowed_template_kinds=[
            "ARG_PERMUTATION",
            "CROSS_PREDICATE_REWRITE",
            "CONSTANT_SCHEMA",
            "FUNCTIONALITY_CONSTRAINT",
            "COMPOSITION_EQUIVALENCE",
        ],
        template_limits={
            "cross_predicate_rewrite": {"allowed_pairs": cross_pairs},
            "composition_equivalence": {
                "chains": [
                    {
                        "predicate": p,
                        "left_chain":  [("A", "B", "S1"), ("S1", "C", "R")],
                        "right_chain": [("B", "C", "S2"), ("A",  "S2", "R")],
                        "label": f"associativity({p})",
                    }
                    for p in enabled
                ],
            },
        },
        mining_thresholds=MiningThresholds(
            min_support=config.min_support,
            min_confidence=config.min_confidence,
            max_counterexamples=config.max_violations,
            min_distinct_bindings=config.min_support,
        ),
    )
