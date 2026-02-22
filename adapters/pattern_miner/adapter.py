"""
Adapter: BruteForcePatternMiner
Implementuje port RuleBootstrap via mining szablonów hipotez.

Strategie (template kinds):
  ARG_PERMUTATION        — przemienność i inne permutacje argumentów pred/n
  CROSS_PREDICATE_REWRITE — reguły między różnymi predykatami (np. add↔sub)
  CONSTANT_SCHEMA        — schematy z neutralną stałą (np. add(X,0,X))
  FUNCTIONALITY_CONSTRAINT — meta-wiedza o determinizmie (nie Horn rule)

Wejście faktów relacyjnych:
  Fact(h="add(2,5,7)", r="instance_of", t="add") → ("add", ("2","5","7"))
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations
from typing import Optional

from adapters.rule_bootstrap.base import RuleBootstrapBase, _parse_atom
from adapters.pattern_miner.spec import (
    HypothesisLanguageSpec,
    default_math_grade1to3_spec,
    spec_from_bootstrap_config,
)
from contracts import Fact, FactStatus
from ports.rule_bootstrap import BootstrapConfig, CandidateRule

# Nazwy zmiennych wg pozycji argumentu (arność ≤ 11)
_VARS = list("ABCDEFGHIJK")


# ---------------------------------------------------------------------------
# Ekstrakcja faktów relacyjnych
# ---------------------------------------------------------------------------

def _extract_relational_facts(
    facts: list[Fact],
    enabled_predicates: set[str],
) -> dict[str, set[tuple[str, ...]]]:
    """
    Filtruje fakty ASSERTED z r="instance_of" i parsuje h jako atom relacyjny.

    Fact(h="add(2,5,7)", r="instance_of", t="add") → {"add": {("2","5","7"), ...}}
    Ignoruje fakty, których funktor nie należy do enabled_predicates.
    """
    result: dict[str, set[tuple[str, ...]]] = {}
    for f in facts:
        if f.status != FactStatus.ASSERTED or f.r != "instance_of":
            continue
        parsed = _parse_atom(f.h)
        if parsed is None:
            continue
        functor, args = parsed
        if functor not in enabled_predicates:
            continue
        if f.t.strip() != functor:
            continue
        result.setdefault(functor, set()).add(tuple(args))
    return result


def _collect_provenance(
    facts: list[Fact],
    pred: str,
    supporting_tuples: set[tuple[str, ...]],
) -> list[str]:
    """Zbiera span_ids ze wspierających faktów (max 40)."""
    sids: set[str] = set()
    for f in facts:
        if f.r != "instance_of" or f.status != FactStatus.ASSERTED:
            continue
        parsed = _parse_atom(f.h)
        if parsed is None or parsed[0] != pred:
            continue
        if tuple(parsed[1]) in supporting_tuples:
            sids.update(s for s in f.provenance if s)
        if len(sids) >= 40:
            break
    return sorted(sids)[:40]


def _base_rate(pred: str, facts_by_pred: dict[str, set[tuple[str, ...]]]) -> float:
    """P(pred krotek zachodzi) wg heurystyki: n_facts / n_values^arity."""
    tuples = facts_by_pred.get(pred, set())
    if not tuples:
        return 0.0
    arity = len(next(iter(tuples)))
    all_vals: set[str] = set()
    for t in tuples:
        all_vals.update(t)
    possible = len(all_vals) ** arity
    return len(tuples) / possible if possible > 0 else 0.0


def _lift(confidence: float, br: float) -> float:
    if br == 0.0:
        return math.inf if confidence > 0.0 else 0.0
    return confidence / br


# ---------------------------------------------------------------------------
# Budowanie kandydatów
# ---------------------------------------------------------------------------

def _build_candidate(
    *,
    head_relation: str,
    body_relations: tuple[str, ...],
    head: str,
    body: tuple[str, ...],
    coverage: int,
    support: int,
    violations: int,
    distinct_bindings: int,
    sample_grounding: Optional[str],
    provenance: list[str],
    br: float,
    thresholds,
    min_coverage: int,
) -> CandidateRule:
    confidence = support / coverage if coverage else 0.0
    rejection: list[str] = []
    if coverage < min_coverage:
        rejection.append("low_coverage")
    if support < thresholds.min_support:
        rejection.append("low_support")
    if confidence < thresholds.min_confidence:
        rejection.append("low_confidence")
    if distinct_bindings < thresholds.min_distinct_bindings:
        rejection.append("low_distinct_bindings")
    if violations > thresholds.max_counterexamples:
        rejection.append("constraint_violations")

    return CandidateRule(
        head_relation=head_relation,
        body_relations=body_relations,
        head=head,
        body=body,
        coverage=coverage,
        support=support,
        confidence=confidence,
        lift=_lift(confidence, br),
        base_rate=br,
        corruption_hits=0,
        local_cwa_negatives=0,
        violations=violations,
        sample_grounding=sample_grounding,
        promotable=len(rejection) == 0,
        rejection_reasons=rejection,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# ARG_PERMUTATION
# ---------------------------------------------------------------------------

def _mine_arg_permutations(
    facts_by_pred: dict[str, set[tuple[str, ...]]],
    asserted_facts: list[Fact],
    spec: HypothesisLanguageSpec,
    min_coverage: int,
) -> list[CandidateRule]:
    """
    Dla każdego enabled predicate testuje wszystkie permutacje argumentów.
    np. add(A,B,C) :- add(B,A,C)  →  test przemienności
    """
    candidates: list[CandidateRule] = []
    limits = spec.template_limits.get("arg_permutation", {})
    allowed_preds = set(limits.get("allowed_predicates", spec.enabled_predicates))
    max_perms = limits.get("max_permutations_per_predicate", 24)

    for pred in spec.enabled_predicates:
        if pred not in allowed_preds:
            continue
        sig = spec.predicate_signatures.get(pred)
        if sig is None:
            continue
        n = sig.arity
        if n < 2:
            continue
        vars_ = _VARS[:n]
        body_atom = f"{pred}({','.join(vars_)})"
        tuples_set = facts_by_pred.get(pred, set())
        if not tuples_set:
            continue
        br = _base_rate(pred, facts_by_pred)

        identity = tuple(range(n))
        perms_tried = 0
        for perm in permutations(range(n)):
            if perm == identity:
                continue
            if perms_tried >= max_perms:
                break
            perms_tried += 1

            permuted_vars = [vars_[i] for i in perm]
            head_atom = f"{pred}({','.join(permuted_vars)})"

            coverage = 0
            support = 0
            distinct: set[tuple[str, ...]] = set()
            sample_body: Optional[tuple[str, ...]] = None
            sample_head: Optional[tuple[str, ...]] = None

            for args in tuples_set:
                coverage += 1
                permuted = tuple(args[i] for i in perm)
                if permuted in tuples_set:
                    support += 1
                    distinct.add(args)
                    if sample_body is None:
                        sample_body = args
                        sample_head = permuted

            sample_gr: Optional[str] = None
            if sample_body is not None:
                sample_gr = (
                    f"{pred}({','.join(sample_head)}) :- "
                    f"{pred}({','.join(sample_body)})"
                )

            prov = _collect_provenance(asserted_facts, pred, distinct)

            cand = _build_candidate(
                head_relation=pred,
                body_relations=(pred,),
                head=head_atom,
                body=(body_atom,),
                coverage=coverage,
                support=support,
                violations=0,
                distinct_bindings=len(distinct),
                sample_grounding=sample_gr,
                provenance=prov,
                br=br,
                thresholds=spec.mining_thresholds,
                min_coverage=min_coverage,
            )
            candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# CROSS_PREDICATE_REWRITE
# ---------------------------------------------------------------------------

def _mine_cross_predicate_rewrites(
    facts_by_pred: dict[str, set[tuple[str, ...]]],
    asserted_facts: list[Fact],
    spec: HypothesisLanguageSpec,
    min_coverage: int,
) -> list[CandidateRule]:
    """
    Testuje reguły między różnymi predykatami za pomocą permutacji argumentów.
    np. sub(C,A,B) :- add(A,B,C)
    """
    candidates: list[CandidateRule] = []
    limits = spec.template_limits.get("cross_predicate_rewrite", {})
    allowed_pairs: list[list[str]] = limits.get("allowed_pairs", [])

    for pair in allowed_pairs:
        if len(pair) != 2:
            continue
        src_pred, dst_pred = pair[0], pair[1]
        src_sig = spec.predicate_signatures.get(src_pred)
        dst_sig = spec.predicate_signatures.get(dst_pred)
        if src_sig is None or dst_sig is None:
            continue
        if src_sig.arity != dst_sig.arity:
            continue
        n = src_sig.arity

        src_tuples = facts_by_pred.get(src_pred, set())
        dst_tuples = facts_by_pred.get(dst_pred, set())
        if not src_tuples or not dst_tuples:
            continue

        src_vars = _VARS[:n]
        body_atom = f"{src_pred}({','.join(src_vars)})"
        br = _base_rate(dst_pred, facts_by_pred)

        for perm in permutations(range(n)):
            dst_vars = [src_vars[i] for i in perm]
            head_atom = f"{dst_pred}({','.join(dst_vars)})"

            coverage = 0
            support = 0
            distinct: set[tuple[str, ...]] = set()
            sample_body: Optional[tuple[str, ...]] = None
            sample_head: Optional[tuple[str, ...]] = None

            for src_args in src_tuples:
                coverage += 1
                mapped = tuple(src_args[i] for i in perm)
                if mapped in dst_tuples:
                    support += 1
                    distinct.add(src_args)
                    if sample_body is None:
                        sample_body = src_args
                        sample_head = mapped

            sample_gr: Optional[str] = None
            if sample_body is not None:
                sample_gr = (
                    f"{dst_pred}({','.join(sample_head)}) :- "
                    f"{src_pred}({','.join(sample_body)})"
                )

            prov = _collect_provenance(asserted_facts, src_pred, distinct)

            cand = _build_candidate(
                head_relation=dst_pred,
                body_relations=(src_pred,),
                head=head_atom,
                body=(body_atom,),
                coverage=coverage,
                support=support,
                violations=0,
                distinct_bindings=len(distinct),
                sample_grounding=sample_gr,
                provenance=prov,
                br=br,
                thresholds=spec.mining_thresholds,
                min_coverage=min_coverage,
            )
            candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------------
# CONSTANT_SCHEMA
# ---------------------------------------------------------------------------

def _mine_constant_schemas(
    facts_by_pred: dict[str, set[tuple[str, ...]]],
    asserted_facts: list[Fact],
    spec: HypothesisLanguageSpec,
    min_coverage: int,
) -> list[CandidateRule]:
    """
    Odkrywa schematy z jedną stałą, np.:
      add(X,0,X)  →  element neutralny
      add(0,X,X)  →  komutacja neutralności

    Dla arity=3 z stałą k na pozycji j:
      Sprawdza czy wynik (pozycja result_pos) == X (pozycja x_pos),
      gdzie x_pos ≠ j i result_pos ≠ j i x_pos ≠ result_pos.
    """
    if not spec.allow_constants_in_rules:
        return []
    candidates: list[CandidateRule] = []
    limits = spec.template_limits.get("constant_schema", {})
    allowed_preds = set(limits.get("allowed_predicates", spec.enabled_predicates))

    for pred in spec.enabled_predicates:
        if pred not in allowed_preds:
            continue
        sig = spec.predicate_signatures.get(pred)
        if sig is None or sig.arity != 3:
            continue
        tuples_set = facts_by_pred.get(pred, set())
        if not tuples_set:
            continue
        br = _base_rate(pred, facts_by_pred)

        for k in spec.allowed_constants:
            # Dla każdej pozycji, na której wstawiamy stałą k
            for const_pos in range(3):
                # Pary pozostałych pozycji (var_pos, result_pos)
                other = [i for i in range(3) if i != const_pos]
                for var_pos, result_pos in [(other[0], other[1]), (other[1], other[0])]:
                    # Schemat: pred(args) gdzie args[const_pos]=k, args[result_pos]=args[var_pos]
                    # Np. const_pos=1, var_pos=0, result_pos=2 → add(X,k,X)

                    # Buduj body i head atom
                    body_vars = list(_VARS[:3])
                    head_vars = list(_VARS[:3])
                    body_vars[const_pos] = k
                    head_vars[const_pos] = k
                    head_vars[result_pos] = body_vars[var_pos]

                    body_atom = f"{pred}({','.join(body_vars)})"
                    head_atom = f"{pred}({','.join(head_vars)})"

                    # Zbierz krotki gdzie args[const_pos] == k
                    coverage_domain: set[str] = set()  # uniq wartości X
                    support_xs: set[str] = set()       # X gdzie result == X
                    violations_xs: set[str] = set()    # X gdzie istnieje wynik ≠ X

                    # Pogrupuj po wartości var_pos
                    grouped: dict[str, set[str]] = defaultdict(set)
                    for args in tuples_set:
                        if args[const_pos] != k:
                            continue
                        x_val = args[var_pos]
                        res_val = args[result_pos]
                        grouped[x_val].add(res_val)

                    for x_val, results in grouped.items():
                        coverage_domain.add(x_val)
                        if x_val in results:
                            support_xs.add(x_val)
                        if any(r != x_val for r in results):
                            violations_xs.add(x_val)

                    coverage = len(coverage_domain)
                    support = len(support_xs)
                    violations = len(violations_xs)

                    if coverage == 0:
                        continue

                    sample_gr: Optional[str] = None
                    if support_xs:
                        x_ex = next(iter(support_xs))
                        sample_gr = (
                            f"{pred}({_example_args(const_pos, k, var_pos, x_ex, result_pos, x_ex)}) "
                            f":- {pred}({_example_args(const_pos, k, var_pos, x_ex, result_pos, x_ex)})"
                        )

                    prov = _collect_provenance(
                        asserted_facts, pred,
                        {args for args in tuples_set if args[const_pos] == k and args[var_pos] in support_xs}
                    )

                    cand = _build_candidate(
                        head_relation=pred,
                        body_relations=(pred,),
                        head=head_atom,
                        body=(body_atom,),
                        coverage=coverage,
                        support=support,
                        violations=violations,
                        distinct_bindings=len(support_xs),
                        sample_grounding=sample_gr,
                        provenance=prov,
                        br=br,
                        thresholds=spec.mining_thresholds,
                        min_coverage=min_coverage,
                    )
                    candidates.append(cand)

    return candidates


def _example_args(
    const_pos: int, k: str,
    var_pos: int, x_val: str,
    result_pos: int, r_val: str,
) -> str:
    args = ["?", "?", "?"]
    args[const_pos] = k
    args[var_pos] = x_val
    args[result_pos] = r_val
    return ",".join(args)


# ---------------------------------------------------------------------------
# FUNCTIONALITY_CONSTRAINT
# ---------------------------------------------------------------------------

def _mine_functionality_constraints(
    facts_by_pred: dict[str, set[tuple[str, ...]]],
    asserted_facts: list[Fact],
    spec: HypothesisLanguageSpec,
    min_coverage: int,
) -> list[CandidateRule]:
    """
    Meta-reguły wyrażające determinizm: det_positions → dep_position.
    Nie są regułami Horn — zawsze mają 'not_a_horn_rule' w rejection_reasons
    i nie będą promowane. Służą jako sygnał jakości danych.

    Dla arity=3 testuje: {0,1}→2, {0,2}→1, {1,2}→0.
    """
    candidates: list[CandidateRule] = []

    for pred in spec.enabled_predicates:
        sig = spec.predicate_signatures.get(pred)
        if sig is None or sig.arity < 2:
            continue
        n = sig.arity
        tuples_set = facts_by_pred.get(pred, set())
        if not tuples_set:
            continue

        vars_ = _VARS[:n]
        body_atom = f"{pred}({','.join(vars_)})"

        # Generuj kandydatów (det_positions, dep_position)
        for dep_pos in range(n):
            det_positions = [i for i in range(n) if i != dep_pos]

            # Zgrupuj po krotce wartości na det_positions
            groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
            for args in tuples_set:
                key = tuple(args[i] for i in det_positions)
                groups[key].add(args[dep_pos])

            support = len(groups)
            violations = sum(1 for vals in groups.values() if len(vals) > 1)
            confidence = (support - violations) / support if support > 0 else 0.0

            det_str = "_".join(str(i) for i in det_positions)
            head_atom = f"func_{pred}_{dep_pos}({','.join(vars_)})"

            rejection = ["not_a_horn_rule"]
            if support < min_coverage:
                rejection.append("low_support")

            prov = _collect_provenance(asserted_facts, pred, tuples_set)

            candidates.append(CandidateRule(
                head_relation=pred,
                body_relations=(pred,),
                head=head_atom,
                body=(body_atom,),
                coverage=support,
                support=support - violations,
                confidence=confidence,
                lift=1.0,
                base_rate=1.0,
                corruption_hits=0,
                local_cwa_negatives=0,
                violations=violations,
                sample_grounding=None,
                promotable=False,
                rejection_reasons=rejection,
                provenance=prov,
            ))

    return candidates


# ---------------------------------------------------------------------------
# COMPOSITION_EQUIVALENCE
# ---------------------------------------------------------------------------

def _mine_composition_equivalences(
    facts_by_pred: dict[str, set[tuple[str, ...]]],
    asserted_facts: list[Fact],
    spec: HypothesisLanguageSpec,
    min_coverage: int,
) -> list[CandidateRule]:
    """
    Testuje równoważność dwóch łańcuchów 2-hop na tym samym predykacie.

    Domyślna konfiguracja (łączność add):
      left:  add(A,B,S1), add(S1,C,R)   →  (A+B)+C = R
      right: add(B,C,S2), add(A,S2,R)   →  A+(B+C) = R

    Dla każdej trójki (A,B,C) gdzie oba łańcuchy są obliczalne:
      coverage    = liczba takich trójek
      support     = trójki gdzie R_left == R_right
      counterexamples = trójki gdzie R_left != R_right

    Emitowana reguła Horn (prawostronny łańcuch z lewostronnym jako warunek):
      add(A,S2,R) :- add(A,B,S1), add(S1,C,R), add(B,C,S2)
    """
    candidates: list[CandidateRule] = []
    limits = spec.template_limits.get("composition_equivalence", {})
    chains_cfg: list[dict] = limits.get("chains", [])

    for chain_def in chains_cfg:
        pred: str = chain_def.get("predicate", "")
        label: str = chain_def.get("label", f"composition_equiv({pred})")
        tuples_set = facts_by_pred.get(pred, set())
        if not tuples_set:
            continue

        # Indeksy dla szybkiego lookupowania
        # by_first[a] = {(b, r)} — dla pred(a, b, r)
        by_first: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for a, b, r in tuples_set:
            by_first[a].add((b, r))

        # left_results[(a,b,c)] = set of R_left
        # Lewy łańcuch: pred(A,B,S1) → pred(S1,C,R)
        left_results: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for a, b, s1 in tuples_set:
            for c, r_left in by_first.get(s1, set()):
                left_results[(a, b, c)].add(r_left)

        # Prawy łańcuch: pred(B,C,S2) → pred(A,S2,R)
        # Potrzebujemy indeksu: by_second_arg[s2] = {(a, r)} dla pred(a, s2, r)
        # by_second_arg[s2] = {(a, r)} — dla pred(a, s2, r)
        by_second_arg: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for a, b, r in tuples_set:
            by_second_arg[b].add((a, r))

        right_results = defaultdict(set)
        for b, c, s2 in tuples_set:
            for a, r_right in by_second_arg.get(s2, set()):
                right_results[(a, b, c)].add(r_right)

        # Wyznacz coverage, support, counterexamples
        common_triples = set(left_results.keys()) & set(right_results.keys())
        coverage = len(common_triples)
        if coverage == 0:
            continue

        support = 0
        counterexamples = 0
        distinct: set[tuple[str, str, str]] = set()
        sample_gr: Optional[str] = None

        for abc in common_triples:
            l_rs = left_results[abc]
            r_rs = right_results[abc]
            if l_rs & r_rs:
                support += 1
                distinct.add(abc)
                if sample_gr is None:
                    a_ex, b_ex, c_ex = abc
                    r_ex = next(iter(l_rs & r_rs))
                    # Znajdź S1 i S2 dla przykładu
                    s1_ex = next((s for a2, b2, s in tuples_set if a2 == a_ex and b2 == b_ex), "?")
                    s2_ex = next((s for b2, c2, s in tuples_set if b2 == b_ex and c2 == c_ex), "?")
                    sample_gr = (
                        f"add({a_ex},{s2_ex},{r_ex}) :- "
                        f"add({a_ex},{b_ex},{s1_ex}), add({s1_ex},{c_ex},{r_ex}), add({b_ex},{c_ex},{s2_ex})"
                    )
            else:
                counterexamples += 1

        confidence = support / coverage if coverage else 0.0
        br = _base_rate(pred, facts_by_pred)

        # Reguła Horn: add(A,S2,R) :- add(A,B,S1), add(S1,C,R), add(B,C,S2)
        head_atom = f"{pred}(A,S2,R)"
        body = (
            f"{pred}(A,B,S1)",
            f"{pred}(S1,C,R)",
            f"{pred}(B,C,S2)",
        )

        rejection: list[str] = []
        if coverage < min_coverage:
            rejection.append("low_coverage")
        if support < spec.mining_thresholds.min_support:
            rejection.append("low_support")
        if confidence < spec.mining_thresholds.min_confidence:
            rejection.append("low_confidence")
        if len(distinct) < spec.mining_thresholds.min_distinct_bindings:
            rejection.append("low_distinct_bindings")
        if counterexamples > spec.mining_thresholds.max_counterexamples:
            rejection.append("constraint_violations")

        prov = _collect_provenance(asserted_facts, pred, {tuple(t) for t in tuples_set})

        candidates.append(CandidateRule(
            head_relation=pred,
            body_relations=(pred, pred, pred),
            head=head_atom,
            body=body,
            coverage=coverage,
            support=support,
            confidence=confidence,
            lift=_lift(confidence, br),
            base_rate=br,
            corruption_hits=0,
            local_cwa_negatives=0,
            violations=counterexamples,
            sample_grounding=sample_gr,
            promotable=len(rejection) == 0,
            rejection_reasons=rejection,
            provenance=prov,
        ))

    return candidates


# ---------------------------------------------------------------------------
# Deduplikacja i ranking
# ---------------------------------------------------------------------------

def _merge_candidates(candidates: list[CandidateRule]) -> list[CandidateRule]:
    """Deduplikuje po kluczu (head, body) — zachowuje kandydata z lepszym wynikiem."""
    best: dict[tuple[str, tuple[str, ...]], CandidateRule] = {}
    for cand in candidates:
        key = (cand.head, cand.body)
        prev = best.get(key)
        if prev is None:
            best[key] = cand
            continue
        prev_score = (prev.confidence, prev.support)
        now_score = (cand.confidence, cand.support)
        if now_score > prev_score:
            best[key] = cand
    return list(best.values())


def _rank_candidates(candidates: list[CandidateRule], top_k: int) -> list[CandidateRule]:
    return sorted(
        candidates,
        key=lambda c: (
            1 if c.promotable else 0,
            c.confidence,
            c.support,
            c.lift if math.isfinite(c.lift) else float("inf"),
            -len(c.rejection_reasons),
        ),
        reverse=True,
    )[:top_k]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class BruteForcePatternMiner(RuleBootstrapBase):
    """
    Miner szablonów hipotez (MVP).
    Implementuje port RuleBootstrap przez mining relacyjnych faktów instancji.

    Wejście: Fact(h="add(2,5,7)", r="instance_of", t="add")
    Wyjście: CandidateRule dla każdego odkrytego wzorca.
    """

    def __init__(
        self,
        config: Optional[BootstrapConfig] = None,
        spec: Optional[HypothesisLanguageSpec] = None,
    ) -> None:
        self.config = config or BootstrapConfig()
        self.spec = spec or spec_from_bootstrap_config(self.config)

    def mine_candidates(
        self, facts: list[Fact]
    ) -> tuple[list[CandidateRule], int, int, int]:
        asserted = [f for f in facts if f.status == FactStatus.ASSERTED]
        facts_by_pred = _extract_relational_facts(
            asserted, set(self.spec.enabled_predicates)
        )

        all_candidates: list[CandidateRule] = []
        pattern_count = 0
        min_cov = self.config.min_coverage

        miners = [
            ("ARG_PERMUTATION", _mine_arg_permutations),
            ("CROSS_PREDICATE_REWRITE", _mine_cross_predicate_rewrites),
            ("CONSTANT_SCHEMA", _mine_constant_schemas),
            ("FUNCTIONALITY_CONSTRAINT", _mine_functionality_constraints),
            ("COMPOSITION_EQUIVALENCE", _mine_composition_equivalences),
        ]

        for kind, fn in miners:
            if kind not in self.spec.allowed_template_kinds:
                continue
            cands = fn(facts_by_pred, asserted, self.spec, min_cov)
            all_candidates.extend(cands)
            pattern_count += len(cands)

        merged = _merge_candidates(all_candidates)
        ranked = _rank_candidates(merged, self.config.top_k)

        return ranked, len(asserted), len(facts_by_pred), pattern_count

    # bootstrap() odziedziczone z RuleBootstrapBase
