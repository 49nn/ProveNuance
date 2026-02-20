"""
Adapter: RuleBootstrapper
Implementuje port RuleBootstrap.

Goal:
- learn candidate Horn rules from asserted facts when no asserted rules exist yet,
- score candidates conservatively,
- optionally persist them as hypothesis and promote only safe rules.

Current learner mines path rules:
- length 2: R1(X,Y) & R2(Y,Z) -> RH(X,Z)
- length 3: R1(X,Y) & R2(Y,W) & R3(W,Z) -> RH(X,Z)
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import DefaultDict, Iterable, Optional

from contracts import Fact, FactStatus, Rule
from ports.knowledge_store import KnowledgeStore
from ports.rule_bootstrap import (
    BootstrapConfig,
    BootstrapSummary,
    CandidateRule,
    RuleBootstrap,
)

_REL_RE = re.compile(r"^[A-Za-z_]\w*$")
_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")
_ATOM_RE = re.compile(r"^[A-Za-z_]\w*\([^)]*\)$")
_VAR_RE = re.compile(r"^[A-Z][A-Za-z0-9_]*$|^\?[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class _Grounding:
    x: str
    z: str
    body_edges: tuple[tuple[str, str, str], ...]  # (rel, src, dst)


@dataclass(frozen=True)
class _PatternSpec:
    body_relations: tuple[str, ...]
    body_vars: tuple[tuple[str, str], ...]  # (src_var, dst_var)

def _is_valid_relation(rel: str) -> bool:
    return bool(_REL_RE.match(rel.strip()))


def _normalized_config(config: BootstrapConfig) -> BootstrapConfig:
    clean = BootstrapConfig(**config.__dict__)
    clean.max_body_literals = 2 if config.max_body_literals <= 2 else 3
    clean.min_coverage = max(1, config.min_coverage)
    clean.min_support = max(1, config.min_support)
    clean.min_confidence = min(1.0, max(0.0, config.min_confidence))
    clean.min_lift = max(0.0, config.min_lift)
    clean.max_violations = max(0, config.max_violations)
    clean.max_corruption_hits = max(0, config.max_corruption_hits)
    clean.max_local_cwa_negatives = max(0, config.max_local_cwa_negatives)
    clean.corruption_samples = max(0, config.corruption_samples)
    clean.max_groundings_per_pattern = max(1, config.max_groundings_per_pattern)
    clean.top_k = max(1, config.top_k)
    clean.priority = int(config.priority)
    clean.relation_whitelist = {r for r in clean.relation_whitelist if _is_valid_relation(r)}
    clean.functional_relations = {
        r for r in clean.functional_relations if _is_valid_relation(r)
    }
    return clean


def _term_type(term: str) -> str:
    raw = term.strip()
    if _INT_RE.match(raw):
        return "int"
    if _FLOAT_RE.match(raw):
        return "float"
    if _ATOM_RE.match(raw):
        return "compound"
    return "symbol"


def _atom(rel: str, src: str, dst: str) -> str:
    return f"{rel}({src},{dst})"


def _extract_vars(atom: str) -> set[str]:
    left = atom.find("(")
    right = atom.rfind(")")
    if left <= 0 or right <= left:
        return set()
    raw_args = atom[left + 1 : right].strip()
    if not raw_args:
        return set()
    out: set[str] = set()
    for arg in raw_args.split(","):
        tok = arg.strip()
        if _VAR_RE.match(tok):
            out.add(tok)
    return out


def _is_safe_horn_rule(head: str, body: Iterable[str], max_body_literals: int) -> bool:
    body_list = [b.strip() for b in body if b.strip()]
    if not head.strip():
        return False
    if len(body_list) == 0 or len(body_list) > max_body_literals:
        return False
    for atom in [head, *body_list]:
        if atom.lstrip().startswith("not "):
            return False
    head_vars = _extract_vars(head)
    body_vars: set[str] = set()
    for atom in body_list:
        body_vars.update(_extract_vars(atom))
    return head_vars.issubset(body_vars)


def _build_indexes(
    facts: list[Fact],
    relation_whitelist: set[str],
) -> tuple[
    list[Fact],
    set[str],
    dict[str, set[tuple[str, str]]],
    dict[str, dict[str, set[str]]],
    dict[tuple[str, str], set[str]],
    dict[str, dict[tuple[str, str], set[str]]],
    dict[str, dict[str, set[tuple[str, str]]]],
    dict[str, dict[str, set[str]]],
    dict[str, dict[str, set[str]]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    asserted = [
        f
        for f in facts
        if f.status == FactStatus.ASSERTED
        and f.r in relation_whitelist
        and _is_valid_relation(f.r)
    ]
    relations = {f.r for f in asserted}

    pairs_by_rel: DefaultDict[str, set[tuple[str, str]]] = defaultdict(set)
    outgoing_by_rel: DefaultDict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    rels_by_pair: DefaultDict[tuple[str, str], set[str]] = defaultdict(set)
    prov_by_rel_pair: DefaultDict[str, dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    span_pairs_by_rel: DefaultDict[str, dict[str, set[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    tails_by_rel_head: DefaultDict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    domain_types: DefaultDict[str, set[str]] = defaultdict(set)
    range_types: DefaultDict[str, set[str]] = defaultdict(set)
    terms_by_type: DefaultDict[str, set[str]] = defaultdict(set)

    for f in asserted:
        pair = (f.h, f.t)
        pairs_by_rel[f.r].add(pair)
        outgoing_by_rel[f.r][f.h].add(f.t)
        rels_by_pair[pair].add(f.r)
        tails_by_rel_head[f.r][f.h].add(f.t)
        for sid in f.provenance:
            if sid:
                prov_by_rel_pair[f.r][pair].add(sid)
                span_pairs_by_rel[f.r][sid].add(pair)
        dh = _term_type(f.h)
        dt = _term_type(f.t)
        domain_types[f.r].add(dh)
        range_types[f.r].add(dt)
        terms_by_type[dh].add(f.h)
        terms_by_type[dt].add(f.t)

    return (
        asserted,
        relations,
        dict(pairs_by_rel),
        {r: dict(v) for r, v in outgoing_by_rel.items()},
        dict(rels_by_pair),
        {r: dict(v) for r, v in prov_by_rel_pair.items()},
        {r: dict(v) for r, v in span_pairs_by_rel.items()},
        {r: dict(v) for r, v in tails_by_rel_head.items()},
        dict(domain_types),
        dict(range_types),
        dict(terms_by_type),
    )


def _base_rate_for_relation(
    rel: str,
    pairs_by_rel: dict[str, set[tuple[str, str]]],
    domain_types: dict[str, set[str]],
    range_types: dict[str, set[str]],
    terms_by_type: dict[str, set[str]],
) -> float:
    edge_count = len(pairs_by_rel.get(rel, set()))
    if edge_count == 0:
        return 0.0

    domains = domain_types.get(rel, set())
    ranges = range_types.get(rel, set())
    possible = 0
    for d_t in domains:
        for r_t in ranges:
            possible += len(terms_by_type.get(d_t, set())) * len(
                terms_by_type.get(r_t, set())
            )

    if possible <= 0:
        all_terms = set()
        for vals in terms_by_type.values():
            all_terms.update(vals)
        size = max(1, len(all_terms))
        possible = size * size
    return edge_count / float(possible)


def _enumerate_patterns(relations: list[str], max_body_literals: int) -> list[_PatternSpec]:
    out: list[_PatternSpec] = []
    for r1 in relations:
        for r2 in relations:
            out.append(
                _PatternSpec(
                    body_relations=(r1, r2),
                    body_vars=(("X", "Y"), ("Y", "Z")),
                )
            )
    if max_body_literals >= 3:
        for r1 in relations:
            for r2 in relations:
                for r3 in relations:
                    out.append(
                        _PatternSpec(
                            body_relations=(r1, r2, r3),
                            body_vars=(("X", "Y"), ("Y", "W"), ("W", "Z")),
                        )
                    )
    return out


def _mine_groundings_for_pattern(
    pattern: _PatternSpec,
    outgoing_by_rel: dict[str, dict[str, set[str]]],
    max_groundings: int,
) -> list[_Grounding]:
    rels = pattern.body_relations
    out: list[_Grounding] = []
    if len(rels) == 2:
        r1, r2 = rels
        out_r1 = outgoing_by_rel.get(r1, {})
        out_r2 = outgoing_by_rel.get(r2, {})
        for x, ys in out_r1.items():
            for y in ys:
                for z in out_r2.get(y, set()):
                    out.append(
                        _Grounding(
                            x=x,
                            z=z,
                            body_edges=((r1, x, y), (r2, y, z)),
                        )
                    )
                    if len(out) >= max_groundings:
                        return out
        return out

    r1, r2, r3 = rels
    out_r1 = outgoing_by_rel.get(r1, {})
    out_r2 = outgoing_by_rel.get(r2, {})
    out_r3 = outgoing_by_rel.get(r3, {})
    for x, ys in out_r1.items():
        for y in ys:
            for w in out_r2.get(y, set()):
                for z in out_r3.get(w, set()):
                    out.append(
                        _Grounding(
                            x=x,
                            z=z,
                            body_edges=((r1, x, y), (r2, y, w), (r3, w, z)),
                        )
                    )
                    if len(out) >= max_groundings:
                        return out
    return out


def _local_cwa_negatives(
    groundings: list[_Grounding],
    head_rel: str,
    pairs_by_rel: dict[str, set[tuple[str, str]]],
    prov_by_rel_pair: dict[str, dict[tuple[str, str], set[str]]],
    span_pairs_by_rel: dict[str, dict[str, set[tuple[str, str]]]],
) -> int:
    head_pairs = pairs_by_rel.get(head_rel, set())
    head_span_pairs = span_pairs_by_rel.get(head_rel, {})
    negatives = 0

    for g in groundings:
        pair = (g.x, g.z)
        if pair in head_pairs:
            continue
        shared: Optional[set[str]] = None
        for rel, src, dst in g.body_edges:
            edge_spans = set(prov_by_rel_pair.get(rel, {}).get((src, dst), set()))
            if shared is None:
                shared = edge_spans
            else:
                shared &= edge_spans
            if not shared:
                break
        if not shared:
            continue
        if all(pair not in head_span_pairs.get(sid, set()) for sid in shared):
            negatives += 1
    return negatives


def _corruption_hits(
    support_groundings: list[_Grounding],
    head_rel: str,
    pairs_by_rel: dict[str, set[tuple[str, str]]],
    outgoing_by_rel: dict[str, dict[str, set[str]]],
    terms_by_type: dict[str, set[str]],
    samples_per_positive: int,
) -> int:
    if samples_per_positive <= 0:
        return 0

    head_pairs = pairs_by_rel.get(head_rel, set())
    hits = 0
    for g in support_groundings:
        _, last_src, _ = g.body_edges[-1]
        last_rel = g.body_edges[-1][0]
        possible_z = outgoing_by_rel.get(last_rel, {}).get(last_src, set())
        if not possible_z:
            continue
        z_type = _term_type(g.z)
        pool = sorted(terms_by_type.get(z_type, set()))
        taken = 0
        for z_alt in pool:
            if z_alt == g.z:
                continue
            if (g.x, z_alt) in head_pairs:
                continue
            taken += 1
            if z_alt in possible_z:
                hits += 1
            if taken >= samples_per_positive:
                break
    return hits


def _violations_after_application(
    head_rel: str,
    predicted_new_pairs: set[tuple[str, str]],
    tails_by_rel_head: dict[str, dict[str, set[str]]],
    functional_relations: set[str],
) -> int:
    if head_rel not in functional_relations:
        return 0
    existing_map = tails_by_rel_head.get(head_rel, {})
    predicted_by_head: DefaultDict[str, set[str]] = defaultdict(set)
    for h, t in predicted_new_pairs:
        predicted_by_head[h].add(t)

    violations = 0
    for h, tails in predicted_by_head.items():
        existing = set(existing_map.get(h, set()))
        if len(existing) > 1:
            # Ignore pre-existing broken rows when scoring new rule.
            continue
        merged = existing | tails
        if len(merged) > 1:
            violations += len(merged) - 1
    return violations


def _sample_grounding_text(head_rel: str, g: Optional[_Grounding]) -> Optional[str]:
    if g is None:
        return None
    body = ", ".join(_atom(rel, src, dst) for rel, src, dst in g.body_edges)
    return f"{_atom(head_rel, g.x, g.z)} :- {body}"


def _score_candidate(
    head_rel: str,
    pattern: _PatternSpec,
    groundings: list[_Grounding],
    support_groundings: list[_Grounding],
    config: BootstrapConfig,
    pairs_by_rel: dict[str, set[tuple[str, str]]],
    outgoing_by_rel: dict[str, dict[str, set[str]]],
    prov_by_rel_pair: dict[str, dict[tuple[str, str], set[str]]],
    span_pairs_by_rel: dict[str, dict[str, set[tuple[str, str]]]],
    tails_by_rel_head: dict[str, dict[str, set[str]]],
    base_rate: float,
    terms_by_type: dict[str, set[str]],
) -> CandidateRule:
    coverage = len(groundings)
    support = len(support_groundings)
    confidence = 0.0 if coverage == 0 else support / float(coverage)
    if base_rate == 0.0:
        lift = math.inf if confidence > 0.0 else 0.0
    else:
        lift = confidence / base_rate

    predicted_new_pairs = {
        (g.x, g.z)
        for g in groundings
        if (g.x, g.z) not in pairs_by_rel.get(head_rel, set())
    }

    violations = _violations_after_application(
        head_rel=head_rel,
        predicted_new_pairs=predicted_new_pairs,
        tails_by_rel_head=tails_by_rel_head,
        functional_relations=config.functional_relations,
    )
    corruption = _corruption_hits(
        support_groundings=support_groundings,
        head_rel=head_rel,
        pairs_by_rel=pairs_by_rel,
        outgoing_by_rel=outgoing_by_rel,
        terms_by_type=terms_by_type,
        samples_per_positive=config.corruption_samples,
    )
    local_cwa = (
        _local_cwa_negatives(
            groundings=groundings,
            head_rel=head_rel,
            pairs_by_rel=pairs_by_rel,
            prov_by_rel_pair=prov_by_rel_pair,
            span_pairs_by_rel=span_pairs_by_rel,
        )
        if config.use_local_cwa
        else 0
    )

    body_atoms = tuple(
        _atom(rel, src_var, dst_var)
        for rel, (src_var, dst_var) in zip(pattern.body_relations, pattern.body_vars)
    )
    head_atom = _atom(head_rel, "X", "Z")

    reasons: list[str] = []
    if not _is_safe_horn_rule(head_atom, body_atoms, config.max_body_literals):
        reasons.append("unsafe_horn")
    if coverage < config.min_coverage:
        reasons.append("low_coverage")
    if support < config.min_support:
        reasons.append("low_support")
    if confidence < config.min_confidence:
        reasons.append("low_confidence")
    if lift < config.min_lift:
        reasons.append("low_lift")
    if violations > config.max_violations:
        reasons.append("constraint_violations")
    if corruption > config.max_corruption_hits:
        reasons.append("corruption_hits")
    if local_cwa > config.max_local_cwa_negatives:
        reasons.append("local_cwa_negatives")

    provenance = sorted(
        {
            sid
            for g in support_groundings
            for rel, src, dst in g.body_edges
            for sid in prov_by_rel_pair.get(rel, {}).get((src, dst), set())
        }
    )[:40]

    return CandidateRule(
        head_relation=head_rel,
        body_relations=pattern.body_relations,
        head=head_atom,
        body=body_atoms,
        coverage=coverage,
        support=support,
        confidence=confidence,
        lift=lift,
        base_rate=base_rate,
        corruption_hits=corruption,
        local_cwa_negatives=local_cwa,
        violations=violations,
        sample_grounding=_sample_grounding_text(
            head_rel, support_groundings[0] if support_groundings else None
        ),
        promotable=len(reasons) == 0,
        rejection_reasons=reasons,
        provenance=provenance,
    )


class RuleBootstrapper(RuleBootstrap):
    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = _normalized_config(config or BootstrapConfig())

    def mine_candidates(
        self, facts: list[Fact]
    ) -> tuple[list[CandidateRule], int, int, int]:
        (
            asserted,
            rels_set,
            pairs_by_rel,
            outgoing_by_rel,
            rels_by_pair,
            prov_by_rel_pair,
            span_pairs_by_rel,
            tails_by_rel_head,
            domain_types,
            range_types,
            terms_by_type,
        ) = _build_indexes(facts, self.config.relation_whitelist)

        relations = sorted(rels_set)
        if not relations:
            return [], 0, 0, 0

        base_rates = {
            rel: _base_rate_for_relation(
                rel=rel,
                pairs_by_rel=pairs_by_rel,
                domain_types=domain_types,
                range_types=range_types,
                terms_by_type=terms_by_type,
            )
            for rel in relations
        }

        patterns = _enumerate_patterns(relations, self.config.max_body_literals)
        out_candidates: dict[tuple[str, tuple[str, ...]], CandidateRule] = {}
        pattern_count = 0

        for pattern in patterns:
            groundings = _mine_groundings_for_pattern(
                pattern=pattern,
                outgoing_by_rel=outgoing_by_rel,
                max_groundings=self.config.max_groundings_per_pattern,
            )
            if not groundings:
                continue

            coverage = len(groundings)
            support_counter: Counter[str] = Counter()
            support_groundings_map: DefaultDict[str, list[_Grounding]] = defaultdict(list)
            for g in groundings:
                for head_rel in rels_by_pair.get((g.x, g.z), set()):
                    if head_rel not in self.config.relation_whitelist:
                        continue
                    support_counter[head_rel] += 1
                    support_groundings_map[head_rel].append(g)

            if not support_counter:
                continue

            pattern_count += 1
            for head_rel, support in support_counter.items():
                if support == 0 or coverage == 0:
                    continue
                candidate = _score_candidate(
                    head_rel=head_rel,
                    pattern=pattern,
                    groundings=groundings,
                    support_groundings=support_groundings_map[head_rel],
                    config=self.config,
                    pairs_by_rel=pairs_by_rel,
                    outgoing_by_rel=outgoing_by_rel,
                    prov_by_rel_pair=prov_by_rel_pair,
                    span_pairs_by_rel=span_pairs_by_rel,
                    tails_by_rel_head=tails_by_rel_head,
                    base_rate=base_rates.get(head_rel, 0.0),
                    terms_by_type=terms_by_type,
                )
                key = (candidate.head, candidate.body)
                prev = out_candidates.get(key)
                if prev is None:
                    out_candidates[key] = candidate
                    continue
                prev_score = (prev.confidence, prev.support, prev.lift)
                now_score = (candidate.confidence, candidate.support, candidate.lift)
                if now_score > prev_score:
                    out_candidates[key] = candidate

        ranked = sorted(
            out_candidates.values(),
            key=lambda c: (
                1 if c.promotable else 0,
                c.confidence,
                c.support,
                c.lift if math.isfinite(c.lift) else float("inf"),
                -len(c.rejection_reasons),
            ),
            reverse=True,
        )[: self.config.top_k]

        return ranked, len(asserted), len(relations), pattern_count

    async def bootstrap(self, store: KnowledgeStore) -> BootstrapSummary:
        asserted_facts: list[Fact] = []
        async for fact in store.get_asserted_facts():
            asserted_facts.append(fact)

        (
            candidates,
            asserted_fact_count,
            relation_count,
            pattern_count,
        ) = self.mine_candidates(asserted_facts)

        created = 0
        updated = 0
        promoted = 0

        for cand in candidates:
            should_store = self.config.store_candidates or (
                self.config.promote and cand.promotable
            )
            if self.config.dry_run or not should_store:
                continue

            rule = Rule(
                head=cand.head,
                body=list(cand.body),
                status=FactStatus.HYPOTHESIS,
                provenance=list(cand.provenance),
                priority=self.config.priority,
            )
            ref = await store.upsert_rule(rule)
            cand.rule_id = ref.rule_id
            if ref.created:
                created += 1
            else:
                updated += 1

            if self.config.promote and cand.promotable:
                current = await store.get_rule(ref.rule_id)
                if current.status != FactStatus.ASSERTED:
                    await store.promote_rule(ref.rule_id, reason=self.config.promotion_reason)
                    cand.promoted = True
                    promoted += 1

        return BootstrapSummary(
            asserted_fact_count=asserted_fact_count,
            relation_count=relation_count,
            pattern_count=pattern_count,
            candidate_count=len(candidates),
            stored_created=created,
            stored_updated=updated,
            promoted=promoted,
            candidates=candidates,
        )
