"""
Adapter: ForwardChainer (SLD-resolution backward chainer)
Implementuje port Reasoner.

Algorytm: SLD-resolution top-down DFS (à la Prolog)
  - Unifikacja Robinson'a
  - Rename zmiennych per krok (zapobiega capture)
  - Głębokość i timeout z LogicQuery

Formaty atomów:
  add(3,4,7)   — funktor + argumenty w nawiasach, przecinkami
  X, Y, Z      — zmienne: wielka pierwsza litera lub prefiks '?'

Konwersja faktów z KnowledgeStore:
  Fakt (h="add(3,4,7)", r="instance_of", t="add") → atom "add(3,4,7)"
  Fakt (h="add(3,4)", r="eq", t="7")              → atom "eq(add(3,4),7)"
"""
from __future__ import annotations

import re
import time
from typing import Iterator, Optional

from contracts import (
    Fact,
    LogicQuery,
    ProofStep,
    ReasonerResult,
    Rule,
    SnapshotRef,
)

# ──────────────────────────────────────────────────────────────────────────────
# Reprezentacja i parsowanie atomów
# ──────────────────────────────────────────────────────────────────────────────

_ATOM_RE = re.compile(r'^([A-Za-z_]\w*)\(([^)]*)\)$')
_VAR_TOKEN_RE = re.compile(r'\b([A-Z][A-Za-z0-9_]*|\?[A-Za-z0-9_]+)\b')


def _parse_atom(s: str) -> tuple[str, list[str]] | None:
    """'add(3,4,7)' → ('add', ['3', '4', '7']).  None jeśli nie atom."""
    m = _ATOM_RE.match(s.strip())
    if not m:
        return None
    functor = m.group(1)
    args_raw = m.group(2).strip()
    args = [a.strip() for a in args_raw.split(",")] if args_raw else []
    return functor, args


def _make_atom(functor: str, args: list[str]) -> str:
    return f"{functor}({','.join(args)})"


def _is_var(term: str) -> bool:
    """Zmienna: zaczyna się wielką literą (A–Z) lub '?'."""
    return bool(term) and (term[0].isupper() or term[0] == "?")


# ──────────────────────────────────────────────────────────────────────────────
# Podstawienia (substitution)
# ──────────────────────────────────────────────────────────────────────────────

Subst = dict[str, str]


def _walk(term: str, subst: Subst) -> str:
    """Podąża łańcuchem podstawień aż do term który sam siebie reprezentuje."""
    seen: set[str] = set()
    while _is_var(term) and term in subst and term not in seen:
        seen.add(term)
        term = subst[term]
    return term


def _apply(term: str, subst: Subst) -> str:
    """Aplikuje podstawienie rekurencyjnie (wewnątrz atomu też)."""
    term = _walk(term, subst)
    parsed = _parse_atom(term)
    if parsed:
        functor, args = parsed
        return _make_atom(functor, [_apply(a, subst) for a in args])
    return term


# ──────────────────────────────────────────────────────────────────────────────
# Unifikacja Robinson'a
# ──────────────────────────────────────────────────────────────────────────────

def _unify(t1: str, t2: str, subst: Subst) -> Subst | None:
    """
    Unifikuje dwa termy w kontekście podstawienia.
    Zwraca rozszerzone podstawienie lub None (niepowodzenie).
    Occur-check pominięty dla wydajności (MVP).
    """
    t1 = _walk(t1, subst)
    t2 = _walk(t2, subst)

    if t1 == t2:
        return subst

    if _is_var(t1):
        return {**subst, t1: t2}
    if _is_var(t2):
        return {**subst, t2: t1}

    p1 = _parse_atom(t1)
    p2 = _parse_atom(t2)
    if p1 and p2:
        f1, args1 = p1
        f2, args2 = p2
        if f1 != f2 or len(args1) != len(args2):
            return None
        current = subst
        for a1, a2 in zip(args1, args2):
            current = _unify(a1, a2, current)
            if current is None:
                return None
        return current

    return None  # dwa różne stałe


# ──────────────────────────────────────────────────────────────────────────────
# Rename zmiennych (zapobiega capture)
# ──────────────────────────────────────────────────────────────────────────────

def _rename(atom: str, suffix: str) -> str:
    """X → X_{suffix}, ?Z → ?Z_{suffix} — wszystkie zmienne w atomie."""
    def _rep(m: re.Match) -> str:
        v = m.group(1)
        if v.startswith("?"):
            return f"?{v[1:]}_{suffix}"
        return f"{v}_{suffix}"
    return _VAR_TOKEN_RE.sub(_rep, atom)


def _rename_rule(
    head: str, body: list[str], suffix: str
) -> tuple[str, list[str]]:
    return _rename(head, suffix), [_rename(b, suffix) for b in body]


# ──────────────────────────────────────────────────────────────────────────────
# SLD solver (DFS)
# ──────────────────────────────────────────────────────────────────────────────

class _SLDSolver:
    """
    Rozwiązuje listę celów przez SLD-resolution.
    Generuje kolejne rozwiązania jako (Subst, proof_steps).
    """

    def __init__(
        self,
        ground_atoms: list[tuple[str, str, list[str]]],  # (atom, fact_id, span_ids)
        rules: list[tuple[str, list[str], str]],          # (head, body, rule_id)
        max_depth: int,
        deadline: float,
    ) -> None:
        self._atoms = ground_atoms
        self._rules = rules
        self._max_depth = max_depth
        self._deadline = deadline
        self._counter = 0
        self.used_fact_ids: set[str] = set()
        self.used_rule_ids: set[str] = set()
        self.depth_reached = 0

    def solve(
        self,
        goals: list[str],
        subst: Subst,
        depth: int,
        proof: list[ProofStep],
    ) -> Iterator[tuple[Subst, list[ProofStep]]]:
        if time.monotonic() > self._deadline:
            raise TimeoutError("Reasoner timeout exceeded")

        self.depth_reached = max(self.depth_reached, depth)

        if not goals:
            yield subst, proof
            return

        if depth >= self._max_depth:
            return

        goal = _apply(goals[0], subst)
        rest_goals = goals[1:]
        step_id = depth + 1

        # ── Próbuj ground atoms (fakty) ──────────────────────────────────────
        for atom_str, fact_id, span_ids in self._atoms:
            new_subst = _unify(goal, atom_str, dict(subst))
            if new_subst is None:
                continue

            self.used_fact_ids.add(fact_id)
            step = ProofStep(
                step_id=step_id,
                rule_or_fact_id=fact_id,
                substitution={k: v for k, v in new_subst.items() if _is_var(k)},
                conclusion=_apply(goal, new_subst),
                span_citations=list(span_ids),
            )
            new_rest = [_apply(g, new_subst) for g in rest_goals]
            yield from self.solve(new_rest, new_subst, depth + 1, proof + [step])

        # ── Próbuj reguły ────────────────────────────────────────────────────
        for head, body, rule_id in self._rules:
            self._counter += 1
            suffix = str(self._counter)
            r_head, r_body = _rename_rule(head, body, suffix)

            new_subst = _unify(goal, r_head, dict(subst))
            if new_subst is None:
                continue

            self.used_rule_ids.add(rule_id)
            new_body = [_apply(b, new_subst) for b in r_body]
            new_goals = new_body + [_apply(g, new_subst) for g in rest_goals]

            step = ProofStep(
                step_id=step_id,
                rule_or_fact_id=rule_id,
                substitution={k: v for k, v in new_subst.items() if _is_var(k)},
                conclusion=_apply(r_head, new_subst),
                span_citations=[],
            )
            yield from self.solve(new_goals, new_subst, depth + 1, proof + [step])


# ──────────────────────────────────────────────────────────────────────────────
# Konwersja faktów z KnowledgeStore → atomy
# ──────────────────────────────────────────────────────────────────────────────

def _fact_to_atoms(fact: Fact) -> list[str]:
    """
    Konwertuje Fact na listę atomów do użycia przez reasoner.
    - h="add(3,4,7)", r="instance_of" → ["add(3,4,7)"]
    - h="add(3,4)",   r="eq", t="7"   → ["eq(add(3,4),7)"]
    """
    atoms: list[str] = []

    if _parse_atom(fact.h):
        if fact.r == "instance_of":
            atoms.append(fact.h)
        else:
            atoms.append(_make_atom(fact.r, [fact.h, fact.t]))
    else:
        atoms.append(_make_atom(fact.r, [fact.h, fact.t]))

    return atoms


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class ForwardChainer:
    """
    SLD-resolution reasoner podłączony do KnowledgeStore.
    Przy każdym zapytaniu ładuje asserted fakty i reguły.
    """

    def __init__(self, knowledge_store, max_depth: int = 30, timeout_ms: int = 5000) -> None:
        self._store = knowledge_store
        self._max_depth = max_depth
        self._timeout_ms = timeout_ms

    # -- Reasoner protocol -------------------------------------------------

    async def query(
        self,
        q: LogicQuery,
        snapshot: Optional[SnapshotRef] = None,
    ) -> ReasonerResult:
        start = time.monotonic()
        deadline = start + min(q.timeout_ms, self._timeout_ms) / 1000.0

        # Załaduj asserted fakty
        ground_atoms: list[tuple[str, str, list[str]]] = []
        async for fact in self._store.get_asserted_facts(snapshot):
            for atom_str in _fact_to_atoms(fact):
                ground_atoms.append((atom_str, fact.fact_id, list(fact.provenance)))

        # Załaduj asserted reguły
        rules_list: list[tuple[str, list[str], str]] = []
        async for rule in self._store.get_asserted_rules(snapshot):
            rules_list.append((rule.head, rule.body, rule.rule_id))

        # SLD resolution
        solver = _SLDSolver(
            ground_atoms,
            rules_list,
            max_depth=min(q.max_depth, self._max_depth),
            deadline=deadline,
        )

        answers: list[dict[str, str]] = []
        proof_steps: list[ProofStep] = []
        timed_out = False

        try:
            for subst, steps in solver.solve([q.goal], {}, 0, []):
                # Wyciągnij bindingi zmiennych z zapytania
                bindings: dict[str, str] = {}
                for var in q.variables:
                    # Obsłuż zarówno "Z" jak i "?Z"
                    for key in (var, f"?{var}"):
                        val = _walk(key, subst)
                        if val != key:
                            bindings[var] = _apply(val, subst)
                            break

                answers.append(bindings)
                if not proof_steps:
                    proof_steps = steps
                break  # MVP: pierwsze rozwiązanie

        except TimeoutError:
            timed_out = True

        duration_ms = (time.monotonic() - start) * 1000.0

        return ReasonerResult(
            goal=q.goal,
            success=len(answers) > 0,
            answers=answers,
            proof=proof_steps,
            used_fact_ids=list(solver.used_fact_ids),
            used_rule_ids=list(solver.used_rule_ids),
            depth_reached=solver.depth_reached,
            duration_ms=duration_ms,
        )
