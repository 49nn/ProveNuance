"""
adapters/rule_bootstrap/base.py — Wspólna baza dla adapterów RuleBootstrap.

Dostarcza:
- _parse_atom: parser płaskich atomów Prolog-owych (np. "add(2,3,5)")
- RuleBootstrapBase: generyczny pipeline bootstrap() reużywalny przez dowolny miner
"""
from __future__ import annotations

import re
from typing import Optional

from contracts import Fact, FactStatus, Rule
from ports.knowledge_store import KnowledgeStore
from ports.rule_bootstrap import (
    BootstrapConfig,
    BootstrapSummary,
    CandidateRule,
)

_ATOM_RE = re.compile(r"^[A-Za-z_]\w*\([^)]*\)$")


def _parse_atom(atom: str) -> tuple[str, list[str]] | None:
    """
    Parsuje płaski atom Prolog-owy, np. "add(2,3,5)" → ("add", ["2","3","5"]).
    Zwraca None dla nie-atomów.
    """
    m = _ATOM_RE.match(atom.strip())
    if not m:
        return None
    functor = atom[: atom.find("(")].strip()
    args_raw = atom[atom.find("(") + 1 : atom.rfind(")")].strip()
    args = [a.strip() for a in args_raw.split(",")] if args_raw else []
    return functor, args


class RuleBootstrapBase:
    """
    Bazowa klasa dla adapterów implementujących port RuleBootstrap.
    Dostarcza generyczny bootstrap() identyczny dla każdego minera.

    Podklasa musi:
    - ustawić self.config: BootstrapConfig
    - zaimplementować mine_candidates(facts) -> tuple[list[CandidateRule], int, int, int]
    """

    config: BootstrapConfig

    def mine_candidates(
        self, facts: list[Fact]
    ) -> tuple[list[CandidateRule], int, int, int]:
        raise NotImplementedError

    async def bootstrap(self, store: KnowledgeStore) -> BootstrapSummary:
        """
        Generyczny pipeline:
          1. Pobierz wszystkie asserted facts ze store
          2. Wywołaj mine_candidates()
          3. Upsert/promote reguły wg self.config
        """
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
