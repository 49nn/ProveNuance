"""
NerLink — EntityResolver: orkiestruje pipeline dict → fuzzy → decyzja.

Pipeline per Mention (sekcja 6 spec):
  1. normalize_key(mention.text)
  2. flexion.analyze(...)
  3. Słownik (dict plugin + store lookup)
  4. Fuzzy (RapidFuzz) — jeśli dostępny
  5. Decyzja: linked / ambiguous / new
  6. Aktualizacje aliasów
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .idgen import IdGenerator, NerLinkIdGenerator
from .match.fuzzy import FuzzyMatch, RapidFuzzMatcher
from .models import DictMatch, FlexionAnalysis, Mention, ResolvedEntity, Span
from .normalize import NormalizationConfig, normalize_key
from .plugins.dictionary import DictionaryPlugin, NullDictionaryPlugin
from .plugins.flexion import FlexionPlugin, IdentityFlexionPlugin
from .store import InMemoryStore


class ResolverPolicy(BaseModel):
    """Konfigurowalne progi decyzyjne (sekcja 6.1 spec)."""

    fuzzy_link_threshold: float = 92.0       # score ≥ → linked
    fuzzy_ambiguous_threshold: float = 86.0  # score ≥ → ambiguous (jeśli < link_threshold)
    topn_fuzzy_candidates: int = 5000        # limit listy kandydatów per label

    # Embedding (przyszłość — nie zaimplementowane w MVP)
    embedding_link_threshold: float = 0.80
    embedding_ambiguous_delta: float = 0.03
    topk_embedding: int = 10


class EntityResolver:
    """
    Implementacja pipeline rozwiązywania wzmianek (sekcja 6 spec).
    Kolejność: dict → fuzzy → new.
    """

    def __init__(
        self,
        store: InMemoryStore,
        flexion_plugin: Optional[FlexionPlugin] = None,
        dictionary_plugin: Optional[DictionaryPlugin] = None,
        fuzzy_matcher: Optional[RapidFuzzMatcher] = None,
        policy: Optional[ResolverPolicy] = None,
        id_generator: Optional[IdGenerator] = None,
        norm_cfg: Optional[NormalizationConfig] = None,
    ) -> None:
        self._store = store
        self._flex: FlexionPlugin = flexion_plugin or IdentityFlexionPlugin()
        self._dict: DictionaryPlugin = dictionary_plugin or NullDictionaryPlugin()
        self._fuzzy: Optional[RapidFuzzMatcher] = fuzzy_matcher
        self._policy = policy or ResolverPolicy()
        self._idgen: IdGenerator = id_generator or NerLinkIdGenerator()
        self._norm_cfg = norm_cfg or NormalizationConfig()

    # ── public API ────────────────────────────────────────────────────────────

    def resolve(self, mention: Mention) -> ResolvedEntity:
        """Rozwiązuje pojedynczą wzmiankę — główny entry point."""
        span = Span(start=mention.start, end=mention.end, text=mention.text)

        # 1–2. Normalizacja i fleksja
        flex: FlexionAnalysis = self._flex.analyze(mention.text, mention.label)

        # 3. Słownik
        dict_match = self._lookup_dict(mention.text, flex, mention.label)
        if dict_match.matched and dict_match.entity_id:
            ent = self._store.get_entity(dict_match.entity_id)
            canonical = ent.canonical_name if ent else mention.text
            alias_to_add = self._compute_alias_to_add(
                dict_match.entity_id, mention.text, mention.label
            )
            return ResolvedEntity(
                mention_id=mention.mention_id,
                span=span,
                label=mention.label,
                status="linked",
                entity_id=dict_match.entity_id,
                canonical_name=canonical,
                confidence=dict_match.confidence,
                method="dict",
                alias_to_add=alias_to_add,
            )

        # 4. Fuzzy
        if self._fuzzy is not None:
            fuzzy_result = self._run_fuzzy(mention, flex, span)
            if fuzzy_result is not None:
                return fuzzy_result

        # 5. Nowa encja
        return self._create_new(mention, span)

    def create_entity(
        self,
        label: str,
        canonical_name: str,
        aliases: Optional[list[str]] = None,
    ) -> str:
        """Tworzy nową encję w store (idempotentne) i zwraca entity_id."""
        entity_id = self._idgen.new_id(label, canonical_name)
        if self._store.get_entity(entity_id) is None:
            canonical_key = normalize_key(canonical_name, self._norm_cfg)
            self._store.save_entity(entity_id, label, canonical_name, canonical_key)
            self._store.save_alias(
                entity_id, canonical_name, canonical_key, label,
                alias_type="surface", is_preferred=True,
            )
            for alias in aliases or []:
                alias_key = normalize_key(alias, self._norm_cfg)
                self._store.save_alias(entity_id, alias, alias_key, label)
        return entity_id

    def add_alias(
        self, entity_id: str, alias: str, label: str, alias_type: str = "surface"
    ) -> None:
        alias_key = normalize_key(alias, self._norm_cfg)
        self._store.save_alias(entity_id, alias, alias_key, label, alias_type=alias_type)

    # ── private ───────────────────────────────────────────────────────────────

    def _lookup_dict(
        self, text: str, flex: FlexionAnalysis, label: str
    ) -> DictMatch:
        """Próbuje dopasowania słownikowego dla mention.text i wszystkich flex.candidates."""
        for candidate in list(dict.fromkeys([text] + flex.candidates)):
            # a) zewnętrzny plugin słownika
            result = self._dict.lookup(candidate, label)
            if result.matched:
                return result
            # b) store jako słownik (alias_key lookup)
            alias_key = normalize_key(candidate, self._norm_cfg)
            entity_id = self._store.get_entity_id_by_alias(alias_key, label)
            if entity_id:
                return DictMatch(matched=True, entity_id=entity_id, matched_alias=candidate)

        return DictMatch(matched=False)

    def _run_fuzzy(
        self, mention: Mention, flex: FlexionAnalysis, span: Span
    ) -> Optional[ResolvedEntity]:
        label = mention.label
        candidates = self._store.get_aliases_for_label(label)
        if not candidates:
            return None

        if len(candidates) > self._policy.topn_fuzzy_candidates:
            candidates = candidates[: self._policy.topn_fuzzy_candidates]

        # Zbierz wszystkie dopasowania dla mention.text i flex.candidates
        queries = list(dict.fromkeys([mention.text] + flex.candidates))
        all_matches: list[FuzzyMatch] = []
        for query in queries:
            all_matches.extend(self._fuzzy.match(query, candidates, limit=2))

        if not all_matches:
            return None

        all_matches.sort(key=lambda m: m.score, reverse=True)
        top = all_matches[0]
        confidence = top.score / 100.0
        policy = self._policy

        if top.score >= policy.fuzzy_link_threshold:
            ent = self._store.get_entity(top.entity_id)
            canonical = ent.canonical_name if ent else mention.text
            alias_to_add = self._compute_alias_to_add(top.entity_id, mention.text, label)
            return ResolvedEntity(
                mention_id=mention.mention_id,
                span=span,
                label=label,
                status="linked",
                entity_id=top.entity_id,
                canonical_name=canonical,
                confidence=confidence,
                method="fuzzy",
                alias_to_add=alias_to_add,
            )

        if top.score >= policy.fuzzy_ambiguous_threshold:
            ent = self._store.get_entity(top.entity_id)
            canonical = ent.canonical_name if ent else mention.text
            return ResolvedEntity(
                mention_id=mention.mention_id,
                span=span,
                label=label,
                status="ambiguous",
                entity_id=top.entity_id,
                canonical_name=canonical,
                confidence=confidence,
                method="fuzzy",
                alias_to_add=None,
            )

        return None

    def _create_new(self, mention: Mention, span: Span) -> ResolvedEntity:
        entity_id = self.create_entity(mention.label, mention.text)
        return ResolvedEntity(
            mention_id=mention.mention_id,
            span=span,
            label=mention.label,
            status="new",
            entity_id=entity_id,
            canonical_name=mention.text,
            confidence=0.0,
            method="new",
            alias_to_add=None,
        )

    def _compute_alias_to_add(
        self, entity_id: str, mention_text: str, label: str
    ) -> Optional[str]:
        """Zwraca mention_text jeśli nie jest jeszcze zarejestrowany jako alias encji."""
        alias_key = normalize_key(mention_text, self._norm_cfg)
        existing = self._store.get_entity_id_by_alias(alias_key, label)
        return mention_text if existing is None else None
