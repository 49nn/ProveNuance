"""
StanzaEntityLinker — adapter EntityLinker oparty na pipeline stanza NLP.

Implementuje port ports.entity_linker.EntityLinker używając:
  - stanza.Pipeline  — tokenizacja + POS + lemmatyzacja + drzewa zależności
  - InMemoryStore    — repozytorium encji (z nerlink, wymienialny)
  - RapidFuzzMatcher — fuzzy matching na lemach (jeśli rapidfuzz dostępny)

Pipeline rozwiązywania:
  1. Normalize + tokenize tekstu wzmianki przez stanza
  2. Wyciągnij lemy (normalizacja morfologiczna)
  3. Dict lookup po lemie i formie powierzchniowej
  4. Fuzzy match (opcjonalne, wymaga rapidfuzz)
  5. Utwórz nową encję jeśli brak dopasowania

Dodatkowe możliwości (nie wymagane przez port):
  - analyze(text)  → StanzaAnalysis z pełną analizą POS + depparse
  - seed_entities([...]) — bulk-load encji jak w NerLinkEntityLinker
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel

from contracts import DocSpan, EntityRef

from ..nerlink.match.fuzzy import RapidFuzzMatcher
from ..nerlink.normalize import NormalizationConfig, normalize_key
from ..nerlink.store import InMemoryStore, StoredAlias, StoredEntity
from .models import StanzaAnalysis
from .pipeline import StanzaPipeline


class StanzaConfig(BaseModel):
    """Konfiguracja adaptera StanzaEntityLinker."""

    lang: str = "pl"
    processors: str = "tokenize,pos,lemma,depparse,ner"
    use_gpu: bool = False
    verbose: bool = False
    fuzzy_enabled: bool = True
    fuzzy_link_threshold: float = 88.0
    fuzzy_ambiguous_threshold: float = 75.0
    topn_fuzzy_candidates: int = 5000
    norm_cfg: NormalizationConfig = NormalizationConfig()
    # Ścieżki do niestandardowego modelu NER (opcjonalne)
    ner_model_path: Optional[str] = None
    ner_charlm_forward_file: Optional[str] = None
    ner_charlm_backward_file: Optional[str] = None


class StanzaEntityLinker:
    """
    Adapter EntityLinker oparty na stanza NLP pipeline.

    Użycie minimalne:
        linker = StanzaEntityLinker()
        ref = linker.link("Komisja Nadzoru Finansowego", "ORG")

    Z analizą NLP:
        analysis = linker.analyze("Prezes NBP zabrał głos.")
        for sent in analysis.sentences:
            for token in sent.tokens:
                print(token.text, token.upos, token.lemma, token.deprel)

    Z pre-seeded encjami:
        linker.seed_entities([
            {"entity_id": "ent_org_knf", "label": "ORG",
             "canonical_name": "Komisja Nadzoru Finansowego",
             "aliases": [{"alias": "KNF", "alias_type": "acronym"}]},
        ])
    """

    def __init__(
        self,
        config: Optional[StanzaConfig] = None,
        store: Optional[InMemoryStore] = None,
    ) -> None:
        cfg = config or StanzaConfig()
        self._cfg = cfg
        self._store = store or InMemoryStore()
        self._pipeline = StanzaPipeline(
            lang=cfg.lang,
            processors=cfg.processors,
            use_gpu=cfg.use_gpu,
            verbose=cfg.verbose,
            ner_model_path=cfg.ner_model_path,
            ner_charlm_forward_file=cfg.ner_charlm_forward_file,
            ner_charlm_backward_file=cfg.ner_charlm_backward_file,
        )
        self._fuzzy: Optional[RapidFuzzMatcher] = None
        if cfg.fuzzy_enabled:
            try:
                self._fuzzy = RapidFuzzMatcher()
            except ImportError:
                pass  # graceful degradation

    # ── Publiczne API NLP ─────────────────────────────────────────────────────

    def analyze(self, text: str) -> StanzaAnalysis:
        """
        Uruchamia pełny pipeline NLP (tokenize + POS + lemma + depparse).

        Zwraca StanzaAnalysis z pełną analizą tokenów.
        """
        return self._pipeline.analyze(text)

    # ── EntityLinker protocol ─────────────────────────────────────────────────

    def link(
        self,
        name: str,
        entity_type: str,
        context: Optional[DocSpan] = None,
    ) -> EntityRef:
        """
        Rozwiązuje wzmiankę do EntityRef.

        Pipeline:
          1. Uruchom stanza na `name` — wyciągnij lemy
          2. Szukaj po lemie w store (exact match)
          3. Szukaj po formie powierzchniowej w store (exact match)
          4. Fuzzy match na aliasach etykiety entity_type
          5. Utwórz nową encję
        """
        lemma = self._lemmatize(name)
        norm_surface = normalize_key(name, self._cfg.norm_cfg)
        norm_lemma = normalize_key(lemma, self._cfg.norm_cfg)

        # 1. Exact match po lemie
        entity_id = self._store.get_entity_id_by_alias(norm_lemma, entity_type)
        if entity_id:
            ent = self._store.get_entity(entity_id)
            if ent:
                self._store.save_alias(
                    entity_id, name, norm_surface, entity_type,
                    alias_type="surface",
                )
                return EntityRef(
                    entity_id=entity_id,
                    canonical_name=ent.canonical_name,
                    entity_type=entity_type,
                )

        # 2. Exact match po formie powierzchniowej
        entity_id = self._store.get_entity_id_by_alias(norm_surface, entity_type)
        if entity_id:
            ent = self._store.get_entity(entity_id)
            if ent:
                return EntityRef(
                    entity_id=entity_id,
                    canonical_name=ent.canonical_name,
                    entity_type=entity_type,
                )

        # 3. Fuzzy matching
        if self._fuzzy:
            candidates = self._store.get_aliases_for_label(entity_type)
            if candidates:
                matches = self._fuzzy.match(
                    norm_lemma, candidates, limit=self._cfg.topn_fuzzy_candidates
                )
                if matches and matches[0].score >= self._cfg.fuzzy_link_threshold:
                    best = matches[0]
                    ent = self._store.get_entity(best.entity_id)
                    if ent:
                        self._store.save_alias(
                            best.entity_id, name, norm_surface, entity_type,
                            alias_type="surface",
                        )
                        return EntityRef(
                            entity_id=best.entity_id,
                            canonical_name=ent.canonical_name,
                            entity_type=entity_type,
                        )

        # 4. Nowa encja
        entity_id = f"ent_{entity_type.lower()}_{uuid.uuid4().hex[:10]}"
        canonical_key = norm_lemma or norm_surface
        self._store.save_entity(entity_id, entity_type, name, canonical_key)
        self._store.save_alias(
            entity_id, name, norm_surface, entity_type,
            alias_type="surface", is_preferred=True,
        )
        if norm_lemma and norm_lemma != norm_surface:
            self._store.save_alias(
                entity_id, lemma, norm_lemma, entity_type,
                alias_type="lemma",
            )
        return EntityRef(
            entity_id=entity_id,
            canonical_name=name,
            entity_type=entity_type,
        )

    def add_alias(self, entity_id: str, alias: str) -> None:
        """Dodaje alternatywną formę do istniejącej encji."""
        ent = self._store.get_entity(entity_id)
        if ent is None:
            raise KeyError(f"Encja nie istnieje: {entity_id!r}")
        alias_key = normalize_key(alias, self._cfg.norm_cfg)
        self._store.save_alias(entity_id, alias, alias_key, ent.label, alias_type="surface")

    def get_entity(self, entity_id: str) -> EntityRef:
        """Zwraca EntityRef po ID. Rzuca KeyError jeśli nie istnieje."""
        ent = self._store.get_entity(entity_id)
        if ent is None:
            raise KeyError(f"Encja nie istnieje: {entity_id!r}")
        return EntityRef(
            entity_id=ent.entity_id,
            canonical_name=ent.canonical_name,
            entity_type=ent.label,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def seed_entities(self, entities: list[dict]) -> None:
        """
        Wczytuje encje do store (format kompatybilny z NerLinkEntityLinker).

        Format:
        [
          {
            "entity_id": "ent_org_knf",
            "label": "ORG",
            "canonical_name": "Komisja Nadzoru Finansowego",
            "aliases": [
              {"alias": "KNF", "alias_type": "acronym"},
              "Komisji Nadzoru Finansowego"
            ]
          }
        ]
        """
        for e in entities:
            entity_id = e["entity_id"]
            label = e["label"]
            canonical = e["canonical_name"]
            canonical_key = normalize_key(canonical, self._cfg.norm_cfg)

            self._store.save_entity(entity_id, label, canonical, canonical_key)
            self._store.save_alias(
                entity_id, canonical, canonical_key, label,
                alias_type="surface", is_preferred=True,
            )
            for alias_entry in e.get("aliases", []):
                if isinstance(alias_entry, str):
                    alias_text, alias_type = alias_entry, "surface"
                else:
                    alias_text = alias_entry["alias"]
                    alias_type = alias_entry.get("alias_type", "surface")
                alias_key = normalize_key(alias_text, self._cfg.norm_cfg)
                self._store.save_alias(
                    entity_id, alias_text, alias_key, label,
                    alias_type=alias_type,
                )

    def _lemmatize(self, text: str) -> str:
        """Uruchamia stanza i zwraca lemy złączone spacją."""
        try:
            analysis = self._pipeline.analyze(text)
            tokens = analysis.all_tokens
            if not tokens:
                return text
            return " ".join(t.lemma for t in tokens)
        except Exception:
            return text

    @staticmethod
    def download_model(lang: str = "pl", verbose: bool = True) -> None:
        """Pobiera model stanza dla danego języka.

        Wywołaj raz przed pierwszym użyciem:
            StanzaEntityLinker.download_model("pl")
        """
        StanzaPipeline.download(lang=lang, verbose=verbose)


__all__ = ["StanzaEntityLinker", "StanzaConfig"]
