"""
NerLink — adapter EntityLinker oparty na pipeline NER Linker (PL).

Implementuje port ports.entity_linker.EntityLinker używając:
  - normalize_key        — normalizacja tekstu
  - FlexionPlugin        — generowanie form fleksyjnych (domyślnie: identity)
  - DictionaryPlugin     — deterministyczny lookup skrótów/aliasów
  - RapidFuzzMatcher     — fuzzy matching na aliasach (jeśli rapidfuzz dostępny)
  - InMemoryStore        — repozytorium encji in-memory (wymienialny na PgStore)

Konfiguracja przez NerLinkConfig.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel

from contracts import DocSpan, EntityRef

from .idgen import NerLinkIdGenerator
from .match.fuzzy import RapidFuzzMatcher
from .models import Mention
from .normalize import NormalizationConfig
from .plugins.dictionary import DictionaryPlugin, NullDictionaryPlugin
from .plugins.flexion import FlexionPlugin, IdentityFlexionPlugin
from .resolver import EntityResolver, ResolverPolicy
from .store import InMemoryStore


class NerLinkConfig(BaseModel):
    """Konfiguracja adaptera NerLinkEntityLinker."""

    fuzzy_enabled: bool = True
    policy: ResolverPolicy = ResolverPolicy()
    norm_cfg: NormalizationConfig = NormalizationConfig()
    id_salt: str = ""


class NerLinkEntityLinker:
    """
    Adapter NerLink → EntityLinker.

    Implementuje port EntityLinker za pomocą pipeline:
      normalize → flexion → dict-lookup → fuzzy → decyzja → EntityRef

    Użycie minimalne:
        linker = NerLinkEntityLinker()
        ref = linker.link("KNF", "ORG")

    Z zewnętrznym słownikiem:
        from adapters.entity_linker.nerlink.plugins.dictionary import SimpleMapDictionaryPlugin
        dict_plugin = SimpleMapDictionaryPlugin(seed_path=Path("dictionary_seed.json"))
        linker = NerLinkEntityLinker(dictionary_plugin=dict_plugin)

    Z pre-seeded encjami:
        linker.seed_entities([
            {"entity_id": "ent_org_3f21a9b7c1", "label": "ORG",
             "canonical_name": "Komisja Nadzoru Finansowego",
             "aliases": [{"alias": "KNF", "alias_type": "acronym"}]},
        ])
    """

    def __init__(
        self,
        config: Optional[NerLinkConfig] = None,
        store: Optional[InMemoryStore] = None,
        flexion_plugin: Optional[FlexionPlugin] = None,
        dictionary_plugin: Optional[DictionaryPlugin] = None,
    ) -> None:
        cfg = config or NerLinkConfig()
        self._store = store or InMemoryStore()
        self._norm_cfg = cfg.norm_cfg

        fuzzy: Optional[RapidFuzzMatcher] = None
        if cfg.fuzzy_enabled:
            try:
                fuzzy = RapidFuzzMatcher()
            except ImportError:
                pass  # graceful degradation: tylko dict + store lookup

        self._resolver = EntityResolver(
            store=self._store,
            flexion_plugin=flexion_plugin or IdentityFlexionPlugin(),
            dictionary_plugin=dictionary_plugin or NullDictionaryPlugin(),
            fuzzy_matcher=fuzzy,
            policy=cfg.policy,
            id_generator=NerLinkIdGenerator(salt=cfg.id_salt),
            norm_cfg=cfg.norm_cfg,
        )

    # ── EntityLinker protocol ─────────────────────────────────────────────────

    def link(
        self,
        name: str,
        entity_type: str,
        context: Optional[DocSpan] = None,
    ) -> EntityRef:
        """
        Rozwiązuje wzmiankę do EntityRef.

        `entity_type` jest używany jako `label` w NerLink pipeline
        (np. PERSON, ORG, LAW, TERM).
        `context` (DocSpan) jest ignorowany w tej implementacji MVP.
        """
        mention = Mention(
            mention_id=str(uuid.uuid4()),
            start=0,
            end=len(name),
            text=name,
            label=entity_type,
        )
        resolved = self._resolver.resolve(mention)

        # Jeśli jest alias do dopisania — zarejestruj go w store
        if resolved.alias_to_add and resolved.entity_id:
            self._resolver.add_alias(
                resolved.entity_id,
                resolved.alias_to_add,
                entity_type,
                alias_type="surface",
            )

        return EntityRef(
            entity_id=resolved.entity_id or "",
            canonical_name=resolved.canonical_name or name,
            entity_type=entity_type,
        )

    def add_alias(self, entity_id: str, alias: str) -> None:
        """Dodaje alternatywną formę powierzchniową do istniejącej encji."""
        ent = self._store.get_entity(entity_id)
        if ent is None:
            raise KeyError(f"Encja nie istnieje: {entity_id!r}")
        self._resolver.add_alias(entity_id, alias, ent.label)

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

    def seed_entities(
        self,
        entities: list[dict],
    ) -> None:
        """Wczytuje encje do store (np. z entities.jsonl / aliases.jsonl).

        Format wejścia (kompatybilny ze spec sekcja 10.1–10.2):
        [
          {
            "entity_id": "ent_org_3f21a9b7c1",
            "label": "ORG",
            "canonical_name": "Komisja Nadzoru Finansowego",
            "aliases": [
              {"alias": "KNF", "alias_type": "acronym"},
              "Komisji Nadzoru Finansowego"   # str też akceptowany
            ]
          }
        ]
        """
        from .normalize import normalize_key

        for e in entities:
            entity_id = e["entity_id"]
            label = e["label"]
            canonical = e["canonical_name"]
            canonical_key = normalize_key(canonical, self._norm_cfg)

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
                alias_key = normalize_key(alias_text, self._norm_cfg)
                self._store.save_alias(
                    entity_id, alias_text, alias_key, label, alias_type=alias_type
                )
