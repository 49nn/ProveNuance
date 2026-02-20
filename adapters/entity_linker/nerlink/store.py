"""
NerLink — InMemoryStore: repozytorium encji i aliasów w pamięci.
Spełnia kontrakt wymagany przez EntityResolver (sekcja 8 spec — wariant in-memory).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StoredEntity:
    entity_id: str
    label: str
    canonical_name: str
    canonical_key: str


@dataclass
class StoredAlias:
    entity_id: str
    alias: str
    alias_key: str
    alias_type: str = "surface"
    weight: float = 1.0
    is_preferred: bool = False


class InMemoryStore:
    """
    Repozytorium encji i aliasów trzymane in-memory.
    Może być wymienione na PgStore implementujący ten sam interfejs.
    """

    def __init__(self) -> None:
        # entity_id → StoredEntity
        self._entities: dict[str, StoredEntity] = {}

        # (label, alias_key) → entity_id  — główny indeks aliasów
        self._alias_index: dict[tuple[str, str], str] = {}

        # entity_id → list[StoredAlias]
        self._aliases: dict[str, list[StoredAlias]] = {}

        # label → list[(alias_key, alias, entity_id)]  — indeks per-label dla fuzzy
        self._label_aliases: dict[str, list[tuple[str, str, str]]] = {}

    # ── lookup ────────────────────────────────────────────────────────────────

    def get_entity_id_by_alias(self, alias_key: str, label: str) -> Optional[str]:
        return self._alias_index.get((label, alias_key))

    def get_entity(self, entity_id: str) -> Optional[StoredEntity]:
        return self._entities.get(entity_id)

    def get_aliases_for_label(self, label: str) -> list[tuple[str, str, str]]:
        """Zwraca listę (alias_key, alias, entity_id) dla danego label.
        Używane przez RapidFuzzMatcher jako pula kandydatów.
        """
        return self._label_aliases.get(label, [])

    # ── write ─────────────────────────────────────────────────────────────────

    def save_entity(
        self,
        entity_id: str,
        label: str,
        canonical_name: str,
        canonical_key: str,
    ) -> None:
        if entity_id not in self._entities:
            self._entities[entity_id] = StoredEntity(
                entity_id=entity_id,
                label=label,
                canonical_name=canonical_name,
                canonical_key=canonical_key,
            )
            self._aliases.setdefault(entity_id, [])

    def save_alias(
        self,
        entity_id: str,
        alias: str,
        alias_key: str,
        label: str,
        alias_type: str = "surface",
        weight: float = 1.0,
        is_preferred: bool = False,
    ) -> None:
        ent = self._entities.get(entity_id)
        if ent is None:
            raise KeyError(f"Encja nie istnieje: {entity_id!r}")

        idx_key = (label, alias_key)
        if idx_key in self._alias_index:
            return  # już zarejestrowany — deduplicate

        self._alias_index[idx_key] = entity_id
        self._aliases.setdefault(entity_id, []).append(
            StoredAlias(
                entity_id=entity_id,
                alias=alias,
                alias_key=alias_key,
                alias_type=alias_type,
                weight=weight,
                is_preferred=is_preferred,
            )
        )
        self._label_aliases.setdefault(label, []).append((alias_key, alias, entity_id))
