"""
NerLink — plugin słownika (sekcja 5.2 spec).

Kontrakt: DictionaryPlugin Protocol + implementacje:
  - NullDictionaryPlugin   : zawsze matched=False
  - SimpleMapDictionaryPlugin : mapa normalized_alias → entity_id
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Protocol

from ..models import DictMatch
from ..normalize import NormalizationConfig, normalize_key


class DictionaryPlugin(Protocol):
    name: str

    def lookup(self, text: str, label: str, lang: str = "pl") -> DictMatch: ...


class NullDictionaryPlugin:
    """Zawsze zwraca matched=False. Używany gdy brak zewnętrznego słownika."""

    name: str = "null"

    def lookup(self, text: str, label: str, lang: str = "pl") -> DictMatch:
        return DictMatch(matched=False)


class SimpleMapDictionaryPlugin:
    """
    Słownik oparty na mapie normalized_alias → entity_id.
    Może być zasilany z:
      - listy entries podanej wprost (entries=[...])
      - pliku dictionary_seed.json (seed_path=Path(...))

    Format seed JSON (sekcja 10.3 spec):
    {
      "version": "...",
      "entries": [
        {"label": "ORG", "alias": "KNF",   "entity_id": "ent_org_..."},
        {"label": "LAW", "alias": "k.p.a.", "entity_id": "ent_law_..."}
      ]
    }
    """

    name: str = "simple_map"

    def __init__(
        self,
        entries: Optional[list[dict[str, Any]]] = None,
        seed_path: Optional[Path] = None,
        norm_cfg: Optional[NormalizationConfig] = None,
    ) -> None:
        self._cfg = norm_cfg or NormalizationConfig()
        # (label, alias_key) → entity_id
        self._index: dict[tuple[str, str], str] = {}

        if seed_path is not None:
            self._load_seed(seed_path)
        for entry in entries or []:
            self._register(entry["label"], entry["alias"], entry["entity_id"])

    def _load_seed(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data.get("entries", []):
            self._register(entry["label"], entry["alias"], entry["entity_id"])

    def _register(self, label: str, alias: str, entity_id: str) -> None:
        key = normalize_key(alias, self._cfg)
        self._index[(label, key)] = entity_id

    def add_entry(self, label: str, alias: str, entity_id: str) -> None:
        self._register(label, alias, entity_id)

    def lookup(self, text: str, label: str, lang: str = "pl") -> DictMatch:
        key = normalize_key(text, self._cfg)
        entity_id = self._index.get((label, key))
        if entity_id is None:
            return DictMatch(matched=False)
        return DictMatch(matched=True, entity_id=entity_id, matched_alias=text)
