"""
Adapter: DictEntityLinker
Implementuje port EntityLinker za pomocą słownika in-memory.

Wbudowane encje:
  - operacje arytmetyczne: add, sub, mul, div
  - cyfry 0–20
  - koncepty: commutative, associative, zero_neutral, distributive
"""
from __future__ import annotations

import uuid

from contracts import DocSpan, EntityRef


def _new_id() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────────
# Wbudowane encje startowe
# ──────────────────────────────────────────────────────────────────

_BUILTIN_ENTITIES: list[tuple[str, str, list[str]]] = [
    # (entity_id, canonical_name, aliases)
    ("op:add", "add",           ["addition", "plus", "dodawanie", "dodać"]),
    ("op:sub", "sub",           ["subtraction", "minus", "odejmowanie", "odjąć"]),
    ("op:mul", "mul",           ["multiplication", "times", "mnożenie", "razy"]),
    ("op:div", "div",           ["division", "divided by", "dzielenie", "podzielić"]),
    ("prop:commutative",  "commutative",  ["przemienne", "commutative"]),
    ("prop:associative",  "associative",  ["łączne", "associative"]),
    ("prop:zero_neutral", "zero_neutral", ["zero neutral", "identity element"]),
    ("prop:distributive", "distributive", ["rozdzielne", "distributive"]),
]

# Cyfry 0–20
for _n in range(21):
    _BUILTIN_ENTITIES.append((f"num:{_n}", str(_n), [str(_n)]))


class DictEntityLinker:
    """
    Prosty EntityLinker oparty na słowniku.
    Thread-safe dla jednowątkowego użycia (brak locka – MVP).
    """

    def __init__(self) -> None:
        # id → EntityRef
        self._by_id: dict[str, EntityRef] = {}
        # alias (lowercase) → entity_id
        self._alias_index: dict[str, str] = {}

        self._load_builtins()

    # -- EntityLinker protocol ---------------------------------

    def link(
        self,
        name: str,
        entity_type: str,
        context: DocSpan | None = None,
    ) -> EntityRef:
        """
        Zwraca istniejącą encję pasującą do `name` lub tworzy nową.
        `context` (DocSpan) jest ignorowany w tej implementacji MVP.
        """
        key = name.strip().lower()
        if key in self._alias_index:
            return self._by_id[self._alias_index[key]]

        # Nowa encja
        entity_id = _new_id()
        ref = EntityRef(
            entity_id=entity_id,
            canonical_name=name.strip(),
            entity_type=entity_type,
        )
        self._by_id[entity_id] = ref
        self._alias_index[key] = entity_id
        return ref

    def add_alias(self, entity_id: str, alias: str) -> None:
        """Dodaje alternatywną formę powierzchniową do istniejącej encji."""
        if entity_id not in self._by_id:
            raise KeyError(f"Nieznane entity_id: {entity_id!r}")
        self._alias_index[alias.strip().lower()] = entity_id

    def get_entity(self, entity_id: str) -> EntityRef:
        """Zwraca EntityRef po ID. Rzuca KeyError jeśli nie istnieje."""
        try:
            return self._by_id[entity_id]
        except KeyError:
            raise KeyError(f"Encja nie istnieje: {entity_id!r}")

    # -- Prywatne ------------------------------------------

    def _load_builtins(self) -> None:
        for entity_id, canonical, aliases in _BUILTIN_ENTITIES:
            entity_type = (
                "operation" if entity_id.startswith("op:") else
                "concept"   if entity_id.startswith("prop:") else
                "number"
            )
            ref = EntityRef(
                entity_id=entity_id,
                canonical_name=canonical,
                entity_type=entity_type,
            )
            self._by_id[entity_id] = ref
            # Rejestruj canonical i wszystkie aliasy
            for alias in [canonical, *aliases]:
                self._alias_index[alias.strip().lower()] = entity_id
