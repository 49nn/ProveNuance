"""
Port: EntityLinker
Odpowiedzialność: lematyzacja/kanonizacja nazw, przypisywanie ID encji.
"""
from typing import Optional, Protocol, runtime_checkable

from contracts import DocSpan, EntityRef


@runtime_checkable
class EntityLinker(Protocol):
    def link(
        self,
        name: str,
        entity_type: str,
        context: Optional[DocSpan] = None,
    ) -> EntityRef:
        """
        Resolves a surface name to a canonical EntityRef.
        Creates a new entity if unknown.
        context (DocSpan) can be used for disambiguation.
        """
        ...

    def add_alias(self, entity_id: str, alias: str) -> None:
        """Adds an alternative surface form for an existing entity."""
        ...

    def get_entity(self, entity_id: str) -> EntityRef:
        """Returns EntityRef by ID. Raises KeyError if not found."""
        ...
