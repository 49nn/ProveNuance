"""
NerLink — generowanie entity_id.
Strategia: ent_{label}_{sha1[:10]} — deterministyczny i powtarzalny.
"""
from __future__ import annotations

import hashlib
from typing import Protocol


def _sha1_prefix(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()[:n]


class IdGenerator(Protocol):
    def new_id(self, label: str, canonical_name: str) -> str: ...


class NerLinkIdGenerator:
    """Generuje deterministyczne entity_id: ent_{label}_{sha1(normalize_key)[:10]}."""

    def __init__(self, salt: str = "") -> None:
        self._salt = salt

    def new_id(self, label: str, canonical_name: str) -> str:
        from .normalize import normalize_key  # lokalne żeby uniknąć circular import

        key = normalize_key(canonical_name) + self._salt
        return f"ent_{label.lower()}_{_sha1_prefix(key)}"
