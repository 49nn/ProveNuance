"""
NerLink — dopasowanie rozmyte (RapidFuzz).
Wymaga: pip install rapidfuzz
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    from rapidfuzz import fuzz, process as rf_process  # type: ignore[import]

    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


@dataclass
class FuzzyMatch:
    entity_id: str
    matched_alias: str
    score: float  # 0..100


class RapidFuzzMatcher:
    """
    Fuzzy matching na liście (alias_key, alias, entity_id) za pomocą RapidFuzz.
    Scorer domyślny: WRatio — dobry kompromis dla krótkich nazw własnych.
    """

    def __init__(self, scorer: str = "WRatio", score_cutoff: float = 0.0) -> None:
        if not _HAS_RAPIDFUZZ:
            raise ImportError(
                "rapidfuzz jest wymagany przez RapidFuzzMatcher. "
                "Zainstaluj: pip install rapidfuzz"
            )
        self._scorer = getattr(fuzz, scorer)
        self._score_cutoff = score_cutoff

    def match(
        self,
        query: str,
        candidates: list[tuple[str, str, str]],  # (alias_key, alias, entity_id)
        limit: int = 5,
    ) -> list[FuzzyMatch]:
        """Zwraca listę najlepszych dopasowań posortowanych malejąco po score."""
        if not candidates:
            return []

        choices = [alias for (_key, alias, _eid) in candidates]
        results = rf_process.extract(
            query,
            choices,
            scorer=self._scorer,
            limit=limit,
            score_cutoff=self._score_cutoff,
        )
        # results: list of (matched_string, score, index)
        return [
            FuzzyMatch(
                entity_id=candidates[idx][2],
                matched_alias=matched_alias,
                score=score,
            )
            for matched_alias, score, idx in results
        ]
