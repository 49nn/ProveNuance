"""
NerLink — plugin fleksji (sekcja 5.1 spec).

Kontrakt: FlexionPlugin Protocol + IdentityFlexionPlugin (referencyjny).
Opcjonalnie: MorfeuszFlexionPlugin (wymaga morfeusz2).
"""
from __future__ import annotations

from typing import Protocol

from ..models import FlexionAnalysis


class FlexionPlugin(Protocol):
    name: str

    def analyze(self, text: str, label: str, lang: str = "pl") -> FlexionAnalysis: ...


class IdentityFlexionPlugin:
    """
    Trywialny plugin: candidates = [text], brak cech fleksyjnych.
    Używany jako domyślny fallback gdy morfeusz2 niedostępny.
    """

    name: str = "identity"

    def analyze(self, text: str, label: str, lang: str = "pl") -> FlexionAnalysis:
        return FlexionAnalysis(surface=text, candidates=[text])


class SimplemmaFlexionPlugin:
    """
    Plugin fleksji oparty na simplemma (lekka alternatywa dla Morfeusz2).
    Wymaga: pip install simplemma

    Zwraca lemat jako dodatkowy kandydat obok surface form.
    Lemat jest zwracany lowercase — normalize_key() i tak robi casefold,
    więc nie ma to znaczenia dla lookup.
    """

    name: str = "simplemma"

    def __init__(self, lang: str = "pl") -> None:
        try:
            import simplemma as _simplemma  # type: ignore[import]

            self._lemmatize = _simplemma.lemmatize
        except ImportError as exc:
            raise ImportError(
                "simplemma jest wymagany przez SimplemmaFlexionPlugin. "
                "Zainstaluj: pip install simplemma"
            ) from exc
        self._lang = lang

    def analyze(self, text: str, label: str, lang: str = "pl") -> FlexionAnalysis:
        effective_lang = lang if lang != "pl" else self._lang
        lemma = self._lemmatize(text, lang=effective_lang)
        # Deduplicate: jeśli lemat == surface (już znormalizowany), nie dubluj
        candidates = list(dict.fromkeys([text, lemma]))
        return FlexionAnalysis(surface=text, candidates=candidates)


class MorfeuszFlexionPlugin:
    """
    Plugin fleksji oparty na Morfeusz2 (opcjonalny).
    Wymaga: pip install morfeusz2
    Zwraca lematy jako candidates.
    """

    name: str = "morfeusz2"

    def __init__(self) -> None:
        try:
            import morfeusz2  # type: ignore[import]

            self._morf = morfeusz2.Morfeusz()
        except ImportError as exc:
            raise ImportError(
                "morfeusz2 jest wymagany przez MorfeuszFlexionPlugin. "
                "Zainstaluj: pip install morfeusz2"
            ) from exc

    def analyze(self, text: str, label: str, lang: str = "pl") -> FlexionAnalysis:
        results = self._morf.analyse(text)
        # results: list of (start, end, (orth, lemma, tag, ...))
        lemmas: list[str] = []
        for _start, _end, interp in results:
            lemma = interp[1].split(":")[0]  # strip inflection suffix if present
            if lemma and lemma not in lemmas:
                lemmas.append(lemma)

        candidates = list(dict.fromkeys([text] + lemmas))  # text first, deduped
        return FlexionAnalysis(surface=text, candidates=candidates)
