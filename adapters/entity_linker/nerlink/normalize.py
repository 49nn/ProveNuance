"""
NerLink — normalizacja kluczy tekstowych.
"""
from __future__ import annotations

import re

from pydantic import BaseModel


class NormalizationConfig(BaseModel):
    strip_acronym_dots: bool = True
    normalize_dashes: bool = True


_DEFAULT_CFG = NormalizationConfig()


def normalize_key(s: str, cfg: NormalizationConfig = _DEFAULT_CFG) -> str:
    """Zwraca znormalizowany klucz dla tekstu powierzchniowego.

    Zachowanie:
    - strip() + casefold()
    - (opcja) usuwanie kropek z akronimów: K.N.F. → knf
    - (opcja) ujednolicenie myślników Unicode → ASCII hyphen
    - wiele whitespace → pojedyncza spacja
    """
    s = s.strip().casefold()
    if cfg.strip_acronym_dots:
        # "K.N.F." → casefold → "k.n.f." → "knf"
        s = re.sub(r"(?<=[a-z])\.(?=[a-z])", "", s)  # dots between letters
        s = re.sub(r"(?<=[a-z])\.$", "", s)           # trailing dot of acronym
    if cfg.normalize_dashes:
        # En-dash, em-dash, minus-sign, etc. → ASCII hyphen
        s = re.sub(r"[\u2010-\u2015\u2212\u2011]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s
