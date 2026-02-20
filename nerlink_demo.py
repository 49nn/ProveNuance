"""
nerlink_demo.py — interaktywne demo NerLink entity linker.

Użycie:
  python nerlink_demo.py                  # demo z wbudowanym tekstem
  python nerlink_demo.py "własny tekst"   # tekst z argumentu
  python nerlink_demo.py - < plik.txt     # tekst ze stdin

Skrypt:
  1. Seeduje zestaw polskich encji (instytucje, osoby, akty prawne)
  2. Wyciąga kandydatów z tekstu prostą heurystyką (wielkie litery, skróty)
  3. Uruchamia NerLinkEntityLinker (normalize → simplemma → dict → fuzzy)
  4. Wypisuje tabelę wyników
"""
from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass

# Windows: wymuś UTF-8 na stdout/stderr
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from adapters.entity_linker.nerlink import NerLinkEntityLinker, NerLinkConfig
from adapters.entity_linker.nerlink.plugins.dictionary import SimpleMapDictionaryPlugin
from adapters.entity_linker.nerlink.plugins.flexion import SimplemmaFlexionPlugin
from adapters.entity_linker.nerlink.resolver import ResolverPolicy

# ── Seed encji ────────────────────────────────────────────────────────────────

_SEED_ENTITIES = [
    {
        "entity_id": "ent_org_knf",
        "label": "ORG",
        "canonical_name": "Komisja Nadzoru Finansowego",
        "aliases": [
            {"alias": "KNF",                              "alias_type": "acronym"},
            {"alias": "Komisji Nadzoru Finansowego",      "alias_type": "inflected"},
            {"alias": "Komisja",                          "alias_type": "surface"},
        ],
    },
    {
        "entity_id": "ent_org_nbp",
        "label": "ORG",
        "canonical_name": "Narodowy Bank Polski",
        "aliases": [
            {"alias": "NBP",                              "alias_type": "acronym"},
            {"alias": "Narodowego Banku Polskiego",       "alias_type": "inflected"},
            {"alias": "Banku Polskiego",                  "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_org_uokik",
        "label": "ORG",
        "canonical_name": "Urząd Ochrony Konkurencji i Konsumentów",
        "aliases": [
            {"alias": "UOKiK",                            "alias_type": "acronym"},
            {"alias": "Urzędu Ochrony Konkurencji",       "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_org_sejm",
        "label": "ORG",
        "canonical_name": "Sejm Rzeczypospolitej Polskiej",
        "aliases": [
            {"alias": "Sejm",                             "alias_type": "surface"},
            {"alias": "Sejmu",                            "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_law_kpa",
        "label": "LAW",
        "canonical_name": "Kodeks postępowania administracyjnego",
        "aliases": [
            {"alias": "k.p.a.",                           "alias_type": "acronym"},
            {"alias": "kpa",                              "alias_type": "acronym"},
            {"alias": "Kodeksu postępowania administracyjnego", "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_law_kc",
        "label": "LAW",
        "canonical_name": "Kodeks cywilny",
        "aliases": [
            {"alias": "k.c.",                             "alias_type": "acronym"},
            {"alias": "kc",                               "alias_type": "acronym"},
            {"alias": "Kodeksu cywilnego",                "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_person_kowalski",
        "label": "PERSON",
        "canonical_name": "Jan Kowalski",
        "aliases": [
            {"alias": "Kowalski",                         "alias_type": "surface"},
            {"alias": "Kowalskiego",                      "alias_type": "inflected"},
            {"alias": "Kowalskiemu",                      "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_person_nowak",
        "label": "PERSON",
        "canonical_name": "Anna Nowak",
        "aliases": [
            {"alias": "Nowak",                            "alias_type": "surface"},
            {"alias": "Nowak Anna",                       "alias_type": "surface"},
        ],
    },
]

# Słownik deterministyczny (skróty)
_DICT_ENTRIES = [
    {"label": "ORG",    "alias": "KNF",    "entity_id": "ent_org_knf"},
    {"label": "ORG",    "alias": "NBP",    "entity_id": "ent_org_nbp"},
    {"label": "ORG",    "alias": "UOKiK",  "entity_id": "ent_org_uokik"},
    {"label": "LAW",    "alias": "k.p.a.", "entity_id": "ent_law_kpa"},
    {"label": "LAW",    "alias": "kpa",    "entity_id": "ent_law_kpa"},
    {"label": "LAW",    "alias": "k.c.",   "entity_id": "ent_law_kc"},
    {"label": "LAW",    "alias": "kc",     "entity_id": "ent_law_kc"},
]

# ── Demo tekst ────────────────────────────────────────────────────────────────

_DEMO_TEXT = """
Komisja Nadzoru Finansowego wydała w ubiegłym tygodniu decyzję dotyczącą
restrukturyzacji trzech banków komercyjnych. Rzecznik KNF poinformował, że
sprawa zostanie przekazana do Urzędu Ochrony Konkurencji i Konsumentów.

Prezes Narodowego Banku Polskiego zabrał głos w debacie budżetowej przed
Sejmem. NBP opublikował raport wskazujący na ryzyko inflacyjne w II kwartale.

Pełnomocnik Jan Kowalski złożył wniosek na podstawie k.p.a., powołując się
na art. 61 Kodeksu postępowania administracyjnego. Sędzia Anna Nowak
zarządziła przerwę w rozprawie. Pełnomocnik Kowalskiego potwierdził
otrzymanie wezwania.

UOKiK wszczął postępowanie wyjaśniające wobec czterech operatorów
telekomunikacyjnych. Zgodnie z Kodeksem cywilnym, stronom przysługuje
prawo do odwołania w terminie 14 dni.
"""

# ── Ekstrakcja kandydatów heurystyką ──────────────────────────────────────────

# Skróty pisane WIELKIMI literami: KNF, NBP, UOKiK
_ABBREV_UPPER_RE = re.compile(
    r"\b[A-ZŁŚŹŻĄĘĆÓŃ]{2,}(?:[a-ząęćółśźżń]{1,3})?\b"
)
# Skróty z kropkami — duże LUB małe litery: K.N.F. / k.p.a. / k.c.
_ABBREV_DOTS_RE = re.compile(
    r"\b(?:[A-ZŁŚŹŻĄĘĆÓŃa-ząęćółśźżń][a-ząęćółśźżń]?\.){2,}"
)
# Ciągi słów z wielkiej litery (nazwy własne, instytucje, osoby)
_CAPITALIZED_RE = re.compile(
    r"\b[A-ZŁŚŹŻĄĘĆÓŃ][a-ząęćółśźżń]+"
    r"(?:\s+(?:i\s+)?[A-ZŁŚŹŻĄĘĆÓŃ][a-ząęćółśźżń]+){0,5}\b"
)


@dataclass
class Candidate:
    text: str
    start: int
    end: int
    label: str


def _guess_label(text: str) -> str:
    """Prosta heurystyka etykiety na potrzeby demo."""
    upper = text.upper()
    law_hints = {"KPA", "KC", "K.P.A.", "K.C.", "KODEKS"}
    if any(h in upper for h in law_hints):
        return "LAW"
    person_hints = {"KOWALSKI", "NOWAK", "JAN", "ANNA", "MAREK", "PIOTR"}
    words = set(text.upper().split())
    if words & person_hints:
        return "PERSON"
    return "ORG"


def extract_candidates(text: str) -> list[Candidate]:
    seen_spans: set[tuple[int, int]] = set()
    candidates: list[Candidate] = []

    def overlaps(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for s, e in seen_spans)

    def add(m: re.Match, label: str) -> None:
        start, end = m.start(), m.end()
        if not overlaps(start, end):
            seen_spans.add((start, end))
            candidates.append(Candidate(m.group().rstrip("."), start, end, label))

    # 1. Skróty z kropkami (małe i wielkie): k.p.a., K.N.F.
    for m in _ABBREV_DOTS_RE.finditer(text):
        add(m, _guess_label(m.group()))
    # 2. Skróty bez kropek (wielkie litery): KNF, NBP, UOKiK
    for m in _ABBREV_UPPER_RE.finditer(text):
        add(m, _guess_label(m.group()))
    # 3. Ciągi słów z wielkiej litery — najszersze dopasowanie
    for m in _CAPITALIZED_RE.finditer(text):
        if not overlaps(m.start(), m.end()):
            add(m, _guess_label(m.group()))

    candidates.sort(key=lambda c: c.start)
    return candidates


# ── Formatowanie wyjścia ──────────────────────────────────────────────────────

_STATUS_ICON = {"linked": "✓", "new": "+", "ambiguous": "?"}
_STATUS_COLOR = {"linked": "\033[32m", "new": "\033[33m", "ambiguous": "\033[35m"}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_result(
    candidate: Candidate,
    status: str,
    entity_id: str,
    canonical: str,
    confidence: float,
    method: str,
    color: bool,
) -> None:
    icon = _STATUS_ICON.get(status, " ")
    if color:
        col = _STATUS_COLOR.get(status, "")
        print(
            f"  {col}{icon} {_BOLD}{candidate.text:<35}{_RESET}"
            f"  [{candidate.label:<6}]"
            f"  {col}{status:<9}{_RESET}"
            f"  → {canonical:<35}"
            f"  {_DIM}{method:<12}  conf={confidence:.2f}  id={entity_id}{_RESET}"
        )
    else:
        print(
            f"  {icon} {candidate.text:<35}"
            f"  [{candidate.label:<6}]"
            f"  {status:<9}"
            f"  → {canonical:<35}"
            f"  {method:<12}  conf={confidence:.2f}  id={entity_id}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def build_linker() -> NerLinkEntityLinker:
    linker = NerLinkEntityLinker(
        config=NerLinkConfig(
            policy=ResolverPolicy(
                fuzzy_link_threshold=88.0,
                fuzzy_ambiguous_threshold=75.0,
            )
        ),
        flexion_plugin=SimplemmaFlexionPlugin(),
        dictionary_plugin=SimpleMapDictionaryPlugin(entries=_DICT_ENTRIES),
    )
    linker.seed_entities(_SEED_ENTITIES)
    return linker


def run(text: str) -> None:
    color = _supports_color()
    linker = build_linker()
    candidates = extract_candidates(text)

    if not candidates:
        print("Brak kandydatów w tekście.")
        return

    # Wewnętrzny resolver daje nam więcej szczegółów niż publiczne link()
    from adapters.entity_linker.nerlink.models import Mention
    import uuid

    print(f"\n{_BOLD}Tekst ({len(text)} znaków):{_RESET}" if color else f"\nTekst ({len(text)} znaków):")
    print("-" * 80)
    print(text.strip())
    print("-" * 80)
    print(f"\n{_BOLD}Wyniki linkowania ({len(candidates)} kandydatów):{_RESET}\n" if color
          else f"\nWyniki linkowania ({len(candidates)} kandydatów):\n")

    linked_count = new_count = ambiguous_count = 0

    for cand in candidates:
        mention = Mention(
            mention_id=str(uuid.uuid4()),
            start=cand.start,
            end=cand.end,
            text=cand.text,
            label=cand.label,
        )
        resolved = linker._resolver.resolve(mention)

        if resolved.status == "linked":
            linked_count += 1
        elif resolved.status == "new":
            new_count += 1
        else:
            ambiguous_count += 1

        # Dopisz alias jeśli potrzeba
        if resolved.alias_to_add and resolved.entity_id:
            linker._resolver.add_alias(
                resolved.entity_id, resolved.alias_to_add, cand.label
            )

        print_result(
            candidate=cand,
            status=resolved.status,
            entity_id=resolved.entity_id or "(brak)",
            canonical=resolved.canonical_name or cand.text,
            confidence=resolved.confidence,
            method=resolved.method,
            color=color,
        )

    print()
    if color:
        print(
            f"  Podsumowanie:  "
            f"{_STATUS_COLOR['linked']}✓ linked={linked_count}{_RESET}  "
            f"{_STATUS_COLOR['new']}+ new={new_count}{_RESET}  "
            f"{_STATUS_COLOR['ambiguous']}? ambiguous={ambiguous_count}{_RESET}"
        )
    else:
        print(f"  Podsumowanie:  linked={linked_count}  new={new_count}  ambiguous={ambiguous_count}")
    print()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        text = sys.stdin.read()
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = _DEMO_TEXT

    run(text)


if __name__ == "__main__":
    main()
