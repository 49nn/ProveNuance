"""
stanza_demo.py — demo adaptera StanzaEntityLinker.

Pokazuje pełny pipeline NLP (tokenize + POS + lemma + depparse + NER) oparty
na stanza, plus linkowanie encji.

Użycie:
  python stanza_demo.py                       # demo z wbudowanym tekstem (PL prawny)
  python stanza_demo.py "własny tekst"        # tekst z argumentu — kandydaci z NER
  python stanza_demo.py - < plik.txt          # tekst ze stdin — kandydaci z NER
  python stanza_demo.py --download            # tylko pobierz model i wyjdź
  python stanza_demo.py --math-ner "tekst"   # użyj wytrenowanego modelu Math NER
  python stanza_demo.py --math-ner            # demo z wbudowanym tekstem matematycznym

Pierwsze uruchomienie automatycznie pobiera model językowy stanza dla języka
polskiego (~500 MB). Kolejne uruchomienia korzystają z cache.
"""
from __future__ import annotations

import io
import os
import sys

# Windows: wymuś UTF-8 na stdout/stderr
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from adapters.entity_linker.stanza_linker import StanzaEntityLinker, StanzaConfig
from adapters.entity_linker.stanza_linker.models import NerEntity, StanzaAnalysis

# ── ANSI kolory ───────────────────────────────────────────────────────────────

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_MAGENTA = "\033[35m"
_BLUE    = "\033[34m"
_RED     = "\033[31m"
_WHITE   = "\033[37m"

_POS_COLOR = {
    "NOUN":  _GREEN,
    "PROPN": _CYAN,
    "VERB":  _YELLOW,
    "ADJ":   _BLUE,
    "ADV":   _MAGENTA,
    "NUM":   _RED,
    "PUNCT": _DIM,
    "ADP":   _DIM,
    "CONJ":  _DIM,
    "CCONJ": _DIM,
    "SCONJ": _DIM,
    "DET":   _DIM,
    "AUX":   _DIM,
    "X":     _DIM,
}

# Kolory tagów NER (BIO prefix stripped)
_NER_COLOR = {
    # standardowe stanza PL
    "PER":        _CYAN,
    "ORG":        _GREEN,
    "LOC":        _YELLOW,
    "MISC":       _MAGENTA,
    "persName":   _CYAN,
    "orgName":    _GREEN,
    "placeName":  _YELLOW,
    # math NER labels
    "THEOREM":    _CYAN,
    "LEMMA":      _BLUE,
    "DEF":        _GREEN,
    "STRUCTURE":  _MAGENTA,
    "FUNCTION":   _YELLOW,
    "CONSTANT":   _RED,
    "AUTHOR":     _CYAN,
    "REF":        _WHITE,
    "CONJECTURE": _MAGENTA,
    "AXIOM":      _BLUE,
    "PROPOSITION":_GREEN,
    "COROLLARY":  _CYAN,
    "SET":        _YELLOW,
    "SEQUENCE":   _RED,
}


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _col(text: str, color: str, use_color: bool) -> str:
    if use_color and color:
        return f"{color}{text}{_RESET}"
    return text


def _bold(text: str, use_color: bool) -> str:
    return _col(text, _BOLD, use_color)


def _ner_type(raw_label: str) -> str:
    """Zwraca typ NER bez prefiksu BIO i bez 'B-'/'I-'."""
    if raw_label.startswith(("B-", "I-", "E-", "S-")):
        return raw_label[2:]
    return raw_label


def _ner_to_entity_type(ner_label: str) -> str:
    """Mapuje etykietę NER stanza → nasz entity_type."""
    t = _ner_type(ner_label).upper()
    if t in ("PER", "PERSON", "PERSNAME", "NAM_LIV_PERSON", "NAM_LIV", "AUTHOR"):
        return "PERSON"
    if t in ("ORG", "ORGNAME", "NAM_ORG", "ORGANIZATION"):
        return "ORG"
    if t in ("LOC", "LOCATION", "PLACENAME", "NAM_LOC", "GPE"):
        return "LOC"
    # math NER labels — przekaż bez zmian jako entity_type
    if t in (
        "THEOREM", "LEMMA", "DEF", "STRUCTURE", "FUNCTION", "CONSTANT",
        "REF", "CONJECTURE", "AXIOM", "PROPOSITION", "COROLLARY", "SET", "SEQUENCE",
    ):
        return t
    return "MISC"


# ── Seed encji dla wbudowanego demo tekstu ────────────────────────────────────

_SEED_ENTITIES = [
    {
        "entity_id": "ent_org_knf",
        "label": "ORG",
        "canonical_name": "Komisja Nadzoru Finansowego",
        "aliases": [
            {"alias": "KNF",                              "alias_type": "acronym"},
            {"alias": "Komisji Nadzoru Finansowego",      "alias_type": "inflected"},
        ],
    },
    {
        "entity_id": "ent_org_nbp",
        "label": "ORG",
        "canonical_name": "Narodowy Bank Polski",
        "aliases": [
            {"alias": "NBP",                              "alias_type": "acronym"},
            {"alias": "Narodowego Banku Polskiego",       "alias_type": "inflected"},
            {"alias": "bank polski",                      "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_org_uokik",
        "label": "ORG",
        "canonical_name": "Urząd Ochrony Konkurencji i Konsumentów",
        "aliases": [
            {"alias": "UOKiK",                            "alias_type": "acronym"},
            {"alias": "urząd ochrony konkurencja konsument", "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_org_sejm",
        "label": "ORG",
        "canonical_name": "Sejm Rzeczypospolitej Polskiej",
        "aliases": [
            {"alias": "Sejm",                             "alias_type": "surface"},
            {"alias": "sejm",                             "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_law_kpa",
        "label": "LAW",
        "canonical_name": "Kodeks postępowania administracyjnego",
        "aliases": [
            {"alias": "k.p.a.",                           "alias_type": "acronym"},
            {"alias": "kpa",                              "alias_type": "acronym"},
            {"alias": "kodeks postępowanie administracyjny", "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_law_kc",
        "label": "LAW",
        "canonical_name": "Kodeks cywilny",
        "aliases": [
            {"alias": "k.c.",                             "alias_type": "acronym"},
            {"alias": "kc",                               "alias_type": "acronym"},
            {"alias": "kodeks cywilny",                   "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_person_kowalski",
        "label": "PERSON",
        "canonical_name": "Jan Kowalski",
        "aliases": [
            {"alias": "Kowalski",                         "alias_type": "surface"},
            {"alias": "jan kowalski",                     "alias_type": "lemma"},
        ],
    },
    {
        "entity_id": "ent_person_nowak",
        "label": "PERSON",
        "canonical_name": "Anna Nowak",
        "aliases": [
            {"alias": "Nowak",                            "alias_type": "surface"},
            {"alias": "anna nowak",                       "alias_type": "lemma"},
        ],
    },
]

# ── Demo tekst z hardcoded kandydatami ────────────────────────────────────────

_DEMO_TEXT = """\
Komisja Nadzoru Finansowego wydała w ubiegłym tygodniu decyzję dotyczącą \
restrukturyzacji trzech banków komercyjnych. Rzecznik KNF poinformował, że \
sprawa zostanie przekazana do Urzędu Ochrony Konkurencji i Konsumentów.

Prezes Narodowego Banku Polskiego zabrał głos w debacie budżetowej przed \
Sejmem. NBP opublikował raport wskazujący na ryzyko inflacyjne w II kwartale.

Pełnomocnik Jan Kowalski złożył wniosek na podstawie k.p.a., powołując się \
na art. 61 Kodeksu postępowania administracyjnego. Sędzia Anna Nowak \
zarządziła przerwę w rozprawie. Pełnomocnik Kowalskiego potwierdził \
otrzymanie wezwania.

UOKiK wszczął postępowanie wyjaśniające wobec czterech operatorów \
telekomunikacyjnych. Zgodnie z Kodeksem cywilnym, stronom przysługuje \
prawo do odwołania w terminie 14 dni.\
"""

# Hardcoded kandydaci dla _DEMO_TEXT (omijamy NER, bo tekst jest znany)
_DEMO_CANDIDATES = [
    ("Komisja Nadzoru Finansowego", "ORG"),
    ("KNF",                          "ORG"),
    ("Urząd Ochrony Konkurencji i Konsumentów", "ORG"),
    ("UOKiK",                        "ORG"),
    ("Narodowego Banku Polskiego",   "ORG"),
    ("NBP",                          "ORG"),
    ("Sejmem",                       "ORG"),
    ("k.p.a.",                       "LAW"),
    ("Kodeksu postępowania administracyjnego", "LAW"),
    ("Kodeksem cywilnym",            "LAW"),
    ("Jan Kowalski",                 "PERSON"),
    ("Anna Nowak",                   "PERSON"),
    ("Kowalskiego",                  "PERSON"),
]

# ── Drukowanie analizy NLP ────────────────────────────────────────────────────

_DEP_IMPORTANT = {"nsubj", "obj", "iobj", "nmod", "amod", "root", "conj", "obl"}


def print_nlp_analysis(analysis: StanzaAnalysis, color: bool) -> None:
    """Drukuje tabelę: ID | TOKEN | LEMMA | UPOS | NER | DEPREL | GŁOWA."""
    show_ner = analysis.has_ner()

    for sent in analysis.sentences:
        print()
        header = f"  Zdanie {sent.sent_id + 1}: {sent.text}"
        print(_bold(header, color))
        print()

        # Szerokości kolumn — z NER lub bez
        if show_ner:
            col_w = [4, 16, 16, 8, 10, 10, 14]
            col_names = ["ID", "TOKEN", "LEMMA", "UPOS", "NER", "DEPREL", "GŁOWA"]
        else:
            col_w = [4, 18, 18, 10, 12, 18]
            col_names = ["ID", "TOKEN", "LEMMA", "UPOS", "DEPREL", "GŁOWA"]

        sep = "  " + "-" * (sum(col_w) + len(col_w) * 3)
        hdr_parts = [
            f"{col_names[0]:>{col_w[0]}}",
            *[f"{col_names[i]:<{col_w[i]}}" for i in range(1, len(col_names))],
        ]
        print("  " + "   ".join(hdr_parts))
        print(sep)

        tok_by_id = {t.id: t for t in sent.tokens}
        for t in sent.tokens:
            head_tok = tok_by_id.get(t.head)
            head_text = head_tok.text if head_tok else "ROOT"
            pos_col = _POS_COLOR.get(t.upos, "")
            dep_col = _MAGENTA if t.deprel in _DEP_IMPORTANT else _DIM

            # NER: strip BIO prefix, koloruj
            ner_raw = t.ner
            ner_type = _ner_type(ner_raw) if ner_raw != "O" else ""
            ner_col = _NER_COLOR.get(ner_type, _WHITE) if ner_type else _DIM

            if show_ner:
                display_ner = ner_raw if ner_raw != "O" else "-"
                row = (
                    _col(f"{t.id:>{col_w[0]}}", _DIM, color),
                    _col(f"{t.text:<{col_w[1]}}", pos_col + _BOLD, color),
                    _col(f"{t.lemma:<{col_w[2]}}", pos_col, color),
                    _col(f"{t.upos:<{col_w[3]}}", pos_col, color),
                    _col(f"{display_ner:<{col_w[4]}}", ner_col, color),
                    _col(f"{t.deprel:<{col_w[5]}}", dep_col, color),
                    _col(f"{head_text:<{col_w[6]}}", _DIM, color),
                )
            else:
                row = (
                    _col(f"{t.id:>{col_w[0]}}", _DIM, color),
                    _col(f"{t.text:<{col_w[1]}}", pos_col + _BOLD, color),
                    _col(f"{t.lemma:<{col_w[2]}}", pos_col, color),
                    _col(f"{t.upos:<{col_w[3]}}", pos_col, color),
                    _col(f"{t.deprel:<{col_w[4]}}", dep_col, color),
                    _col(f"{head_text:<{col_w[5]}}", _DIM, color),
                )
            print("  " + "   ".join(row))

        print(sep)

        # NER entities zgrupowane
        if sent.ner_entities:
            print()
            print(_col("  Encje NER w zdaniu:", _CYAN, color))
            for ent in sent.ner_entities:
                ner_col = _NER_COLOR.get(ent.label, _WHITE)
                label_s = _col(f"[{ent.label}]", ner_col, color)
                print(f"    [{ent.start_char}:{ent.end_char}]  {label_s}  {ent.text!r}")
        elif show_ner:
            print()
            print(_col("  (brak encji NER w tym zdaniu)", _DIM, color))


# ── Drukowanie wyników linkowania ─────────────────────────────────────────────

def print_linking_results(
    linker: StanzaEntityLinker,
    candidates: list[tuple[str, str]],
    color: bool,
) -> None:
    if not candidates:
        print(_col("  (brak kandydatów do linkowania)", _DIM, color))
        return

    print()
    print(_bold("  Wyniki linkowania encji:", color))
    print()

    col_w = [33, 8, 9, 33, 28]
    sep = "  " + "-" * (sum(col_w) + len(col_w) * 3)
    hdr = (
        f"  {'WZMIANKA':<{col_w[0]}}"
        f"   {'ETYKIETA':<{col_w[1]}}"
        f"   {'STATUS':<{col_w[2]}}"
        f"   {'CANONICAL':<{col_w[3]}}"
        f"   {'ID':<{col_w[4]}}"
    )
    print(_bold(hdr, color))
    print(sep)

    linked_count = new_count = 0
    # entity_id → canonical_name zapamiętane z seed, żeby odróżnić new od linked
    seeded_ids: set[str] = {
        e["entity_id"] for e in _SEED_ENTITIES
    }

    for name, label in candidates:
        ref = linker.link(name, label)
        is_seeded = ref.entity_id in seeded_ids

        if is_seeded:
            status = "linked"
            icon = "✓"
            status_col = _GREEN
            linked_count += 1
        else:
            status = "new"
            icon = "+"
            status_col = _YELLOW
            new_count += 1
            seeded_ids.add(ref.entity_id)  # żeby drugi raz ten sam był "linked"

        name_s   = f"{name:<{col_w[0]}}"
        label_s  = f"{label:<{col_w[1]}}"
        status_s = f"{icon} {status:<{col_w[2] - 2}}"
        canon_s  = f"{ref.canonical_name:<{col_w[3]}}"
        id_s     = f"{ref.entity_id:<{col_w[4]}}"

        if color:
            print(
                f"  {_BOLD}{name_s}{_RESET}"
                f"   {label_s}"
                f"   {status_col}{status_s}{_RESET}"
                f"   {canon_s}"
                f"   {_DIM}{id_s}{_RESET}"
            )
        else:
            print(f"  {name_s}   {label_s}   {status_s}   {canon_s}   {id_s}")

    print(sep)
    print(_bold(f"  Podsumowanie: linked={linked_count}  new={new_count}", color))


# ── Budowanie linkera ─────────────────────────────────────────────────────────

def build_linker() -> StanzaEntityLinker:
    linker = StanzaEntityLinker(
        config=StanzaConfig(
            lang="pl",
            processors="tokenize,pos,lemma,depparse,ner",
            fuzzy_link_threshold=85.0,
            fuzzy_ambiguous_threshold=70.0,
        )
    )
    linker.seed_entities(_SEED_ENTITIES)
    return linker


# ── Math NER demo ──────────────────────────────────────────────────────────────

_MATH_DEMO_TEXT = """\
Twierdzenie Pitagorasa stwierdza, że w trójkącie prostokątnym suma kwadratów \
przyprostokątnych jest równa kwadratowi przeciwprostokątnej.

Lemat Zorna jest narzędziem używanym w dowodzie aksjomatu wyboru. \
Definicja przestrzeni metrycznej wymaga określenia funkcji odległości \
spełniającej warunki symetrii i nierówności trójkąta.

Euler opublikował hipotezę dotyczącą wielościanów, a Gauss udowodnił \
twierdzenie zasadnicze algebry. Aksjomat regularności wyklucza \
zbiory należące do siebie samych.\
"""

_MATH_NER_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output", "math_ner", "pl_math_ner.pt",
)
_STANZA_RESOURCES = os.path.expanduser("~/stanza_resources/pl")


def build_math_ner_linker() -> StanzaEntityLinker:
    """Buduje linker z wytrenowanym modelem Math NER."""
    return StanzaEntityLinker(
        config=StanzaConfig(
            lang="pl",
            processors="tokenize,pos,lemma,depparse,ner",
            fuzzy_link_threshold=85.0,
            fuzzy_ambiguous_threshold=70.0,
            ner_model_path=_MATH_NER_MODEL,
            ner_charlm_forward_file=os.path.join(_STANZA_RESOURCES, "forward_charlm", "oscar.pt"),
            ner_charlm_backward_file=os.path.join(_STANZA_RESOURCES, "backward_charlm", "oscar.pt"),
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(text: str, use_hardcoded_candidates: bool = False, math_ner: bool = False) -> None:
    color = _supports_color()

    print()
    print(_bold("=" * 80, color))
    print(_bold("  STANZA NLP PIPELINE DEMO — ProveNuance", color))
    print(_bold("=" * 80, color))
    print()
    mode = "Math NER (custom model)" if math_ner else "standard PL NER"
    print(f"  Język: Polish (pl)  |  Pipeline: tokenize + pos + lemma + depparse + ner  |  Tryb: {mode}")
    print()

    if math_ner:
        print(_col(f"  [1/3] Ładowanie Math NER linker ({_MATH_NER_MODEL})...", _DIM, color))
        linker = build_math_ner_linker()
    else:
        print(_col("  [1/3] Inicjalizacja StanzaEntityLinker + seedowanie encji...", _DIM, color))
        linker = build_linker()

    print(_col("  [2/3] Analiza NLP tekstu...", _DIM, color))
    print()
    print(_bold("-" * 80, color))
    print(_bold(f"  TEKST ({len(text)} znaków):", color))
    print(_bold("-" * 80, color))
    print()
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print()

    analysis = linker.analyze(text)
    ner_entities = analysis.all_ner_entities

    print(_bold("-" * 80, color))
    print(_bold(
        f"  ANALIZA NLP ({len(analysis.sentences)} zdań, "
        f"{len(analysis.all_tokens)} tokenów, "
        f"{len(ner_entities)} encji NER):",
        color,
    ))
    print(_bold("-" * 80, color))

    print_nlp_analysis(analysis, color)

    # Wybierz kandydatów: hardcoded (demo text) lub dynamicznie z NER
    if use_hardcoded_candidates:
        candidates = _DEMO_CANDIDATES
        source = "hardcoded"
    elif ner_entities:
        candidates = [(e.text, _ner_to_entity_type(e.label)) for e in ner_entities]
        source = "NER"
    else:
        # fallback: noun_spans gdy brak NER
        candidates = [(surf, "MISC") for surf, _lemma, _s, _e in analysis.noun_spans()]
        source = "noun_spans (fallback)"

    print()
    print(_bold("-" * 80, color))
    print(_bold(f"  [3/3] LINKOWANIE ENCJI (kandydaci z: {source}):", color))
    print(_bold("-" * 80, color))

    print_linking_results(linker, candidates, color)

    # Legenda POS + NER
    print()
    print(_col("  Legenda POS (Universal Dependencies):", _DIM, color))
    pos_legend = [
        ("NOUN", "rzeczownik"), ("PROPN", "nazwa własna"), ("VERB", "czasownik"),
        ("ADJ",  "przymiotnik"), ("ADV",  "przysłówek"),   ("NUM",  "liczebnik"),
        ("ADP",  "przyimek"),   ("CONJ", "spójnik"),       ("PUNCT","interpunkcja"),
    ]
    for pos, desc in pos_legend:
        pos_col = _POS_COLOR.get(pos, "")
        print(f"    {_col(f'{pos:<8}', pos_col, color)}  {desc}")

    print()
    print(_col("  Legenda NER:", _DIM, color))
    if math_ner:
        ner_legend = [
            ("THEOREM",    "twierdzenie"),
            ("LEMMA",      "lemat"),
            ("DEF",        "definicja"),
            ("AXIOM",      "aksjomat"),
            ("PROPOSITION","propozycja"),
            ("COROLLARY",  "wniosek"),
            ("CONJECTURE", "hipoteza"),
            ("STRUCTURE",  "struktura matematyczna"),
            ("FUNCTION",   "funkcja"),
            ("CONSTANT",   "stała"),
            ("SET",        "zbiór"),
            ("SEQUENCE",   "ciąg"),
            ("AUTHOR",     "autor"),
            ("REF",        "odniesienie"),
        ]
    else:
        ner_legend = [
            ("PER / persName",  "osoba"),
            ("ORG / orgName",   "organizacja"),
            ("LOC / placeName", "lokalizacja"),
            ("MISC",            "inne"),
        ]
    for ner_tag, desc in ner_legend:
        raw = ner_tag.split("/")[0].strip()
        ner_col = _NER_COLOR.get(raw, _WHITE)
        print(f"    {_col(f'{ner_tag:<22}', ner_col, color)}  {desc}")
    print()


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "--download":
        print("Pobieranie modelu stanza dla języka polskiego...")
        StanzaEntityLinker.download_model("pl", verbose=True)
        print("Model pobrany. Możesz teraz uruchomić demo bez --download.")
        return

    math_ner = "--math-ner" in args
    if math_ner:
        args = [a for a in args if a != "--math-ner"]
        if not os.path.isfile(_MATH_NER_MODEL):
            print(f"Błąd: model Math NER nie znaleziony: {_MATH_NER_MODEL}", file=sys.stderr)
            print("Uruchom najpierw: python train_math_ner.py", file=sys.stderr)
            sys.exit(1)

    if args and args[0] == "-":
        text = sys.stdin.read()
        run(text, use_hardcoded_candidates=False, math_ner=math_ner)
    elif args:
        text = " ".join(args)
        run(text, use_hardcoded_candidates=False, math_ner=math_ner)
    elif math_ner:
        run(_MATH_DEMO_TEXT, use_hardcoded_candidates=False, math_ner=True)
    else:
        run(_DEMO_TEXT, use_hardcoded_candidates=True)


if __name__ == "__main__":
    main()
