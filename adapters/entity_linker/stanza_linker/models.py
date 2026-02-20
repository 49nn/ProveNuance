"""
stanza_linker/models.py — wewnętrzne modele danych dla adaptera StanzaEntityLinker.

Oddzielone od contracts.py celowo — są szczegółem implementacyjnym pipeline'u.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StanzaToken:
    """Pojedynczy token z pełną analizą NLP."""
    id: int             # indeks w zdaniu (1-based jak w CoNLL-U)
    text: str           # forma powierzchniowa
    lemma: str          # forma słownikowa / lemat
    upos: str           # Universal POS tag (NOUN, VERB, ADJ, …)
    xpos: str           # Penn/NKJP POS tag (szczegółowy)
    feats: str          # cechy morfologiczne (Gender=Masc|Number=Sing|…)
    head: int           # indeks głowy w drzewie zależności
    deprel: str         # etykieta relacji zależności (nsubj, obj, …)
    start_char: int     # pozycja startowa w tekście źródłowym
    end_char: int       # pozycja końcowa w tekście źródłowym
    ner: str = "O"      # etykieta NER na poziomie tokenu (B-PER, I-ORG, O, …)


@dataclass
class NerEntity:
    """Encja rozpoznana przez NER (wynik scalenia tagów BIO z sent.ents)."""
    text: str       # forma powierzchniowa ("Alicja K.", "Komisja Nadzoru…")
    label: str      # typ wg stanza: PER, ORG, LOC, MISC, persName, orgName, …
    start_char: int
    end_char: int


@dataclass
class StanzaSentence:
    """Zdanie z listą tokenów i metadanymi."""
    sent_id: int
    text: str
    tokens: list[StanzaToken] = field(default_factory=list)
    ner_entities: list[NerEntity] = field(default_factory=list)

    def get_lemmas(self) -> list[str]:
        return [t.lemma for t in self.tokens]

    def get_pos_tags(self) -> list[tuple[str, str]]:
        """Zwraca listę (tekst, POS) dla każdego tokenu."""
        return [(t.text, t.upos) for t in self.tokens]

    def get_dep_triples(self) -> list[tuple[str, str, str]]:
        """Zwraca listę (głowa, relacja, zależny) jako trójki tekstowe."""
        triples = []
        tok_by_id = {t.id: t for t in self.tokens}
        for t in self.tokens:
            head_tok = tok_by_id.get(t.head)
            head_text = head_tok.text if head_tok else "ROOT"
            triples.append((head_text, t.deprel, t.text))
        return triples


@dataclass
class StanzaAnalysis:
    """Wynik analizy całego tekstu przez pipeline stanza."""
    text: str
    sentences: list[StanzaSentence] = field(default_factory=list)
    lang: str = "pl"

    @property
    def all_tokens(self) -> list[StanzaToken]:
        return [t for s in self.sentences for t in s.tokens]

    @property
    def all_ner_entities(self) -> list[NerEntity]:
        """Wszystkie encje NER ze wszystkich zdań."""
        return [e for s in self.sentences for e in s.ner_entities]

    def has_ner(self) -> bool:
        """Zwraca True jeśli pipeline zawierał procesor NER."""
        return any(
            t.ner != "O" for s in self.sentences for t in s.tokens
        ) or any(
            s.ner_entities for s in self.sentences
        )

    def noun_spans(self) -> list[tuple[str, str, int, int]]:
        """Wyciąga ciągłe fragmenty NOUN/PROPN jako kandydatów encji.

        Fallback gdy NER niedostępny.
        Zwraca listę (tekst_powierzchniowy, lemma, start_char, end_char).
        """
        spans: list[tuple[str, str, int, int]] = []
        tokens = self.all_tokens
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.upos in ("NOUN", "PROPN"):
                j = i + 1
                # rozszerz ciąg o kolejne NOUN/PROPN lub ADJ/DET
                while j < len(tokens) and tokens[j].upos in ("NOUN", "PROPN", "ADJ", "DET"):
                    j += 1
                span_tokens = tokens[i:j]
                surface = " ".join(t.text for t in span_tokens)
                lemma = " ".join(t.lemma for t in span_tokens)
                start = span_tokens[0].start_char
                end = span_tokens[-1].end_char
                spans.append((surface, lemma, start, end))
                i = j
            else:
                i += 1
        return spans
