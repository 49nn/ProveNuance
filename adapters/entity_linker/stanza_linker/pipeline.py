"""
stanza_linker/pipeline.py — leniwie ładowany wrapper wokół stanza.Pipeline.

Obsługuje:
  - lazy init (stanza ładuje modele przy pierwszym użyciu)
  - processors: tokenize + pos + lemma + depparse + ner
  - wypełnianie token.ner z poziomów BIO (word.parent.ner)
  - wypełnianie sent.ner_entities z sent.ents (scalone encje)
  - graceful degradation jeśli stanza nie jest zainstalowana
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .models import NerEntity, StanzaAnalysis, StanzaSentence, StanzaToken

if TYPE_CHECKING:
    import stanza as _stanza_module


class StanzaPipeline:
    """
    Wrapper wokół stanza.Pipeline z lazy init i obsługą błędów.

    Użycie:
        pipe = StanzaPipeline(lang="pl")
        analysis = pipe.analyze("Alicja K. siedzi na kanapie.")
        for ent in analysis.all_ner_entities:
            print(ent.text, ent.label)
        for token in analysis.all_tokens:
            print(token.text, token.upos, token.lemma, token.ner)
    """

    def __init__(
        self,
        lang: str = "pl",
        processors: str = "tokenize,pos,lemma,depparse,ner",
        use_gpu: bool = False,
        verbose: bool = False,
        ner_model_path: str | None = None,
        ner_charlm_forward_file: str | None = None,
        ner_charlm_backward_file: str | None = None,
    ) -> None:
        self.lang = lang
        self.processors = processors
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.ner_model_path = ner_model_path
        self.ner_charlm_forward_file = ner_charlm_forward_file
        self.ner_charlm_backward_file = ner_charlm_backward_file
        self._nlp: object | None = None  # lazy

    def _load(self) -> None:
        """Ładuje pipeline stanza (pobiera model jeśli potrzeba)."""
        try:
            import stanza
        except ImportError as exc:
            raise ImportError(
                "Stanza nie jest zainstalowana. "
                "Zainstaluj ją: pip install stanza"
            ) from exc

        # Argumenty specyficzne dla procesora NER przekazujemy przez kwargs
        # z prefiksem "ner_": stanza.Pipeline routuje je do procesora NER.
        extra_kwargs: dict = {}
        if self.ner_model_path:
            extra_kwargs["ner_model_path"] = self.ner_model_path
        if self.ner_charlm_forward_file:
            extra_kwargs["ner_charlm_forward_file"] = self.ner_charlm_forward_file
        if self.ner_charlm_backward_file:
            extra_kwargs["ner_charlm_backward_file"] = self.ner_charlm_backward_file

        self._nlp = stanza.Pipeline(
            lang=self.lang,
            processors=self.processors,
            use_gpu=self.use_gpu,
            verbose=self.verbose,
            **extra_kwargs,
        )

    def analyze(self, text: str) -> StanzaAnalysis:
        """
        Uruchamia pipeline NLP na tekście.

        Zwraca StanzaAnalysis z pełną analizą tokenów, POS, lemm, zależności
        i encji NER (jeśli procesor 'ner' jest załadowany).
        """
        if self._nlp is None:
            self._load()

        doc = self._nlp(text)  # type: ignore[operator]
        sentences: list[StanzaSentence] = []

        for sent_id, sent in enumerate(doc.sentences):
            stanza_sent = StanzaSentence(
                sent_id=sent_id,
                text=sent.text,
            )

            # ── Tokeny z pełną analizą ─────────────────────────────────────
            for word in sent.words:
                # NER na poziomie tokenu (word.parent to Token ze stanza)
                ner_tag = "O"
                try:
                    parent = getattr(word, "parent", None)
                    if parent is not None:
                        raw = getattr(parent, "ner", None)
                        if raw:
                            ner_tag = raw
                except Exception:
                    pass

                token = StanzaToken(
                    id=word.id,
                    text=word.text,
                    lemma=word.lemma or word.text,
                    upos=word.upos or "X",
                    xpos=word.xpos or "_",
                    feats=word.feats or "_",
                    head=word.head or 0,
                    deprel=word.deprel or "dep",
                    start_char=word.start_char if hasattr(word, "start_char") else 0,
                    end_char=word.end_char if hasattr(word, "end_char") else 0,
                    ner=ner_tag,
                )
                stanza_sent.tokens.append(token)

            # ── Scalone encje NER z sent.ents ──────────────────────────────
            try:
                for ent in sent.ents:
                    stanza_sent.ner_entities.append(
                        NerEntity(
                            text=ent.text,
                            label=ent.type,
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                        )
                    )
            except Exception:
                pass  # ner nie załadowany lub brak encji

            sentences.append(stanza_sent)

        return StanzaAnalysis(text=text, sentences=sentences, lang=self.lang)

    @staticmethod
    def download(lang: str = "pl", verbose: bool = True) -> None:
        """Pobiera modele stanza dla danego języka.

        Wywołaj raz przed pierwszym użyciem:
            StanzaPipeline.download("pl")
        """
        try:
            import stanza
        except ImportError as exc:
            raise ImportError(
                "Stanza nie jest zainstalowana. "
                "Zainstaluj ją: pip install stanza"
            ) from exc
        stanza.download(lang, verbose=verbose)
