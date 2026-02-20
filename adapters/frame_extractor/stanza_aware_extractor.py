"""
Adapter: StanzaAwareExtractor

Pipeline NLP → Frame uruchamiany PO NER (wejście: DocSpan lub StanzaAnalysis).

Implementuje port FrameExtractor:
  extract_frames(span)   → kroki A+B+C → list[Frame]
  validate_frame(frame)  → krok D      → list[ValidationIssue]

Dodatkowe publiczne API (poza portem):
  extract(analysis)      → A+B+C+D    → list[Frame | HypothesisFrame]

Kroki:
  A  segmentacja na kandydatów (słowa-sygnały + klauzule warunkowe)
  B  klasyfikacja typu frame   (reguły na tokenach / lemma / POS / feats)
  C  wypełnianie slotów        (pattern matching na drzewie zależności)
  D  walidacja przez schemat   (→ Frame lub HypothesisFrame)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from contracts import (
    ArithExampleFrame,
    ConditionFrame,
    DefinitionFrame,
    DocSpan,
    Frame,
    FrameType,
    HypothesisFrame,
    PropertyFrame,
    TaskFrame,
    ValidationIssue,
)
from adapters.frame_extractor._printer import print_frames
from adapters.entity_linker.stanza_linker.models import (
    StanzaAnalysis,
    StanzaSentence,
    StanzaToken,
)

if TYPE_CHECKING:
    from adapters.entity_linker.stanza_linker import StanzaEntityLinker

# ── Słowniki cue words ────────────────────────────────────────────────────────

# lemma → podpowiedź frame_type (None = brak pewności, dalej klasyfikuj w B)
_SIGNAL_LEMMAS: dict[str, FrameType | None] = {
    "przykład":  FrameType.ARITH_EXAMPLE,
    "np":        FrameType.ARITH_EXAMPLE,
    "zadanie":   FrameType.TASK,
    "oblicz":    FrameType.TASK,
    "uzupełnij": FrameType.TASK,
    "reguła":    FrameType.PROPERTY,
    "uwaga":     None,   # generic — Step B zadecyduje
    # nieformalne sygnały właściwości obecne w podręcznikach klas 1-3
    "kolejność": FrameType.PROPERTY,   # "Kolejność dodawania nie zmienia wyniku"
    "zero":      FrameType.PROPERTY,   # "Dodanie zera nic nie zmienia"
}

_CONDITIONAL_LEMMAS: frozenset[str] = frozenset({"jeśli", "jeżeli", "gdy", "kiedy"})
_CONDITIONAL_CONSQ:  frozenset[str] = frozenset({"to", "wtedy"})

# definicja
_DEF_CUES: frozenset[str] = frozenset({
    "być", "oznaczać", "nazywać", "mówić", "definiować", "stanowić",
})

# właściwość/reguła
_PROP_CUES: frozenset[str] = frozenset({
    "zawsze", "każdy", "każda", "własność", "przemienne", "łączny",
    "rozdzielny", "przemienność", "łączność", "rozdzielność",
})
_PROP_NAME_MAP: dict[str, str] = {
    "przemienne":    "commutative",
    "przemienność":  "commutative",
    "kolejność":     "commutative",   # "kolejność nie zmienia" → przemienność
    "zamienić":      "commutative",   # "zamienić miejscami"
    "przestawić":    "commutative",
    "łączny":        "associative",
    "łączność":      "associative",
    "rozdzielny":    "distributive",
    "rozdzielność":  "distributive",
    "zero":          "zero_neutral",  # "dodanie zera nic nie zmienia"
}

# normalizacja operatorów
_OP_SYMBOL_MAP: dict[str, str] = {
    "+": "add", "-": "sub",
    "×": "mul", "·": "mul", "*": "mul",
    "/": "div", "÷": "div",
}
_SUBJ_LEMMA_MAP: dict[str, str] = {
    "dodawanie": "addition",   "dodać": "addition",
    "dodanie":   "addition",   # rzeczownik odsłowny: "Dodanie zera"
    "odejmowanie": "subtraction", "odjąć": "subtraction",
    "odjęcie":   "subtraction",
    "mnożenie": "multiplication", "mnożyć": "multiplication",
    "dzielenie": "division",   "dzielić": "division",
}

_INLINE_ARITH_RE = re.compile(r"(\d+)\s*([+\-×·*/÷])\s*(\d+)\s*=\s*(\d+)")


# ── Wewnętrzny model kandydata ────────────────────────────────────────────────

@dataclass
class CandidateSpan:
    """Segment zdania z metadanymi (szczegół implementacyjny adaptera)."""
    tokens: list[StanzaToken]
    sent: StanzaSentence
    trigger_lemma: Optional[str] = None   # lemma słowa-sygnału z Kroku A
    frame_hint: Optional[FrameType] = None  # wstępna wskazówka z Kroku A

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self.tokens)

    @property
    def lemmas(self) -> list[str]:
        return [t.lemma.lower() for t in self.tokens]

    @property
    def lemma_set(self) -> frozenset[str]:
        return frozenset(self.lemmas)

    def tokens_before(self, idx: int) -> list[StanzaToken]:
        return self.tokens[:idx]

    def tokens_after(self, idx: int) -> list[StanzaToken]:
        return self.tokens[idx + 1:]

    def dep_children(self, head_id: int, rels: set[str] | None = None) -> list[StanzaToken]:
        """Dzieci tokenu w drzewie zależności (opcjonalnie filtrowane po deprel)."""
        return [
            t for t in self.tokens
            if t.head == head_id and (rels is None or t.deprel in rels)
        ]

    def root_token(self) -> Optional[StanzaToken]:
        return next((t for t in self.tokens if t.deprel == "root"), None)


# ── Główny adapter ────────────────────────────────────────────────────────────

class StanzaAwareExtractor:
    """
    Wyciąga Frame'y z DocSpan (przez stanza) lub StanzaAnalysis.

    Implementuje port FrameExtractor — może zastąpić RuleBasedExtractor
    bez żadnych zmian w routerze /extract.

    Użycie jako port (router /extract):
        extractor = StanzaAwareExtractor(linker)
        frames = extractor.extract_frames(span)          # A+B+C
        issues = extractor.validate_frame(frame)         # D

    Użycie bezpośrednie (pełny pipeline):
        results = extractor.extract(analysis)            # A+B+C+D
    """

    def __init__(self, linker: "StanzaEntityLinker", verbose: bool = False) -> None:
        self._linker = linker
        self._verbose = verbose

    # ── Port: FrameExtractor ──────────────────────────────────────────────────

    def extract_frames(self, span: DocSpan) -> list[Frame]:
        """
        Konformuje do portu FrameExtractor.
        Uruchamia stanza na surface_text spanu → kroki A, B, C → list[Frame].
        Kroku D (walidacja) tu nie ma — router woła validate_frame osobno.
        """
        analysis = self._linker.analyze(span.surface_text)
        frames: list[Frame] = []
        for sent in analysis.sentences:
            for cand in self._step_a_segment(sent):
                frame_type = self._step_b_classify(cand)
                if frame_type is None:
                    continue
                frame = self._step_c_fill_slots(cand, frame_type)
                if frame is not None:
                    frames.append(frame)

        # Fallback: extract simple arithmetic equations directly from text.
        # This handles spans with dense answer lists where POS tags are noisy.
        seen_arith = {
            (
                f.operation,
                tuple(f.operands),
                f.result,
            )
            for f in frames
            if isinstance(f, ArithExampleFrame)
        }
        for frame in _extract_inline_arith_frames_from_text(span.surface_text, span.span_id):
            sig = (frame.operation, tuple(frame.operands), frame.result)
            if sig not in seen_arith:
                frames.append(frame)
                seen_arith.add(sig)
        if self._verbose:
            print_frames(span.surface_text, frames)
        return frames

    def validate_frame(self, frame: Frame) -> list[ValidationIssue]:
        """
        Konformuje do portu FrameExtractor.
        Odpowiada krokowi D — sprawdza kompletność i spójność slotów.
        """
        return self._check_schema(frame)

    # ── Pełny pipeline (poza portem) ─────────────────────────────────────────

    def extract(
        self, analysis: StanzaAnalysis
    ) -> list[Frame | HypothesisFrame]:
        results: list[Frame | HypothesisFrame] = []
        for sent in analysis.sentences:
            for cand in self._step_a_segment(sent):
                frame_type = self._step_b_classify(cand)
                if frame_type is None:
                    continue
                frame = self._step_c_fill_slots(cand, frame_type)
                if frame is None:
                    continue
                results.append(self._step_d_validate(frame))
        return results

    # ── Krok A: Segmentacja ───────────────────────────────────────────────────

    def _step_a_segment(self, sent: StanzaSentence) -> list[CandidateSpan]:
        """
        Wyszukuje zdania z sygnałami lub klauzule warunkowe.
        Każde trafienie → jeden CandidateSpan (na razie = całe zdanie).
        """
        tokens = sent.tokens
        candidates: list[CandidateSpan] = []
        lemmas = [t.lemma.lower() for t in tokens]

        # (1) Słowa-sygnały — pierwsze trafienie w zdaniu wystarczy
        for tok in tokens:
            lemma = tok.lemma.lower()
            if lemma in _SIGNAL_LEMMAS:
                candidates.append(CandidateSpan(
                    tokens=tokens,
                    sent=sent,
                    trigger_lemma=lemma,
                    frame_hint=_SIGNAL_LEMMAS[lemma],
                ))
                break  # jedno zdanie → jeden kandydat z sygnału

        # (2) Klauzula warunkowa "Jeśli … to …"
        has_cond  = any(l in _CONDITIONAL_LEMMAS for l in lemmas)
        has_consq = any(l in _CONDITIONAL_CONSQ  for l in lemmas)
        if has_cond and has_consq:
            # Nie duplikuj jeśli już dodano z powodu słowa-sygnału
            already = any(c.frame_hint == FrameType.CONDITION for c in candidates)
            if not already:
                candidates.append(CandidateSpan(
                    tokens=tokens,
                    sent=sent,
                    trigger_lemma=next(l for l in lemmas if l in _CONDITIONAL_LEMMAS),
                    frame_hint=FrameType.CONDITION,
                ))

        # (3) "Ile to jest…" — pytanie o obliczenie
        for i, tok in enumerate(tokens):
            if tok.lemma.lower() == "ile":
                already = any(c.frame_hint == FrameType.TASK for c in candidates)
                if not already:
                    candidates.append(CandidateSpan(
                        tokens=tokens,
                        sent=sent,
                        trigger_lemma="ile",
                        frame_hint=FrameType.TASK,
                    ))
                break

        # (4) Wyrażenie arytmetyczne: znak '=' + co najmniej 2 NUM
        #     Np. "1 + 1 = 2", "3 + 3 = 6" — bez żadnego słowa-sygnału
        has_eq = any(t.text == "=" for t in tokens)
        n_nums = sum(1 for t in tokens if t.upos == "NUM")
        if has_eq and n_nums >= 2:
            already = any(c.frame_hint == FrameType.ARITH_EXAMPLE for c in candidates)
            if not already:
                candidates.append(CandidateSpan(
                    tokens=tokens,
                    sent=sent,
                    trigger_lemma=None,
                    frame_hint=FrameType.ARITH_EXAMPLE,
                ))

        return candidates

    # ── Krok B: Klasyfikacja ──────────────────────────────────────────────────

    def _step_b_classify(self, cand: CandidateSpan) -> Optional[FrameType]:
        """
        Zwraca typ frame lub None jeśli zdanie nie pasuje do żadnego wzorca.
        Hint z Kroku A ma pierwszeństwo; jeśli None — pełna klasyfikacja.
        """
        if cand.frame_hint is not None:
            return cand.frame_hint

        ls = cand.lemma_set

        # ARITH_EXAMPLE: co najmniej 2 NUM + znak '='
        has_eq  = any(t.text == "=" for t in cand.tokens)
        n_nums  = sum(1 for t in cand.tokens if t.upos == "NUM")
        if has_eq and n_nums >= 2:
            return FrameType.ARITH_EXAMPLE

        # TASK: tryb rozkazujący (Mood=Imp w feats)
        for tok in cand.tokens:
            if tok.upos == "VERB" and "Mood=Imp" in tok.feats:
                return FrameType.TASK

        # DEFINITION: czasownik definiujący + rzeczownik jako podmiot
        if ls & _DEF_CUES:
            subj_before = any(
                t.upos in ("NOUN", "PROPN")
                for t in cand.tokens_before(
                    next(i for i, t in enumerate(cand.tokens) if t.lemma.lower() in _DEF_CUES)
                )
            )
            if subj_before:
                return FrameType.DEFINITION

        # PROPERTY / RULE: słowa kluczowe
        if ls & _PROP_CUES:
            return FrameType.PROPERTY

        return None

    # ── Krok C: Wypełnianie slotów ────────────────────────────────────────────

    def _step_c_fill_slots(
        self, cand: CandidateSpan, frame_type: FrameType
    ) -> Optional[Frame]:
        span_id = f"sent_{cand.sent.sent_id}"
        dispatch = {
            FrameType.ARITH_EXAMPLE: self._fill_arith,
            FrameType.PROPERTY:      self._fill_property,
            FrameType.DEFINITION:    self._fill_definition,
            FrameType.TASK:          self._fill_task,
            FrameType.CONDITION:     self._fill_condition,
        }
        filler = dispatch.get(frame_type)
        return filler(cand, span_id) if filler else None

    def _fill_arith(self, cand: CandidateSpan, span_id: str) -> Optional[ArithExampleFrame]:
        """
        Szuka wzorca: NUM OP NUM = NUM
        Używa pozycji znaku '=' do podziału na lewo (operandy) / prawo (wynik).
        """
        eq_pos = next(
            (i for i, t in enumerate(cand.tokens) if t.text == "="), None
        )
        if eq_pos is None:
            return None

        lhs = cand.tokens_before(eq_pos)
        rhs = cand.tokens_after(eq_pos)

        num_lhs = [t for t in lhs if t.upos == "NUM"]
        num_rhs = [t for t in rhs if t.upos == "NUM"]
        ops     = [t for t in lhs if t.text in _OP_SYMBOL_MAP]

        if len(num_lhs) < 2 or not num_rhs or not ops:
            return None

        def _to_int(tok: StanzaToken) -> int | str:
            try:
                return int(tok.text)
            except ValueError:
                return tok.text

        return ArithExampleFrame(
            operation=_OP_SYMBOL_MAP[ops[0].text],
            operands=[_to_int(num_lhs[0]), _to_int(num_lhs[1])],
            result=_to_int(num_rhs[0]),
            source_span_id=span_id,
        )

    def _fill_property(self, cand: CandidateSpan, span_id: str) -> Optional[PropertyFrame]:
        """
        Wzorzec: <podmiot-operacja> [jest/nic] <cecha>.
        Podmiot → pierwsze NOUN/PROPN z _SUBJ_LEMMA_MAP (token-trigger pomijany).
        Cecha   → token z _PROP_NAME_MAP (sprawdzany u wszystkich tokenów).

        Trigger token (np. "reguła", "kolejność", "zero") jest pomijany
        przy szukaniu podmiotu, bo sam nie jest operacją matematyczną.
        Cecha jest jednak sprawdzana również dla triggera — np. "kolejność"
        jest jednocześnie triggerem i nazwą własności (commutative).
        """
        subj: Optional[str] = None
        prop: Optional[str] = None
        trigger = cand.trigger_lemma  # np. "reguła", "kolejność", "zero"

        for tok in cand.tokens:
            ll = tok.lemma.lower()
            # Cecha: szukaj u wszystkich tokenów (trigger może być też własnością)
            if ll in _PROP_NAME_MAP:
                prop = _PROP_NAME_MAP[ll]
            # Podmiot: tylko znane operacje, pomiń token-trigger
            if tok.upos in ("NOUN", "PROPN") and subj is None and ll != trigger:
                mapped = _SUBJ_LEMMA_MAP.get(ll)
                if mapped:
                    subj = mapped

        if subj and prop:
            return PropertyFrame(subject=subj, property_name=prop, source_span_id=span_id)
        return None

    def _fill_definition(self, cand: CandidateSpan, span_id: str) -> Optional[DefinitionFrame]:
        """
        Wzorzec: <termin (nsubj)> <cue: jest/oznacza/…> <treść definicji>.
        Próbuje najpierw przez relację nsubj w drzewie zależności,
        fallback: pierwsze NOUN/PROPN przed cue-słowem.
        """
        cue_idx: Optional[int] = None
        cue_tok: Optional[StanzaToken] = None
        for i, tok in enumerate(cand.tokens):
            if tok.lemma.lower() in _DEF_CUES:
                cue_idx, cue_tok = i, tok
                break
        if cue_idx is None or cue_tok is None:
            return None

        # Próba 1: nsubj z drzewa zależności
        nsubj_toks = cand.dep_children(cue_tok.id, rels={"nsubj"})
        if nsubj_toks:
            term = nsubj_toks[0].lemma.lower()
        else:
            # Fallback: ostatnie NOUN/PROPN przed cue
            pre = [t for t in cand.tokens_before(cue_idx) if t.upos in ("NOUN", "PROPN")]
            if not pre:
                return None
            term = pre[-1].lemma.lower()

        definition = " ".join(t.text for t in cand.tokens_after(cue_idx))
        return DefinitionFrame(term=term, definition=definition, source_span_id=span_id)

    def _fill_task(self, cand: CandidateSpan, span_id: str) -> Optional[TaskFrame]:
        """
        Wzorzec: <imperatyw> <obiekt>.
        Obiekt: dzieci deprel obj/obl/nmod w drzewie;
        fallback: wszystko po czasowniku.
        """
        # Poszukaj imperatywu (Mood=Imp) lub lemmy-sygnału
        verb_tok: Optional[StanzaToken] = None
        for tok in cand.tokens:
            if tok.upos == "VERB" and "Mood=Imp" in tok.feats:
                verb_tok = tok
                break
        if verb_tok is None:
            verb_tok = next(
                (t for t in cand.tokens if t.lemma.lower() in {"oblicz", "uzupełnij"}),
                None,
            )
        if verb_tok is None:
            return None

        # Obiekt: przez drzewo zależności (obj / obl / nmod)
        obj_toks = cand.dep_children(verb_tok.id, rels={"obj", "obl", "nmod", "xcomp"})
        if obj_toks:
            target = " ".join(t.text for t in sorted(obj_toks, key=lambda t: t.id))
        else:
            # Fallback: tokeny po czasowniku
            verb_pos = next(i for i, t in enumerate(cand.tokens) if t.id == verb_tok.id)
            rest = cand.tokens_after(verb_pos)
            target = " ".join(t.text for t in rest).strip(" :,")

        return TaskFrame(
            verb=verb_tok.lemma.lower(),
            target=target or cand.text,
            source_span_id=span_id,
        )

    def _fill_condition(self, cand: CandidateSpan, span_id: str) -> Optional[ConditionFrame]:
        """
        Wzorzec: <Jeśli> <warunek> <to> <wniosek>.
        Podział na 'to'/'wtedy' z upos PART/SCONJ/ADV.
        """
        cond_start: Optional[int] = None
        consq_start: Optional[int] = None

        for i, tok in enumerate(cand.tokens):
            ll = tok.lemma.lower()
            if ll in _CONDITIONAL_LEMMAS and cond_start is None:
                cond_start = i
            elif ll in _CONDITIONAL_CONSQ and cond_start is not None:
                consq_start = i
                break

        if cond_start is None or consq_start is None:
            return None

        condition  = " ".join(t.text for t in cand.tokens[cond_start + 1 : consq_start])
        conclusion = " ".join(t.text for t in cand.tokens[consq_start + 1 :])

        if not condition or not conclusion:
            return None

        return ConditionFrame(
            condition=condition,
            conclusion=conclusion,
            source_span_id=span_id,
        )

    # ── Krok D: Walidacja ─────────────────────────────────────────────────────

    def _step_d_validate(self, frame: Frame) -> Frame | HypothesisFrame:
        """
        Sprawdza kompletność i spójność slotów.
        Frame poprawny → zwraca Frame.
        Frame niepewny → zwraca HypothesisFrame z flagą needs_review.
        """
        issues = self._check_schema(frame)
        if not issues:
            return frame

        errors   = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        # confidence: 1.0 − 0.3*błędy − 0.1*ostrzeżenia (min 0.0)
        confidence = max(0.0, 1.0 - 0.3 * len(errors) - 0.1 * len(warnings))
        return HypothesisFrame(
            frame=frame,
            confidence=confidence,
            issues=issues,
            needs_review=bool(errors),
        )

    def _check_schema(self, frame: Frame) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if isinstance(frame, ArithExampleFrame):
            if frame.operation not in {"add", "sub", "mul", "div"}:
                issues.append(ValidationIssue(
                    severity="error", code="UNKNOWN_OP",
                    message=f"Nieznana operacja: {frame.operation!r}",
                    field_path="operation",
                ))
            if len(frame.operands) != 2:
                issues.append(ValidationIssue(
                    severity="error", code="WRONG_ARITY",
                    message=f"Oczekiwano 2 operandów, znaleziono {len(frame.operands)}",
                    field_path="operands",
                ))
            elif all(isinstance(x, int) for x in (*frame.operands, frame.result)):
                a, b, r = frame.operands[0], frame.operands[1], frame.result  # type: ignore[misc]
                expected = {"add": a + b, "sub": a - b, "mul": a * b}.get(frame.operation)
                if expected is not None and expected != r:
                    issues.append(ValidationIssue(
                        severity="warning", code="ARITH_MISMATCH",
                        message=f"Wynik {r} ≠ oczekiwany {expected}",
                        field_path="result",
                    ))

        elif isinstance(frame, PropertyFrame):
            if not frame.subject:
                issues.append(ValidationIssue(
                    severity="error", code="MISSING_SUBJECT",
                    message="Brak podmiotu właściwości",
                    field_path="subject",
                ))
            known_props = {"commutative", "associative", "zero_neutral", "distributive", "identity"}
            if frame.property_name not in known_props:
                issues.append(ValidationIssue(
                    severity="warning", code="UNKNOWN_PROPERTY",
                    message=f"Nieznana właściwość: {frame.property_name!r}",
                    field_path="property_name",
                ))

        elif isinstance(frame, DefinitionFrame):
            if not frame.term:
                issues.append(ValidationIssue(
                    severity="error", code="MISSING_TERM",
                    message="Brak definiowanego terminu",
                    field_path="term",
                ))
            if not frame.definition or len(frame.definition.split()) < 2:
                issues.append(ValidationIssue(
                    severity="warning", code="THIN_DEFINITION",
                    message="Definicja jest zbyt krótka lub pusta",
                    field_path="definition",
                ))

        elif isinstance(frame, TaskFrame):
            if not frame.verb:
                issues.append(ValidationIssue(
                    severity="error", code="MISSING_VERB",
                    message="Brak czasownika polecenia",
                    field_path="verb",
                ))
            if not frame.target:
                issues.append(ValidationIssue(
                    severity="warning", code="MISSING_TARGET",
                    message="Nie znaleziono obiektu polecenia",
                    field_path="target",
                ))

        elif isinstance(frame, ConditionFrame):
            if not frame.condition:
                issues.append(ValidationIssue(
                    severity="error", code="MISSING_CONDITION",
                    message="Brak członu warunkowego",
                    field_path="condition",
                ))
            if not frame.conclusion:
                issues.append(ValidationIssue(
                    severity="error", code="MISSING_CONCLUSION",
                    message="Brak członu wynikowego",
                    field_path="conclusion",
                ))

        return issues


def _extract_inline_arith_frames_from_text(text: str, span_id: str) -> list[ArithExampleFrame]:
    """Extracts all simple equations `a op b = c` found in text lines."""
    frames: list[ArithExampleFrame] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for m in _INLINE_ARITH_RE.finditer(line):
            # Avoid matching the middle part of a chained expression like: 5 + 1 + 2 = 8
            prefix = line[:m.start()].rstrip()
            if prefix and prefix[-1] in "+-*/×÷·":
                continue
            suffix = line[m.end():].lstrip()
            if suffix and suffix[0] in "+-*/×÷·":
                continue

            a = int(m.group(1))
            op = m.group(2)
            b = int(m.group(3))
            r = int(m.group(4))
            op_norm = _OP_SYMBOL_MAP.get(op)
            if op_norm is None:
                continue
            frames.append(
                ArithExampleFrame(
                    operation=op_norm,
                    operands=[a, b],
                    result=r,
                    source_span_id=span_id,
                )
            )
    return frames
