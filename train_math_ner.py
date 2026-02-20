"""
train_math_ner.py — konwersja BIO CoNLL → JSON + trening stanza NER dla matematycznych NE.

Pipeline:
  1. Wczytaj i zwaliduj pliki CoNLL (BIO format: TOKEN<TAB>TAG)
  2. Eksportuj JSON (stanza Document-format, per zdanie) + stats.json
  3. Uruchom trening stanza NER z polskim charlm (oscar) jako backbone
  4. Ewaluuj na test.conll i wypisz F1 per label
  5. Pokaż przykład użycia wytrenowanego modelu w StanzaEntityLinker

Użycie:
  python train_math_ner.py                  # pełny pipeline (CPU, 2000 kroków)
  python train_math_ner.py --steps 200      # szybki debug run
  python train_math_ner.py --convert-only   # tylko konwersja do JSON, bez treningu
  python train_math_ner.py --eval-only      # tylko ewaluacja istniejącego modelu

Wynik:
  output/math_ner/
    train.json / dev.json / test.json  — dane w stanza Document JSON format
    stats.json                         — statystyki datasetu
    pl_math_ner.pt                     — wytrenowany model NER
    eval_results.json                  — wyniki ewaluacji per label
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
from collections import Counter
from typing import Optional

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Stałe ─────────────────────────────────────────────────────────────────────

DATASET_DIR = pathlib.Path("stanza_math_ner_dataset")
OUTPUT_DIR  = pathlib.Path("output/math_ner")
STANZA_RES  = pathlib.Path.home() / "stanza_resources" / "pl"

SPLITS = ["train", "dev", "test"]

# Etykiety NER z datasetu (dla dokumentacji)
EXPECTED_LABELS = {
    "THEOREM", "LEMMA", "DEF", "STRUCTURE", "FUNCTION",
    "CONSTANT", "AUTHOR", "REF", "CONJECTURE", "AXIOM",
    "PROPOSITION", "COROLLARY", "SET", "SEQUENCE",
}

# ── Typ danych ────────────────────────────────────────────────────────────────

Sentence = list[tuple[str, str]]  # lista (token, bio_tag)


# ══════════════════════════════════════════════════════════════════════════════
# Krok 1: Wczytanie i walidacja CoNLL
# ══════════════════════════════════════════════════════════════════════════════

def read_conll(path: pathlib.Path) -> list[Sentence]:
    """
    Wczytuje plik CoNLL (TOKEN<TAB>TAG, zdania oddzielone pustą linią).

    Akceptuje też format bez tabulatora (spacja jako separator) jako fallback.
    Rzuca ValueError przy niezgodności formatu.
    """
    sentences: list[Sentence] = []
    current: Sentence = []

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip()
        if not line:
            if current:
                sentences.append(current)
                current = []
            continue

        if "\t" in line:
            parts = line.split("\t", 1)
        elif " " in line:
            parts = line.rsplit(" ", 1)
        else:
            raise ValueError(f"{path.name}:{lineno}: nie można sparsować linii: {line!r}")

        if len(parts) != 2:
            raise ValueError(f"{path.name}:{lineno}: oczekiwano 2 kolumn, znaleziono: {line!r}")

        token, tag = parts[0], parts[1]
        current.append((token, tag))

    if current:
        sentences.append(current)

    return sentences


def validate_bio(sentences: list[Sentence], filename: str) -> list[str]:
    """
    Sprawdza spójność tagów BIO. Zwraca listę komunikatów o błędach.

    Reguły:
    - I-X może wystąpić tylko po B-X lub I-X
    - Etykiety powinny być ze zbioru EXPECTED_LABELS (ostrzeżenie jeśli nie)
    """
    errors: list[str] = []
    seen_labels: set[str] = set()

    for i, sent in enumerate(sentences, 1):
        prev_tag = "O"
        for j, (tok, tag) in enumerate(sent, 1):
            if tag == "O":
                prev_tag = tag
                continue

            if "-" not in tag:
                errors.append(f"{filename} zdanie {i} token {j}: nieprawidłowy tag {tag!r}")
                prev_tag = tag
                continue

            prefix, label = tag.split("-", 1)
            seen_labels.add(label)

            if prefix == "I":
                # I-X musi być poprzedzone B-X lub I-X
                if not (prev_tag == f"B-{label}" or prev_tag == f"I-{label}"):
                    errors.append(
                        f"{filename} zdanie {i} token {j}: "
                        f"I-{label} po {prev_tag!r} (oczekiwano B-{label} lub I-{label})"
                    )
            elif prefix not in ("B",):
                errors.append(
                    f"{filename} zdanie {i} token {j}: nieznany prefix BIO {prefix!r} w tagu {tag!r}"
                )

            prev_tag = tag

    unknown = seen_labels - EXPECTED_LABELS
    if unknown:
        errors.append(f"{filename}: nieznane etykiety (nie w README): {sorted(unknown)}")

    return errors


def compute_stats(sentences: list[Sentence]) -> dict:
    """Oblicza statystyki datasetu (liczba zdań, tokenów, encji per etykieta)."""
    token_count = 0
    entity_counts: Counter[str] = Counter()
    label_token_counts: Counter[str] = Counter()

    for sent in sentences:
        token_count += len(sent)
        in_entity: Optional[str] = None
        for tok, tag in sent:
            if tag.startswith("B-"):
                label = tag[2:]
                entity_counts[label] += 1
                label_token_counts[label] += 1
                in_entity = label
            elif tag.startswith("I-"):
                label = tag[2:]
                label_token_counts[label] += 1
                in_entity = label
            else:
                in_entity = None

    return {
        "sentence_count": len(sentences),
        "token_count": token_count,
        "entity_count": sum(entity_counts.values()),
        "entities_per_label": dict(entity_counts.most_common()),
        "label_token_count": dict(label_token_counts.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Krok 2: Konwersja do JSON
# ══════════════════════════════════════════════════════════════════════════════

def sentences_to_json(sentences: list[Sentence]) -> list[list[dict]]:
    """
    Konwertuje zdania CoNLL na format JSON wymagany przez stanza NER tagger.

    stanza.models.ner_tagger wczytuje plik jako:
        train_doc = Document(json.load(fin))

    Oczekiwany format to lista list tokenów — outer = zdania, inner = tokeny:

        [
          [{"id": "1-1", "text": "Twierdzenie", "ner": "B-THEOREM"},
           {"id": "1-2", "text": "Pitagorasa",  "ner": "I-THEOREM"}, ...],
          [{"id": "2-1", "text": "Turing", "ner": "B-AUTHOR"}, ...],
        ]

    Pola "text" i "ner" są obowiązkowe. "id" musi być stringiem.
    """
    result: list[list[dict]] = []
    for sent in sentences:
        # ID musi być prostą liczbą całkowitą (1-based w obrębie zdania).
        # Format "X-Y" (np. "1-2") jest zarezerwowany dla MWT w CoNLL-U
        # i byłby błędnie interpretowany przez stanza Document jako token wielosłowowy.
        result.append([
            {"id": j, "text": tok, "ner": tag}
            for j, (tok, tag) in enumerate(sent, 1)
        ])
    return result


def sentences_to_human_json(sentences: list[Sentence]) -> list[dict]:
    """
    Czytelna wersja JSON z polem text i entities — do podglądu i debugowania.

    NIE jest formatem treningowym stanza (do tego służy sentences_to_json).
    """
    result: list[dict] = []
    for sent_id, sent in enumerate(sentences):
        entities: list[dict] = []
        i = 0
        while i < len(sent):
            tok, tag = sent[i]
            if tag.startswith("B-"):
                label = tag[2:]
                j = i + 1
                while j < len(sent) and sent[j][1] == f"I-{label}":
                    j += 1
                entities.append({
                    "text": " ".join(t for t, _ in sent[i:j]),
                    "label": label,
                    "start_token": i + 1,
                    "end_token": j,
                })
                i = j
            else:
                i += 1

        result.append({
            "id": sent_id,
            "text": " ".join(tok for tok, _ in sent),
            "tokens": [{"id": j + 1, "text": tok, "ner": tag} for j, (tok, tag) in enumerate(sent)],
            "entities": entities,
        })
    return result


def export_split_json(sentences: list[Sentence], path: pathlib.Path) -> None:
    """
    Zapisuje dwa pliki JSON:
    - path            — format treningowy stanza (lista list tokenów)
    - path.stem_human.json — czytelna wersja z tekstem i scalonymi encjami
    """
    # Format treningowy stanza: [[{tok}, ...], ...]
    train_data = sentences_to_json(sentences)
    path.write_text(json.dumps(train_data, ensure_ascii=False), encoding="utf-8")

    # Format czytelny: [{text, tokens, entities}, ...]
    human_path = path.with_stem(path.stem + "_human")
    human_data = sentences_to_human_json(sentences)
    human_path.write_text(json.dumps(human_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  Zapisano {len(train_data)} zdań → {path.name}  (trening)")
    print(f"  Zapisano {len(human_data)} zdań → {human_path.name}  (podgląd)")


def export_stats_json(stats_by_split: dict, labels: list[str], path: pathlib.Path) -> None:
    """Zapisuje statystyki datasetu do pliku."""
    output = {
        "dataset": "ProveNuance Math NER (PL)",
        "labels": sorted(labels),
        "splits": stats_by_split,
        "total": {
            "sentence_count": sum(s["sentence_count"] for s in stats_by_split.values()),
            "token_count": sum(s["token_count"] for s in stats_by_split.values()),
            "entity_count": sum(s["entity_count"] for s in stats_by_split.values()),
        },
    }
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Zapisano statystyki → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Krok 3: Trening stanza NER
# ══════════════════════════════════════════════════════════════════════════════

def find_stanza_resource(subpath: str) -> pathlib.Path:
    """Znajduje plik zasobu stanza lub rzuca FileNotFoundError."""
    p = STANZA_RES / subpath
    if not p.exists():
        raise FileNotFoundError(
            f"Nie znaleziono zasobu stanza: {p}\n"
            f"Uruchom: python -c \"import stanza; stanza.download('pl')\""
        )
    return p


def build_training_args(
    output_dir: pathlib.Path,
    max_steps: int,
    eval_interval: int,
    batch_size: int,
    use_cpu: bool,
    finetune: bool,
) -> list[str]:
    """
    Buduje listę argumentów dla stanza.models.ner_tagger.main().

    Strategia: trening nowego modelu NER używając polskiego charlm (oscar)
    jako feature backbone. Nie importujemy etykiet z istniejącego modelu NKJP
    bo nasze labele (THEOREM, LEMMA, …) są zupełnie inne.
    """
    pretrain   = find_stanza_resource("pretrain/conll17.pt")
    fwd_charlm = find_stanza_resource("forward_charlm/oscar.pt")
    bwd_charlm = find_stanza_resource("backward_charlm/oscar.pt")

    # Stanza NER tagger czyta JSON (nie CoNLL): Document(json.load(fin))
    train_file = OUTPUT_DIR / "train.json"
    dev_file   = OUTPUT_DIR / "dev.json"

    if not train_file.exists() or not dev_file.exists():
        raise FileNotFoundError(
            "Brak plików JSON w output/math_ner/. "
            "Uruchom najpierw: python train_math_ner.py --convert-only"
        )

    model_name = "pl_math_ner.pt"

    args = [
        "--mode", "train",
        "--train_file", str(train_file),
        "--eval_file",  str(dev_file),
        "--shorthand",  "pl_math",
        # charlm_shorthand wymagany gdy --charlm (nawet jeśli podajemy ścieżki bezpośrednio)
        "--charlm_shorthand", "pl_oscar",
        # Charlm jako backbone (pretrenowane cechy)
        "--charlm",
        "--charlm_forward_file",  str(fwd_charlm),
        "--charlm_backward_file", str(bwd_charlm),
        "--wordvec_pretrain_file", str(pretrain),
        # Schemat tagowania: dane wejściowe BIO → wewnętrznie BIOES
        "--train_scheme", "BIO",
        "--scheme",       "BIOES",
        # Hiperparametry
        "--max_steps",           str(max_steps),
        "--eval_interval",       str(eval_interval),
        "--batch_size",          str(batch_size),
        "--max_steps_no_improve", str(max(200, max_steps // 4)),
        "--lr",    "0.01",
        "--dropout", "0.5",
        "--word_dropout", "0.05",
        # Zapis modelu
        "--save_dir",  str(output_dir),
        "--save_name", model_name,
        "--seed", "42",
    ]

    if use_cpu:
        args += ["--cpu"]

    if finetune:
        existing = STANZA_RES / "ner" / "nkjp.pt"
        if existing.exists():
            args += ["--finetune", "--finetune_load_name", str(existing)]
            print(f"  Fine-tuning z istniejącego modelu: {existing}")
        else:
            print("  UWAGA: --finetune podano, ale nkjp.pt nie istnieje — trening od zera")

    return args


def run_training(
    output_dir: pathlib.Path,
    max_steps: int = 2000,
    eval_interval: int = 200,
    batch_size: int = 8,
    use_cpu: bool = True,
    finetune: bool = False,
) -> pathlib.Path:
    """
    Uruchamia trening stanza NER.

    Zwraca ścieżkę do wytrenowanego modelu.
    """
    from stanza.models.ner_tagger import main as ner_main

    output_dir.mkdir(parents=True, exist_ok=True)

    args = build_training_args(
        output_dir=output_dir,
        max_steps=max_steps,
        eval_interval=eval_interval,
        batch_size=batch_size,
        use_cpu=use_cpu,
        finetune=finetune,
    )

    print()
    print("  Argumenty treningu:")
    for i in range(0, len(args), 2):
        if i + 1 < len(args):
            print(f"    {args[i]:<32} {args[i+1]}")
        else:
            print(f"    {args[i]}")
    print()

    ner_main(args)

    model_path = output_dir / "pl_math_ner.pt"
    return model_path


# ══════════════════════════════════════════════════════════════════════════════
# Krok 4: Ewaluacja
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(model_path: pathlib.Path, output_dir: pathlib.Path) -> dict:
    """
    Uruchamia ewaluację na test.conll i zwraca wyniki per etykieta.

    Używa stanza NER tagger w trybie predict na pliku testowym.
    """
    from stanza.models.ner_tagger import main as ner_main

    pretrain   = find_stanza_resource("pretrain/conll17.pt")
    fwd_charlm = find_stanza_resource("forward_charlm/oscar.pt")
    bwd_charlm = find_stanza_resource("backward_charlm/oscar.pt")

    # Stanza NER tagger w trybie predict też czyta JSON
    test_json  = OUTPUT_DIR / "test.json"
    gold_conll = DATASET_DIR / "test.conll"
    eval_output = output_dir / "test_predictions.conll"

    if not test_json.exists():
        raise FileNotFoundError(f"Brak {test_json}. Uruchom --convert-only najpierw.")

    args = [
        "--mode", "predict",
        "--eval_file", str(test_json),
        "--eval_output_file", str(eval_output),
        "--shorthand", "pl_math",
        "--charlm_shorthand", "pl_oscar",
        "--charlm",
        "--charlm_forward_file",  str(fwd_charlm),
        "--charlm_backward_file", str(bwd_charlm),
        "--wordvec_pretrain_file", str(pretrain),
        "--scheme", "BIOES",
        "--save_dir",  str(output_dir),
        "--save_name", model_path.name,
        "--cpu",
    ]

    ner_main(args)

    # F1 liczymy ze złotych CoNLL vs predykowanych CoNLL (stanza zapisuje CoNLL)
    results = _compute_f1(gold_conll, eval_output)

    results_path = output_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Wyniki ewaluacji → {results_path}")
    return results


def _compute_f1(gold_file: pathlib.Path, pred_file: pathlib.Path) -> dict:
    """Oblicza precision/recall/F1 per label (entity-level) ze złotych vs predykowanych."""
    gold_sents = read_conll(gold_file)
    pred_sents = read_conll(pred_file)

    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()

    for gold_sent, pred_sent in zip(gold_sents, pred_sents):
        gold_spans = _extract_spans(gold_sent)
        pred_spans = _extract_spans(pred_sent)
        all_labels = {label for _, label in gold_spans | pred_spans}
        for label in all_labels:
            g = {span for span, lbl in gold_spans if lbl == label}
            p = {span for span, lbl in pred_spans if lbl == label}
            tp[label] += len(g & p)
            fp[label] += len(p - g)
            fn[label] += len(g - p)

    results: dict = {"per_label": {}}
    total_tp = total_fp = total_fn = 0

    for label in sorted(set(list(tp) + list(fp) + list(fn))):
        t, f_p, f_n = tp[label], fp[label], fn[label]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec  = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results["per_label"][label] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   t + f_n,
        }
        total_tp += t; total_fp += f_p; total_fn += f_n

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    results["micro"] = {
        "precision": round(micro_prec, 4),
        "recall":    round(micro_rec, 4),
        "f1":        round(micro_f1, 4),
    }
    return results


def _extract_spans(sent: Sentence) -> set[tuple[tuple[int, ...], str]]:
    """Wyciąga spany encji jako (tuple indeksów tokenów, etykieta)."""
    spans: set[tuple[tuple[int, ...], str]] = set()
    i = 0
    while i < len(sent):
        tok, tag = sent[i]
        if tag.startswith("B-"):
            label = tag[2:]
            indices = [i]
            j = i + 1
            while j < len(sent) and sent[j][1] == f"I-{label}":
                indices.append(j)
                j += 1
            spans.add((tuple(indices), label))
            i = j
        else:
            i += 1
    return spans


def print_eval_results(results: dict) -> None:
    """Drukuje wyniki ewaluacji jako tabelę."""
    print()
    print(f"  {'LABEL':<20} {'PREC':>6} {'REC':>6} {'F1':>6} {'SUPP':>6}")
    print("  " + "-" * 46)
    for label, m in sorted(results["per_label"].items(), key=lambda x: -x[1]["f1"]):
        print(
            f"  {label:<20} {m['precision']:>6.3f} {m['recall']:>6.3f}"
            f" {m['f1']:>6.3f} {m['support']:>6}"
        )
    print("  " + "-" * 46)
    m = results["micro"]
    print(
        f"  {'MICRO'::<20} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Krok 5: Instrukcja użycia modelu
# ══════════════════════════════════════════════════════════════════════════════

def print_usage_example(model_path: pathlib.Path) -> None:
    abs_path = model_path.resolve()
    print()
    print("=" * 70)
    print("  Model gotowy. Jak go uzywac w StanzaEntityLinker:")
    print("=" * 70)
    print()
    print("  from adapters.entity_linker.stanza_linker import StanzaEntityLinker, StanzaConfig")
    print()
    print("  linker = StanzaEntityLinker(")
    print("      config=StanzaConfig(")
    print("          lang='pl',")
    print("          processors='tokenize,pos,lemma,depparse,ner',")
    print(f"          # Nadpisz model NER:")
    print(f"          # (podaj przez stanza.Pipeline(..., ner_model_path=...))")
    print("      )")
    print("  )")
    print()
    print("  # Lub bezposrednio przez stanza.Pipeline:")
    print("  import stanza")
    print("  nlp = stanza.Pipeline(")
    print("      lang='pl',")
    print("      processors='tokenize,ner',")
    print(f"      ner_model_path=r'{abs_path}',")
    print("      ner_charlm_forward_file=r'" + str(find_stanza_resource('forward_charlm/oscar.pt')) + "',")
    print("      ner_charlm_backward_file=r'" + str(find_stanza_resource('backward_charlm/oscar.pt')) + "',")
    print("  )")
    print("  doc = nlp('Twierdzenie Pitagorasa zostalo udowodnione przez Eulera.')")
    print("  for ent in doc.ents:")
    print("      print(ent.text, ent.type)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Konwertuje BIO CoNLL do JSON i trenuje stanza NER dla matematycznych NE (PL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir",    type=pathlib.Path, default=DATASET_DIR,
                   help="Katalog z plikami train/dev/test.conll")
    p.add_argument("--output-dir",     type=pathlib.Path, default=OUTPUT_DIR,
                   help="Katalog wyjściowy dla JSON i modelu")
    p.add_argument("--steps",          type=int, default=2000,
                   help="Liczba kroków treningu (max_steps)")
    p.add_argument("--eval-interval",  type=int, default=200,
                   help="Co ile kroków ewaluować na dev")
    p.add_argument("--batch-size",     type=int, default=8,
                   help="Rozmiar batcha")
    p.add_argument("--cuda",           action="store_true",
                   help="Użyj GPU (domyślnie CPU)")
    p.add_argument("--finetune",       action="store_true",
                   help="Fine-tune istniejącego modelu pl/ner/nkjp.pt (eksperymentalnie)")
    p.add_argument("--convert-only",   action="store_true",
                   help="Tylko konwersja do JSON, pomiń trening")
    p.add_argument("--eval-only",      action="store_true",
                   help="Tylko ewaluacja istniejącego modelu (pomiń trening)")
    p.add_argument("--skip-eval",      action="store_true",
                   help="Pomiń ewaluację po treningu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Krok 1: Wczytaj i zwaliduj ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  [1/4] Wczytywanie i walidacja danych CoNLL")
    print("=" * 70)

    all_sentences: dict[str, list[Sentence]] = {}
    all_labels: set[str] = set()
    had_errors = False

    for split in SPLITS:
        conll_path = args.dataset_dir / f"{split}.conll"
        if not conll_path.exists():
            print(f"  BLAD: brak pliku {conll_path}")
            sys.exit(1)

        sents = read_conll(conll_path)
        errors = validate_bio(sents, conll_path.name)

        # Zbierz etykiety
        for sent in sents:
            for _, tag in sent:
                if "-" in tag and not tag.startswith("I-"):
                    all_labels.add(tag.split("-", 1)[1])

        stats = compute_stats(sents)
        print(
            f"  {split:5s}: {stats['sentence_count']:4d} zdań "
            f"| {stats['token_count']:6d} tokenów "
            f"| {stats['entity_count']:4d} encji"
        )
        if errors:
            for e in errors:
                print(f"    WARN: {e}")
            had_errors = True

        all_sentences[split] = sents

    print(f"  Etykiety ({len(all_labels)}): {', '.join(sorted(all_labels))}")

    if had_errors:
        print("  UWAGA: znaleziono błędy walidacji — kontynuuję mimo to.")

    # ── Krok 2: Konwersja do JSON ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  [2/4] Konwersja do JSON")
    print("=" * 70)

    stats_by_split: dict[str, dict] = {}
    for split, sents in all_sentences.items():
        json_path = output_dir / f"{split}.json"
        export_split_json(sents, json_path)
        stats_by_split[split] = compute_stats(sents)

    export_stats_json(stats_by_split, list(all_labels), output_dir / "stats.json")

    if args.convert_only:
        print()
        print("  --convert-only: zakończono. Pliki JSON w:", output_dir)
        return

    # ── Krok 3: Trening ───────────────────────────────────────────────────────
    model_path = output_dir / "pl_math_ner.pt"

    if not args.eval_only:
        print()
        print("=" * 70)
        print("  [3/4] Trening stanza NER")
        print("=" * 70)
        print(f"  Backbone: charlm oscar (PL) + pretrain conll17")
        print(f"  Kroki: {args.steps}  |  Eval co: {args.eval_interval}  |  Batch: {args.batch_size}")
        print(f"  Urzadzenie: {'GPU (CUDA)' if args.cuda else 'CPU'}")
        print()

        model_path = run_training(
            output_dir=output_dir,
            max_steps=args.steps,
            eval_interval=args.eval_interval,
            batch_size=args.batch_size,
            use_cpu=not args.cuda,
            finetune=args.finetune,
        )
        print()
        print(f"  Model zapisany: {model_path}")
    else:
        print()
        print("  [3/4] --eval-only: pomijam trening")
        if not model_path.exists():
            print(f"  BLAD: nie znaleziono modelu {model_path}")
            print("  Uruchom bez --eval-only żeby wytrenować model.")
            sys.exit(1)

    # ── Krok 4: Ewaluacja ─────────────────────────────────────────────────────
    if not args.skip_eval:
        print()
        print("=" * 70)
        print("  [4/4] Ewaluacja na test.conll")
        print("=" * 70)

        try:
            results = run_eval(model_path, output_dir)
            print_eval_results(results)
        except Exception as exc:
            print(f"  BLAD ewaluacji: {exc}")
            print("  Pomiń ewaluację z --skip-eval jeśli model nie jest jeszcze dostępny.")
    else:
        print()
        print("  [4/4] --skip-eval: pomijam ewaluację")

    # ── Instrukcja użycia ─────────────────────────────────────────────────────
    print_usage_example(model_path)


if __name__ == "__main__":
    main()
