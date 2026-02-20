#!/usr/bin/env python3
"""
provenuance.py — CLI narzędzie ProveNuance.

Działa całkowicie lokalnie — łączy się bezpośrednio z PostgreSQL,
nie wymaga uruchomionego serwera API.

Konfiguracja DB: zmienne środowiskowe z prefiksem PROVE_NUANCE_
lub plik .env (np. PROVE_NUANCE_DB_URL=postgresql://...).

Podkomendy:
    ingest   — wyślij dokument do DocumentStore
    extract  — ekstrahuj ramki z dokumentu
    promote  — promuj fakty/reguły do 'asserted'
    facts    — listuj fakty z KnowledgeStore
    rules    — listuj reguły z KnowledgeStore
    ner      — uruchom NER na tekście (wymaga backendu NER)
    health   — sprawdź połączenie z bazą danych
    reset    — usuń dokument z DocumentStore
    run      — lokalnie ekstrahuj ramki z tekstu (bez DB)

Użycie:
    python provenuance.py ingest --title "Dodawanie" --text "3 + 4 = 7."
    python provenuance.py ingest --title "Rozdział 1" --file doc.md --source-type markdown
    python provenuance.py extract --verbose 52e50486-04eb-4e5d-867f-1bb8440643f7
    python provenuance.py promote --facts abc123 def456
    python provenuance.py facts --status all
    python provenuance.py rules
    python provenuance.py ner --text "Addition is commutative"
    python provenuance.py health
    python provenuance.py reset 52e50486-04eb-4e5d-867f-1bb8440643f7
    python provenuance.py run --text "3 + 4 = 7"
"""
from __future__ import annotations

import argparse
import asyncio
import sys


# ── helpers ───────────────────────────────────────────────────────────────────

def _dsn() -> str:
    from config import Settings
    return Settings().db_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "file", None):
        try:
            return open(args.file, encoding="utf-8").read()
        except OSError as e:
            print(f"Błąd odczytu pliku: {e}", file=sys.stderr)
            sys.exit(1)
    text = getattr(args, "text", None) or sys.stdin.read().strip()
    if not text:
        print("Błąd: podaj tekst przez --text, --file lub stdin", file=sys.stderr)
        sys.exit(1)
    return text


# ── podkomendy async ──────────────────────────────────────────────────────────

async def _ingest(args: argparse.Namespace) -> None:
    from adapters.document_store.postgres_document_store import PostgresDocumentStore
    from contracts import DocumentIn

    text = _read_text(args)
    store = await PostgresDocumentStore.create(_dsn())
    try:
        doc_ref = await store.ingest_document(DocumentIn(
            title=args.title,
            raw_text=text,
            source_type=args.source_type,
        ))
        spans = await store.list_spans(doc_ref.doc_id)
    finally:
        await store.close()

    print(f"doc_id:     {doc_ref.doc_id}")
    print(f"title:      {doc_ref.title}")
    print(f"version:    {doc_ref.version}")
    print(f"span_count: {len(spans)}")


async def _extract(args: argparse.Namespace) -> None:
    from adapters.document_store.postgres_document_store import PostgresDocumentStore
    from adapters.frame_extractor._printer import print_frames
    from adapters.frame_extractor.rule_based_extractor import RuleBasedExtractor
    from adapters.frame_mapper.math_grade1to3_mapper import MathGrade1to3Mapper
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore
    from contracts import Provenance

    dsn = _dsn()
    doc_store = await PostgresDocumentStore.create(dsn)
    kn_store  = await PostgresKnowledgeStore.create(dsn)
    try:
        spans = await doc_store.list_spans(args.doc_id)
        if not spans:
            print(f"Brak spanów dla dokumentu: {args.doc_id}", file=sys.stderr)
            sys.exit(1)

        extractor = RuleBasedExtractor()
        mapper    = MathGrade1to3Mapper()
        total_frames = total_facts = total_rules = 0

        for span in spans:
            frames = extractor.extract_frames(span)
            if args.verbose:
                print_frames(span.surface_text, frames)

            for frame in frames:
                issues = extractor.validate_frame(frame)
                if any(i.severity == "error" for i in issues):
                    continue
                mapping = mapper.map(frame, Provenance(
                    span_ids=[span.span_id], doc_ref=args.doc_id
                ))
                for fact in mapping.facts:
                    await kn_store.upsert_fact(fact)
                    total_facts += 1
                    if args.auto_promote:
                        await kn_store.promote_fact(fact.fact_id, reason="auto_promote")
                for rule in mapping.rules:
                    await kn_store.upsert_rule(rule)
                    total_rules += 1
                    if args.auto_promote:
                        await kn_store.promote_rule(rule.rule_id, reason="auto_promote")
            total_frames += len(frames)
    finally:
        await doc_store.close()
        await kn_store.close()

    print(f"Dokument:  {args.doc_id}")
    print(f"Spany:     {len(spans)}")
    print(f"Ramki:     {total_frames}")
    print(f"Fakty:     {total_facts}")
    print(f"Reguły:    {total_rules}")


async def _promote(args: argparse.Namespace) -> None:
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore

    store = await PostgresKnowledgeStore.create(_dsn())
    promoted_facts = promoted_rules = 0
    errors: list[str] = []
    try:
        for fid in (args.facts or []):
            try:
                await store.promote_fact(fid, reason=args.reason)
                promoted_facts += 1
            except KeyError as e:
                errors.append(str(e))
        for rid in (args.rules or []):
            try:
                await store.promote_rule(rid, reason=args.reason)
                promoted_rules += 1
            except KeyError as e:
                errors.append(str(e))
    finally:
        await store.close()

    print(f"Promowane fakty:  {promoted_facts}")
    print(f"Promowane reguły: {promoted_rules}")
    for e in errors:
        print(f"  BŁĄD: {e}", file=sys.stderr)


async def _facts(args: argparse.Namespace) -> None:
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore

    store = await PostgresKnowledgeStore.create(_dsn())
    try:
        facts = await store.list_facts(status=args.status, limit=args.limit)
    finally:
        await store.close()

    if not facts:
        print("Brak faktów.")
        return
    print(f"Fakty ({args.status}): {len(facts)}")
    for f in facts:
        print(f"  {f.fact_id[:8]}…  {f.h} —[{f.r}]→ {f.t}"
              f"  [{f.status.value}]  conf={f.confidence:.2f}")


async def _rules(args: argparse.Namespace) -> None:
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore

    store = await PostgresKnowledgeStore.create(_dsn())
    try:
        rules = await store.list_rules(status=args.status, limit=args.limit)
    finally:
        await store.close()

    if not rules:
        print("Brak reguł.")
        return
    print(f"Reguły ({args.status}): {len(rules)}")
    for r in rules:
        body = ", ".join(r.body) if r.body else "∅"
        print(f"  {r.rule_id[:8]}…  {r.head} :- {body}"
              f"  [{r.status.value}]  prio={r.priority}")


async def _ner(args: argparse.Namespace) -> None:
    from adapters.frame_extractor._printer import print_frames
    from adapters.frame_extractor.llm_extractor import LLMFrameExtractor
    from config import Settings

    text = args.text or sys.stdin.read().strip()
    if not text:
        print("Błąd: podaj tekst przez --text lub stdin", file=sys.stderr)
        sys.exit(1)

    s = Settings()
    extractor = LLMFrameExtractor(ner_backend_url=s.ner_backend_url, timeout_ms=s.ner_timeout_ms)
    try:
        output = extractor.call_backend(text)
    except Exception as exc:
        print(f"Błąd backendu NER: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Encje ({len(output.entities)}):")
    for e in output.entities:
        print(f"  [{e.type}] {e.text!r} → {e.canonical}  conf={e.confidence:.2f}")
    frames = extractor.map_frames(output.frames, "ner-cli")
    if frames:
        print_frames(text, frames)


async def _health(args: argparse.Namespace) -> None:
    import asyncpg
    dsn = _dsn()
    try:
        conn = await asyncpg.connect(dsn, timeout=5)
        await conn.fetchval("SELECT 1")
        await conn.close()
        host_db = dsn.split("@")[-1]
        print("status:  ok")
        print(f"db:      {host_db}")
    except Exception as exc:
        print("status:  error", file=sys.stderr)
        print(f"db:      {exc}", file=sys.stderr)
        sys.exit(1)


async def _reset(args: argparse.Namespace) -> None:
    from adapters.document_store.postgres_document_store import PostgresDocumentStore

    store = await PostgresDocumentStore.create(_dsn())
    try:
        await store.delete_document(args.doc_id)
    except KeyError:
        print(f"Dokument nie znaleziony: {args.doc_id}", file=sys.stderr)
        sys.exit(1)
    finally:
        await store.close()

    print(f"Dokument {args.doc_id} usunięty.")


# ── podkomenda sync ───────────────────────────────────────────────────────────

def _run(args: argparse.Namespace) -> None:
    from adapters.frame_extractor.rule_based_extractor import RuleBasedExtractor
    from contracts import DocSpan

    text = args.text or sys.stdin.read().strip()
    if not text:
        print("Błąd: podaj tekst przez --text lub stdin", file=sys.stderr)
        sys.exit(1)

    extractor = RuleBasedExtractor(verbose=not args.quiet)
    span = DocSpan(doc_id="cli", version=0, surface_text=text,
                   start_char=0, end_char=len(text))
    frames = extractor.extract_frames(span)
    print(f"Znaleziono {len(frames)} ramek.")


# ── podkomendy stub ───────────────────────────────────────────────────────────

def _solve(args: argparse.Namespace) -> None:
    text = args.text or sys.stdin.read().strip()
    if not text:
        print("Błąd: podaj tekst przez --text lub stdin", file=sys.stderr)
        sys.exit(1)
    print("solve: nie zaimplementowane (wymaga Etap 3+4)")
    print(f"  zadanie: {text[:120]}")


def _prove(args: argparse.Namespace) -> None:
    print("prove: nie zaimplementowane (wymaga Etap 4)")
    print(f"  proof_id: {args.proof_id}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="provenuance",
        description="ProveNuance — CLI (lokalny, bez serwera API)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p = sub.add_parser("ingest", help="Wyślij dokument do DocumentStore")
    p.add_argument("--title", "-T", required=True, help="Tytuł dokumentu")
    p.add_argument("--text", "-t", help="Treść dokumentu")
    p.add_argument("--file", "-f", help="Ścieżka do pliku z treścią")
    p.add_argument("--source-type", default="text",
                   choices=["text", "markdown", "pdf"])

    # extract
    p = sub.add_parser("extract", help="Ekstrahuj ramki z dokumentu")
    p.add_argument("doc_id", help="UUID dokumentu")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Wyświetl znalezione ramki")
    p.add_argument("--auto-promote", action="store_true",
                   help="Promuj fakty/reguły od razu")

    # promote
    p = sub.add_parser("promote", help="Promuj fakty i/lub reguły do 'asserted'")
    p.add_argument("--facts", nargs="*", default=[], metavar="ID")
    p.add_argument("--rules", nargs="*", default=[], metavar="ID")
    p.add_argument("--reason", default="manual promotion")

    # facts
    p = sub.add_parser("facts", help="Listuj fakty z KnowledgeStore")
    p.add_argument("--status", default="asserted",
                   choices=["hypothesis", "asserted", "retracted", "all"])
    p.add_argument("--limit", type=int, default=200, metavar="N")

    # rules
    p = sub.add_parser("rules", help="Listuj reguły z KnowledgeStore")
    p.add_argument("--status", default="asserted",
                   choices=["hypothesis", "asserted", "all"])
    p.add_argument("--limit", type=int, default=200, metavar="N")

    # ner
    p = sub.add_parser("ner", help="Uruchom NER na tekście (wymaga backendu NER)")
    p.add_argument("--text", "-t", help="Tekst do analizy (lub stdin)")

    # solve
    p = sub.add_parser("solve", help="Rozwiąż zadanie matematyczne [stub – Etap 3+4]")
    p.add_argument("--text", "-t", help="Treść zadania (lub stdin)")

    # prove
    p = sub.add_parser("prove", help="Pobierz dowód [stub – Etap 4]")
    p.add_argument("proof_id", help="UUID dowodu")

    # health
    sub.add_parser("health", help="Sprawdź połączenie z bazą danych")

    # reset
    p = sub.add_parser("reset", help="Usuń dokument z DocumentStore (nieodwracalne)")
    p.add_argument("doc_id", help="UUID dokumentu")

    # run
    p = sub.add_parser("run", help="Lokalnie ekstrahuj ramki z tekstu (bez DB)")
    p.add_argument("--text", "-t", help="Tekst do analizy (lub stdin)")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Tylko liczba ramek, bez szczegółów")

    args = parser.parse_args()

    async_cmds = {
        "ingest":  _ingest,
        "extract": _extract,
        "promote": _promote,
        "facts":   _facts,
        "rules":   _rules,
        "ner":     _ner,
        "health":  _health,
        "reset":   _reset,
    }

    if args.command in async_cmds:
        asyncio.run(async_cmds[args.command](args))
    elif args.command == "run":
        _run(args)


if __name__ == "__main__":
    main()
