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
    reset    — usun wszystkie dane z KnowledgeStore
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
    python provenuance.py reset
    python provenuance.py run --text "3 + 4 = 7"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table


# -- helpers ---------------------------------------------------------------

_CONSOLE: Console | None = None


def _console() -> Console:
    global _CONSOLE
    if _CONSOLE is None:
        _CONSOLE = Console(highlight=False)
    return _CONSOLE


def _safe_terminal_text(value: Any) -> str:
    s = str(value)
    s = (
        s.replace("→", "->")
        .replace("—", "-")
        .replace("…", "...")
        .replace("∅", "EMPTY")
        .replace("×", "x")
        .replace("÷", "/")
    )
    encoding = sys.stdout.encoding or "utf-8"
    try:
        s.encode(encoding)
        return s
    except UnicodeEncodeError:
        return s.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _short(value: Any, limit: int = 64) -> str:
    s = _safe_terminal_text(value).replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


def _print_kv_table(title: str, rows: list[tuple[str, Any]]) -> None:
    table = Table(title=title, box=box.ASCII, show_header=False, pad_edge=False)
    table.add_column("Key", no_wrap=True, style="bold cyan")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(_safe_terminal_text(key), _safe_terminal_text(value))
    _console().print(table)


def _print_facts_table(status: str, facts: list[Any]) -> None:
    table = Table(
        title=f"Facts ({status}) [{len(facts)}]",
        box=box.ASCII,
        show_lines=False,
    )
    table.add_column("ID", no_wrap=True, style="cyan")
    table.add_column("H")
    table.add_column("R", no_wrap=True)
    table.add_column("T")
    table.add_column("Status", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    for fact in facts:
        table.add_row(
            _safe_terminal_text(str(fact.fact_id)[:8]),
            _short(fact.h, 44),
            _safe_terminal_text(fact.r),
            _short(fact.t, 44),
            _safe_terminal_text(fact.status.value),
            f"{fact.confidence:.2f}",
        )
    _console().print(table)


def _print_rules_table(status: str, rules: list[Any]) -> None:
    table = Table(
        title=f"Rules ({status}) [{len(rules)}]",
        box=box.ASCII,
        show_lines=False,
    )
    table.add_column("ID", no_wrap=True, style="cyan")
    table.add_column("Head")
    table.add_column("Body")
    table.add_column("Status", no_wrap=True)
    table.add_column("Prio", justify="right", no_wrap=True)
    for rule in rules:
        body = ", ".join(rule.body) if rule.body else "EMPTY"
        table.add_row(
            _safe_terminal_text(str(rule.rule_id)[:8]),
            _short(rule.head, 48),
            _short(body, 56),
            _safe_terminal_text(rule.status.value),
            _safe_terminal_text(rule.priority),
        )
    _console().print(table)


def _print_bootstrap_candidates_table(candidates: list[Any], show: int) -> None:
    table = Table(
        title=f"Top {show} bootstrap candidates",
        box=box.ASCII,
        show_lines=False,
    )
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Rule")
    table.add_column("Support", justify="right", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Lift", justify="right", no_wrap=True)
    table.add_column("Viol", justify="right", no_wrap=True)
    table.add_column("Corr", justify="right", no_wrap=True)
    table.add_column("CWA", justify="right", no_wrap=True)
    table.add_column("OK", justify="center", no_wrap=True)
    table.add_column("Rule ID", no_wrap=True)
    for idx, cand in enumerate(candidates[:show], 1):
        body = ", ".join(cand.body)
        rule = f"{cand.head} :- {body}"
        lift = "inf" if cand.lift == float("inf") else f"{cand.lift:.3f}"
        table.add_row(
            str(idx),
            _short(rule, 78),
            f"{cand.support}/{cand.coverage}",
            f"{cand.confidence:.3f}",
            lift,
            str(cand.violations),
            str(cand.corruption_hits),
            str(cand.local_cwa_negatives),
            "yes" if cand.promotable else "no",
            _safe_terminal_text(cand.rule_id or "-"),
        )
    _console().print(table)

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


def _proof_store_path() -> Path:
    return Path("data") / "cli_proofs.json"


def _load_proofs() -> dict[str, Any]:
    path = _proof_store_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Blad odczytu magazynu dowodow: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, dict):
        print("Blad: magazyn dowodow ma niepoprawny format.", file=sys.stderr)
        sys.exit(1)
    return data


def _save_proof(record: dict[str, Any], proof_id: str | None = None) -> str:
    path = _proof_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    all_proofs = _load_proofs()

    pid = proof_id or str(record.get("proof_id") or uuid.uuid4())
    record["proof_id"] = pid
    all_proofs[pid] = record

    try:
        path.write_text(
            json.dumps(all_proofs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Blad zapisu magazynu dowodow: {exc}", file=sys.stderr)
        sys.exit(1)
    return pid


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _normalize_operation(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    op = value.strip().lower()
    mapping = {
        "add": "add",
        "+": "add",
        "plus": "add",
        "sub": "sub",
        "-": "sub",
        "minus": "sub",
        "subtract": "sub",
        "mul": "mul",
        "*": "mul",
        "x": "mul",
        "×": "mul",
        "times": "mul",
        "div": "div",
        "/": "div",
        "÷": "div",
        "divide": "div",
    }
    return mapping.get(op)


def _extract_timeline_plan(
    slots: dict[str, Any],
    default_op: str | None,
) -> tuple[int, list[tuple[str, int]]] | None:
    initial = _coerce_int(slots.get("initial"))
    events_raw = slots.get("events")
    if initial is None or not isinstance(events_raw, list) or not events_raw:
        return None

    normalized_events: list[tuple[str, int]] = []
    for event in events_raw:
        if not isinstance(event, dict):
            return None
        op = _normalize_operation(event.get("op") or event.get("operation") or default_op)
        value = _coerce_int(event.get("value"))
        if value is None:
            value = _coerce_int(event.get("delta"))
        if value is None:
            value = _coerce_int(event.get("amount"))
        if value is None:
            value = _coerce_int(event.get("b"))
        if op is None or value is None:
            return None
        normalized_events.append((op, value))

    return initial, normalized_events


def _extract_expr_chain_plan(expr_ast: Any) -> tuple[int, list[tuple[str, int]]] | None:
    def _number_value(node: Any) -> int | None:
        if node is None:
            return None
        node_type = getattr(node, "node_type", None)
        if node_type == "number":
            return _coerce_int(getattr(node, "value", None))
        if node_type == "unary" and getattr(node, "op", None) == "-":
            inner = _number_value(getattr(node, "operand", None))
            return None if inner is None else -inner
        return None

    def _walk(node: Any) -> tuple[int, list[tuple[str, int]]] | None:
        if node is None:
            return None

        number = _number_value(node)
        if number is not None:
            return number, []

        if getattr(node, "node_type", None) != "binop":
            return None

        left_plan = _walk(getattr(node, "left", None))
        if left_plan is None:
            return None
        right_number = _number_value(getattr(node, "right", None))
        op = _normalize_operation(getattr(node, "op", None))
        if right_number is None or op is None:
            return None

        initial, events = left_plan
        return initial, [*events, (op, right_number)]

    plan = _walk(expr_ast)
    if plan is None:
        return None
    initial, events = plan
    if not events:
        return None
    return initial, events


def _extract_flat_expr_chain_plan(
    operation_hint: Any,
    numbers: list[Any],
) -> tuple[int, list[tuple[str, int]]] | None:
    op = _normalize_operation(operation_hint)
    if op is None or len(numbers) < 3:
        return None

    ints: list[int] = []
    for raw in numbers:
        value = _coerce_int(raw)
        if value is None:
            return None
        ints.append(value)

    return ints[0], [(op, v) for v in ints[1:]]


def _first_binding(answer: dict[str, str], preferred_var: str = "Z") -> str | None:
    for key in (preferred_var, f"?{preferred_var}"):
        value = answer.get(key)
        if value is not None:
            return str(value)
    if not answer:
        return None
    first_value = next(iter(answer.values()))
    return str(first_value) if first_value is not None else None


# -- podkomendy async ------------------------------------------------------

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
    from adapters.entity_linker.stanza_linker import StanzaConfig, StanzaEntityLinker
    from adapters.frame_extractor._printer import print_frames
    from adapters.frame_extractor.stanza_aware_extractor import StanzaAwareExtractor
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

        stanza_linker = StanzaEntityLinker(
            config=StanzaConfig(lang="pl", processors="tokenize,pos,lemma,depparse,ner")
        )
        extractor = StanzaAwareExtractor(linker=stanza_linker)
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
                    fact_ref = await kn_store.upsert_fact(fact)
                    total_facts += 1
                    if args.auto_promote:
                        await kn_store.promote_fact(fact_ref.fact_id, reason="auto_promote")
                for rule in mapping.rules:
                    rule_ref = await kn_store.upsert_rule(rule)
                    total_rules += 1
                    if args.auto_promote:
                        await kn_store.promote_rule(rule_ref.rule_id, reason="auto_promote")
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
        print("Brak faktow.")
        return
    _print_facts_table(args.status, facts)


async def _rules(args: argparse.Namespace) -> None:
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore

    store = await PostgresKnowledgeStore.create(_dsn())
    try:
        rules = await store.list_rules(status=args.status, limit=args.limit)
    finally:
        await store.close()

    if not rules:
        print("Brak regul.")
        return
    _print_rules_table(args.status, rules)


async def _bootstrap_rules(args: argparse.Namespace) -> None:
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore
    from adapters.rule_bootstrap import RuleBootstrapper
    from ports.rule_bootstrap import BootstrapConfig, DEFAULT_RELATION_WHITELIST

    relations = set(args.relations) if args.relations else set(DEFAULT_RELATION_WHITELIST)
    functional = set(args.functional_relations or [])

    cfg = BootstrapConfig(
        relation_whitelist=relations,
        max_body_literals=args.max_body,
        min_coverage=args.min_coverage,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_violations=args.max_violations,
        max_corruption_hits=args.max_corruption_hits,
        max_local_cwa_negatives=args.max_local_cwa_negatives,
        corruption_samples=args.corruption_samples,
        use_local_cwa=not args.no_local_cwa,
        functional_relations=functional,
        max_groundings_per_pattern=args.max_groundings_per_pattern,
        top_k=args.top_k,
        priority=args.priority,
        store_candidates=not args.no_store_candidates,
        promote=not args.no_promote,
        promotion_reason=args.reason,
        dry_run=args.dry_run,
    )

    store = await PostgresKnowledgeStore.create(_dsn())
    try:
        bootstrapper = RuleBootstrapper(cfg)
        summary = await bootstrapper.bootstrap(store)
    finally:
        await store.close()

    summary_rows: list[tuple[str, Any]] = [
        ("asserted_facts", summary.asserted_fact_count),
        ("relations", summary.relation_count),
        ("patterns", summary.pattern_count),
        ("candidates", summary.candidate_count),
        ("stored_new", summary.stored_created),
        ("stored_update", summary.stored_updated),
        ("promoted", summary.promoted),
    ]
    if args.dry_run:
        summary_rows.append(("mode", "dry-run (no DB writes)"))
    _print_kv_table("Rule Bootstrap Summary", summary_rows)

    if not summary.candidates:
        print("Brak kandydatow reguly.")
        return

    show = min(args.show, len(summary.candidates))
    _print_bootstrap_candidates_table(summary.candidates, show)


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
    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore

    store = await PostgresKnowledgeStore.create(_dsn())
    try:
        counts = await store.reset_all()
    finally:
        await store.close()

    print("KnowledgeStore wyczyszczony.")
    print(f"  fakty:        {counts['facts']}")
    print(f"  reguly:       {counts['rules']}")
    print(f"  snapshoty:    {counts['snapshots']}")
    print(f"  fact_history: {counts['fact_history']}")
    print(f"  rule_history: {counts['rule_history']}")

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


# -- podkomendy stub -------------------------------------------------------



# -- main ------------------------------------------------------------------

async def _solve(args: argparse.Namespace) -> None:
    from adapters.math_problem_parser.llm_parser import LLMMathParser
    from adapters.math_problem_parser.regex_parser import RegexMathParser
    from contracts import LogicQuery, ProblemType
    from config import Settings

    text = args.text or sys.stdin.read().strip()
    if not text:
        print("Blad: podaj tekst przez --text lub stdin", file=sys.stderr)
        sys.exit(1)

    settings = Settings()
    parser = LLMMathParser(
        backend_url=settings.math_parser_backend_url,
        timeout_ms=settings.math_parser_timeout_ms,
        fallback_parser=RegexMathParser(),
    )
    parsed = parser.parse(text)

    if parsed.problem_type == ProblemType.UNKNOWN:
        print("Blad: nie udalo sie sparsowac zadania.", file=sys.stderr)
        sys.exit(1)

    timeline_plan: tuple[int, list[tuple[str, int]]] | None = None
    if parsed.problem_type == ProblemType.WORD_PROBLEM:
        timeline_plan = _extract_timeline_plan(parsed.slots, parsed.operation_hint)
    elif parsed.problem_type == ProblemType.EXPR:
        timeline_plan = _extract_expr_chain_plan(parsed.expr_ast)
        if timeline_plan is None:
            regex_parsed = RegexMathParser().parse(text)
            if regex_parsed.problem_type == ProblemType.EXPR:
                timeline_plan = _extract_expr_chain_plan(regex_parsed.expr_ast)
                if timeline_plan is None:
                    timeline_plan = _extract_flat_expr_chain_plan(
                        regex_parsed.operation_hint,
                        regex_parsed.extracted_numbers,
                    )
        if timeline_plan is None:
            timeline_plan = _extract_flat_expr_chain_plan(
                parsed.operation_hint,
                parsed.extracted_numbers,
            )

    query = parsed.logic_query
    if parsed.problem_type == ProblemType.EXPR and query is None and timeline_plan is None:
        # W trybie KB-only wyrażenie musi dać się sprowadzić do jednego atomu op(a,b,?Z).
        op = parsed.operation_hint
        nums = parsed.extracted_numbers
        if op and len(nums) >= 2:
            query = LogicQuery(goal=f"{op}({nums[0]},{nums[1]},?Z)", variables=["Z"])
        else:
            print(
                "Blad: wyrazenie nieobslugiwane w trybie KB-only "
                "(wymagany prosty format z 2 liczbami).",
                file=sys.stderr,
            )
            sys.exit(1)

    if timeline_plan is None and query is None:
        print("Blad: brak logic_query dla zadania.", file=sys.stderr)
        sys.exit(1)

    from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore
    from adapters.reasoner.forward_chainer import ForwardChainer

    answer_value = ""
    query_goal_for_output = ""
    short_proof = ""
    proof_steps: list[dict[str, Any]] = []
    step_goals: list[str] = []
    used_fact_ids_set: set[str] = set()
    used_rule_ids_set: set[str] = set()
    span_ids: set[str] = set()
    duration_ms = 0.0
    proof_query_id: str | None = None

    try:
        kn_store = await PostgresKnowledgeStore.create(_dsn())
        reasoner = ForwardChainer(
            knowledge_store=kn_store,
            max_depth=settings.max_reasoner_depth,
            timeout_ms=settings.reasoner_timeout_ms,
        )
        try:
            if timeline_plan is not None:
                initial, events = timeline_plan
                current_value = str(initial)

                for idx, (op, value) in enumerate(events, 1):
                    step_query = LogicQuery(
                        goal=f"{op}({current_value},{value},?Z)",
                        variables=["Z"],
                    )
                    step_goals.append(step_query.goal)
                    reasoning = await reasoner.query(step_query)

                    duration_ms += reasoning.duration_ms
                    if proof_query_id is None and reasoning.query_id:
                        proof_query_id = reasoning.query_id

                    if not reasoning.success or not reasoning.answers:
                        print(
                            f"Brak odpowiedzi w knowledge base dla kroku {idx}: {step_query.goal}",
                            file=sys.stderr,
                        )
                        sys.exit(1)

                    first = reasoning.answers[0]
                    step_value = _first_binding(first, preferred_var="Z")
                    if step_value is None:
                        print(
                            f"Brak podstawienia zmiennych dla kroku {idx}: {step_query.goal}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    current_value = step_value

                    used_fact_ids_set.update(reasoning.used_fact_ids)
                    used_rule_ids_set.update(reasoning.used_rule_ids)
                    for step in reasoning.proof:
                        data = step.model_dump(mode="json")
                        data["step_id"] = len(proof_steps) + 1
                        proof_steps.append(data)
                        for sid in step.span_citations:
                            span_ids.add(sid)

                answer_value = current_value
                query_goal_for_output = " -> ".join(step_goals)
                short_proof = (
                    f"Wykonano {len(step_goals)} kroki reasonera. Wynik: {answer_value}."
                )
            else:
                assert query is not None
                reasoning = await reasoner.query(query)
                duration_ms = reasoning.duration_ms
                if reasoning.query_id:
                    proof_query_id = reasoning.query_id

                if not reasoning.success or not reasoning.answers:
                    print(
                        f"Brak odpowiedzi w knowledge base dla zapytania: {query.goal}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                first = reasoning.answers[0]
                answer = _first_binding(first, preferred_var="Z")
                if answer is None:
                    print(
                        f"Brak podstawienia zmiennych dla zapytania: {query.goal}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                answer_value = answer
                query_goal_for_output = query.goal
                short_proof = f"Znaleziono odpowiedz przez reasoner: {answer_value}."

                used_fact_ids_set.update(reasoning.used_fact_ids)
                used_rule_ids_set.update(reasoning.used_rule_ids)
                step_goals = [query.goal]
                for step in reasoning.proof:
                    data = step.model_dump(mode="json")
                    data["step_id"] = len(proof_steps) + 1
                    proof_steps.append(data)
                    for sid in step.span_citations:
                        span_ids.add(sid)
        finally:
            await kn_store.close()
    except Exception as exc:
        print(f"Blad reasonera/knowledge base: {exc}", file=sys.stderr)
        sys.exit(1)

    used_fact_ids = sorted(used_fact_ids_set)
    used_rule_ids = sorted(used_rule_ids_set)

    citation_spans: list[dict[str, Any]] = []
    if span_ids:
        from adapters.document_store.postgres_document_store import PostgresDocumentStore

        doc_store = await PostgresDocumentStore.create(_dsn())
        try:
            for sid in sorted(span_ids):
                try:
                    span = await doc_store.get_span(sid)
                except KeyError:
                    continue
                citation_spans.append(span.model_dump(mode="json"))
        finally:
            await doc_store.close()

    proof_id = proof_query_id if proof_query_id else str(uuid.uuid4())
    proof_record = {
        "proof_id": proof_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": parsed.problem_type.value,
        "original_text": text,
        "query": query_goal_for_output,
        "result": answer_value,
        "short_proof": short_proof,
        "steps": step_goals,
        "proof_steps": proof_steps,
        "used_fact_ids": used_fact_ids,
        "used_rule_ids": used_rule_ids,
        "citations": citation_spans,
        "duration_ms": duration_ms,
    }
    proof_id = _save_proof(proof_record, proof_id=proof_id)

    print(f"typ:       {parsed.problem_type.value}")
    print(f"query:     {query_goal_for_output}")
    print(f"wynik:     {answer_value}")
    print(f"proof_id:  {proof_id}")
    print(f"facts:     {len(used_fact_ids)}")
    print(f"rules:     {len(used_rule_ids)}")
    print(f"citations: {len(citation_spans)}")


async def _prove(args: argparse.Namespace) -> None:
    proofs = _load_proofs()
    rec = proofs.get(args.proof_id)
    if rec is None:
        print(
            f"Blad: nie znaleziono proof_id={args.proof_id} "
            f"w {_proof_store_path()}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"proof_id:   {args.proof_id}")
    print(f"created_at: {rec.get('created_at', '-')}")
    print(f"type:       {rec.get('kind', '-')}")
    print(f"text:       {rec.get('original_text', '-')}")
    print(f"result:     {rec.get('result', '-')}")
    print(f"summary:    {rec.get('short_proof', '-')}")

    steps = rec.get("steps") or []
    if steps:
        print("steps:")
        for idx, step in enumerate(steps, 1):
            print(f"  {idx}. {step}")

    proof_steps = rec.get("proof_steps") or []
    if proof_steps:
        print("proof_steps:")
        for st in proof_steps:
            sid = st.get("step_id", "?")
            rid = st.get("rule_or_fact_id", "?")
            concl = st.get("conclusion", "?")
            print(f"  {sid}. {rid} => {concl}")

    used_fact_ids = rec.get("used_fact_ids") or []
    used_rule_ids = rec.get("used_rule_ids") or []
    print(f"used_facts: {len(used_fact_ids)}")
    print(f"used_rules: {len(used_rule_ids)}")

    citations = rec.get("citations") or []
    if citations:
        print("citations:")
        for c in citations:
            span_id = c.get("span_id", "?")
            doc_id = c.get("doc_id", "?")
            surface = (c.get("surface_text", "") or "").replace("\n", " ").strip()
            if len(surface) > 120:
                surface = surface[:117] + "..."
            print(f"  {span_id} [{doc_id}] {surface}")


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

    # bootstrap-rules
    p = sub.add_parser(
        "bootstrap-rules",
        help="Ucz reguly z asserted facts i opcjonalnie promuj do asserted",
    )
    p.add_argument("--relations", nargs="*", default=[], metavar="REL")
    p.add_argument("--max-body", type=int, default=2, choices=[2, 3])
    p.add_argument("--min-coverage", type=int, default=3)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--min-confidence", type=float, default=0.95)
    p.add_argument("--min-lift", type=float, default=1.0)
    p.add_argument("--max-violations", type=int, default=0)
    p.add_argument("--max-corruption-hits", type=int, default=0)
    p.add_argument("--max-local-cwa-negatives", type=int, default=0)
    p.add_argument("--functional-relations", nargs="*", default=["eq"], metavar="REL")
    p.add_argument("--corruption-samples", type=int, default=2)
    p.add_argument("--max-groundings-per-pattern", type=int, default=5000)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--priority", type=int, default=20)
    p.add_argument("--reason", default="rule_bootstrap")
    p.add_argument("--no-local-cwa", action="store_true")
    p.add_argument("--no-promote", action="store_true")
    p.add_argument("--no-store-candidates", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # ner
    p = sub.add_parser("ner", help="Uruchom NER na tekście (wymaga backendu NER)")
    p.add_argument("--text", "-t", help="Tekst do analizy (lub stdin)")

    # solve
    p = sub.add_parser("solve", help="Rozwiaz zadanie matematyczne")
    p.add_argument("--text", "-t", help="Treść zadania (lub stdin)")

    # prove
    p = sub.add_parser("prove", help="Pobierz zapisany dowod po proof_id")
    p.add_argument("proof_id", help="UUID dowodu")

    # health
    sub.add_parser("health", help="Sprawdź połączenie z bazą danych")

    # reset
    sub.add_parser("reset", help="Usun wszystkie dane z KnowledgeStore (nieodwracalne)")

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
        "bootstrap-rules": _bootstrap_rules,
        "ner":     _ner,
        "health":  _health,
        "reset":   _reset,
        "solve":   _solve,
        "prove":   _prove,
    }

    if args.command in async_cmds:
        asyncio.run(async_cmds[args.command](args))
    elif args.command == "run":
        _run(args)


if __name__ == "__main__":
    main()
