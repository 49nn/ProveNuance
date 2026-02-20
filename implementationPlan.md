# ProveNuance — Plan implementacji

## Stack technologiczny
- Python 3.12 + FastAPI + asyncpg (Postgres) + Docker
- Architektura: Hexagonal / Ports-and-Adapters
- Reasoner MVP: PythonForwardChainer (czyste Python)

---

## Struktura katalogów

```
prove-nuance/
├── implementationPlan.md        ← ten plik
├── contracts.py                 # Pydantic modele — jedyne źródło prawdy
├── config.py                    # pydantic-settings; prefix PROVE_NUANCE_
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── ports/                       # interfejsy (typing.Protocol)
│   ├── __init__.py
│   ├── document_store.py
│   ├── knowledge_store.py
│   ├── frame_extractor.py
│   ├── frame_mapper.py
│   ├── entity_linker.py
│   ├── validator.py
│   ├── reasoner.py
│   ├── math_problem_parser.py
│   └── evaluator.py
├── adapters/                    # wymienne implementacje
│   ├── __init__.py
│   ├── document_store/
│   │   ├── __init__.py
│   │   └── postgres_document_store.py
│   ├── knowledge_store/
│   │   ├── __init__.py
│   │   └── postgres_knowledge_store.py
│   ├── frame_extractor/         # [Etap 2]
│   │   └── rule_based_extractor.py
│   ├── frame_mapper/            # [Etap 2]
│   │   └── math_grade1to3_mapper.py
│   ├── entity_linker/           # [Etap 2]
│   │   └── dict_entity_linker.py
│   ├── validator/               # [Etap 2]
│   │   └── simple_validator.py
│   ├── reasoner/                # [Etap 4]
│   │   └── forward_chainer.py
│   ├── math_problem_parser/     # [Etap 3]
│   │   └── regex_parser.py
│   └── evaluator/               # [Etap 3]
│       └── ast_evaluator.py
├── api/                         # [Etap 5]
│   ├── main.py
│   ├── dependencies.py
│   ├── schemas.py
│   └── routers/
│       ├── ingest.py
│       ├── extract.py
│       ├── promote.py
│       ├── solve.py
│       └── proof.py
├── data/                        # volume mount
│   └── .gitkeep
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   │   ├── qa_pairs.json
│   │   └── snapshots/
│   └── test_golden_qa.py
└── scripts/
    ├── seed_textbook.py
    └── run_golden.py
```

---

## Etapy implementacji

### ✅ Etap 0 — Kontrakty i konfiguracja — **ZAIMPLEMENTOWANY**
- [x] `contracts.py` — wszystkie modele Pydantic (DocSpan, Fact, Rule, Frame, ProofStep, …)
- [x] `config.py` — Settings z pydantic-settings (PROVE_NUANCE_ prefix, env_file=.env)
- [x] `ports/__init__.py` + 9 portów (Protocol interfaces): document_store, knowledge_store, frame_extractor, frame_mapper, entity_linker, validator, reasoner, math_problem_parser, evaluator
- [x] `requirements.txt` (fastapi, uvicorn, pydantic, asyncpg, pytest)
- [x] `Dockerfile` (python:3.12-slim, non-root user, workers=4)
- [x] `docker-compose.yml` (db: postgres:16-alpine z healthcheck, api: depends_on db)

### ✅ Etap 1 — DocumentStore + KnowledgeStore (PostgreSQL) — **ZAIMPLEMENTOWANY**
- [x] `adapters/document_store/postgres_document_store.py`
  - Schema DDL wbudowany w plik (tworzona przy starcie)
  - `ingest_document()` — segmentacja regexem `(?<=[.!?])\s+` + executemany
  - `search_spans()` — `tsvector + plainto_tsquery('simple', …) + ts_rank`
  - `get_span()`, `list_spans()`, `delete_document()` (CASCADE)
  - Factory: `PostgresDocumentStore.create(dsn)` — asyncpg pool
- [x] `adapters/knowledge_store/postgres_knowledge_store.py`
  - Schema: facts, rules (UNIQUE constraints), snapshots, snapshot_facts/rules, fact/rule_history
  - `upsert_fact/rule()` — `INSERT … ON CONFLICT DO UPDATE`, zwraca `created` flag
  - `promote_fact/rule()` + `retract_fact()` — UPDATE + INSERT do history (transakcja)
  - `create_snapshot()` — `INSERT INTO snapshot_facts SELECT … WHERE status='asserted'`
  - `diff_snapshots()` — set-based SQL diff między dwoma snapshot_facts/rules
  - Factory: `PostgresKnowledgeStore.create(dsn)` — asyncpg pool

### ✅ Etap 2 — FrameExtractor + FrameMapper — **ZAIMPLEMENTOWANY**
- [x] `adapters/frame_extractor/rule_based_extractor.py`
  - Regex dla ARITH_EXAMPLE (cyfry + słowa, symbole +-×÷)
  - Regex dla PROPERTY (commutative, associative, zero_neutral, distributive)
  - `validate_frame()` — sprawdza arność, arytmetyczną spójność, znane właściwości
- [x] `adapters/frame_mapper/math_grade1to3_mapper.py`
  - ARITH_EXAMPLE → 2× Fact (ground atom + instance_of)
  - PROPERTY(commutative/associative/zero_neutral/distributive) → Rule Horn
- [x] `adapters/entity_linker/dict_entity_linker.py`
  - Wbudowane encje: operacje, cyfry 0–20, koncepty właściwości
  - `link()` — tworzenie nowych encji + indeks aliasów
  - `add_alias()`, `get_entity()`
- [x] `adapters/validator/simple_validator.py`
  - `validate_fact()` — puste pola, wartości nieujemne, znane relacje
  - `validate_rule()` — pusty head, require_body, check_arity
  - `find_conflicts()` — naruszenia ograniczenia funkcjonalnego (h,r) → t

### ✅ Etap 3 — MathProblemParser + Evaluator — **ZAIMPLEMENTOWANY**
- [x] `adapters/math_problem_parser/regex_parser.py`
  - Klasyfikacja: EXPR / WORD_PROBLEM / UNKNOWN
  - Precedence-climbing parser dla wyrażeń arytmetycznych (obsługa nawiasów, priorytety)
  - Ekstrakcja slotów + operation hints z word problems (EN + PL słowa kluczowe)
  - `LogicQuery` budowany automatycznie: `add(5,3,?Z)`
- [x] `adapters/evaluator/ast_evaluator.py`
  - `eval_expr()` — rekurencyjne przejście ExprAST z Fraction
  - `simplify()` — constant-folding: BinOp(Num, Num) → Num
  - Fractions dla dokładnej arytmetyki (brak błędów float)
  - steps[] — krok po kroku dla proof

### ✅ Etap 4 — Reasoner (ForwardChainer / SLD) — **ZAIMPLEMENTOWANY**
- [x] `adapters/reasoner/forward_chainer.py`
  - SLD-resolution top-down DFS (backtracking, generatorowy)
  - Unifikacja Robinson'a (`_unify` + `_walk` + `_apply`)
  - Rename zmiennych per krok zapobiegający capture (`X` → `X_{suffix}`)
  - ProofStep z substitution dict i span_citations z proweniencji faktów
  - Timeout przez `time.monotonic()` + `deadline`
  - Reguły ładowane z KnowledgeStore (asserted) — `load-rules` lub `extract`
- [x] `provenuance solve` — EXPR przez ASTEvaluator, WORD_PROBLEM przez ForwardChainer
  - Weryfikacja przemienności: `add(3,5,?Z)` → reguła → `add(5,3,8)` ✓

### ⬜ Etap 5 — API FastAPI
- [ ] `api/main.py` — lifespan, DI wiring
- [ ] `api/dependencies.py` — get_document_store(), get_knowledge_store(), …
- [ ] `api/routers/ingest.py` — POST /ingest
- [ ] `api/routers/extract.py` — POST /extract
- [ ] `api/routers/promote.py` — POST /promote
- [ ] `api/routers/solve.py` — POST /solve (główny endpoint)
- [ ] `api/routers/proof.py` — GET /proof/{id}

### ⬜ Etap 6 — Testy + Promotion
- [ ] `tests/unit/test_forward_chainer.py`
- [ ] `tests/unit/test_math_parser.py`
- [ ] `tests/unit/test_evaluator.py`
- [ ] `tests/integration/test_postgres_stores.py`
- [ ] `tests/golden/qa_pairs.json` (20+ par Q/A)
- [ ] `tests/test_golden_qa.py`
- [ ] `scripts/seed_textbook.py`
- [ ] `scripts/run_golden.py`

---

## Kluczowe decyzje techniczne

| Temat | Decyzja | Uzasadnienie |
|-------|---------|--------------|
| Baza danych | PostgreSQL + asyncpg | Produkcyjnie od razu; multiprocess-safe |
| Reasoner | PythonForwardChainer | Zero deps, prosty debug; Soufflé można dodać przez port-adapter |
| FTS | `tsvector` + `plainto_tsquery` | Native Postgres, bez dodatkowych usług |
| HTTP | FastAPI async | Spójne z asyncpg |
| Migracje | Schema SQL aplikowana na starcie | Prostsze dla PoC; Alembic można dodać |
| Wersjonowanie | `contracts_version = "1.0.0"` w contracts.py | Stabilny kontrakt między modułami |

---

## Weryfikacja end-to-end

```bash
# 1. Build i start
docker-compose up --build

# 2. Ingest podręcznika
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"title":"Math Grade 2","raw_text":"3 + 4 = 7. Addition is commutative."}'

# 3. Extract + auto-promote
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<id>","auto_promote":true}'

# 4. Solve
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"text":"Mary has 5 apples. She gets 3 more. How many in all?"}'
# → {"result":8,"short_proof":"The answer is 8...","citations":[...]}

# 5. Testy
pytest tests/ -v --tb=short
python scripts/run_golden.py
```

---

## Podmiana implementacji (bez zmian reszty)

```python
# Swap Reasoner: PythonForwardChainer → SouffléReasoner
app.dependency_overrides[get_reasoner] = lambda: SouffléReasoner(...)

# Swap store dla testów
app.dependency_overrides[get_knowledge_store] = lambda: InMemoryKnowledgeStore()
```
