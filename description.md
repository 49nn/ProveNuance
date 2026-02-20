## Plan implementacji PoC (on-prem) z modułami wymiennymi i stabilnymi interfejsami

Poniżej architektura modułowa (hexagonal/ports-and-adapters): każdy moduł ma **kontrakt (interfejs)**, implementacje można podmieniać bez zmian reszty.

### Cele PoC

* Rozwiązywanie prostych zadań matematycznych (1–3 klasa) na podstawie wiedzy z podręcznika.
* Odpowiedź: wynik + krótki dowód + cytaty 1:1 (provenance).
* Lokalnie (on-prem), uruchamiane w Dockerze.

---

# 1) Moduły i interfejsy (ports)

## 1.1 DocumentStore (pamięć źródeł 1:1)

**Odpowiedzialność:** przechowywanie dokumentów i fragmentów (DocSpan), wersjonowanie.

**Interfejs (Python Protocol / abstract base class)**

* `ingest_document(doc: DocumentIn) -> DocumentRef`
* `list_spans(doc_ref: DocumentRef) -> list[DocSpan]`
* `get_span(span_id: str) -> DocSpan`
* `search_spans(query: str, doc_ref: Optional[DocumentRef]) -> list[DocSpanHit]`

**Wymienne implementacje**

* `SQLiteDocumentStore` (MVP)
* `PostgresDocumentStore` (później)
* `Filesystem+JSON` (najprostsze)

---

## 1.2 KnowledgeStore (fakty/reguły/schemat)

**Odpowiedzialność:** przechowywanie trójek i reguł, status `hypothesis/asserted`, provenance, snapshoty.

**Interfejs**

* `upsert_fact(f: Fact) -> FactRef`
* `upsert_rule(r: Rule) -> RuleRef`
* `get_asserted_facts(snapshot: Optional[SnapshotRef]) -> Iterable[Fact]`
* `get_asserted_rules(snapshot: Optional[SnapshotRef]) -> Iterable[Rule]`
* `promote_fact(fact_id: str, reason: PromotionReason) -> None`
* `promote_rule(rule_id: str, reason: PromotionReason) -> None`
* `create_snapshot() -> SnapshotRef`
* `diff_snapshots(a: SnapshotRef, b: SnapshotRef) -> SnapshotDiff`

**Wymienne implementacje**

* SQLite/Postgres
* Pliki JSON (na start)

---

## 1.3 FrameExtractor (LLM/regex → frames)

**Odpowiedzialność:** z DocSpan (lub zdania) wyciąga „frames” w kontrolowanym DSL.

**Interfejs**

* `extract_frames(span: DocSpan) -> list[Frame]`
* `validate_frame(frame: Frame) -> list[ValidationIssue]`

**Wymienne implementacje**

* `RuleBasedFrameExtractor` (MVP; ręczne patterny)
* `LLMFrameExtractor` (później; lokalny lub zdalny)
* `HybridFrameExtractor` (łączony)

---

## 1.4 FrameMapper (frames → fakty/reguły)

**Odpowiedzialność:** deterministyczne mapowanie frame na fakty/reguły (z provenance).

**Interfejs**

* `map(frame: Frame, provenance: Provenance) -> MappingResult`

  * `facts: list[Fact]`
  * `rules: list[Rule]`
  * `entities: list[Entity]` (opcjonalnie; np. HomeOf(x))

**Wymienne implementacje**

* Wersja domenowa „MathGrade1to3Mapper”
* Inna domena = nowy mapper, bez zmian w extractorze i solverze

---

## 1.5 EntityLinker (normalizacja encji)

**Odpowiedzialność:** lematyzacja/kanonizacja nazw, ID encji.

**Interfejs**

* `link(name: str, typ: Optional[str], context: Optional[DocSpan]) -> EntityRef`
* `add_alias(entity: EntityRef, alias: str) -> None`

**Wymienne implementacje**

* prosty słownik + heurystyki (MVP)
* embedding-based linker (później)

---

## 1.6 Validator (schema + constraints)

**Odpowiedzialność:** wykrywanie błędów mapowania i konfliktów.

**Interfejs**

* `validate_fact(f: Fact, schema: Schema) -> list[ValidationIssue]`
* `validate_rule(r: Rule, schema: Schema) -> list[ValidationIssue]`
* `find_conflicts(facts: Iterable[Fact], constraints: Constraints) -> list[ConflictSet]`

**Wymienne implementacje**

* proste walidacje typów + funkcjonalność (MVP)
* rozszerzone constraints / MaxSAT (później)

---

## 1.7 Reasoner (deterministyczne wnioskowanie)

**Odpowiedzialność:** odpowiedź na zapytania, proof trace.

**Interfejs**

* `query(q: LogicQuery, snapshot: SnapshotRef) -> ReasonerResult`

  * `answers: list[Answer]`
  * `proof: ProofTrace`
  * `used_facts: list[FactRef]`
  * `used_rules: list[RuleRef]`

**Wymienne implementacje**

* `DatalogReasoner` (np. Soufflé) – rekomendowane
* `PrologReasoner` (SWI-Prolog) – opcjonalnie
* `PythonForwardChainer` (MVP, bardzo szybki start)

---

## 1.8 MathProblemParser (pytanie ucznia → AST/query)

**Odpowiedzialność:** parsowanie wyrażeń i prostych zadań tekstowych.

**Interfejs**

* `parse(text: str) -> ParsedProblem`

  * `problem_type: "expr"|"word_problem"`
  * `expr_ast` (dla rachunkowych)
  * `logic_query` (np. `value(expr, ?)` lub `solve_word_problem(...)`)
  * `slots` (liczby, jednostki, obiekty)

**Wymienne implementacje**

* regex+parser (MVP)
* LLM-based semantic parser (później)

---

## 1.9 Evaluator (arytmetyka bez LLM)

**Odpowiedzialność:** deterministyczne liczenie AST.

**Interfejs**

* `eval_expr(ast: ExprAST) -> int|Fraction`
* `simplify(ast: ExprAST) -> ExprAST`

**Wymienne implementacje**

* własny evaluator (MVP)
* sympy (później, jeśli potrzebne)

---

## 1.10 Orchestrator / API (jedyny moduł „sklejający”)

**Odpowiedzialność:** workflow: ingest → extract → map → validate → store → query → response.

**Interfejs (HTTP)**

* `POST /ingest` (dokument lub fragmenty)
* `POST /extract` (uruchom ekstrakcję na dokumencie)
* `POST /promote` (promocja faktów/reguł)
* `POST /solve` (zadanie ucznia)
* `GET /proof/{id}` (dowód i cytaty)

Orchestrator zależy tylko od portów, nie od implementacji.

---

# 2) Minimalny DSL danych (stabilny kontrakt między modułami)

## 2.1 DocSpan

```json
{ "span_id":"...", "doc_id":"...", "version":"...", "surface_text":"..." }
```

## 2.2 Frame (MVP: tylko 3–5 typów)

```json
{ "type":"ARITH_EXAMPLE", "lhs":"2+3", "rhs":"5" }
{ "type":"PROPERTY", "name":"commutative_addition" }
{ "type":"PROCEDURE", "name":"column_addition", "text_span_id":"..." }
```

## 2.3 Fact / Rule

Fact:

```json
{ "h":"Tomek", "r":"owns", "t":"Pilka", "status":"hypothesis", "provenance":["span_17"] }
```

Rule (Horn):

```json
{ "head":"add(X,Y,Z)", "body":["add(Y,X,Z)"], "status":"asserted", "provenance":["span_21"] }
```

W PoC matematycznym sensowniej trzymać predykaty typu:

* `add(a,b,c)` zamiast `("=" (add a b) c)` — łatwiej w Datalog/Prolog.

---

# 3) Wnioskowanie w PoC: rekomendowany rdzeń

### Najprościej i stabilnie

* **Evaluator** liczy wynik wyrażeń.
* **Reasoner** używa reguł głównie do:

  * przekształceń (np. przemienność),
  * doboru właściwego przykładu z podręcznika (1:1),
  * generowania kroków dowodu.

To pozwala szybko osiągnąć „działa” bez budowy pełnego CAS.

---

# 4) Docker (on-prem) – proponowany układ

## 4.1 Usługi

1. `api` – FastAPI orchestrator
2. `db` – Postgres (albo SQLite w volume na start)
3. `reasoner` – Soufflé worker (opcjonalnie jako osobny container)
4. `ui` – opcjonalnie panel (może być później)

## 4.2 docker-compose (koncept)

* `api` montuje `/data` (dokumenty, eksporty faktów do solvera)
* `reasoner` czyta snapshot asserted → liczy closure → zapisuje wyniki do `/data/closure`

---

# 5) Plan implementacji krok po kroku (wersja PoC)

## Etap 0: repo + kontrakty (1–2 dni)

* Monorepo:

  * `contracts/` (Pydantic modele: DocSpan/Frame/Fact/Rule/ProofTrace)
  * `ports/` (interfejsy modułów)
  * `adapters/` (implementacje)
  * `api/` (orchestrator)
* Ustal wersjonowanie kontraktów (`contracts_version`).

## Etap 1: DocumentStore + KnowledgeStore (2–4 dni)

* SQLite (MVP) albo Postgres od razu.
* CRUD dla DocSpan, facts, rules, snapshots.

## Etap 2: FrameExtractor (rule-based) + FrameMapper (3–5 dni)

* Ręcznie przygotuj 30–100 DocSpan z podręcznika:

  * przykłady typu „2+3=5”
  * zdania „Dodawanie jest przemienne”, „0 nie zmienia sumy”
* Extractor:

  * regex na przykłady równań,
  * słownik fraz na własności.
* Mapper:

  * przykłady → `example(expr,result)` + opcjonalnie `add(a,b,c)`
  * własności → reguły Horn (`commutative_add`).

## Etap 3: Parser zadań + Evaluator (3–6 dni)

* Parser wyrażeń: `+ - ( )` (0–100).
* Parser prostych zadań tekstowych: 5–10 wzorców.
* Evaluator: deterministyczne liczenie + generowanie kroków.

## Etap 4: Reasoner (MVP) (2–5 dni)

Najpierw `PythonForwardChainer`:

* stosuje asserted rules do generowania nowych faktów i kroków,
* zwraca proof trace.

Następnie (opcjonalnie) podmień na `SouffléReasoner` bez zmian w API:

* adapter generuje pliki .facts i .dl z asserted,
* uruchamia Soufflé,
* parsuje wyniki + buduje proof trace (w MVP może być „explain by replay”: log derivacji w Pythonie).

## Etap 5: API /solve + format odpowiedzi (2–3 dni)

* `/solve`:

  1. parse problem,
  2. retrieval z pamięci (examples/properties),
  3. reasoner + evaluator,
  4. proof + citations.

Zwracaj zawsze:

* wynik,
* kroki,
* cytaty 1:1 (DocSpan).

## Etap 6: Promotion + testy regresji (2–4 dni)

* Progi promocji (support/confidence) dla fact/rule.
* Zestaw testów Q/A:

  * 20–50 pytań z oczekiwanymi wynikami.
* Snapshoty i diff.

---

# 6) Minimalny zestaw reguł do PoC (start)

* przemienność dodawania: `add(X,Y,Z) :- add(Y,X,Z).` *(lub jako transformacja AST)*
* 0 jako neutralny: `add(X,0,X). add(0,X,X).`
* relacja odejmowania: `sub(Z,X,Y) :- add(X,Y,Z).`
* porównanie: `gt(X,Y) :- ...` (opcjonalnie na później)

W PoC możesz trzymać te reguły jako:

* reguły na faktach (`add(a,b,c)`) i/lub
* reguły transformacji AST (często prostsze).

---

# 7) Podmiana implementacji bez zmian reszty (przykłady)

* Zamieniasz `RuleBasedFrameExtractor` → `LLMFrameExtractor`:

  * kontrakt `FrameExtractor.extract_frames()` ten sam.
* Zamieniasz `PythonForwardChainer` → `SouffléReasoner`:

  * kontrakt `Reasoner.query()` ten sam.
* Zamieniasz SQLite → Postgres:

  * tylko adaptery Store.

To jest główna korzyść architektury port-adapter.

---

# 8) Artefakty projektu (co powinno powstać w repo)

* `contracts.py` (Pydantic modele)
* `ports/*.py` (interfejsy)
* `adapters/sqlite_store.py`, `adapters/postgres_store.py`
* `adapters/rule_based_extractor.py`
* `adapters/math_mapper.py`
* `adapters/python_reasoner.py`, `adapters/souffle_reasoner.py`
* `adapters/parser_expr.py`, `adapters/parser_word.py`
* `api/main.py` (FastAPI)
* `docker-compose.yml`
* `tests/` (golden Q/A + regresje snapshotów)


