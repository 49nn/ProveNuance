# LLM-Based Named Entity Recognition — Design Document

## Overview

This document describes the design for an LLM-powered Named Entity Recognition (NER)
pipeline for the ProveNuance system, targeting the **elementary mathematics domain (grades 1–3)**.

The LLM replaces (or supplements) the regex-based `RuleBasedExtractor` with a semantic
understanding layer. All entity canonical names are in **English** regardless of input language.

The integration calls a **custom NER backend service** over HTTP.
The service contract is minimal: send plain text, receive structured JSON.
No authentication, no provider-specific protocol.

---

## 1. Mathematical Entity Types

| Type | Description | Examples | Canonical form |
|------|-------------|----------|----------------|
| `OPERATION` | Arithmetic operations | addition, subtraction, multiplication, division, dodawanie, mnożenie | `add`, `sub`, `mul`, `div` |
| `PROPERTY` | Mathematical properties | commutative, associative, distributive, zero neutral, identity, przemienne, łączne | `commutative`, `associative`, `distributive`, `zero_neutral`, `identity` |
| `NUMBER` | Specific numeric values | 5, 3.14, one hundred, trzy, sto | `"5"`, `"3.14"`, `"100"` |
| `VARIABLE` | Algebraic variables | x, y, n, a, b | lowercase letter as-is |
| `RELATION` | Mathematical relations | equals, less than, greater than, równa się, mniejszy niż | `eq`, `lt`, `gt`, `lte`, `gte`, `neq` |
| `CONCEPT` | Abstract math concepts | fraction, digit, place value, remainder, ułamek, cyfra | snake_case English name |

---

## 2. LLM Output Format

The LLM returns a single JSON object. The schema is designed to map directly onto
the existing `Frame` union type from `contracts.py`.

### 2.1 JSON Schema

```json
{
  "entities": [
    {
      "text": "<original surface text of the entity>",
      "type": "<OPERATION|PROPERTY|NUMBER|VARIABLE|RELATION|CONCEPT>",
      "start": 0,
      "end": 8,
      "canonical": "<normalized English name>",
      "confidence": 0.98
    }
  ],
  "frames": [
    {
      "frame_type": "<ARITH_EXAMPLE|PROPERTY|PROCEDURE|DEFINITION>",
      "<frame-specific fields...>": "..."
    }
  ],
  "unmapped": ["<text fragments the model could not classify>"]
}
```

### 2.2 Frame Sub-schemas

> **Note:** `source_span_id` is injected by the ProveNuance adapter — the backend must NOT include it.

#### ARITH_EXAMPLE
Maps to `ArithExampleFrame` in `contracts.py`.
```json
{
  "frame_type": "ARITH_EXAMPLE",
  "operation": "add",
  "operands": [3, 4],
  "result": 7
}
```

#### PROPERTY
Maps to `PropertyFrame` in `contracts.py`.
```json
{
  "frame_type": "PROPERTY",
  "subject": "add",
  "property_name": "commutative",
  "value": true
}
```

#### PROCEDURE
Maps to `ProcedureFrame` in `contracts.py`.
```json
{
  "frame_type": "PROCEDURE",
  "operation": "add",
  "steps": [
    "Align digits by place value",
    "Add ones column first",
    "Carry if sum >= 10",
    "Add tens column"
  ]
}
```

#### DEFINITION
Maps to `DefinitionFrame` in `contracts.py`.
Use for sentences that define what a mathematical term means (e.g. "X is Y", "X is called Y").
- `term` — the concept being defined, canonical English name (e.g. `"add"`, `"sum"`, `"subtraction"`)
- `definition` — the description in the original language of the text

```json
{
  "frame_type": "DEFINITION",
  "term": "add",
  "definition": "łączenie dwóch (albo więcej) liczb"
}
```

```json
{
  "frame_type": "DEFINITION",
  "term": "sum",
  "definition": "wynik dodawania"
}
```

### 2.3 Full Example

**Input text:** `"3 plus 4 equals 7, because addition is commutative"`

**LLM output:**
```json
{
  "entities": [
    {"text": "3", "type": "NUMBER", "start": 0, "end": 1, "canonical": "3", "confidence": 0.99},
    {"text": "plus", "type": "OPERATION", "start": 2, "end": 6, "canonical": "add", "confidence": 0.99},
    {"text": "4", "type": "NUMBER", "start": 7, "end": 8, "canonical": "4", "confidence": 0.99},
    {"text": "equals", "type": "RELATION", "start": 9, "end": 15, "canonical": "eq", "confidence": 0.99},
    {"text": "7", "type": "NUMBER", "start": 16, "end": 17, "canonical": "7", "confidence": 0.99},
    {"text": "addition", "type": "OPERATION", "start": 30, "end": 38, "canonical": "add", "confidence": 0.97},
    {"text": "commutative", "type": "PROPERTY", "start": 42, "end": 53, "canonical": "commutative", "confidence": 0.98}
  ],
  "frames": [
    {
      "frame_type": "ARITH_EXAMPLE",
      "operation": "add",
      "operands": [3, 4],
      "result": 7
    },
    {
      "frame_type": "PROPERTY",
      "subject": "add",
      "property_name": "commutative",
      "value": true
    }
  ],
  "unmapped": []
}
```

---

## 3. LLM Prompt Design

### 3.1 System Prompt

```
You are a Named Entity Recognition (NER) system specializing in elementary mathematics
(grades 1–3). Your role is to extract mathematical entities and semantic frames from text.

## Entity Types
- OPERATION: arithmetic operations (addition, subtraction, multiplication, division)
- PROPERTY: mathematical properties (commutative, associative, distributive,
  zero_neutral, identity)
- NUMBER: specific numeric values (integers, decimals, number words)
- VARIABLE: algebraic variables (single letters used as unknowns)
- RELATION: mathematical relations (equals, less than, greater than, not equal to)
- CONCEPT: abstract math concepts (fraction, digit, remainder, place value)

## Rules
1. All canonical names MUST be in English, regardless of input language.
2. Canonical forms for OPERATION: add, sub, mul, div
3. Canonical forms for RELATION: eq, lt, gt, lte, gte, neq
4. Canonical forms for PROPERTY: commutative, associative, distributive,
   zero_neutral, identity
5. Extract frames that capture relationships between entities.
6. For sentences that define a term ("X is Y", "X is called Y", "X means Y") extract a
   DEFINITION frame with `term` (canonical English) and `definition` (original language text).
7. `start` and `end` are character offsets into the original text (0-indexed, end exclusive).
7. Confidence is a float in [0.0, 1.0].
8. Place unrecognized but potentially relevant fragments in `unmapped`.

## Output
Return ONLY valid JSON matching this schema — no explanation, no markdown fences.
```

### 3.2 User Prompt

```
Extract named entities and frames from the following text:

"{surface_text}"
```

### 3.3 API Call Strategy

The adapter calls a **custom NER backend** over plain HTTP.
The contract is intentionally minimal — the backend owns all LLM details internally.

**HTTP request** (`POST {ner_backend_url}`):

```
POST /ner
Content-Type: text/plain

3 plus 4 equals 7, because addition is commutative
```

**HTTP response** (`200 OK`, `Content-Type: application/json`):

```json
{
  "entities": [...],
  "frames": [...],
  "unmapped": []
}
```

The adapter parses the response body and validates it against `LLMNEROutput` (Pydantic).

**HTTP client**: `httpx.AsyncClient` (already in `requirements.txt` as `httpx==0.27.0`).
**No auth headers.** The backend is assumed to be internal / network-isolated.

---

## 4. Backend API Design

### 4.1 New Endpoint: `POST /ner`

**File:** `api/routers/ner.py`

#### Request Schema (`api/schemas.py`)

```python
class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    doc_id: UUID | None = None
    span_id: UUID | None = None

class NEREntity(BaseModel):
    text: str
    type: Literal["OPERATION", "PROPERTY", "NUMBER", "VARIABLE", "RELATION", "CONCEPT"]
    start: int
    end: int
    canonical: str
    confidence: float = Field(ge=0.0, le=1.0)

class NERResponse(BaseModel):
    span_id: UUID | None
    text: str
    entities: list[NEREntity]
    frames: list[FrameResult]   # reuse existing FrameResult from api/schemas.py
    processing_time_ms: int
```

#### HTTP Contract

```
POST /ner
Content-Type: application/json

{
  "text": "3 plus 4 equals 7",
  "span_id": null
}

→ 200 OK
{
  "span_id": null,
  "text": "3 plus 4 equals 7",
  "entities": [...],
  "frames": [...],
  "processing_time_ms": 342
}

→ 422 Unprocessable Entity   (validation error — text missing or too long)
→ 503 Service Unavailable    (NER backend unreachable)
→ 504 Gateway Timeout        (NER backend exceeded ner_timeout_ms)
```

#### Optional: `POST /ner/batch`

```
POST /ner/batch
{
  "spans": [
    {"text": "...", "span_id": "uuid"},
    ...
  ],
  "domain": "math_elementary"
}

→ 200 OK
{
  "results": [<NERResponse>, ...]
}
```

---

## 5. Adapter Architecture

### 5.1 `LLMFrameExtractor`

**File:** `adapters/frame_extractor/llm_extractor.py`

Implements the `FrameExtractor` Protocol from `ports/frame_extractor.py` — fully
drop-in compatible with `RuleBasedExtractor`.

```
LLMFrameExtractor
├── __init__(ner_backend_url, timeout_ms)
│       ner_backend_url — URL of the custom NER service, e.g. "http://ner-service:8001/ner"
│       timeout_ms      — request timeout
│
├── extract_frames(span: DocSpan) -> list[Frame]
│   1. POST {ner_backend_url}
│        Content-Type: text/plain
│        body: span.surface_text   (raw text, nothing else)
│   2. Read response body as JSON
│   3. Validate with LLMNEROutput (Pydantic) → raises on schema mismatch
│   4. Map LLMNEROutput.frames → list[Frame] from contracts.py
│   5. Inject source_span_id = span.span_id into each frame
│   6. Return frames
│
└── validate_frame(frame: Frame) -> list[ValidationIssue]
    → Delegates to existing SimpleValidator or inline checks
```

### 5.2 Internal Pydantic Models

```python
class LLMNEROutput(BaseModel):
    entities: list[NEREntity]
    frames: list[dict]       # raw, before mapping to Frame union
    unmapped: list[str] = []

class _LLMArithFrame(BaseModel):
    frame_type: Literal["ARITH_EXAMPLE"]
    operation: str
    operands: list[int | float | str]
    result: int | float | str
    source_span_id: UUID | None = None

class _LLMPropertyFrame(BaseModel):
    frame_type: Literal["PROPERTY"]
    subject: str
    property_name: str
    value: bool | str
    source_span_id: UUID | None = None

class _LLMProcedureFrame(BaseModel):
    frame_type: Literal["PROCEDURE"]
    operation: str
    steps: list[str]
    source_span_id: UUID | None = None
```

---

## 6. Configuration Changes

**File:** `config.py` — add to `Settings`:

```python
# NER backend service
ner_backend_url: str = "http://localhost:8001/ner"
ner_timeout_ms: int = 10_000
```

**Environment variables** (prefix `PROVE_NUANCE_`):
```
PROVE_NUANCE_NER_BACKEND_URL=http://localhost:8001/ner
PROVE_NUANCE_NER_TIMEOUT_MS=10000
```

**No new dependency** — `httpx` is already listed in `requirements.txt` (version 0.27.0).
No API key configuration needed.

---

## 7. Integration with Existing Pipeline

### Strategy: New Independent Endpoint (Recommended)

The `/ner` endpoint runs **independently** of the existing `/ingest` → `/extract` pipeline.
This is the least invasive option and allows parallel testing.

```
Existing pipeline (unchanged):
  POST /ingest  →  segmentation  →  DocSpans in DB
  POST /extract →  RuleBasedExtractor  →  Frames  →  Facts/Rules in DB

New LLM pipeline (additive):
  POST /ner     →  LLMFrameExtractor  →  NERResponse (stateless, not persisted by default)
```

### Future Integration Options

**Option A — Swap adapter**: Replace `RuleBasedExtractor` with `LLMFrameExtractor` in
`api/main.py` lifespan. The `/extract` endpoint continues working unchanged.

**Option B — Hybrid**: `HybridFrameExtractor` runs regex first; LLM only for unmatched spans.
Best cost/coverage trade-off for production.

**Option C — Ingestion flag**: Add `use_llm_ner: bool = False` to `ExtractRequest`.
Allows per-request choice of extractor.

---

## 8. File Map

| File | Status | Change |
|------|--------|--------|
| `api/routers/ner.py` | New | NER router (`POST /ner`, `POST /ner/batch`) |
| `api/schemas.py` | Modify | Add `NERRequest`, `NEREntity`, `NERResponse` |
| `api/main.py` | Modify | Register `/ner` router; init `LLMFrameExtractor` in lifespan |
| `api/dependencies.py` | Modify | Expose `LLMFrameExtractor` via `request.app.state.llm_extractor` |
| `adapters/frame_extractor/llm_extractor.py` | New | `LLMFrameExtractor` adapter |
| `config.py` | Modify | Add `ner_backend_url`, `ner_timeout_ms` |
| `pyproject.toml` | Modify | Promote `httpx` from test-only to runtime dependency |
| `.env.example` | Modify | Add `PROVE_NUANCE_NER_BACKEND_URL=`, `PROVE_NUANCE_NER_TIMEOUT_MS=` |

---

## 9. Verification

1. **Smoke test** — `POST /ner` with English math text:
   - Input: `"Addition is commutative: 3 + 4 = 4 + 3"`
   - Expected: `OPERATION(add)` + `PROPERTY(commutative)` + 2× `ARITH_EXAMPLE` frames

2. **Multilingual test** — input in Polish:
   - Input: `"3 dodać 4 równa się 7"`
   - Expected: entities with canonical names in English (`add`, `eq`), `ARITH_EXAMPLE` frame

3. **Regression** — existing endpoints unaffected:
   - `POST /ingest` → 200 OK (unchanged)
   - `POST /extract` → 200 OK using `RuleBasedExtractor` (unchanged)
   - `GET /health` → `{"status": "ok"}`

4. **Error handling**:
   - `ner_backend_url` unreachable → 503 Service Unavailable
   - Backend timeout → 504 Gateway Timeout
   - Backend returns invalid JSON → 500 with logged raw response
   - Switch backend URL by changing env var only — no code change needed
