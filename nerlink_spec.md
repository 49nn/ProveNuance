# NER Linker (PL) — Specyfikacja implementacyjna (lekki wariant) + Postgres/pgvector

## 1. Cel
System normalizacji i linkowania encji dla wejściowych wzmianek (mentions) z NER:
- mapuje wzmiankę do istniejącego `entity_id` (słownik → fuzzy → embedding),
- jeśli brak dopasowania → tworzy nowy `entity_id`,
- utrzymuje `canonical_name` i `aliases[]`,
- moduły fleksji i słownika są **pluginami** wymienialnymi przez zdefiniowany interfejs.

Zakres: **entity linking/normalization**. Detekcja mentionów (NER) jest poza zakresem — system przyjmuje już listę `Mention`.

---

## 2. Minimalny stack (Python)
Wymagane:
- `pydantic>=2`
- `rapidfuzz`
- `sqlalchemy>=2`
- `psycopg>=3`
- `orjson`
- `numpy`
- `pgvector` (python client; do typów i integracji z SQLAlchemy)

Opcjonalne:
- fleksja PL: `morfeusz2` (plugin)
- szybszy słownik w tekście: `pyahocorasick` lub `flashtext` (plugin)
- serwis: `fastapi`, `uvicorn`
- testy: `pytest`, `hypothesis`, `ruff`, `mypy`

---

## 3. Kontrakty danych (Pydantic)

### 3.1. Wejście
**Mention**
- `mention_id: str`
- `start: int` (offset znakowy, inclusive)
- `end: int` (offset znakowy, exclusive)
- `text: str`
- `label: str` (np. `PERSON`, `ORG`, `LAW`, `TERM`)

**DocumentInput**
- `doc_id: str`
- `text: str`
- `language: Literal["pl"] = "pl"`
- `mentions: list[Mention]`

### 3.2. Wyjście (lekki wariant)
**ResolvedEntity**
- `mention_id: str`
- `span: {start:int, end:int, text:str}`
- `label: str`
- `status: Literal["linked","new","ambiguous"]`
- `entity_id: str | None`
- `canonical_name: str | None`
- `confidence: float` (0..1)
- `method: str` (np. `dict`, `fuzzy`, `emb`, `dict+fuzzy`, `fuzzy+emb`, `new`)
- `alias_to_add: str | None`

**LightResult**
- `doc_id: str`
- `entities: list[ResolvedEntity]`
- `updates:`
  - `aliases_to_add: list[{entity_id:str, alias:str, mention_id:str, alias_type:str}]`
  - `entities_to_create: list[{entity_id:str, label:str, canonical_name:str, aliases:list[str]}]`

---

## 4. Normalizacja tekstu (wspólna funkcja)
Wszystkie moduły korzystają z jednej funkcji normalizacji klucza:
`normalize_key(s: str) -> str`

Wymagane zachowanie:
- `strip()`
- `casefold()`
- zamiana wielu whitespace na pojedynczą spację
- (opcjonalnie konfig) usuwanie kropek z akronimów: `K.N.F.` → `knf`
- (opcjonalnie konfig) ujednolicenie myślników/łączników

Konfiguracja: `NormalizationConfig`.

---

## 5. Pluginy (wymienialne interfejsem)

### 5.1. Plugin fleksji (FlexionPlugin)
Cel: wygenerować kandydatów (np. lemmy) i cechy fleksyjne.

**Kontrakt**
```python
from typing import Protocol
from pydantic import BaseModel

class FlexionAnalysis(BaseModel):
    surface: str
    candidates: list[str]          # zawsze co najmniej [surface]
    features: dict[str, str] = {}  # np. {"case":"GEN","number":"SG"}

class FlexionPlugin(Protocol):
    name: str
    def analyze(self, text: str, label: str, lang: str = "pl") -> FlexionAnalysis: ...
```

**Referencyjne implementacje**
- `IdentityFlexionPlugin`: `candidates=[text]`
- `MorfeuszFlexionPlugin` (opcjonalnie): lematy + heurystyka wyboru

### 5.2. Plugin słownika (DictionaryPlugin)
Cel: deterministyczne dopasowanie aliasów (w szczególności skrótów i nazw instytucji).

**Kontrakt**
```python
from typing import Protocol, Optional
from pydantic import BaseModel

class DictMatch(BaseModel):
    matched: bool
    entity_id: Optional[str] = None
    matched_alias: Optional[str] = None
    confidence: float = 1.0
    meta: dict = {}

class DictionaryPlugin(Protocol):
    name: str
    def lookup(self, text: str, label: str, lang: str = "pl") -> DictMatch: ...
```

**Referencyjne implementacje**
- `NullDictionaryPlugin`: zawsze `matched=False`
- `SimpleMapDictionaryPlugin`: mapa `normalized_alias -> entity_id` (z seed JSON lub z DB)

---

## 6. Pipeline rozwiązywania (EntityResolver)

Dla każdego `Mention`:
1) `key = normalize_key(mention.text)`
2) `flex = flexion_plugin.analyze(mention.text, mention.label)`
3) **Słownik**: `dictionary.lookup(candidate)` dla `candidate` w `[mention.text] + flex.candidates`
   - jeśli matched → `linked`
4) **Fuzzy** (RapidFuzz) na aliasach/canonical w danym `label`:
   - query: `mention.text` i wszystkie `flex.candidates`
   - wynik: najlepszy alias → `entity_id`
5) **Embedding** (pgvector) jako fallback/rozstrzygnięcie:
   - wygeneruj embedding dla query (jeśli włączone) i wykonaj `topK` w DB
6) Decyzja:
   - `linked` jeśli spełnione progi
   - `ambiguous` jeśli top2 są blisko (zdefiniowane w polityce)
   - `new` jeśli brak dopasowania
7) Aktualizacje:
   - jeśli `linked` i `mention.text` nie jest aliasem encji → `alias_to_add`

### 6.1. Polityka progów (ResolverPolicy)
Konfigurowalne parametry (przykładowe):
- `fuzzy_link_threshold = 92` (0..100)
- `fuzzy_ambiguous_threshold = 86`
- `embedding_link_threshold = 0.80` (cosine)
- `embedding_ambiguous_delta = 0.03` (różnica top1-top2)
- `topk_embedding = 10`
- `topn_fuzzy_candidates = 5000` (limit listy kandydatów per label)

Reguły:
- słownik wygrywa (jeśli matched)
- fuzzy ≥ link → linked
- jeśli fuzzy w [ambiguous..link) → spróbuj embedding
- embedding ≥ link → linked
- jeśli top2 zbliżone → ambiguous
- inaczej → new

---

## 7. Generowanie `entity_id`
Interfejs:
```python
class IdGenerator(Protocol):
    def new_id(self, label: str, canonical_name: str) -> str: ...
```

Strategia minimalna:
- `ent_{label.lower()}_{sha1(normalize_key(canonical_name))[:10]}` (+ opcjonalny namespace/salt)

---

## 8. Postgres + pgvector — schemat DB

### 8.1. Wymagania DB
- PostgreSQL
- `pgvector` extension

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 8.2. Tabela encji
```sql
CREATE TABLE entities (
  entity_id       TEXT PRIMARY KEY,
  label           TEXT NOT NULL,
  canonical_name  TEXT NOT NULL,
  canonical_key   TEXT NOT NULL,               -- normalize_key(canonical_name)
  status          TEXT NOT NULL DEFAULT 'active',
  meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX entities_label_idx ON entities(label);
CREATE INDEX entities_label_canonical_key_idx ON entities(label, canonical_key);
```

### 8.3. Tabela aliasów
```sql
CREATE TABLE entity_aliases (
  alias_id        BIGSERIAL PRIMARY KEY,
  entity_id       TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
  alias           TEXT NOT NULL,
  alias_key       TEXT NOT NULL,               -- normalize_key(alias)
  alias_type      TEXT NOT NULL DEFAULT 'surface', -- surface/inflected/acronym/synonym/llm
  weight          REAL NOT NULL DEFAULT 1.0,
  is_preferred    BOOLEAN NOT NULL DEFAULT false,
  source          JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX entity_aliases_unique_per_entity
ON entity_aliases(entity_id, alias_key);

CREATE INDEX entity_aliases_key_idx ON entity_aliases(alias_key);
CREATE INDEX entity_aliases_entity_idx ON entity_aliases(entity_id);
```

### 8.4. Embeddingi aliasów (pgvector)
Embedding trzymamy na poziomie aliasu.

```sql
-- dim zależny od modelu, np. 768
CREATE TABLE alias_embeddings (
  alias_id        BIGINT PRIMARY KEY REFERENCES entity_aliases(alias_id) ON DELETE CASCADE,
  model_id        TEXT NOT NULL,
  embedding       vector(768) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indeks ANN (dobór zależy od wersji pgvector). Przykład:
-- CREATE INDEX alias_embeddings_hnsw_idx
-- ON alias_embeddings USING hnsw (embedding vector_cosine_ops);
```

### 8.5. Audyt decyzji (opcjonalnie)
```sql
CREATE TABLE entity_links_audit (
  audit_id        BIGSERIAL PRIMARY KEY,
  doc_id          TEXT NOT NULL,
  mention_id      TEXT NOT NULL,
  label           TEXT NOT NULL,
  mention_text    TEXT NOT NULL,
  mention_key     TEXT NOT NULL,
  status          TEXT NOT NULL,               -- linked/new/ambiguous
  entity_id       TEXT NULL,
  confidence      REAL NOT NULL,
  method          TEXT NOT NULL,
  evidence        JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX entity_links_audit_doc_idx ON entity_links_audit(doc_id);
CREATE INDEX entity_links_audit_entity_idx ON entity_links_audit(entity_id);
```

---

## 9. Zapytania kluczowe (logika)

### 9.1. Lookup słownikowy (DB jako słownik)
```sql
SELECT e.entity_id
FROM entity_aliases a
JOIN entities e ON e.entity_id = a.entity_id
WHERE e.label = :label AND a.alias_key = :alias_key
LIMIT 1;
```

### 9.2. pgvector topK (cosine) — koncepcyjnie
- join `alias_embeddings` → `entity_aliases` → `entities`
- filtr: `entities.label = :label` oraz `alias_embeddings.model_id = :model_id`
- sort: odległość kosinusowa (wg operatorów pgvector)

---

## 10. JSON/JSONL (import/export)

### 10.1. `entities.jsonl`
```json
{"entity_id":"ent_org_3f21a9b7c1","label":"ORG","canonical_name":"Komisja Nadzoru Finansowego","meta":{"country":"PL"}}
```

### 10.2. `aliases.jsonl`
```json
{"entity_id":"ent_org_3f21a9b7c1","alias":"KNF","alias_type":"acronym","weight":1.0,"is_preferred":false,"source":{"seed":"regulations_v1"}}
{"entity_id":"ent_org_3f21a9b7c1","alias":"Komisji Nadzoru Finansowego","alias_type":"inflected","weight":0.7,"source":{"doc_id":"d1","mention_id":"m_0007"}}
```

### 10.3. `dictionary_seed.json`
```json
{
  "version": "2026-02-19",
  "entries": [
    {"label":"ORG","alias":"KNF","entity_id":"ent_org_3f21a9b7c1"},
    {"label":"LAW","alias":"k.p.a.","entity_id":"ent_law_91d2c1a0ef"}
  ]
}
```

### 10.4. `link_results.jsonl`
```json
{"doc_id":"d1","mention_id":"m_0001","label":"PERSON","span":{"start":125,"end":130,"text":"Tomka"},"status":"linked","entity_id":"ent_person_1f3c9a","canonical_name":"Tomek","confidence":0.94,"method":"dict+fuzzy","alias_to_add":"Tomka"}
```

---

## 11. Struktura paczki (propozycja)
```
nerlink/
  models.py
  normalize.py
  idgen.py
  store_pg.py
  plugins/
    flexion_base.py
    flexion_identity.py
    flexion_morfeusz.py
    dict_base.py
    dict_null.py
    dict_map.py
    dict_db.py
  match/
    fuzzy.py
    embedding_pgvector.py
  resolver.py
  cli.py
  tests/
```

---

## 12. MVP: wymagane implementacje
- `IdentityFlexionPlugin`
- `NullDictionaryPlugin` lub `SimpleMapDictionaryPlugin`
- `normalize_key`
- `RapidFuzzMatcher`
- `PgStore` (repo na Postgres)
- `EntityResolver` + `ResolverPolicy`
- `PgVectorEmbeddingMatcher` (opcjonalnie; jeśli włączone embeddingi)

---

## 13. Kryteria akceptacji
1) Fleksja: `Tomek` / `Tomka` → `linked`, alias dopisany
2) Skróty: `KNF` → `linked` przez słownik, alias dopisany jeśli brak
3) Synonimy/zdrobnienia: `piłka` vs `piłeczka`
   - bez embedding: dopuszczalne `new`
   - z embedding: preferowane `linked`
4) Ambiguity: alias wspólny dla wielu encji → `ambiguous`
