# ARCH — AI Financial Analyst (Multimodal RAG + Structured Data + Tools)
_Date: 2025-08-30_


## 1) High-Level Diagram (Mermaid)
```mermaid
flowchart LR
  U[User (Web)] -->|HTTPS| W[Next.js 14]
  W -->|SSE/WebSocket| A[FastAPI Gateway]
  A -->|RPC| ORCH[Planner/Orchestrator]
  subgraph Ingestion & Storage
    S3[(MinIO/S3 Raw Artifacts)]
    LG[(Parquet Lake 
 bronze/silver/gold)]
    PG[(Postgres + pgvector)]
    RD[(Redis Queue/Cache)]
  end
  subgraph Workers
    PDF[PDF Parser 
 Unstructured/PyMuPDF/Camelot]
    ASR[WhisperX + pyannote]
    XBRL[XBRL Loader]
    IDX[Indexers: dense/sparse/table/graph]
    VAL[Valuation & Risk Models]
  end
  ORCH --> RD
  RD --> PDF
  RD --> ASR
  RD --> XBRL
  PDF --> LG
  ASR --> LG
  XBRL --> LG
  LG --> IDX --> PG
  ORCH --> PG
  ORCH --> VAL
  ORCH --> MD[(Market Data APIs)]
  VAL --> PG
  A --> PG
  W -->|Downloads| S3
```

## 2) Module Breakdown
### 2.1 Web (Next.js 14 / React 18)
- App Router, RSC where helpful; server actions call API gateway.
- Auth via NextAuth (OIDC/OAuth + email); org switcher.
- Pages: `/board`, `/uploads`, `/companies`, `/alerts`, `/eval`.
- Components: PromptBar, EvidenceCard, TableSlice, ChartCard, DrillThroughModal, WhatIfPanel, AlertCenter.
- Streaming: SSE/WebSocket for long jobs; optimistic UI for board.

### 2.2 API Gateway (FastAPI)
- Endpoints (simplified):
  - `POST /v1/query`: plan + answer with citations; stream tokens.
  - `POST /v1/upload`: presigned URL; begins ingestion job.
  - `GET /v1/artifacts/{id}`: download assets (XLSX/PPTX/CSV/PDF).
  - `POST /v1/valuation/dcf`: compute with inputs or auto-from-filings.
  - `GET /v1/alerts/stream`: WebSocket feed.
  - `POST /v1/admin/reindex`: rebuild indexes for doc/org.
- Pydantic models: `QueryRequest`, `Citation`, `EvidenceItem`, `ToolCall`, `ValuationInputs`, `Alert`.

### 2.3 Orchestrator (Planner + Tools)
- LangChain agent with **explicit tools** only; JSON schemas; retries/backoff.
- Plan steps: retrieve → verify (cross-source) → compute → explain → export.
- Tool registry: `market_data.get_ohlcv`, `fundamentals.get_metric`, `sql.query_readonly`, `python_sandbox.run`, `export.excel`, `export.pptx`.
- Evidence gating: require ≥2 independent sources for numeric claims or downgrade to “analysis with caveats.”

### 2.4 Ingestion & Normalization
- PDF → sections, tables (with multi-header), figures; image OCR fallback.
- Audio → segments with speaker roles; transcript alignment to slide/page numbers.
- XBRL → fact tables with taxonomy & units; restatement history.
- Normalization: currency/units, FY alignment, non-GAAP reconciliation mapping.

### 2.5 Indexing & Retrieval
- Embeddings: text (passages), table cells (header+row stripes), transcript segments, slide text.
- Stores: **pgvector** (`collection_id`, `embedding`, `payload JSONB`).
- Hybrid search: BM25 (pg_tgrm/tsvector) + dense cosine; **RRF** fusion.
- GraphRAG: `entity(name,type)`; `relation(src,dst,label,confidence)`; hop-limited traversal.

### 2.6 Analytics Engines
- **Valuation**: DCF (3-stage/H), comps; sensitivity (WACC, tg, growth), tornado, Monte Carlo.
- **Risk**: factor regressions (FF3/FF5), rolling betas; VaR; drawdown curves.
- **Event Study**: abnormal returns around events; visualization helpers.
- **SaaS KPIs**: LTV/CAC, NDR, cohort survival; unit economics.

## 3) Data Model (Postgres)
```
orgs(id, name, created_at)
users(id, org_id FK, email, role, created_at)
documents(id, org_id, kind, title, ticker, fiscal_year, fiscal_period, language, checksum, path_s3, meta JSONB, created_at)
artifacts(id, org_id, document_id, type, path_s3, meta JSONB, created_at)
facts_xbrl(id, org_id, document_id, taxonomy, tag, value_num, unit, period_start, period_end, decimals, restated_from_id FK, meta JSONB)
transcripts(id, org_id, document_id, speaker, text, start_sec, end_sec, meta JSONB)
vector_index(id, org_id, document_id, collection, embedding VECTOR, payload JSONB)
queries(id, org_id, user_id, prompt, plan JSONB, created_at)
answers(id, org_id, query_id, text, confidence, citations JSONB, exports JSONB, created_at)
alerts(id, org_id, type, payload JSONB, status, created_at)
audit_log(id, org_id, user_id, action, resource, meta JSONB, created_at)
```
- RLS: all tables scoped by `org_id`; policies per role (admin, analyst, viewer).

## 4) Caching & Performance
- **Two-tier cache:** (a) semantic answer cache keyed by prompt+snapshot; (b) vector query cache per collection.
- **Latency budgets:** retrieval < 800ms p95; end-to-end < 2s cached.
- **Batching:** embedding and API calls; async streaming to UI.
- **Backpressure:** queue limits; drop to warm cache when overload.

## 5) Security & Compliance
- Secret vault (env/provider), KMS for S3; encryption at rest & transit.
- MNPI/PII classifier; quarantine bucket; audit trail for access.
- Tool allowlist; input sanitization; Python sandbox seccomp profile.
- Policy banners & disclaimers appended by compliance mode.

## 6) Evaluation
- Golden QA with numeric tolerance; table extraction F1; ASR WER thresholds.
- Hallucination detector: uncited-claim rule; “no-answer” path.
- CI gates: run eval on 5% nightly; block merge if score below threshold.

## 7) ADRs (Architecture Decision Records)
1. **pgvector in Postgres** for locality & simplicity; external vector DB optional for scale.
2. **MinIO/S3** for raw artifacts; Parquet lake for lineage & reproducibility.
3. **LangChain tools** for orchestration due to ecosystem maturity; keep adapters thin & testable.
4. **Hybrid retrieval (BM25 + dense + table-aware)** to maximize precision on financial docs.
5. **Next.js 14** for modern RSC & streaming UX; deploy to Vercel for simplicity.

## 8) Example Schemas (Pydantic)
```python
class QueryRequest(BaseModel):
    org_id: str
    prompt: str
    tickers: list[str] = []
    options: dict[str, Any] = {}

class Citation(BaseModel):
    source: str  # s3 path or URL
    kind: str    # pdf|xbrl|audio|slide|api
    locator: str # page:line or cell coord or timestamp
    confidence: float

class ValuationInputs(BaseModel):
    ticker: str
    wacc: float | None = None
    terminal_growth: float | None = None
    years: int = 5
    scenario: Literal["base","bull","bear"] = "base"
```
## 9) Endpoints (OpenAPI excerpt)
```
POST /v1/query
POST /v1/upload (presign) → PUT s3
GET  /v1/artifacts/{id}
POST /v1/valuation/dcf
GET  /v1/alerts/stream   (WS)
POST /v1/admin/reindex
```

## 10) Deployment & Ops
- **Local:** docker compose all services; make targets: `make up`, `make seed`, `make eval`.
- **Staging/Prod:** Vercel (web), Render/Fly.io (API/workers), managed Postgres (pgvector), S3/MinIO, Redis.
- **Runbooks:** API outage, provider failover, index rebuild, snapshot restoration.
