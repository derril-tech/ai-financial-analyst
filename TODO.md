# TODO — AI Financial Analyst (Multimodal RAG + Structured Data + Tools)
_Date: 2025-08-30_


> Tech stack: **Next.js 14 / React 18**, **FastAPI** (Python 3.11), **LangChain** (+ optional LlamaIndex utilities), **Postgres 16** (+ pgvector), **DuckDB 1.x**, **Redis** (queue + cache), **WhisperX** (ASR), **PyMuPDF/Camelot/Unstructured** (PDF), **MinIO/S3** (object store), **Parquet** (lake), **Vercel** (web), **Render/Fly.io** (API workers), **Docker Compose** (local).

---

## PHASE 0 — Monorepo Scaffold & DevEx
- [x] Create monorepo: `apps/web` (Next 14), `apps/api` (FastAPI), `packages/shared` (types, prompts), `infra/` (docker, compose, env), `datasets/`, `eval/`.
- [x] Configure **pnpm** workspaces (or npm) and **poetry** for Python.
- [x] Dockerfiles: slim API image (poetry, uvicorn, cpu build), local dev image (with poppler, tesseract, ffmpeg).
- [x] `docker-compose.yml`: Postgres + pgvector, Redis, MinIO, API, Nginx (optional), Next.js.
- [x] Env management: `.env`, `.env.example`, secrets mounting for prod.
- [x] Pre-commit hooks: black, isort, ruff, mypy, eslint, prettier, type-check.
- [x] GitHub Actions: lint/test/build matrix; cache poetry & pnpm.
- [x] Seed scripts for DB migrations (**alembic**) and initial admin user.
- [x] Feature flags (env-based + runtime overrides).
- [x] Observability: request/trace IDs, OpenTelemetry stubs.

**DoD:** Monorepo boots locally with `docker compose up`, web & API health checks green.  
**Tests:** CI job runs lint + unit tests; sample ping and DB migration tested.

---

## PHASE 1 — Ingestion & Normalization
- [x] Upload pipeline: PDFs, images, PPTX, audio (MP3/MP4), spreadsheets → MinIO/S3 `s3://org/{org_id}/raw/{uuid}`.
- [x] Metadata capture: title, ticker(s), FY, fiscal period, language, source (10-K, 10-Q, Call, Deck), checksum, uploader, ACL.
- [x] PDF parse: **Unstructured + PyMuPDF**; table extraction with **Camelot/Tabula**; image extraction (PNG); figure captions.
- [x] Audio/Video: **WhisperX** ASR + **pyannote** diarization; word-level timestamps; speaker roles (CEO/CFO/Analyst) via classifier.
- [x] XBRL ingest: parse US-GAAP/IFRS, taxonomy map, units, decimals; fact provenance & restatement tracking.
- [x] Unit/currency normalization: registry for USD/EUR/...; FX snapshot on filing date; scale/precision (thousands/millions).
- [x] Fiscal calendar aligner: map company FY to calendar; quarter shifting rules.
- [x] Data validation: cross-check totals vs subtotals; reconcile non-GAAP to GAAP if provided.
- [x] Parquet lake write: `datasets/bronze|silver|gold` with reproducible snapshot IDs.
**DoD:** A 10-K PDF + call audio + XBRL produce normalized silver tables and artifacts with provenance.  
**Tests:** Golden fixtures for 2 public companies; assert unit conversions, table row counts, ASR WER thresholds.

---

## PHASE 2:
# Indexing & Retrieval
- [x] Embedding service: select model (OpenAI text-embedding-3-large or local), **pgvector** store, cosine + ivfflat index.
- [x] Chunkers: passage (layout-aware), table-row/region, slide, transcript segment; attach rich metadata.
- [x] Hybrid retrieval: sparse (BM25) + dense; **Fusion** re-ranking (Reciprocal Rank Fusion).
- [x] Table-aware retriever: schema for table cell coordinates; return slices with header context.
- [x] GraphRAG: entity-resolution (subsidiaries, execs, products); relation triples; neighborhood traversal.
- [x] Confidence scoring: ensemble of retrievers; minimum evidence gating.
- [x] Caching: vector query cache (semantic key), passage cache; invalidation on new filings.
**DoD:** Queries return top-k with passages, table slices, slide refs, transcript spans; latency < 800ms p95 (cold).  
**Tests:** Retrieval precision/recall vs labeled QA; ablate components (dense-only, hybrid, etc.).
## Tooling (Structured Data & Live APIs)
- [x] Market data adapters: equities (OHLCV intraday+daily), fundamentals, estimates; pluggable providers; rate-limit + retry.
- [x] SQL semantic layer: NL→SQL with guarded templates; schema registry; row-level org ACL.
- [x] Python sandbox: restricted packages, time/mem limits; read-only dataset mounts; export artifacts (CSV/XLSX).
- [x] Spreadsheet I/O: export any answer; ingest user Excel to join with warehouse tables.
- [x] What-if engine: parameter registry (WACC, tg, churn, ARPU), constraint checks, sensitivity runs; produce tornado charts.
**DoD:** Question triggers both RAG and tools, with cited results and downloadable XLSX.  
**Tests:** Unit tests for each adapter; sandbox timeout and import guards.

---

## PHASE 3: 
# Analytics & Valuation
- [x] DCF suite: 3-stage + H-model; WACC calc; working capital & capex modeling; terminal methods (Gordon, Exit multiple).
- [x] Comps engine: peer set curation, winsorization; EV/EBITDA, P/S, PEG, EV/S, EV/FCF; unit/currency normalization.
- [x] Factor + risk: FF3/FF5 + momentum/quality; rolling beta by regime; VaR, CVaR; drawdown analytics.
- [x] Event study: abnormal returns vs factor model; windows (-5, +5); bootstrapped significance.
- [x] SaaS/consumer KPIs: LTV/CAC, NDR, payback, cohort charts; variance explanations from filings.
**DoD:** A full NVDA vs AMD valuation pack with Base/Bull/Bear + sensitivities.  
**Tests:** Deterministic results on fixed snapshots; tolerances documented.
# UX & Delivery
- [x] Next.js 14 App Router skeleton; Mantine UI (or Tailwind + shadcn); auth (NextAuth, email+OAuth).
- [x] "Research Board" page: prompt → cards (thesis bullets, charts, tables), drag/sort, inline edits.
- [x] Drill-through: click any number → "Show Work" modal (inputs, formula, citations).
- [x] Export: PPTX/Keynote via server; CSV/XLSX; PDF report.
- [x] Alerts: watchlists, KPI drift, guidance changes; WebSocket toasts; alert JSON with "explain-why."

**DoD:** End-to-end: ask → board → drill → export; live feed for alerts.  
**Tests:** Playwright flows; Lighthouse perf budget; a11y baseline (axe).

---

## PHASE 4:
# Security, Governance, Compliance
- [x] Multitenancy isolation (org_id in every row); RLS policies in Postgres.
- [x] Secret management; per-provider API key vault; audit log (who saw what).
- [x] MNPI/PII guard: classifier + regex; quarantine workflow; approval override.
- [x] Policy modes: "Informational only" vs "Opinionated" answer framing; disclosures appended.
- [x] Red-team prompt-injection tests; tool-use allowlist & input sanitization.
**DoD:** External pen-test checklist; guardrails enforced in CI.  
**Tests:** Attack prompts library; verify blocked actions + logs.
# Evaluation & Telemetry
- [x] Golden QA set (200+ Qs) with numeric truth sets; unit/currency aware grading.
- [x] Hallucination score: uncited claim detector; evidence coverage metric.
- [x] Cost/latency telemetry; per-agent flamegraphs; cache hit ratios.
- [x] A/B framework for retrievers and prompts; model gates by eval score.

**DoD:** Eval dashboard with trendlines; "fail to merge if eval < threshold."  
**Tests:** CI gates run nightly on small slice, full weekly.

---
## PHASE 5:
## Milestone 8 — Scale & Ops
- [x] Worker autoscaling; priority queues; backpressure.
- [x] Sharded pgvector or external vector store (optional).
- [x] Snapshotting & reproducibility: data snapshot IDs on every report.
- [x] Backups & DR playbook; SLOs (99.9% API), SLAs for enterprise.
**DoD:** Load test with 200 concurrent analyst sessions; p95 < 2s (cached).  
**Tests:** Locust/Gatling scripts; chaos experiments.
## Backlog (Nice-to-haves)
- [x] Guidance Consistency Map, Footnote Time Machine, Counterfactual Explainer.
- [x] Portfolio API (batch tickers, PM reports).
- [x] Alt-data pack (web, jobs, app ranks, satellite).
