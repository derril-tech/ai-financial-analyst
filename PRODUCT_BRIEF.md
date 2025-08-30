AI Financial Analyst — Multimodal RAG + Structured Data + Tools (Super-Granular Spec) 

0) Why this exists 

Analysts waste hours reconciling numbers across 10-Ks/10-Qs, XBRL, slides, and earnings calls, then re-doing it when live data shifts. This system reads filings (PDF/XBRL), listens to call audio, extracts tables & chart data, cross-checks with live market data + SQL warehouses, and runs valuation & risk models—returning cited, spreadsheet-ready answers. 

 

1) Product surface (what the user gets) 

Research Board (primary UI): one query → stacked cards: thesis bullets, cited passages, table slices, charts, and a “Show Your Work” panel with formulas & inputs. 

Ask-Anything bar: natural-language questions; toggles for company, period, report type, compliance mode. 

Drill-through: click any number to see: source doc, table cell coordinates, currency/unit normalization, transformations, and tool calls. 

What-if panel: WACC/terminal growth sliders; Monte Carlo; tornado charts. 

Exports: Excel/CSV of every table and calc; PPTX slides with charts; PDF report. 

Alerts: watchlists (tickers), KPI drift, guidance edits, unusual moves, factor shocks; Slack/email/WebSocket stream. 

 

2) Ingestion & normalization (deep) 

2.1 Sources 

Filings: 10-K, 10-Q, 8-K, annual reviews, investor decks (PDF/PPTX), proxy statements. 

Audio/video: earnings calls (MP3/MP4) + transcripts if supplied; otherwise ASR. 

XBRL: US-GAAP/IFRS facts + taxonomy; restatement linkage. 

Structured: equities OHLCV, fundamentals, estimates; SQL warehouses (read-only). 

User spreadsheets: optional joins for bespoke KPIs. 

2.2 Pipelines 

Upload → Object store (MinIO/S3) s3://org/{org_id}/raw/{uuid} with checksum. 

PDF parsing: PyMuPDF + Unstructured → sections; Camelot/Tabula → tables (handles multi-header, merged cells); figure extraction (PNG) + caption OCR fallback. 

Chart-to-data: detect chart regions; OCR axes/labels; curve digitization heuristic → numeric series + CI + caveat flag. 

Audio: WhisperX (ASR) → word timestamps; pyannote (diarization) → speakers; classifier to tag CEO/CFO/Analyst; sentiment + stance per turn. 

XBRL: parse contextRef to periods; units/decimals; track restatements; map common tags (Revenue, COGS, R&D, SBC, FCF, leases). 

Normalization: 

Currency/FX: value × fx_rate(filing_date) → USD (or org currency). Persist raw + normalized. 

Scale/precision: thousands/millions; decimals → numeric. 

Fiscal alignment: company FY → calendar; shift quarters. 

Non-GAAP reconciliation: link mgmt definitions to GAAP; capture adjustments (SBC, restructuring, etc.). 

Validation: totals = Σ subtotals (± tolerance), YoY logic checks; if fail → quarantine with reviewer task. 

Lineage: every transformed number stores {source_doc_id, locator, transform_chain, fx_snapshot_id}. 

2.3 Lake layout 

bronze/ raw binaries & naive extracts 

silver/ normalized tables (facts_xbrl, tables_pdf, transcript_turns, fx_snapshots) 

gold/ curated marts (kpi_quarterly, segment_revenue, comps_sets, valuation_runs) 

 

3) Indexing & retrieval (deep) 

3.1 Embedding collections (pgvector) 

passages_text: paragraphs with layout cues (section, page, heading tree). 

tables_cells: row/col stripes with header contexts; cell_id, row_key, col_key. 

slides_text: slide text + speaker notes. 

transcript_spans: 2–5 sentence windows tagged with speaker+timestamp. 

All payloads have rich metadata: doc_id, ticker, fy, fiscal_period, source_kind, locator, unit, currency, scale, confidence. 

3.2 Hybrid retrieval 

Sparse: BM25 (tsvector/pg_trgm) to anchor exact symbols & tags. 

Dense: text-embedding-3-large (or local alternative) for semantics. 

Fusion: Reciprocal Rank Fusion → rerank with small cross-encoder (optional). 

Table-aware: top-k cells clustered into slices w/ header roll-ups; ensure row completeness. 

GraphRAG: entities (subsidiaries/products/executives) and relations (owns, mentions, reports_to); neighbor expansion for context beyond a single doc. 

3.3 Answer assembly 

Evidence bundle must include ≥2 independent sources for numeric claims (e.g., XBRL + PDF table or PDF + slide). 

If independence not met → “analysis with caveats” and no hard numeric returned unless user accepts lower confidence. 

 

4) Tooling (structured data + computation) 

4.1 Market adapters 

Interfaces: get_ohlcv(ticker, start, end, interval), get_fundamentals(ticker, fields, asof), get_estimates(ticker, metric, horizon). 

Resilience: token bucket rate limiting, retries with jitter, circuit breaker, provider rotation, per-org API key vault. 

Snapshot caching: store normalized time series keyed by (provider, ticker, params_hash). 

4.2 SQL semantic layer 

Schema registry → NL templates → guarded SQL (param-bound; READ-ONLY). 

Row-level security by org_id. 

Explain plan preview & cost limit; abort on full scans of large tables unless whitelisted. 

4.3 Python sandbox 

Restricted modules (numpy, pandas, scipy, statsmodels, openpyxl, python-pptx, no network). 

Time/memory caps; temp dir mounted read-only to datasets; results saved as artifacts (CSV/XLSX/PPTX/PNG). 

Deterministic mode for reproducible runs. 

4.4 What-if & sensitivity 

Parameter registry with ranges/validations (WACC, terminal growth, churn, ARPU, CAC, SBC treatment). 

Tornado charts auto-generated; Monte Carlo with seed + percentile bands. 

 

5) Analytics engines 

Valuation: DCF (3-stage & H-model), dividend discount, residual income; working capital/capex schedules; terminal (Gordon/Exit multiple). Peer comps with winsorization/outlier trims; EV bridges. 

Risk/Factor: FF3/FF5 + momentum/quality; rolling beta (regime segmentation); VaR/CVaR; drawdowns. 

Event studies: pre/post windows, factor-adjusted abnormal returns; bootstrap p-values. 

SaaS KPIs: LTV/CAC, NDR, payback months, cohort analysis; reconcile to reported metrics. 

Sentiment: call tone (valence, certainty), hedge words, Q&A pressure; link to surprise magnitude. 

 

6) Orchestration & reasoning 

Planner agent (LangChain): build a plan → invoke tools → verify → compute → explain. 

Tool allowlist per compliance mode; prompt injection filters; numeric claims require verify_evidence() to pass. 

Outputs: markdown + tables; citations[] with {source, kind, locator, confidence}; exports{} linking files. 

“Show Your Work” returns intermediate dataframes, formulas, and decision trace (but not private chain-of-thought). 

 

7) UX/Frontend (Next.js 14/React 18) 

App Router + RSC; streaming responses via SSE to Research Board. 

Auth: NextAuth (email + OAuth), org switcher; roles: admin/analyst/viewer. 

Components: PromptBar, EvidenceCard, TableSlice (with row/col highlights), ChartCard, WhatIfPanel, DrillThroughModal, AlertCenter, ExportMenu. 

Charts: Recharts/ECharts; consistent number/currency formatting (Intl). 

Data grids: AG Grid (pin headers, sticky subtotals, copy/export). 

A11y & i18n: keyboard nav, ARIA labels; pluggable locale + currency unit. 

 

8) Data, schema, and caching 

Postgres + pgvector (primary app DB + vectors). 

DuckDB (local fast analytics/join on Parquet). 

Redis for queues and response caches: 

Semantic Answer Cache keyed by (org, prompt_hash, snapshot_id). 

Vector Query Cache keyed by (collection, embedding_hash). 

 

9) Security & compliance 

Row-Level Security by org_id; attribute-based access for documents (labels: public/private/MNPI). 

Secret vault for provider keys; KMS for S3; TLS everywhere. 

MNPI/PII guard: classifier + regex; quarantine flows; audit trail. 

Compliance modes: 

Informational: avoids prescriptive investment advice; adds disclaimers. 

Opinionated: allows target ranges with stronger caveats and evidence thresholds. 

Sandbox hardening: seccomp profile, import guards, max file handles, no network. 

 

10) Evaluation & QA 

Golden QA set (≥200 items) with numeric truth + tolerance; covers tables, footnotes, XBRL vs PDF discrepancies. 

Table extraction metrics: header detection F1, cell accuracy; unit normalization success. 

ASR metrics: WER < target; diarization DER < target. 

Hallucination score: uncited-claim penalty; fail-closed to “no answer with reasons” when evidence insufficient. 

A/B testing: retriever variants, prompt versions; gating by score. 

 

11) Ops & SRE 

CI/CD: lint, unit, schema tests, golden QA subset on PR; full eval nightly/weekly. 

Observability: OpenTelemetry traces (tool calls, DB queries), structured logs with trace IDs; dashboards for cost/latency/cache hits. 

Scaling: worker autoscaling; priority queues; backpressure strategies. 

Backups/DR: daily snapshots of Postgres + S3; restore drills; snapshot IDs stamped into exported reports. 

 

12) Deployment 

Local: Docker Compose for Postgres (pgvector), Redis, MinIO, API, Web. 

Staging/Prod: Vercel (web), Render/Fly.io (API/workers), managed Postgres, S3/MinIO, CDN for artifacts. Env-based feature flags. 

 

13) Example tool contracts (concise) 

# market_data.py 
def get_ohlcv(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame: ... 
def get_fundamentals(ticker: str, fields: list[str], asof: str|None=None) -> pd.DataFrame: ... 
 
# sql_readonly.py 
def query(sql: str, params: dict[str, Any]) -> pd.DataFrame: ...  # validates against registry 
 
# python_sandbox.py 
def run(code: str, inputs: dict[str, Any]) -> dict: ...  # returns artifacts + stdout/stderr 
 
# export.py 
def excel(tables: dict[str, pd.DataFrame]) -> str   # returns s3 path 
def pptx(slides: list[dict]) -> str 
 

 

14) Example prompts 

“Build a DCF for TSLA with Base/Bull/Bear using FY2022–FY2024 filings; normalize to USD; WACC 8–10%; export XLSX + PPTX; show tornado sensitivities.” 

“From Q2 call audio, extract updated gross margin guidance; compare to printed deck; flag definition changes in non-GAAP.” 

“Compare R&D % revenue for NVDA/AMD/INTC last 8 quarters; align fiscal calendars; explain deltas with citations.” 

 

15) Non-goals (keeps us honest) 

Providing individualized investment advice or brokerage-like features. 

Price targets without disclosed assumptions, methods, and evidence. 

 