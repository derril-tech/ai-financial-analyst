# PLAN — AI Financial Analyst (Multimodal RAG + Structured Data + Tools)
_Date: 2025-08-30_


## 1) Product Goals
- Deliver **auditable, cited** answers about company performance by fusing filings (PDF/XBRL), earnings call audio, and **live market data**.
- Provide **valuation models** (DCF/comps), **risk analytics**, and **explainable outputs** exportable to Excel/PowerPoint.
- Operate under **governed, multi-tenant** data controls with robust guardrails and evals.

## 2) Personas & Core Jobs-to-be-Done
- **Sell-side/Buy-side analyst:** rapid thesis building, comps, event studies, earnings prep.
- **CFO/FP&A:** benchmarking peers, sanity-checking guidance, sensitivity/scenario analysis.
- **Fintech app dev:** embed reports via Portfolio API.
- **Compliance:** audit trace, disclosure checks.

## 3) User Journeys (MVP → Pro)
1. **Ask a KPI question** → RAG over 10-K → get table slice + citation → export CSV.
2. **Valuation pack** → DCF + comps with Base/Bull/Bear → download PPTX.
3. **Earnings study** → call audio tone + transcript diff → flag reconciliations.
4. **Watchlist alert** → KPI drift/guidance change detected → Slack/email/WebSocket alert with “why”.

## 4) Roadmap (Quarterized, scope-based — adapt dates as needed)
- **Q1 (M0–M3):** Scaffold, ingestion (PDF/XBRL/Audio), normalization, hybrid retrieval, tools+APIs, Python sandbox.
- **Q2 (M4–M5):** Valuation suite, event study, SaaS KPIs; Research Board UI, exports, alerts.
- **Q3 (M6–M7):** Security/compliance hardening, red-team; eval harness + telemetry gating.
- **Q4 (M8):** Scaling, DR, enterprise features; alt-data pack (optional).

## 5) Success Metrics (KPIs)
- Analyst time-to-answer: **< 30s** for common KPIs.
- Evidence coverage: **> 95%** answers include at least 2 independent citations.
- Numeric accuracy vs golden set: **MAE < 3%** on normalized figures.
- Latency: **p95 < 2s** for cached queries; **< 800ms** retrieval stage p95 (cold).

## 6) Scope & Guardrails
- **Non-goals:** retail-trading signals, price targets without disclosure, broker-like functions.
- **Compliance modes:** Informational vs Opinionated; investment-disclaimer injection.
- **Data ethics:** alt-data labeled experimental; bias/limitation notes required.

## 7) Technical Plan (Condensed)
- **Data layer:** S3/MinIO for raw; Parquet lake (bronze/silver/gold); Postgres (metadata, auth, results); pgvector store.
- **Processing:** FastAPI + workers; Celery/Dramatiq + Redis; ASR (WhisperX), diarization (pyannote), PDF (Unstructured/Camelot).
- **Retrieval:** Dense + sparse + table-aware; GraphRAG entities; fusion re-rank; confidence gating.
- **Reasoning:** LangChain agents with strict tool schemas; planner → retriever → verifier → calculator → explainer.
- **UX:** Next 14 App Router; Mantine; Recharts/ECharts; AG Grid; WebSocket streams.
- **Exports:** XLSX/PPTX/CSV/PDF via python-docx/pptx, pandas, reportlab.
- **Deploy:** Vercel (web), Render/Fly.io (API, workers), managed Postgres, S3-compatible object store.

## 8) Risks & Mitigations
- **Parsing brittleness:** multi-parser fallback; manual review UI; golden tests.
- **Hallucinations:** evidence gating, “no answer” mode, uncited-claim penalties.
- **API limits/outages:** provider rotation, backoff/retry, snapshot caching.
- **Costs:** caching, smaller models for retrieval, batch embedding, budget alerts.
- **Security:** RLS, secret vault, sanitization, code sandboxing, audit logs.

## 9) Go-to-Market / Packaging (suggested)
- **Free tier:** 3 companies, limited filings; delayed market data.
- **Pro:** unlimited companies, live data, exports, alerts.
- **Enterprise:** SSO/SAML, private S3 bucket, VPC peering, premium SLAs.

## 10) Acceptance Criteria for GA
- E2E flows stable across 10 diverse tickers; reproducible reports by snapshot ID.
- Eval dashboard shows targets met for accuracy, latency, and hallucination control.
- Security audit completed; DR runbook validated.

## 11) Current Status (2025-08-30)
**PHASES 0 & 1 COMPLETED** ✅

### Phase 0 - Monorepo Scaffold & DevEx
- ✅ Complete monorepo structure with Next.js 14, FastAPI, shared packages
- ✅ Docker containerization with development and production images
- ✅ Docker Compose orchestration (Postgres+pgvector, Redis, MinIO, API, Web, Worker)
- ✅ Environment management with .env.example template
- ✅ Pre-commit hooks (black, isort, ruff, mypy, eslint, prettier)
- ✅ GitHub Actions CI/CD pipeline with lint/test/build matrix
- ✅ Alembic database migrations setup
- ✅ Feature flags system with environment-based configuration
- ✅ Observability framework with request/trace IDs and structured logging

### Phase 1 - Ingestion & Normalization
- ✅ Upload service with MinIO/S3 integration and presigned URLs
- ✅ Document metadata capture and database models
- ✅ PDF processing with Unstructured + PyMuPDF + Camelot table extraction
- ✅ Audio processing with WhisperX ASR + pyannote speaker diarization
- ✅ XBRL parsing with taxonomy mapping and fact extraction
- ✅ Currency/unit normalization services with exchange rate handling
- ✅ Fiscal calendar alignment for period mapping
- ✅ Data validation framework with financial consistency checks
- ✅ Parquet data lake with bronze/silver/gold layers and snapshot IDs
- ✅ Celery background task system for async processing

### Phase 2 - Indexing & Retrieval (COMPLETED) ✅
- ✅ Embedding service with OpenAI text-embedding-3-large and pgvector indexing
- ✅ Multi-modal chunkers for passages, tables, slides, transcripts with rich metadata
- ✅ Hybrid retrieval combining BM25 sparse search + dense vector search with RRF fusion
- ✅ Table-aware retriever with cell coordinate mapping and header context
- ✅ GraphRAG implementation with entity resolution and relation extraction
- ✅ Confidence scoring ensemble with evidence gating and minimum thresholds
- ✅ Vector query caching with semantic keys and passage-level invalidation
- ✅ Market data adapters with Alpha Vantage + Yahoo Finance failover and rate limiting
- ✅ SQL semantic layer with NL→SQL conversion and row-level security
- ✅ Python sandbox with restricted execution and export capabilities
- ✅ Spreadsheet I/O for Excel/CSV import/export with data joining
- ✅ What-if parameter engine with sensitivity analysis and tornado charts

### Phase 3 - Analytics & Valuation (COMPLETED) ✅
- ✅ DCF valuation suite with 3-stage, H-model, and sensitivity analysis
- ✅ Comparable company analysis with peer curation and winsorization
- ✅ Factor models (CAPM, Fama-French 3/5-factor) with rolling beta analysis
- ✅ Risk analytics including VaR, CVaR, drawdown, and regime analysis
- ✅ Event study framework with abnormal returns and statistical significance
- ✅ SaaS/consumer KPI calculations (LTV/CAC, NDR, cohort analysis)
- ✅ Research Board UI with drag-and-drop cards and interactive analysis
- ✅ Drill-through modals with "Show Work" functionality and citations
- ✅ Multi-format exports (Excel, PowerPoint, PDF) with professional templates
- ✅ Real-time alerts system with WebSocket feeds and explanatory context

### Phase 4 - Security, Governance & Compliance (COMPLETED) ✅
- ✅ Multitenancy isolation with row-level security policies and org_id scoping
- ✅ Secret management system with encrypted API key vault and audit logging
- ✅ MNPI/PII content filtering with risk assessment and quarantine workflow
- ✅ Policy mode framework for informational vs opinionated answer framing
- ✅ Red-team prompt injection protection with pattern detection and sanitization
- ✅ Golden QA dataset with 200+ questions and numeric truth validation
- ✅ Hallucination detection with uncited claim analysis and evidence coverage
- ✅ Comprehensive telemetry with cost/latency tracking and flame graph profiling
- ✅ A/B testing framework for model experiments with statistical analysis

### Phase 5 - Scale & Operations (COMPLETED) ✅
- ✅ Auto-scaling system with worker metrics and intelligent scaling decisions
- ✅ Priority queue management with backpressure and load balancing
- ✅ Enhanced snapshot system with reproducible data lineage and 90-day retention
- ✅ Disaster recovery framework with automated backups and recovery playbooks
- ✅ Load testing suite with 200+ concurrent user scenarios and chaos engineering
- ✅ System monitoring with real-time health checks and performance dashboards
- ✅ Production-ready scaling with SLO/SLA compliance and operational runbooks

### PRODUCTION READY 🚀
All phases complete. The AI Financial Analyst is now enterprise-ready with comprehensive security, governance, scaling, and operational capabilities.
