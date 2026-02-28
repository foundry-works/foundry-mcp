# Academic Deep Research Improvement Plans

Five plans that transform foundry-mcp's deep research from a general-purpose web research pipeline into a composable academic research platform.

## Motivation

Aaron Tay's article ("The agentic researcher," Feb 2026) argues that generic LLMs + MCP servers are competitive with — and in some ways superior to — specialized academic deep research tools (Undermind, Elicit, Consensus). The key advantage is composability: researchers define their own workflows rather than being constrained by vendor-designed pipelines.

These plans respond to that thesis by making foundry-mcp's deep research excellent at academic use cases while preserving its general-purpose capabilities. The strategy: instead of trying to be the entire academic research stack, become the **orchestration layer** that produces composable, structured outputs other tools can consume.

## Plan Overview

| Plan | Theme | Impl LOC | Test LOC | Dependencies |
|------|-------|----------|----------|-------------|
| [PLAN-0: Prerequisites](PLAN-0-PREREQUISITES.md) | Remaining supervision refactoring, state model architecture | ~200-400 | ~200-300 | None |
| [PLAN-1: Foundations](PLAN-1-FOUNDATIONS.md) | Profiles, provenance, academic output | ~1000-1350 | ~450-580 | PLAN-0 |
| [PLAN-2: Academic Tools](PLAN-2-ACADEMIC-TOOLS.md) | OpenAlex, Crossref, citation tools, strategic primitives | ~580-850 | ~370-500 | PLAN-0, PLAN-1.1 |
| [PLAN-3: Research Intelligence](PLAN-3-RESEARCH-INTELLIGENCE.md) | Ranking, landscape, export | ~660-940 | ~350-460 | PLAN-0 (soft: PLAN-1, PLAN-2) |
| [PLAN-4: Deep Analysis](PLAN-4-DEEP-ANALYSIS.md) | PDF, citation networks, methodology | ~650-900 | ~240-360 | PLAN-0, PLAN-1, PLAN-2.1 |

**Total**: ~3,090-4,440 LOC implementation + ~1,610-2,200 LOC tests

> **Revised Feb 2026**: Scope reduced ~25% after tool evaluation. Unpaywall, OpenCitations, and CORE providers removed (redundant with OpenAlex). MCP Bridge reframed as external documentation. See [revision notes](#revisions-from-feb-2026-tool-evaluation).

## Key Design Decisions

### Prerequisites first
The supervision module (originally 3,445 lines across three files: `supervision.py` at 2,174, `supervision_coverage.py` at 424, and `supervision_prompts.py` at 847) is modified by 3 of 4 plans. The high-value extractions (coverage helpers, prompt builders) are already done. PLAN-0 completes the remaining refactoring (first-round decomposition extraction) before feature work begins. Similarly, `DeepResearchState` (1,659 lines) gains ~7 new field groups — PLAN-0 introduces a `ResearchExtensions` container to prevent state model bloat.

### Profiles over modes
Instead of a monolithic `research_mode` enum (`GENERAL` | `ACADEMIC` | `TECHNICAL`), research profiles compose capabilities declaratively. Named profiles (e.g., `systematic-review`, `bibliometric`) provide sensible defaults; per-request overrides provide flexibility. The old `research_mode` parameter continues to work via backward-compatible mapping.

### Provenance by default
Every research session produces a machine-readable audit trail: which providers were queried, which sources were found, how coverage was assessed, what gaps were identified. This addresses the reproducibility criticism of the MCP approach and provides debugging infrastructure for when results are unexpected.

### Structured output for composability
Deep research produces both a markdown report AND structured JSON output (source catalog, findings, gaps, contradictions, landscape metadata, BibTeX/RIS exports). This makes foundry-mcp an orchestration layer — downstream tools (Zotero MCP, visualization tools, reference managers) can consume the structured data.

### Provider tiers
New academic providers are split into **Tier 1** (OpenAlex — highest value, broad index, sole hard dependency for downstream features) and **Tier 2** (Crossref — optional metadata enrichment). Unpaywall, OpenCitations, and CORE were evaluated and removed: Unpaywall is redundant (OpenAlex uses the same OA engine since the Walden merger), OpenCitations is redundant (OpenAlex handles citation traversal at 100 req/s vs OpenCitations' 3 req/s), and CORE's rate limits (5 req/10s) make it impractical when OpenAlex provides OA PDF URLs for the same content.

### External MCP ecosystem over built-in bridges
Rather than building bespoke MCP bridge wrappers for remote academic servers (Scite, Consensus, PubMed), these are documented as external MCP servers users configure directly. Scite MCP (launched Feb 26, 2026) requires a paid subscription (~16 EUR/mo). Consensus MCP offers a free tier (3 results/query). PubMed has production-grade community MCP servers (fully free). This avoids coupling the pipeline to paid services while preserving composability.

### Relaxed dependency chains
PLAN-3 (Research Intelligence) works with whatever metadata is available from existing providers. It doesn't hard-require profiles or OpenAlex — it produces richer output when they're present but functions with just Semantic Scholar metadata. This decouples it from the PLAN-1/PLAN-2 critical path.

### Strategic researcher primitives
Topic researchers don't just get new tools (citation search, related papers) — they get strategic guidance on *when* to use different approaches: BROADEN (widen search), DEEPEN (follow citation chains), VALIDATE (corroborate claims), SATURATE (recognize coverage).

### Adaptive provider selection
The brief phase output — which already identifies discipline, scope, and source preferences — drives provider selection. A biomedical query auto-activates PubMed; an education query preferences OpenAlex. No hardcoded provider chains.

### Extend, don't duplicate
PLAN-4's PDF analysis extends the existing production-ready `pdf_extractor.py` (833 lines, with SSRF protection, pypdf + pdfminer.six fallback, Prometheus metrics) rather than creating a parallel module with a different library.

## Execution Order

```
PLAN-0.1  Complete supervision refactoring ────────────────┐
PLAN-0.2  ResearchExtensions container model ───────────────┤ (parallel)
                                                             │
PLAN-1.1  Research Profiles ─────────────────────────────────┤
PLAN-1.2  Provenance Audit Trail ────────────────────────────┤ (parallel with 1.1)
PLAN-1.6  Structured Output Mode ────────────────────────────┤ (parallel with 1.1)
                                                              │
PLAN-1.3  Literature Review Query Type ──────────────────────┤ (needs 1.1)
PLAN-1.4  APA Citation Formatting ───────────────────────────┤ (needs 1.1)
PLAN-1.5  Academic Brief Enrichment ─────────────────────────┘ (needs 1.1)

PLAN-2.1  OpenAlex Provider (Tier 1) ──────────────────────────────────┐
PLAN-2.2  Crossref Provider (Tier 2, metadata enrichment) ────────────┤ (parallel)
PLAN-2.6  Per-Provider Rate Limiting ──────────────────────────────────┤ (parallel)
                                                                        │
PLAN-2.3  Citation Graph Tools ────────────────────────────────────────┤ (needs 2.1)
PLAN-2.4  Strategic Research Primitives ────────────────────────────────┤ (needs 2.3)
PLAN-2.5  Adaptive Provider Selection ─────────────────────────────────┘ (needs 2.1)

PLAN-3.1-5  All items (influence ranking, landscape, gaps, tables, export)
            (largely independent, can start after PLAN-0)
            (enhanced by PLAN-1 + PLAN-2 when available)

PLAN-4.1    PDF Analysis (extend existing extractor — can start after PLAN-0)
PLAN-4.2    Citation Network (user-triggered — needs PLAN-2.1, uses OpenAlex)
PLAN-4.3    Methodology Assessment (experimental — independent)
```

### Parallelism opportunities

After PLAN-0 completes, significant parallel work is possible:
- **PLAN-3 item 1** (influence ranking) can start immediately — `supervision_coverage.py` already exists as a standalone module
- **PLAN-1 items 1/2/6** + **PLAN-3 items 2-5** + **PLAN-4 items 1/3** can all proceed concurrently after PLAN-0
- **PLAN-2 Tier 1** (OpenAlex) is the critical path item — unlocks citation graph tools and PLAN-4.2

## What's NOT in these plans

- **Cost visibility / budgeting**: Not included. Pressuring agents with token awareness degrades quality. The existing token budget management handles context window fitting.
- **Replacing specialized tools**: These plans complement tools like Undermind and Elicit, not replace them. The goal is composability, not completeness.
- **ERIC, DBLP, or domain-specific databases**: Beyond the current scope. The adaptive provider selection framework (PLAN-2.5) makes adding these later straightforward.
- **Interactive steering**: Real-time researcher control of the pipeline mid-execution. Worth exploring but architecturally complex — deferred.
- **Numeric rigor scores**: Removed from PLAN-4.3. LLM-derived numeric scores from abstracts are unreliable and invite misuse. Qualitative methodology metadata is provided instead.
- **Built-in MCP bridge wrappers**: Scite, Consensus, and PubMed MCP servers exist but are documented as external user-configured servers rather than embedded in the pipeline. This avoids coupling to paid services (Scite) and keeps the dependency surface minimal.
- **Unpaywall / OpenCitations / CORE providers**: Evaluated and removed — see [Feb 2026 tool evaluation](#revisions-from-feb-2026-tool-evaluation).

## Revisions from review

| Change | Rationale |
|--------|-----------|
| Added PLAN-0 (Prerequisites) | Supervision module is a merge bottleneck — refactor before feature work |
| PLAN-0 scope reduced | Coverage helpers and prompt builders already extracted to `supervision_coverage.py` (424 lines) and `supervision_prompts.py` (847 lines); only first-round decomposition extraction remains |
| ResearchExtensions container | Prevents `DeepResearchState` (1,659 lines) bloat from ~7 new field groups |
| Provider tiers in PLAN-2 | OpenAlex is Tier 1 (required); Unpaywall/Crossref/OpenCitations are Tier 2 (optional enrichment) |
| PLAN-2 resilience config aligned | Rate limiting uses existing `ProviderResilienceConfig` pattern, not flat RPS dict |
| Relaxed PLAN-3 dependencies | Works with existing Semantic Scholar metadata; upstream plans enhance but don't gate |
| PLAN-3/4 state fields on extensions | ResearchLandscape, CitationNetwork, MethodologyAssessments stored on `state.extensions`, not directly on DeepResearchState |
| PDF extractor reconciliation | PLAN-4.1 extends existing `pdf_extractor.py` (834 lines) instead of creating a duplicate with `pymupdf` |
| Citation network deferred to user-triggered | 30+ API calls is too expensive for automatic pipeline step |
| Methodology scoring demoted | Removed numeric rigor score; kept qualitative metadata extraction |
| MCP bridge contingent | Speculative server URLs must be validated before building bridge infrastructure |
| Testing LOC budgeted | Each plan includes testing estimates (~30-40% of implementation LOC) |

## Revisions from Feb 2026 tool evaluation

| Change | Rationale |
|--------|-----------|
| Unpaywall provider removed | Redundant — OpenAlex uses same OA engine since Walden merger (late 2025). `open_access.oa_url` provides identical data. |
| OpenCitations provider removed | Redundant — OpenAlex handles citation traversal at 100 req/s (vs OpenCitations' 3 req/s). Self-citation metadata is niche. |
| CORE provider removed | Low throughput (5 req/10s), largely redundant with OpenAlex OA coverage and PDF URLs. |
| OpenAlex auth model updated | API key now required (free, Feb 13 2026). Polite pool discontinued. Usage-based pricing: $1/day free budget covers ~1,000 searches. |
| OpenAlex field changes noted | Walden rewrite removed/renamed fields: `concepts` → `topics`, `grants` → `awards`, several fields dropped. Defensive parsing required. |
| MCP Bridge reframed | Scite MCP (paid, ~16 EUR/mo), Consensus MCP (free limited), PubMed MCP (free) exist — documented as external user-configured servers, not built-in wrappers. |
| Citation network simplified | Uses OpenAlex exclusively (not OpenCitations). 100 req/s makes interactive graph building practical. |
| Crossref rate limits updated | Dec 2025 restructure: simple requests (DOI lookups) get higher limits, complex (filtered) get lower. Favorable for our use case. |
| PLAN-2 scope reduced ~40% | 4 providers → 2 providers. ~380-500 LOC removed from implementation, ~160-220 LOC from tests. |
| PLAN-4 scope reduced ~48% | MCP Bridge and CORE removed. ~450-600 LOC removed from implementation, ~160-220 LOC from tests. |
| External MCP ecosystem documented | PubMed (cyanheads), Consensus, Scite, Docling, AI2 Asta S2 MCP, OpenAlex community MCPs — documented as composable external servers. |

## New strategic items

These plans reorganize features from earlier planning along architectural boundaries (prerequisites → foundations → tools → intelligence → analysis) rather than effort tiers, and add several new strategic items:

| Item | Rationale |
|------|-----------|
| Supervision Refactoring (remaining) | First-round decomposition extraction — coverage and prompt extractions already complete |
| ResearchExtensions Container | State model discipline — one field for all new capabilities |
| Research Profiles | Replaces monolithic `research_mode` with composable, named profiles |
| Provenance Audit Trail | Addresses reproducibility gap — the sharpest criticism of the MCP approach |
| Structured Output Mode | Enables composability — foundry-mcp as orchestration layer, not terminal output |
| Provider Tiers | Reduces operational risk — OpenAlex (Tier 1) + Crossref (Tier 2) only |
| Strategic Research Primitives | Improves researcher quality with BROADEN/DEEPEN/VALIDATE/SATURATE strategies |
| Adaptive Provider Selection | Brief-driven pipeline configuration — no hardcoded provider chains |
| External MCP Ecosystem | Document Scite, Consensus, PubMed, Docling, Asta MCP servers as composable externals |
