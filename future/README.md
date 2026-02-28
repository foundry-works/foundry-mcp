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
| [PLAN-2: Academic Tools](PLAN-2-ACADEMIC-TOOLS.md) | Providers, citation tools, strategic primitives | ~960-1350 | ~600-800 | PLAN-0, PLAN-1.1 |
| [PLAN-3: Research Intelligence](PLAN-3-RESEARCH-INTELLIGENCE.md) | Ranking, landscape, export | ~660-940 | ~350-460 | PLAN-0 (soft: PLAN-1, PLAN-2) |
| [PLAN-4: Deep Analysis](PLAN-4-DEEP-ANALYSIS.md) | PDF, citation networks, methodology, MCP bridge | ~1250-1650 | ~420-580 | PLAN-0, PLAN-1, PLAN-2.1 |

**Total**: ~4,070-5,690 LOC implementation + ~2,020-2,720 LOC tests

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
New academic providers are split into **Tier 1** (OpenAlex — highest value, free, broad index, sole hard dependency for downstream features) and **Tier 2** (Unpaywall, Crossref, OpenCitations — optional enrichment). This reduces the "all 4 APIs must work" risk and establishes a minimum viable academic pipeline.

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
PLAN-2.8  Per-Provider Rate Limiting ──────────────────────────────────┤ (parallel)
                                                                        │
PLAN-2.5  Citation Graph Tools ────────────────────────────────────────┤ (needs 2.1)
PLAN-2.6  Strategic Research Primitives ────────────────────────────────┤ (needs 2.5)
PLAN-2.7  Adaptive Provider Selection ─────────────────────────────────┘ (needs 2.1)

PLAN-2.2-4  Tier 2 Providers (Unpaywall, Crossref, OpenCitations)
            (optional, can be added at any point after 2.8)

PLAN-3.1-5  All items (influence ranking, landscape, gaps, tables, export)
            (largely independent, can start after PLAN-0)
            (enhanced by PLAN-1 + PLAN-2 when available)

PLAN-4.1    PDF Analysis (extend existing extractor — can start after PLAN-0)
PLAN-4.2    Citation Network (user-triggered — needs PLAN-2.1)
PLAN-4.3    Methodology Assessment (experimental — independent)
PLAN-4.4    MCP Bridge (CONTINGENT — blocked until server validation)
PLAN-4.5    CORE Provider (independent)
```

### Parallelism opportunities

After PLAN-0 completes, significant parallel work is possible:
- **PLAN-3 item 1** (influence ranking) can start immediately — `supervision_coverage.py` already exists as a standalone module
- **PLAN-1 items 1/2/6** + **PLAN-3 items 2-5** + **PLAN-4 items 1/3/5** can all proceed concurrently after PLAN-0
- **PLAN-2 Tier 1** (OpenAlex) is the critical path item — unlocks items 5-7 and PLAN-4.2

## What's NOT in these plans

- **Cost visibility / budgeting**: Not included. Pressuring agents with token awareness degrades quality. The existing token budget management handles context window fitting.
- **Replacing specialized tools**: These plans complement tools like Undermind and Elicit, not replace them. The goal is composability, not completeness.
- **ERIC, DBLP, or domain-specific databases**: Beyond the current scope. The adaptive provider selection framework (PLAN-2.7) makes adding these later straightforward.
- **Interactive steering**: Real-time researcher control of the pipeline mid-execution. Worth exploring but architecturally complex — deferred.
- **Numeric rigor scores**: Removed from PLAN-4.3. LLM-derived numeric scores from abstracts are unreliable and invite misuse. Qualitative methodology metadata is provided instead.

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

## New strategic items

These plans reorganize features from earlier planning along architectural boundaries (prerequisites → foundations → tools → intelligence → analysis) rather than effort tiers, and add several new strategic items:

| Item | Rationale |
|------|-----------|
| Supervision Refactoring (remaining) | First-round decomposition extraction — coverage and prompt extractions already complete |
| ResearchExtensions Container | State model discipline — one field for all new capabilities |
| Research Profiles | Replaces monolithic `research_mode` with composable, named profiles |
| Provenance Audit Trail | Addresses reproducibility gap — the sharpest criticism of the MCP approach |
| Structured Output Mode | Enables composability — foundry-mcp as orchestration layer, not terminal output |
| Provider Tiers | Reduces operational risk — only OpenAlex is a hard dependency |
| Strategic Research Primitives | Improves researcher quality with BROADEN/DEEPEN/VALIDATE/SATURATE strategies |
| Adaptive Provider Selection | Brief-driven pipeline configuration — no hardcoded provider chains |
| MCP Bridge Pattern | Generic remote MCP integration — contingent on server validation |
