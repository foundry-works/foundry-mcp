# Academic Deep Research Improvement Plans

Four plans that transform foundry-mcp's deep research from a general-purpose web research pipeline into a composable academic research platform.

## Motivation

Aaron Tay's article ("The agentic researcher," Feb 2026) argues that generic LLMs + MCP servers are competitive with — and in some ways superior to — specialized academic deep research tools (Undermind, Elicit, Consensus). The key advantage is composability: researchers define their own workflows rather than being constrained by vendor-designed pipelines.

These plans respond to that thesis by making foundry-mcp's deep research excellent at academic use cases while preserving its general-purpose capabilities. The strategy: instead of trying to be the entire academic research stack, become the **orchestration layer** that produces composable, structured outputs other tools can consume.

## Plan Overview

| Plan | Theme | Scope | Dependencies |
|------|-------|-------|-------------|
| [PLAN-1: Foundations](PLAN-1-FOUNDATIONS.md) | Profiles, provenance, academic output | ~1000-1400 LOC | None |
| [PLAN-2: Academic Tools](PLAN-2-ACADEMIC-TOOLS.md) | Providers, citation tools, strategic primitives | ~1500-2000 LOC | PLAN-1.1 |
| [PLAN-3: Research Intelligence](PLAN-3-RESEARCH-INTELLIGENCE.md) | Ranking, landscape, export | ~900-1300 LOC | PLAN-1, PLAN-2.1 |
| [PLAN-4: Deep Analysis](PLAN-4-DEEP-ANALYSIS.md) | PDF, citation networks, methodology, MCP bridge | ~2000-3000 LOC | PLAN-1, PLAN-2 |

## Key Design Decisions

### Profiles over modes
Instead of a monolithic `research_mode` enum (`GENERAL` | `ACADEMIC` | `TECHNICAL`), research profiles compose capabilities declaratively. Named profiles (e.g., `systematic-review`, `bibliometric`) provide sensible defaults; per-request overrides provide flexibility. The old `research_mode` parameter continues to work via backward-compatible mapping.

### Provenance by default
Every research session produces a machine-readable audit trail: which providers were queried, which sources were found, how coverage was assessed, what gaps were identified. This addresses the reproducibility criticism of the MCP approach and provides debugging infrastructure for when results are unexpected.

### Structured output for composability
Deep research produces both a markdown report AND structured JSON output (source catalog, findings, gaps, contradictions, landscape metadata, BibTeX/RIS exports). This makes foundry-mcp an orchestration layer — downstream tools (Zotero MCP, visualization tools, reference managers) can consume the structured data.

### Strategic researcher primitives
Topic researchers don't just get new tools (citation search, related papers) — they get strategic guidance on *when* to use different approaches: BROADEN (widen search), DEEPEN (follow citation chains), VALIDATE (corroborate claims), SATURATE (recognize coverage).

### Adaptive provider selection
The brief phase output — which already identifies discipline, scope, and source preferences — drives provider selection. A biomedical query auto-activates PubMed; an education query preferences OpenAlex. No hardcoded provider chains.

## Execution Order

```
PLAN-1.1  Research Profiles ─────────────────────────────────┐
PLAN-1.2  Provenance Audit Trail ────────────────────────────┤ (parallel)
PLAN-1.6  Structured Output Mode ────────────────────────────┤
                                                              │
PLAN-1.3  Literature Review Query Type ──────────────────────┤ (needs 1.1)
PLAN-1.4  APA Citation Formatting ───────────────────────────┤ (needs 1.1)
PLAN-1.5  Academic Brief Enrichment ─────────────────────────┘ (needs 1.1)

PLAN-2.1-4  New Providers (OpenAlex, Unpaywall, Crossref, OpenCitations) ─┐
PLAN-2.8    Per-Provider Rate Limiting ───────────────────────────────────┤ (parallel)
                                                                          │
PLAN-2.5    Citation Graph Tools ─────────────────────────────────────────┤ (needs providers)
PLAN-2.6    Strategic Research Primitives ─────────────────────────────────┤ (needs 2.5)
PLAN-2.7    Adaptive Provider Selection ──────────────────────────────────┘ (needs 2.1-4)

PLAN-3.1-5  All items (influence ranking, landscape, gaps, tables, export)
            (largely independent, can be done in any order)

PLAN-4.1-5  All items (PDF, citation network, methodology, MCP bridge, CORE)
            (largely independent, all opt-in)
```

## What's NOT in these plans

- **Cost visibility / budgeting**: Not included. Pressuring agents with token awareness degrades quality. The existing token budget management handles context window fitting.
- **Replacing specialized tools**: These plans complement tools like Undermind and Elicit, not replace them. The goal is composability, not completeness.
- **ERIC, DBLP, or domain-specific databases**: Beyond the current scope. The adaptive provider selection framework (PLAN-2.7) makes adding these later straightforward.
- **Interactive steering**: Real-time researcher control of the pipeline mid-execution. Worth exploring but architecturally complex — deferred.

## New strategic items

These plans reorganize features from earlier planning along architectural boundaries (foundations → tools → intelligence → analysis) rather than effort tiers, and add several new strategic items:

| Item | Rationale |
|------|-----------|
| Research Profiles | Replaces monolithic `research_mode` with composable, named profiles |
| Provenance Audit Trail | Addresses reproducibility gap — the sharpest criticism of the MCP approach |
| Structured Output Mode | Enables composability — foundry-mcp as orchestration layer, not terminal output |
| Strategic Research Primitives | Improves researcher quality with BROADEN/DEEPEN/VALIDATE/SATURATE strategies |
| Adaptive Provider Selection | Brief-driven pipeline configuration — no hardcoded provider chains |
| MCP Bridge Pattern | Generic remote MCP integration for Scite, Consensus, PubMed, etc. |
