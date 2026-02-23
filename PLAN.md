# Deep Research Enhancements — Adopting open_deep_research Strengths

Date: 2026-02-23
Owner: foundry-mcp core
Status: Draft
Reference: Comparison analysis of `open_deep_research` (LangGraph-based) vs foundry-mcp deep research

## 1. Objective

Adopt proven patterns from the `open_deep_research` project to improve foundry-mcp's deep research quality, token efficiency, and search targeting — without abandoning foundry-mcp's existing strengths (resilience stack, fidelity tracking, multi-provider support, session persistence, audit infrastructure).

## 2. Desired End State

1. Search results are summarized at fetch time using a cheap/fast model, reducing token pressure on downstream phases.
2. Individual topic researchers use forced reflection pauses (think-tool pattern) between search actions, producing better-targeted follow-up queries.
3. Per-topic findings are compressed into citation-rich summaries before aggregation, improving synthesis quality.
4. LLMs can explicitly signal research sufficiency via a `ResearchComplete` tool-call pattern, rather than relying solely on iteration limits.
5. Clarification phase uses a structured binary decision (clarify vs. verify understanding) for better UX.
6. Token limit errors are detected per-provider with progressive truncation recovery.
7. All changes are backward-compatible; existing sessions and configs continue to work.

## 3. Non-Goals

1. Replacing the phase-based architecture with LangGraph.
2. Full supervisor-researcher agent decomposition (considered in §10 as a future phase).
3. Changing the background execution model (daemon threads for sync MCP).
4. Removing or replacing the existing fidelity/budget system — enhancements layer on top.

## 4. Guiding Principles

1. **Additive, not disruptive** — New capabilities gate behind config flags with sensible defaults.
2. **Cheap models for cheap work** — Summarization and reflection don't need the strongest model.
3. **Measure before/after** — Token usage and report quality must be comparable via audit metrics.
4. **Preserve existing test coverage** — All contract tests pass before and after each phase.

---

## 5. Design Rationale

### Why fetch-time summarization?

Raw webpage content from Tavily (up to 50K chars per source) is noisy. Currently this raw content flows into the analysis phase's budget allocator, which must compress/drop sources to fit context. Summarizing at fetch time with a cheap model (e.g., gpt-4.1-mini, haiku) reduces each source to ~25-30% of original size *before* it enters the pipeline. This means:
- The budget allocator can fit more sources at higher fidelity levels.
- Analysis LLM sees cleaner, pre-structured input → better finding extraction.
- Key verbatim excerpts are preserved for citation accuracy.
- Timeout protection (60s per source) with fallback to raw content ensures no degradation.

### Why forced reflection (think-tool)?

The topic research mixin already has a `_topic_reflect` step, but it's optional and the LLM can skip it by generating no tool calls. open_deep_research's key insight: making reflection *the only tool available* at certain points forces deliberate assessment. This produces better follow-up queries and prevents reflexive over-searching. The implementation: after each search iteration, inject a reflection-only turn where search tools are temporarily unavailable.

### Why per-topic compression before aggregation?

Currently, all sources flow into a single analysis phase that processes them together. With 5+ sub-queries × 5+ sources each, this creates a large, undifferentiated source pool. Per-topic compression:
- Groups sources by the sub-query that found them.
- Produces a structured summary per topic with inline citations.
- The analysis/synthesis phases then work with pre-organized, cited material.
- This is analogous to each researcher writing up their notes before the team meeting.

### Why explicit research completion signals?

The topic research loop currently terminates on: (a) max iterations reached, or (b) no sources found. Missing is: "I have enough information." Adding a `research_complete` signal in the reflection response lets the LLM express sufficiency — saving search budget on well-covered topics and spending more on under-covered ones.

### Why progressive token-limit recovery?

The current `execute_llm_call` catches `ContextWindowError` but returns a hard error. open_deep_research's approach: detect the error, truncate input by 10%, retry up to 3 times. This recovers from near-limit cases without failing the entire phase. Provider-specific detection patterns improve accuracy.

---

## 6. Phases

### Phase 1: Fetch-Time Source Summarization

**Goal:** Summarize raw search results at the provider level before they enter the research state.

**Changes:**
- `providers/base.py` — Add optional `summarizer` hook to `SearchProvider` base class.
- `providers/shared.py` — New `SourceSummarizer` class: takes raw content + cheap model config → returns `Summary` (executive summary + key excerpts).
- `providers/tavily.py` — After `search()` returns raw results, run summarizer on each source's `content` field. Parallel execution with 60s timeout per source, fallback to original content.
- `core/research/config.py` — Add `summarization_model` and `summarization_provider` config fields (default: use cheapest available provider).
- `models/sources.py` — Add `raw_content: Optional[str]` field to `ResearchSource` to preserve original alongside summary.

**Key design decisions:**
- Summarization is opt-in via config (`fetch_time_summarization: bool = True`).
- The `Summary` output includes `executive_summary` (narrative) and `key_excerpts` (list of verbatim quotes, max 5). Both stored in `ResearchSource.content` (formatted) and `ResearchSource.metadata["excerpts"]`.
- Parallel summarization across all sources in a batch, bounded by `max_concurrent`.
- Token tracking: summarization tokens counted separately in `PhaseMetrics`.

**Files touched:**
- `src/foundry_mcp/core/research/providers/base.py`
- `src/foundry_mcp/core/research/providers/shared.py` (new or extend)
- `src/foundry_mcp/core/research/providers/tavily.py`
- `src/foundry_mcp/core/research/config.py`
- `src/foundry_mcp/core/research/models/sources.py`
- Tests: `tests/research/test_source_summarization.py` (new)

---

### Phase 2: Forced Reflection in Topic Research

**Goal:** Make the think-tool pattern mandatory between search iterations in the topic research loop.

**Changes:**
- `phases/topic_research.py` — Restructure `_execute_topic_research_async` loop:
  1. Search step (unchanged).
  2. **Mandatory reflection step** — Always call `_topic_reflect` after search (not conditional on source count).
  3. Reflection response includes structured decision: `{continue_searching: bool, refined_query: str | null, research_complete: bool, rationale: str}`.
  4. If `research_complete=True`, exit loop early regardless of remaining budget.
  5. If `continue_searching=False` and not `research_complete`, exit loop (sufficient for now, may revisit in refinement).
- `phases/topic_research.py` — Update reflection prompt to include:
  - Current source count and quality distribution.
  - Explicit instruction to assess sufficiency (3+ relevant sources = likely sufficient).
  - Option to signal completion.
- `_helpers.py` — Add `parse_reflection_decision()` for structured extraction from LLM response.

**Key design decisions:**
- Reflection uses the same provider as topic research (no separate model — keep it simple for Phase 2).
- The `research_complete` signal is advisory: the loop still respects `max_searches` as a hard cap.
- Reflection tokens tracked in `TopicResearchResult.tokens_used`.

**Files touched:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`
- Tests: `tests/research/test_topic_reflection.py` (new)

---

### Phase 3: Per-Topic Compression Before Aggregation

**Goal:** Compress each topic's sources into a structured, citation-rich summary before the analysis phase processes them.

**Changes:**
- `phases/gathering.py` — After all topic researchers complete, run a per-topic compression step:
  1. For each `TopicResearchResult`, collect its sources from `state.sources`.
  2. Build a compression prompt: "Reformat these findings. DO NOT summarize — preserve all relevant information. Add inline citations [1], [2], etc. Output: queries made, comprehensive findings, source list."
  3. Call LLM (compression model/provider from config).
  4. Store compressed output in `TopicResearchResult.compressed_findings` (new field).
- `models/deep_research.py` — Add `compressed_findings: Optional[str]` to `TopicResearchResult`.
- `phases/analysis.py` — When `compressed_findings` is available on topic results, use those as primary input instead of raw source content. This reduces the budget allocator's workload.
- `config.py` — Add `compression_model` and `compression_provider` config fields.

**Key design decisions:**
- Compression is distinct from summarization (Phase 1): summarization reduces individual sources, compression organizes a topic's findings into a coherent narrative with citations.
- Progressive token-limit handling on compression (3 retries, 10% reduction each time) — this is the pattern from open_deep_research.
- If compression fails, fall through to existing analysis pipeline (raw sources).
- Compression prompt explicitly says "not summarizing, just reformatting" to prevent information loss.

**Files touched:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py`
- `src/foundry_mcp/core/research/models/deep_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/analysis.py`
- `src/foundry_mcp/core/research/config.py`
- Tests: `tests/research/test_topic_compression.py` (new)

---

### Phase 4: Structured Clarification Gate

**Goal:** Replace the optional clarification phase with a structured binary decision.

**Changes:**
- `phases/clarification.py` — Use structured output extraction:
  ```
  {need_clarification: bool, question: str, verification: str}
  ```
  - If `need_clarification=True`: return the question to the user (existing flow).
  - If `need_clarification=False`: log `verification` (the LLM's restatement of its understanding) as an audit event, then proceed to planning.
- `phases/_lifecycle.py` — Add `execute_structured_llm_call()` variant that requests JSON-mode output and validates against a schema.

**Key design decisions:**
- The verification text is stored in `state.clarification_constraints` for traceability.
- Structured output uses JSON extraction (existing `extract_json` helper), not a new dependency.
- Retry up to 3 times on parse failure, then fall through to existing behavior (treat as "no clarification needed").

**Files touched:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/clarification.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`
- Tests: `tests/research/test_clarification_structured.py` (new)

---

### Phase 5: Progressive Token-Limit Recovery

**Goal:** Replace hard failures on context window errors with progressive truncation and retry.

**Changes:**
- `phases/_lifecycle.py` — Enhance `execute_llm_call()`:
  1. On `ContextWindowError`, estimate token limit from provider/model metadata.
  2. Truncate the user prompt content by 10%.
  3. Retry up to 3 times with progressive truncation.
  4. Track retry attempts in `PhaseMetrics.metadata["token_limit_retries"]`.
- `providers/base.py` — Add `TOKEN_LIMITS: dict[str, int]` class-level registry mapping known model names to context window sizes. Used for truncation estimation.
- `_helpers.py` — Add `truncate_to_token_estimate(text: str, max_tokens: int) -> str` utility.

**Key design decisions:**
- Token estimation uses 4 chars/token heuristic (same as open_deep_research). Good enough for truncation.
- Only the user prompt is truncated, never the system prompt.
- Provider-specific error detection patterns (OpenAI, Anthropic, Google) added to `ContextWindowError` classification.
- Falls back to existing hard-error behavior if all retries exhausted.

**Files touched:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`
- `src/foundry_mcp/core/research/providers/base.py`
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`
- Tests: `tests/research/test_token_limit_recovery.py` (new)

---

### Phase 6: Multi-Model Cost Optimization

**Goal:** Wire cheap-model routing for summarization and reflection, distinct from the main research model.

**Changes:**
- `config.py` — Formalize the model hierarchy:
  - `research_model` / `research_provider` — Main reasoning (analysis, synthesis). Default: strongest available.
  - `summarization_model` / `summarization_provider` — Fetch-time summarization (Phase 1). Default: cheapest available.
  - `compression_model` / `compression_provider` — Per-topic compression (Phase 3). Default: same as research.
  - `reflection_model` / `reflection_provider` — Think-tool pauses (Phase 2). Default: same as research.
  - `report_model` / `report_provider` — Final synthesis. Default: same as research.
- `phases/_lifecycle.py` — `execute_llm_call()` accepts `role: str` parameter. Resolves provider/model from config based on role.
- All phase callsites updated to pass the appropriate role.

**Key design decisions:**
- Role-based resolution is backward-compatible: if role-specific config is absent, falls back to the phase-level provider/model, then to the global default.
- The `server(action="capabilities")` response includes available model roles.
- Cost tracking per role in `PhaseMetrics.metadata["model_roles"]`.

**Files touched:**
- `src/foundry_mcp/core/research/config.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/analysis.py`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
- Tests: `tests/research/test_model_routing.py` (new)

---

## 7. Phase Dependencies

```
Phase 1 (Fetch-Time Summarization)  ──────┐
                                           ├──→ Phase 6 (Multi-Model Routing)
Phase 2 (Forced Reflection)         ──────┤
                                           │
Phase 3 (Per-Topic Compression)     ──────┘
         depends on Phase 1 (summarized sources are better compression input)

Phase 4 (Structured Clarification)  ─── independent
Phase 5 (Token-Limit Recovery)      ─── independent (but benefits all other phases)
```

**Recommended execution order:** 5 → 1 → 2 → 3 → 4 → 6

Rationale: Phase 5 (token-limit recovery) provides safety for all subsequent phases that add LLM calls. Phase 1 (summarization) reduces token pressure, making Phases 2-3 more effective. Phase 6 (multi-model routing) is best done last since it touches all phase callsites.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Summarization loses critical information | Medium | High | Preserve `raw_content`; excerpts are verbatim quotes; fallback to original on timeout |
| Reflection adds latency without quality gain | Low | Medium | Measure search count before/after; gate behind config flag |
| Per-topic compression hits token limits | Medium | Medium | Progressive truncation (Phase 5); fallback to raw sources |
| Multi-model routing config complexity | Low | Low | Sensible defaults; only override what you need |
| Backward-incompatible state changes | Low | High | New fields are Optional with defaults; existing sessions deserialize cleanly |

---

## 9. Success Metrics

1. **Token efficiency:** ≥30% reduction in total tokens consumed per research session (measured via `PhaseMetrics`).
2. **Search precision:** Fewer wasted search iterations per topic (measure via `TopicResearchResult.searches_performed`).
3. **Report quality:** Subjective assessment — reports use more specific citations and fewer generic statements.
4. **Error recovery:** Token-limit errors result in truncated-but-complete reports instead of hard failures.
5. **Latency:** No more than 20% increase in wall-clock time (summarization parallelism offsets reflection overhead).

---

## 10. Future Considerations (Out of Scope)

### Full Supervisor-Researcher Agent Decomposition

open_deep_research's most architecturally ambitious feature is the supervisor spawning independent researcher subgraphs that run in parallel, each with their own search-reflect loops, returning compressed findings to the supervisor for aggregation.

foundry-mcp's `TopicResearchMixin` is already 80% of this pattern — it runs per-topic ReAct loops in parallel. The gap is:
- Researchers can't see each other's work (isolation).
- Supervisor can't spawn *additional* researchers after reviewing aggregated results (single-round gathering).
- No cross-topic deduplication awareness at the researcher level.

This would require rethinking the phase pipeline into a more graph-like execution model. Worth exploring after Phases 1-6 prove out the incremental improvements.

### MCP Tool Integration in Research

open_deep_research supports dynamically loading MCP tools as research instruments (beyond search). This would let researchers call specialized tools (code execution, database queries, API calls) during research. Potentially powerful but significantly expands the trust boundary.
