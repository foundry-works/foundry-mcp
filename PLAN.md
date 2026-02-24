# PLAN: Deep Research Workflow — ODR Alignment & Efficiency Improvements

## Context

Comparison of the foundry-mcp deep research workflow against open_deep_research (ODR)
reveals that **leaf-level behaviors** (stop heuristics, retry strategies, prompt philosophy)
are well-aligned after Phases 3-6, but several **structural gaps** remain that affect
research quality and context efficiency:

1. ODR summarizes each search result at retrieval time; we defer all summarization to compression.
   Researchers consume raw content, burning context budget on unstructured text.
2. Supervision directive results are accumulated as raw findings in message history.
   No inline compression means the supervisor's context grows linearly with research volume.
3. Researchers have no novelty signal — they can't see whether a search result is truly new
   or duplicative of earlier finds, weakening the stop heuristics introduced in Phase 5.
4. Supervision message truncation is flat — all message types are truncated equally,
   even though think messages (gap analyses) are critical for multi-round reasoning continuity.
5. The supervisor re-analyzes the same coverage state each round with no delta awareness,
   leading to redundant gap analysis when most sub-queries are already sufficient.

These improvements are ordered by dependency and impact, prioritizing context efficiency
gains that compound across the pipeline.

---

## Phase 1 — Per-Result Summarization at Search Time

**Problem:** Researchers receive raw search content (truncated at 500 chars or full snippets),
consuming context budget on unstructured text. ODR's `summarize_webpage()` (utils.py:175-213)
summarizes each result at retrieval time using a fast model with a 60s timeout, producing
structured `{summary, key_excerpts}` output. We have no equivalent.

**Why it matters:** Per-result summarization reduces researcher context consumption by 60-70%
per search call, allowing more search iterations within the same budget. It also produces
cleaner inputs for the compression phase downstream.

**Changes:**

1. Add a `_summarize_search_result` async helper to `TopicResearchMixin` that takes a
   `ResearchSource` with raw content and returns a structured summary using a fast model
   (same provider as compression, or a dedicated summarization model config).
2. In `_handle_web_search_tool`, after sources are added to state (line ~733), invoke
   `_summarize_search_result` for each source with content longer than a configurable
   threshold (default: 300 chars). Use `asyncio.gather` with per-result timeout (30s).
3. Store the summary in `src.content` (replacing raw content) and set
   `src.metadata["summarized"] = True`. Preserve the original in `src.metadata["raw_content"]`
   for compression-phase fidelity if needed.
4. Fall back to raw snippet/truncation on timeout or failure (matching ODR's fallback pattern).
5. Format the message history entry using `SUMMARY:` block (already handled at line ~751
   via the `src.metadata.get("summarized")` check — just needs the data populated).

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — add summarization helper, modify `_handle_web_search_tool`
- `src/foundry_mcp/core/research/models/deep_research.py` — add optional `summarization_model` config field if not already present

---

## Phase 2 — Inline Compression of Supervision Directive Results

**Problem:** When the supervision delegation loop executes directives (topic research for
follow-up queries), results are accumulated in `state.supervision_messages` as raw
`tool_result` messages. These findings are not compressed until the global compression
phase much later. This inflates supervision message history, accelerating truncation and
losing supervisor reasoning context.

**Why it matters:** Compressing directive results immediately — before appending to supervision
messages — reduces message history growth by ~45% per supervision round. This preserves
more think/delegation context for subsequent rounds.

**Changes:**

1. After `_execute_directives_async` returns in the delegation loop (supervision.py ~line 313),
   iterate over results and invoke `_compress_single_topic_async` for any result that has
   source IDs but no `compressed_findings` yet.
2. Use the compressed output (not raw message history) as the `content` of the `tool_result`
   message appended to `state.supervision_messages`.
3. Guard with a per-result timeout (same as topic compression timeout) so a single slow
   compression doesn't block the delegation loop.
4. If compression fails, fall back to a truncated raw summary (first 800 chars of findings).

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — add inline compression call after directive execution
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` — ensure `_compress_single_topic_async` is usable from supervision context (may need minor interface adjustment)

---

## Phase 3 — Novelty-Tagged Search Results for Researcher Stop Decisions

**Problem:** The researcher's stop heuristics (Phase 5) check for "last 2 searches returned
similar results," but this check relies entirely on the LLM's judgment from reading raw
result text. The researcher has no explicit signal about whether a new result overlaps
with previously found sources for this sub-query.

**Why it matters:** Explicit novelty signals let the researcher make faster, more accurate
stop decisions. This reduces wasted search calls on marginal sources and improves the
signal-to-noise ratio of the stop heuristics.

**Changes:**

1. In `_handle_web_search_tool`, after adding new sources, compute a novelty score for each
   new source against existing sources for this sub-query. Use the existing
   `content_similarity()` or a lightweight text-overlap heuristic (Jaccard on tokens or
   URL domain matching).
2. Annotate each source in the formatted message with a tag:
   - `[NEW]` — content is substantially novel (similarity < 0.3)
   - `[RELATED: <existing-source-title>]` — content overlaps with a prior source (0.3-0.7)
   - `[DUPLICATE]` — near-identical to existing source (> 0.7)
3. Inject a summary line at the top of the search results message:
   `"Novelty: N new, M related, K duplicate out of T results"`
4. Update the think-tool stop-criteria injection (line ~670) to reference novelty tags:
   `"- Check novelty tags: if most recent results are [RELATED] or [DUPLICATE], consider calling research_complete."`

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — add novelty scoring in `_handle_web_search_tool`, update think injection
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` — add lightweight similarity helper if `content_similarity` is not already suitable

---

## Phase 4 — Type-Aware Supervision Message Truncation

**Problem:** `truncate_supervision_messages()` treats all message types equally. When the
supervisor's context fills up, think messages (containing gap analyses and strategic
reasoning) are truncated alongside verbose tool_result messages (containing compressed
findings). Losing think context degrades multi-round supervision quality.

**Why it matters:** The supervisor's gap analyses are the highest-value content for subsequent
rounds — they encode what's missing, what was tried, and what to prioritize. Findings can
be re-derived from state, but reasoning cannot.

**Changes:**

1. Replace flat truncation with type-aware budgeting in `truncate_supervision_messages()`:
   - Reserve 60% of token budget for `think` + `delegation` messages (reasoning)
   - Allocate 40% for `tool_result` / `research_findings` messages (findings)
   - Within each bucket, drop oldest messages first
2. When truncating findings messages, prefer keeping the summary/header portion and dropping
   detailed content (truncate each message body rather than dropping whole messages).
3. Add a `preserve_last_n_thinks` parameter (default: 2) that unconditionally preserves
   the most recent N think messages regardless of budget.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — update `truncate_supervision_messages` or add `_smart_truncate_supervision_messages`

---

## Phase 5 — Coverage Delta Injection for Supervisor Think Step

**Problem:** Each supervision round's think step re-analyzes the full coverage state from
scratch. The supervisor sees current per-query coverage data but has no structured view of
what changed since the last round — which gaps were addressed, which are new, and which
remain unchanged. This leads to redundant analysis and slower convergence.

**Why it matters:** With delta awareness, the supervisor can focus its gap analysis on
what's actually new, reducing think-step latency and improving directive quality for
subsequent rounds.

**Changes:**

1. Before calling `_supervision_think_step` in the post-execution assessment (~line 336),
   compute a coverage delta by comparing current per-query coverage against the coverage
   snapshot from the previous round (stored in supervision history metadata).
2. Build a compact delta summary:
   ```
   Coverage delta (round N-1 → N):
   - query_1: +2 sources, +1 domain (now: 4 sources, 3 domains) — SUFFICIENT
   - query_2: +0 sources — STILL INSUFFICIENT
   - query_3 [NEW]: 1 source from this round's directives
   ```
3. Inject this delta into the think step's user prompt alongside current coverage data.
4. Store the current coverage snapshot in `state.metadata["coverage_snapshots"]` keyed by
   round number, for use in subsequent rounds.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — add `_compute_coverage_delta`, modify `_supervision_think_step` to accept delta, store snapshots in state metadata

---

## Phase 6 — Confidence-Scored Coverage Heuristic

**Problem:** `_assess_coverage_heuristic()` returns a binary `should_continue_gathering`
flag. The decision between "barely sufficient" and "definitively sufficient" is invisible
to downstream consumers and the audit trail. This makes it hard to tune cost-quality
trade-offs or understand why research stopped.

**Why it matters:** A confidence score enables threshold-based configuration (e.g., "require
0.8 confidence for cost-sensitive queries, 0.6 for exploratory ones") and provides better
observability into research quality decisions.

**Changes:**

1. Extend `_assess_coverage_heuristic()` to return multi-dimensional scores:
   - **Source adequacy**: `min(1.0, sources_per_query / min_sources)` averaged across queries
   - **Domain diversity**: `unique_domains / (query_count * 2)` capped at 1.0
   - **Query completion rate**: `completed_queries / total_queries`
2. Compute an overall confidence as the weighted mean (weights configurable).
3. Add `confidence`, `dominant_factors`, and `weak_factors` to the return dict.
4. Use `confidence >= threshold` (default 0.75) instead of the current binary check for
   the `should_continue_gathering` decision.
5. Log the confidence breakdown in the audit event for observability.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` — extend `_assess_coverage_heuristic`, update callers to use confidence threshold
