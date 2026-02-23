# PLAN: Align Deep Research Compression with open_deep_research

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23
**Reference:** `~/GitHub/open_deep_research`

---

## Problem Statement

The foundry-mcp compression implementation is architecturally misaligned with open_deep_research. The two systems have a similar two-level design but foundry-mcp's L2 (per-topic compression) feeds truncated raw sources instead of the full research context, defeating the purpose.

### Current State

```
open_deep_research                       foundry-mcp
──────────────────                       ───────────
L1: summarize_webpage()                  SourceSummarizer.summarize_source()
  input:  raw_content[:50,000 chars]       input:  source.content (no limit)
  output: <summary> + <key_excerpts>       output: executive_summary + excerpts
  model:  gpt-4.1-mini (cheap)            model:  configurable via role
  when:   inside tavily_search()           when:   inside tavily search()
  ✓ Reasonably aligned                    ✓ Reasonably aligned

L2: compress_research()                  CompressionMixin._compress_topic_findings_async()
  input:  FULL message history             input:  source title/url/content[:2000]
          (all tool calls, reflections,    loses:  reflection notes, refined queries,
           search results, reasoning)              reasoning, search iteration context
  output: queries + findings + sources     output: queries + findings + sources
  model:  gpt-4.1 (strong)                model:  configurable via role
  when:   after researcher finishes        when:   after gathering finishes
          all ReAct iterations                     (before analysis)
  ✗ MISALIGNED — wrong input              ✗ MISALIGNED — wrong input
```

### Key Gaps

1. **L2 input is wrong.** Compression receives re-truncated raw source content (2000 chars each) instead of the full research context accumulated by each topic researcher (search results, reflections, refined queries, rationale).

2. **L2 throws away the ReAct context.** Each topic researcher runs search → reflect → refine cycles. The reflections, query refinements, and early-completion rationale are stored in `TopicResearchResult` but never fed into compression. The compression prompt only sees source title/url/content.

3. **L1 has no input char limit.** open_deep_research caps `raw_content` at 50,000 chars before summarization. foundry-mcp's `SourceSummarizer` has no such cap — it passes `source.content` directly. For most Tavily results this is fine (snippets are small), but extracted full-page content could be unbounded.

---

## Phase 1: Fix L2 Compression Input

The core fix: compression should operate on the **full topic research context**, not re-truncated raw sources.

### 1.1 Build the full topic research context for compression

Each `TopicResearchResult` already stores:
- `sub_query_id` → the original query
- `refined_queries` → all query refinements from ReAct loop
- `reflection_notes` → assessments from each reflection step
- `early_completion` / `completion_rationale` → why research stopped
- `source_ids` → which sources were found

The compression prompt should include ALL of this, not just source content. Rewrite `_compress_topic_findings_async` to build a prompt like:

```
Research sub-query: {original_query}

Search iterations:
  1. Query: "{original_query}" → {n} sources found
     Reflection: {reflection_notes[0]}
  2. Query: "{refined_queries[0]}" → {n} sources found
     Reflection: {reflection_notes[1]}
  ...
  Completion: {completion_rationale}

Sources ({n} total):
  [1] Title: ...
      URL: ...
      Content: {full content, not truncated to 2000 chars}
  [2] ...

Clean up these findings preserving all relevant information with inline citations.
```

**Files:** `compression.py`

### 1.2 Raise source content limit to match open_deep_research

Replace `_COMPRESSION_SOURCE_CHAR_LIMIT = 2000` with a configurable `deep_research_compression_max_content_length` defaulting to **50,000 chars** (matching open_deep_research's `max_content_length`).

**Files:** `compression.py`, `research.py` (add config field)

### 1.3 Align compression prompt with open_deep_research

Rewrite the system prompt to match open_deep_research's compress_research directives:

- "DO NOT summarize. Return raw information in a cleaner format."
- "Preserve ALL relevant information verbatim."
- "A later LLM will merge this with other topic reports — don't lose sources."
- Output format: Queries Made → Comprehensive Findings → Source List
- Inline citations [1], [2] numbered sequentially
- "It is extremely important that any information even remotely relevant is preserved verbatim."

The current prompt says "DO NOT summarize or remove information" but the 2000-char truncation contradicts this. With the raised limit and full context, the prompt becomes honest.

**Files:** `compression.py`

### 1.4 Use `execute_llm_call` instead of duplicated retry logic

The compression mixin currently re-implements ContextWindowError → truncate → retry. Refactor to use the shared `execute_llm_call` from `_lifecycle.py` which already handles this.

**Files:** `compression.py`

---

## Phase 2: Fix L1 Summarization Input Limit

### 2.1 Add max content length cap to SourceSummarizer

open_deep_research truncates `raw_content[:max_content_length]` (50,000 chars) before passing to summarization. foundry-mcp's `SourceSummarizer.summarize_source()` has no such cap.

Add a `max_content_length` parameter to `SourceSummarizer.__init__()` defaulting to 50,000. Truncate in `summarize_source()` before building the prompt.

**Files:** `shared.py` (SourceSummarizer), `gathering.py` (_attach_source_summarizer)

### 2.2 Add `deep_research_max_content_length` config field

Add to `ResearchConfig` so the cap is configurable, matching open_deep_research's pattern. Wire it through `_attach_source_summarizer`.

**Files:** `research.py`, `gathering.py`

---

## Phase 3: Update Tests

### 3.1 Update compression tests for new prompt/input structure

`test_topic_compression.py` tests need to verify:
- Full ReAct context (reflections, refined queries, rationale) appears in compression prompt
- Source content uses the raised char limit
- Prompt matches the open_deep_research-aligned directives

### 3.2 Add summarization input limit tests

`test_source_summarization.py` needs tests for:
- Content exceeding `max_content_length` is truncated before summarization
- Configurable limit is respected

### 3.3 Verify analysis prompt still works with richer compressed findings

`test_cross_phase_integration.py` should verify that analysis correctly consumes the new compression output format (which now includes queries-made and research-iteration context).

---

## Phase 4: Cleanup from Code Review

Remaining review findings not related to compression alignment:

### 4.1 Fix `_load_model_token_limits()` path resolution
Use `foundry_mcp.config.__file__` instead of `parents[5]`.

### 4.2 Extract `safe_resolve_model_for_role()` helper
Replace 5+ try/except sites with a single helper in `_helpers.py`.

### 4.3 Remove unused `provider_hint` from `_CONTEXT_WINDOW_ERROR_PATTERNS`
The hint field is never used — remove it or use it.

### 4.4 Add sync test for `_FALLBACK_MODEL_TOKEN_LIMITS` vs JSON
Prevent the two sources of truth from diverging.

### 4.5 Move per-call imports to module level in `_lifecycle.py`
Avoid import overhead on every `execute_llm_call` invocation.

---

## References

- **open_deep_research L1:** `~/GitHub/open_deep_research/src/open_deep_research/utils.py` — `tavily_search()` (lines 44-136), `summarize_webpage()` (lines 175-213)
- **open_deep_research L2:** `~/GitHub/open_deep_research/src/open_deep_research/deep_researcher.py` — `compress_research()` (lines 511-585)
- **open_deep_research prompts:** `~/GitHub/open_deep_research/src/open_deep_research/prompts.py` — `summarize_webpage_prompt` (lines 311-368), `compress_research_system_prompt` (lines 186-222)
- **open_deep_research config:** `~/GitHub/open_deep_research/src/open_deep_research/configuration.py` — `max_content_length=50000`, `summarization_model=gpt-4.1-mini`, `compression_model=gpt-4.1`
