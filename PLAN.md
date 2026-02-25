# PLAN: Deep Research ODR Alignment — Batch Queries, Brief Phrasing, Stop Heuristics

## Context

Comparison of foundry-mcp's deep research workflow against open_deep_research (ODR) identified three actionable misalignments where adopting ODR patterns improves quality without degradation:

1. **Batch search queries** — ODR lets a researcher send multiple queries in one tool call (`queries: List[str]`), returning one deduplicated result set. Foundry requires one `web_search` call per query, producing separate overlapping result sets that waste researcher context tokens.
2. **First-person phrasing in research brief** — ODR explicitly instructs "Use the First Person — Phrase the request from the perspective of the user." Foundry generates briefs in third person.
3. **Explicit hard-stop heuristic** — ODR has "Always stop after 5 search tool calls if you cannot find the right sources." Foundry says "use up to your budget limit" without an explicit futility escape.

A fourth candidate (compression output structure) was found to already be aligned — the compression system prompt at `compression.py:449-522` already specifies ODR-style structured sections and citation rules.

---

## Phase 1: Batch Search Queries in `web_search` Tool

**Goal**: Allow the researcher to send multiple queries in one tool call, returning one consolidated, deduplicated, summarized result set.

### Files to modify

- `src/foundry_mcp/core/research/models/deep_research.py` — `WebSearchTool` model (line 393)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — `_handle_web_search_tool` (line 957), `_RESEARCHER_SYSTEM_PROMPT` (line 162), ReAct loop budget accounting (line 635)

### Changes

#### 1a. Extend `WebSearchTool` schema

Add optional `queries: list[str]` field alongside existing `query`. A `model_validator` normalizes both forms into `self.queries` so the handler always works with a list. `query="foo"` becomes `queries=["foo"]`. Neither provided raises `ValueError`.

#### 1b. Update `_handle_web_search_tool` for batch dispatch

Replace single-query dispatch with `asyncio.gather` over `_topic_search` for all queries in the batch. Rate limits are respected because `_topic_search` already acquires the semaphore and goes through the provider resilience stack. Cross-query dedup is automatic via shared `seen_urls`/`seen_titles` under `state_lock`. One consolidated `_format_search_results_batch` result set returned to the researcher.

#### 1c. Budget accounting

Change `_handle_web_search_tool` to return `(tool_result_text, queries_charged)` tuple. The outer ReAct loop adjusts `tool_calls_used`/`budget_remaining` by `queries_charged`. Batch size is capped to `budget_remaining` before dispatch so the researcher can't overspend.

#### 1d. Update researcher system prompt

- Document `queries` parameter in `web_search` tool docs
- Replace the first-turn multi-call exception ("On your first turn only, you may issue multiple web_search calls") with batch guidance: "Use the `queries` parameter to search multiple angles at once for initial broad coverage"

---

## Phase 2: First-Person Phrasing in Research Brief

**Goal**: Brief phrased from user's perspective, preserving voice and intent.

### File to modify

- `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` — `_build_brief_system_prompt` (line 203)

### Change

Add rule 5 to the existing numbered list: "Use the first person — Phrase the brief from the perspective of the user (e.g. 'I am looking for...' rather than 'Research the following topic...'). This preserves the user's voice and helps downstream researchers understand intent."

---

## Phase 3: Explicit Hard-Stop Heuristic for Researcher

**Goal**: Add an explicit futility escape matching ODR's pattern.

### File to modify

- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — `_RESEARCHER_SYSTEM_PROMPT` (line 211)

### Change

Add rule 4 to the "Stop Immediately When" section: "**Futility stop**: Always call `research_complete` after 5 search tool calls if you have not found adequate sources — the topic may not be well-covered online. Report what you found and note the gaps."

---

## Verification

```bash
python -m pytest tests/core/research/workflows/deep_research/ -x -q
python -m pytest tests/core/research/workflows/test_topic_research.py -x -q
python -m pytest tests/core/research/workflows/test_synthesis_prompts.py -x -q
python -c "from foundry_mcp.core.research.models.deep_research import WebSearchTool"
```

Add test cases for `WebSearchTool` validation (queries, query, neither, budget capping).
