# PLAN-CHECKLIST: Deep Research ODR Alignment — Batch Queries, Brief Phrasing, Stop Heuristics

## Phase 1: Batch Search Queries in `web_search` Tool

- [x] **1a.** Extend `WebSearchTool` model with optional `queries: list[str]` field and `model_validator` that normalizes `query` → `queries`
  - File: `src/foundry_mcp/core/research/models/deep_research.py` (class at line 393)
  - Backward compat: `WebSearchTool(query="foo")` → `queries=["foo"]`
  - Validation: `WebSearchTool()` with neither field raises error

- [x] **1b.** Update `_handle_web_search_tool` to dispatch batch queries via `asyncio.gather` over `_topic_search`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (method at line 957)
  - Parallel execution through existing semaphore + provider rate limiter
  - Cross-query dedup via shared `seen_urls`/`seen_titles` under `state_lock`
  - Single consolidated `_format_search_results_batch` result set returned

- [x] **1c.** Refactor budget accounting: `_handle_web_search_tool` returns `(tool_result_text, queries_charged)` tuple
  - Update caller in ReAct loop (line 635-655) to unpack tuple and adjust `tool_calls_used`/`budget_remaining` by `queries_charged`
  - Cap batch size to `budget_remaining` before dispatch

- [x] **1d.** Update `_RESEARCHER_SYSTEM_PROMPT` with batch query documentation and first-turn guidance
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (line 162)
  - Document `queries` parameter in tool docs
  - Replace first-turn exception with batch guidance

- [x] **1e.** Add unit tests for batch query flow
  - `WebSearchTool` validation (queries, query, neither)
  - Budget charging (batch of 3 charges 3)
  - Budget capping (batch of 5 with budget=2 executes 2)

## Phase 2: First-Person Phrasing in Research Brief

- [x] **2a.** Add first-person instruction to `_build_brief_system_prompt`
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (line 231)
  - Add rule 5: "Use the first person — phrase from user's perspective"

## Phase 3: Researcher Hard-Stop Heuristic

- [x] **3a.** Add futility stop rule to `_RESEARCHER_SYSTEM_PROMPT` Stop Immediately section
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (line 211)
  - Rule 4: "Always call research_complete after 5 search tool calls if sources not found"

## Verification

- [x] Run `pytest tests/core/research/workflows/deep_research/ -x -q`
- [x] Run `pytest tests/core/research/workflows/test_topic_research.py -x -q`
- [x] Run `pytest tests/core/research/workflows/test_synthesis_prompts.py -x -q`
- [x] Verify no import errors: `python -c "from foundry_mcp.core.research.models.deep_research import WebSearchTool"`
