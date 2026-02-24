# PLAN-CHECKLIST: Deep Research — Config Consolidation & Tool-Calling Researchers

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24

---

## Phase 1: Collapse Config Flags (Delete Dead Paths) ✅ COMPLETE

### 1.1 — Audit and catalog all flag references ✅

- [x] **1.1.1** Grep codebase for each flag name; record every file that references it
- [x] **1.1.2** Catalog which code paths each flag gates (production path vs dead path)
- [x] **1.1.3** Identify tests that exercise dead paths (will be deleted or rewritten)

### 1.2 — Delete dead phase code ✅

- [x] **1.2.1** Remove ANALYSIS phase (enum, transitions, flags, phase code)
- [x] **1.2.2** Remove global COMPRESSION phase (keep per-topic `_compress_single_topic_async`)
- [x] **1.2.3** Remove REFINEMENT phase (loop, transitions, enum, flags)
- [x] **1.2.4** Remove per-phase reflection pauses (`async_think_pause()` calls, flag, prompts)
- [x] **1.2.5** Remove legacy query-generation delegation model
- [x] **1.2.6** Remove standalone PLANNING phase from default path
- [x] **1.2.7** Remove post-gathering Tavily extract (`tavily_extract_in_deep_research` flag)
- [x] **1.2.8** Remove `deep_research_enable_topic_agents` flag and gated code
- [x] **1.2.9** Remove `deep_research_digest_policy` flag and proactive/lazy branching

### 1.3 — Hardwire always-on behavior ✅

- [x] **1.3.1** Remove `deep_research_enable_brief` flag — brief phase always runs
- [x] **1.3.2** Remove `deep_research_fetch_time_summarization` flag — summarization always active

### 1.4 — Simplify workflow_execution.py ✅

- [x] **1.4.1** Rewrite `_determine_next_phase()` as linear sequence: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS → COMPLETE
- [x] **1.4.2** Remove iteration loop (was for REFINEMENT)
- [x] **1.4.3** Remove dead phase transitions and state checks

### 1.5 — Clean up config and models ✅

- [x] **1.5.1** Remove all deleted flags from `research.py` config class
- [x] **1.5.2** Remove deleted flags from `from_toml_dict()` parsing
- [x] **1.5.3** Clean up `DeepResearchPhase` enum — remove dead phase values
- [x] **1.5.4** Remove any config validation referencing deleted flags
- [x] **1.5.5** Clean up orchestration.py — remove dead AgentRole values, dead PHASE_TO_AGENT entries, dead elif branches

### 1.6 — Update tests ✅

- [x] **1.6.1** Delete tests for removed phases (6 test files, ~4,500 lines):
  - `test_global_compression.py` — global compression phase
  - `test_planning_critique.py` — planning self-critique
  - `test_contradiction_detection.py` — analysis contradiction detection
  - `test_reflection.py` — per-phase reflection pauses
  - `test_deep_research_digest.py` — analysis digest pipeline
  - `test_proactive_digest.py` — proactive digest policy
- [x] **1.6.2** Fix 22 test files — replace dead phase refs (PLANNING→BRIEF, ANALYSIS→SYNTHESIS, etc.), remove dead config attr refs, delete test methods for removed features
- [x] **1.6.3** Verify production pipeline tests pass

### 1.7 — Final verification ✅

- [x] **1.7.1** Grep entire codebase for each deleted phase name — zero references remain in tests
- [x] **1.7.2** Run full test suite — 6107 passed, 0 failures, 48 skipped
- [x] **1.7.3** Committed in two commits: `3f67676` (source changes) + `a1a1640` (cleanup)

---

## Phase 2: Tool-Calling Researchers (Merge Reflect + Think into ReAct Agent)

### 2.1 — Define researcher tool schemas

- [ ] **2.1.1** Add `WebSearchTool` schema to `models/deep_research.py`:
  ```python
  class WebSearchTool(BaseModel):
      query: str
      max_results: int = 5
  ```
- [ ] **2.1.2** Add `ExtractContentTool` schema:
  ```python
  class ExtractContentTool(BaseModel):
      urls: list[str]  # max 2 URLs per call
  ```
- [ ] **2.1.3** Add `ThinkTool` schema:
  ```python
  class ThinkTool(BaseModel):
      reasoning: str
  ```
- [ ] **2.1.4** Add `ResearchCompleteTool` schema:
  ```python
  class ResearchCompleteTool(BaseModel):
      summary: str
  ```
- [ ] **2.1.5** Create tool registry mapping tool names to schemas and dispatch functions

### 2.2 — Create researcher system prompt

- [ ] **2.2.1** Write system prompt adapted from open_deep_research's `research_system_prompt`:
  - Role: "You are a focused research agent assigned to investigate: {topic}"
  - Strategy guidance: "Start with broader searches, then narrow based on findings"
  - Think guidance: "Use Think to pause and assess before deciding next steps"
  - Completion guidance: "Call ResearchComplete when findings address the research question"
  - Budget visibility: "You have {remaining} of {total} tool calls remaining"
  - Scaling guidance: "Simple queries: 2-3 searches. Complex topics: up to budget limit"
  - Date context: "Today's date is {date}"
- [ ] **2.2.2** Include source quality expectations:
  - "Prefer primary sources, official documentation, and peer-reviewed content"
  - "Seek diverse perspectives — multiple domains and viewpoints"
- [ ] **2.2.3** Include search result format documentation:
  - Explain that search results are pre-summarized with `<summary>` and `<key_excerpts>` tags
  - Explain that ExtractContent returns full page content in markdown

### 2.3 — Implement ReAct loop

- [ ] **2.3.1** Refactor `_execute_topic_research_async()`:
  - Initialize message history: `[system_prompt, user_assignment]`
  - Enter ReAct loop: `while tool_calls_remaining > 0`
  - Each iteration: one LLM call with tools → process tool calls → append results
  - Exit conditions: `ResearchComplete` called, no tool calls returned, budget exhausted
- [ ] **2.3.2** Implement tool call dispatch:
  - Parse tool calls from LLM response (provider-specific format)
  - Route to handler by tool name:
    - `WebSearch` → `_topic_search()` with query from tool call
    - `ExtractContent` → `_topic_extract()` with URLs from tool call
    - `Think` → log reasoning, return acknowledgment
    - `ResearchComplete` → record summary, set `early_completion=True`, break
  - Format tool results as tool response messages
- [ ] **2.3.3** Implement Think constraint:
  - If Think and other tools called in same turn: execute Think first
  - Log Think reasoning before executing action tools
- [ ] **2.3.4** Implement tool call budget tracking:
  - Each tool call (WebSearch, ExtractContent) decrements budget by 1
  - Think calls do NOT count against budget (same as open_deep_research)
  - ResearchComplete does NOT count against budget
  - When budget reaches 0: stop loop, log "budget exhausted"
- [ ] **2.3.5** Implement message history management:
  - Append each tool call + result to history
  - Cap history at reasonable token limit (truncate oldest tool results if needed)
  - Each researcher has independent history (no cross-contamination)

### 2.4 — Wire tool calls to existing infrastructure

- [ ] **2.4.1** `WebSearch` dispatch:
  - Call existing `_topic_search()` with tool call's query
  - Return formatted search results (already summarized via fetch-time summarization)
  - Format: `Found {N} sources:\n\n--- SOURCE 1: {title} ---\nURL: ...\nSUMMARY: ...`
  - Add sources to shared state under `state_lock` (existing dedup logic)
- [ ] **2.4.2** `ExtractContent` dispatch:
  - Call existing `_topic_extract()` with tool call's URLs
  - Return extracted markdown content
  - Add extracted sources to shared state under `state_lock`
  - Gate on `deep_research_enable_extract` config (preserved tuning knob)
- [ ] **2.4.3** `Think` dispatch:
  - Log reasoning at INFO level with topic context
  - Return simple acknowledgment: "Reflection recorded."
  - Track think step content for inline compression context
- [ ] **2.4.4** `ResearchComplete` dispatch:
  - Record summary in `TopicResearchResult.completion_rationale`
  - Set `early_completion = True`
  - Return confirmation: "Research complete. Findings recorded."

### 2.5 — Update LLM provider integration

- [ ] **2.5.1** Add tool-calling support to provider request:
  - Extend `ProviderRequest` (or create specialized request) with tools list
  - Map Pydantic tool schemas to provider-specific tool format:
    - Anthropic: `tool_use` blocks
    - OpenAI: `function` definitions
    - Other providers: provider-specific adaptation
- [ ] **2.5.2** Parse tool calls from provider response:
  - Extract tool call name, arguments, and ID from response
  - Handle multiple tool calls per response
  - Handle responses with both text and tool calls
- [ ] **2.5.3** Add provider capability check:
  - `provider.supports_tool_calling() -> bool`
  - Gate ReAct loop on this check
  - Fall back to structured-JSON approach when not supported

### 2.6 — Delete merged code

- [ ] **2.6.1** Delete `_topic_reflect()` method from `topic_research.py`
- [ ] **2.6.2** Delete `_topic_think()` method from `topic_research.py`
- [ ] **2.6.3** Delete `_parse_reflection_decision()` and related parsing functions
- [ ] **2.6.4** Delete `_format_topic_sources_for_reflection()` method
- [ ] **2.6.5** Delete rigid threshold rules from any remaining prompt code:
  - "STOP IMMEDIATELY if 3+ sources FROM 2+ DISTINCT DOMAINS"
  - "ADEQUATE if 2+ sources but same domain"
  - Source-count / domain-count injection logic
- [ ] **2.6.6** Clean up imports referencing deleted functions

### 2.7 — Preserve existing behavior

- [ ] **2.7.1** Inline per-topic compression still runs after ReAct loop completes
  - `_compress_single_topic_async()` unchanged
  - Input: accumulated sources from researcher's tool calls
  - Output: `compressed_findings` on `TopicResearchResult`
- [ ] **2.7.2** Source deduplication unchanged:
  - URL dedup, title normalization, content similarity — all preserved
  - Applied when `WebSearch` results are added to state
- [ ] **2.7.3** Concurrent execution unchanged:
  - Multiple topic researchers run in parallel (semaphore-bounded)
  - Each has independent message history and state lock
- [ ] **2.7.4** `TopicResearchResult` fields still populated:
  - `sources_found`, `searches_performed`, `refined_queries`
  - `early_completion`, `completion_rationale`
  - `compressed_findings` (from post-loop compression)

### 2.8 — Add provider fallback for non-tool-calling providers

- [ ] **2.8.1** Detect provider capability before entering ReAct loop
- [ ] **2.8.2** When provider lacks tool calling: fall back to structured-JSON loop
  - Use simplified version of existing reflect approach
  - Single LLM call per iteration with JSON output: `{action, query, reasoning, complete}`
  - No separate think step (merged into single call)
- [ ] **2.8.3** Log when fallback path is used

### 2.9 — Tests

- [ ] **2.9.1** Test: researcher makes WebSearch tool calls and receives summarized results
- [ ] **2.9.2** Test: researcher makes ExtractContent tool calls and receives markdown content
- [ ] **2.9.3** Test: Think tool logs reasoning and returns acknowledgment
- [ ] **2.9.4** Test: ResearchComplete terminates ReAct loop and records summary
- [ ] **2.9.5** Test: tool call budget enforced (WebSearch + ExtractContent decrement, Think does not)
- [ ] **2.9.6** Test: multiple tool calls per turn processed correctly
- [ ] **2.9.7** Test: Think constraint — Think executed before action tools in same turn
- [ ] **2.9.8** Test: message history accumulates across turns
- [ ] **2.9.9** Test: concurrent researchers have independent histories
- [ ] **2.9.10** Test: source deduplication works through WebSearch tool path
- [ ] **2.9.11** Test: inline compression runs after ReAct loop completes
- [ ] **2.9.12** Test: fallback to structured-JSON loop when provider lacks tool calling
- [ ] **2.9.13** Test: LLM call count is 1 per turn (not 2 per iteration)
- [ ] **2.9.14** Test: "no tool calls" response terminates loop gracefully
- [ ] **2.9.15** Test: budget exhaustion terminates loop with appropriate log message
