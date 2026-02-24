# PLAN: Deep Research — Config Consolidation & Tool-Calling Researchers

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24
**Reference:** `~/GitHub/open_deep_research`
**Status:** Draft
**Depends on:** All prior plan phases (per-result summarization, supervisor-owned decomposition, collapsed pipeline, structured outputs, simplified reflection — all complete)

---

## Context

After completing three rounds of architectural alignment with `open_deep_research` (RACE 0.4344, #6 Deep Research Bench), the foundry-mcp deep research pipeline has converged on the same high-level flow: clarification → brief → supervisor-led decomposition → parallel topic research → per-topic compression → synthesis. The core loop is structurally aligned.

However, comparative analysis reveals two remaining gaps that limit research quality and developer velocity:

1. **Config flag proliferation** — foundry-mcp has 13+ feature flags controlling phase execution (`enable_analysis_digest`, `enable_global_compression`, `enable_refinement`, `enable_reflection`, `enable_brief`, `digest_policy`, `delegation_model`, etc.). Most default to False, meaning they gate dead code paths never exercised in production. open_deep_research has zero such flags — every phase always runs. The flags create untested combinatorial paths, inflate the codebase, and confuse contributors.

2. **Rigid researcher loop** — foundry-mcp's topic researchers follow a fixed `search → reflect → think → refine` sequence where "decisions" are structured JSON fields parsed from separate LLM calls. open_deep_research gives researchers real tool-calling agency: they receive `search`, `extract`, `think_tool`, and `ResearchComplete` as tools and the LLM decides which to call. This saves one LLM call per iteration (merge reflect+think) and lets researchers pursue unexpected leads naturally.

Phases are ordered by risk profile: Phase 1 is pure deletion (lowest risk, highest maintenance payoff). Phase 2 is an architectural improvement that reduces LLM calls and improves research quality.

---

## Phase 1: Collapse Config Flags (Delete Dead Paths)

**Effort:** Medium (mostly deletion) | **Impact:** High (maintenance, testability, readability)
**Files:**
- `src/foundry_mcp/config/research.py` (remove flags)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (simplify phase flow)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/` (remove gated code paths)
- `src/foundry_mcp/core/research/models/deep_research.py` (clean up phase enum)

### Problem

foundry-mcp has 13+ feature flags controlling deep research execution:

| Flag | Default | Status |
|------|---------|--------|
| `deep_research_enable_analysis_digest` | `False` | Dead — analysis phase never runs in default config |
| `deep_research_enable_global_compression` | `False` | Dead — global compression never runs in default config |
| `deep_research_enable_refinement` | `False` | Dead — refinement loop never runs in default config |
| `deep_research_enable_reflection` | `False` | Dead — per-phase reflection pauses never run |
| `deep_research_enable_brief` | `True` | Always on — no reason to disable |
| `deep_research_digest_policy` | `"auto"` | Moot — proactive vs lazy distinction no longer meaningful with inline compression |
| `deep_research_delegation_model` | `True` | Always on — legacy query-generation model superseded |
| `deep_research_supervisor_owned_decomposition` | `True` | Always on — planning phase superseded |
| `deep_research_enable_planning_critique` | `True` | Moot — planning phase no longer in default path |
| `deep_research_enable_topic_agents` | `False` | Dead — never enabled |
| `deep_research_enable_contradiction_detection` | `True` | Moot — analysis phase disabled |
| `tavily_extract_in_deep_research` | `False` | Dead — post-gathering extract never runs |
| `deep_research_fetch_time_summarization` | `True` | Always on — per-result summarization is now standard |

open_deep_research has zero feature flags. Every phase always runs. The system is readable in a single sitting.

Each flag creates a conditional branch in `workflow_execution.py`, a code path in a phase file, and (in theory) a test case. But because most flags default to False, the gated paths are effectively untested and unmaintained. They add cognitive overhead without value.

### Design

Commit to the production pipeline and delete dead paths:

1. **Keep the production pipeline**: CLARIFICATION → BRIEF → SUPERVISION (round 0 = decompose, round 1+ = gap-fill) → SYNTHESIS. This is what runs today with default config.

2. **Delete dead phase code**:
   - Global COMPRESSION phase — per-topic compression handles this
   - ANALYSIS phase (digest/rank/select) — per-result summarization + per-topic compression make this redundant
   - REFINEMENT phase — supervision gap-filling replaces this
   - Per-phase reflection pauses — never proved useful
   - Legacy query-generation delegation model — superseded by directive delegation
   - Standalone PLANNING phase — superseded by supervisor-owned decomposition
   - Post-gathering Tavily extract — inline extract handles this

3. **Hardwire always-on behavior**:
   - Brief generation always runs (remove the enable flag, keep the phase)
   - Supervisor-owned decomposition always runs (remove the flag)
   - Delegation model always runs (remove the flag)
   - Fetch-time summarization always runs (remove the flag)

4. **Keep tuning knobs**: `max_iterations`, `max_searches`, `max_supervision_rounds`, `max_concurrent_research_units`, `max_sources_per_query`, `deep_research_enable_extract`, `deep_research_enable_content_dedup`. These control behavior within phases, not whether phases exist.

5. **Simplify workflow_execution.py**: The `_determine_next_phase()` function currently has 6+ conditional branches. After this change, it's a linear sequence: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS → COMPLETE.

### Changes

1. Delete config flags from `research.py`:
   - `deep_research_enable_analysis_digest`
   - `deep_research_enable_global_compression`
   - `deep_research_enable_refinement`
   - `deep_research_enable_reflection`
   - `deep_research_enable_brief`
   - `deep_research_digest_policy`
   - `deep_research_delegation_model`
   - `deep_research_supervisor_owned_decomposition`
   - `deep_research_enable_planning_critique`
   - `deep_research_enable_topic_agents`
   - `deep_research_enable_contradiction_detection`
   - `tavily_extract_in_deep_research`
   - `deep_research_fetch_time_summarization` (hardwire to True)
2. Simplify `workflow_execution.py`:
   - Remove all conditional phase-skip logic
   - Hardwire phase sequence: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS
   - Remove iteration loop (refinement is gone)
   - Remove dead phase transitions
3. Delete or archive dead phase files:
   - `phases/analysis_*.py` — analysis digest code
   - `phases/refinement.py` — refinement loop code (if exists as separate file)
   - Legacy delegation model code path in `phases/supervision.py`
   - Legacy planning phase entry point in `workflow_execution.py`
   - Global compression orchestration in `phases/compression.py` (keep per-topic compression)
   - Post-gathering extract code in `phases/gathering.py`
   - Per-phase reflection pause code in orchestration module
4. Clean up `DeepResearchPhase` enum — remove phases that no longer exist
5. Remove dead config parsing in `from_toml_dict()` and any config validation referencing deleted flags
6. Update tests — remove tests for deleted paths, verify production pipeline still passes
7. Update any documentation referencing deleted flags

### Validation

- Test: production pipeline (CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS) works end-to-end
- Test: no config flags required — default config produces working research
- Test: workflow_execution.py has no conditional phase-skip branches
- Test: deleted phases cannot be invoked (phase enum values removed)
- Test: per-topic compression still works (not deleted, only global compression removed)
- Test: fetch-time summarization always active (no flag to disable)
- Test: existing passing tests still pass (with test deletions for removed features)
- Verify: grep for all deleted flag names across codebase — zero references remain

---

## Phase 2: Tool-Calling Researchers (Merge Reflect + Think into ReAct Agent)

**Effort:** Medium-High | **Impact:** High (research quality + fewer LLM calls)
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (major refactor)
- `src/foundry_mcp/core/research/models/deep_research.py` (tool schemas)
- `src/foundry_mcp/config/research.py` (cleanup)

### Problem

foundry-mcp's topic researcher runs a rigid 4-step sequence per iteration:

```
search(query)           # 0 LLM calls — just API dispatch
reflect(sources)        # 1 LLM call — structured JSON: continue/stop/refine/extract
think(gap_analysis)     # 1 LLM call — structured JSON: reasoning + next_query
refine(query)           # 0 LLM calls — string replacement
```

That's **2 LLM calls per search iteration** for decision-making alone. With 3 iterations per topic and 3 topics, that's ~18 decision LLM calls per research run.

open_deep_research's researcher is a true ReAct agent:

```
researcher_turn:        # 1 LLM call — decides AND acts
  → calls search("refined query")       # tool call
  → calls think_tool("gap analysis")    # tool call
  → calls search("follow-up query")     # tool call
  → calls ResearchComplete()            # tool call (exit signal)
```

That's **1 LLM call per turn**, and the researcher can make multiple tool calls per turn. The model chooses its own search queries, decides when to reflect, and signals completion — all in the same inference pass.

Benefits of tool-calling:
- **Fewer LLM calls**: 1 per turn vs 2 per iteration. With 3 turns per topic, that's 9 vs 18 calls.
- **Better agency**: The researcher can search, reflect, and search again in a single turn if it spots a lead.
- **Natural exit**: `ResearchComplete` tool call is cleaner than a JSON field in a reflection response.
- **Simpler code**: No separate `_topic_reflect()`, `_topic_think()`, `_parse_reflection_decision()` functions. The tool-calling framework handles dispatch.

### Design

Replace the fixed search→reflect→think→refine loop with a tool-calling ReAct agent, modeled on open_deep_research's researcher:

1. **Researcher tools**: Define as Pydantic tool schemas:
   - `WebSearch(query: str, max_results: int = 5)` — dispatches to configured search providers
   - `ExtractContent(urls: list[str])` — calls Tavily extract for full page content
   - `Think(reasoning: str)` — strategic reflection pause (logged, no side effect)
   - `ResearchComplete(summary: str)` — signals completion with summary of findings

2. **ReAct loop**: Each iteration is one LLM call that produces tool calls:
   ```
   while tool_calls_remaining > 0:
       response = llm.generate(messages, tools=[WebSearch, ExtractContent, Think, ResearchComplete])
       for tool_call in response.tool_calls:
           result = execute_tool(tool_call)
           messages.append(tool_result(result))
           tool_calls_remaining -= 1
       if ResearchComplete called: break
       if no tool calls: break  # model chose to stop
   ```

3. **System prompt**: Adapted from open_deep_research's `research_system_prompt` (prompts.py:138-183):
   - "You are a focused research agent. Your task is to thoroughly research: {topic}"
   - "Start with broader searches, then narrow based on what you find"
   - "Use Think to pause and assess your findings before deciding next steps"
   - "Call ResearchComplete when you are confident the findings address the research question"
   - Budget visibility: "You have {max_tool_calls} tool calls remaining"

4. **Message history**: Each researcher maintains its own message list. Search results (already summarized via fetch-time summarization) are returned as tool results. This gives the model full context of what it's searched and found.

5. **Concurrency**: Multiple topic researchers still run in parallel (bounded by semaphore). Each researcher is an independent ReAct loop with its own message history.

6. **Source management**: `WebSearch` tool results include source metadata. Sources are added to shared state under the existing `state_lock`. Deduplication (URL, title, content similarity) still applies.

7. **Think tool constraint**: Following open_deep_research's pattern, `Think` cannot be called in parallel with other tools. If the model calls Think alongside WebSearch in the same turn, execute Think first, then WebSearch. This ensures reflection happens before action.

8. **Provider compatibility**: Tool-calling requires a provider that supports function calling. For providers that don't (rare for modern models), fall back to the existing structured-JSON approach as a compatibility path.

### Changes

1. Define researcher tool schemas in `models/deep_research.py`:
   - `WebSearchTool(query: str, max_results: int = 5)`
   - `ExtractContentTool(urls: list[str])`
   - `ThinkTool(reasoning: str)`
   - `ResearchCompleteTool(summary: str)`
2. Refactor `_execute_topic_research_async()` in `topic_research.py`:
   - Replace search→reflect→think→refine loop with ReAct tool-calling loop
   - Build message history per researcher: system prompt + tool calls + tool results
   - Dispatch tool calls to existing infrastructure:
     - `WebSearchTool` → `_topic_search()` (reuse existing provider dispatch)
     - `ExtractContentTool` → `_topic_extract()` (reuse existing extract logic)
     - `ThinkTool` → log reasoning, add to message history
     - `ResearchCompleteTool` → set `early_completion=True`, break loop
   - Track `tool_calls_used` across all tool types (unified budget)
3. Create researcher system prompt:
   - Adapt from open_deep_research's `research_system_prompt`
   - Include topic assignment, budget visibility, search strategy guidance
   - Include date context and source quality expectations
4. Update LLM call path in topic_research.py:
   - Use provider's tool-calling API (function calling / tool use)
   - Parse tool call responses from provider result
   - Execute tools sequentially (respecting Think constraint)
   - Append tool results to message history
5. Delete merged functions:
   - `_topic_reflect()` — merged into researcher's own reasoning
   - `_topic_think()` — merged into `ThinkTool`
   - `_parse_reflection_decision()` — no longer needed
   - `_format_topic_sources_for_reflection()` — sources are in message history
6. Preserve inline per-topic compression after ReAct loop completes (unchanged)
7. Add provider compatibility check — if provider doesn't support tool calling, fall back to structured-JSON loop
8. Tests

### Validation

- Test: researcher makes tool calls (WebSearch, Think, ResearchComplete) in a single LLM turn
- Test: WebSearch dispatches to configured providers and returns summarized results
- Test: ExtractContent calls Tavily extract and returns markdown content
- Test: Think tool logs reasoning and adds to message history
- Test: ResearchComplete terminates loop and records summary
- Test: tool call budget enforced across all tool types
- Test: multiple tool calls per turn (search + think in same response)
- Test: Think constraint respected (Think before other tools in same turn)
- Test: message history accumulates across turns (researcher has full context)
- Test: concurrent researchers run in parallel with independent histories
- Test: source deduplication still works (URL, title, content similarity)
- Test: inline compression still runs after ReAct loop completes
- Test: provider without tool-calling support falls back to structured-JSON loop
- Test: LLM call count reduced vs prior approach (1 per turn vs 2 per iteration)

---

## Dependency Graph

```
Phase 1 (Collapse Config Flags)
    ↓ simplifies codebase for
Phase 2 (Tool-Calling Researchers)
```

Phase 1 should land first — it produces a cleaner codebase that's easier to refactor in Phase 2.

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Deleting code paths breaks unknown consumers | Grep for all deleted flag names. Check config files, tests, docs. Staged deletion with deprecation warnings first if needed. |
| 1 | Users relying on disabled-by-default features | These features were never in the default path. Any user who enabled them knowingly opted into non-default behavior — provide migration notes. |
| 2 | Tool-calling not supported by all LLM providers | Compatibility check before entering ReAct loop. Fall back to structured-JSON approach for providers without tool calling. |
| 2 | Researcher over-searches without rigid loop structure | Budget hard cap (max_tool_calls) still enforced. ResearchComplete provides clean exit. "No new sources" heuristic prevents infinite loops. |
| 2 | Message history grows too large for context window | Cap message history at N most recent turns. Older tool results summarized or dropped. Per-topic research is bounded by max_tool_calls. |
