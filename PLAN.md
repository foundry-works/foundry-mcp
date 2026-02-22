# Foundry Deep Research Improvement Plan

**Date:** 2026-02-22
**Scope:** foundry-mcp deep research workflow (`foundry_mcp/core/research/workflows/deep_research/`)
**Reference:** Comparative analysis with [open_deep_research](https://github.com/langchain-ai/open_deep_research)

---

## Executive Summary

Foundry's deep research implementation is operationally robust (provider resilience, token budgeting, content archival) but its research *quality* is limited by a rigid sequential pipeline and rule-based orchestration. open_deep_research demonstrates that hierarchical multi-agent delegation with LLM-driven reflection produces significantly more thorough research by dynamically identifying and filling knowledge gaps.

This plan proposes 6 improvements in priority order. Each is independently shippable.

---

## 1. Query Clarification Phase

**Problem:** Foundry takes research queries as-is. Vague queries ("how does X work?") waste search credits producing unfocused results.

**What open_deep_research does:** Before any research, it runs a clarification step that analyzes the query for completeness and asks the user to disambiguate scope, timeframe, or domain preferences. Uses a structured `ClarifyWithUser` Pydantic model.

**Proposed change:**

Add an optional `CLARIFICATION` phase before `PLANNING`. When enabled (`deep_research_allow_clarification = true`):

1. Send the original query to a fast model (flash/haiku-tier) with a system prompt asking: "Is this query specific enough for focused research? If not, what 1-3 clarifying questions would help?"
2. LLM returns structured JSON: `{"needs_clarification": bool, "questions": [...], "inferred_constraints": {...}}`
3. If clarification is needed, return the questions to the user via the MCP tool response (the user can answer or skip)
4. Feed the clarified query + constraints into the planning phase

**Files to modify:**
- `models/deep_research.py` — Add `CLARIFICATION` to `DeepResearchPhase` enum, add `clarification_constraints` field to state
- `phases/` — New `clarification.py` mixin
- `workflow_execution.py` — Insert clarification before planning
- `core.py` — Register `ClarificationPhaseMixin`
- `orchestration.py` — Add `CLARIFIER` agent role

**Config additions** (`.foundry-mcp.toml`):
```toml
deep_research_allow_clarification = true
deep_research_clarification_provider = "[cli]gemini:flash"
```

**Estimated complexity:** Low-medium. Single LLM call, simple structured output, clean insertion point.

---

## 2. LLM-Driven Supervisor Reflection (Think-Tool)

**Problem:** Foundry's `SupervisorOrchestrator` uses hardcoded heuristics for quality gates (e.g., "at least 2 sub-queries", "at least 3 sources", report > 100 chars). These thresholds don't account for query complexity or content quality. The `SupervisorHooks.think_pause` mechanism exists but is never wired to an actual LLM call.

**What open_deep_research does:** The supervisor uses a `think_tool` (a no-op tool that forces the LLM to articulate reasoning) before and after every delegation. This creates natural reflection points: "Do I have enough coverage? What angles am I missing? Should I delegate more research or synthesize?"

**Proposed change:**

Wire the existing `SupervisorHooks.think_pause` to actual LLM-based reflection at phase transitions:

1. After each phase completes, send the phase results + state summary to a fast model with the existing `get_reflection_prompt()` as input
2. LLM returns structured JSON: `{"quality_assessment": str, "proceed": bool, "adjustments": [...], "rationale": str}`
3. If `proceed` is false, the supervisor can suggest retry or skip strategies
4. Replace hardcoded thresholds in `_evaluate_phase_quality()` with LLM-assessed quality when reflection is enabled

**Files to modify:**
- `orchestration.py` — Add `async_think_pause()` method that calls LLM, wire into `evaluate_phase_completion()`
- `workflow_execution.py` — Call `async_think_pause()` at phase boundaries (where hooks are already emitted)
- Config: `deep_research_enable_reflection = true`, `deep_research_reflection_provider = "[cli]gemini:flash"`

**Fallback:** Keep existing heuristics as the default when reflection is disabled. LLM reflection is opt-in.

**Estimated complexity:** Medium. Needs async LLM call in the orchestrator, structured output parsing, and graceful fallback.

---

## 3. Parallel Sub-Topic Researcher Agents

**Problem:** Foundry's gathering phase executes all sub-queries in parallel across search providers, but each query gets the same treatment. There's no per-topic reasoning — the LLM that analyzes sources sees all sources in one batch, losing per-topic coherence.

**What open_deep_research does:** The supervisor spawns independent researcher agents (up to 5) for each sub-topic. Each researcher runs its own ReAct loop: search → reflect → search → ... → complete. Researchers are topic-specialized and can decide independently when they have enough information.

**Proposed change:**

Add a `TOPIC_RESEARCH` sub-phase within gathering that, for each sub-query, runs a mini ReAct loop:

1. Search (existing gathering logic, scoped to one sub-query)
2. Reflect: Fast LLM evaluates search results — "Did I find what I need? What's missing?"
3. If gaps remain and search budget allows, refine query and search again (max 3 loops per topic)
4. Compile per-topic findings summary

These mini-loops run in parallel (bounded by `max_concurrent`). Their per-topic summaries feed into the analysis phase.

**Implementation approach:**
- New `phases/topic_research.py` mixin with `_execute_topic_research_async(sub_query, ...)`
- Modify `phases/gathering.py` to optionally delegate to topic researchers instead of flat parallel search
- Config: `deep_research_enable_topic_agents = true`, `deep_research_topic_max_searches = 3`
- State model: Add `TopicResearchResult` to track per-topic findings

**Estimated complexity:** High. Requires nested async loops, per-topic state tracking, budget splitting across topics.

---

## 4. Proactive Content Summarization at Retrieval

**Problem:** Foundry's digest pipeline is budget-triggered (`auto` mode digests only when content exceeds `digest_min_chars`). This means the analysis phase sometimes receives a mix of raw (long) and digested (short) content, making LLM analysis inconsistent.

**What open_deep_research does:** Summarizes every webpage immediately at retrieval time via a fast model (gpt-4.1-mini), producing a uniform ~25-30% summary with `key_excerpts`. This ensures all downstream phases work with pre-processed, consistent-quality content.

**Proposed change:**

Add a `proactive` digest policy alongside the existing `off`/`auto`/`always`:

1. When `deep_research_digest_policy = "proactive"`, run the digest pipeline on every source immediately after retrieval in the gathering phase (not deferred to analysis)
2. Use the existing `DocumentDigestor` but invoke it as part of `_execute_gathering_async()` completion
3. Digest results are stored on the source object, so the analysis phase receives uniform content
4. Budget allocation in analysis can then use accurate digested token counts

**Files to modify:**
- `phases/gathering.py` — Add post-gather digest step when policy is `proactive`
- Config: Add `"proactive"` to `deep_research_digest_policy` options
- Documentation update for the new policy

**Estimated complexity:** Low. The digest pipeline already exists. This just moves its invocation earlier.

---

## 5. Contradiction Detection in Analysis

**Problem:** When multiple sources provide conflicting information, Foundry's analysis phase has no mechanism to detect or flag contradictions. All findings are treated independently. The final report may contain inconsistencies.

**What open_deep_research does:** This is actually a shared weakness — neither implementation handles contradictions well. But open_deep_research's per-researcher approach naturally surfaces contradictions when different researchers report conflicting findings on the same topic.

**Proposed change:**

Add a contradiction detection step at the end of the analysis phase:

1. After findings extraction, send all findings to an LLM with a focused prompt: "Identify any contradictions or conflicting claims between these findings. For each contradiction, note the finding IDs, the nature of the conflict, and which source(s) are more authoritative."
2. LLM returns: `{"contradictions": [{"finding_ids": [...], "description": str, "resolution": str, "preferred_source": str}]}`
3. Store contradictions in state, surface them in the synthesis prompt so the report can address them explicitly
4. Optionally flag contradictions as a specific gap type for the refinement phase

**Files to modify:**
- `models/deep_research.py` — Add `Contradiction` model, `contradictions` list to state
- `phases/_analysis_parsing.py` — Parse contradiction output
- `phases/analysis.py` — Add post-analysis contradiction step
- `phases/synthesis.py` — Include contradictions in synthesis prompt

**Estimated complexity:** Medium. New LLM call, new model, prompt engineering for the synthesis integration.

---

## 6. End-to-End Citation Tracking

**Problem:** Foundry tracks `source_ids` on findings, but the citation chain breaks in the synthesis phase. The final report's source references are generated by the LLM from memory rather than linked to specific findings and their source IDs.

**What open_deep_research does:** Implements numbered inline citations `[1], [2]...` that flow from individual researchers through compression to the final report. Each source gets a stable number, and the report's Sources section uses those numbers.

**Proposed change:**

1. Assign each source a stable citation number (1-indexed) when it enters the state
2. In the synthesis prompt, present findings with their citation numbers: "According to [3], ..."
3. Instruct the synthesis LLM to use inline citations `[N]` in the report
4. Append a numbered Sources section automatically (don't rely on the LLM to reconstruct it)
5. Post-process the report to verify citation consistency (all referenced numbers exist, no dangling references)

**Files to modify:**
- `models/sources.py` or `models/deep_research.py` — Add `citation_number` to source model
- `phases/synthesis.py` — Include citation numbers in prompt, auto-generate Sources section
- New `_citation_postprocess.py` helper for verification

**Estimated complexity:** Low-medium. Mostly prompt and post-processing changes.

---

## Implementation Order & Dependencies

```
Phase 1 (Quick wins, no dependencies):
  [1] Query Clarification
  [4] Proactive Digest
  [6] Citation Tracking

Phase 2 (Core architecture):
  [2] LLM-Driven Reflection  (enables smarter orchestration for Phase 3)

Phase 3 (Advanced):
  [3] Parallel Topic Researchers  (benefits from [2] for per-topic reflection)
  [5] Contradiction Detection  (benefits from [3] for cross-topic comparison)
```

---

## Out of Scope

These were considered but deferred:

- **LangGraph migration**: Foundry's mixin architecture works. Rewriting to LangGraph would be high effort for marginal benefit.
- **OpenAI/Anthropic native web search**: Foundry already has Tavily/Perplexity/Google/Semantic Scholar. Native search APIs add complexity without clear advantage.
- **Evaluation framework**: open_deep_research's LangSmith integration is useful but not a research quality improvement. Can be added later.
- **MCP tool integration in research agents**: open_deep_research supports giving researchers access to arbitrary MCP tools. Interesting but complex and out of scope for this plan.
