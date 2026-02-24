# PLAN: Deep Research — Supervisor Orchestration, Compression & Synthesis Alignment

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-24
**Reference:** `~/GitHub/open_deep_research` (RACE 0.4344, #6 Deep Research Bench)
**Status:** Draft
**Depends on:** All prior plan phases complete (summarization, brief, inline compression, structured outputs, delegation, pipeline collapse, reflection simplification)

---

## Context

After completing seven prior plan phases that aligned foundry-mcp with open_deep_research on per-result summarization, research briefs, inline compression, structured outputs, supervisor delegation, pipeline simplification, and reflection — comparative analysis reveals four remaining structural gaps, ranked by downstream quality impact.

These four changes focus on the core research loop (how work gets orchestrated) and the output boundary (how findings become reports). No changes to tool calling, search providers, or evaluation infrastructure.

### Gap Analysis Summary

| # | Gap | Root cause | Quality impact |
|---|-----|-----------|----------------|
| 1 | **Rigid phase pipeline** | Separate PLANNING → GATHERING → SUPERVISION phases commit to a full decomposition before seeing any results | Supervisor can only patch gaps after ALL initial gathering completes; no adaptive decomposition |
| 2 | **Compression from metadata, not conversation** | Compression prompt receives structured `TopicResearchResult` fields (reflection notes, refined queries) instead of the raw researcher message history | Loses the researcher's reasoning chain, failed attempts, and iterative refinements — degrades synthesis quality on complex topics |
| 3 | **Synthesis prompt underspecified** | Foundry synthesis prompt has good structure guidance but misses open_deep_research's concrete citation rules, language-matching emphasis, verbosity expectations, and section fluidity | Final report quality gap: less comprehensive sections, weaker citation formatting, occasional language mismatches |
| 4 | **Token recovery truncates state fields, not messages** | On token limit errors, foundry-mcp uniformly degrades state fields via fidelity tracking rather than pruning old message context while preserving recent reasoning | Loses recent LLM reasoning (most valuable) proportionally with old context (least valuable) |

Phases are ordered by leverage: Phase 1 is the largest architectural change and unblocks the most downstream improvement. Phases 2-4 are smaller, independent prompt/recovery changes that each improve a specific quality dimension.

---

## Phase 1: Unify Supervision as the Research Orchestrator

**Effort:** Medium-High | **Impact:** Very High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (expand first-round decomposition to be the primary path)
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (simplify phase flow)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py` (deprecate from default path)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py` (deprecate from default path)
- `src/foundry_mcp/core/research/models/deep_research.py` (phase enum adjustments)
- `src/foundry_mcp/config/research.py` (config)

### Problem

open_deep_research has no PLANNING or GATHERING phase. The `supervisor` function (deep_researcher.py:178-224) is the sole orchestrator: it receives the research brief, decomposes on its first iteration via `ConductResearch` tool calls, sees the results, assesses gaps, and delegates follow-ups — all in one continuous loop. The decomposer and the gap assessor are the same entity, sharing a mental model.

foundry-mcp currently has supervisor-owned decomposition as an *implemented feature* (`_is_first_round_decomposition()` at supervision.py:337-352, first-round think/delegate prompts, and the BRIEF → SUPERVISION transition at workflow_execution.py:194-199). However, it coexists with the legacy PLANNING and GATHERING phases. The current `workflow_execution.py` flow (lines 194-252):

```
BRIEF → (skip PLANNING/GATHERING) → SUPERVISION loop → SYNTHESIS
```

...already works as intended. But the codebase retains PLANNING and GATHERING as fully wired phases that legacy saved states can resume into, and the supervision loop only runs the first-round decomposition path when no prior topic results exist. The task now is to complete this transition:

1. **Make supervisor-owned decomposition the only default path** — remove the conditional that checks for `deep_research_supervisor_owned_decomposition` config flag and make it unconditional.
2. **Ensure the supervision loop handles the full lifecycle** — round 0 decomposes, rounds 1+ assess and delegate, with no expectation of GATHERING as a separate phase.
3. **Clean up dead code paths** — the PLANNING phase's decomposition logic is now dead code on the default path. Mark it as legacy/deprecated.
4. **Harden the loop termination** — currently round 0 always delegates (no heuristic early-exit). Ensure that round 0 delegates even for trivially simple queries (single directive) and that the heuristic gate at round > 0 properly assesses the decomposition round's results.

The key quality win: the supervisor can *adaptively* decompose. For a simple factual query, it issues 1 directive; sees the results are sufficient in round 1; stops. For a complex comparison, it issues 3 parallel directives in round 0; sees gaps in round 1; issues 2 targeted follow-ups; stops in round 2. The planning phase can't do this because it commits to a decomposition before any research happens.

### Design

Complete the transition that prior plan phases started:

1. **Remove config guard**: The `deep_research_supervisor_owned_decomposition` flag becomes always-True. The BRIEF → SUPERVISION transition in `workflow_execution.py` is unconditional (it already is on line 198-199, but verify no other code paths check this flag).

2. **GATHERING as legacy-resume-only**: The GATHERING block in `workflow_execution.py` (lines 205-219) already has a comment "Legacy saved states may resume at GATHERING; let it proceed." Formalize this: GATHERING only executes if `state.phase == DeepResearchPhase.GATHERING` on entry (resumed from saved state). New workflows never enter GATHERING.

3. **PLANNING as legacy-resume-only**: Same treatment. If a saved state resumes at PLANNING, run it; new workflows skip it entirely.

4. **Supervision round 0 prompt quality**: The first-round think and delegate prompts (already implemented) absorb open_deep_research's supervisor guidance:
   - "Bias toward a single researcher for simple factual queries"
   - "Parallelize for explicit comparisons (one per element)"
   - "2-5 directives for typical queries"
   - Self-critique: "Before delegating, verify no redundant directives and no missing perspectives"

5. **Post-decomposition assessment**: After round 0 directives execute, the heuristic at round > 0 (`_assess_coverage_heuristic`) sees the decomposition results and decides whether gaps exist. This is the key adaptive behavior: the supervisor *reacts* to what the researchers found.

6. **Remove the config flag**: Delete `deep_research_supervisor_owned_decomposition` from `research.py` and its `from_toml_dict()` parsing. It's no longer optional.

### Changes

1. Remove `deep_research_supervisor_owned_decomposition` config flag from `research.py` — behavior is now always-on
2. Update `workflow_execution.py`:
   - Remove conditional logic around PLANNING and GATHERING for new workflows
   - Keep PLANNING/GATHERING blocks for legacy resume only (guarded by `state.phase ==` check)
   - Ensure the BRIEF → SUPERVISION transition is the sole default path
3. Clean up any references to the config flag in supervision.py, planning.py, gathering.py
4. Review first-round think/delegate prompts for quality — ensure they match open_deep_research's supervisor guidance level
5. Add deprecation log when PLANNING or GATHERING phase runs from legacy resume
6. Verify round 0 → round 1 handoff: heuristic should assess round 0 results before deciding on round 1
7. Tests

### Validation

- Test: new workflow goes BRIEF → SUPERVISION (round 0 decomposition) → SUPERVISION (round 1+ gap assessment) → SYNTHESIS
- Test: PLANNING and GATHERING phases never execute for new workflows
- Test: legacy saved state at GATHERING resumes correctly
- Test: legacy saved state at PLANNING resumes correctly
- Test: simple query produces 1 directive; sufficient in round 1
- Test: complex comparison produces 3+ directives; round 1 may delegate follow-ups
- Test: round 0 results feed into round 1 heuristic assessment
- Test: config flag removed — no runtime error from old config files referencing it
- Test: no regression in overall research quality (manual inspection of sample queries)

---

## Phase 2: Pass Full Message History to Compression

**Effort:** Small | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (prompt enhancement)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (persist message history on result)
- `src/foundry_mcp/core/research/models/deep_research.py` (model field)

### Problem

open_deep_research's `compress_research` function (deep_researcher.py:511-585) passes the **full researcher message history** — all tool outputs, AI reflections, search results, and reasoning — to the compression model. The compression prompt says: "All relevant information should be repeated and rewritten verbatim, but in a cleaner format... Only these fully comprehensive cleaned findings are going to be returned to the user, so it's crucial that you don't lose any information."

foundry-mcp's `_compress_single_topic_async` (compression.py:50-150) builds the compression prompt from **structured metadata**: `TopicResearchResult` fields like `reflection_notes`, `refined_queries`, `completion_rationale`, and source content. It reconstructs an "iteration history" from these fields (lines 106-139).

The problem: the structured metadata is a *lossy projection* of the actual conversation. The raw message history contains:
- The researcher's explicit reasoning at each step ("I found X but it doesn't answer Y, so I'll search for Z")
- Failed search context ("This search returned irrelevant results about A instead of B")
- Tool response content as the researcher actually saw it
- Think-tool deliberation ("The sources contradict each other on point P...")

This context helps the compression model preserve the right information and produce better-cited summaries. The structured metadata loses the reasoning chain and reduces compression quality, especially for complex topics where the research journey matters.

### Design

Store the raw message history on `TopicResearchResult` and pass it to the compression prompt:

1. **Persist message history**: The topic researcher's ReAct loop already maintains `message_history: list[dict]` (topic_research.py). After the loop completes, store it on the `TopicResearchResult`.

2. **Compression prompt enhancement**: Update `_compress_single_topic_async` to include the raw message history in the prompt when available, in addition to the existing structured metadata. Format it as open_deep_research does: sequential tool calls and AI responses in chronological order.

3. **Prompt structure alignment**: Adopt open_deep_research's compression prompt structure:
   - `<Task>`: Clean up findings, preserve ALL relevant information verbatim
   - `<Guidelines>`: Comprehensive, include ALL sources, inline citations, Sources section
   - `<Output Format>`: Queries/tool calls made, Fully comprehensive findings, Sources list
   - `<Citation Rules>`: Sequential numbering, `[1] Title: URL` format
   - Critical reminder: preserve verbatim, don't summarize

4. **Fallback**: If `message_history` is empty (e.g., legacy results without it), fall back to the existing structured metadata approach. No breaking change.

5. **Token budget**: Message history can be large. Apply the existing `max_content_length` cap to the total message history block. Truncate oldest messages first (preserving the most recent reasoning).

### Changes

1. Add `message_history: list[dict] = Field(default_factory=list)` to `TopicResearchResult` in `models/deep_research.py`
2. Store `message_history` on result object at end of ReAct loop in `topic_research.py` (after line ~540 where the loop ends)
3. Update `_compress_single_topic_async` in `compression.py`:
   - When `topic_result.message_history` is non-empty, build prompt from message history (chronological tool calls + AI responses)
   - Adopt open_deep_research's `compress_research_system_prompt` structure: Task, Guidelines, Output Format, Citation Rules sections
   - When `message_history` is empty, fall back to existing structured metadata prompt
4. Apply `max_content_length` cap to total message history block in prompt
5. Tests

### Validation

- Test: message history stored on TopicResearchResult after ReAct loop
- Test: compression prompt includes raw message history when available
- Test: compression prompt falls back to structured metadata when message_history empty
- Test: message history truncated to max_content_length (oldest messages dropped first)
- Test: compression output includes citation format matching open_deep_research (`[N] Title: URL`)
- Test: compression output includes "Queries and Tool Calls Made" section
- Test: no regression — existing compression tests pass with new prompt structure

---

## Phase 3: Align Synthesis Prompt with open_deep_research

**Effort:** Small | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (prompt updates)

### Problem

Foundry-mcp's synthesis system prompt (synthesis.py:329-376) is good but diverges from open_deep_research's `final_report_generation_prompt` (prompts.py:228-308) in several specific ways that affect output quality:

1. **Section verbosity expectations**: open_deep_research explicitly says "Each section should be as long as necessary to deeply answer the question... It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer." Foundry's prompt says "provide depth, not surface-level summaries" but doesn't set the expectation of length.

2. **Section fluidity**: open_deep_research says "Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!" Foundry's structure guidance is more prescriptive (fixed template per query type) with a mandatory "Analysis" section containing "Supporting Evidence", "Conflicting Information", and "Limitations" subsections.

3. **Citation format**: open_deep_research uses `[Title](URL)` markdown links inline AND numbered `[1] Source Title: URL` at the end. Foundry uses only numbered `[N]` inline with automatic Sources section appended by `postprocess_citations()`.

4. **Language matching emphasis**: open_deep_research repeats the language-matching instruction three times with increasing urgency ("CRITICAL", "REMEMBER"). Foundry mentions it once.

5. **Self-referential language**: open_deep_research says "Do NOT ever refer to yourself as the writer" and "Do not say what you are doing in the report. Just write the report without any commentary." Foundry says "Never refer to yourself" and "Never use meta-commentary" which is similar but less emphatic.

6. **Mandatory Analysis section**: Foundry requires an "Analysis" section with "Supporting Evidence", "Conflicting Information", and "Limitations" subsections for every query type. open_deep_research has no such requirement — the structure is query-driven. This forced structure can feel formulaic for simple queries.

### Design

Targeted prompt updates to close the specific gaps, while keeping foundry-mcp's strengths (query-type classification, structure guidance):

1. **Add verbosity expectation**: Explicitly tell the model that sections should be thorough and lengthy. Users expect deep research.

2. **Soften structure prescriptiveness**: Keep query-type structure hints but add open_deep_research's "you can structure however you think is best" permission. Make the Analysis section optional rather than mandatory.

3. **Strengthen language matching**: Add a second, more emphatic language-matching instruction at the end of the system prompt (matching open_deep_research's emphasis pattern).

4. **Citation format**: Keep the current `[N]` inline numbering (it works well with `postprocess_citations()`), but add guidance about citation importance: "Citations are extremely important. Pay attention to getting these right. Users will use citations to look into more information."

5. **Section writing rules**: Add open_deep_research's per-section rules: "Use ## for section titles", "Write in paragraph form by default; use bullet points only when listing discrete items", "Each section should be as long as necessary."

6. **Remove mandatory Analysis subsections**: Replace the mandatory "Supporting Evidence / Conflicting Information / Limitations" structure with optional guidance: "Include analysis of conflicting information and limitations where relevant, but do not force these into separate subsections if they don't apply."

### Changes

1. Update `_build_synthesis_system_prompt` in synthesis.py:
   - Add section verbosity expectation ("thorough, lengthy sections expected")
   - Soften structure guidance ("these are suggestions — structure however makes sense")
   - Make Analysis subsections optional ("include where relevant, not mandatory")
   - Add citation importance emphasis
   - Add second language-matching reminder at end
   - Add per-section writing rules from open_deep_research
2. No changes to `_build_synthesis_user_prompt` — the data formatting is already good
3. Tests

### Validation

- Test: system prompt includes verbosity expectation language
- Test: structure guidance includes "structure however you think is best" permission
- Test: Analysis subsections are described as optional, not mandatory
- Test: citation guidance includes importance emphasis
- Test: language matching instruction appears at least twice in system prompt
- Test: prompt still includes query-type classification and structure hints (no regression)

---

## Phase 4: Message-Aware Token Limit Recovery

**Effort:** Small | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` (retry logic)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py` (compression-specific recovery)
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (synthesis-specific recovery)

### Problem

open_deep_research handles token limit errors with a message-aware truncation strategy (utils.py:848-866): `remove_up_to_last_ai_message(messages)` prunes the oldest messages while preserving the most recent AI reasoning. On retry, if still over limit, it reduces by 10% character count. Max 3 attempts.

foundry-mcp's `execute_llm_call` in `_lifecycle.py` has token limit detection and recovery but uses a different strategy: it relies on the `final_fit_validate` preflight check and fidelity-based content degradation to stay within limits. When a token limit error occurs at runtime (preflight passed but actual call fails), the recovery options are limited.

The problem: the most valuable content in a prompt is typically the *most recent* context — the latest research findings, the most recent reasoning, the current gap analysis. Uniform truncation (or fidelity degradation across all content) loses recent context proportionally with old context. Message-aware truncation preserves recency.

### Design

Add message-aware truncation as a retry strategy for compression and synthesis phases:

1. **Compression retry**: When `_compress_single_topic_async` gets a token limit error:
   - First retry: truncate the message history block (drop oldest messages, preserving recent)
   - Second retry: additionally reduce source content by 10%
   - Third retry: reduce by another 10%
   - After 3 retries: return without compression (non-fatal, use raw findings)

2. **Synthesis retry**: When `_execute_synthesis_async` gets a token limit error:
   - First retry: truncate the per-topic findings (drop oldest topics' compressed_findings first, preserving most recent/highest priority)
   - Second retry: reduce remaining content by 10%
   - Third retry: reduce by another 10%
   - After 3 retries: generate partial report with available content

3. **Generic helper**: Add `truncate_prompt_for_retry(prompt: str, attempt: int, max_attempts: int = 3) -> str` to `_lifecycle.py`:
   - Attempt 1: remove first 20% of content
   - Attempt 2: remove first 30% of content
   - Attempt 3: remove first 40% of content
   - Preserves system prompt (never truncated)
   - Preserves the last N characters (most recent context)

4. **Provider-specific detection**: The existing `ContextWindowError` detection is good. Ensure it covers the patterns from open_deep_research (utils.py:665-785):
   - OpenAI: `BadRequestError` + "maximum context length" / "too many tokens"
   - Anthropic: `BadRequestError` + "prompt is too long" / "too many tokens"
   - Google: `ResourceExhausted` / `InvalidArgument` with token keywords

### Changes

1. Add `truncate_prompt_for_retry()` helper to `_lifecycle.py`
2. Wrap the LLM call in `_compress_single_topic_async` with a retry loop:
   - Catch token limit errors
   - Apply `truncate_prompt_for_retry()` to user prompt
   - Retry up to 3 times
   - Log each retry with truncation percentage
3. Wrap the LLM call in `_execute_synthesis_async` with a retry loop:
   - Same pattern as compression
   - Truncate per-topic findings (oldest/lowest-priority first)
4. Verify provider-specific error detection patterns cover OpenAI, Anthropic, Google
5. Tests

### Validation

- Test: compression retries on token limit error with progressive truncation
- Test: synthesis retries on token limit error with progressive truncation
- Test: system prompt is never truncated (only user prompt content)
- Test: most recent context preserved (truncation removes oldest content first)
- Test: max 3 retries before graceful fallback
- Test: provider-specific error detection for OpenAI, Anthropic, Google patterns
- Test: non-token-limit errors are NOT retried (only token limit triggers retry)
- Test: retry metadata recorded in audit events

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|-----------|
| **1: Supervisor orchestrator** | Legacy saved states break if PLANNING/GATHERING phases removed | Keep phase code for resume; only new workflows skip them |
| **1: Supervisor orchestrator** | Round 0 decomposition quality regresses vs. planning phase | Compare decomposition outputs; first-round prompts already tested |
| **2: Message history compression** | Message history inflates prompt beyond token limits | Cap at `max_content_length`; truncate oldest messages |
| **2: Message history compression** | Pydantic serialization size increase | `message_history` excluded from compact repr |
| **3: Synthesis prompt** | Longer, more verbose reports increase token costs | Acceptable tradeoff for quality; matches user expectations |
| **4: Token recovery** | Retry loops add latency on token limit errors | Max 3 retries (bounded); only triggers on actual errors |

---

## Dependency Graph

```
Phase 1 (Supervisor orchestrator)   ← independent, highest leverage
Phase 2 (Message history compression) ← independent
Phase 3 (Synthesis prompt)           ← independent
Phase 4 (Token limit recovery)       ← independent, benefits from Phase 2
```

All phases are independent and can be implemented in parallel. Phase 1 is recommended first due to highest leverage. Phase 4 benefits slightly from Phase 2 (message history gives the token recovery more granular truncation targets).
