# PLAN: Deep Research Alignment with open_deep_research

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23
**Reference:** `~/GitHub/open_deep_research`
**Status:** Draft

---

## Context

Comparative review of foundry-mcp deep research against open_deep_research identified six alignment gaps. Prior work on this branch already aligned L1 summarization, L2 compression semantics, token retry counts, truncation factors, and citation formatting. This plan targets the remaining gaps, ordered by effort-to-impact ratio.

---

## Phase 1: Update Model Token Limits

**Effort:** Low | **Impact:** Medium
**Files:** `src/foundry_mcp/config/model_token_limits.json`

### Problem

foundry-mcp uses rounded-down token limits (1,000,000 for GPT-4.1, 1,000,000 for Gemini) while open_deep_research uses provider-accurate values. This causes foundry-mcp to trigger truncation ~5% earlier than necessary for GPT-4.1 and uses half the actual limit for Gemini models.

### Upstream Values (open_deep_research `utils.py:788-829`)

| Model | Current (foundry) | Upstream (open_deep_research) |
|-------|-------------------|-------------------------------|
| gpt-4.1 | 1,000,000 | 1,047,576 |
| gpt-4.1-mini | 1,000,000 | 1,047,576 |
| gpt-4.1-nano | 1,000,000 | 1,047,576 |
| gpt-4o | — | 128,000 |
| gpt-4o-mini | — | 128,000 |
| claude-opus-4-6 | 200,000 | 200,000 (aligned) |
| claude-sonnet-4-6 | 200,000 | 200,000 (aligned) |
| gemini-2.5-pro | — | 1,048,576 |
| gemini-2.5-flash | — | 1,048,576 |
| gemini-1.5-pro | — | 2,097,152 |

### Changes

1. Update `model_token_limits.json` with accurate values
2. Add missing models (gpt-4o, gpt-4o-mini, gemini variants)
3. Update `_lifecycle.py` inline `MODEL_TOKEN_LIMITS` fallback dict (line 95) to match

### Validation

- Existing token truncation tests should pass with updated values
- Verify `_lifecycle.py:truncate_to_token_estimate` uses the json file, not just the hardcoded dict

---

## Phase 2: Research Brief Generation

**Effort:** Medium | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py` (lines 49-309)
- `src/foundry_mcp/core/research/models/deep_research.py` (state model)

### Problem

open_deep_research has a dedicated `write_research_brief` phase (deep_researcher.py:118-175) that transforms raw user messages into a structured research question before any planning occurs. This step maximizes specificity, fills unstated dimensions, and biases toward primary sources. foundry-mcp generates `state.research_brief` inline during planning (planning.py:116) without a dedicated refinement step.

### Design

Add a brief-refinement step at the start of the planning phase. Not a new phase — extend the existing planning phase to first refine the query into a structured brief, then decompose it into sub-queries. This avoids adding orchestration complexity while capturing the upstream benefit.

The brief-refinement prompt should be original (not copied from open_deep_research) but model after these principles from their `transform_messages_into_research_topic_prompt`:
- Maximize specificity and detail from the user's query
- Fill unstated dimensions as open-ended rather than assuming
- Prefer primary/official sources over aggregators
- Preserve the user's original language preference
- Output a single focused research brief paragraph

### Changes

1. Add `_build_brief_refinement_prompt()` to `planning.py`
2. Execute a brief-refinement LLM call before the existing sub-query decomposition call
3. Store result in `state.research_brief` (already exists in the state model)
4. Use the refined brief as input to `_build_planning_user_prompt()` instead of the raw query
5. Use a cheap model role (e.g. `"summarization"` or add `"brief"` to `_ROLE_RESOLUTION_CHAIN`)

### Validation

- Test: refined brief is more specific than raw query for ambiguous inputs
- Test: planning sub-queries are grounded in the refined brief, not raw query
- Test: existing planning tests still pass (brief refinement is additive)

---

## Phase 3: Synthesis Prompt Engineering

**Effort:** Medium | **Impact:** High
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (lines 263-456)

### Problem

open_deep_research's `final_report_generation_prompt` (prompts.py:228-308) includes several quality-improving directives that foundry-mcp's synthesis prompt lacks:
- Multi-language detection and matching (write report in user's language)
- Flexible report structure selection (comparison vs. list vs. summary)
- Anti-patterns ("never start with 'based on the research'")
- Progressive disclosure of information
- Specific citation format rules (`[Title](URL)` inline + numbered source list)

### Design

Enhance `_build_synthesis_system_prompt()` with our own versions of these directives. The prompt should be original but address the same quality dimensions. Key additions:

1. **Language matching** — detect language from user query, generate report in same language
2. **Structure adaptation** — select report format based on query type (comparison, enumeration, explanation, how-to)
3. **Anti-pattern guardrails** — avoid meta-commentary, self-reference, hedging openers
4. **Citation consistency** — enforce inline `[Title](URL)` + numbered source section at end

### Changes

1. Extend `_build_synthesis_system_prompt()` with structure-adaptive and language-aware directives
2. Add a query-type classifier to `_build_synthesis_user_prompt()` that hints at appropriate structure
3. Update citation formatting rules to match the inline + numbered list pattern

### Validation

- Test: synthesis output matches user query language when non-English input provided
- Test: comparison queries produce comparison-structured reports
- Test: output does not contain meta-commentary anti-patterns
- Test: citations are consistently formatted

---

## Phase 4: Message-Aware Token Recovery

**Effort:** Medium | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` (lines 199-478)
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`

### Problem

When token limits are hit, open_deep_research uses `remove_up_to_last_ai_message()` (utils.py:848-866) to progressively strip messages from conversation history, preserving the system prompt and earliest context. foundry-mcp uses `_truncate_for_retry()` which chops content from the end of a single user prompt string. The foundry-mcp approach risks cutting the most recently gathered (potentially most relevant) findings.

### Design

Add a structured content truncation strategy to `execute_llm_call()` that understands the prompt contains multiple sections (sources, findings, context). Instead of blind tail-chopping:

1. On first retry: truncate individual source content blocks (longest first)
2. On second retry: drop lowest-quality sources entirely (if quality scores available)
3. On third retry: fall back to current char-based truncation

This is better suited to foundry-mcp's architecture than open_deep_research's message-list approach, since foundry-mcp builds single prompts rather than multi-turn message histories.

### Changes

1. Add `_structured_truncation()` helper to `_helpers.py` that understands source/finding block boundaries
2. Update retry loop in `execute_llm_call()` to try structured truncation before char truncation
3. Keep current char-based truncation as final fallback

### Validation

- Test: structured truncation preserves high-quality sources while dropping verbose low-quality ones
- Test: char-based fallback still works when structured truncation is insufficient
- Test: no regression in existing token recovery tests

---

## Phase 5: Reflection Enforcement

**Effort:** Low | **Impact:** Medium
**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (lines 355-483)

### Problem

open_deep_research enforces `think_tool` usage before every research action via explicit prompt directives ("ALWAYS use think_tool BEFORE each ConductResearch call"). foundry-mcp's `_topic_reflect()` runs after searches but the enforcement is softer — reflection is called but the decision rules could be tighter.

### Design

Strengthen the reflection prompt (topic_research.py:395-417) with clearer decision boundaries, modeled after open_deep_research's hard limits:

1. **Hard stop rules**: research_complete=true when 3+ relevant sources from distinct domains
2. **Continue rules**: continue_searching=true only if fewer than 2 relevant sources
3. **Absolute ceiling**: never exceed `deep_research_topic_max_searches` regardless of reflection decision
4. **Rationale requirement**: reflection must articulate what specific gap would be filled by continuing

Also ensure the reflection call is sequenced strictly — never run reflection and search in parallel.

### Changes

1. Update reflection system prompt in `_topic_reflect()` with tighter decision rules
2. Add guard: if iteration count >= max_searches, force research_complete regardless of reflection
3. Add logging of reflection rationale for observability

### Validation

- Test: reflection returns research_complete=true after 3+ sources from distinct domains
- Test: reflection never recommends continuing past max_searches
- Test: rationale field is always non-empty

---

## Phase 6: Iterative Supervisor Architecture (Future)

**Effort:** High | **Impact:** High
**Files:** Multiple — requires new orchestration layer

### Problem

open_deep_research has a true supervisor agent (deep_researcher.py:178-349) that iteratively plans, delegates to parallel sub-researchers, assesses coverage gaps, and spawns additional research rounds (up to 6 iterations). foundry-mcp's architecture is "plan once, execute in parallel" — the planning phase generates sub-queries, topic agents execute them, and there's no meta-level loop that re-assesses and fills gaps.

### Design (Sketch — not for immediate implementation)

This is the largest architectural enhancement. The key insight from open_deep_research is the assess-delegate-assess loop:

1. Supervisor reviews current findings after each batch of topic research completes
2. Identifies coverage gaps (topics with insufficient sources, unanswered sub-questions)
3. Generates new sub-queries targeting gaps
4. Spawns additional topic agents for the new queries
5. Repeats until satisfied or iteration limit reached

This would require:
- A new `supervision` phase between gathering/compression and analysis
- State tracking for "coverage assessment" across iterations
- Budget management (tokens, API calls) across iterations
- Integration with the existing phase lifecycle in `_lifecycle.py`

### Deferral Rationale

Phases 1-5 improve every research run with moderate effort. Phase 6 is the biggest quality jump but also the biggest architectural change. Implement Phases 1-5 first, measure impact, then evaluate whether Phase 6 is warranted based on observed gap-coverage issues in production research runs.

---

## Reference: Alignment Status After Prior Work

These areas were aligned in earlier commits on this branch and do NOT need further work:

| Area | Status | Aligned In |
|------|--------|-----------|
| L1 summarization input cap (50K chars) | Aligned | `b684a7c` |
| L2 compression prompt semantics | Aligned | `7d449bd` |
| Compression verbatim preservation | Aligned | `7d449bd` |
| Token retry count (3) | Aligned | prior work |
| Truncation factor (0.9/retry) | Aligned | prior work |
| Citation format (sequential numbering) | Aligned | `7d449bd` |
| Summarization targets (25-30%, 5 excerpts) | Aligned | `b684a7c` |
| Fetch-time summarization (L1) | Aligned | `0350c98` |
| Per-topic compression (L2) | Aligned | `3981035` |
| Forced reflection in topic research | Aligned | `9961686` |
| Clarification gate | Aligned | `ebb8c76` |
| Multi-model cost routing | Aligned | `c9e88ac` |
