# PLAN: Pre-Merge Hardening — Review Findings Remediation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-25
**Status:** Draft

---

## Context

Senior engineering review of the full branch (84 commits, +32.5k/-6.4k lines) identified 5 critical, 15 major, and 20+ minor issues across the deep research workflow. This plan addresses all critical items and the highest-impact major items, organized into 6 phases by dependency order and blast radius.

---

## Phase 1: Runtime Crash Fixes (Critical Path)

**Effort:** Low | **Impact:** Critical
**Goal:** Eliminate issues that cause runtime crashes or incorrect error recovery.

### 1A. Add missing `deep_research_reflection_timeout` config field

**Problem:** `orchestration.py:496` does direct attribute access on `self.config.deep_research_reflection_timeout`, but the field is never defined on `ResearchConfig`. This raises `AttributeError` during LLM-driven reflection at phase boundaries.

**Files:**
- `src/foundry_mcp/config/research.py`

**Changes:**
1. Add field `deep_research_reflection_timeout: float = 60.0` to `ResearchConfig`
2. Add corresponding entry in `from_toml_dict()`: `float(data.get("deep_research_reflection_timeout", 60.0))`
3. Audit all `getattr(self.config, "deep_research_reflection_timeout", ...)` call sites — they can remain as-is (defensive fallback) but should use the same default (60.0)

### 1B. Tighten context-window error detection patterns

**Problem:** `_lifecycle.py:401` pattern `r"(?i)\b(?:token|context|length|maximum.*context)\b"` false-positives on "invalid authentication token", "context parameter is required", etc. `InvalidArgument` (line 414) is a generic gRPC class — classifying all instances as context-window errors triggers incorrect truncation-retry loops.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`

**Changes:**
1. Replace the first regex pattern with tighter alternatives:
   - `r"(?i)\b(?:token|context)\s*(?:limit|window|exceeded|too\s+long|overflow)\b"`
   - `r"(?i)maximum\s+(?:context|token)"`
   - `r"(?i)(?:exceeds?|over)\s+(?:the\s+)?(?:token|context|length)\s+limit"`
2. Remove `"InvalidArgument"` from `_CONTEXT_WINDOW_ERROR_CLASSES`
3. Add a combined check: if exception class name is `"InvalidArgument"` AND message matches a token-related pattern, then classify as context-window error
4. Add unit tests for false-positive cases: "invalid authentication token", "context parameter is required", "length must be positive"

### 1C. Fix async/sync bridge in evaluation action handler

**Problem:** `action_handlers.py:622-659` spawns `asyncio.run()` inside a `ThreadPoolExecutor` when a loop is already running. This bypasses structured concurrency, breaks cancellation, and can deadlock.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`

**Changes:**
1. Replace the `ThreadPoolExecutor` + `asyncio.run` pattern with `asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=...)`
2. Alternatively (preferred): make `_evaluate_research` an async method and call it directly since the MCP dispatch layer already supports async handlers
3. Remove the `RuntimeError` fallback that was masking the fragility

---

## Phase 2: Data Quality & Correctness

**Effort:** Low-Medium | **Impact:** High
**Goal:** Fix issues that produce incorrect results or silently corrupt data.

### 2A. Consolidate duplicate `_extract_domain` implementations

**Problem:** `_helpers.py:784` (manual string splitting) and `source_quality.py:19` (`urlparse`-based) disagree on URLs without schemes, with ports, and malformed URLs. Both are used in the topic research path — novelty scoring uses one, search result formatting uses the other.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`
- `src/foundry_mcp/core/research/workflows/deep_research/source_quality.py` (reference — no changes needed)

**Changes:**
1. Delete `_extract_domain` from `_helpers.py` (lines 784-803)
2. Import `_extract_domain` from `source_quality` in `_helpers.py`
3. Update `compute_novelty_tag` (line 753) to use the imported version
4. Verify no other callers reference the deleted version

### 2B. Unify `max_supervision_rounds` defaults

**Problem:** Default is 6 in `config/research.py:106`, 6 in `research_sub_configs.py:119`, but 3 in `models/deep_research.py:870` and 3 in `action_handlers.py:132` `getattr` fallback. Sessions created without explicit config get half the intended rounds.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`
- `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`

**Changes:**
1. Define `DEFAULT_MAX_SUPERVISION_ROUNDS = 6` as a constant in `models/deep_research.py`
2. Update `DeepResearchState.max_supervision_rounds` Field default to use the constant
3. Update `action_handlers.py:132` `getattr` fallback to use the constant
4. Import the constant in both locations to ensure single source of truth

### 2C. Track imputed evaluation dimensions

**Problem:** Missing dimensions in LLM evaluation response default to score 3 (0.5 normalized) with no tracking. Composite score can be misleadingly inflated.

**Files:**
- `src/foundry_mcp/core/research/evaluation/scoring.py`
- `src/foundry_mcp/core/research/evaluation/evaluator.py`

**Changes:**
1. Add `imputed: bool = False` field to `DimensionScore` dataclass
2. Set `imputed=True` when building fallback scores in `evaluator.py:199-201`
3. Add `imputed_count` to `EvaluationResult.metadata` in the composite calculation
4. Exclude imputed dimensions from variance calculation (they add noise, not signal)
5. Add `warnings` list to `EvaluationResult.metadata` when imputed dimensions > 0

### 2D. Fix unweighted variance in evaluation scoring

**Problem:** Composite uses weighted scores but variance is computed from unweighted scores with an unweighted mean.

**Files:**
- `src/foundry_mcp/core/research/evaluation/scoring.py`

**Changes:**
1. Compute weighted variance using effective weights and the weighted mean (the composite itself)
2. Only include non-imputed dimensions in variance calculation (from 2C)

---

## Phase 3: Concurrency & Error Handling

**Effort:** Medium | **Impact:** High
**Goal:** Fix error handling paths that can crash, hang, or silently corrupt state under concurrency.

### 3A. Fix `asyncio.gather` exception handling in directive execution

**Problem:** `supervision.py:1034` uses `return_exceptions=False`. Unexpected exceptions before the per-task try/except block crash the entire supervision phase.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. Change `return_exceptions=False` to `return_exceptions=True` at line 1034
2. After gather, filter results: if any is `asyncio.CancelledError`, re-raise it
3. For other `BaseException` results, log as non-fatal and count as failed directives
4. Mirror the pattern already used at line 1144 (`_compress_directive_results_inline`)

### 3B. Propagate cancellation from compression gather

**Problem:** `compression.py:757` uses `return_exceptions=True` which silently catches `CancelledError`, hiding session cancellation.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Changes:**
1. After gather at line 757, check if any result is `asyncio.CancelledError`
2. If found, re-raise to propagate cancellation upward
3. Process remaining non-cancellation results normally

### 3C. Add global budget cap to structured metadata prompt

**Problem:** `compression.py:166-174` caps per-source content at 50k chars but has no global cap. 20 sources = 1M chars potential prompt size.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Changes:**
1. Track cumulative content length in `_build_structured_metadata_prompt`
2. Stop adding source content once global budget is exhausted (e.g., 200k chars total)
3. Add truncation marker for omitted sources: `[... N additional sources omitted for context limits]`

### 3D. Fix budget rebalance overflow in supervision message truncation

**Problem:** `_lifecycle.py:315-351` rebalance phase restores messages checking only per-bucket constraints, not the global total. `findings_remaining + reasoning_remaining` can exceed `budget_chars`.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py`

**Changes:**
1. Track `total_remaining = findings_remaining + reasoning_remaining` during rebalance
2. Add guard: `if total_remaining >= budget_chars: break` before each restoration

---

## Phase 4: State Model Hygiene

**Effort:** Medium | **Impact:** Medium
**Goal:** Fix state model issues that create confusion or latent bugs.

### 4A. Rename conflicting `ReflectionDecision` class

**Problem:** Two unrelated classes named `ReflectionDecision` — Pydantic model in `models/deep_research.py:204` (topic researcher) and stdlib dataclass in `orchestration.py:97` (supervisor phase boundary).

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`

**Changes:**
1. Rename `ReflectionDecision` in `orchestration.py` to `PhaseReflectionDecision`
2. Update all references in `orchestration.py` (class definition, type annotations, instantiation sites)
3. Search for any external imports and update them

### 4B. Skip deprecated GATHERING phase in `advance_phase()`

**Problem:** `advance_phase()` traverses enum order BRIEF → GATHERING → SUPERVISION. GATHERING is deprecated. `workflow_execution.py` works around this by direct assignment.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`

**Changes:**
1. Add a skip-set to `advance_phase()`: `_SKIP_PHASES = {DeepResearchPhase.GATHERING}`
2. After advancing, if `self.phase in _SKIP_PHASES`, advance again
3. Add a comment explaining this is for backward compatibility with the deprecated phase
4. Remove the direct `state.phase = DeepResearchPhase.SUPERVISION` workaround in `workflow_execution.py` (it can now use `advance_phase()`)

### 4C. Cap `state.directives` growth

**Problem:** `supervision.py:327` appends directives every round without pruning. State serialization grows monotonically.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Changes:**
1. After `state.directives.extend(directives)` at line 327, cap the list: `state.directives = state.directives[-30:]`
2. Add a constant `_MAX_STORED_DIRECTIVES = 30` at module level
3. Document the cap in a comment explaining the serialization concern

---

## Phase 5: Prompt Injection Surface Reduction

**Effort:** Low | **Impact:** Medium
**Goal:** Reduce prompt injection risk from web-scraped content flowing into supervision prompts.

### 5A. Sanitize external content in supervision messages

**Problem:** `supervision.py:818-839` renders web-scraped source titles and content snippets directly into supervision user prompts without sanitization.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` (add sanitizer)

**Changes:**
1. Add `sanitize_external_content(text: str) -> str` to `_helpers.py`:
   - Strip XML-like tags that could override instructions: `<system>`, `<instructions>`, `<tool_use>`, `<human>`, `<assistant>`, etc.
   - Strip markdown heading sequences that could inject structure: `# SYSTEM`, `## INSTRUCTIONS`
   - Preserve normal content (citations, formatting, data)
2. Apply sanitizer to source titles and content snippets before inclusion in supervision messages
3. Apply sanitizer in `_build_directive_fallback_summary` (lines 1204-1224) for source-derived content

---

## Phase 6: Test Gaps & Quality

**Effort:** Medium | **Impact:** Medium
**Goal:** Fix broken tests and add coverage for critical untested paths.

### 6A. Fix vacuously true test assertion

**Problem:** `test_evaluator.py:746` has `assert ... or True` which can never fail.

**Files:**
- `tests/core/research/evaluation/test_evaluator.py`

**Changes:**
1. Fix the mock to capture the user prompt at the right layer (intercept `execute_llm_call` instead of `_execute_provider_async`)
2. Remove the `or True` fallback
3. Assert that raw notes content appears in the captured prompt

### 6B. Add cancellation propagation tests

**Files:**
- `tests/core/research/workflows/deep_research/test_supervision.py`
- `tests/core/research/workflows/deep_research/test_inline_compression.py`

**Changes:**
1. Add test: cancellation during directive execution propagates `CancelledError` upward
2. Add test: cancellation during compression gather propagates `CancelledError` upward
3. Add test: partial results are preserved when cancellation fires mid-batch

### 6C. Add context-window error detection regression tests

**Files:**
- `tests/core/research/workflows/deep_research/test_compression_retry_truncation.py` (or new file)

**Changes:**
1. Test: "invalid authentication token" is NOT classified as context-window error
2. Test: "context parameter is required" is NOT classified as context-window error
3. Test: "maximum context length exceeded" IS classified as context-window error
4. Test: `InvalidArgument` with non-token message is NOT classified as context-window error

---

## Verification

```bash
# Full deep research test suite
python -m pytest tests/core/research/ -x -q

# Specific areas touched by each phase
python -m pytest tests/core/research/workflows/deep_research/ -x -q
python -m pytest tests/core/research/evaluation/ -x -q
python -m pytest tests/unit/test_config_tavily.py -x -q

# Import smoke test for renamed class
python -c "from foundry_mcp.core.research.workflows.deep_research.orchestration import PhaseReflectionDecision"

# Config field smoke test
python -c "from foundry_mcp.config.research import ResearchConfig; c = ResearchConfig(); print(c.deep_research_reflection_timeout)"
```

---

## Out of Scope (Backlog)

Items identified in review but deferred to follow-up work:

- **Supervision mixin decomposition** (3,200-line mixin → SupervisionOrchestrator + DirectiveExecutor + CoverageAnalyzer + PromptBuilder)
- **ResearchConfig decomposition** (80+ flat fields → composed sub-configs as actual storage)
- **Typed terminal status field** on `DeepResearchState` (replace metadata-based status tracking)
- **Language-aware token estimation** (4 chars/token heuristic → language-aware multiplier for CJK/Arabic)
- **ReAct message history truncation** (context window budget check on growing conversation)
- **O(n^2) content-similarity dedup optimization** (move computation outside state lock, add shingle cache)
- **Test infrastructure consolidation** (shared stubs/fixtures in conftest.py)
- **Legacy supervision code removal** (`_execute_supervision_query_generation_async` ~500 lines)
- **Module-level import cleanup** (move deferred imports out of hot paths)
- **`EvaluationResult.from_dict()` deserialization**
- **`_extract_markdown_report` nested code block handling**
