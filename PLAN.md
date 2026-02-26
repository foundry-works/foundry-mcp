# PLAN: Address Review Findings — Deep Research Workflow Refactor

> **Goal**: Fix all critical, high, and medium severity issues identified during senior-engineer review of the deep research workflow refactor branch.
>
> **Scope**: 6 phases, estimated ~500-800 LOC changes + ~200 LOC test additions
>
> **Risk**: Low-Medium. Most changes are correctness/cleanup with no behavioral change to the happy path. Phase 1 (critical/high) requires careful async reasoning.

---

## Phase 1: Critical & High-Priority Fixes

**Objective**: Eliminate the deadlock risk, restore cancellation semantics, fix legacy resume crash, and harden SSRF protection.

### 1a. Fix `_evaluate_research` deadlock risk (CRITICAL)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`
**Lines**: 573-585

**Problem**: `run_coroutine_threadsafe(coro, loop)` followed by `future.result()` will deadlock if called from the event loop thread. The `_run_sync` method (lines 612-675) in the same file handles this correctly using `ThreadPoolExecutor`.

**Fix**: Refactor `_evaluate_research` to use the same dispatch pattern as `_run_sync`:
1. Extract the async evaluation logic into a standalone async function.
2. In the sync entry point, detect whether we're on the event loop thread using `loop.is_running()`.
3. If on the event loop thread, dispatch to a `ThreadPoolExecutor` that runs `asyncio.run()` internally (same pattern as `_run_sync` lines 647-649).
4. If no event loop, call `asyncio.run()` directly.

**Test**: Add a test that calls `_evaluate_research` from within an async context (simulating the deadlock scenario) and verifies it completes without hanging.

### 1b. Restore `CancelledError` propagation (HIGH)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
**Lines**: 379-485

**Problem**: `except asyncio.CancelledError` catches and returns `WorkflowResult` instead of re-raising. This breaks Python's cancellation contract — callers using `asyncio.wait_for()` or `Task.cancel()` won't see cancellation propagate.

**Fix**:
1. Keep the existing state cleanup / rollback logic (lines 384-469).
2. After cleanup and state persistence, **re-raise** `CancelledError` instead of returning a `WorkflowResult`.
3. In `background_tasks.py`, the existing guard `if state.completed_at is None` (around lines 96-116) already handles this — verify it catches the re-raised `CancelledError` and performs the same finalization.

**Test**: Verify that `asyncio.Task.cancel()` on a running workflow results in `CancelledError` being raised to the caller, and that state is still properly rolled back.

### 1c. Fix legacy resume crash for PLANNING phase (HIGH)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
**Lines**: ~200 (after BRIEF handling, before GATHERING)
**File**: `src/foundry_mcp/core/research/workflows/deep_research/core.py` (lines 104-121)

**Problem**: `PlanningPhaseMixin` was removed from the `DeepResearchWorkflow` MRO but `phases/__init__.py` still exports it. A saved state at `PLANNING` phase would hit the `advance_phase()` fallthrough but `self._execute_planning_async()` doesn't exist on the class — `AttributeError` at runtime.

**Fix** (option b — clean removal):
1. In `workflow_execution.py`, add explicit handling for `DeepResearchPhase.PLANNING` between the `BRIEF` and `GATHERING` blocks:
   ```python
   if state.phase == DeepResearchPhase.PLANNING:
       logger.warning(
           "PLANNING phase running from legacy saved state (research %s) "
           "— advancing to SUPERVISION",
           state.id,
       )
       self._write_audit_event(state, "legacy_phase_resume", data={
           "phase": "planning", "deprecated_phase": True,
       }, level="warning")
       state.advance_phase()  # PLANNING → GATHERING → SUPERVISION
   ```
2. Remove `PlanningPhaseMixin` from `phases/__init__.py` `__all__` and the import (or keep the import with a comment that it's intentionally not in the MRO).
3. Verify `DeepResearchPhase.PLANNING` still exists in the enum for deserialization of legacy states.

**Test**: Create a saved state at PLANNING phase, resume it, and verify it advances to SUPERVISION without error.

### 1d. Harden SSRF validation with DNS rebinding note (HIGH)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`
**Lines**: 809-857

**Problem**: `validate_extract_url()` checks IP literals but not DNS resolution. A hostname resolving to `169.254.169.254` bypasses the check. The Tavily provider's version (in `tavily_extract.py`) supports `resolve_dns` parameter.

**Fix**:
1. Add a docstring note explicitly documenting the TOCTOU gap: "This function validates URL syntax and blocks obvious IP-literal attacks. DNS rebinding attacks (hostname resolving to private IP) are NOT blocked here — the fetch layer (Tavily API) operates in its own network context."
2. Add an optional `resolve_dns: bool = False` parameter that, when True, performs `socket.getaddrinfo()` and checks the resolved IP against `_BLOCKED_NETWORKS`.
3. In the `ReflectionDecision._coerce_urls` validator (deep_research.py line 232-248), call with `resolve_dns=True` since these URLs come from LLM output and are used for server-side fetch.

**Test**: Add tests for DNS rebinding detection when `resolve_dns=True`.

---

## Phase 2: Dead Code Cleanup

**Objective**: Remove unreachable code that inflates the MRO and maintenance burden.

### 2a. Remove `AnalysisPhaseMixin` from MRO

**File**: `src/foundry_mcp/core/research/workflows/deep_research/core.py` (line 115)
**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/__init__.py`

The ANALYSIS phase is no longer reachable in the active workflow. The `advance_phase()` logic skips directly from GATHERING to SUPERVISION.

1. Remove `AnalysisPhaseMixin` from `DeepResearchWorkflow` base classes in `core.py`.
2. Mark the export in `__init__.py` as deprecated (like PlanningPhaseMixin).
3. Verify no runtime code calls `self._execute_analysis_async()`.

### 2b. Remove `RefinementPhaseMixin` from MRO

**File**: `src/foundry_mcp/core/research/workflows/deep_research/core.py` (line 117)

Same treatment as AnalysisPhaseMixin — the REFINEMENT phase was removed from the pipeline.

1. Remove from base classes in `core.py`.
2. Mark export as deprecated in `__init__.py`.

### 2c. Delete `supervision_legacy.py`

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_legacy.py` (627 lines)

`LegacySupervisionMixin` is never in the class hierarchy and no config path reaches it.

1. Delete the file entirely.
2. Remove any imports/references.
3. If there's concern about losing the code, note that it's preserved in git history.

### 2d. Clean up `PlanningPhaseMixin` export

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/__init__.py` (lines 12, 21)

After 1c adds explicit skip handling, the mixin export is only needed for external consumers that may import it directly (unlikely).

1. Remove from `__all__`.
2. Keep the import line with comment: `# Retained for git history reference only; not in active MRO`
3. Or delete outright if no external consumers exist.

---

## Phase 3: Type Safety & Config Hardening

**Objective**: Eliminate `getattr(self.config, ...)` anti-pattern and wire up the protocol/sub-config infrastructure that was created but not connected.

### 3a. Wire `DeepResearchWorkflowProtocol` into phase mixins

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/_protocols.py` (lines 33-66)
**Files**: All phase mixin files (supervision.py, topic_research.py, synthesis.py, compression.py, brief.py, gathering.py, analysis.py, clarification.py)

The protocol already defines `config`, `memory`, `_write_audit_event`, `_check_cancellation`, and `_execute_provider_async`. Currently each mixin redeclares these independently.

1. In each mixin's `TYPE_CHECKING` block, import `DeepResearchWorkflowProtocol`.
2. Add a class-level type annotation: `_self: "DeepResearchWorkflowProtocol"` or use `Self` from `typing_extensions`.
3. Remove the duplicated `config: Any`, `memory: Any`, and per-mixin `TYPE_CHECKING` stubs.
4. Extend the protocol with any additional methods that mixins currently stub (e.g., `_execute_topic_research_async`).

### 3b. Replace `getattr(self.config, ...)` with direct attribute access

**Files**: ~19 occurrences across:
- `action_handlers.py` (lines 115, 135, 556-558)
- `phases/supervision.py`
- `phases/topic_research.py`
- `phases/gathering.py`
- `audit.py`
- `persistence.py`

For each `getattr(self.config, "deep_research_foo", default)`:
1. Verify the attribute exists on `ResearchConfig` dataclass.
2. Replace with `self.config.deep_research_foo`.
3. If the attribute genuinely might not exist (backward compat), add a `hasattr` check with explicit comment explaining why.

This ensures typos are caught at type-check time rather than silently returning defaults.

---

## Phase 4: Module-Level Side Effects & Bounded Growth

**Objective**: Fix module-level I/O and ensure in-round message growth stays bounded.

### 4a. Lazy-load `MODEL_TOKEN_LIMITS`

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` (lines 96-130)

Replace module-level `MODEL_TOKEN_LIMITS = _load_model_token_limits()` with lazy loading:
```python
@functools.lru_cache(maxsize=1)
def get_model_token_limits() -> dict[str, int]:
    return _load_model_token_limits()
```

Update all references (line 582 in same file, plus any imports) to call `get_model_token_limits()`.

### 4b. Promote `_FALLBACK_CONTEXT_WINDOW` to public

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py` (line 511)
**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (line 34)

Rename `_FALLBACK_CONTEXT_WINDOW` → `FALLBACK_CONTEXT_WINDOW` (remove leading underscore).
Update all import sites (synthesis.py, _lifecycle.py internal usages).

### 4c. Post-round supervision message truncation

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
**Lines**: ~468-507 (after `_execute_and_merge_directives`)

Add a `truncate_supervision_messages()` call after all directive results have been appended within a round, not just at the start of the next round. This prevents temporary memory spikes from large directive result sets.

Specifically, add truncation at the end of `_post_round_bookkeeping()` or at the exit of `_execute_and_merge_directives()`.

---

## Phase 5: Test Improvements

**Objective**: Address the most impactful test gaps and quality issues.

### 5a. Add topic research error path tests

**File**: `tests/core/research/workflows/test_topic_research.py`

Add tests for:
1. Unknown tool name in researcher response (e.g., `{"tool": "invalid_tool", ...}`)
2. Budget exhaustion mid-loop (tool calls hit max without `research_complete`)
3. `extract_content` tool failure (provider returns error or empty content)

### 5b. Add supervision to cross-phase integration test

**File**: `tests/core/research/workflows/test_cross_phase_integration.py`

Add `SupervisionPhaseMixin` to the `StubWorkflow` class composition. Add a test case that exercises the full path: CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS with a simplified mock.

### 5c. Fix `test_supervision_skipped_when_disabled`

**File**: `tests/core/research/workflows/deep_research/test_supervision.py` (line 522)

The test reimplements production logic inline. Rewrite to call actual `_execute_workflow_async` with `config.deep_research_enable_supervision = False` and verify the phase transition skips SUPERVISION.

### 5d. Build evaluation mock responses from `DIMENSIONS` keys

**File**: `tests/core/research/evaluation/test_evaluator.py`

Replace hand-written dimension names in `_make_valid_eval_response()` with dynamic construction from the `DIMENSIONS` constant to prevent drift.

---

## Phase 6: Low-Priority Cleanup

**Objective**: Address low-severity items that improve code quality but are not blocking.

### 6a. Extract `build_sanitized_context(state)` helper

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py`

Create a helper that returns pre-sanitized versions of common state fields:
```python
def build_sanitized_context(state) -> dict[str, str]:
    return {
        "original_query": sanitize_external_content(state.original_query or ""),
        "system_prompt": sanitize_external_content(state.system_prompt or ""),
        "constraints": sanitize_external_content(
            str(state.clarification_constraints) if state.clarification_constraints else ""
        ),
        "research_brief": sanitize_external_content(state.research_brief or ""),
    }
```

Update prompt builders in `brief.py`, `planning.py`, `supervision_prompts.py` to use it.

### 6b. Fix `extract_json()` backslash handling

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` (lines 50-77)

The `escape` flag logic should only trigger inside strings:
```python
if char == "\\":
    if in_string:
        escape = True
    continue  # ← this 'continue' is wrong; a backslash outside a string
              # should not skip the next character
```

Fix: Only set `escape = True` when `in_string` is True, and remove the `continue` for the non-string case.

### 6c. Add JSON/fallback token limits sync test

**File**: `tests/unit/test_config_phase4.py` (new test or existing)

Add a test that verifies `model_token_limits.json` and `_FALLBACK_MODEL_TOKEN_LIMITS` contain the same keys and values, preventing drift.

### 6d. Consolidate `_make_state()` test helpers

**Files**: 17 test files with duplicate `_make_state()` definitions

Consolidate the most common patterns into `tests/core/research/workflows/deep_research/conftest.py`. Add preset factories like `make_supervision_state()`, `make_gathering_state()` that wrap `make_test_state()` with phase-specific defaults. Migrate test files to use the shared helpers.

> **Note**: This is a large but low-risk refactor. Can be done incrementally per test file.
