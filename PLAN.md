# PLAN: Pre-Merge Fixes from Senior Engineer Review

Fixes identified during comprehensive senior engineer review of the deep research workflow overhaul
(branch `tyler/foundry-mcp-20260223-0747`, 100 commits, +39k/-6.6k lines across 100 files).

Organized into 5 phases: architectural must-fix, security/sanitization, test coverage gaps, config bugs, then cleanup.

---

## Phase 1 — Architectural: Split `supervision.py`

### 1.1 Extract prompt builders into `supervision_prompts.py`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
**New file:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_prompts.py`

**Problem:** `supervision.py` is 3365 lines — a God File mixing orchestration logic, prompt construction, coverage assessment, and legacy compat. Three of four future plans (PLAN-0 through PLAN-4) modify this file, making it a merge bottleneck.

**Fix:** Extract all `_build_*_system_prompt` and `_build_*_user_prompt` methods (~700 lines) into `supervision_prompts.py` as module-level functions. These are pure functions that take state/config/data and return strings — they don't need `self`. Update imports in `supervision.py` to call the extracted functions.

Methods to extract:
- `_build_delegation_system_prompt` / `_build_delegation_user_prompt`
- `_build_combined_think_delegate_system_prompt` / `_build_combined_think_delegate_user_prompt`
- `_build_first_round_think_prompt` / `_build_first_round_delegation_system_prompt` / `_build_first_round_delegation_user_prompt`
- `_build_critique_system_prompt` / `_build_critique_user_prompt`
- `_build_revision_system_prompt` / `_build_revision_user_prompt`
- `_build_think_prompt`
- `_build_supervision_system_prompt` / `_build_supervision_user_prompt` (legacy)

### 1.2 Extract coverage assessment into `supervision_coverage.py`

**New file:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

**Problem:** Coverage assessment (~400 lines) is analysis utility code independent of the LLM orchestration loop.

**Fix:** Extract these methods as module-level functions:
- `_build_per_query_coverage`
- `_store_coverage_snapshot`
- `_compute_coverage_delta`
- `_assess_coverage_heuristic`
- The `_VERDICT_*` and `_ISSUE_MARKER_RE` regex patterns
- `_critique_has_issues`

### 1.3 Decompose `_execute_supervision_delegation_async`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Problem:** This method is 428 lines containing the entire supervision loop: wall-clock timeout, heuristic early-exit, think/delegate dispatch, directive execution, inline compression, raw notes trimming, evidence inventory, post-execution think, history recording, and state saving. A bug in any mutation point can corrupt state.

**Fix:** Decompose into named sub-methods:
- `_should_exit_wall_clock(wall_clock_start, wall_clock_limit, state) -> bool`
- `_should_exit_heuristic(state, min_sources) -> bool`
- `_run_think_delegate_step(state, ...) -> tuple[think_output, directives, research_complete]`
- `_execute_and_merge_directives(state, directives, timeout) -> None`
- `_post_round_bookkeeping(state, directives, think_output, ...) -> None`

The main loop becomes ~40 lines calling these sub-methods.

### 1.4 Extract duplicate supervision message rendering

**Problem:** Supervision message rendering code is duplicated between `_build_delegation_user_prompt` (lines 1599-1633) and `_build_combined_think_delegate_user_prompt` (lines 909-943).

**Fix:** Extract into a shared helper `_render_supervision_conversation_history(state, messages) -> str` used by both prompt builders. This goes in the new `supervision_prompts.py`.

### 1.5 Move inline import to module level

**Problem:** `_normalize_title` is imported inline inside `_execute_directives_async` (line 1071). While there's no actual circular dependency, inline imports obscure the dependency graph.

**Fix:** Move to module-level import block.

---

## Phase 2 — Security: Sanitization Consistency

### 2.1 Sanitize `state.original_query` in supervision prompt builders

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (or the new `supervision_prompts.py`)

**Problem:** `state.original_query` is sanitized in brief and clarification phases but interpolated raw in 8+ supervision prompt builders (`_build_delegation_user_prompt`, `_build_combined_think_delegate_user_prompt`, `_build_first_round_think_prompt`, `_build_first_round_delegation_user_prompt`, `_build_critique_user_prompt`, `_build_revision_user_prompt`, `_build_think_prompt`, etc.). While the user controls their own query (same trust domain), this inconsistency violates defense-in-depth.

**Fix:** Apply `sanitize_external_content()` to `state.original_query` at all interpolation points. Also sanitize `state.research_brief` and `state.system_prompt` where they appear in supervision prompts.

### 2.2 Sanitize `state.original_query` in planning prompt builders

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/planning.py`

**Problem:** Same inconsistency as 2.1. The planning phase interpolates `state.original_query` raw in `_build_brief_refinement_prompt` (line 357), `_build_planning_user_prompt` (line 386), and `_build_decomposition_critique_prompt` (line 535). Also `state.system_prompt` (lines 361, 400) and `state.clarification_constraints` (lines 365-366) are interpolated raw.

**Fix:** Apply `sanitize_external_content()` to all user-supplied and LLM-derived content in planning prompts, matching what brief/clarification already do.

### 2.3 Sanitize second-order injection vectors in supervision

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Problem:** LLM-influenced content is interpolated unsanitized into supervision prompts:
- `entry['query']` from coverage data (lines 1640, 950) — sub-query text that could be influenced by a compromised web source via a previous LLM round
- `d.research_topic[:120]` from directives (lines 1683, 962) — LLM-generated text

**Fix:** Apply `sanitize_external_content()` to:
- `entry['query']` in `_build_delegation_user_prompt` and `_build_combined_think_delegate_user_prompt`
- `d.research_topic` in the "Previously Executed Directives" sections

---

## Phase 3 — Test Coverage: Restore Deleted Test Coverage

### 3.1 Restore digest integration test coverage

**Deleted file:** `tests/core/research/test_deep_research_digest.py` (933 lines, 28 tests)
**Deleted file:** `tests/core/research/test_proactive_digest.py` (385 lines, 12 tests)
**Production code still exists:** `_analysis_digest.py`, `_execute_digest_step_async`

**Problem:** The digest pipeline's integration with deep research is untested. The production code was NOT removed — only the tests were deleted. This is the most significant coverage gap.

**Fix:** Either:
- **(a)** Confirm digest integration is dead code on the new pipeline (supervision replaces it) and remove the production code, OR
- **(b)** Write replacement integration tests covering: budget allocation, citation rendering, multi-iteration dedup, timeout budgeting, max sources limits, fidelity tracking on errors

### 3.2 Restore contradiction detection test coverage

**Deleted file:** `tests/core/research/workflows/test_contradiction_detection.py` (749 lines, 25 tests)
**Production code still exists:** `_detect_contradictions()` in `analysis.py` (line 353)

**Problem:** The contradiction detection logic (JSON parsing, finding ID validation, severity validation, empty description filtering, error handling) is no longer unit-tested. Cross-phase tests cover synthesis prompt building *with* contradictions present, but not the detection logic itself.

**Fix:** Either:
- **(a)** Write focused unit tests for `_detect_contradictions()` covering: valid JSON, malformed JSON fallback, invalid finding IDs, severity validation, empty descriptions, token tracking, OR
- **(b)** If contradiction detection is vestigial (contradictions now come from compression/supervision), remove the dead code

### 3.3 Restore orchestration reflection test coverage

**Deleted file:** `tests/core/research/workflows/test_reflection.py` (751 lines, 31 tests)
**Production code still exists:** `PhaseReflectionDecision` in `orchestration.py`, `_async_think_pause`, `maybe_reflect`

**Problem:** Orchestration-level reflection is untested. The supervision think-tool tests cover a related but architecturally different code path.

**Fix:** Either:
- **(a)** Write replacement tests for `_async_think_pause` and `maybe_reflect`, OR
- **(b)** If these code paths are superseded by supervision think-tool, remove the dead code and mark it in the commit

### 3.4 Add dedicated brief phase tests

**Production code:** `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (284 lines)

**Problem:** Brief phase is only tested via 3 cross-phase integration tests in `test_cross_phase_integration.py`. No dedicated edge case coverage (LLM failure fallback, malformed JSON, empty brief, brief with missing fields).

**Fix:** Add `tests/core/research/workflows/deep_research/test_brief.py` with:
- Non-fatal failure graceful degradation (workflow continues with original query)
- Malformed JSON fallback to plain-text brief
- `ResearchBriefOutput` parsing with missing optional fields
- Brief generation with clarification constraints

---

## Phase 4 — Config & Data Model Bugs

### 4.1 Fix `per_provider_rate_limits` default mismatch

**File:** `src/foundry_mcp/config/research.py`

**Problem:** The dataclass field default (line 199) sets `"semantic_scholar": 20` but `from_toml_dict()` fallback (line 379) sets `"semantic_scholar": 100`. The sample config also uses `100`. Different construction paths produce different rate limits.

**Fix:** Align both to `20` (matches the documented unauthenticated limit of ~20 req/min). Update `from_toml_dict()` line 379 and `samples/foundry-mcp.toml` line 650.

### 4.2 Add `deep_research_mode` validation

**File:** `src/foundry_mcp/config/research.py`

**Problem:** `deep_research_mode` (line 190) accepts `"general" | "academic" | "technical"` but no validator enforces valid values. Invalid modes are silently accepted.

**Fix:** Add a check in `__post_init__` or a new `_validate_deep_research_mode()`:
```python
_VALID_MODES = {"general", "academic", "technical"}
if self.deep_research_mode not in _VALID_MODES:
    raise ValueError(f"deep_research_mode must be one of {_VALID_MODES}, got '{self.deep_research_mode}'")
```

### 4.3 Add missing Gemini 2.0 model entries

**File:** `src/foundry_mcp/config/model_token_limits.json`

**Problem:** No entries for `gemini-2.0-flash` or `gemini-2.0-pro`. These fall through to no match. The code handles `None` gracefully but token budget estimation fails for these models.

**Fix:** Add `"gemini-2.0-pro": 1048576` and `"gemini-2.0-flash": 1048576` entries. Position before the `"gemini-2"` catch-all would be ideal but since sorting is by key length, just append them.

---

## Phase 5 — Cleanup: Code Quality

### 5.1 Rename `DeepResearchConfig` sub-config to avoid collision

**File:** `src/foundry_mcp/config/research_sub_configs.py`

**Problem:** `DeepResearchConfig` exists in both `research_sub_configs.py` (frozen dataclass, read-only view) and `core/research/models/deep_research.py` (Pydantic BaseModel, per-request config). Same name, different modules, different purposes.

**Fix:** Rename the sub-config to `DeepResearchSettings` in `research_sub_configs.py`. Update all references.

### 5.2 Isolate legacy supervision query-generation model

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Problem:** `_execute_supervision_query_generation_async` (~360 lines) is dead code for new workflows. Only reachable from a config path that new workflows never trigger. Inflates an already oversized file.

**Fix:** Move to `supervision_legacy.py` alongside the legacy prompt builders (`_build_supervision_system_prompt`, `_build_supervision_user_prompt`, `_parse_supervision_response`). Import conditionally from `supervision.py`.

### 5.3 Add explicit phase transition table

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

**Problem:** The phase skip logic on line 232 (`elif state.phase not in (SUPERVISION, SYNTHESIS): state.advance_phase()`) is a subtle implicit mechanism. The `_SKIP_PHASES` set in the model and the conditional advance in workflow execution work together in a way that requires reading both files to understand.

**Fix:** Add a comment block or constant defining the explicit phase transition:
```python
# Phase transition table (new workflows):
#   CLARIFICATION -> BRIEF -> SUPERVISION -> SYNTHESIS
# Legacy states may resume at GATHERING, which auto-advances to SUPERVISION.
```

### 5.4 Add `Protocol` class for mixin interface

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/_protocols.py` (new)

**Problem:** All mixin class annotations use `Any` for `config`, `memory`, etc. The `TYPE_CHECKING` protocol stubs help document the interface, but a proper `Protocol` class would catch interface drift at type-check time.

**Fix:** Define a `DeepResearchWorkflowProtocol` in a new `_protocols.py` file:
```python
class DeepResearchWorkflowProtocol(Protocol):
    config: ResearchConfig
    memory: Any  # ResearchMemory
    def _write_audit_event(self, state, event_name, *, data=None, level="info"): ...
    def _check_cancellation(self, state): ...
    async def _execute_provider_async(self, **kwargs) -> WorkflowResult: ...
```
Update mixin stubs to reference this protocol. This is nice-to-have and can be deferred.
