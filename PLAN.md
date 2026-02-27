# PLAN: Address Senior Review Findings — Correctness, Security & Robustness

> **Goal**: Fix all critical and major issues identified during senior-engineer review of the deep research workflow branch.
>
> **Scope**: 5 phases, estimated ~400-600 LOC changes + ~150 LOC test additions/fixes
>
> **Risk**: Medium. Phase 1 (critical) fixes a confirmed token double-counting bug and a silent data-loss path. Phase 2 (security) addresses an SSRF gap. Phases 3-5 are robustness and cleanup.

---

## Phase 1: Critical Fixes

**Objective**: Eliminate the token double-counting bug and the silent compression failure path.

### 1a. Fix token double-counting in parallel compression (CRITICAL)

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_lifecycle.py:954`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py:1178-1179`

**Problem**: `execute_llm_call()` at `_lifecycle.py:954` adds `result.tokens_used` to `state.total_tokens_used`. Then `_apply_inline_compression_async` in `topic_research.py:1178-1179` adds the *same* compression tokens again under the `state_lock`. Every successful inline compression inflates token counts by 2x.

**Fix** (option A — preferred): Remove the token addition from `execute_llm_call()` for the compression path. The cleanest approach is:
1. Add an optional `skip_token_tracking: bool = False` parameter to `execute_llm_call()`.
2. When called from the compression path (`_compress_single_topic_async`), pass `skip_token_tracking=True`.
3. The caller in `topic_research.py` already handles token tracking under the lock — let it be the sole owner.

**Fix** (option B — alternative): Remove the manual token tracking from `_apply_inline_compression_async` and rely on `execute_llm_call`'s tracking. This is simpler but loses the lock-protected update discipline.

**Preferred**: Option A, because lock-protected updates are the correct pattern for parallel contexts.

**Test**: Add a test that runs parallel compression on 3 topics, asserts `state.total_tokens_used` equals the sum of individual compression calls (not 2x).

### 1b. Add logging on empty compression LLM response (CRITICAL)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py:700-719`

**Problem**: When `_compress_single_topic_async` gets a successful LLM call but `result.content` is empty or `result.success is False`, it returns `(0, 0, False)` with no log message and no audit event. Silent data loss.

**Fix**:
1. Before the `return (0, 0, False)` at the end of the method, add:
   ```python
   logger.warning(
       "Compression produced empty/failed result for topic %r (research %s)",
       topic_label, state.id,
   )
   self._write_audit_event(state, "compression_empty_result", data={
       "topic": topic_label,
       "success": result.success if result else False,
       "had_content": bool(result.content) if result else False,
   }, level="warning")
   ```
2. This gives operators per-topic visibility into compression failures.

**Test**: Mock an LLM call that returns `success=True, content=""`, verify warning is logged and audit event is written.

---

## Phase 2: Security Hardening

**Objective**: Close the DNS rebinding SSRF gap and add defense-in-depth sanitization.

### 2a. Enable DNS resolution for extract URL validation (MAJOR)

**Files**:
- `src/foundry_mcp/core/research/workflows/deep_research/_injection_protection.py:38`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py:1647`

**Problem**: `validate_extract_url()` defaults `resolve_dns=False`. The sole call site in `_handle_extract_tool` doesn't override this. A malicious hostname resolving to `169.254.169.254` (cloud metadata) bypasses all checks.

**Fix**:
1. Change the call site at `topic_research.py:1647` to pass `resolve_dns=True`:
   ```python
   urls = [u for u in urls if validate_extract_url(u, resolve_dns=True)]
   ```
2. Keep the default as `False` for backward compatibility, but the security-critical path now uses the stronger check.

**Test**: Add a test that mocks `socket.getaddrinfo` to return `127.0.0.1` for a hostname, verify `validate_extract_url(url, resolve_dns=True)` rejects it.

### 2b. Sanitize `supervisor_summary` and `per_topic_summary` at construction (MAJOR)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py:1351-1356, 1465-1474`

**Problem**: `per_topic_summary` and `supervisor_summary` are interpolated into evidence inventories without `sanitize_external_content()`. Mitigated by downstream sanitization in `render_supervision_conversation_history`, but inconsistent with the project's sanitize-at-source pattern.

**Fix**:
1. At line 1351, wrap: `summary = sanitize_external_content(topic_result.per_topic_summary)`
2. At line 1467, wrap: `brief = sanitize_external_content(topic_result.supervisor_summary)`
3. Import `sanitize_external_content` if not already imported in the file.

**Test**: Existing `test_sanitize_external_content.py` covers the sanitization function itself. Add a focused test that constructs a `TopicResearchResult` with injection tags in `per_topic_summary`, calls `_build_directive_fallback_summary`, and verifies tags are stripped.

---

## Phase 3: Token Budget & Truncation Robustness

**Objective**: Fix token math edge cases that cause silent content erasure or context-window overflows.

### 3a. Guard against zero/negative token budgets in truncation (MAJOR)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_token_budget.py:60-77`

**Problem**: `truncate_to_token_estimate()` with `max_tokens <= 0` silently returns a truncation marker with no content. Upstream budget underflows become invisible.

**Fix**:
1. Add a guard at the top of `truncate_to_token_estimate`:
   ```python
   if max_tokens <= 0:
       logger.warning(
           "truncate_to_token_estimate called with max_tokens=%d; "
           "returning full text (budget exhausted upstream)",
           max_tokens,
       )
       return text  # Return full text rather than silently erasing
   ```
2. This makes budget underflows visible via logging while preserving data (let the LLM call fail with a clear token-limit error rather than silently eating content).

**Test**: Call `truncate_to_token_estimate("some text", 0)` and `truncate_to_token_estimate("some text", -5)`, verify full text is returned and warning is logged.

### 3b. Account for `token_safety_margin` in supplementary notes injection (MAJOR)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py:1179-1190`

**Problem**: `_inject_supplementary_raw_notes` uses raw `context_window` for budget math without subtracting the configured `token_safety_margin` (default 15%). Final prompt can consume 100% of context window.

**Fix**:
1. Apply the safety margin to the budget calculation:
   ```python
   safety_margin = getattr(self.config, 'deep_research_token_safety_margin', 0.15)
   effective_context = int(context_window * (1.0 - safety_margin))
   headroom_tokens = effective_context - current_tokens - SYNTHESIS_OUTPUT_RESERVED
   ```
2. This ensures the prompt stays within the safe token budget.

**Test**: Mock a scenario where raw `context_window` headroom is 20k tokens but effective headroom (after 15% margin) is 1.2k — verify supplementary injection is skipped.

### 3c. Fix model token limit substring matching order (MAJOR)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_model_resolution.py:21-41`

**Problem**: `estimate_token_limit_for_model()` does first-match substring search. `"gpt-4"` matches `"gpt-4o-2024-08-06"` before `"gpt-4o"` depending on dict insertion order.

**Fix**:
1. Sort matches by descending key length before returning:
   ```python
   matches = [(k, v) for k, v in token_limits.items() if k.lower() in model_lower]
   if matches:
       # Longest match is most specific
       matches.sort(key=lambda kv: len(kv[0]), reverse=True)
       return matches[0][1]
   ```
2. This ensures `"gpt-4o"` matches before `"gpt-4"` for `"gpt-4o-2024-08-06"`.

**Test**: Build a token_limits dict with both `"gpt-4": 8192` and `"gpt-4o": 128000`, verify `estimate_token_limit_for_model("gpt-4o-2024-08-06", limits)` returns 128000.

---

## Phase 4: Configuration Validation & Consistency

**Objective**: Close validation gaps in research config that allow invalid values to propagate silently.

### 4a. Validate phase-specific timeouts and thresholds (MAJOR)

**File**: `src/foundry_mcp/config/research.py` — `_validate_deep_research_bounds()`

**Problem**: Phase-specific timeouts (`planning_timeout`, `synthesis_timeout`, `supervision_wall_clock_timeout`, `reflection_timeout`, `evaluation_timeout`, `digest_timeout`, `summarization_timeout`, `retry_delay`), `content_dedup_threshold`, and `compression_max_content_length` have no lower-bound validation. Negative values propagate silently.

**Fix**: Add to `_validate_deep_research_bounds()`:
```python
# Validate phase-specific timeouts (must be positive)
timeout_fields = [
    "deep_research_planning_timeout",
    "deep_research_synthesis_timeout",
    "deep_research_supervision_wall_clock_timeout",
    "deep_research_reflection_timeout",
    "deep_research_evaluation_timeout",
    "deep_research_digest_timeout",
    "deep_research_summarization_timeout",
    "deep_research_retry_delay",
]
for field_name in timeout_fields:
    val = getattr(self, field_name)
    if val is not None and val <= 0:
        setattr(self, field_name, getattr(ResearchConfig, field_name))
        logger.warning("%s=%s invalid (must be > 0), reset to default", field_name, val)

# Validate threshold range
if not (0.0 <= self.deep_research_content_dedup_threshold <= 1.0):
    self.deep_research_content_dedup_threshold = 0.85
    logger.warning("content_dedup_threshold out of [0,1] range, reset to 0.85")

# Validate positive content length
if self.deep_research_compression_max_content_length <= 0:
    self.deep_research_compression_max_content_length = 150000
    logger.warning("compression_max_content_length must be > 0, reset to 150000")
```

**Test**: Construct `ResearchConfig` with negative timeouts, verify they're clamped to defaults with warnings.

### 4b. Fix `structured_truncate_blocks` single-pass insufficiency (MAJOR)

**File**: `src/foundry_mcp/core/research/workflows/deep_research/_token_budget.py:177`

**Problem**: `cut = min(excess, orig_len // 2)` caps each section cut at 50%. If the prompt is >2x over budget with one large section, a single pass leaves the prompt still over budget, wasting an LLM round-trip.

**Fix**: Wrap the truncation loop in an outer pass loop (max 3 passes) that recomputes excess after each pass:
```python
for _pass in range(3):
    excess = total - max_chars
    if excess <= 0:
        break
    for section in sorted(sections, key=lambda s: len(s), reverse=True):
        # existing cut logic
        ...
```

**Test**: Create a prompt 3x over budget with one large section, verify it fits within budget after `structured_truncate_blocks`.

---

## Phase 5: Test Quality & Dead Code Cleanup

**Objective**: Remove bogus tests, dead model classes, and add missing edge-case coverage.

### 5a. Delete bogus wall-clock timeout tests (TEST QUALITY)

**File**: `tests/core/research/workflows/deep_research/test_sanitize_external_content.py:874-978`

**Problem**: `TestSupervisionWallClockTimeout` (4 test methods) re-implements production logic inline instead of calling actual code. Tests `dict.__setitem__`, not production behavior. The same timeout is properly tested in `test_supervision.py:4488-4536`.

**Fix**: Delete lines 874-978 (the entire `TestSupervisionWallClockTimeout` class).

### 5b. Remove or test dead `Contradiction` model (DEAD CODE)

**File**: `src/foundry_mcp/core/research/models/deep_research.py:609`

**Problem**: `Contradiction` model class exists but all tests for it were deleted (along with `_detect_contradictions`). No remaining production code uses it.

**Fix** (preferred): Delete the `Contradiction` class and any `contradictions` field on `DeepResearchState` if it exists. If deserialization of old states is a concern, keep the field as `Optional[list] = None` with a deprecation comment.

### 5c. Add missing test for compression hard-truncation mid-message (COVERAGE)

**File**: `tests/core/research/workflows/deep_research/test_compression_retry_truncation.py` (or new test)

**Problem**: The `history_block[-max_content_length:]` fallback in `compression.py:95` that slices mid-message is untested.

**Fix**: Add a test that constructs a message history with clear `[Assistant]` / `[Tool: ...]` markers, applies compression with a very small `max_content_length`, and verifies the truncation behavior. This documents the known limitation.

---

## Scope Boundary — Deferred Items

The following issues from the review are acknowledged but **not addressed in this plan** due to lower severity or higher refactoring cost:

1. **`from_toml_dict` default duplication** (M4): Correct fix requires refactoring the entire 230-line factory method to use `dataclasses.fields()` introspection. Deferred to a dedicated config refactor.

2. **Sub-config validation bypass** (M6): Adding validators to frozen dataclasses requires either `__post_init__` on each sub-config or a factory pattern. Deferred until the config refactor.

3. **Unsynchronized `execute_llm_call` state mutation** (M2): In CPython's single-threaded event loop, this is not a corruption risk. The lock discipline inconsistency is a documentation/maintenance issue. Deferred with a TODO comment.

4. **`_classify_query_type` regex misclassification** (synthesis M5): Low impact — affects structural hints in synthesis prompts but not correctness. Deferred.

5. **Cancellation rollback leaves accumulated data** (concurrency M4): Acknowledged design trade-off with existing documentation. URL dedup covers the common case.
