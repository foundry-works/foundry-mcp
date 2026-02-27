# Review Remediation Plan

Branch: `tyler/foundry-mcp-20260223-0747`
Source: `review.txt` — senior engineer review + extended security findings

---

## Phase 1: MUST-FIX (Critical)

### 1.1 Fix failing test — `_execute_analysis_async` missing

**File:** `tests/unit/test_core/research/test_deep_research_public_api.py` (lines 87-106)

The `test_workflow_inherits_all_phase_methods` test asserts `hasattr(workflow, "_execute_analysis_async")` but this method was removed during the refactoring. The analysis phase was folded into supervision.

**Action:** Remove `_execute_analysis_async` from the expected method list in the test since the method no longer exists in the workflow. Verify no other tests reference this method.

### 1.2 Sanitize all 8 prompt injection gaps

Apply `sanitize_external_content()` to every trust-boundary crossing identified by the security review. The sanitization framework already exists in `_helpers.py` — these files were created before the sanitization sweep.

| # | File | Line | Field | Fix |
|---|------|------|-------|-----|
| A | `_analysis_prompts.py` | 103 | `state.original_query` | Wrap with `sanitize_external_content()` |
| B | `_analysis_prompts.py` | 106 | `state.research_brief` | Wrap with `sanitize_external_content()` |
| C | `_analysis_prompts.py` | 141 | `source.title` | Wrap with `sanitize_external_content()` |
| D | `_analysis_prompts.py` | 215 | `source.title` | Wrap with `sanitize_external_content()` |
| E | `_analysis_prompts.py` | 217 | `source.url` | Wrap with `sanitize_external_content()` |
| F | `refinement.py` | 338 | `state.original_query` | Wrap with `sanitize_external_content()` |
| G | `synthesis.py` | 905 | `source.url` | Wrap with `sanitize_external_content()` |
| H | `synthesis.py` | 942 | `source.url` | Wrap with `sanitize_external_content()` |
| I | `compression.py` | 185 | `src.url` | Wrap with `sanitize_external_content()` |

For state fields (A, B, F): prefer `build_sanitized_context(state)` where the function already provides a pre-sanitized dict, then use the sanitized values from that dict.

For source fields (C, D, E, G, H, I): wrap each interpolation with `sanitize_external_content(value)`.

Ensure `sanitize_external_content` and/or `build_sanitized_context` are imported in each file.

---

## Phase 2: SHOULD-FIX (Important)

### 2.1 Reduce method complexity (Issue #3)

Extract sub-operations from the four oversized methods. Target: no method > 150 lines.

| File | Method | Lines | Extraction targets |
|------|--------|-------|--------------------|
| `supervision.py` | `_first_round_decompose_critique_revise()` | ~270 | Extract: parse/retry logic, directive validation, critique handling |
| `supervision.py` | `_execute_supervision_delegation_async()` | ~215 | Extract: think+delegate orchestration, gap analysis construction |
| `topic_research.py` | `_execute_topic_research_async()` | ~500+ | Extract: tool dispatch, findings truncation, result merging, retry loops |
| `synthesis.py` | `_execute_synthesis_async()` | ~400+ | Extract: source block construction, report extraction, citation formatting |

**Approach:** Create private helper methods within the same class/module. Don't create new files — keep extractions colocated. Preserve all existing behavior and error handling.

### 2.2 Split `_helpers.py` into focused modules (Issue #4)

Split the 987-line `_helpers.py` into domain-focused modules. The file already has clear sections:

| New module | Source lines (approx) | Contents |
|-----------|----------------------|----------|
| `_json_parsing.py` | 28-77 | `extract_json()` |
| `_token_budget.py` | 80-371 | `fidelity_level_from_score()`, `truncate_at_boundary()`, `truncate_to_token_estimate()`, `_split_prompt_sections()`, `structured_truncate_blocks()`, `structured_drop_sources()` |
| `_model_resolution.py` | 373-611 | `estimate_token_limit_for_model()`, `TopicReflectionDecision`, `parse_reflection_decision()`, `ClarificationDecision`, `parse_clarification_decision()`, `safe_resolve_model_for_role()`, `resolve_phase_provider()` |
| `_content_dedup.py` | 614-786 | `_char_ngrams()`, `_normalize_content_for_dedup()`, `content_similarity()`, `NoveltyTag`, `compute_novelty_tag()` |
| `_injection_protection.py` | 788-987 | `validate_extract_url()`, `sanitize_external_content()`, `build_sanitized_context()`, `build_novelty_summary()`, SSRF constants |

**Approach:**
1. Create new modules under `phases/` (or a `_utils/` subpackage).
2. Re-export everything from `_helpers.py` for backward compatibility.
3. Update direct imports in other files to point to the new modules.
4. Keep `_helpers.py` as a thin re-export shim initially, remove in a follow-up.

### 2.3 Config bounds validation (Issue #5)

Add bounds checking in `research.py` config validation for fields that currently lack it, following the existing `max_supervision_rounds` pattern (define ClassVar constant → warn + clamp upper → raise on lower).

| Field | Lower bound | Upper bound | Notes |
|-------|-------------|-------------|-------|
| `deep_research_max_iterations` | >= 1 | <= 20 | Iteration count |
| `deep_research_max_sub_queries` | >= 1 | <= 50 | Sub-query count |
| `deep_research_max_sources` | >= 1 | <= 100 | Source limit per query |
| `deep_research_max_concurrent` | >= 1 | <= 20 | Concurrency limit |
| `default_timeout` | >= 1 | <= 3600 | Seconds |
| `token_safety_margin` | >= 0 | <= 50000 | Token count |

Add a new validation method `_validate_deep_research_bounds()` called from the existing validation chain.

### 2.4 Coverage heuristic improvement (Issue #6)

**File:** `supervision_coverage.py` — `assess_coverage_heuristic()`

**Problem:** Source adequacy uses `mean()` averaging, so a query with 10x minimum sources compensates for one with 0 sources, allowing premature exit.

**Fix:** Replace `mean()` with `min()` for the source adequacy dimension. This ensures coverage is only declared sufficient when *every* sub-query meets its minimum source threshold. The `min()` approach is the most conservative and prevents lopsided coverage.

Alternative: Use a geometric mean or `median()` if `min()` proves too strict in practice (can be adjusted later).

### 2.5 Model token limits validation (Issue #7)

**File:** `_helpers.py` (or new `_model_resolution.py`) — `estimate_token_limit_for_model()` loader

**Problem:** Accepts any integer from JSON, including 0 or negative. A typo like `200` instead of `200000` would silently break context window management.

**Fix:** After loading `model_token_limits.json`, validate every value is `>= 1000`. Log a warning for any entry below the threshold and skip it (don't include in lookup). This catches obvious typos without requiring exact values.

---

## Phase 3: NICE-TO-HAVE (Polish)

### 3.1 Reduce prompt duplication (Issue #8)

**File:** `supervision_prompts.py`

~40% of text is repeated between `build_delegation_system_prompt()` (lines 229-271) and `build_first_round_delegation_system_prompt()` (lines 493-529).

**Fix:** Extract shared prompt sections into a `_build_delegation_core_prompt()` helper. Have both functions call it and append their specific sections (gap-analysis guidance vs. decomposition guidance).

### 3.2 Centralize supervision round increment (Issue #9)

**File:** `supervision.py`

`state.supervision_round += 1` appears at 4 locations (lines 286, 312, 330, 687).

**Fix:** Extract to a `_advance_supervision_round(state)` method that increments the counter and emits any associated audit event. Replace all 4 call sites. This makes the round lifecycle explicit and reduces risk of missed updates.

### 3.3 Consolidate `_CHARS_PER_TOKEN` constant (Issue #10)

Currently defined independently in:
- `_lifecycle.py:149`
- `topic_research.py:435`
- Also used inline in `_helpers.py` `truncate_to_token_estimate()`

**Fix:** Define `CHARS_PER_TOKEN = 4` once in `_helpers.py` (or the new `_token_budget.py`). Import from there in all three files.

### 3.4 Fix RuntimeWarning in test (Issue #11)

**File:** `tests/core/research/workflows/deep_research/test_workflow_execution.py` (lines 298-320)

`test_cancellation_after_brief_before_supervision` produces:
```
RuntimeWarning: coroutine 'StubWorkflow._execute_supervision_async' was never awaited
```

**Fix:** Ensure the stub coroutine is properly awaited or closed in the test teardown. Likely need to either:
- Add `asyncio.get_event_loop().run_until_complete(coro)` to consume it, or
- Mock the method to return a regular value instead of a coroutine, or
- Use `coro.close()` in cleanup to suppress the warning.

---

## Execution Order

1. **Phase 1** first — failing test and security fixes are blockers.
2. **Phase 2.2** (split `_helpers.py`) before **2.1** (method complexity) — the split creates cleaner import targets for extracted helpers.
3. **Phase 2.3-2.5** are independent, can be done in parallel.
4. **Phase 3** items are independent, can be done in any order after Phase 2.
5. **Phase 3.3** (`_CHARS_PER_TOKEN`) should happen after **2.2** (split) since the constant's home file changes.

## Testing

Run the full test suite after each phase:
```bash
python -m pytest tests/ -x -q
```

After Phase 1.2 (sanitization), verify with:
```bash
python -m pytest tests/ -k "sanitiz" -v
```

After Phase 2.2 (split), verify all imports resolve:
```bash
python -c "from foundry_mcp.core.research.workflows.deep_research._helpers import *"
```
