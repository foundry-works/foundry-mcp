# Plan: Deep Research Iteration Resilience

## Problem Statement

Deep research sessions waste full iteration cycles when the search provider stops returning results. Observed in session `deepres-6d696c22d090`:

- **Iteration 1:** 16 sources found, fidelity 0.379
- **Iteration 2:** 0 sources found (all 4 directives failed), fidelity 0.658 (improved via source deepening only)
- **Iteration 3:** 0 sources found (all 3 directives failed), fidelity 0.470 (regressed)

Root causes identified:
1. **No zero-yield short-circuit** — if an iteration's topic research phase adds zero new sources, the system still proceeds through full synthesis → verification → fidelity scoring before discovering nothing changed.
2. **No provider health tracking** — search provider errors and empty results are silently swallowed. The system cannot distinguish "nothing relevant exists" from "provider is broken/rate-limited."
3. **`deepen_thin_sources` is dead code** — classified into the `deepen_extract` bucket but never called.
4. **Regression early-stop doesn't rollback** — fidelity regression triggers COMPLETED but keeps the regressed report instead of rolling back to the better previous-iteration version.

## Changes

### Phase 1: Zero-source-yield short-circuit

**Goal:** When an iteration's supervision + topic research phase produces zero new sources, skip synthesis and stop iterating immediately. The previous iteration's report is already the best we can produce.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

After supervision completes and before synthesis begins (~line 528 area), add a source-gain check:

```python
sources_before = iteration_source_count_before  # captured before supervision
sources_after = len(state.sources)
sources_gained = sources_after - sources_before

if state.iteration > 1 and sources_gained == 0:
    # No new sources found — re-synthesizing won't improve the report
    self._write_audit_event(
        state,
        "iteration_short_circuit",
        data={
            "iteration": state.iteration,
            "reason": "zero_source_yield",
            "sources_total": sources_after,
        },
    )
    # Keep previous iteration's report, finalize and complete
    await self._finalize_report(state, trigger="zero_yield_short_circuit")
    state.status = "completed"
    state.completion_reason = "zero_source_yield"
    break
```

Capture `iteration_source_count_before = len(state.sources)` at the top of each iteration (before supervision starts). Only applies to iteration > 1 — the first iteration should always proceed to synthesis even with zero sources (it falls back to ungrounded synthesis with a disclaimer).

#### Tests: `tests/core/research/workflows/deep_research/test_workflow_execution.py`

- `test_zero_source_yield_short_circuits_iteration` — mock supervision to add 0 sources on iteration 2; verify synthesis is skipped, previous report preserved, audit event logged.
- `test_first_iteration_zero_sources_proceeds_to_synthesis` — verify iteration 1 with 0 sources does NOT short-circuit (preserves existing ungrounded synthesis behavior).
- `test_zero_yield_still_finalizes_report` — verify citations and confidence section are appended despite short-circuit.

### Phase 2: Provider health tracking and degraded-mode awareness

**Goal:** Track per-provider failure rates during a session. When a provider's recent failure rate exceeds a threshold, log a degraded-mode audit event. Surface provider health in the confidence section context.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

Add a `ProviderHealthTracker` class (lightweight, session-scoped):

```python
class ProviderHealthTracker:
    """Track per-provider success/failure rates within a research session."""

    def __init__(self, *, degraded_threshold: float = 0.8):
        self._stats: dict[str, ProviderStats] = {}
        self._degraded_threshold = degraded_threshold

    def record_success(self, provider: str) -> None: ...
    def record_failure(self, provider: str, error_type: str) -> None: ...
    def is_degraded(self, provider: str) -> bool:
        """True if failure_rate >= degraded_threshold over last N calls."""
    def all_degraded(self) -> bool:
        """True if ALL configured providers are degraded."""
    def summary(self) -> dict[str, Any]:
        """Structured summary for audit/confidence section."""
```

Integrate into `_topic_search_single` (lines 2659-2780):
- Call `record_success` after successful search (line 2755)
- Call `record_failure` in each `except` block with the error type

Wire the tracker through `_execute_topic_research` so it's shared across all sub-queries within an iteration.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

After supervision + topic research completes:
- Check `provider_tracker.all_degraded()`
- If all providers degraded AND zero new sources: include `"provider_health": "all_degraded"` in the `iteration_short_circuit` audit event
- Store provider health summary in `state.metadata["_provider_health"]` for the confidence section

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py`

In `build_confidence_context`, include provider health data if available:
```python
if "_provider_health" in state.metadata:
    context["provider_health"] = state.metadata["_provider_health"]
```

Update the confidence LLM prompt to mention provider issues when present (e.g., "Search provider availability was limited during later research cycles, which may have constrained source coverage").

#### Tests

In a new section of `test_topic_research.py`:
- `test_provider_health_tracker_records_success_and_failure`
- `test_provider_health_tracker_degraded_threshold`
- `test_provider_health_tracker_all_degraded`
- `test_provider_health_summary_format`

In `test_confidence_section.py`:
- `test_confidence_context_includes_provider_health` — verify provider health data flows into confidence context

### Phase 3: Wire `deepen_thin_sources`

**Goal:** The `deepen_thin_sources` function in `_source_deepening.py` is complete but never called. Wire it into the workflow after `deepen_window` claims are processed.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

After the `reverify_with_expanded_window` call for deepen_window claims (~line 760 area), add:

```python
if classification.deepen_extract and extract_provider:
    deepened_count = await deepen_thin_sources(
        classification.deepen_extract,
        citation_map,
        state,
        extract_provider,
    )
    if deepened_count > 0:
        # Re-verify the deepened claims with the enriched content
        reverified = await reverify_with_expanded_window(
            classification.deepen_extract,
            citation_map,
            llm_call_fn,
            max_chars=24000,
        )
        # Update aggregate counts
        ...
```

Need to resolve the extract provider — currently the Tavily extract provider is available through the provider registry. Pass it through the workflow execution context.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py`

The `deepen_thin_sources` function (lines 272-346) needs a minor fix: the `_state` parameter was removed per review issue in 7c03416, but the function still needs access to the source list. Verify the current signature is correct and the function works with the citation_map approach.

#### Tests: `tests/core/research/workflows/deep_research/test_source_deepening.py`

- `test_deepen_thin_sources_enriches_content` — mock Tavily extract returning richer content; verify raw_content updated
- `test_deepen_thin_sources_followed_by_reverify_upgrades` — deepened source + re-verification upgrades verdict
- `test_deepen_thin_sources_no_extract_provider_skipped` — gracefully skipped when no extract provider available

### Phase 4: Fidelity regression rollback

**Goal:** When fidelity regresses between iterations, roll back to the previous iteration's report rather than keeping the worse one.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`

In `decide_iteration` (line ~380), when regression is detected (`delta < 0`):

Add a `"rollback_to_iteration"` field to the decision result:

```python
if delta < 0:
    rationale = f"Completing: fidelity regressed by {abs(delta):.3f} ..."
    decision["rollback_to_iteration"] = state.iteration - 1
```

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

When `decide_iteration` returns `rollback_to_iteration`:
- Restore the previous iteration's report from `state.iteration_reports[rollback_to_iteration]`
- Restore the previous iteration's claim verification results
- Log audit event `"fidelity_regression_rollback"`
- Proceed to finalize with the restored (better) report

This requires storing per-iteration report snapshots — verify `state.iteration_reports` exists or add it.

#### Tests: `tests/core/research/workflows/deep_research/test_workflow_execution.py`

- `test_fidelity_regression_rolls_back_report` — fidelity drops from 0.65 to 0.47; verify previous report restored
- `test_fidelity_regression_rollback_audit_event` — verify audit event logged with iteration details
- `test_regression_rollback_finalizes_restored_report` — verify citations/confidence run on the restored report

## Files Modified

| File | Change |
|------|--------|
| `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` | Zero-yield short-circuit, provider health wiring, deepen_extract wiring, regression rollback |
| `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py` | Rollback field on regression decision |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` | `ProviderHealthTracker` class, integrate into search calls |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py` | Provider health in confidence context + prompt |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py` | Minor: verify `deepen_thin_sources` signature |
| `tests/core/research/workflows/deep_research/test_workflow_execution.py` | 9 new tests (Phases 1, 3, 4) |
| `tests/core/research/workflows/deep_research/test_topic_research.py` | 4 new tests (Phase 2) |
| `tests/core/research/workflows/deep_research/test_confidence_section.py` | 1 new test (Phase 2) |
| `tests/core/research/workflows/deep_research/test_source_deepening.py` | 3 new tests (Phase 3) |

## Out of Scope

- Multi-provider fallback/rotation (adding Perplexity/Google as active providers) — config-level change, not code
- Provider-level retry with backoff — the tracker observes but doesn't retry; adding retry is a separate concern
- Changing the fidelity threshold (0.70) or min_improvement (0.10) defaults — tuning, not code
- Per-iteration report versioning if `iteration_reports` doesn't already exist — scoped as needed in Phase 4
