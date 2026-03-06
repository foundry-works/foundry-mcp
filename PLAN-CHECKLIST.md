# Plan Checklist: Deep Research Iteration Resilience

## Phase 1: Zero-source-yield short-circuit
- [x] Capture `iteration_source_count_before = len(state.sources)` at top of each iteration loop in `workflow_execution.py`
- [x] Add source-gain check after supervision completes, before synthesis begins
- [x] Guard: only short-circuit on iteration > 1 (first iteration always proceeds)
- [x] Write `iteration_short_circuit` audit event with reason and source counts
- [x] Call `_finalize_report` on previous iteration's report before completing
- [x] Set `state.metadata["completion_reason"] = "zero_source_yield"` (adapted: field stored in metadata, not a model attr)
- [x] Test: `test_zero_source_yield_short_circuits_iteration`
- [x] Test: `test_first_iteration_zero_sources_proceeds_to_synthesis`
- [x] Test: `test_zero_yield_still_finalizes_report`
- [x] Fix: Updated `test_cancellation_after_completed_iteration_finalizes_citations` to add source during supervision (prevents false short-circuit)

## Phase 2: Provider health tracking
- [x] Create `ProviderHealthTracker` class in `topic_research.py`
  - [x] `record_success(provider)` method
  - [x] `record_failure(provider, error_type)` method
  - [x] `is_degraded(provider)` — failure rate >= threshold over recent calls
  - [x] `all_degraded()` — all configured providers degraded
  - [x] `summary()` — structured dict for audit/confidence
- [x] Integrate `record_success` after successful search in `_topic_search_single`
- [x] Integrate `record_failure` in each `except` block in `_topic_search_single`
- [x] Wire tracker through `_execute_topic_research` (shared across sub-queries)
- [x] Store `state.metadata["_provider_health"]` after topic research completes
- [x] Include provider health in `iteration_short_circuit` audit event when all degraded
- [x] Add provider health to `build_confidence_context` in `_confidence_section.py`
- [x] Update confidence LLM prompt to mention provider issues when present
- [x] Test: `test_provider_health_tracker_records_success_and_failure`
- [x] Test: `test_provider_health_tracker_degraded_threshold`
- [x] Test: `test_provider_health_tracker_all_degraded`
- [x] Test: `test_provider_health_summary_format`
- [x] Test: `test_confidence_context_includes_provider_health`

## Phase 3: Wire `deepen_thin_sources`
- [x] Verify `deepen_thin_sources` signature in `_source_deepening.py` is correct post-7c03416
- [x] Resolve extract provider from provider registry in workflow execution context
- [x] Call `deepen_thin_sources` after `reverify_with_expanded_window` for deepen_extract claims
- [x] If sources deepened, re-verify those claims with expanded window
- [x] Recompute aggregate verification counts after deepen_extract
- [x] Add deepen_extract counts to existing source_deepening audit event
- [x] Gracefully skip when no extract provider is available
- [x] Test: `test_deepen_thin_sources_enriches_content`
- [x] Test: `test_deepen_thin_sources_followed_by_reverify_upgrades`
- [x] Test: `test_deepen_thin_sources_no_extract_provider_skipped`

## Phase 4: Fidelity regression rollback
- [x] Check if `state.iteration_reports` (per-iteration report snapshots) exists; add if needed
- [x] Store report snapshot after each synthesis completes (before verification modifies it)
- [x] Add `rollback_to_iteration` field to `decide_iteration` result when `delta < 0`
- [x] In workflow execution, when rollback indicated:
  - [x] Restore report from `iteration_reports[rollback_iteration]`
  - [x] Re-run claim verification on restored report (or restore previous verification results)
  - [x] Log `fidelity_regression_rollback` audit event
- [x] Proceed to `_finalize_report` with the restored (better) report
- [x] Test: `test_fidelity_regression_rolls_back_report`
- [x] Test: `test_fidelity_regression_rollback_audit_event`
- [x] Test: `test_regression_rollback_finalizes_restored_report`
