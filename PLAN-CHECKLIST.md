# Plan Checklist: Deep Research Iteration Resilience

## Phase 1: Zero-source-yield short-circuit
- [ ] Capture `iteration_source_count_before = len(state.sources)` at top of each iteration loop in `workflow_execution.py`
- [ ] Add source-gain check after supervision completes, before synthesis begins
- [ ] Guard: only short-circuit on iteration > 1 (first iteration always proceeds)
- [ ] Write `iteration_short_circuit` audit event with reason and source counts
- [ ] Call `_finalize_report` on previous iteration's report before completing
- [ ] Set `state.completion_reason = "zero_source_yield"`
- [ ] Test: `test_zero_source_yield_short_circuits_iteration`
- [ ] Test: `test_first_iteration_zero_sources_proceeds_to_synthesis`
- [ ] Test: `test_zero_yield_still_finalizes_report`

## Phase 2: Provider health tracking
- [ ] Create `ProviderHealthTracker` class in `topic_research.py`
  - [ ] `record_success(provider)` method
  - [ ] `record_failure(provider, error_type)` method
  - [ ] `is_degraded(provider)` — failure rate >= threshold over recent calls
  - [ ] `all_degraded()` — all configured providers degraded
  - [ ] `summary()` — structured dict for audit/confidence
- [ ] Integrate `record_success` after successful search in `_topic_search_single`
- [ ] Integrate `record_failure` in each `except` block in `_topic_search_single`
- [ ] Wire tracker through `_execute_topic_research` (shared across sub-queries)
- [ ] Store `state.metadata["_provider_health"]` after topic research completes
- [ ] Include provider health in `iteration_short_circuit` audit event when all degraded
- [ ] Add provider health to `build_confidence_context` in `_confidence_section.py`
- [ ] Update confidence LLM prompt to mention provider issues when present
- [ ] Test: `test_provider_health_tracker_records_success_and_failure`
- [ ] Test: `test_provider_health_tracker_degraded_threshold`
- [ ] Test: `test_provider_health_tracker_all_degraded`
- [ ] Test: `test_provider_health_summary_format`
- [ ] Test: `test_confidence_context_includes_provider_health`

## Phase 3: Wire `deepen_thin_sources`
- [ ] Verify `deepen_thin_sources` signature in `_source_deepening.py` is correct post-7c03416
- [ ] Resolve extract provider from provider registry in workflow execution context
- [ ] Call `deepen_thin_sources` after `reverify_with_expanded_window` for deepen_extract claims
- [ ] If sources deepened, re-verify those claims with expanded window
- [ ] Recompute aggregate verification counts after deepen_extract
- [ ] Add deepen_extract counts to existing source_deepening audit event
- [ ] Gracefully skip when no extract provider is available
- [ ] Test: `test_deepen_thin_sources_enriches_content`
- [ ] Test: `test_deepen_thin_sources_followed_by_reverify_upgrades`
- [ ] Test: `test_deepen_thin_sources_no_extract_provider_skipped`

## Phase 4: Fidelity regression rollback
- [ ] Check if `state.iteration_reports` (per-iteration report snapshots) exists; add if needed
- [ ] Store report snapshot after each synthesis completes (before verification modifies it)
- [ ] Add `rollback_to_iteration` field to `decide_iteration` result when `delta < 0`
- [ ] In workflow execution, when rollback indicated:
  - [ ] Restore report from `iteration_reports[rollback_iteration]`
  - [ ] Re-run claim verification on restored report (or restore previous verification results)
  - [ ] Log `fidelity_regression_rollback` audit event
- [ ] Proceed to `_finalize_report` with the restored (better) report
- [ ] Test: `test_fidelity_regression_rolls_back_report`
- [ ] Test: `test_fidelity_regression_rollback_audit_event`
- [ ] Test: `test_regression_rollback_finalizes_restored_report`
