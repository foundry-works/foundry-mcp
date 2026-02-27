# PLAN-CHECKLIST: Post-Review Fixes for Deep Research Pipeline

## Phase 1 — Critical: Timeout Budget & Execution Model
- [x] 1.1 Restore `background=True` in `handlers_deep_research.py`
- [x] 1.2 Fix timeout budget inversion (raise `deep_research_timeout` or lower per-phase timeouts; add cross-validation)
- [x] 1.3 Add aggregate timeout guard to planning phase (wall-clock check between 3 sequential LLM calls)
- [x] 1.4 Deprecate or wire `max_iterations` (emit warning or implement iteration loop)

## Phase 2 — Critical: Bounded State Growth
- [x] 2.1 Cap `state.sources` with content eviction (e.g., 500 sources max)
- [x] 2.2 Clear `TopicResearchResult.message_history` after compression succeeds
- [x] 2.3 Cap `topic_research_results` list (e.g., 50 max)
- [x] 2.4 Call `cleanup_stale_tasks()` at start of `_start_background_task`
- [x] 2.5 Cap `supervision_messages` entry count (e.g., 100 max)
- [x] 2.6 Bound `_active_research_sessions` dict (max sessions or periodic cleanup)

## Phase 3 — Security: SSRF, Sanitization, Injection
- [ ] 3.1 Unify SSRF validators — add `0.0.0.0/8`, `ff00::/8`, `.local`/`.internal`/`.localhost` to `_injection_protection.py`
- [ ] 3.2 Fix double HTML entity encoding bypass (loop `html.unescape()` until stable)
- [ ] 3.3 Sanitize `report` and `query` in evaluation prompt (`evaluator.py`)
- [ ] 3.4 Expand zero-width character stripping (`U+00AD`, `U+034F`, `U+2060-2064`, `U+180E`)
- [ ] 3.5 Fix `\b` word boundary to catch underscore-extended tags (`<system_prompt>` etc.)

## Phase 4 — Correctness: Orchestration & Config Validation
- [ ] 4.1 Validate `coverage_confidence_weights` schema (numeric values, known keys)
- [ ] 4.2 Add supervision quality gate in `evaluate_phase_completion`
- [ ] 4.3 Fix cancellation rollback — check `rollback_note` on resume or clean partial data
- [ ] 4.4 Handle overlapping redundancy indices in critique merge logic
- [ ] 4.5 Fix gap priority parsing — wrap in `try/except` with default
- [ ] 4.6 Add `deep_research_timeout` to timeout validation list
- [ ] 4.7 Add PLANNING to `PHASE_TO_AGENT` mapping (or document omission)

## Phase 5 — Code Quality & Observability
- [ ] 5.1 Add debug logging for `resolve_phase_provider` fallthrough
- [ ] 5.2 Inject `max_sub_queries` into critique system prompt
- [ ] 5.3 Add warning when catch-all `elif` fires in `advance_phase`
- [ ] 5.4 Update status handler polling guidance (coordinate with Phase 1.1)
- [ ] 5.5 Build compression history incrementally to avoid memory spikes
