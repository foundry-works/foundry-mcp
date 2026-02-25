# PLAN CHECKLIST: Deep Research Workflow — ODR Alignment & Efficiency Improvements

## Phase 1 — Per-Result Summarization at Search Time
- [x] 1.1 Add `_summarize_search_result` async helper to `TopicResearchMixin`
- [x] 1.2 Wire summarization into `_handle_web_search_tool` after source addition, with `asyncio.gather` and per-result timeout (30s)
- [x] 1.3 Store summary in `src.content`, set `src.metadata["summarized"] = True`, preserve raw in `src.metadata["raw_content"]`
- [x] 1.4 Fall back to raw snippet/truncation on summarization timeout or failure
- [x] 1.5 Add `summarization_model` config field if not present (or reuse compression model)
- [x] 1.6 Test: verify researcher message history contains `SUMMARY:` blocks after search
- [x] 1.7 Test: verify fallback to raw content on summarization timeout
- [x] 1.8 Test: verify raw content preserved in metadata for compression phase

## Phase 2 — Inline Compression of Supervision Directive Results
- [x] 2.1 After `_execute_directives_async`, invoke `_compress_single_topic_async` for results without `compressed_findings`
- [x] 2.2 Use compressed output as `content` in `tool_result` messages appended to `state.supervision_messages`
- [x] 2.3 Add per-result compression timeout guard
- [x] 2.4 Fall back to truncated raw summary (800 chars) on compression failure
- [x] 2.5 Test: verify supervision messages contain compressed findings (not raw)
- [x] 2.6 Test: verify compression failure falls back to truncated summary
- [x] 2.7 Test: verify supervision message history growth rate is reduced vs baseline

## Phase 3 — Novelty-Tagged Search Results for Researcher Stop Decisions
- [x] 3.1 Add novelty scoring in `_handle_web_search_tool` comparing new sources against existing sub-query sources
- [x] 3.2 Annotate formatted results with `[NEW]`, `[RELATED: <title>]`, or `[DUPLICATE]` tags
- [x] 3.3 Add novelty summary line to search results message header
- [x] 3.4 Update think-tool stop-criteria injection to reference novelty tags
- [x] 3.5 Add or reuse lightweight content similarity helper in `_helpers.py`
- [x] 3.6 Test: verify novelty tags appear in researcher message history
- [x] 3.7 Test: verify duplicate sources are correctly tagged
- [x] 3.8 Test: verify think injection references novelty context

## Phase 4 — Type-Aware Supervision Message Truncation
- [x] 4.1 Implement type-aware budgeting: 60% reasoning (think + delegation), 40% findings (tool_result)
- [x] 4.2 Within findings bucket, truncate message bodies (keep headers) before dropping whole messages
- [x] 4.3 Add `preserve_last_n_thinks` parameter (default: 2) to unconditionally keep recent think messages
- [x] 4.4 Replace or extend `truncate_supervision_messages` with new logic
- [x] 4.5 Test: verify think messages are preserved when findings messages are truncated
- [x] 4.6 Test: verify last N think messages survive aggressive truncation
- [x] 4.7 Test: verify total token usage stays within model limits after truncation

## Phase 5 — Coverage Delta Injection for Supervisor Think Step
- [x] 5.1 Add `_compute_coverage_delta` helper comparing current vs previous round's per-query coverage
- [x] 5.2 Store coverage snapshots in `state.metadata["coverage_snapshots"]` keyed by round number
- [x] 5.3 Build compact delta summary string and inject into think step user prompt
- [x] 5.4 Modify `_supervision_think_step` to accept and incorporate coverage delta
- [x] 5.5 Test: verify delta correctly identifies newly sufficient, still-insufficient, and new queries
- [x] 5.6 Test: verify delta is injected into think prompt on rounds > 0
- [x] 5.7 Test: verify coverage snapshots persist across supervision rounds

## Phase 6 — Confidence-Scored Coverage Heuristic
- [ ] 6.1 Extend `_assess_coverage_heuristic` with multi-dimensional scoring (source adequacy, domain diversity, query completion rate)
- [ ] 6.2 Compute weighted confidence score with configurable weights
- [ ] 6.3 Add `confidence`, `dominant_factors`, `weak_factors` to return dict
- [ ] 6.4 Use `confidence >= threshold` for `should_continue_gathering` decision
- [ ] 6.5 Log confidence breakdown in audit events
- [ ] 6.6 Test: verify confidence score reflects actual coverage quality
- [ ] 6.7 Test: verify threshold-based decision matches expected behavior at boundary values
- [ ] 6.8 Test: verify audit events contain confidence breakdown
