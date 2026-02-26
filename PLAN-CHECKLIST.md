# PLAN-CHECKLIST: Post-Review Fixes

Track completion of each fix. Mark `[x]` when implemented and verified.

---

## Phase 1 — Critical: Blocking Timeout & Crash Handler

- [x] **1.1** Add `asyncio.wait_for` timeout to synchronous execution paths in `action_handlers.py`
  - [x] Wrap coroutine in `_start_research()` with `asyncio.wait_for(coro, timeout=task_timeout)`
  - [x] Wrap coroutine in `_continue_research()` with `asyncio.wait_for(coro, timeout=task_timeout)`
  - [x] Add `asyncio.TimeoutError` handler that calls `state.mark_failed()` and saves state
  - [x] Simplify dead `loop.run_until_complete()` branch — replace three-way dispatch with two-way (`get_running_loop` + `asyncio.run`)
  - [x] Add unit test: verify timeout fires and state is marked failed
- [x] **1.2** Fix `_active_research_memory` global variable scoping
  - [x] Add `set_active_research_memory()` setter in `infrastructure.py`
  - [x] Call setter from `core.py` `__init__` instead of `global` statement
  - [x] Remove stale `global _active_research_memory` and `_active_research_memory = None` from `core.py`
  - [x] Add unit test: verify `infrastructure._active_research_memory` is set after workflow init
- [x] **1.3** Fix test regression from blocking behavior change
  - [x] Update `test_config_default_applies_when_param_omitted` to expect two `execute` calls (start + report)
  - [x] Configure `side_effect` with different results for start vs report calls
  - [x] Verify test passes with `pytest tests/tools/unified/test_research.py::TestDeepResearchTimeoutConfig -x`

---

## Phase 2 — Critical: Authorization & Prompt Injection Sanitization

- [x] **2.1** Fix `UnboundLocalError` in authorization denial path
  - [x] Move `raw_action = (action or "").lower()` above the `if not tool_name:` block in `authorization.py`
  - [x] Add unit test: denial with `tool_name` truthy and `normalized_action` falsy doesn't crash
- [x] **2.2** Sanitize user query and system_prompt in clarification and brief phases
  - [x] Wrap `state.original_query` in `sanitize_external_content()` in `_build_clarification_user_prompt()`
  - [x] Wrap `state.system_prompt` in `sanitize_external_content()` in `_build_clarification_user_prompt()`
  - [x] Wrap `state.original_query` in `sanitize_external_content()` in `_build_brief_user_prompt()`
  - [x] Wrap `state.system_prompt` in `sanitize_external_content()` in `_build_brief_user_prompt()`
  - [x] Sanitize `clarification_constraints` keys and values in `brief.py`
- [x] **2.3** Sanitize `think_output` in supervision prompts
  - [x] Apply `sanitize_external_content()` before `<gap_analysis>` tag interpolation
  - [x] Apply `sanitize_external_content()` before `<decomposition_strategy>` tag interpolation
- [x] **2.4** Sanitize assistant messages and global compression content
  - [x] Apply `sanitize_external_content()` to assistant message content in `_build_message_history_prompt()`
  - [x] Apply `sanitize_external_content()` to default/unknown role content in `_build_message_history_prompt()`
  - [x] Wrap `content` and `query_text` in `sanitize_external_content()` in `_execute_global_compression_async()`
- [x] **2.5** Sanitize contradiction/gap fields in synthesis prompt
  - [x] Apply `sanitize_external_content()` to `contradiction.severity`, `.description`, `.resolution`
  - [x] Apply `sanitize_external_content()` to `gap.description` and related fields
- [x] **2.6** Expand injection tag pattern coverage
  - [x] Add `message|messages|context|document|thinking|reflection` to `_INJECTION_TAG_PATTERN`
  - [x] Add `_SPECIAL_TOKEN_PATTERN` for `<\|.*?\|>` (OpenAI-family special tokens)
  - [x] Relax `_INJECTION_HEADING_PATTERN` `$` anchor to `\s*$` (already present)
  - [x] Apply new pattern in `sanitize_external_content()`
  - [x] Add unit tests for new patterns (message tags, special tokens, heading with trailing whitespace)
- [x] **2.7** Add `system_prompt` length validation
  - [x] Add `system_prompt` length check alongside `query` validation in `_start_research()`
  - [x] Add unit test: oversized system_prompt returns validation error
- [x] **2.8** Remove full prompt/response logging from legacy supervision audit path
  - [x] Replace `system_prompt`, `user_prompt`, `raw_response` with bounded summaries in audit data
  - [x] Keep structured fields (coverage_outcome, model, token counts)

---

## Phase 3 — Major: Cancellation Race Conditions

- [x] **3.1** Fix triple `mark_cancelled` race
  - [x] `_cancel_research()`: remove direct `load/mark_cancelled/save` — only set cancel flag on `BackgroundTask`
  - [x] `_execute_workflow_async()`: return failure `WorkflowResult` instead of re-raising `CancelledError`
  - [x] `background_tasks.py`: guard `mark_cancelled()` with `if state.completed_at is None`
  - [x] Add test: cancellation preserves the inner handler's partial-result discard state

---

## Phase 4 — Major: Performance & Resource Management

- [x] **4.1** Move content-similarity dedup outside the async lock
  - [x] Snapshot `state.sources` under lock before per-source loop
  - [x] Compute content similarity against snapshot outside lock
  - [x] Re-acquire lock briefly for final add-or-skip decision with URL re-check
  - [x] Add test: concurrent topic researchers don't block each other during dedup
- [x] **4.2** Cap `supervision_history` growth
  - [x] Add `_MAX_SUPERVISION_HISTORY_ENTRIES = 10` constant
  - [x] Add `_MAX_THINK_OUTPUT_STORED_CHARS = 2000` constant
  - [x] Truncate think outputs before storing in history
  - [x] Trim history list to most recent N entries after append
- [x] **4.3** Fix synthesis retry truncation starting point
  - [x] Compute actual findings section length from `user_prompt`
  - [x] Initialize `max_findings_chars` as `int(findings_section_len * 0.7)` on first retry
  - [x] Reduce by 10% from that meaningful starting point each subsequent retry
- [x] **4.4** Fix O(n^2) history truncation in topic researcher
  - [x] Compute `total_chars` once before loop
  - [x] Subtract dropped element size instead of recomputing sum

---

## Phase 5 — Major: Config & Model Correctness

- [ ] **5.1** Fix Semantic Scholar rate limit value
  - [ ] Change `"semantic_scholar": 100` to `"semantic_scholar": 20`
  - [ ] Update comment to `# Semantic Scholar: ~20 req/min (100 req/5min unauthenticated)`
- [ ] **5.2** Clarify `deep_research_timeout` semantics
  - [ ] Update class docstring to say "wall-clock timeout for entire workflow"
  - [ ] Update field comment at line ~172 to match
  - [ ] Add cross-reference to per-phase timeout fields
- [ ] **5.3** Fix `gpt-4.1-*` token limit typo
  - [ ] Verify actual value against OpenAI docs
  - [ ] Update `model_token_limits.json` (likely `1047576` → `1048576`)
- [ ] **5.4** Add `terminal_status` to `mark_failed`
  - [ ] Add `self.metadata["terminal_status"] = "failed"` to `mark_failed()` in `deep_research.py`

---

## Phase 6 — Minor: Cleanup & Hardening

- [ ] **6.1** Sanitize URLs in topic researcher prompt
  - [ ] Apply `sanitize_external_content()` to `src.url` in `_format_source_block()`
- [ ] **6.2** Add content-similarity dedup to `_topic_extract`
  - [ ] Factor dedup logic into shared `_dedup_and_add_source()` helper
  - [ ] Call from both `_topic_search` and `_topic_extract`
- [ ] **6.3** Rebuild topic researcher system prompt each turn
  - [ ] Move `system_prompt = _build_researcher_system_prompt(...)` inside the turn loop
  - [ ] Pass current `budget_remaining`
- [ ] **6.4** Remove dead `PlanningPhaseMixin` from class hierarchy
  - [ ] Remove from `DeepResearchWorkflow` MRO in `core.py`
- [ ] **6.5** Remove `idea.md` (copyrighted content)
  - [ ] Delete `idea.md`
- [ ] **6.6** Stage deletion of `PLAN-BG-CHECKLIST.md` and `PLAN-BG.md`
  - [ ] Include in fix commit

---

## Validation

After all phases:

- [ ] Full test suite passes: `pytest tests/ -x --timeout=120`
- [ ] Research-specific tests pass: `pytest tests/core/research/ -x --timeout=120`
- [ ] No new `grep -r "global _active_research_memory" src/` hits in `core.py`
- [ ] `sanitize_external_content` grep shows coverage at all prompt interpolation sites
- [ ] Blocking execution respects `task_timeout` (manual or integration test)
