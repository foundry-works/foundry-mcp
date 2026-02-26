# PLAN: Post-Review Fixes for Deep Research Workflow

Fixes identified during senior engineer review of the deep research workflow overhaul.
Organized into 6 phases by priority: critical correctness/security first, then major bugs, then cleanup.

---

## Phase 1 — Critical: Blocking Timeout & Crash Handler

### 1.1 Add `asyncio.wait_for` timeout to synchronous execution paths

**File:** `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`

**Problem:** After commit `0097b14` switched from background to blocking execution, `task_timeout` is accepted but never enforced. The synchronous path calls `_execute_workflow_async()` directly without any timeout wrapping. A malfunctioning provider or network issue blocks the MCP tool handler indefinitely.

**Fix:** In both `_start_research()` (lines ~166-201) and `_continue_research()` (lines ~262-297), wrap the workflow coroutine in `asyncio.wait_for(coro, timeout=task_timeout)`. Add an `except asyncio.TimeoutError` handler that calls `state.mark_failed()` with a timeout message and saves state, mirroring the logic in `background_tasks.py:116-138`.

Apply to all three event-loop branches (running loop via ThreadPoolExecutor, and `asyncio.run()` fallback). Also simplify the dead `loop.run_until_complete()` branch (unreachable on Python 3.12+) — replace the three-way dispatch with a two-way check using `asyncio.get_running_loop()`.

### 1.2 Fix `_active_research_memory` global variable scoping

**File:** `src/foundry_mcp/core/research/workflows/deep_research/core.py` (lines ~165-167)
**File:** `src/foundry_mcp/core/research/workflows/deep_research/infrastructure.py`

**Problem:** `core.py` uses `global _active_research_memory` which sets a variable in `core`'s namespace, not `infrastructure`'s. The crash handler in `infrastructure.py` always sees `None`.

**Fix:** In `infrastructure.py`, add a setter function:
```python
def set_active_research_memory(memory):
    global _active_research_memory
    _active_research_memory = memory
```
In `core.py` `__init__`, replace the `global` statement with:
```python
from foundry_mcp.core.research.workflows.deep_research.infrastructure import set_active_research_memory
set_active_research_memory(self.memory)
```
Remove the stale `global _active_research_memory` and `_active_research_memory = None` from `core.py`.

### 1.3 Fix test regression from blocking behavior change

**File:** `tests/tools/unified/test_research.py` (lines ~830-851)

**Problem:** `test_config_default_applies_when_param_omitted` expects `mock_workflow.execute` called once, but blocking mode calls it twice (start + report). The mock returns the same `WorkflowResult` for both calls, and the second call (report action) gets a result whose `metadata` structure doesn't match expectations.

**Fix:** Configure `mock_workflow.execute` with `side_effect` that returns different results for the start call vs the report call. Assert the start call's `task_timeout` argument, and verify the overall response structure accounts for the report enrichment path.

---

## Phase 2 — Critical: Authorization & Prompt Injection Sanitization

### 2.1 Fix `UnboundLocalError` in authorization denial path

**File:** `src/foundry_mcp/core/authorization.py` (lines ~336-356)

**Problem:** `raw_action` is only defined inside `if not tool_name:` but referenced unconditionally in the denial logging at line ~347 (`normalized_action or raw_action`). When `tool_name` is truthy and `normalized_action` is falsy, this raises `UnboundLocalError`, potentially degrading to fail-open.

**Fix:** Move `raw_action = (action or "").lower()` above the `if not tool_name:` block so it's always defined. Add a unit test covering the denial path when `tool_name` is provided but the action is not in the allowlist.

### 2.2 Sanitize user query and system_prompt in clarification and brief phases

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/clarification.py` (~line 295)
**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py` (~line 261)

**Problem:** Every downstream phase sanitizes external content before prompt interpolation, but the entry-point phases (clarification, brief) interpolate `state.original_query` and `state.system_prompt` raw. Unsanitized output flows into `state.research_brief` and `state.clarification_constraints`, which downstream phases trust.

**Fix:** In `_build_clarification_user_prompt()`, wrap `state.original_query` and `state.system_prompt` in `sanitize_external_content()`. Same in `_build_brief_user_prompt()`. Also sanitize `clarification_constraints` keys and values in `brief.py` (~line 272).

### 2.3 Sanitize `think_output` in supervision prompts

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Problem:** `think_output` (from a previous LLM call influenced by external data) is embedded raw inside `<gap_analysis>` and `<decomposition_strategy>` XML tags — privileged prompt positions. Multi-hop injection vectors could propagate.

**Fix:** Apply `sanitize_external_content()` to `think_output` at the interpolation points:
- Line ~1647: before embedding in `<gap_analysis>` tag
- Line ~1949: before embedding in `<decomposition_strategy>` tag

### 2.4 Sanitize assistant messages and global compression content

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Problem:** In `_build_message_history_prompt()` (line ~65), assistant messages bypass `sanitize_external_content()` despite potentially echoing web-sourced content. In `_execute_global_compression_async()` (lines ~896-912), `compressed_findings`, `per_topic_summary`, and query text are interpolated without sanitization.

**Fix:**
- In `_build_message_history_prompt()`, apply `sanitize_external_content()` to assistant message content and the default/unknown role path.
- In `_execute_global_compression_async()`, wrap `content` and `query_text` in `sanitize_external_content()` before building `topic_sections`.

### 2.5 Sanitize contradiction/gap fields in synthesis prompt

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (~lines 816-835)

**Problem:** `contradiction.severity`, `contradiction.description`, `contradiction.resolution`, `gap.description` are LLM-derived fields from analysis of web content, interpolated into the synthesis prompt without sanitization.

**Fix:** Apply `sanitize_external_content()` to all contradiction and gap fields in `_append_contradictions_and_gaps()` and `_build_synthesis_user_prompt()`.

### 2.6 Expand injection tag pattern coverage

**File:** `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` (lines ~794-803)

**Problem:** `_INJECTION_TAG_PATTERN` doesn't cover `<message>`, `<context>`, `<document>`, `<thinking>`, `<reflection>` tags or OpenAI special tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`). Heading pattern bypassed by trailing whitespace.

**Fix:**
- Add `message|messages|context|document|thinking|reflection` to `_INJECTION_TAG_PATTERN`.
- Add a new `_SPECIAL_TOKEN_PATTERN` for `<\|.*?\|>` (catches OpenAI-family special tokens).
- Relax `_INJECTION_HEADING_PATTERN`'s `$` anchor to `\s*$`.
- Apply the new pattern in `sanitize_external_content()`.

### 2.7 Add `system_prompt` length validation

**File:** `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py` (~line 88)

**Problem:** `query` is validated against `MAX_PROMPT_LENGTH`, but `system_prompt` passes through without any length check. Enables cost amplification via arbitrarily large system prompts.

**Fix:** Add `system_prompt` length validation alongside the existing `query` check in the `violations` list in `_start_research()`.

### 2.8 Remove full prompt/response logging from legacy supervision audit path

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (~lines 2605-2615)

**Problem:** Legacy `_execute_supervision_query_generation_async` logs full system prompt, user prompt, and raw LLM response to audit events — data-at-rest exposure and state bloat. The delegation model path correctly logs only structured metadata.

**Fix:** Replace `system_prompt`, `user_prompt`, and `raw_response` in the audit data dict with bounded summaries: `system_prompt_length`, `user_prompt_length`, `response_length`, and the existing structured fields (coverage_outcome, model, token counts).

---

## Phase 3 — Major: Cancellation Race Conditions

### 3.1 Fix triple `mark_cancelled` race on cancellation

**File:** `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py` (~line 559)
**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` (~line 447)
**File:** `src/foundry_mcp/core/research/workflows/deep_research/background_tasks.py` (~line 98)

**Problem:** Cancellation triggers `mark_cancelled()` + `save_deep_research()` from up to three racing paths with different state objects. The inner handler's careful iteration-rollback gets overwritten by simpler writes.

**Fix:**
1. In `_cancel_research()` (`action_handlers.py`): only set the cancel flag on the `BackgroundTask`. Remove the direct `state = self.memory.load_deep_research(...)` / `mark_cancelled()` / `save_deep_research()` call. The workflow's own `CancelledError` handler is the sole writer of terminal state.
2. In `_execute_workflow_async()` (`workflow_execution.py`): after the elaborate partial-result discard and state rollback, do NOT re-raise `CancelledError`. Instead, return a failure `WorkflowResult` with `error="Research cancelled"`. This prevents the outer `background_tasks.py` handler from overwriting the rollback state.
3. In `background_tasks.py`: keep the `CancelledError` handler as a safety net, but add a guard: only call `mark_cancelled()` if `state.completed_at is None` (not already terminated by the inner handler).

---

## Phase 4 — Major: Performance & Resource Management

### 4.1 Move content-similarity dedup outside the async lock

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~lines 1431-1475)

**Problem:** Content-similarity dedup iterates over all `state.sources` for each new source while holding the shared async lock, creating O(n^2) lock contention that blocks all concurrent topic researchers.

**Fix:** Restructure `_topic_search()`:
1. Before the per-source loop, snapshot `state.sources` (under lock) into a local list.
2. Compute content similarity against the snapshot **outside** the lock.
3. Acquire the lock briefly only for the final add-or-skip decision (re-check URL dedup under lock to handle races).

### 4.2 Cap `supervision_history` growth in state metadata

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` (~lines 257-264)

**Problem:** `supervision_history` is appended on every round with full `think_output` and `post_execution_think` strings but never trimmed. Can grow to hundreds of KB.

**Fix:** After appending to `supervision_history`, trim to the most recent `_MAX_SUPERVISION_HISTORY_ENTRIES = 10` entries (consistent with `_MAX_STORED_DIRECTIVES`). Also truncate `think_output` and `post_execution_think` fields to `_MAX_THINK_OUTPUT_STORED_CHARS = 2000` characters each before storing.

### 4.3 Fix synthesis retry truncation to start from actual findings length

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (~lines 420-435)

**Problem:** `max_findings_chars` starts from `model_token_limit * 4` (full context window) not the actual findings length. Early retries often do nothing because the budget exceeds the content.

**Fix:** Initialize `max_findings_chars` based on the actual findings section length:
```python
findings_section_len = _estimate_findings_section_length(user_prompt)
max_findings_chars = int(findings_section_len * 0.7)  # 30% cut on first retry
```
Then reduce by 10% each subsequent retry from this meaningful starting point.

### 4.4 Fix O(n^2) history truncation in topic researcher

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~lines 383-389)

**Problem:** `_truncate_researcher_history` recomputes `sum(len(...))` on every iteration.

**Fix:** Compute `total_chars` once before the loop, then subtract `len(result[0].get("content", ""))` before each `result.pop(0)`.

---

## Phase 5 — Major: Config & Model Correctness

### 5.1 Fix Semantic Scholar rate limit value

**File:** `src/foundry_mcp/config/research.py` (~line 199)

**Problem:** Config says `100` (req/min) but comment says "100 req/5min" (= 20 req/min). The value should match the actual API limit.

**Fix:** Change to `"semantic_scholar": 20` and update comment to `# Semantic Scholar: ~20 req/min (100 req/5min unauthenticated)`.

### 5.2 Clarify `deep_research_timeout` semantics

**File:** `src/foundry_mcp/config/research.py` (~lines 45, 172)

**Problem:** Docstring says "per operation", field comment says "whole workflow", Pydantic model has different field name and default.

**Fix:** Update the class docstring (line ~45) to say "Default wall-clock timeout for the entire deep research workflow in seconds". Update the field comment (line ~172) to match. Add a cross-reference to `deep_research_planning_timeout` / `deep_research_synthesis_timeout` as the per-phase timeouts.

### 5.3 Fix `gpt-4.1-*` token limit typo

**File:** `src/foundry_mcp/config/model_token_limits.json`

**Problem:** `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano` all show `1047576` which is 1000 less than `1048576` (2^20). Likely a typo.

**Fix:** Verify against OpenAI docs. If 1M is correct, change to `1048576`. If not, use the actual value.

### 5.4 Add `terminal_status` to `mark_failed`

**File:** `src/foundry_mcp/core/research/models/deep_research.py`

**Problem:** `mark_cancelled` and `mark_interrupted` set `metadata["terminal_status"]` but `mark_failed` does not, making status-checking inconsistent.

**Fix:** Add `self.metadata["terminal_status"] = "failed"` to `mark_failed()`.

---

## Phase 6 — Minor: Cleanup & Hardening

### 6.1 Sanitize URLs in topic researcher prompt

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~line 74)

**Fix:** Apply `sanitize_external_content()` to `src.url` in `_format_source_block()`.

### 6.2 Add content-similarity dedup to `_topic_extract`

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~lines 1580-1607)

**Fix:** Factor dedup logic from `_topic_search` into a shared `_dedup_and_add_source()` helper. Call it from both `_topic_search` and `_topic_extract`.

### 6.3 Rebuild topic researcher system prompt each turn

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (~lines 517-522)

**Fix:** Move `system_prompt = _build_researcher_system_prompt(...)` inside the turn loop, passing the current `budget_remaining`.

### 6.4 Remove dead `PlanningPhaseMixin` from class hierarchy

**File:** `src/foundry_mcp/core/research/workflows/deep_research/core.py` (~line 115)

**Fix:** Remove `PlanningPhaseMixin` from the `DeepResearchWorkflow` MRO. Keep the import and the module for now (legacy resume states may reference it), but remove it from the active class composition.

### 6.5 Remove `idea.md` (copyrighted content)

**File:** `idea.md`

**Fix:** Delete `idea.md` — it's a full reproduction of a third-party Substack article. The `future/README.md` already references it properly by description.

### 6.6 Stage deletion of `PLAN-BG-CHECKLIST.md` and `PLAN-BG.md`

**Fix:** These are already deleted in the working tree but not staged. Include in the fix commit.
