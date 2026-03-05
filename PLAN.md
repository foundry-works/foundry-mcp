# Plan: Wire topic research LLM calls through `execute_llm_call` lifecycle wrapper

## Context

Deep research session `deepres-4fbeb6b8568d` failed silently — all 4 topic researcher sub-queries completed with 0 turns, 0 searches, 0 sources, but the audit trail contained **no information about why**. Investigation revealed two compounding problems:

1. **`_execute_researcher_llm_call` bypasses the `execute_llm_call` lifecycle wrapper.** Every other phase (clarification, brief, supervision) routes LLM calls through `execute_llm_call()` in `_lifecycle.py`, which provides audit events (`llm.call.started` / `llm.call.completed`), heartbeats, PhaseMetrics recording, and progressive ContextWindowError recovery. Topic research calls `_execute_provider_async` directly, losing all of this instrumentation.

2. **Two completely silent failure paths** in `_execute_researcher_llm_call` (lines 1052, 1055) return `None` with no `logger.warning` and no audit event.

The fix is to migrate `_execute_researcher_llm_call` to delegate to `execute_llm_call`, which automatically provides audit trail, metrics, and error visibility — making failures like this diagnosable from the session JSON alone.

## Files to modify

- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — primary change
- `tests/` — update any tests that mock `_execute_researcher_llm_call` or `_execute_provider_async` in the topic research context

## Implementation

### Step 1: Rewrite `_execute_researcher_llm_call` to use `execute_llm_call`

Replace the body of `_execute_researcher_llm_call` (lines 894-1062) to:

1. Build `system_prompt` and `user_prompt` (already done at lines 929-942 — keep this)
2. Call `execute_llm_call()` instead of `self._execute_provider_async()`:
   ```python
   from ._lifecycle import execute_llm_call, LLMCallResult

   ret = await execute_llm_call(
       workflow=self,
       state=state,
       phase_name="topic_research",
       system_prompt=system_prompt,
       user_prompt=user_prompt,
       provider_id=provider_id,
       model=researcher_model,
       temperature=0.3,
       timeout=self.config.deep_research_reflection_timeout,
       role="topic_reflection",
       skip_token_tracking=True,  # caller tracks tokens via local_tokens_used
   )
   ```
3. Handle the return type:
   - `isinstance(ret, WorkflowResult)` → error path (replaces the current `None` return)
   - `isinstance(ret, LLMCallResult)` → success, return `ret.result` (the inner `WorkflowResult`)

4. **Keep the history-truncation recovery** as an outer wrapper: if `execute_llm_call` returns a `WorkflowResult` with context-window metadata, truncate `message_history` via `_truncate_researcher_history(..., budget_fraction=0.5)`, rebuild the prompt, and retry once. Use `_is_context_window_exceeded()` from `_lifecycle.py` to detect this.

5. **Delete the hand-rolled ContextWindowError/exception handling** (lines 966-1062) — `execute_llm_call` handles this internally with progressive truncation retries. The outer history-truncation loop handles the case `execute_llm_call` can't (rebuilding from truncated history).

### Step 2: Update the caller in `_execute_topic_research_async`

Add `state` to the `_execute_researcher_llm_call` call (line 753). The caller already has `state` in scope.

The return type contract stays the same: success → `WorkflowResult` object, failure → `None`. No changes needed at the call site (lines 765-770) since `_execute_researcher_llm_call` will still return `WorkflowResult | None` — it unwraps `LLMCallResult.result` internally.

### Step 3: Update tests

Find and update tests that:
- Mock `_execute_provider_async` in topic research test scenarios → should mock `execute_llm_call` or its provider instead
- Assert on `_execute_researcher_llm_call` return types

## What this fixes

| Before | After |
|--------|-------|
| 0 audit events for topic research LLM calls | `llm.call.started` + `llm.call.completed` per call (with provider, duration, status) |
| Silent failures on lines 1052/1055 | All failures captured in audit with error details |
| No PhaseMetrics for topic research turns | PhaseMetrics recorded per turn |
| No heartbeat updates during researcher loop | Heartbeat updated each turn |
| Hand-rolled CWE handling with gaps | Battle-tested progressive truncation + outer history-truncation fallback |

## What this does NOT change

- The ReAct loop structure, tool dispatch, reflection enforcement — untouched
- The `_finalize_topic_result` audit event — still fires, now complemented by per-turn events
- Provider resolution logic — same `safe_resolve_model_for_role` + `resolve_phase_provider` chain
- External behavior — same prompts, same tool calls, same results

## Verification

1. **Unit tests**: Run existing topic research tests — `pytest tests/ -k topic_research`
2. **Integration**: Run a deep research session and verify:
   - `llm.call.started` / `llm.call.completed` events appear in audit JSONL for topic_research phase
   - On provider failure, the audit shows `status: "error"` with provider info
   - `phase_metrics` includes `topic_research` entries
3. **Regression**: Verify supervision/clarification/brief phases still work (they don't touch this code)
