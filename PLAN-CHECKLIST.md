# Implementation Checklist

## Step 1: Rewrite `_execute_researcher_llm_call`
- [ ] Add `state: DeepResearchState` parameter to `_execute_researcher_llm_call` signature
- [ ] Add import for `execute_llm_call`, `LLMCallResult`, `_is_context_window_exceeded` from `._lifecycle`
- [ ] Keep prompt-building logic (system_prompt via `_build_researcher_system_prompt`, user_prompt via `_truncate_researcher_history` + `_build_react_user_prompt`)
- [ ] Replace `self._execute_provider_async(...)` call with `execute_llm_call(workflow=self, state=state, ...)` using `skip_token_tracking=True`
- [ ] Add outer history-truncation retry: if `execute_llm_call` returns `WorkflowResult` and `_is_context_window_exceeded(ret)`, truncate `message_history` at 50% budget, rebuild prompt, retry once
- [ ] On non-CWE failure (`isinstance(ret, WorkflowResult)` but not context-window): `logger.warning` + return `None`
- [ ] On success (`isinstance(ret, LLMCallResult)`): return `ret.result` (the inner `WorkflowResult`)
- [ ] Delete all hand-rolled `ContextWindowError` / exception handling (lines 966-1062)

## Step 2: Update caller in `_execute_topic_research_async`
- [ ] Pass `state=state` to `_execute_researcher_llm_call(...)` call at line 753

## Step 3: Update tests
- [ ] Find tests mocking `_execute_provider_async` for topic research paths
- [ ] Update mocks to target `execute_llm_call` or the underlying provider
- [ ] Verify no tests assert `_execute_researcher_llm_call` returns raw `_execute_provider_async` output

## Verification
- [ ] `pytest tests/ -k topic_research` passes
- [ ] `pytest tests/ -k deep_research` passes (full suite)
- [ ] Manual: confirm `llm.call.started` / `llm.call.completed` audit events appear for topic_research phase in a test run
