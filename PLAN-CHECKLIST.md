# Implementation Checklist

## Step 1: Change `execute_fn` callback to `workflow` parameter
- [ ] In `extract_and_verify_claims`: replace `execute_fn: ExecuteFn` with `workflow: Any` parameter
- [ ] In `apply_corrections`: replace `execute_fn: ExecuteFn` with `workflow: Any` parameter
- [ ] In `remap_unsupported_citations`: replace `execute_fn: ExecuteFn` with `workflow: Any` parameter
- [ ] Thread `workflow` + `state` down to internal helpers: `_extract_claims_chunked`, `_extract_claims_single_chunk`, `_verify_claims_batch`, `_verify_single_claim`, `_apply_single_correction`, `_remap_single_claim`

## Step 2: Replace `await execute_fn(...)` calls with `execute_llm_call(...)`
- [ ] Add imports: `execute_llm_call`, `LLMCallResult` from `._lifecycle`
- [ ] `_extract_claims_single_chunk` (line 402): replace `execute_fn(...)` with `execute_llm_call(workflow=workflow, state=state, phase_name="claim_extraction", ...)`
- [ ] Handle return: `LLMCallResult` → use `ret.result`, `WorkflowResult` → return `[]`
- [ ] `_verify_single_claim` (line 904): replace `execute_fn(...)` with `execute_llm_call(workflow=workflow, state=state, phase_name="claim_verification", ...)`
- [ ] Handle return: `LLMCallResult` → use `ret.result`, error → set verdict to UNSUPPORTED
- [ ] `_apply_single_correction` (line 1147): replace `execute_fn(...)` with `execute_llm_call(workflow=workflow, state=state, phase_name="claim_verification_correction", ...)`
- [ ] Handle return: `LLMCallResult` → use `ret.result`, error → return False
- [ ] `_remap_single_claim` inner call (line 1408): replace `execute_fn(...)` with `execute_llm_call(workflow=workflow, state=state, phase_name="claim_verification_remap", ...)`
- [ ] Handle return: `LLMCallResult` → use `ret.result`, error → return False

## Step 3: Update `workflow_execution.py` callers
- [ ] Line 477: `execute_fn=self._execute_provider_async` → `workflow=self`
- [ ] Line 489: `execute_fn=self._execute_provider_async` → `workflow=self`
- [ ] Line 497: `execute_fn=self._execute_provider_async` → `workflow=self`

## Step 4: Clean up `ExecuteFn` type alias
- [ ] Delete `ExecuteFn = Callable[..., Any]` (line 51) if no longer used
- [ ] Remove the `Callable` import if no longer needed
- [ ] Update `TYPE_CHECKING` imports as needed

## Step 5: Wire topic research parse retry
- [ ] Add `state: DeepResearchState` parameter to `_parse_with_retry_async`
- [ ] Replace `self._execute_provider_async(...)` call at line 1126 with `execute_llm_call(workflow=self, state=state, phase_name="topic_research_parse_retry", ...)`
- [ ] Handle `LLMCallResult` / `WorkflowResult` return type
- [ ] Update caller at line 778 to pass `state=state`
- [ ] Delete bare `except (asyncio.TimeoutError, OSError, ValueError, RuntimeError)` — lifecycle handles this

## Step 6: Update tests
- [ ] Find tests mocking `execute_fn` in claim verification test files
- [ ] Update mocks to provide a mock workflow with `_execute_provider_async` + lifecycle-required fields (`provider_id`, `model_used`, `duration_ms`, `input_tokens`, `output_tokens`, `cached_tokens`, `metadata`)
- [ ] Update `test_deep_research.py` claim verification integration paths if affected
- [ ] Update `_parse_with_retry_async` test scenarios for new `state` parameter

## Verification
- [ ] `pytest tests/ -k claim_verification` passes
- [ ] `pytest tests/ -k topic_research` passes
- [ ] `pytest tests/ -k deep_research` passes (full suite)
- [ ] No regressions in other phases: `pytest tests/ -k "planning or synthesis or analysis or supervision"` passes
