# Plan: Wire remaining direct `_execute_provider_async` calls through `execute_llm_call`

## Context

After wiring topic research LLM calls through the `execute_llm_call` lifecycle wrapper (0.18.0a13), two areas remain that bypass the lifecycle and call `_execute_provider_async` directly:

1. **Claim verification pipeline** (`claim_verification.py`) — 4 LLM call sites across extraction, verification, correction, and citation remapping. All receive `self._execute_provider_async` as an `execute_fn` callback from `workflow_execution.py`. This is the **high-priority** gap: an entire sub-pipeline with no audit events, no heartbeats, no PhaseMetrics, and no lifecycle-managed context-window recovery.

2. **Topic research parse retry** (`topic_research.py:1126`) — 1 LLM call in `_parse_with_retry_async` for re-prompting on malformed JSON. **Lower priority**: rare path, main researcher call is already instrumented.

### What's missing without lifecycle wiring

| Capability | Claim verification | Parse retry |
|---|---|---|
| `llm.call.started` / `llm.call.completed` audit events | Missing | Missing |
| Heartbeat updates | Missing | Missing |
| PhaseMetrics recording | Missing | Missing |
| Progressive context-window recovery (tiered truncation) | Missing (bare try/except) | Missing (bare try/except) |
| Silent failure visibility | Partial (logger.warning exists but no audit) | Partial |

## Files to modify

### Primary
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py` — replace `execute_fn` callback pattern with `execute_llm_call` delegation
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` — update caller to pass `workflow` instead of `execute_fn`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — wire parse retry through lifecycle

### Tests
- `tests/core/research/workflows/test_deep_research.py` — update claim verification mocks
- `tests/core/research/workflows/test_topic_research.py` — update parse retry tests if affected
- Any dedicated claim verification test files

## Implementation

### Step 1: Change the `execute_fn` callback pattern to pass `workflow` + `state`

The current design passes `self._execute_provider_async` as an `execute_fn: ExecuteFn` callback to the claim verification free functions. This made sense when claim verification was a standalone module, but it prevents lifecycle integration because `execute_llm_call` needs both `workflow` and `state`.

**Change the public API** of the three entry-point functions:

```python
# Before
async def extract_and_verify_claims(
    state, config, provider_id, execute_fn, timeout
)
async def apply_corrections(
    state, config, verification_result, execute_fn, provider_id
)
async def remap_unsupported_citations(
    state, verification_result, execute_fn, provider_id, timeout, max_concurrent
)

# After — replace execute_fn with workflow
async def extract_and_verify_claims(
    state, config, provider_id, workflow, timeout
)
async def apply_corrections(
    state, config, verification_result, workflow, provider_id
)
async def remap_unsupported_citations(
    state, verification_result, workflow, provider_id, timeout, max_concurrent
)
```

Thread `workflow` down to the 4 internal call sites (`_extract_claims_single_chunk`, `_verify_single_claim`, `_apply_single_correction`, `_remap_single_claim`).

### Step 2: Replace each `await execute_fn(...)` with `await execute_llm_call(...)`

For each of the 4 call sites:

**a) `_extract_claims_single_chunk` (line 402)**
```python
ret = await execute_llm_call(
    workflow=workflow,
    state=state,
    phase_name="claim_extraction",
    system_prompt=system_prompt,
    user_prompt=chunk_prompt,
    provider_id=provider_id,
    model=None,
    temperature=0.0,
    timeout=timeout,
    role="claim_verification",
)
```
Handle `LLMCallResult` → use `ret.result`, `WorkflowResult` → log warning + return `[]`.

**b) `_verify_single_claim` (line 904)**
```python
ret = await execute_llm_call(
    workflow=workflow,
    state=state,
    phase_name="claim_verification",
    system_prompt=_VERIFICATION_SYSTEM_PROMPT,
    user_prompt=user_prompt,
    provider_id=provider_id,
    model=None,
    temperature=0.0,
    timeout=timeout,
    role="claim_verification",
)
```

**c) `_apply_single_correction` (line 1147)**
```python
ret = await execute_llm_call(
    workflow=workflow,
    state=state,
    phase_name="claim_verification_correction",
    system_prompt=_CORRECTION_SYSTEM_PROMPT,
    user_prompt=user_prompt,
    provider_id=provider_id,
    model=None,
    temperature=0.0,
    timeout=timeout,
    role="claim_verification",
)
```

**d) `_remap_single_claim` inner call (line 1408)**
```python
ret = await execute_llm_call(
    workflow=workflow,
    state=state,
    phase_name="claim_verification_remap",
    system_prompt=_REMAP_SYSTEM_PROMPT,
    user_prompt=user_prompt,
    provider_id=provider_id,
    model=None,
    temperature=0.0,
    timeout=timeout,
    role="claim_verification",
)
```

### Step 3: Update `workflow_execution.py` callers

Replace `execute_fn=self._execute_provider_async` with `workflow=self` at lines 477, 489, 497:

```python
verification_result = await extract_and_verify_claims(
    state=state,
    config=self.config,
    provider_id=resolve_phase_provider(self.config, "claim_verification", "synthesis"),
    workflow=self,  # was: execute_fn=self._execute_provider_async
    timeout=self.config.deep_research_claim_verification_timeout,
)

await apply_corrections(
    state=state,
    config=self.config,
    verification_result=verification_result,
    workflow=self,  # was: execute_fn=self._execute_provider_async
)

await remap_unsupported_citations(
    state=state,
    verification_result=verification_result,
    workflow=self,  # was: execute_fn=self._execute_provider_async
    ...
)
```

### Step 4: Remove the `ExecuteFn` type alias

Delete `ExecuteFn = Callable[..., Any]` (line 51) since it's no longer used. Update the `TYPE_CHECKING` import block to add `WorkflowResult` if not already present, and add `execute_llm_call`/`LLMCallResult` imports.

### Step 5: Wire topic research parse retry through lifecycle

In `_parse_with_retry_async` (topic_research.py:1126), replace the `self._execute_provider_async(...)` call with `execute_llm_call(...)`. This requires threading `state` into `_parse_with_retry_async` — add `state: DeepResearchState` parameter.

The caller at line 778 already has `state` in scope (from `_execute_topic_research_async`).

### Step 6: Update tests

1. **Claim verification tests**: Find tests that mock `execute_fn` or pass `_execute_provider_async` — update to pass a mock workflow with `_execute_provider_async` + lifecycle-required attributes.
2. **Topic research tests**: Update any parse-retry test scenarios to account for `state` parameter and lifecycle mock fields.
3. **Deep research integration tests**: Update `_mock_llm_provider` patterns in `test_deep_research.py` that exercise claim verification paths.

## What this fixes

| Before | After |
|--------|-------|
| 0 audit events for claim verification LLM calls | `llm.call.started` / `llm.call.completed` per call |
| No heartbeat during claim verification (can appear stale/stuck) | Heartbeat updated per LLM call |
| No PhaseMetrics for extraction/verification/correction/remap | PhaseMetrics recorded per call |
| Bare try/except for context-window errors | Progressive tiered truncation recovery |
| Parse retry in topic research uninstrumented | Full lifecycle coverage |

## What this does NOT change

- Claim verification logic (extraction prompts, verdict parsing, correction application, remap logic) — untouched
- The parallel verification structure (`_verify_claims_batch` with semaphore) — untouched
- Sequential correction application order — untouched
- External behavior — same prompts, same verdicts, same corrections

## Risk assessment

- **claim_verification.py is a standalone module** with free functions (not a mixin). Changing `execute_fn` to `workflow` changes the public API of 3 functions. Only one caller (`workflow_execution.py`). Low coupling risk.
- **Parallel verification** uses `asyncio.Semaphore` — `execute_llm_call` is async-safe (no shared mutable state). No concurrency risk.
- **`skip_token_tracking=False`** for claim verification (unlike topic research where the caller tracks tokens) — lifecycle wrapper handles token accounting automatically.

## Verification

1. **Unit tests**: `pytest tests/ -k claim_verification` passes
2. **Topic research tests**: `pytest tests/ -k topic_research` passes
3. **Full deep research suite**: `pytest tests/ -k deep_research` passes
4. **Integration**: Run a deep research session with claim verification enabled and verify:
   - `llm.call.started` / `llm.call.completed` events appear for `claim_extraction`, `claim_verification`, `claim_verification_correction`, `claim_verification_remap` phases
   - PhaseMetrics include claim verification entries
   - Heartbeat stays fresh during verification (no stale-task warnings)
