# Deep Research: Timeout Retry Cascade & Zero-Source Synthesis Fix

## Problem Statement

Deep research sessions fail silently when topic research LLM calls timeout. The retry
cascade in `_execute_provider_async` retries timeouts identically to transient errors,
burning ~130s per call with near-zero chance of recovery. When all topic researchers
timeout, 0 sources are collected, and synthesis generates an entirely ungrounded report
from LLM knowledge — with no citations and no warning to the user.

**Evidence:** Session `deepres-06df190929b5` — all 4 topic research agents timed out
at exactly 190s each (60s timeout x 3 attempts + 5s delay x 2 = 190s), producing
0 searches, 0 sources, 0 citations in the final report.

## Root Cause

### 1. Timeout Retry Cascade (`base.py:479-517`)

`_execute_provider_async` retries `ProviderTimeoutError` and `asyncio.TimeoutError` the
same as transient API errors (e.g., 500, rate limit). A timeout on a 60s call with the
same prompt will almost certainly timeout again — the prompt hasn't changed, the model
context hasn't shrunk. Retrying wastes `(max_retries * timeout) + (max_retries * retry_delay)`
seconds per call for zero benefit.

Default config: `max_retries=2`, `timeout=60s`, `delay=5s` → **190s wasted per call**.

### 2. No Zero-Source Guard Before Synthesis

When supervision completes with 0 sources (all topic researchers failed), the workflow
proceeds directly to synthesis without warning or recovery attempt. The synthesis LLM
generates a plausible-sounding report entirely from parametric knowledge, with no
citations. The user receives what appears to be a researched report but is actually
ungrounded.

### 3. Reflection Timeout May Be Too Tight

`deep_research_reflection_timeout` defaults to 60s, which is used for all topic research
LLM calls. First-turn calls include a long system prompt (~2KB) plus a 1000+ char
research directive. Some models/providers may need more time, especially under load.

## Scope

- **In scope:** Timeout retry policy, zero-source guard, topic research timeout config
- **Out of scope:** Fidelity-gated iteration loop (already working correctly), claim
  verification pipeline, citation postprocessing, general provider retry logic for
  non-timeout errors

## Design

### Fix 1: Cap Timeout Retries in `_execute_provider_async`

**File:** `src/foundry_mcp/core/research/workflows/base.py`

Differentiate timeout retries from transient error retries. Timeouts should either:
- Not be retried at all (skip to next provider immediately), OR
- Be retried at most once (in case of transient network issue vs. model overload)

Approach: Add a `max_timeout_retries` parameter (default: 0) to `_execute_provider_async`.
When a timeout occurs and timeout retries are exhausted, break immediately to the next
provider (or fail) instead of retrying with the same prompt.

The existing `max_retries` continues to govern non-timeout transient errors (API 500s,
rate limits, etc.) where retrying is genuinely useful.

### Fix 2: Zero-Source Warning in Synthesis Phase

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

Before entering synthesis, check `len(state.sources)`. If 0:
- Log a warning
- Emit an audit event (`zero_source_synthesis_warning`)
- Set `state.metadata["ungrounded_synthesis"] = True`
- Prepend a disclaimer to the report after synthesis:
  `> **Note:** This report was generated without web sources due to search failures.
  > All claims are based on the model's training data and may be outdated or inaccurate.`

This makes the ungrounded nature visible to the user rather than silently delivering
a hallucinated report.

### Fix 3: Configurable Topic Research Timeout

**File:** `src/foundry_mcp/config/research.py`

Add `deep_research_topic_research_timeout` (default: 90s) as a separate config from
`deep_research_reflection_timeout` (60s). Topic research first-turn prompts are
substantially larger than reflection prompts and warrant a higher timeout.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

Use the new config value in `_execute_researcher_llm_call` instead of
`deep_research_reflection_timeout`.

## Files Modified

| File | Change |
|------|--------|
| `src/foundry_mcp/core/research/workflows/base.py` | Add `max_timeout_retries` param, separate timeout retry logic |
| `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` | Add zero-source guard before synthesis |
| `src/foundry_mcp/config/research.py` | Add `deep_research_topic_research_timeout` config |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` | Use new timeout config |
| `tests/core/research/workflows/test_deep_research.py` | Test zero-source guard behavior |
| `tests/core/research/workflows/test_timeout_resilience.py` | Test timeout retry cap |

## Risks & Mitigations

- **Risk:** Reducing timeout retries could cause failures on transiently slow providers.
  **Mitigation:** `max_timeout_retries=0` only affects timeout errors; transient API
  errors (500, rate limit) still get full `max_retries`. If a provider is genuinely slow
  (not broken), the higher `topic_research_timeout` (90s vs 60s) gives it more room.

- **Risk:** Zero-source disclaimer could be confusing if sources were expected.
  **Mitigation:** The disclaimer is factual and actionable — the user knows to retry
  or check provider configuration.

- **Risk:** Separating topic research timeout from reflection timeout adds config surface.
  **Mitigation:** The new config has a sensible default (90s) and only needs adjustment
  in edge cases. Existing `deep_research_reflection_timeout` remains unchanged.
