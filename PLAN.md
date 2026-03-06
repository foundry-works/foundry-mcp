# Deep Research Post-Synthesis Session Review — Bug Fixes

**Session analyzed:** `deepres-23798130b3a3` (credit card research, 47.6 min, 198K tokens)
**Branch:** `alpha`

---

## Bug 1: `status` Field Never Set in Completed Session Metadata

### Problem

When `_get_status()` returns metadata for a completed deep research session, it sets `is_complete: True` but never includes a `"status"` key. The unified handler at `research_handlers/handlers_deep_research.py:202` reads `status_data.get("status", "unknown")`, so every completed session reports `status: "unknown"` instead of `"completed"`.

The active-task path (line 352-354 in `action_handlers.py`) returns `"task_status"` — also the wrong key name.

### Root Cause

- `action_handlers.py:505-544` — completed session path returns `is_complete`, `is_failed`, `timed_out`, `cancelled` as separate booleans but never synthesizes them into a single `"status"` string.
- `action_handlers.py:352` — active task path uses key `"task_status"` instead of `"status"`.
- `handlers_deep_research.py:202` — handler expects `"status"` key.

### Fix

In `_get_status()` completed-session path (`action_handlers.py`), derive and include a `"status"` field:
- If `state.metadata.get("cancelled")` → `"cancelled"`
- If `state.metadata.get("timeout")` → `"timed_out"`
- If `is_failed` → `"failed"`
- If `state.completed_at is not None` → `"completed"`
- Else → `"in_progress"`

Also fix the active-task path to use `"status"` instead of `"task_status"`.

### Files

- `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py` (lines 352, 505-544)
- `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py` (line 202, consumer — no change needed)

---

## Bug 2: `input_tokens` Always Reports 3 (CLI Message Count, Not Token Count)

### Problem

Every phase metric records `input_tokens=3` regardless of actual prompt size. Output tokens vary correctly. Cached tokens are always 0.

### Root Cause

The Claude CLI `--output-format json` response includes a top-level `usage` object where `input_tokens` represents the **number of conversation messages/turns**, not token counts. The actual token counts live under `modelUsage.{model_name}.input_tokens`.

`_claude_base.py:292-299` (`_extract_usage`) reads `payload["usage"]["input_tokens"]` — getting the message count (3 = system + user + assistant messages).

Line 357 already extracts `modelUsage` to identify the model, but its token data is never used.

### Fix

Update `_extract_usage` in `_claude_base.py` to prefer `modelUsage` token counts when available, falling back to `usage` for non-CLI callers:

```python
def _extract_usage(self, payload: Dict[str, Any]) -> TokenUsage:
    # Claude CLI puts real token counts in modelUsage.{model}.{field}
    # The top-level "usage" in CLI output contains message counts, not tokens.
    model_usage = payload.get("modelUsage") or {}
    if model_usage:
        # Sum across all models (usually just one)
        input_t = sum(v.get("input_tokens", 0) for v in model_usage.values())
        output_t = sum(v.get("output_tokens", 0) for v in model_usage.values())
        cached_t = sum(v.get("cache_read_input_tokens", 0) for v in model_usage.values())
        return TokenUsage(
            input_tokens=input_t,
            output_tokens=output_t,
            cached_input_tokens=cached_t,
            total_tokens=input_t + output_t,
        )
    # Fallback for non-CLI providers or older formats
    usage = payload.get("usage") or {}
    return TokenUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
        total_tokens=int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0),
    )
```

Note: The `modelUsage` field uses `cache_read_input_tokens` (CLI convention) while the `usage` fallback uses `cached_input_tokens` (API convention).

### Files

- `src/foundry_mcp/core/providers/_claude_base.py` (lines 292-299)
- `tests/unit/test_providers_implementations.py` (update fixtures to include `modelUsage` test cases)

---

## Implementation Order

1. **Bug 1** (status field) — Simplest fix, highest user-facing impact, no risk
2. **Bug 2** (input_tokens) — Straightforward provider fix, improves all session metrics
