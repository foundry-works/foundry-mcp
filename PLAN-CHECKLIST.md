# Deep Research Post-Synthesis Fixes — Checklist

## Bug 1: Status Field Never Set in Completed Session Metadata

- [x] In `action_handlers.py` `_get_status()` completed-session path (~line 505), add `"status"` key derived from `is_complete`/`is_failed`/`timed_out`/`cancelled` booleans
- [x] In `action_handlers.py` active-task path (~line 352), add `"status"` key (kept `"task_status"` for back-compat)
- [ ] Add/update test for `_get_status()` return value to assert `"status"` key exists with correct values for completed, failed, cancelled, timed_out sessions
- [x] Verify `handlers_deep_research.py:202` correctly reads the new `"status"` field (no change needed)

## Bug 2: `input_tokens` Reports CLI Message Count Instead of Token Count

- [x] In `_claude_base.py` `_extract_usage()`, add `modelUsage` extraction path that sums token counts across models
- [x] Keep `usage` fallback for non-CLI providers
- [x] Handle `cache_read_input_tokens` (CLI key) vs `cached_input_tokens` (API key) correctly
- [x] Add test case with `modelUsage` payload to `test_providers_implementations.py`
- [x] Add test case verifying fallback to `usage` when `modelUsage` is absent
- [x] Verify `output_tokens` extraction is also correct from `modelUsage` (yes — `modelUsage` includes it)

## Integration / Smoke

- [x] Run existing deep research test suite — 1,465 passed
- [x] Run provider implementation tests — 55 passed
- [x] Run contract tests — 292 passed
