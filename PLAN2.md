# Code Review Remediation Plan

**Date:** 2026-02-22
**Scope:** Fixes and test coverage for issues identified in senior engineering review of branch `tyler/foundry-mcp-20260222-1744`
**Reference:** Code review covering autonomy subsystem, deep research improvements, module refactoring, and test quality

---

## Executive Summary

A senior engineering review of the branch identified 8 code-level issues (bugs, security gaps, and code smells) and 3 test coverage gaps (untested security-critical modules). All items are independently shippable. No architectural changes are required — fixes are targeted edits to existing code.

---

## 1. Falsy Zero Bug in Topic Research

**Problem:** `max_sources_per_provider or state.max_sources_per_query` treats an explicit `0` as falsy, silently falling back to the state default.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` (line 227)

**Fix:** Replace `or` with explicit `None` check:
```python
# Before
effective_max_results = max_sources_per_provider or state.max_sources_per_query

# After
effective_max_results = (
    max_sources_per_provider
    if max_sources_per_provider is not None
    else state.max_sources_per_query
)
```

---

## 2. Bool-to-String Inconsistency in Clarification

**Problem:** `str(True)` produces `"True"` instead of JSON-conventional `"true"`. Additionally, the `if v` filter excludes `False` and `0`, which are valid constraint values.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/clarification.py` (line 248-250)

**Fix:**
```python
# Before
result["inferred_constraints"] = {
    k: str(v) for k, v in constraints.items() if v and isinstance(v, (str, int, float, bool))
}

# After
result["inferred_constraints"] = {
    k: (str(v).lower() if isinstance(v, bool) else str(v))
    for k, v in constraints.items()
    if v is not None and v != "" and isinstance(v, (str, int, float, bool))
}
```

---

## 3. Authorization Fail-Open on Invalid Role

**Problem:** `set_server_role("garbage")` falls back to `Role.MAINTAINER` (full mutation access). Security best practice is fail-closed.

**File:** `src/foundry_mcp/core/authorization.py` (lines 164-169)

**Fix:** Fall back to `Role.OBSERVER` (read-only) instead of `Role.MAINTAINER`:
```python
# Before
role = Role.MAINTAINER.value

# After
role = Role.OBSERVER.value
```

Update the log message to reflect the change.

---

## 4. Spec Cache Thread Safety

**Problem:** `_spec_cache` in `StepOrchestrator` is a plain instance attribute with no lock. Concurrent threads calling `compute_next_step` can interleave reads and writes.

**File:** `src/foundry_mcp/core/autonomy/orchestrator.py`

**Fix:**
1. Add `import threading`
2. Add `self._spec_cache_lock = threading.Lock()` in `__init__`
3. Wrap all `_spec_cache` read/write sites (in `invalidate_spec_cache` and spec load paths) with `with self._spec_cache_lock:`

---

## 5. Integrity Checksum Versioning

**Problem:** HMAC payload `f"{gate_attempt_id}:{step_id}:{phase_id}:{verdict}"` has no version field, making future algorithm rotation impossible without breaking all in-flight evidence.

**File:** `src/foundry_mcp/core/autonomy/server_secret.py`

**Fix:**
1. Prefix payload with `v1:` in `compute_integrity_checksum`
2. Add legacy fallback in `verify_integrity_checksum` that accepts unversioned checksums during migration

---

## 6. Hard-Coded Stale Task Timeout

**Problem:** `bg_task.is_stale(300.0)` is hardcoded. Operators cannot tune staleness thresholds.

**Files:**
- `src/foundry_mcp/config/research.py` — Add `deep_research_stale_task_seconds: float = 300.0`
- `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py` — Use config value

---

## 7. Provider Fallback Chain Duplication

**Problem:** Provider resolution via chained `getattr(config, ...) or ...` pattern is duplicated in `topic_research.py` and `clarification.py`.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/_helpers.py` — Add `resolve_phase_provider(config, *phase_names)` helper
- `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py` — Use helper
- `src/foundry_mcp/core/research/workflows/deep_research/phases/clarification.py` — Use helper (if applicable)

---

## 8. Guard Script Tests (New)

**Problem:** `scripts/guard_autonomous_bash.py` and `scripts/guard_autonomous_write.py` are security-critical boundary scripts with zero test coverage.

**New file:** `tests/unit/test_guard_scripts.py`

**Coverage:**
- Git read commands allowed, write commands blocked
- Shell write patterns to protected paths blocked
- Environment variable bypasses (FOUNDRY_GUARD_DISABLED, FOUNDRY_GUARD_ALLOW_GIT_COMMIT)
- Edge cases (flags before subcommand, unknown commands)
- Write guard: spec JSON, config TOML, session state, journal paths blocked
- Write guard: normal paths allowed
- EXTRA_BLOCKED / EXTRA_ALLOWED env var precedence

---

## 9. Server Secret Tests (New)

**Problem:** `server_secret.py` provides HMAC integrity for gate evidence but has no dedicated tests.

**New file:** `tests/unit/test_core/autonomy/test_server_secret.py`

**Coverage:**
- Secret generation, caching, and file permissions (0o600)
- Environment variable override (`FOUNDRY_MCP_GATE_SECRET`)
- Cache invalidation via `clear_secret_cache()`
- Checksum determinism and different-input divergence
- Versioned payload verification (after fix 5)
- Legacy checksum backward compatibility
- Delimiter collision resistance
- Empty/invalid checksum rejection

---

## 10. Orchestrator Concurrency Tests (New)

**Problem:** No tests exercise concurrent `compute_next_step` calls or spec cache invalidation under contention.

**File:** `tests/unit/test_core/autonomy/test_concurrency_scale.py` (extend existing)

**Coverage:**
- Multiple threads calling `compute_next_step` with shared spec cache
- Spec cache invalidation during concurrent reads
- No errors or state corruption under contention

---

## 11. Write Lock TOCTOU Documentation

**Problem:** `_find_active_session_for_spec()` calls `get_active_session()` then `load()` — a redundant double-load. The TOCTOU is already mitigated by the storage layer's internal verification.

**File:** `src/foundry_mcp/core/autonomy/write_lock.py`

**Fix:** Add clarifying comment documenting that the double-load is benign and the TOCTOU is mitigated by the storage layer. No code change needed.

---

## Implementation Order

| Phase | Items | Dependencies |
|-------|-------|-------------|
| 1 | Fixes 1-3 (small bug fixes) | None |
| 2 | Fixes 4-5 (security hardening) | Fix 5 before test 9 |
| 3 | Fixes 6-7 (config + helper cleanup) | None |
| 4 | Tests 8-9 (new test files) | Fix 5 for test 9 |
| 5 | Test 10 + Fix 11 (concurrency + docs) | Fix 4 for test 10 |
