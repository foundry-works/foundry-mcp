# Code Review Remediation — Implementation Checklist

Tracks progress against [PLAN2.md](./PLAN2.md). Check items as completed.

---

## Phase 1: Bug Fixes

### 1. Falsy Zero Bug
- [ ] Change `or` to `is not None` check in `topic_research.py` line 227
- [ ] Verify existing topic research tests still pass

### 2. Bool-to-String Inconsistency
- [ ] Fix `str(v)` to use `str(v).lower()` for bools in `clarification.py` line 249
- [ ] Fix falsy filter (`if v` → `if v is not None and v != ""`) to preserve `False` and `0`
- [ ] Verify existing clarification tests still pass

### 3. Authorization Fail-Closed
- [ ] Change fallback from `Role.MAINTAINER` to `Role.OBSERVER` in `authorization.py` line 169
- [ ] Update log message to say "falling back to observer"
- [ ] Verify existing authorization tests still pass

---

## Phase 2: Security Hardening

### 4. Spec Cache Thread Safety
- [ ] Add `import threading` to `orchestrator.py`
- [ ] Add `self._spec_cache_lock = threading.Lock()` in `__init__`
- [ ] Wrap `invalidate_spec_cache()` body with lock
- [ ] Wrap all spec cache read/write sites in `_load_spec_data` (or equivalent) with lock
- [ ] Verify existing orchestrator tests still pass

### 5. Integrity Checksum Versioning
- [ ] Add `v1:` prefix to payload in `compute_integrity_checksum()` in `server_secret.py`
- [ ] Add legacy fallback path in `verify_integrity_checksum()` for unversioned checksums
- [ ] Add comment documenting migration window for legacy fallback removal

---

## Phase 3: Config and Helper Cleanup

### 6. Configurable Stale Task Timeout
- [ ] Add `deep_research_stale_task_seconds: float = 300.0` to `ResearchConfig` in `config/research.py`
- [ ] Add field to `from_toml_dict()` parser
- [ ] Replace hardcoded `300.0` with `self.config.deep_research_stale_task_seconds` in `handlers_deep_research.py`

### 7. Provider Fallback Chain Helper
- [ ] Add `resolve_phase_provider(config, *phase_names) -> str` to `_helpers.py`
- [ ] Replace manual fallback chain in `topic_research.py` with helper call
- [ ] Replace manual fallback chain in `clarification.py` with helper call (if applicable)
- [ ] Verify existing tests pass

---

## Phase 4: New Test Files

### 8. Guard Script Tests
- [ ] Create `tests/unit/test_guard_scripts.py`
- [ ] `TestBashGuardReadOnly`: parametrized test for all git read commands (status, diff, log, show, branch, etc.)
- [ ] `TestBashGuardWriteBlocked`: parametrized test for all git write commands (commit, push, reset, rebase, etc.)
- [ ] `TestBashGuardBypass`: FOUNDRY_GUARD_DISABLED, FOUNDRY_GUARD_ALLOW_GIT_COMMIT env vars
- [ ] `TestBashGuardEdgeCases`: unknown commands, flags before subcommand, shell write patterns
- [ ] `TestWriteGuardBlocked`: spec JSON, config TOML, session state, journal, audit, proof paths
- [ ] `TestWriteGuardAllowed`: normal source files, test files, temp paths
- [ ] `TestWriteGuardBypass`: FOUNDRY_GUARD_DISABLED, EXTRA_BLOCKED, EXTRA_ALLOWED
- [ ] All tests pass with `pytest tests/unit/test_guard_scripts.py -v`

### 9. Server Secret Tests
- [ ] Create `tests/unit/test_core/autonomy/test_server_secret.py`
- [ ] `TestGetServerSecret`: generation, caching, file permissions (0o600), env var override, cache invalidation
- [ ] `TestComputeIntegrityChecksum`: determinism, input divergence, versioned payload verification
- [ ] `TestVerifyIntegrityChecksum`: valid accepted, invalid rejected, empty rejected, legacy fallback, delimiter collision resistance
- [ ] All tests pass with `pytest tests/unit/test_core/autonomy/test_server_secret.py -v`

---

## Phase 5: Concurrency Tests and Documentation

### 10. Orchestrator Concurrency Tests
- [ ] Add `TestSpecCacheConcurrency` class to `tests/unit/test_core/autonomy/test_concurrency_scale.py`
- [ ] `test_concurrent_compute_next_step_shared_cache`: 8 threads, no errors
- [ ] `test_invalidate_during_concurrent_reads`: invalidation loop + 4 reader threads, no crashes
- [ ] All tests pass with `pytest tests/unit/test_core/autonomy/test_concurrency_scale.py -v`

### 11. Write Lock TOCTOU Documentation
- [ ] Add clarifying comment in `_find_active_session_for_spec()` in `write_lock.py`
- [ ] Document that double-load is benign and TOCTOU is mitigated by storage layer

---

## Verification

- [ ] Run full test suite: `pytest tests/ -x --maxfail=5`
- [ ] Run lint: `ruff check src/ tests/`
- [ ] Verify no import errors: `python -c "from foundry_mcp.core.autonomy.orchestrator import StepOrchestrator"`
- [ ] Verify no regressions in existing tests
