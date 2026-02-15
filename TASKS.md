# Branch Review Follow-ups

## Progress (2026-02-15)

- Completed: all merge blockers (1-4) are implemented and validated in tests.
- Completed: non-blocking improvement #1 (audit role source unified with authorization role).
- Remaining: non-blocking improvements #2-#3 and research tracks.

## Merge blockers (fix before merge)

1. Wire role initialization at process startup.
   - Problem: `initialize_role_from_config()` is defined but never called, so runtime role stays default `observer`.
   - Evidence: `src/foundry_mcp/core/authorization.py:300`, `src/foundry_mcp/server.py:203`, `src/foundry_mcp/tools/unified/common.py:201`.
   - Impact: non-read actions can be denied unexpectedly; hardening role config (`autonomy_security.role` / `FOUNDRY_MCP_ROLE`) is effectively inert.
   - Suggestion: call `initialize_role_from_config(config.autonomy_security.role)` during server boot (and CLI boot if needed), then add an integration test that verifies a configured `maintainer` role can execute mutation actions.

2. Reconcile schema-version bump with test/contracts.
   - Problem: state schema moved to v3, but existing migration and persistence tests still assert v2.
   - Evidence: `src/foundry_mcp/core/autonomy/state_migrations.py:39`, `src/foundry_mcp/core/autonomy/models.py:554`, failing tests in `tests/unit/test_core/autonomy/test_context_tracker.py:614` and `tests/unit/test_core/autonomy/test_memory.py:173`.
   - Impact: branch is red (`11 failed, 395 passed` in targeted autonomy/core run).
   - Suggestion: update affected tests/fixtures/spec docs to v3 expectations, and add explicit v2->v3 migration assertions where behavior changed.

3. Land contract-change updates for `session-end` and `session-reset` reason codes.
   - Problem: handlers now require `reason_code`, but existing handler/integration tests and likely callers still use old shape.
   - Evidence: `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py:1084`, `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py:1641`, failures in `tests/unit/test_core/autonomy/test_handlers_session.py:651` and `tests/unit/test_core/autonomy/test_integration.py:166`.
   - Impact: breaking behavior change without full downstream migration.
   - Suggestion: update tests/callers to pass `reason_code`, and document this as a versioned contract change in specs/changelog/capabilities notes.

4. Align write-lock bypass semantics/tests with new role gate.
   - Problem: bypass now denies non-maintainer before config checks, but tests still expect previous status/error precedence.
   - Evidence: `src/foundry_mcp/core/autonomy/write_lock.py:426`, failures in `tests/unit/test_core/autonomy/test_write_lock.py:150`, `:193`, `:212`.
   - Impact: regression in test suite and unclear external error expectations.
   - Suggestion: decide intended precedence (`role` first vs `config` first), codify it in tests, and add explicit role setup in bypass tests.

## Important non-blocking improvements

1. Unify audit role source with authorization role.
   - Problem: audit ledger uses `FOUNDRY_SERVER_ROLE` fallback, while authorization uses `FOUNDRY_MCP_ROLE` + context variable.
   - Evidence: `src/foundry_mcp/core/autonomy/audit.py:183`, `src/foundry_mcp/core/authorization.py:315`.
   - Suggestion: source audit role from `get_server_role()` (or the same initialization path) to avoid role attribution drift.

2. Strengthen rebase gate-reconciliation mapping.
   - Problem: phase impact from added tasks uses string heuristics and includes placeholder logic.
   - Evidence: `src/foundry_mcp/tools/unified/task_handlers/handlers_session.py:430`.
   - Suggestion: derive task->phase mapping from parsed spec structures instead of ID-pattern heuristics.

3. Replace placeholder integration tests with assertive coverage.
   - Problem: fallback integration tests allow broad pass conditions and include no-op assertions.
   - Evidence: `tests/integration/test_fallback_integration.py:31`, `:45`, `:63`.
   - Suggestion: add deterministic mocks/fixtures and assert actual fallback order, retry behavior, and caps.

## Research tracks

1. Backward-compat rollout strategy for autonomy hardening changes.
   - Questions:
     - Should `reason_code` become conditionally required via schema version/capability flag first?
     - Do we need temporary compatibility mode for older clients?
   - Deliverable: migration plan with cutover date and compatibility window.

2. Authorization model for non-stdio / multi-tenant futures.
   - Questions:
     - Is process-wide role sufficient, or should role be request-scoped with signed identity?
     - How should rate limiting and audit attribution change under per-request identity?
   - Deliverable: ADR addendum with threat model and operational recommendations.

3. Gate invariant durability under complex rebases.
   - Questions:
     - What guarantees are needed when phases/tasks are renamed or moved?
     - Should structural diff return explicit task->phase deltas?
   - Deliverable: data-model proposal + test matrix for rename/move/add/remove scenarios.

## Validation command used during review

`pytest -q tests/unit/test_core/autonomy tests/unit/test_core/test_authorization.py tests/tools/unified/test_common.py tests/integration/test_fallback_integration.py --maxfail=0`
