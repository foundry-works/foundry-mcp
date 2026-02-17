# Codebase Refactoring Research

> Senior engineering review of foundry-mcp (~27K LOC source, ~71K LOC tests)
> Date: 2026-02-16

---

## 1. Declarative Parameter Validation Framework — DONE

### Status: Waves 1-5 complete (69 handlers migrated)

**Framework built** (`tools/unified/param_schema.py`, ~270 lines):
- 6 schema types: `Str`, `Num`, `Bool`, `List_`, `Dict_`, `AtLeastOne`
- `validate_payload()` engine with in-place normalization
- Error envelopes identical to `make_validation_error_fn()` output
- 72 unit tests passing (`tests/tools/unified/test_param_schema.py`)

**Wave 1 migrated** (3 handlers):
- `handlers_phase.py::_handle_phase_add` — migrated
- `handlers_mutation.py::_handle_add` — migrated (research-specific conditional validation kept imperative)
- `handlers_spec.py::_handle_spec_create` — migrated

**Wave 2 migrated** (13 handlers, 874 tests passing):
- `handlers_phase.py` — 4 handlers (`phase_update_metadata` w/ AtLeastOne, `phase_add_bulk` top-level only, `phase_move`, `phase_remove`); `phase_template` kept imperative (branching on template_action)
- `handlers_spec.py` — 3 handlers (`spec_update_frontmatter`, `spec_find_replace`, `spec_rollback`); `spec_template` kept imperative
- `handlers_metadata.py` — 3 handlers (`assumption_add`, `assumption_list`, `revision_add`)
- `handlers_intake.py` — 3 handlers (partial: `intake_add`, `intake_list`, `intake_dismiss`)

**Wave 3 migrated** (28 task handlers, ~110 `_validation_error` calls eliminated):
- `handlers_lifecycle.py` — 6 handlers (19 calls eliminated)
- `handlers_query.py` — 8 handlers (19 calls eliminated); `_handle_session_config` skipped (0 calls)
- `handlers_session_rebase.py` — 1 handler (`gate_waiver`, 3 calls); `_handle_session_rebase` unchanged
- `handlers_session_step.py` — 1 handler (`session_step_report`, 4 calls); uses `AtLeastOne` cross-field rule
- `handlers_batch.py` — 6 handlers (6 schemas, imperative remainders for per-element loops and AtLeastOne)
- `handlers_mutation.py` — 7 handlers (6 schemas, imperative remainders for AtLeastOne and conditional research validation)
- `handlers_session_lifecycle.py` — 3 handlers (`session_start`, `session_end`, `session_reset`; 6 calls eliminated); `_resolve_session_config` stays imperative (derived value validation)

**Known framework limitation**: Schema supports single `error_code` per field. Fields needing `MISSING_REQUIRED` for absence but `INVALID_FORMAT` for type errors must stay imperative (e.g., `phase_move.position`). Fields with strip-or-None semantics (`description.strip() or None`) also stay imperative since schema strips but doesn't convert empty to None.

**Wave 4 migrated** (10 handlers, ~40 `_validation_error` calls eliminated):
- `lifecycle.py` — 5 handlers (move, activate, complete, archive, state); shared `_SPEC_PATH_SCHEMA` base with per-handler extensions
- `verification.py` — 2 handlers (add, execute); `result` field uses `choices` for PASSED/FAILED/PARTIAL
- `provider.py` — 3 handlers (list, status, execute); `_handle_execute` was largest (11 imperative calls → 1 schema)

**Wave 5 migrated** (15 research handlers, ~21 `_validation_error`/`error_response` validation calls eliminated):
- `handlers_threads.py` — 3 handlers (thread-list w/ enum choices, thread-get, thread-delete)
- `handlers_workflows.py` — 4 handlers (chat, consensus w/ enum choices, thinkdeep w/ AtLeastOne, ideate w/ AtLeastOne)
- `handlers_deep_research.py` — 3 handlers (status, report, delete); `_handle_deep_research` kept imperative (2 action-conditional checks)
- `handlers_spec_nodes.py` — 4 handlers (execute, record w/ choices, status, findings); `_handle_node_execute` kept imperative (2 runtime-dependent checks: query metadata fallback, research_type validation)
- `handlers_extract.py` — 1 handler (extract w/ List_); API key config check and exception routing kept imperative

**Imperative remainders** (4 calls across 2 handlers — irreducibly runtime-dependent):
- `_handle_deep_research`: 2 calls (query required when action=="start", research_id required when action=="continue")
- `_handle_node_execute`: 2 calls (query from param or metadata fallback, unsupported research_type)

### Final Tally

| Metric | Before | After |
|--------|--------|-------|
| Handlers with imperative validation | 69 | 4 (runtime-dependent remainders) |
| `_validation_error()` calls | ~370 | 4 |
| Distinct parameter names | 85+ | — |
| Files with validation boilerplate | 19 | 2 (deep_research, spec_nodes) |
| Test suite | 5,201 passing | 5,201 passing (0 regressions) |

### Impact

- ~15-20% codebase reduction in handler files
- Eliminates entire class of copy-paste validation bugs
- Makes adding new handlers trivial (schema only, no validation code)
- All 5 waves completed across authoring, task, lifecycle, provider, verification, and research handler families

---

## 2. God Object Decomposition — DONE

### Status: Waves 1-4 complete (2026-02-16)

Three core modules decomposed into sub-module packages with backward-compatible re-exports.

**Wave 1** — Decomposed `core/validation.py` (2,342 lines) → `core/validation/` package (10 sub-modules: constants, models, normalization, input, rules, fixes, application, stats, verification, `__init__`). 62 unit + 34 property + 241 integration tests passing.

**Wave 2** — Decomposed `core/ai_consultation.py` (1,774 lines) → `core/ai_consultation/` package (4 sub-modules: types, cache, orchestrator, `__init__`). 3,078 unit + 241 integration tests passing.

**Wave 3** — Decomposed `core/observability.py` (1,218 lines) → `core/observability/` package (6 sub-modules: manager, metrics, audit, redaction, decorators, `__init__`). 34 prod importers verified via re-exports.

**Wave 4** — Migrated 12 caller files (10 prod + 2 test) to canonical sub-module import paths for `validation/`. All callers now import from specific sub-modules (e.g., `validation.rules`, `validation.constants`) instead of `validation.__init__`. 3,353 tests passing, 42 skipped. `__init__.py` re-exports preserved for third-party consumers.

**Remaining**: Waves 5-6 (caller import migration for ai_consultation + observability, cleanup + validation dispatch dict) are optional follow-ups.

---

## 3. Unified Error Hierarchy — DONE

### Status: All 3 waves complete (2026-02-16)

**Wave 1** — Created `core/errors/` package with 10 modules, moved all 37 error classes, added backward-compatible re-exports in 17 original source files.

**Wave 2** — Migrated ~28 caller import sites to canonical `core/errors/` paths across tool layer, core layer, and research providers.

**Wave 3** — Populated `ERROR_MAPPINGS` registry with 24 error→(ErrorCode, ErrorType) mappings. `error_to_response()` helper available for future adoption. Existing handlers retain their custom enrichments (remediation, details, metrics).

**Package structure**:
```
core/errors/
    __init__.py          # Re-exports all errors + registry
    base.py              # ERROR_MAPPINGS (24 entries) + error_to_response()
    llm.py               # 6 LLM error classes
    provider.py          # 6 provider error classes
    search.py            # 3 search provider error classes
    research.py          # 8 research error classes
    resilience.py        # 4 resilience error classes
    storage.py           # 5 storage error classes (MigrationError consolidated)
    authorization.py     # 2 authorization error classes
    execution.py         # 3 execution error classes
```

---

## 4. Research Tool Module Decomposition — DONE

### Status: Waves 1-4 complete (2026-02-16)

**Wave 1** — Created `research_handlers/` package scaffold with `_helpers.py` (shared state, validation factory, constants) and `__init__.py` (router, dispatch, registration).

**Wave 2** — Extracted all 17 handlers into 5 domain-focused modules:
- `handlers_workflows.py` — chat, consensus, thinkdeep, ideate
- `handlers_deep_research.py` — deep-research, -status, -report, -list, -delete
- `handlers_threads.py` — thread-list, thread-get, thread-delete
- `handlers_spec_nodes.py` — node-execute, node-record, node-status, node-findings
- `handlers_extract.py` — extract

**Wave 3** — Updated test patch paths in `test_research.py` and `test_deep_research.py`. Converted `research.py` to backward-compatible shim. Migrated `_validation_error` to `make_validation_error_fn("research")` factory.

**Wave 4** — Simplified registration body: replaced 40-line manual parameter passthrough with `return _dispatch_research_action(**locals())`. Fixed stale patch paths in `test_research_e2e.py` (18 tests). Note: full `**kwargs` signature not viable — FastMCP derives MCP schema from function signature. 5183 tests passing.

---

## 5. Tool Registration Signature Bloat — DONE

**Finding**: `**kwargs` passthrough is **not viable** — FastMCP generates MCP tool parameter schemas by inspecting function signatures. Removing explicit params produces a broken schema. The viable approach is `locals()` passthrough in the function body, which eliminates the manual dict-packing while preserving the schema.

**Research tool** — Done (Wave 4 above). Body reduced from 40 lines to `return _dispatch_research_action(**locals())`.

**Task/Authoring tools** — Done. Applied `locals()` filter pattern:
- `task_handlers/__init__.py` — replaced ~83-line manual payload dict with 1-line `locals()` filter
- `authoring_handlers/__init__.py` — replaced ~40-line manual payload dict with 1-line `locals()` filter
- Pattern: `payload = {k: v for k, v in locals().items() if k not in ("action", "config")}`

---

## 6. Observability/Metrics Consolidation — DONE

### Status: All 4 waves complete (2026-02-17)

Consolidated 6 standalone `core/` files into two well-bounded packages (`core/metrics/` and `core/observability/`). Zero test failures (5,183 tests verified).

**Wave 1** — Created `core/metrics/` package. Moved `metrics_store.py` → `metrics/store.py`, `metrics_persistence.py` → `metrics/persistence.py`, `metrics_registry.py` → `metrics/registry.py`. Created `__init__.py` with re-exports.

**Wave 2** — Moved `prometheus.py` → `observability/prometheus.py`, `otel.py` → `observability/otel.py`, `otel_stubs.py` → `observability/stubs.py`. Updated internal cross-references.

**Wave 3** — Migrated ~11 production import sites and ~6 test import sites to canonical paths. Full test suite green.

**Wave 4** — Verified zero remaining old-path imports. Added deprecation warnings to all 6 shim files at old locations. Shims preserved for third-party consumers.

### Previous Problem

Metrics and observability logic was spread across **7 files totaling 5,000+ lines** with unclear boundaries.

### Final Structure

```
core/metrics/          # store.py, persistence.py, registry.py, __init__.py
core/observability/    # manager.py, metrics.py, audit.py, decorators.py, redaction.py,
                       # prometheus.py, otel.py, stubs.py, __init__.py
```

---

## 7. Test Fixture Consolidation — DONE

### Status: All 4 waves complete (2026-02-17)

Created 5 new `conftest.py` files to centralize duplicate fixtures across the test suite. Zero test failures, zero test count changes (5,083 tests verified).

**Wave 1** — Created `tests/tools/unified/conftest.py` with shared `mock_config` fixture. Removed from 11 files; `test_dispatch_common.py` overrides to set `specs_dir=None`; `test_research.py` kept local (yield-based with module patching).

**Wave 2** — Created `tests/integration/conftest.py` with `test_specs_dir`, `test_config`, `mcp_server`. Fully removed from `test_provider_tools.py` and `test_json_minification.py`. Partial removal from 4 files that keep local `test_specs_dir` overrides (richer spec data, templates, role overrides).

**Wave 3** — Created `tests/unit/test_core/conftest.py` and `tests/unit/test_contracts/conftest.py` with `temp_specs_dir`. Removed from 6 test_core files and 4 test_contracts files. `test_spec_history.py` and `test_phase6_contracts.py` kept local (need `.backups` dir).

**Wave 4** — Created `tests/core/research/workflows/conftest.py` with `mock_config` and `mock_memory`. 4 workflow test files now extend the base fixtures with workflow-specific attributes. `test_deep_research.py` kept local (15+ attrs + helper methods).

### Remaining (out of scope, separate efforts)

- Splitting monolithic test files (test_deep_research.py, test_document_digest.py, etc.)
- CLI layer test coverage gaps
- `sample_spec` consolidation (too many intentional variations)

---

## 8. Deep Research Phase Execution Framework — DONE

### Status: Waves 1-5 complete (2026-02-17)

Extracted shared LLM call lifecycle boilerplate from 4 phase mixins (planning, analysis, synthesis, refinement) into `phases/_lifecycle.py`. Then extracted shared orchestrator dispatch boilerplate from `core.py` into `_run_phase()`. Gathering excluded from lifecycle helpers (uses search providers, not LLM calls).

**Wave 1** — Created `phases/_lifecycle.py` (~100 lines) with `LLMCallResult` dataclass, `execute_llm_call()` async helper, and `finalize_phase()` helper. 18 unit tests in `test_phase_lifecycle.py`.

**Wave 2** — Migrated `planning.py` and `refinement.py` to use lifecycle helpers. ~160 lines of boilerplate removed.

**Wave 3** — Migrated `synthesis.py` to use lifecycle helpers. ~80 lines of boilerplate removed.

**Wave 4** — Migrated `analysis.py` to use lifecycle helpers. ~80 lines of boilerplate removed.

**Wave 5** — Extracted `_run_phase()` helper in `core.py` (~30 lines) encapsulating the 9-step phase dispatch lifecycle (cancel → timer → hooks → audit → execute → error → hooks → audit → transition). Migrated all 5 phase blocks in `_execute_workflow_async`:
- PLANNING, ANALYSIS — clean migration (standard lifecycle)
- GATHERING — `iteration_in_progress` pre-hook, extract followup post-hook
- SYNTHESIS — `skip_transition=True`, custom orchestrator logic inline
- REFINEMENT — `skip_error_check=True, skip_transition=True`, iteration tracking + recursion inline

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Phase mixin boilerplate (Waves 1-4) | ~350 lines | ~0 |
| Orchestrator dispatch boilerplate (Wave 5) | ~250 lines (5 copies) | ~30 lines (helper) + ~55 lines (calls) |
| `core.py` line count | 1,629 | 1,595 |
| Copy-paste bug surface | 4 phase copies + 5 dispatch copies | 2 sources of truth |
| New helpers | 0 | 2 (`_lifecycle.py`, `_run_phase()`) |
| New tests | 0 | 18 (`test_phase_lifecycle.py`) + 4 (`_run_phase` unit tests) |

130 tests passing (95 primary + 35 secondary), 0 regressions.

---

## 9. Dispatch Dictionary for Validation Fixes — DONE

Replaced 12 sequential `if code ==` statements in `core/validation/fixes.py` with two dispatch dicts split by argument shape:
- `_SPEC_DATA_HANDLERS` — handlers taking `(diag, spec_data)` (1 entry)
- `_HIERARCHY_HANDLERS` — handlers taking `(diag, hierarchy)` (11 entries)
- `_COUNTS_CODES` frozenset — 4 codes routing to `_build_counts_fix(diag, spec_data)`

`_build_fix_action` body reduced from 30+ lines to 10 lines (3 dict lookups + 1 set membership test).

---

## 10. Config Module Decomposition — DONE

### Status: All 4 waves complete (2026-02-17)

Decomposed `config.py` (2,771 lines, 24 dataclasses — the largest god object in the codebase) into a `config/` package with 7 focused sub-modules.

**Wave 1** — Extracted `ResearchConfig` (974 lines) → `config/research.py`. Extracted parsing helpers → `config/parsing.py` (163 lines). Created `config/__init__.py` with re-exports. Remaining symbols in `config/_legacy.py`.

**Wave 2** — Extracted domain configs → `config/domains.py` (348 lines: `GitSettings`, `ObservabilityConfig`, `HealthConfig`, `ErrorCollectionConfig`, `MetricsPersistenceConfig`, `RunnerConfig`, `TestConfig`). Extracted autonomy configs → `config/autonomy.py` (142 lines: `AutonomySecurityConfig`, `AutonomySessionDefaultsConfig`, `AutonomyPostureConfig`, posture constants).

**Wave 3** — Split `ServerConfig` into `config/server.py` (~190 lines, class definition + globals) and `config/loader.py` (~560 lines, `_ServerConfigLoader` mixin with `from_env`, `_load_toml`, `_load_env`). Extracted decorators → `config/decorators.py` (~120 lines). Deleted `_legacy.py`.

**Wave 4** — Migrated ~60 caller import sites across `src/` and `tests/` to canonical sub-module paths (`config.server`, `config.research`, `config.domains`). Re-exports in `__init__.py` retained for external consumers.

### Final Structure

```
config/
    __init__.py       # Re-exports all public symbols (70 lines)
    server.py         # ServerConfig + get_config/set_config (~190 lines)
    loader.py         # _ServerConfigLoader mixin (~560 lines)
    research.py       # ResearchConfig (974 lines)
    autonomy.py       # Autonomy configs + posture constants (142 lines)
    domains.py        # 7 domain config dataclasses (348 lines)
    parsing.py        # Parse/normalize helpers (163 lines)
    decorators.py     # log_call, timed, require_auth (~120 lines)
```

4,812+ tests passing, 0 regressions across all 4 waves.

---

## What's Working Well

- **Response envelope consistency**: 100% of production code uses `asdict(success_response(...))` / `asdict(error_response(...))` — zero hand-rolled responses
- **Error code/type enums**: Proper `ErrorCode` (26 codes) and `ErrorType` (9 categories) throughout
- **Pagination**: Clean cursor-based `paginated_response()` wrapper
- **Security primitives**: `core/security.py` provides good input validation with size limits and injection detection
- **Test volume**: 71K lines across unit, integration, contract, and property-based tests
- **Documentation**: Excellent MCP best practices docs (15 guides) and response schema standards
- **Spec/task domain**: Well-structured `core/spec/` and `core/task/` packages with clean IO/hierarchy/mutation layers
- **Context management**: Clean `core/context.py` (540 lines) with W3C trace support
- **Resilience**: Good timeout budget strategy in `core/resilience.py`

---

## Recommended Execution Order

| Priority | Item | Status | Effort | Impact |
|----------|------|--------|--------|--------|
| 1 | Declarative parameter validation framework | **Done** (69 handlers, Waves 1-5) | — | Highest ROI, ~15-20% handler reduction |
| 2 | Unify error hierarchies | **Done** | — | Prerequisite for cleaner error handling |
| 3 | Split god objects | **Done** (Waves 1-4) | — | Improves testability and maintainability |
| 4 | Research sub-handlers | **Done** (Waves 1-4) | — | Consistency win, follows existing pattern |
| 5 | Tool signature cleanup | **Done** | — | `locals()` passthrough, ~350 lines |
| 6 | Test fixture consolidation | **Done** (Waves 1-4) | — | 5 conftest files, ~820 lines deduplicated |
| 7 | Observability consolidation | **Done** (Waves 1-4) | — | 6 files → 2 packages, deprecation shims |
| 8 | Deep research framework | **Done** (Waves 1-5) | — | ~320 lines removed, 2 sources of truth |
| 9 | Validation fix dispatch dict | **Done** | — | Quick win |
| 10 | Config module decomposition | **Done** (Waves 1-4) | — | Largest god object eliminated |
