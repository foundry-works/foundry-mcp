# Codebase Refactoring Research

> Senior engineering review of foundry-mcp (~27K LOC source, ~71K LOC tests)
> Date: 2026-02-16

---

## 1. Declarative Parameter Validation Framework — IN PROGRESS

### Status: Waves 1-4 complete (54 handlers migrated)

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

**Remaining work** (Wave 5 — deferred):
- 20+ handlers in `research_handlers/` (uses simplified validation pattern, separate test suites)
- `spec.py` handlers already use `error_response()` directly — no migration needed
- `plan.py` and `review.py` — 0 `_validation_error` calls, no migration needed

### Original Scale

| Metric | Count |
|--------|-------|
| Handler functions | 46 |
| `_validation_error()` calls | 346 |
| Distinct parameter names | 85 |
| Files with validation boilerplate | 14+ |

### Impact

- ~15-20% codebase reduction in handler files
- Eliminates entire class of copy-paste validation bugs
- Makes adding new handlers trivial (schema only, no validation code)

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

## 6. Observability/Metrics Sprawl (Medium)

### Problem

Metrics and observability logic is spread across **7 files totaling 5,000+ lines** with unclear boundaries.

### File Inventory

| File | Lines | Responsibility |
|------|-------|---------------|
| `core/observability.py` | 1,218 | Orchestration + decorators |
| `core/metrics_store.py` | 641 | Abstract base + implementations |
| `core/metrics_persistence.py` | 584 | File I/O for metrics |
| `core/prometheus.py` | 564 | Prometheus exporter |
| `core/otel.py` | 452 | OpenTelemetry integration |
| `core/metrics_registry.py` | 327 | In-memory registry |
| `core/otel_stubs.py` | 264 | Fallback stubs when OTel unavailable |

### Recommendation

Consolidate into two packages:
```
core/metrics/        # Registry, data models, persistence
core/observability/  # Logging, tracing, auditing, decorators, exporters
```

---

## 7. Test Fixture Duplication (Medium)

### Problem

Common test fixtures are independently redefined across many test files.

### Duplicate Fixture Counts

| Fixture | Occurrences | Lines Wasted |
|---------|-------------|-------------|
| `mock_config` | 20 files | ~300 |
| `temp_specs_dir` | 16 files | ~100 |
| `test_config` | 10 files | ~80 |
| `mock_memory` | 10 files | ~100 |
| `test_specs_dir` | 9 files | ~70 |
| `sample_spec` | 8 files | ~120 |
| `mcp_server` | 5 files | ~50 |

### Missing Conftest Files

```
tests/conftest.py                    # Exists (305 lines, only 3 fixtures)
tests/core/research/conftest.py      # MISSING — should centralize mock_config, mock_memory
tests/unit/test_core/conftest.py     # MISSING — should centralize test_config, temp_specs_dir
tests/tools/unified/conftest.py      # MISSING — should centralize shared tool fixtures
```

### Monolithic Test Files

| File | Lines | Issue |
|------|-------|-------|
| `tests/core/research/workflows/test_deep_research.py` | 2,403 | Mix of fixture setup + 40+ test scenarios |
| `tests/core/research/test_document_digest.py` | 2,356 | 9 test classes, should be 3 files |
| `tests/unit/test_core/test_spec.py` | 2,209 | Large fixture overhead |
| `tests/unit/test_batch_operations.py` | 1,643 | 18 test classes in 1 file |

### Test Coverage Gaps

- **Entire CLI layer untested** (22+ source files in `cli/` with zero tests)
- Missing tests for: `core/cache.py`, `core/health.py`, `core/observability.py`, `core/prometheus.py`, `core/otel.py`, `core/error_collection.py`, `core/concurrency.py`
- Estimated gap: ~30-40% of source modules lack dedicated unit tests

---

## 8. Deep Research Phase Execution Framework (Medium)

### Problem

`DeepResearchWorkflow` uses 6+ mixin classes (5,366+ lines total) where each phase repeats identical patterns for error handling, audit logging, and state mutation.

### Files

| File | Lines | Mixin |
|------|-------|-------|
| `core/research/workflows/deep_research/core.py` | 1,639 | Main orchestrator |
| `phases/analysis.py` | 1,203 | AnalysisPhaseMixin |
| `phases/gathering.py` | 824 | GatheringPhaseMixin |
| `phases/refinement.py` | 660 | RefinementPhaseMixin |
| `phases/synthesis.py` | 619 | SynthesisPhaseMixin |
| `phases/planning.py` | 421 | PlanningPhaseMixin |

### Repeated Pattern in Each Phase

```python
async def _execute_<phase>_async(...):
    try:
        result = await self._execute_provider_async(...)
        state.<phase> = result
        self._write_audit_event(...)
    except ContextWindowError:
        # Error handling
    except Exception as e:
        # More error handling
    return WorkflowResult(...)
```

### Recommendation

Extract generic `_execute_phase_async(phase_name, executor, timeout)` method. Each phase only defines its unique logic; the lifecycle (error handling, audit, state mutation) is handled by the framework.

---

## 9. Dispatch Dictionary for Validation Fixes — DONE

Replaced 12 sequential `if code ==` statements in `core/validation/fixes.py` with two dispatch dicts split by argument shape:
- `_SPEC_DATA_HANDLERS` — handlers taking `(diag, spec_data)` (1 entry)
- `_HIERARCHY_HANDLERS` — handlers taking `(diag, hierarchy)` (11 entries)
- `_COUNTS_CODES` frozenset — 4 codes routing to `_build_counts_fix(diag, spec_data)`

`_build_fix_action` body reduced from 30+ lines to 10 lines (3 dict lookups + 1 set membership test).

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
| 1 | Declarative parameter validation framework | **In progress** (44/46 handlers, Waves 1-3 done) | 2-3 days remaining (Wave 4) | Highest ROI, ~15-20% handler reduction |
| 2 | Unify error hierarchies | **Done** | — | Prerequisite for cleaner error handling |
| 3 | Split god objects | **Done** (Waves 1-4) | — | Improves testability and maintainability |
| 4 | Research sub-handlers | **Done** (Waves 1-4) | — | Consistency win, follows existing pattern |
| 5 | Tool signature cleanup | **Partial** (research done, task/authoring remaining) | 0.5 day | `locals()` passthrough, ~350 lines |
| 6 | Test fixture consolidation | Pending | 2-3 days | Reduces test maintenance friction |
| 7 | Observability consolidation | Pending | 3-4 days | Lower urgency, improves clarity |
| 8 | Deep research framework | Pending | 3-4 days | Contained to one subsystem |
| 9 | Validation fix dispatch dict | Pending | 0.5 day | Quick win |
