# Codebase Refactoring Research

> Senior engineering review of foundry-mcp (~27K LOC source, ~71K LOC tests)
> Date: 2026-02-16

---

## 1. Declarative Parameter Validation Framework — IN PROGRESS

### Status: Waves 1-3 complete (44/46 handlers migrated)

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

**Remaining work** (Wave 4):
- 21 handlers in other tool modules (`spec.py`, `lifecycle.py`, `verification.py`, `provider.py`); plus 20 deferred (`research.py`, `plan.py`, `review.py`)

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

## 4. Research Tool Module Decomposition (High)

### Problem

`tools/unified/research.py` (1,710 lines, 17 handler functions) is monolithic — unlike `task.py` and `authoring.py` which correctly delegate to sub-handler modules.

### Current State

- `task.py` (1 KB) delegates to `task_handlers/` (4 sub-modules)
- `authoring.py` (1 KB) delegates to `authoring_handlers/` (4 sub-modules)
- `research.py` (1,710 lines) — everything inline, 17 handlers with repetitive response wrapping

### 17 Research Handlers

| Handler | Responsibility |
|---------|---------------|
| `_handle_chat` | Single-model conversation |
| `_handle_consensus` | Multi-model parallel consultation |
| `_handle_thinkdeep` | Hypothesis-driven investigation |
| `_handle_ideate` | Creative brainstorming |
| `_handle_deep_research` | Multi-phase deep research |
| `_handle_deep_research_status` | Get research session status |
| `_handle_deep_research_report` | Get final report |
| `_handle_deep_research_list` | List sessions |
| `_handle_deep_research_delete` | Delete session |
| `_handle_thread_list` | List threads |
| `_handle_thread_get` | Get thread details |
| `_handle_thread_delete` | Delete thread |
| `_handle_node_execute` | Execute research node |
| `_handle_node_record` | Record node result |
| `_handle_node_status` | Get node status |
| `_handle_node_findings` | Get node findings |
| `_handle_extract` | Extract content |

### Repeated Pattern

Each handler repeats:
```python
if result.success:
    return asdict(success_response(data={...}))
else:
    return asdict(error_response(...))
```

### Recommendation

Split into `research_handlers/` following existing pattern:
```
research_handlers/
    __init__.py              # Router/dispatcher
    handlers_chat.py         # chat, thread-list, thread-get, thread-delete
    handlers_consensus.py    # consensus
    handlers_deep.py         # deep-research, status, report, list, delete
    handlers_ideate.py       # ideate
    handlers_investigation.py # thinkdeep, node-*, extract
```

Also hand-rolls `_validation_error()` locally (lines 105-122) instead of using the `make_validation_error_fn()` factory.

---

## 5. Tool Registration Signature Bloat (Medium-High)

### Problem

Tool entry points have 40-55 parameter signatures that exist only to repack into a dict:

```python
def task(action, spec_id, task_id, workspace, status_filter,
         # ... 50 more parameters ...
         ) -> dict:
    payload = {"spec_id": spec_id, "task_id": task_id, ...}
    return _dispatch(action=action, payload=payload)
```

### Files Affected

| File | Parameters | Boilerplate Lines |
|------|-----------|-------------------|
| `task_handlers/__init__.py` | 55+ | ~200 |
| `authoring_handlers/__init__.py` | 40+ | ~150 |
| `research.py` register function | 40+ | ~150 |

### Recommendation

Use `**kwargs` passthrough since FastMCP already handles parameter parsing from the MCP schema. Eliminates ~500 lines of pure dict-packing boilerplate.

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

## 9. Dispatch Dictionary for Validation Fixes (Low — Quick Win)

### Problem

`core/validation.py:1164-1218` uses 12 sequential `if code ==` statements:

```python
if code == "INVALID_DATE_FORMAT":
    return _build_date_fix(diag, spec_data)
if code == "PARENT_CHILD_MISMATCH":
    return _build_hierarchy_align_fix(diag, hierarchy)
# ... 10 more
```

### Fix

```python
FIX_HANDLERS = {
    "INVALID_DATE_FORMAT": _build_date_fix,
    "PARENT_CHILD_MISMATCH": _build_hierarchy_align_fix,
    ...
}
handler = FIX_HANDLERS.get(code)
return handler(diag, spec_data) if handler else None
```

3-line change, cleaner and extensible.

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
| 4 | Research sub-handlers | Pending | 2-3 days | Consistency win, follows existing pattern |
| 5 | Tool signature cleanup | Pending | 1-2 days | Small effort, ~500 lines removed |
| 6 | Test fixture consolidation | Pending | 2-3 days | Reduces test maintenance friction |
| 7 | Observability consolidation | Pending | 3-4 days | Lower urgency, improves clarity |
| 8 | Deep research framework | Pending | 3-4 days | Contained to one subsystem |
| 9 | Validation fix dispatch dict | Pending | 0.5 day | Quick win |
