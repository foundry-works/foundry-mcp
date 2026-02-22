# Plan 1 Checklist: Schema Removals — estimated_hours + instructions

## estimated_hours Removal

### Core spec module
- [x] Remove `estimated_hours` from all 5 phase template definitions in `core/spec/templates.py` `get_phase_template_structure()`
- [x] Remove `estimated_hours` parameter from `apply_phase_template()` in `core/spec/templates.py`
- [x] Remove `estimated_hours` parameter from `add_phase()` in `core/spec/hierarchy.py`
- [x] Remove `estimated_hours` validation and node construction from `add_phase_bulk()` in `core/spec/hierarchy.py`
- [x] Remove `update_phase_metadata()` `estimated_hours` parameter and handling in `core/spec/hierarchy.py`
- [x] Delete `recalculate_estimated_hours()` function entirely from `core/spec/hierarchy.py`
- [x] Remove "estimates" weight (25%) from `check_spec_completeness()` in `core/spec/analysis.py` and redistribute weights (25/40/35)
- [x] Remove `recalculate_estimated_hours` from `core/spec/__init__.py` exports
- [x] Remove `recalculate_estimated_hours` re-export from `core/spec/_monolith.py`

### Validation
- [x] `core/validation/rules.py` — confirmed no direct `estimated_hours` validation to remove (no changes needed)

### Tool handlers
- [x] Remove `estimated_hours` from `_handle_phase_add()` schema and handler in `authoring_handlers/handlers_phase.py`
- [x] Remove `estimated_hours` from `_handle_phase_update_metadata()` in `authoring_handlers/handlers_phase.py`
- [x] Remove `estimated_hours` from `_handle_phase_add_bulk()` task and phase validation in `authoring_handlers/handlers_phase.py`
- [x] Remove hours display from `_handle_phase_template()` list/show responses in `authoring_handlers/handlers_phase.py`
- [x] Remove `_handle_recalculate_hours()` action handler from `tools/unified/spec.py`
- [x] Remove `recalculate-hours` action from spec router `_ACTIONS` list
- [x] Remove `recalculate_estimated_hours` import from `tools/unified/spec.py`

### Documentation
- [x] Remove `estimated_hours` from `docs/concepts/spec-schema.md` examples and metadata table
- [x] Remove `estimated_hours` from `docs/05-mcp-tool-reference.md` authoring parameters — confirmed clean (not present)
- [x] Remove `recalculate-hours` from `docs/05-mcp-tool-reference.md` spec actions — confirmed clean (not present)
- [x] Remove hours references from `docs/04-cli-command-reference.md` — confirmed clean (not present)

### Remaining file cleanups
- [x] `authoring_handlers/__init__.py` — removed `estimated_hours` param from function signature
- [x] `schemas/foundry-spec-schema.json` — removed `estimated_hours` from both spec-level and phase-level schema definitions
- [x] `mcp/capabilities_manifest.json` — removed `estimated_hours` parameter and example usage

## instructions Removal

### Skill guidance
- [x] Searched all docs and skill files — no `instructions` references found (already clean)

## Test file updates
- [x] `tests/unit/test_core/test_spec.py` — removed `recalculate_estimated_hours` import, deleted `TestRecalculateEstimatedHours` class, fixed template tests and add_phase tests
- [x] `tests/unit/test_core/test_spec_public_api.py` — removed from expected exports lists, updated baseline counts (33→32, 43→42)
- [x] `tests/unit/test_core/test_phase_metadata_update.py` — rewrote all tests to use description/purpose instead of estimated_hours
- [x] `tests/unit/test_core/test_spec_validation.py` — removed "estimates" category assertion from completeness check
- [x] `tests/unit/test_phase_add_bulk.py` — removed estimated_hours validation tests (task, phase, metadata_defaults)
- [x] `tests/integration/test_authoring_tools.py` — removed `test_phase_add_validates_hours` test
- [x] Remaining test files with `estimated_hours` references audited — all are in the **task domain** (core/task, task_handlers) which still supports estimated_hours; no changes needed

## Verification
- [x] Run full test suite — 5344 passed, 48 skipped, 0 failures
- [x] Verify existing specs with `estimated_hours` still load without errors (additionalProperties allows old data)
- [x] Verify `check_spec_completeness()` returns valid scores (0-100) without the estimates dimension
- [x] Verify `add_phase_bulk()` succeeds without `estimated_hours` on tasks

## Out of Scope (noted for future plans)
- Task domain (`core/task/mutations.py`, `core/task/queries.py`, `core/task/batch.py`) still uses `estimated_hours` for `update-estimate` action and task creation
- Task tool handlers (`task_handlers/handlers_mutation.py`, `task_handlers/handlers_batch.py`, `task_handlers/__init__.py`) still expose `estimated_hours`
- CLI commands (`cli/commands/modify.py`, `cli/commands/specs.py`) still reference `estimated_hours`
- `core/review.py` still warns about missing estimates in validation
