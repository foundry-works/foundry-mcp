# Plan 3 Checklist: Phase Template Removal + Verification Scaffolding Migration

**Implemented:** 2026-02-22

## Verification scaffolding migration

- [x] Add auto-append of verification scaffolding (run-tests + fidelity) at the end of `add_phase_bulk()` in `core/spec/hierarchy.py`
- [x] Update `add_phase_bulk()` docstring to reflect new auto-append behavior
- [x] Add deduplication check: skip auto-append if caller already provided verify-type nodes in task list
- [x] Auto-appended verify nodes tracked in `tasks_created` result

## Phase template code removal

### Core functions
- [x] Remove `get_phase_template_structure()` from `core/spec/templates.py`
- [x] Remove `apply_phase_template()` from `core/spec/templates.py`
- [x] Remove `PHASE_TEMPLATES` and `add_phase_bulk` imports from templates.py (no longer needed)

### Constants
- [x] Remove `PHASE_TEMPLATES` from `core/spec/_constants.py`
- [x] Update `TEMPLATE_DESCRIPTIONS` text to reference `phase-add-bulk` instead of phase templates

### Exports (`core/spec/__init__.py`)
- [x] Remove `PHASE_TEMPLATES` import and `__all__` entry
- [x] Remove `apply_phase_template` import and `__all__` entry
- [x] Remove `get_phase_template_structure` import and `__all__` entry
- [x] Update module docstring

### Authoring handler
- [x] Remove `_handle_phase_template()` from `authoring_handlers/handlers_phase.py` (290 lines)
- [x] Remove `ActionDefinition(name="phase-template", ...)` from `authoring_handlers/__init__.py`
- [x] Remove `_handle_phase_template` import from `authoring_handlers/__init__.py`
- [x] Remove `PHASE_TEMPLATES`, `apply_phase_template`, `get_phase_template_structure` imports from handlers_phase.py
- [x] Remove `"phase-template"` from `_ACTION_SUMMARY` in `_helpers.py`
- [x] Update `phase-add-bulk` summary to note auto-appended verification scaffolding

### Monolith re-exports
- [x] Remove `apply_phase_template` and `get_phase_template_structure` from `core/spec/_monolith.py`

### Spec template handler (handlers_spec.py)
- [x] Remove `PHASE_TEMPLATES` import
- [x] Remove `phase_templates` list from spec-template list response
- [x] Update remediation messages and instructions to reference `phase-add-bulk`

### CLI commands
- [x] Remove local `PHASE_TEMPLATES` constant from `cli/commands/specs.py`
- [x] Update all text references from "phase templates" to "phase-add-bulk"

## Documentation

- [x] Remove `phase-template` from authoring action list in `docs/05-mcp-tool-reference.md`
- [x] Update `phase-add-bulk` docs to note auto-appended verification scaffolding
- [x] Update `docs/04-cli-command-reference.md` template option text

## Tests

- [x] Remove `TestPhaseTemplates` class from `tests/unit/test_core/test_spec.py`
- [x] Remove `TestApplyPhaseTemplate` class from `tests/unit/test_core/test_spec.py`
- [x] Remove imports of `PHASE_TEMPLATES`, `apply_phase_template`, `get_phase_template_structure` from test_spec.py
- [x] Update `test_spec_public_api.py`: remove symbols from baselines, update counts (38 functions, 10 constants, 48 total)
- [x] Update consumer import map for `foundry_mcp.tools.unified.authoring`

## Verification

- [x] No remaining imports of removed functions anywhere in `src/` or `tests/`
- [x] No remaining references to `phase-template` action in `src/`
- [x] Full test suite passes: 5340 passed, 48 skipped, 0 failures
