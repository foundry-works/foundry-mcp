# Plan 3 Checklist: Phase Template Removal + Verification Scaffolding Migration

## Verification scaffolding migration

- [ ] Add auto-append of verification scaffolding (run-tests + fidelity) at the end of `add_phase_bulk()` in `core/spec/hierarchy.py`
- [ ] Add deduplication check: skip auto-append if caller already provided verify-type nodes in task list
- [ ] Verify `_add_phase_verification()` (lines 94-152) handles the bulk case correctly (phase_num, phase_id, children list, total_tasks update)

## Phase template code removal

### Core functions
- [ ] Remove `get_phase_template_structure()` from `core/spec/templates.py` (lines 74-233)
- [ ] Remove `apply_phase_template()` from `core/spec/templates.py` (lines 236-324)

### Constants
- [ ] Remove `PHASE_TEMPLATES` from `core/spec/_constants.py`

### Exports
- [ ] Remove `PHASE_TEMPLATES` from `core/spec/__init__.py` exports
- [ ] Remove `get_phase_template_structure` from `core/spec/__init__.py` exports
- [ ] Remove `apply_phase_template` from `core/spec/__init__.py` exports

### Authoring handler
- [ ] Remove `_handle_phase_template()` from `authoring_handlers/handlers_phase.py` (lines 665-950)
- [ ] Remove `ActionDefinition(name="phase-template", ...)` from `authoring_handlers/__init__.py`
- [ ] Remove `_handle_phase_template` import from `authoring_handlers/__init__.py`

## Documentation

- [ ] Remove `phase-template` action from `docs/05-mcp-tool-reference.md`
- [ ] Update `phase-add-bulk` docs to note auto-appended verification scaffolding
- [ ] Remove phase template references from `docs/04-cli-command-reference.md`
- [ ] Remove phase template references from `docs/07-troubleshooting.md`

## Verification

- [ ] `add_phase_bulk()` auto-appends run-tests + fidelity verify nodes when caller doesn't include them
- [ ] `add_phase_bulk()` does NOT double-append when caller includes verify nodes
- [ ] `add_phase()` behavior unchanged (already has scaffolding)
- [ ] Spec creation with `"empty"` template still works
- [ ] No remaining imports of removed functions anywhere in codebase
- [ ] Run full test suite — update/remove tests that reference phase templates
