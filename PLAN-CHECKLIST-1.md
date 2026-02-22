# Plan 1 Checklist: Schema Removals — estimated_hours + instructions

## estimated_hours Removal

### Core spec module
- [ ] Remove `estimated_hours` from all 5 phase template definitions in `core/spec/templates.py` `get_phase_template_structure()`
- [ ] Remove `estimated_hours` parameter from `apply_phase_template()` in `core/spec/templates.py`
- [ ] Remove `estimated_hours` from `generate_spec_data()` in `core/spec/templates.py`
- [ ] Remove `estimated_hours` parameter from `add_phase()` in `core/spec/hierarchy.py`
- [ ] Remove `estimated_hours` validation and node construction from `add_phase_bulk()` in `core/spec/hierarchy.py`
- [ ] Remove "estimates" weight (25%) from `check_spec_completeness()` in `core/spec/analysis.py` and redistribute weights
- [ ] Remove `recalculate_estimated_hours` from `core/spec/__init__.py` exports

### Validation
- [ ] Remove `estimated_hours` validation warnings from `core/validation/rules.py`

### Tool handlers
- [ ] Remove `estimated_hours` from `_handle_phase_add()` schema and handler in `authoring_handlers/handlers_phase.py`
- [ ] Remove `estimated_hours` from `_handle_phase_update_metadata()` in `authoring_handlers/handlers_phase.py`
- [ ] Remove `estimated_hours` from `_handle_phase_add_bulk()` task validation in `authoring_handlers/handlers_phase.py`
- [ ] Remove hours display from `_handle_phase_template()` list/show responses in `authoring_handlers/handlers_phase.py`
- [ ] Remove `_handle_recalculate_hours()` action handler from `tools/unified/spec.py`
- [ ] Remove `recalculate-hours` action from spec router registration

### Documentation
- [ ] Remove `estimated_hours` from `docs/05-mcp-tool-reference.md` authoring parameters
- [ ] Remove `recalculate-hours` from `docs/05-mcp-tool-reference.md` spec actions
- [ ] Remove hours references from `docs/04-cli-command-reference.md`

## instructions Removal

### Skill guidance
- [ ] Search for `instructions` references in all docs and skill files
- [ ] Remove any `instructions` examples from `docs/05-mcp-tool-reference.md`
- [ ] Ensure no docs tell LLMs to include `instructions` on tasks

## Verification

- [ ] Run full test suite — no test should depend on `estimated_hours` being required
- [ ] Verify existing specs with `estimated_hours` still load without errors
- [ ] Verify `check_spec_completeness()` returns valid scores (0-100) without the estimates dimension
- [ ] Verify `add_phase_bulk()` succeeds without `estimated_hours` on tasks
