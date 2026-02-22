# Plan 2 Checklist: Schema Enrichment — Metadata Fields + Task Complexity

## New metadata fields (success_criteria, constraints, risks, open_questions)

### Schema initialization
- [ ] Add `success_criteria: []` to `generate_spec_data()` metadata in `core/spec/templates.py`
- [ ] Add `constraints: []` to `generate_spec_data()` metadata
- [ ] Add `risks: []` to `generate_spec_data()` metadata
- [ ] Add `open_questions: []` to `generate_spec_data()` metadata

### Validation
- [ ] Add validation warning for empty `success_criteria` in `core/validation/rules.py`
- [ ] Add validation warning for empty `constraints` in `core/validation/rules.py`
- [ ] Add `risks` object validation: required `description` string; optional `likelihood` (low|medium|high), `impact` (low|medium|high), `mitigation` (string)
- [ ] No structural validation needed for `open_questions` (string array)

### Core add/list functions
- [ ] Add `add_constraint(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [ ] Add `add_risk(spec_id, description, likelihood, impact, mitigation, specs_dir)` in `core/spec/templates.py`
- [ ] Add `add_question(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [ ] Add `add_success_criterion(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [ ] Add `list_constraints(spec_id, specs_dir)` in `core/spec/templates.py`
- [ ] Add `list_risks(spec_id, specs_dir)` in `core/spec/templates.py`
- [ ] Add `list_questions(spec_id, specs_dir)` in `core/spec/templates.py`
- [ ] Add `list_success_criteria(spec_id, specs_dir)` in `core/spec/templates.py`
- [ ] Export new functions from `core/spec/__init__.py`

### Authoring handlers
- [ ] Add `_handle_constraint_add()` handler in `authoring_handlers/handlers_metadata.py`
- [ ] Add `_handle_risk_add()` handler with description/likelihood/impact/mitigation params
- [ ] Add `_handle_question_add()` handler
- [ ] Add `_handle_success_criterion_add()` handler
- [ ] Register `constraint-add`, `risk-add`, `question-add`, `success-criterion-add` actions in `authoring_handlers/__init__.py`

### Frontmatter blocking
- [ ] Add `success_criteria`, `constraints`, `risks`, `open_questions` to blocked fields in `update_frontmatter()` in `core/spec/templates.py`

## Task complexity

### Constants
- [ ] Add `COMPLEXITY_LEVELS = ("low", "medium", "high")` to `core/spec/_constants.py`
- [ ] Export `COMPLEXITY_LEVELS` from `core/spec/__init__.py`

### Validation
- [ ] Add `metadata.complexity` validation for task nodes in `core/validation/rules.py`: valid values low|medium|high
- [ ] Add validation warning if complexity is missing on task nodes

### Hierarchy
- [ ] Accept and store `complexity` in task metadata via `add_phase_bulk()` in `core/spec/hierarchy.py`
- [ ] Validate complexity values in `add_phase_bulk()` if provided

## Fidelity context enrichment

- [ ] Include `success_criteria`, `constraints`, `risks`, `open_questions` in `_build_spec_overview()` in `documentation_helpers.py`
- [ ] Include task `complexity` in `_build_spec_requirements()` in `documentation_helpers.py`

## Documentation

- [ ] Add `constraint-add`, `risk-add`, `question-add`, `success-criterion-add` to `docs/05-mcp-tool-reference.md`
- [ ] Document new metadata fields in spec schema section of docs

## Verification

- [ ] Existing specs without new fields load without errors
- [ ] `generate_spec_data()` produces valid spec with new empty-array fields
- [ ] `validate_spec()` warns (not errors) on empty success_criteria/constraints
- [ ] `add_risk()` validates required `description` field
- [ ] `add_risk()` validates `likelihood` and `impact` enum values
- [ ] `add_phase_bulk()` accepts and stores complexity on task metadata
- [ ] Fidelity review context includes new metadata fields
- [ ] Run full test suite
