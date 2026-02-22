# Plan 2 Checklist: Schema Enrichment — Metadata Fields + Task Complexity

## New metadata fields (success_criteria, constraints, risks, open_questions)

### Schema initialization
- [x] Add `success_criteria: []` to `generate_spec_data()` metadata in `core/spec/templates.py`
- [x] Add `constraints: []` to `generate_spec_data()` metadata
- [x] Add `risks: []` to `generate_spec_data()` metadata
- [x] Add `open_questions: []` to `generate_spec_data()` metadata

### Validation
- [x] Add validation warning for empty `success_criteria` in `core/validation/rules.py`
- [x] Add validation warning for empty `constraints` in `core/validation/rules.py`
- [x] Add `risks` object validation: required `description` string; optional `likelihood` (low|medium|high), `impact` (low|medium|high), `mitigation` (string)
- [x] No structural validation needed for `open_questions` (string array)

### Core add/list functions
- [x] Add `add_constraint(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [x] Add `add_risk(spec_id, description, likelihood, impact, mitigation, specs_dir)` in `core/spec/templates.py`
- [x] Add `add_question(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [x] Add `add_success_criterion(spec_id, text, specs_dir)` in `core/spec/templates.py`
- [x] Add `list_constraints(spec_id, specs_dir)` in `core/spec/templates.py`
- [x] Add `list_risks(spec_id, specs_dir)` in `core/spec/templates.py`
- [x] Add `list_questions(spec_id, specs_dir)` in `core/spec/templates.py`
- [x] Add `list_success_criteria(spec_id, specs_dir)` in `core/spec/templates.py`
- [x] Export new functions from `core/spec/__init__.py`

### Authoring handlers
- [x] Add `_handle_constraint_add()` handler in `authoring_handlers/handlers_metadata.py`
- [x] Add `_handle_risk_add()` handler with description/likelihood/impact/mitigation params
- [x] Add `_handle_question_add()` handler
- [x] Add `_handle_success_criterion_add()` handler
- [x] Register `constraint-add`, `risk-add`, `question-add`, `success-criterion-add` actions in `authoring_handlers/__init__.py`

### Frontmatter blocking
- [x] Add `success_criteria`, `constraints`, `risks`, `open_questions` to blocked fields in `update_frontmatter()` in `core/spec/templates.py`

## Task complexity

### Constants
- [x] Add `COMPLEXITY_LEVELS = ("low", "medium", "high")` to `core/spec/_constants.py`
- [x] Export `COMPLEXITY_LEVELS` from `core/spec/__init__.py`

### Validation
- [x] Add `metadata.complexity` validation for task nodes in `core/validation/rules.py`: valid values low|medium|high
- [x] Add validation warning if complexity is missing on task nodes

### Hierarchy
- [x] Accept and store `complexity` in task metadata via `add_phase_bulk()` in `core/spec/hierarchy.py`
- [x] Validate complexity values in `add_phase_bulk()` if provided

## Fidelity context enrichment

- [x] Include `success_criteria`, `constraints`, `risks`, `open_questions` in `_build_spec_overview()` in `documentation_helpers.py`
- [x] Include task `complexity` in `_build_spec_requirements()` in `documentation_helpers.py`

## Documentation

- [x] Add `constraint-add`, `risk-add`, `question-add`, `success-criterion-add` to `docs/05-mcp-tool-reference.md`
- [x] Document new metadata fields in spec schema section of docs

## Verification

- [x] Existing specs without new fields load without errors
- [x] `generate_spec_data()` produces valid spec with new empty-array fields
- [x] `validate_spec()` warns (not errors) on empty success_criteria/constraints
- [x] `add_risk()` validates required `description` field
- [x] `add_risk()` validates `likelihood` and `impact` enum values
- [x] `add_phase_bulk()` accepts and stores complexity on task metadata
- [x] Fidelity review context includes new metadata fields
- [x] Run full test suite — **5362 passed, 0 failed**
