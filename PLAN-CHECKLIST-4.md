# Plan 4 Checklist: Plan Linkage & Traceability

## Spec metadata fields

### Schema initialization
- [ ] Add `plan_path` parameter to `generate_spec_data()` in `core/spec/templates.py`
- [ ] Add `plan_review_path` parameter to `generate_spec_data()` in `core/spec/templates.py`
- [ ] Store both in `metadata` dict during spec generation
- [ ] Add `plan_path` and `plan_review_path` parameters to `create_spec()` in `core/spec/templates.py`
- [ ] Pass both through to `generate_spec_data()`

### spec-create handler
- [ ] Add `plan_path` (required, string) to `_SPEC_CREATE_SCHEMA` in `authoring_handlers/handlers_spec.py`
- [ ] Add `plan_review_path` (required, string) to `_SPEC_CREATE_SCHEMA`
- [ ] Extract both from payload in `_handle_spec_create()`
- [ ] Pass both to `create_spec()` call

### Validation
- [ ] Add validation warning for missing/empty `metadata.plan_path` in `core/validation/rules.py` `_validate_structure()`
- [ ] Add validation warning for missing/empty `metadata.plan_review_path` in `core/validation/rules.py` `_validate_structure()`
- [ ] Add file existence check for `plan_path` in `create_spec()` — error if file not found
- [ ] Add file existence check for `plan_review_path` in `create_spec()` — error if file not found

## Fidelity context enrichment

- [ ] Include `plan_path` and `plan_review_path` in `_build_spec_overview()` output in `documentation_helpers.py`
- [ ] Add `_build_plan_context()` helper in `documentation_helpers.py` — reads plan file, extracts key sections
- [ ] Include plan content in fidelity review request context in `review.py` (when plan_path exists and file is readable)

## Documentation

- [ ] Add `plan_path` and `plan_review_path` to `spec-create` params in `docs/05-mcp-tool-reference.md`
- [ ] Document fields as required for new specs, warning-level for existing specs

## Verification

- [ ] `spec-create` requires `plan_path` and `plan_review_path` — fails without them
- [ ] `spec-create` validates both files exist on disk — fails if not found
- [ ] `validate_spec()` warns (not errors) on specs missing plan_path (backward compat)
- [ ] Existing specs without plan_path still load and validate with warnings only
- [ ] Fidelity review context includes plan content when plan_path is present
- [ ] Fidelity review works normally when plan_path is absent (old specs)
- [ ] Run full test suite
