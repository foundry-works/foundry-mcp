# Plan 4 Checklist: Plan Linkage & Traceability

## Spec metadata fields

### Schema initialization
- [x] Add `plan_path` parameter (Optional[str], default None) to `generate_spec_data()` in `core/spec/templates.py`
- [x] Add `plan_review_path` parameter (Optional[str], default None) to `generate_spec_data()` in `core/spec/templates.py`
- [x] Store both in `metadata` dict during spec generation (alongside existing fields at line 122)
- [x] Add `plan_path` and `plan_review_path` parameters to `create_spec()` in `core/spec/templates.py`
- [x] Pass both through to `generate_spec_data()` call at line 165

### spec-create handler
- [x] Add `plan_path` (required, string) to `_SPEC_CREATE_SCHEMA` in `authoring_handlers/handlers_spec.py`
- [x] Add `plan_review_path` (required, string) to `_SPEC_CREATE_SCHEMA`
- [x] Extract both from payload in `_handle_spec_create()` (after line 107)
- [x] Pass both to `generate_spec_data()` call (dry_run path, line 116)
- [x] Pass both to `create_spec()` call (real create path, line 174)

### Validation — general validator
- [x] Add validation warning for missing/empty `metadata.plan_path` in `_validate_structure()` in `core/validation/rules.py` (after line 305)
- [x] Add validation warning for missing/empty `metadata.plan_review_path` in `_validate_structure()`

### Validation — creation-time file existence
- [x] Add file existence check for `plan_path` in `create_spec()` — resolve relative to specs_dir, error if file not found
- [x] Add file existence check for `plan_review_path` in `create_spec()` — resolve relative to specs_dir, error if file not found

## Fidelity context enrichment

### Spec overview
- [x] Include `plan_path` and `plan_review_path` in `_build_spec_overview()` output in `documentation_helpers.py` (before return at line 396)

### Plan context builder
- [x] Add `_build_plan_context(spec_data, workspace_root)` function in `documentation_helpers.py`
  - Resolves `metadata.plan_path` relative to workspace_root
  - Reads plan file content
  - Returns formatted markdown or empty string if unavailable
- [x] Include plan context in fidelity review request context in `review.py` (add `"plan_context"` key to ConsultationRequest.context at line 860)

## Documentation

- [x] Add `plan_path` and `plan_review_path` to `spec-create` params in `docs/05-mcp-tool-reference.md` parameter table
- [x] Update `spec-create` example to include plan_path and plan_review_path
- [x] Document fields as required for new specs, warning-level for existing specs

## Tests

- [x] Test `generate_spec_data()` stores plan_path/plan_review_path in metadata when provided
- [x] Test `generate_spec_data()` omits plan_path/plan_review_path from metadata when not provided (backward compat)
- [x] Test `create_spec()` fails when plan_path file doesn't exist on disk
- [x] Test `create_spec()` fails when plan_review_path file doesn't exist on disk
- [x] Test `create_spec()` succeeds when both files exist
- [x] Test `_handle_spec_create()` rejects payload missing plan_path — covered by schema validation (required field)
- [x] Test `_handle_spec_create()` rejects payload missing plan_review_path — covered by schema validation (required field)
- [x] Test `validate_spec()` warns (not errors) on specs missing plan_path (backward compat)
- [x] Test `_build_spec_overview()` includes plan_path when present
- [x] Test `_build_plan_context()` returns content when plan file exists
- [x] Test `_build_plan_context()` returns empty string when plan file missing
- [x] Test `_build_plan_context()` returns empty string when plan_path absent from metadata

## Verification

- [x] `spec-create` requires `plan_path` and `plan_review_path` — fails without them
- [x] `spec-create` validates both files exist on disk — fails if not found
- [x] `validate_spec()` warns (not errors) on specs missing plan_path (backward compat)
- [x] Existing specs without plan_path still load and validate with warnings only
- [x] Fidelity review context includes plan content when plan_path is present
- [x] Fidelity review works normally when plan_path is absent (old specs)
- [x] Run full test suite
