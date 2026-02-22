# Plan 4: Plan Linkage & Traceability

**Decision:** "Require Plan Linkage on Specs"
**Dependencies:** None strictly, but better after Plan 2 (metadata enrichment establishes the pattern)
**Risk:** Low-medium — adds required fields, which could break spec creation workflows if not handled carefully

---

## Rationale

A spec should know its origin. The mandatory markdown plan defines what was agreed upon with the human. The spec is the JSON translation. Without `plan_path` and `plan_review_path`, there's no way to trace from implementation back to planning rationale.

This is data integrity (the server can enforce it), not process integrity (the server can't verify human approval).

---

## Scope

### 1. Add `plan_path` and `plan_review_path` to spec metadata

**Schema:**
```json
{
  "metadata": {
    "plan_path": "specs/.plans/feature-name.md",
    "plan_review_path": "specs/.plan-reviews/feature-name-review-full.md"
  }
}
```

**File: `src/foundry_mcp/core/spec/templates.py`**
- `generate_spec_data()` (lines 327-391): Accept `plan_path` and `plan_review_path` as parameters; store in metadata
- `create_spec()` (lines 394-470): Pass `plan_path` and `plan_review_path` through to `generate_spec_data()`

### 2. Add `plan_path` and `plan_review_path` parameters to `spec-create`

**File: `src/foundry_mcp/tools/unified/authoring_handlers/handlers_spec.py`**
- `_SPEC_CREATE_SCHEMA` (lines 78-88): Add `plan_path` (required, string) and `plan_review_path` (required, string) to the validation schema
- `_handle_spec_create()` (lines 91-228): Extract and pass both paths to `create_spec()`

### 3. Add validation rules

**File: `src/foundry_mcp/core/validation/rules.py`**
- In `_validate_structure()` (lines 178-258): Add validation errors (not warnings) for:
  1. `metadata.plan_path` missing or empty → error
  2. `metadata.plan_review_path` missing or empty → error

**File: `src/foundry_mcp/core/spec/templates.py`** (or a new validation helper)
- Add file existence validation: check that `plan_path` points to an existing file
- Add file existence validation: check that `plan_review_path` points to an existing file
- These checks need a workspace/specs_dir context to resolve relative paths
- **Decision:** Validate at creation time (in `create_spec()`), not in `validate_spec()`. The general validator may not have filesystem context, but creation always does.

### 4. Update fidelity gate to use plan context

**File: `src/foundry_mcp/tools/unified/documentation_helpers.py`**
- `_build_spec_overview()` (lines 300-354): Include `plan_path` and `plan_review_path` in overview
- Consider adding a new context builder: `_build_plan_context()` — reads the linked plan file and extracts key sections (mission, success criteria, constraints, risks) for the fidelity review LLM

**File: `src/foundry_mcp/tools/unified/review.py`**
- In fidelity review context building (around lines 821-874): If `metadata.plan_path` exists and the file is readable, include plan content in the fidelity request context. This gives the fidelity reviewer access to the original intent, not just the spec's interpretation of it.

### 5. Update documentation

**File: `docs/05-mcp-tool-reference.md`**
- Add `plan_path` and `plan_review_path` to `spec-create` parameters
- Document as required fields

---

## Design Notes

- **File existence validation at creation, not general validation.** `validate_spec()` is a pure data validator (takes JSON, returns diagnostics). Adding filesystem checks would break its contract. Instead, validate file existence in `create_spec()` before writing the spec file. If `validate_spec()` is also called with a `specs_dir` context (check current signature), the file checks could go there too — but creation-time is the primary gate.
- **Relative vs absolute paths.** Store paths as relative to the specs directory (e.g., `specs/.plans/feature.md` or `.plans/feature.md`). Resolve to absolute at read time using the workspace context. This makes specs portable across machines.
- **plan_review_path stores the synthesized review**, not individual per-provider reviews. The `.plan-reviews/` directory may contain multiple files (one per provider + one synthesized). The `plan_review_path` points to the synthesized one.
- **Backward compatibility.** Existing specs won't have these fields. Options:
  - (A) Hard break: validation fails on old specs → forces migration
  - (B) Soft break: warn on old specs, error on new specs created after this change
  - **Recommendation:** (B). Add validation warning (not error) for missing `plan_path`/`plan_review_path`. But `spec-create` requires them as input parameters. This means old specs get a warning; new specs can't be created without them.
