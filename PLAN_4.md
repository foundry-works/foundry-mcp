# Plan 4: Plan Linkage & Traceability

**Decision:** "Require Plan Linkage on Specs"
**Dependencies:** None strictly, but better after Plan 2 (metadata enrichment establishes the pattern)
**Risk:** Low-medium — adds required fields, which could break spec creation workflows if not handled carefully
**Line references verified:** 2026-02-22 (post-Plan 3 implementation)

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
- `generate_spec_data()` (lines 71-139): Accept `plan_path` and `plan_review_path` as optional parameters; store in metadata dict (alongside existing fields at lines 122-131)
- `create_spec()` (lines 142-218): Accept `plan_path` and `plan_review_path` parameters; pass through to `generate_spec_data()` call at lines 165-170

### 2. Add `plan_path` and `plan_review_path` parameters to `spec-create`

**File: `src/foundry_mcp/tools/unified/authoring_handlers/handlers_spec.py`**
- `_SPEC_CREATE_SCHEMA` (lines 77-87): Add `plan_path` (required, string) and `plan_review_path` (required, string) to the validation schema
- `_handle_spec_create()` (lines 90-227): Extract both paths from payload (after line 107); pass to `generate_spec_data()` call at line 116-121 (dry_run) and `create_spec()` call at lines 174-180 (real create)

### 3. Add validation rules

**File: `src/foundry_mcp/core/validation/rules.py`**
- In `_validate_structure()` (lines 179-326): After the existing metadata checks (ends at line 305), add validation warnings (not errors) for:
  1. `metadata.plan_path` missing or empty → warning
  2. `metadata.plan_review_path` missing or empty → warning

**File: `src/foundry_mcp/core/spec/templates.py`**
- Add file existence validation in `create_spec()` (after specs_dir resolution at line 176):
  - Check that `plan_path` points to an existing file (resolve relative to specs_dir)
  - Check that `plan_review_path` points to an existing file (resolve relative to specs_dir)
- **Decision:** Validate at creation time (in `create_spec()`), not in `validate_spec()`. The general validator works with pure JSON data and has no filesystem context. Creation always has `specs_dir`.

### 4. Update fidelity gate to use plan context

**File: `src/foundry_mcp/tools/unified/documentation_helpers.py`**
- `_build_spec_overview()` (lines 305-396): Include `plan_path` and `plan_review_path` in overview output (after existing metadata fields, before the return at line 396)

**New function in `documentation_helpers.py`:**
- `_build_plan_context(spec_data, workspace_root)` — resolves `metadata.plan_path` relative to workspace_root, reads the plan file, and returns formatted markdown with key sections (mission, success criteria, constraints, risks). Returns empty string if plan_path is absent or file is unreadable. Note: unlike other `_build_*` functions which are pure data transforms, this one reads from disk. It needs `workspace_root` (already available in the fidelity review caller at `ws_path`).

**File: `src/foundry_mcp/tools/unified/review.py`**
- In fidelity review context building (lines 821-837): After `spec_overview` (line 822), call `_build_plan_context(spec_data, ws_path)` and include result in the `ConsultationRequest.context` dict (lines 860-871) as `"plan_context"` key. This gives the fidelity reviewer access to the original intent, not just the spec's interpretation of it.

### 5. Update documentation

**File: `docs/05-mcp-tool-reference.md`**
- Add `plan_path` and `plan_review_path` to `spec-create` parameters (parameter table around lines 377-387)
- Document as required fields for new specs
- Update example at line 419

---

## Design Notes

- **File existence validation at creation, not general validation.** `validate_spec()` is a pure data validator (takes JSON, returns diagnostics). Adding filesystem checks would break its contract. Instead, validate file existence in `create_spec()` before writing the spec file.
- **Relative vs absolute paths.** Store paths as relative to the specs directory (e.g., `.plans/feature.md` or `.plan-reviews/feature-review-full.md`). Resolve to absolute at read time using workspace context. This makes specs portable across machines.
- **plan_review_path stores the synthesized review**, not individual per-provider reviews. The `.plan-reviews/` directory may contain multiple files (one per provider + one synthesized). The `plan_review_path` points to the synthesized one.
- **`_build_plan_context()` reads from disk**, unlike other `_build_*` helpers which transform spec data in-memory. This is acceptable because: (a) it's called from review.py which already has filesystem context via `ws_path`, (b) it gracefully degrades (returns empty string) if the file is missing, and (c) plan context is supplementary — fidelity review works without it.
- **Backward compatibility.** Existing specs won't have these fields. Approach:
  - (A) Hard break: validation fails on old specs → forces migration
  - (B) Soft break: warn on old specs, error on new specs created after this change
  - **Recommendation:** (B). Add validation warning (not error) for missing `plan_path`/`plan_review_path`. But `spec-create` requires them as input parameters. This means old specs get a warning; new specs can't be created without them.
- **Parameters are required in `spec-create` but optional in `generate_spec_data()`/`create_spec()`.** The core functions accept `None` (for backward compat and testing), but the handler enforces them as required. This follows the same pattern as `mission` — optional at the core level, strongly recommended at the tool level.
