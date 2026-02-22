# Plan 2: Schema Enrichment — Metadata Fields + Task Complexity

**Decisions:** "Add Missing Plan Sections to Spec Schema" + "Add Task-Level Complexity"
**Dependencies:** None strictly, but cleaner after Plan 1 removals
**Risk:** Low-medium — additive changes, backward compatible

---

## Rationale

The plan template captures success criteria, constraints, risks, and open questions, but the spec JSON has nowhere to store them. This means the implementing LLM has less context than the planning LLM. Similarly, `complexity` is the useful core of what `estimated_hours` tried to express, without the false precision.

---

## Scope

### 1. Add metadata fields to spec schema

**New fields in `metadata`:**

```json
{
  "metadata": {
    "success_criteria": [],
    "constraints": [],
    "risks": [
      {
        "description": "...",
        "likelihood": "low|medium|high",
        "impact": "low|medium|high",
        "mitigation": "..."
      }
    ],
    "open_questions": []
  }
}
```

**File: `src/foundry_mcp/core/spec/templates.py`**
- `generate_spec_data()` (lines 327-391): Initialize `success_criteria: []`, `constraints: []`, `risks: []`, `open_questions: []` in metadata alongside existing `objectives` and `assumptions`

**File: `src/foundry_mcp/core/spec/_constants.py`**
- Update `FRONTMATTER_KEYS` to include the new fields if they should be updatable via `spec-update-frontmatter`. But per the design notes in DECISIONS.md — block array fields that have dedicated handlers. So these should NOT be in `FRONTMATTER_KEYS`. Instead, they get dedicated add/list actions like `assumptions` currently has.

### 2. Add validation for new metadata fields

**File: `src/foundry_mcp/core/validation/rules.py`**
- In `_validate_structure()` (lines 178-258): Add validation warnings (not errors) when `success_criteria` or `constraints` are empty arrays
- Add `risks` object validation: each risk must have `description` (required string); `likelihood`, `impact`, `mitigation` are optional
- Valid `likelihood` values: `low`, `medium`, `high`
- Valid `impact` values: `low`, `medium`, `high`
- `open_questions` and `success_criteria` — no structural validation needed (string arrays)

### 3. Add authoring actions for new fields

**File: `src/foundry_mcp/tools/unified/authoring_handlers/handlers_metadata.py`** (or a new file if this one is already large)

New actions, following the existing `assumption-add` / `list-assumptions` pattern:

| Action | Handler | Pattern |
|--------|---------|---------|
| `constraint-add` | `_handle_constraint_add()` | Like `assumption-add` — appends string to `metadata.constraints` |
| `risk-add` | `_handle_risk_add()` | Appends risk object to `metadata.risks`; validates required `description` |
| `question-add` | `_handle_question_add()` | Appends string to `metadata.open_questions` |
| `success-criterion-add` | `_handle_success_criterion_add()` | Appends string to `metadata.success_criteria` |

**Registration in `authoring_handlers/__init__.py`:**
- Add ActionDefinition entries for each new action
- Follow existing naming convention (noun-verb)

**File: `src/foundry_mcp/core/spec/templates.py`**
- Add handler functions parallel to `add_assumption()` (lines 473-561):
  - `add_constraint(spec_id, text, specs_dir)` → appends to `metadata.constraints`
  - `add_risk(spec_id, description, likelihood, impact, mitigation, specs_dir)` → appends to `metadata.risks`
  - `add_question(spec_id, text, specs_dir)` → appends to `metadata.open_questions`
  - `add_success_criterion(spec_id, text, specs_dir)` → appends to `metadata.success_criteria`
- Add corresponding list functions parallel to `list_assumptions()` (lines 664-723)

### 4. Block new array fields in update_frontmatter

**File: `src/foundry_mcp/core/spec/templates.py`**
- `update_frontmatter()` (lines 726-826): Add `success_criteria`, `constraints`, `risks`, `open_questions` to the blocked fields list (alongside `assumptions`, `revision_history`)

### 5. Add `complexity` to task nodes

**Valid values:** `low`, `medium`, `high`

**File: `src/foundry_mcp/core/validation/rules.py`**
- In `_validate_nodes()` (lines 415-523): Add validation for `metadata.complexity` on task nodes
  - Valid values: `low`, `medium`, `high`
  - Warning (not error) if complexity is missing on task nodes

**File: `src/foundry_mcp/core/spec/hierarchy.py`**
- `add_phase_bulk()` (lines 313-656): Accept and store `complexity` in task metadata
  - Validate values if provided: must be `low`, `medium`, or `high`
  - No default — if not provided, it's simply absent (warning, not error)

**File: `src/foundry_mcp/core/spec/_constants.py`**
- Add `COMPLEXITY_LEVELS = ("low", "medium", "high")`

### 6. Update spec overview for fidelity context

**File: `src/foundry_mcp/tools/unified/documentation_helpers.py`**
- `_build_spec_overview()` (lines 300-354): Include new metadata fields (success_criteria, constraints, risks, open_questions) in the overview passed to fidelity review LLM
- `_build_spec_requirements()` (lines 12-87): Include task complexity in requirement extraction

### 7. Update documentation

**File: `docs/05-mcp-tool-reference.md`**
- Add new authoring actions: `constraint-add`, `risk-add`, `question-add`, `success-criterion-add`
- Document new metadata fields in spec schema reference

---

## Design Notes

- **All new fields are optional with empty-array defaults.** Existing specs remain valid. No migration.
- **Validation warns, not errors**, for empty `success_criteria` and `constraints`. These are strongly recommended but shouldn't block spec creation.
- **`risks` is an array of objects** because structure (likelihood/impact/mitigation) is the whole point. String arrays would repeat the `assumptions` mistake.
- **`complexity` is task-level only.** Phases don't have independent complexity — they inherit it from their tasks. Spec-level complexity is the aggregate.
- **No `list-constraints`, `list-risks`, etc. in v1.** The `spec get` action already returns the full spec JSON including these fields. Dedicated list actions can be added later if needed. Actually — follow the `assumptions` pattern: add `list-constraints`, `list-risks`, `list-questions`, `list-success-criteria` for consistency.
