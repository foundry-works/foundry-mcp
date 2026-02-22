# Plan 3: Phase Template Removal + Verification Scaffolding Migration

**Decision:** "Remove Phase Templates, Keep `phase-add-bulk` as Primary Path"
**Dependencies:** Ideally after Plan 1 (estimated_hours removed from templates), but can proceed independently
**Risk:** Medium — removing a working feature, must ensure verification scaffolding migrates correctly
**Line references verified:** 2026-02-22 (post-Plan 2 implementation)

---

## Rationale

Phase templates are a side door that duplicates what the LLM already does when building custom phases from an approved plan. The one useful thing templates provide — auto-appending verification scaffolding (run-tests + fidelity review) — should be built into `phase-add-bulk` itself so every phase gets verification regardless of how it was created.

---

## Scope

### 1. Move verification scaffolding into `phase-add-bulk`

**File: `src/foundry_mcp/core/spec/hierarchy.py`**

The function `_add_phase_verification()` (line 95) already exists and adds run-tests + fidelity verify nodes to a phase. Currently called by:
- `add_phase()` (single phase creation, call site at line 285) — already has scaffolding
- `apply_phase_template()` in templates.py — called after template application

`add_phase_bulk()` (line 307) does **NOT** auto-append verification. Its docstring (lines 319-324) explicitly states: _"Creates a phase and all specified tasks/verify nodes without auto-generating verification scaffolding."_ It relies on the caller (or template) to include verify nodes in the task list.

**Change:** At the end of `add_phase_bulk()`, after all tasks are added to the phase, call `_add_phase_verification(hierarchy, phase_num, phase_id)` to auto-append run-tests + fidelity verify nodes — BUT only if the caller hasn't already included verify nodes in the task list. Check for existing verify-type nodes in the provided tasks to avoid duplication. Also update the docstring to reflect the new behavior.

**Logic:**
```python
# After adding all tasks from the tasks array
has_verify_nodes = any(t.get("type") == "verify" for t in tasks)
if not has_verify_nodes:
    _add_phase_verification(hierarchy, phase_num, phase_id)
```

### 2. Remove phase template code from `core/spec/templates.py`

**Functions to remove:**
- `get_phase_template_structure()` (line 74) — the 5 template definitions and the function that returns them
- `apply_phase_template()` (line 220) — applies template to spec

**Functions to keep:**
- `get_template_structure()` (line 30) — this handles the `"empty"` spec-level template, which is a different concept
- `generate_spec_data()`, `create_spec()` — spec creation, unrelated to phase templates
- `add_assumption()`, `add_revision()`, etc. — metadata operations

### 3. Remove `PHASE_TEMPLATES` constant

**File: `src/foundry_mcp/core/spec/_constants.py`**
- Remove `PHASE_TEMPLATES = ("planning", "implementation", "testing", "security", "documentation")` (line 26)

**File: `src/foundry_mcp/core/spec/__init__.py`**
- Remove `PHASE_TEMPLATES` from import (line 24) and `__all__` (line 99)
- Remove `get_phase_template_structure` from import (line 79) and `__all__` (line 127)
- Remove `apply_phase_template` from import (line 75) and `__all__` (line 126)

### 4. Remove `phase-template` action from authoring router

**File: `src/foundry_mcp/tools/unified/authoring_handlers/handlers_phase.py`**
- Remove `_handle_phase_template()` (lines 622-911) — the entire handler for list/show/apply template operations (290 lines)

**File: `src/foundry_mcp/tools/unified/authoring_handlers/__init__.py`**
- Remove `_handle_phase_template` import (line 37)
- Remove `ActionDefinition(name="phase-template", ...)` registration (lines 93-96)

### 5. Update skill reference docs

**File: `docs/05-mcp-tool-reference.md`**
- Remove `phase-template` from authoring action table (line 17 — full action list row, line 357 — detailed row)
- Update `phase-add-bulk` documentation to note that verification scaffolding is auto-appended

**File: `docs/04-cli-command-reference.md`**
- Remove indirect phase template reference (line 61 — `--template` option note mentioning "use phase templates to add structure")

**File: `docs/07-troubleshooting.md`**
- No phase template references found — no changes needed

### 6. Keep `spec-create` with `"empty"` template

The `"empty"` template in `get_template_structure()` and `TEMPLATES = ("empty",)` is a different concept — it's the initial spec skeleton, not a phase template. This stays untouched.

---

## Design Notes

- **Verification scaffolding deduplication:** The check `any(t.get("type") == "verify" for t in tasks)` prevents double-appending. If a caller explicitly includes verify nodes (e.g., they want custom verification), the auto-append skips. This is the right behavior — explicit beats implicit.
- **No behavioral change for `add_phase()`:** Single phase creation already calls `_add_phase_verification()` (line 285). No change needed.
- **Docstring update required:** `add_phase_bulk()` docstring (lines 319-324) currently says "without auto-generating verification scaffolding" — this must be updated to reflect the new auto-append behavior.
- **Test impact:** Any tests that call `apply_phase_template()` or `get_phase_template_structure()` need removal or rewriting. Tests for `add_phase_bulk()` should be updated to verify that verification scaffolding is auto-appended.
- **The `_add_phase_verification` function stays in hierarchy.py.** It's shared between `add_phase()` and `add_phase_bulk()` — it's a hierarchy concern, not a template concern.
