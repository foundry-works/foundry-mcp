# Plan 1: Schema Removals — estimated_hours + instructions

**Decisions:** "Remove Estimated Hours from Specs" + "Drop instructions Array from Task Schema Guidance"
**Dependencies:** None — this is the starting point
**Risk:** Low — pure removals with no new features

---

## Rationale

Remove dead weight before adding new fields. Both `estimated_hours` and `instructions` are noise:
- `estimated_hours` is fiction (LLMs guess), creates false precision, and drives no behavior
- `instructions` is redundant with `description` + `acceptance_criteria`, over-specifies implementation, and no server code reads it

---

## Scope

### 1. Remove `estimated_hours` from core/spec

**File: `src/foundry_mcp/core/spec/templates.py`**
- `get_phase_template_structure()` (lines 74-233): Remove `estimated_hours` from all 5 phase template definitions and their task definitions
- `apply_phase_template()` (lines 236-324): Remove `estimated_hours` parameter handling
- `generate_spec_data()` (lines 327-391): Remove any `estimated_hours` initialization
- `create_spec()` (lines 394-470): Remove `estimated_hours` from spec creation flow
- `add_assumption()`, `add_revision()`, etc.: Unlikely to reference hours, but verify

**File: `src/foundry_mcp/core/spec/hierarchy.py`**
- `add_phase()` (lines 167-310): Remove `estimated_hours` parameter and any hours-related logic
- `add_phase_bulk()` (lines 313-656): Remove `estimated_hours` validation (line ~408: "estimated_hours must be non-negative"), remove hours from task node construction
- `_add_phase_verification()` (lines 94-152): Check for hours references in verify node creation

**File: `src/foundry_mcp/core/spec/analysis.py`**
- `check_spec_completeness()` (lines 23-200+): Remove "estimates" from completeness weights (currently 25% weight at lines 17-20). Redistribute weights across remaining dimensions (titles, descriptions, file_paths)

**File: `src/foundry_mcp/core/spec/_constants.py`**
- No direct `estimated_hours` constant, but verify no references

### 2. Remove `estimated_hours` from validation

**File: `src/foundry_mcp/core/validation/rules.py`**
- Remove validation warnings for missing estimates (check around lines 415-523 in `_validate_nodes()`)

### 3. Remove `estimated_hours` from tool handlers

**File: `src/foundry_mcp/tools/unified/authoring_handlers/handlers_phase.py`**
- `_handle_phase_add()` (lines 109-223): Remove `estimated_hours` from schema and handler
- `_handle_phase_update_metadata()` (lines 226-349): Remove `estimated_hours` from updatable fields
- `_handle_phase_add_bulk()` (lines 352-650): Remove `estimated_hours` from task validation and construction
- `_handle_phase_template()` (lines 665-950): Hours shown in template listings — remove from "show" and "list" responses

**File: `src/foundry_mcp/tools/unified/spec.py`**
- `_handle_recalculate_hours()` (lines 983-1023): Remove this action handler entirely
- `_handle_completeness_check()` (lines 893-927): Will reflect analysis.py changes automatically

### 4. Remove `estimated_hours` from spec router

**File: `src/foundry_mcp/tools/unified/spec.py` (or its handler registration)**
- Remove `recalculate-hours` action from the spec router's action definitions
- Keep `recalculate-actual-hours` only if it serves a different purpose (actual vs estimated). If `actual_hours` is also being tracked, evaluate whether to keep or remove.

### 5. Remove hours recalculation exports

**File: `src/foundry_mcp/core/spec/__init__.py`**
- Remove `recalculate_estimated_hours` from exports (line ~130-131)
- Keep `recalculate_actual_hours` if it remains useful

### 6. Remove `instructions` from skill guidance

- Search for any `instructions` references in skill prompts, docs, or template files
- Since the skill files don't exist as formal files yet (the foundry-spec workflow is defined in the claude-foundry plugin, not in this repo), this action item becomes: "ensure no new guidance references `instructions`"
- Check `docs/05-mcp-tool-reference.md` and `docs/04-cli-command-reference.md` for any `instructions` mentions

### 7. Update documentation

**File: `docs/05-mcp-tool-reference.md`**
- Remove `estimated_hours` from authoring tool parameter documentation
- Remove `recalculate-hours` from spec tool action list

**File: `docs/04-cli-command-reference.md`**
- Remove any CLI references to `estimated_hours` or hour recalculation

---

## Design Notes

- **Completeness scoring redistribution:** With estimates removed (25% weight), redistribute to: titles (25%), descriptions (40%), file_paths (35%). Description quality matters most.
- **actual_hours:** The `actual_hours` field (and `recalculate-actual-hours`) may still be useful for tracking time spent. Evaluate independently — if nothing reads it, remove it too. If the autonomy runner logs actual time, keep it.
- **Backward compatibility:** Existing specs with `estimated_hours` in their JSON should still load. Don't strip the field on read — just stop writing, validating, or scoring it. The schema allows `additionalProperties`, so old data is harmless.
- **No migration needed:** We're not rewriting existing spec files. They keep their hours data; we just ignore it going forward.
