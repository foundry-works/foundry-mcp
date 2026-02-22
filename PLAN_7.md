# Plan 7: Flatten Plan Templates

**Decision:** Remove the simple/detailed template selection; single plan template
**Dependencies:** None — can run before or after Plan 5 (template content), but Plan 5 should update the single template, not two
**Risk:** Low — removes code and a parameter; no new logic

---

## Rationale

The `plan create` action currently offers two templates (`simple` and `detailed`) via a `template` parameter. In practice, `detailed` is always the right choice — a "simple" plan that omits phases, risks, and structure just produces a worse spec. The selection mechanism adds parameter surface, validation code, CLI options, and a duplicated dict for zero user value. Remove it.

---

## Scope

### 1. Replace `PLAN_TEMPLATES` dict with single `PLAN_TEMPLATE` string

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Lines 136-207: Replace `PLAN_TEMPLATES = { "simple": ..., "detailed": ... }` with a single `PLAN_TEMPLATE = """..."""` string containing the (current or updated) detailed template content
- Line 645: Change `PLAN_TEMPLATES[template].format(name=name)` → `PLAN_TEMPLATE.format(name=name)`

**File: `src/foundry_mcp/cli/commands/plan.py`**
- Lines 385-456: Same replacement — `PLAN_TEMPLATES` dict → single `PLAN_TEMPLATE` string
- Line 536: Change `PLAN_TEMPLATES[template].format(name=name)` → `PLAN_TEMPLATE.format(name=name)`

### 2. Remove `template` parameter from `perform_plan_create()`

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Line 574: `perform_plan_create(name: str, template: str = "detailed")` → `perform_plan_create(name: str)`
- Lines 579-591: Delete the `if template not in PLAN_TEMPLATES` validation block entirely
- Line 661: Remove `"template": template` from metric labels
- Line 670: Remove `"template": template` from response details

### 3. Remove `template` from MCP tool handler and registration

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Lines 742-754 (`_handle_plan_create`): Remove `template = payload.get("template", "detailed")` (line 744) and remove `template=template` from `perform_plan_create()` call (line 754)
- Lines 810-838 (`register_unified_plan_tool`): Remove `template: str = "detailed"` parameter (line 821) and `"template": template` from payload dict (line 833)

### 4. Remove `--template` from CLI command

**File: `src/foundry_mcp/cli/commands/plan.py`**
- Lines 469-473: Delete the `@click.option("--template", ...)` decorator
- Line 482: Remove `template: str` from `plan_create_cmd()` signature
- Line 557: Remove `"template": template` from response output

### 5. Update action summary

**File: `src/foundry_mcp/tools/unified/plan.py`**
- Line 736: Change `"create": "Create markdown plan templates in specs/.plans"` → `"create": "Create a markdown implementation plan in specs/.plans"`

### 6. Update documentation

**File: `docs/04-cli-command-reference.md`**
- Line 579: Remove `--template` row from parameters table
- Line 581: Update description (remove "with the specified template")
- Line 587: Remove or update the `--template simple` example

---

## Design Notes

- **Backward compatibility for the MCP tool parameter.** If a caller passes `template="detailed"` or `template="simple"`, the parameter is simply ignored (it's removed from the handler). No error, no behavior change — the single template is always used. This is safe because the parameter was optional with a default.
- **Backward compatibility for the CLI.** Removing a `click.option` means `--template simple` will error. This is an acceptable breaking change for a CLI beta. The option was rarely used (default was already `detailed`).
- **Ordering with Plan 5.** If Plan 7 runs first, Plan 5 updates one template string instead of two. If Plan 5 runs first, Plan 7 collapses the two updated templates into one. Either order works — but Plan 7 first is cleaner (Plan 5 has less to touch).
- **The template content itself is not this plan's concern.** Plan 7 removes the selection mechanism. Plan 5 updates the template content. They are orthogonal.
