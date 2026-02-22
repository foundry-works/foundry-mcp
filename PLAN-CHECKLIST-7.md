# Plan 7 Checklist: Flatten Plan Templates

## Template dict → single string

- [x] `tools/unified/plan.py` lines 136-207: Replace `PLAN_TEMPLATES` dict with single `PLAN_TEMPLATE` string (use current detailed content)
- [x] `tools/unified/plan.py` line 645: `PLAN_TEMPLATES[template].format(...)` → `PLAN_TEMPLATE.format(...)`
- [x] `cli/commands/plan.py` lines 385-456: Replace `PLAN_TEMPLATES` dict with single `PLAN_TEMPLATE` string (must match unified version)
- [x] `cli/commands/plan.py` line 536: `PLAN_TEMPLATES[template].format(...)` → `PLAN_TEMPLATE.format(...)`

## Remove `template` parameter — unified tool

- [x] `tools/unified/plan.py` line 574: Remove `template` param from `perform_plan_create()` signature
- [x] `tools/unified/plan.py` lines 579-591: Delete `if template not in PLAN_TEMPLATES` validation block
- [x] `tools/unified/plan.py` line 661: Remove `"template": template` from metric labels
- [x] `tools/unified/plan.py` line 670: Remove `"template": template` from response details
- [x] `tools/unified/plan.py` line 744: Remove `template = payload.get("template", "detailed")` from `_handle_plan_create`
- [x] `tools/unified/plan.py` line 754: Remove `template=template` from `perform_plan_create()` call
- [x] `tools/unified/plan.py` line 821: Remove `template: str = "detailed"` from tool registration signature
- [x] `tools/unified/plan.py` line 833: Remove `"template": template` from payload dict

## Remove `--template` — CLI

- [x] `cli/commands/plan.py` lines 469-473: Delete `@click.option("--template", ...)` decorator
- [x] `cli/commands/plan.py` line 482: Remove `template: str` from `plan_create_cmd()` function signature
- [x] `cli/commands/plan.py` line 557: Remove `"template": template` from response output

## Update text

- [x] `tools/unified/plan.py` line 736: Update `_ACTION_SUMMARY["create"]` — remove "templates" wording

## Documentation

- [x] `docs/04-cli-command-reference.md` line 579: Remove `--template` row from parameters table
- [x] `docs/04-cli-command-reference.md` line 581: Remove "with the specified template" from description
- [x] `docs/04-cli-command-reference.md` line 587: Remove `--template simple` example

## Verification

- [x] `plan create` (MCP tool) works without `template` parameter
- [x] `plan create` (MCP tool) silently ignores `template` if passed (no error)
- [x] `foundry plan create "Test"` (CLI) works without `--template` flag
- [x] Generated plan content matches the single template
- [x] No remaining references to `PLAN_TEMPLATES` (plural) in `src/`
- [x] No remaining references to `"simple"` template choice in `src/`
- [x] Run test suite — no regressions
