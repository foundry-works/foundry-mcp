# Plan 5 Checklist: Plan Template Alignment

## Plan template updates

### Detailed template (`PLAN_TEMPLATES["detailed"]`)
- [x] Update `tools/unified/plan.py`: Replace detailed template with new structure
- [x] Update `cli/commands/plan.py`: Replace detailed template (must match unified version)
- [x] New template includes: Mission, Objectives (bullet list), Success Criteria (checkboxes), Assumptions, Constraints, Risks (table with Likelihood/Impact/Mitigation), Open Questions, Dependencies
- [x] New template phase structure includes: Goal + Description (separate fields)
- [x] New template per-task structure includes: `[task_category]` tag, `[complexity]` tag, Description, File path (or "N/A"), Acceptance criteria (list), Depends on
- [x] New template phase verification section includes: Run tests, Fidelity review, Manual checks
- [x] No `estimated_hours` in template
- [x] Template uses `{name}` placeholder correctly for `.format(name=name)` call

### Simple template (`PLAN_TEMPLATES["simple"]`)
- [x] Update `tools/unified/plan.py`: Replace simple template with lighter new structure
- [x] Update `cli/commands/plan.py`: Replace simple template (must match unified version)
- [x] Simple template includes: Mission, Scope, Tasks with `[complexity]`/File/Acceptance criteria, Success Criteria
- [x] Template uses `{name}` placeholder correctly

## Markdown plan review prompt updates

### Full review template
- [x] Update `MARKDOWN_PLAN_REVIEW_FULL_V1` user_template in `core/prompts/markdown_plan_review.py`
- [x] Expand Completeness dimension to check for new required sections: Mission, Objectives, Success Criteria, Assumptions, Constraints, Risks (structured), Open Questions
- [x] Add per-task completeness checks: category tag, complexity tag, file path, acceptance criteria, dependencies
- [x] Replace Feasibility dimension with Over-engineering dimension (unnecessary abstractions, premature generalization, features beyond what was asked)

### Response schema
- [x] Replace `[Feasibility]` with `[Over-engineering]` in category tags
- [x] Dropped `[Template Compliance]` — redundant with expanded Completeness dimension

### What not to change
- [x] Confirm Quick review template (`MARKDOWN_PLAN_REVIEW_QUICK_V1`) left unchanged
- [x] Confirm Security review template (`MARKDOWN_PLAN_REVIEW_SECURITY_V1`) left unchanged
- [x] Confirm Feasibility review template (`MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1`) left unchanged

## Skill workflow documentation

- [x] Document plan section → spec JSON field mapping table in this repo (in PLAN_5.md design notes)
- [x] Document that plan-to-spec translation should be mechanical, not creative
- [x] Note which plan sections are new vs existing for plugin maintainer reference

## Verification

- [x] `plan.create` (MCP tool) generates plan with all new sections (template tested via unit tests)
- [x] Both templates render correctly as markdown (no broken formatting)
- [x] Unified tool template and CLI template are identical (no drift)
- [x] No references to `estimated_hours` or `instructions` in generated plans
- [x] Run test suite to confirm no regressions — 2,584 tests pass
