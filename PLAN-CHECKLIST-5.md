# Plan 5 Checklist: Plan Template Alignment

## Plan template update

- [ ] Locate plan template content in `tools/unified/plan.py` `_handle_plan_create()` (or wherever it's defined)
- [ ] Update template to include: Mission, Objectives (list), Success Criteria, Assumptions, Constraints, Risks (table), Open Questions, Dependencies
- [ ] Update per-phase template to include: Goal + Description (separate fields)
- [ ] Update per-task template to include: `[task_category]` tag, `[complexity]` tag, Description, File path (or "N/A"), Acceptance criteria (list), Depends on
- [ ] Update per-phase verification section template
- [ ] Remove `estimated_hours` from plan template (if present)
- [ ] Remove `instructions` from plan template (if present)

## Plan review prompt updates

- [ ] Update plan review prompt in `core/prompts/plan_review.py` to evaluate new sections
- [ ] Add review dimension: Are assumptions listed?
- [ ] Add review dimension: Does each task have acceptance criteria?
- [ ] Add review dimension: Does each task have `[task_category]`?
- [ ] Add review dimension: Does each task have `[complexity]`?
- [ ] Add review dimension: Does each task have file path or "N/A"?
- [ ] Add review dimension: Are task-to-task dependencies stated?

## Skill workflow documentation

- [ ] Document plan section → spec JSON field mapping for skill maintainer
- [ ] Document that plan-to-spec translation should be mechanical, not creative
- [ ] Note which plan sections are new vs existing

## Verification

- [ ] `plan create` generates a plan with all new sections
- [ ] Plan review evaluates the new sections without errors
- [ ] Plan template renders correctly in markdown
- [ ] No references to `estimated_hours` or `instructions` in generated plans
