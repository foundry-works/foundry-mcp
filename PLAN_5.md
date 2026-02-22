# Plan 5: Plan Template Alignment

**Decision:** "Align Plan Template with Spec Schema"
**Dependencies:** Plan 1 (removals), Plan 2 (enrichment), Plan 4 (plan linkage)
**Risk:** Low — documentation/template changes, no server code logic changes

---

## Rationale

The spec review's purpose is to check alignment between the plan and the spec. But half the spec's fields have no corresponding section in the plan template. This means the LLM invents information during plan-to-spec translation — information the spec review can't verify.

After Plans 1-4, the spec schema is settled. Now the plan template must match it, making plan-to-spec translation mechanical instead of creative.

---

## Scope

### 1. Update plan template

**File: Plan template in the foundry-spec skill (managed by the claude-foundry plugin)**

The plan template lives outside this repo (in the claude-foundry plugin). However, the `plan create` action in `tools/unified/plan.py` (lines 574-674) generates plan templates. Check if the template content is embedded there or loaded from a file.

**File: `src/foundry_mcp/tools/unified/plan.py`**
- `_handle_plan_create()` (lines 574-674): Find where the template content is defined
- Update the template to the new structure:

```markdown
# {Feature Name} Implementation Plan

## Mission
{Single sentence — becomes metadata.mission}

## Objective
- {Objective 1 — each becomes an array item in metadata.objectives}
- {Objective 2}

## Success Criteria
- [ ] {Measurable criterion 1}
- [ ] {Measurable criterion 2}

## Assumptions
- {What we believe to be true — becomes metadata.assumptions}

## Constraints
- {Hard limits we must work within — becomes metadata.constraints}

## Risks
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|

## Open Questions
- {Unresolved questions — becomes metadata.open_questions}

## Dependencies
- {External/internal dependencies}

## Phases

### Phase 1: {Phase Name}
**Goal:** {What this phase accomplishes — becomes phase purpose}
**Description:** {Detailed description — becomes phase description}

#### Tasks

- **{Task title}** [{task_category}] [{complexity}]
  - Description: {What to do}
  - File: {repo-relative path, or "N/A" for investigation/research/decision}
  - Acceptance criteria:
    - {How to verify this task is done}
    - {Another criterion}
  - Depends on: {other task titles, or "none"}

#### Verification
- **Run tests:** {test command}
- **Fidelity review:** Compare implementation to spec
- **Manual checks:** {any manual steps, or "none"}
```

### 2. Key changes from current template

| Change | Why |
|--------|-----|
| `[task_category]` inline per task | No guessing during translation |
| `[complexity]` inline per task | Plans 1+2: replaces estimated_hours |
| Per-task acceptance criteria | Not invented during translation |
| Per-task file path | "N/A" is explicit, not omitted |
| Per-task dependencies | Task-to-task deps stated in plan, not inferred |
| **Assumptions** section added | Separate from constraints (different concepts) |
| **Phase description** added | Both `purpose` and `description` have sources |
| **Objectives** as list | Array items, not a paragraph |
| **No estimated_hours** | Plan 1: removed from spec |

### 3. What stays out of the plan

- `estimated_hours` — removed from spec (Plan 1)
- `spec_id`, timestamps, status, counts — auto-generated
- Node IDs (`phase-1`, `task-1-1`) — structural, assigned during translation

### 4. Update skill workflow instructions

The foundry-spec skill's workflow instructions (in the claude-foundry plugin) need updates to:
- Reference the new plan template sections
- Explain the mapping: plan section → spec JSON field
- Note that translation should be mechanical, not creative

Since the skill lives in the plugin, document the expected changes here for the plugin maintainer.

### 5. Update plan review dimensions

**File: `src/foundry_mcp/core/prompts/plan_review.py`**
- Update the plan review prompt to evaluate the new sections:
  - Are assumptions listed? (Previously not in template)
  - Does each task have acceptance criteria? (Previously plan-level only)
  - Does each task have a `[task_category]` tag?
  - Does each task have a `[complexity]` tag?
  - Does each task have a file path or explicit "N/A"?
  - Are task-to-task dependencies stated?

### 6. Update spec review to check plan section coverage

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- The spec review context (lines 235-247) should include information about what the plan template requires, so the review can check that the spec covers all plan sections
- This is preparation for Plan 6 (spec review diffs against plan)

---

## Design Notes

- **Template versions:** If the plan template is embedded in code, consider versioning it (e.g., `PLAN_TEMPLATE_V2`). This lets old reviews reference the old template while new plans use the new one.
- **The plan template is guidance, not enforcement.** The server doesn't validate plan markdown structure. The template tells the LLM what sections to include; the plan review checks compliance.
- **"N/A" for file_path is explicit.** Investigation, research, and decision tasks don't touch files. Making the LLM write "N/A" is better than letting it omit the field and having the spec translator guess.
