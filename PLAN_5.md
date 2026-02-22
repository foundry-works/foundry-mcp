# Plan 5: Plan Template Alignment

**Decision:** "Align Plan Template with Spec Schema"
**Dependencies:** Plan 1 (removals), Plan 2 (enrichment), Plan 4 (plan linkage)
**Risk:** Low — template strings and review prompt changes; no server logic changes
**Line references verified:** 2026-02-22 (post-Plans 1-4 implementation)

---

## Rationale

The spec review's purpose is to check alignment between the plan and the spec. But half the spec's fields have no corresponding section in the plan template. This means the LLM invents information during plan-to-spec translation — information the spec review can't verify.

After Plans 1-4, the spec schema is settled. Now the plan template must match it, making plan-to-spec translation mechanical instead of creative.

---

## Scope

### 1. Update plan creation templates (two locations)

The plan templates are embedded markdown strings in a `PLAN_TEMPLATES` dict. This dict is duplicated in two files:

**File: `src/foundry_mcp/tools/unified/plan.py`** (lines 136-207)
- `PLAN_TEMPLATES["simple"]` (lines 137-157): Basic template
- `PLAN_TEMPLATES["detailed"]` (lines 158-206): Comprehensive template
- Used by `perform_plan_create()` at line 645: `PLAN_TEMPLATES[template].format(name=name)`

**File: `src/foundry_mcp/cli/commands/plan.py`** (lines 385-456)
- Identical `PLAN_TEMPLATES` dict (copy-pasted)
- Used at line 536: `PLAN_TEMPLATES[template].format(name=name)`

**Action:** Update both `"detailed"` templates to the new structure. The `"simple"` template gets a lighter update (add Mission and Success Criteria, keep flat task list).

#### New "detailed" template

```markdown
# {name}

## Mission

[Single sentence describing the primary goal — becomes metadata.mission]

## Objectives

- [Objective 1 — each becomes an array item in metadata.objectives]
- [Objective 2]

## Success Criteria

- [ ] [Measurable criterion 1 — becomes metadata.success_criteria]
- [ ] [Measurable criterion 2]

## Assumptions

- [What we believe to be true — becomes metadata.assumptions]

## Constraints

- [Hard limits we must work within — becomes metadata.constraints]

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | [low/medium/high] | [low/medium/high] | [Mitigation strategy] |

## Open Questions

- [Unresolved question — becomes metadata.open_questions]

## Dependencies

- [External/internal dependencies]

## Phases

### Phase 1: [Phase Name]

**Goal:** [What this phase accomplishes — becomes phase purpose]

**Description:** [Detailed description — becomes phase description]

#### Tasks

- **[Task title]** `[task_category]` `[complexity]`
  - Description: [What to do]
  - File: [repo-relative path, or "N/A" for investigation/research/decision]
  - Acceptance criteria:
    - [How to verify this task is done]
    - [Another criterion]
  - Depends on: [other task titles, or "none"]

#### Verification

- **Run tests:** [test command]
- **Fidelity review:** Compare implementation to spec
- **Manual checks:** [any manual steps, or "none"]
```

#### New "simple" template

```markdown
# {name}

## Mission

[Single sentence describing the primary goal]

## Scope

[What is included/excluded from this plan]

## Tasks

1. **[Task 1]** `[complexity]`
   - File: [path or "N/A"]
   - Acceptance criteria:
     - [criterion]
2. **[Task 2]** `[complexity]`
   - File: [path or "N/A"]
   - Acceptance criteria:
     - [criterion]

## Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
```

### 2. Key changes from current templates

| Change | Why |
|--------|-----|
| **Mission** section added | Maps to `metadata.mission` — currently no template source |
| **Objectives** as bullet list | Maps to `metadata.objectives` — currently a paragraph under "Objective" |
| **Success Criteria** with spec mapping note | Maps to `metadata.success_criteria` (Plan 2) |
| **Assumptions** section added | Maps to `metadata.assumptions` — separate from constraints |
| **Constraints** section added | Maps to `metadata.constraints` (Plan 2) |
| **Risks** as structured table with Likelihood column | Maps to `metadata.risks` with `likelihood`/`impact`/`mitigation` (Plan 2) |
| **Open Questions** section added | Maps to `metadata.open_questions` (Plan 2) |
| `[task_category]` inline per task | No guessing during spec translation |
| `[complexity]` inline per task | Plans 1+2: replaces `estimated_hours` |
| Per-task **acceptance criteria** | Not invented during translation |
| Per-task **file path** (or "N/A") | Explicit — not omitted and guessed |
| Per-task **dependencies** | Stated in plan, not inferred during translation |
| Phase **Goal** + **Description** separated | Both `purpose` and `description` have plan sources |
| **No estimated_hours** | Plan 1: removed from spec |

### 3. What stays out of the plan

- `estimated_hours` — removed from spec (Plan 1)
- `spec_id`, timestamps, status, counts — auto-generated during spec creation
- Node IDs (`phase-1`, `task-1-1`) — structural, assigned during translation
- `plan_path`, `plan_review_path` — set during `spec-create` (Plan 4), not in plan content

### 4. DRY up the duplicate templates (optional improvement)

Currently the same `PLAN_TEMPLATES` dict is copy-pasted in two files. Options:
- **(A) Keep both in sync manually.** Simple, two places to update. This is acceptable since the templates change rarely.
- **(B) Extract to a shared module.** Create `src/foundry_mcp/core/plan_templates.py` and import from both locations.

**Recommendation:** (A) for now. The duplication is a known debt but not Plan 5's problem. Just update both in this plan. If template changes become frequent, extract later.

### 5. Update markdown plan review prompts

**File: `src/foundry_mcp/core/prompts/markdown_plan_review.py`**

This is where the review prompts live for reviewing **markdown plans** (not specs). The full review template `MARKDOWN_PLAN_REVIEW_FULL_V1` (lines 102-167) evaluates 6 dimensions: Completeness, Architecture, Sequencing, Feasibility, Risk, Clarity.

Update the `MARKDOWN_PLAN_REVIEW_FULL_V1` user_template to add specific checks for the new template sections. In the **Completeness** dimension (line 133), add:

```
1. **Completeness** - Are all required sections present?
   - Mission statement (single sentence)?
   - Objectives listed as discrete items?
   - Success criteria with measurable checkboxes?
   - Assumptions listed?
   - Constraints listed?
   - Risks with likelihood/impact/mitigation columns?
   - Open questions (if any unresolved items exist)?
   - Per-task: category tag, complexity tag, file path (or "N/A"), acceptance criteria, dependencies?
```

Also update the **response schema** `_RESPONSE_SCHEMA` (lines 27-74) category tags to include `[Template Compliance]` alongside existing categories.

**What NOT to change:** The Quick, Security, and Feasibility review templates. They have specialized focus areas and don't need template-compliance checks.

### 6. Update skill workflow documentation

The foundry-spec skill's workflow instructions (in the claude-foundry plugin) need updates to:
- Reference the new plan template sections
- Provide a plan section → spec JSON field mapping table
- Note that plan-to-spec translation should be mechanical, not creative

Since the skill lives in the plugin (not this repo), document the expected mapping here so the plugin maintainer can apply it:

| Plan Section | Spec JSON Field | Notes |
|-------------|----------------|-------|
| `## Mission` | `metadata.mission` | Single sentence |
| `## Objectives` bullet list | `metadata.objectives` array | One string per bullet |
| `## Success Criteria` checkboxes | `metadata.success_criteria` array | Strip `- [ ]` prefix |
| `## Assumptions` bullets | `metadata.assumptions` array | One string per bullet |
| `## Constraints` bullets | `metadata.constraints` array | One string per bullet |
| `## Risks` table rows | `metadata.risks` array of objects | `{description, likelihood, impact, mitigation}` |
| `## Open Questions` bullets | `metadata.open_questions` array | One string per bullet |
| `### Phase N: Name` | Phase node `title` | |
| Phase `**Goal:**` | Phase `metadata.purpose` | |
| Phase `**Description:**` | Phase `metadata.description` | |
| Task `**Title**` | Task node `title` | |
| Task `[task_category]` | Task `metadata.task_category` | |
| Task `[complexity]` | Task `metadata.complexity` | low/medium/high |
| Task `File:` | Task `metadata.file_path` | "N/A" → omit or set null |
| Task `Acceptance criteria:` | Task `metadata.acceptance_criteria` array | |
| Task `Depends on:` | Task blocker relationships | Via `task block` after creation |

---

## Design Notes

- **No template versioning needed.** The old templates were v1 scaffolds. Replacing them in-place is fine — there's no need to support old-format plans alongside new-format plans. The templates are guidance for LLMs, not parsed by code.
- **The plan template is guidance, not enforcement.** The server doesn't validate plan markdown structure. The template tells the LLM what sections to include; the plan review checks compliance.
- **"N/A" for file_path is explicit.** Investigation, research, and decision tasks don't touch files. Making the LLM write "N/A" is better than letting it omit the field and having the spec translator guess.
- **`_RESPONSE_SCHEMA` is shared** across all four markdown plan review templates via the builder. Updating it once propagates to all review types. The `[Template Compliance]` category tag is additive — it doesn't break existing review responses.
- **Plan 6 dependency.** Plan 6 (spec review diffs against plan) depends on this plan to align the plan structure. Once the plan template matches the spec schema, Plan 6's comparison prompts can reference specific plan sections with confidence.
