# Spec System Design Decisions

## Decision: Remove Phase Templates, Keep `phase-add-bulk` as Primary Path

**Date:** 2026-02-22
**Status:** Decided

### Context

The spec system has two ways to add phases to a spec:

1. **Phase templates** (`authoring action="phase-template"`) — 5 pre-built templates (`planning`, `implementation`, `testing`, `security`, `documentation`) that generate phases with pre-configured tasks, acceptance criteria, estimated hours, and auto-appended verification scaffolding (run-tests + fidelity review nodes).

2. **`phase-add-bulk`** (`authoring action="phase-add-bulk"`) — Manual phase construction where the LLM builds custom phases with arbitrary task structures from a user-approved markdown plan.

The server code for templates is solid — they generate valid phases, pass validation, and work correctly. However, the foundry-spec skill's 9-step workflow uses `phase-add-bulk` as the primary mechanism, not templates. Templates are mentioned in reference docs (`phase-authoring.md`, `troubleshooting.md`) as a convenience alternative but are not woven into the core workflow. The skill provides no decision rules for when to use each specific template.

The actual workflow is: **markdown plan → human approval → custom phases via `phase-add-bulk`**. Templates are a side door.

### Decision

Simplify by removing phase templates. `phase-add-bulk` is the sole phase creation path.

### Rationale

- **The LLM already does the thinking.** The mandatory markdown plan defines custom phases with goals, files, tasks, and verification. Templates would override or duplicate that thinking.
- **Template content gets thrown away.** For real specs, the LLM ends up replacing all template task content with what the plan actually calls for. The template is a wasted intermediate step.
- **Templates constrain when they shouldn't.** When the template doesn't match what was planned, it produces a worse outcome than building directly from the plan.
- **Most phases need custom structure.** Cookie-cutter task patterns rarely match real work. The "standard" phases (testing, docs, security) still need project-specific tasks.
- **The one useful thing templates do can move elsewhere.** Auto-appending verification scaffolding (run-tests + fidelity review) should be built into `phase-add-bulk` itself, so every phase gets verification regardless of creation method.

### Action Items

- [ ] Move verification scaffolding auto-append into `phase-add-bulk`
- [ ] Remove phase template code from `core/spec/templates.py` (the `get_phase_template_structure`, `apply_phase_template` functions and the 5 template definitions)
- [ ] Remove `phase-template` action from `authoring` router
- [ ] Remove `PHASE_TEMPLATES` constant and related exports from `core/spec/__init__.py`
- [ ] Update skill reference docs (`phase-authoring.md`, `troubleshooting.md`) to remove template mentions
- [ ] Keep `spec-create` with the `"empty"` template — that's just the initial skeleton, not the same concept

---

## Decision: Add Missing Plan Sections to Spec Schema

**Date:** 2026-02-22
**Status:** Decided

### Context

The foundry-spec skill requires a mandatory markdown plan before JSON spec creation. The plan template includes sections that have no home in the spec JSON schema. This means the LLM implementing the spec has less context than the LLM that planned it.

#### What transfers cleanly

| Plan Section | Spec JSON Location |
|---|---|
| Mission | `metadata.mission` |
| Objective | `metadata.objectives` (array) |
| Phases (goal, files, tasks) | `hierarchy` nodes |
| Verification | `verify` nodes with `verification_type` |
| Dependencies | `node.dependencies.{blocks, blocked_by, depends}` |

#### What gets lost

| Plan Section | Problem |
|---|---|
| **Success Criteria** | Only exists at task level as `acceptance_criteria`. No spec-level "definition of done." The plan says "here's how we know this whole spec is done" — but the spec can't store that. |
| **Constraints** | Shoved into `metadata.assumptions` as plain strings. Constraints and assumptions are different things. "Must support Python 3.9+" is a constraint. "The API response format won't change" is an assumption. |
| **Risks** | No field at all. Risk likelihood, impact, and mitigation are lost entirely or crammed into assumptions as unstructured text. |
| **Open Questions** | No field at all. Questions that could block tasks or change approach are lost between planning and implementation. |

### Decision

Add first-class fields for these four sections in `metadata`:

```json
{
  "metadata": {
    "mission": "...",
    "objectives": [],
    "assumptions": [],
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

### Rationale

- **Success criteria** are the spec-level definition of done. Without them, there's no way to verify the spec as a whole is complete — only individual tasks. The fidelity gate should be able to check these.
- **Constraints** mixed into assumptions is a type error. They serve different purposes: assumptions are things you believe to be true; constraints are hard limits you must work within. Separating them makes both more useful.
- **Risks** have structure (likelihood, impact, mitigation) that plain strings destroy. Structured risks let the implementation LLM make informed tradeoffs and let reviewers assess whether mitigations were followed.
- **Open questions** that stay visible during implementation prevent the LLM from silently making decisions that should have been escalated. They can also inform task blocking — a task that depends on an unresolved question should be blocked.

### Design Notes

- `success_criteria`, `constraints`, and `open_questions` are simple string arrays — same pattern as `assumptions` and `objectives`. No over-engineering.
- `risks` is an array of objects because the structure (likelihood/impact/mitigation) is the whole point. Flattening to strings would repeat the assumptions mistake.
- All new fields are optional with empty-array defaults. Existing specs remain valid.
- Validation should warn (not error) if these are empty — they're strongly recommended but shouldn't block spec creation.

### Action Items

- [ ] Add `success_criteria`, `constraints`, `open_questions` (string arrays) and `risks` (object array) to spec schema
- [ ] Update `generate_spec_data()` in `core/spec/templates.py` to initialize these as empty arrays
- [ ] Add validation warnings in `core/spec/rules.py` for empty `success_criteria` and `constraints`
- [ ] Add `risks` object validation (required keys: `description`; optional: `likelihood`, `impact`, `mitigation`)
- [ ] Update `spec-update-frontmatter` to support the new fields (block array fields that have dedicated handlers if we add them)
- [ ] Add authoring actions: `constraint-add`, `risk-add`, `question-add` (parallel to existing `assumption-add`)
- [ ] Update skill plan template and spec creation workflow to map these sections into the new fields
- [ ] Update fidelity gate to check spec-level `success_criteria` in addition to task-level `acceptance_criteria`

---

## Decision: Drop `instructions` Array from Task Schema Guidance

**Date:** 2026-02-22
**Status:** Decided

### Context

The foundry-spec skill's `json-spec.md` shows an `instructions` array on tasks — actionable steps like "Create users table with id, email, password_hash", "Add sessions table for token storage". The server doesn't validate, reference, or read this field anywhere (`core/spec/` has zero hits). It survives only because node metadata allows `additionalProperties: true`, so it gets silently stored if the LLM includes it.

### Decision

Remove `instructions` from the skill's guidance. Do not add server-side support.

### Rationale

- **Redundant with `description` + `acceptance_criteria`.** The description says what to do. Acceptance criteria say how to verify it's done. A third "here are the steps" field adds nothing.
- **Over-specifies implementation.** Prescribing exact steps micromanages the implementing LLM instead of giving it a goal to hit. If the implementing LLM is capable enough to write code, it's capable enough to figure out the steps from a clear description and acceptance criteria.
- **If steps matter, they belong in the description.** A description that says "Create users table with id/email/password_hash columns and a sessions table for token storage, with appropriate indexes" carries the same information without a separate field.
- **Dead weight in the schema.** No code reads it, no validation checks it, no review gates reference it. It's noise.

### Action Items

- [ ] Remove `instructions` from `json-spec.md` example in the skill
- [ ] Remove any skill guidance that tells the LLM to include `instructions` on tasks

---

## Decision: Require Plan Linkage on Specs

**Date:** 2026-02-22
**Status:** Decided

### Context

The foundry-spec skill mandates a markdown plan before JSON spec creation, but the server has no awareness of this relationship. A spec doesn't know which plan it came from, and there's no way to trace back from implementation to planning rationale.

Server-side enforcement of the full planning *workflow* (plan → review → human approval → spec) was considered and rejected — the server can enforce data integrity but not process integrity. It can't distinguish "human approved" from "LLM said human approved."

However, **traceability** is a data concern, not a process concern. The spec should know its origin.

### Decision

Add `metadata.plan_path` as a required field on specs. The server validates that:

1. The field is present and non-empty (error if missing)
2. The path points to an existing file (error if not found)

### Schema

```json
{
  "metadata": {
    "plan_path": "specs/.plans/feature-name.md",
    "plan_review_path": "specs/.plan-reviews/feature-name-review-full.md",
    ...
  }
}
```

### Rationale

- **Traceability.** Anyone reviewing the spec (human, LLM, fidelity gate) can follow the link back to the planning rationale — why this approach was chosen, what risks were identified, what alternatives were considered.
- **Review receipt.** The plan review path proves the plan was evaluated before the spec was created. The server can't verify the review was read or acted on, but it can verify one was produced.
- **Fidelity gate context.** The fidelity review can read both the plan and its review to understand intent and known concerns — not just check task-level acceptance criteria.
- **Enforcement of planning + review.** Requiring both fields means a spec can't be created without a plan that was reviewed. Two file-existence checks. No workflow engine needed.
- **Cheap to implement.** Two metadata fields, two file-existence checks in validation. No new MCP actions, no workflow changes.

### What this does NOT enforce

- That a human approved the plan (or read the review)
- That the plan content is any good
- That review feedback was addressed

"That the spec actually reflects the plan" is now enforced — see next decision.

### Action Items

- [ ] Add `metadata.plan_path` and `metadata.plan_review_path` to spec schema
- [ ] Update `spec-create` to accept `plan_path` and `plan_review_path` parameters
- [ ] Add validation error in `core/spec/rules.py` if either path is missing or empty
- [ ] Add validation error if either path points to a non-existent file
- [ ] Update `generate_spec_data()` to include both paths
- [ ] Update skill workflow to pass both paths when calling `spec-create`
- [ ] Update fidelity gate to read the linked plan and review for additional context

---

## Decision: Spec Review Should Diff Against the Plan

**Date:** 2026-02-22
**Status:** Decided

### Context

Currently, spec review (`review action="spec"`) evaluates a spec in isolation — checking completeness, clarity, feasibility, architecture, risk management, and verification coverage. It doesn't compare the spec to the plan it was derived from.

With `metadata.plan_path` now required (see previous decision), the spec review has a concrete reference point. The plan represents what was agreed upon with the human. The spec is the JSON translation of that agreement. Any gap between the two is either a mistake or an undocumented deviation.

### Decision

After spec creation, the spec review process should read the linked plan and identify gaps or deviations between the plan and the spec. This becomes the primary purpose of spec review — not "is this spec good in the abstract?" but "does this spec faithfully represent what was planned?"

### What the review should check

1. **Coverage** — Every phase, task, and verification step in the plan has a corresponding node in the spec. Nothing was dropped during translation.
2. **Fidelity** — Spec tasks match the plan's intent. A plan task that says "implement OAuth2 with PKCE" shouldn't become a spec task that just says "add authentication."
3. **Success criteria mapping** — The plan's success criteria are reflected in the spec's `metadata.success_criteria` and/or task-level `acceptance_criteria`.
4. **Constraints preserved** — The plan's constraints appear in `metadata.constraints`.
5. **Risks preserved** — The plan's risk table is reflected in `metadata.risks`.
6. **Open questions preserved** — The plan's open questions appear in `metadata.open_questions`.
7. **Undocumented additions** — Anything in the spec that wasn't in the plan should be flagged. Not necessarily wrong, but should be justified.

### What the review should NOT do

- Re-evaluate whether the plan itself was good (that's the plan review's job)
- Check implementation code (that's the fidelity gate's job during task execution)
- Block on minor wording differences — the plan is markdown prose, the spec is structured JSON. Exact string matching is wrong. Semantic alignment is the goal.

### Rationale

- **Closes the traceability loop.** Plan → plan review → spec → spec review (against plan) → implementation → fidelity gate (against spec). Every stage checks against the previous one.
- **Catches translation errors.** The most common failure mode is the LLM dropping or diluting requirements when converting prose to JSON. This review catches that before implementation starts.
- **Makes plan_path useful.** Without this, `plan_path` is just metadata. With this, it's an active input to quality control.

### Action Items

- [ ] Update spec review to read `metadata.plan_path` and load plan content
- [ ] Add plan-vs-spec comparison as the primary review dimension (coverage, fidelity, preserved metadata)
- [ ] Flag undocumented spec additions that don't trace back to the plan
- [ ] Store spec review results with `metadata.spec_review_path` (parallel to `plan_review_path`)
- [ ] Update skill workflow to run spec review automatically after spec creation

---

## Decision: Remove Estimated Hours from Specs

**Date:** 2026-02-22
**Status:** Decided

### Context

The spec schema currently has `estimated_hours` at multiple levels — spec-level metadata, phase metadata, and per-task metadata. The validation warns if tasks are missing estimates. Phase templates include default hour estimates (planning: 4h, implementation: 8h, testing: 6h, etc.).

### Decision

Remove `estimated_hours` from the spec schema entirely — spec level, phase level, and task level.

### Rationale

- **Estimates are fiction.** LLMs generating specs have no reliable basis for time estimates. They produce plausible-looking numbers that don't reflect actual complexity, developer skill, or codebase familiarity.
- **They create false precision.** "2 hours" for a task suggests a confidence that doesn't exist. This misleads rather than informs.
- **They don't drive any behavior.** No gate, review, or workflow decision depends on estimated hours. They're collected but never acted on.
- **They're noise in the spec review.** If the spec review checks alignment with the plan, estimated hours would need to be in the plan too — propagating fiction further up the chain.

### Action Items

- [ ] Remove `estimated_hours` from spec schema at all levels (spec metadata, phase metadata, task metadata)
- [ ] Remove `estimated_hours` validation warnings from `core/spec/rules.py`
- [ ] Remove `estimated_hours` from completeness scoring in `check_spec_completeness()`
- [ ] Remove `estimated_hours` from phase template definitions
- [ ] Remove `estimated_hours` from skill guidance (`json-spec.md`, plan template)

---

## Decision: Align Plan Template with Spec Schema

**Date:** 2026-02-22
**Status:** Decided

### Context

The spec review's purpose is to check alignment between the plan and the spec (see "Spec Review Should Diff Against the Plan" decision). But half the spec's required fields have no corresponding section in the plan template. This means the LLM invents information during plan-to-spec translation that the spec review can't verify against the plan.

Fields the spec requires that the plan doesn't ask for:

| Spec field | Plan gap |
|---|---|
| `task_category` per task | No guidance. LLM guesses investigation/implementation/refactoring/decision/research. |
| `acceptance_criteria` per task | Not in plan. Only spec-level success criteria. LLM invents per-task criteria. |
| `file_path` per task | Plan has "Files:" at phase level only. LLM distributes file assignments. |
| `assumptions` | No section. Plan has constraints and risks but not assumptions. |
| Task-level dependencies | Plan has "Dependencies" section but external/internal, not task-to-task blocks/blocked_by. |
| Phase `description` | Plan has "Goal:" (maps to `purpose`) but `description` is a separate field with no source. |
| Verification node structure | Plan lists verification as bullets, not structured nodes with commands/expected output. |

If the plan doesn't contain this information, the spec review can't check it for alignment, and the translation step becomes a creative exercise instead of a mechanical one.

### Decision

Expand the plan template so that every substantive spec field has a plan-side source. The plan-to-spec translation should be as mechanical as possible — the LLM structures existing information, not invents new information.

### Plan Template Changes

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

### Key Changes from Current Template

1. **`task_category` inline** — Each task is tagged with its category in brackets. No guessing during translation.
2. **Per-task acceptance criteria** — Each task lists its own criteria. Not invented during translation.
3. **Per-task file path** — Each task names its file. "N/A" for non-implementation tasks is explicit, not omitted.
4. **Per-task dependencies** — Task-to-task dependencies are stated in the plan, not inferred during translation.
5. **Assumptions section added** — Separate from constraints (different concepts, see earlier decision).
6. **Phase description added** — Separate from "Goal:" so both `purpose` and `description` have sources.
7. **Objectives as list** — Array items, not a paragraph, matching the schema.

### What stays out of the plan

- `estimated_hours` — removed from spec entirely (see previous decision)
- `spec_id`, timestamps, status, counts — auto-generated, not plan content
- Node IDs (`phase-1`, `task-1-1`) — structural, assigned during translation

### Rationale

- **Mechanical translation.** Every substantive spec field maps to a specific plan section. The LLM structures; it doesn't create.
- **Spec review becomes checkable.** With full coverage, the spec review can verify every field against the plan. No blind spots.
- **Better plans.** Forcing the planner to think about per-task acceptance criteria, file paths, and categories produces better plans — not just better specs.
- **Human sees everything.** The human approving the plan sees all the information that will end up in the spec. No surprises during translation.

### Action Items

- [ ] Update plan template in skill (`phase-plan-template.md`) with the new structure
- [ ] Update skill workflow instructions to reflect new plan sections
- [ ] Update spec review checklist to verify all plan sections map to spec fields
- [ ] Update plan review dimensions to cover the new sections (assumptions, per-task criteria, etc.)

---

## Decision: Add Task-Level Complexity

**Date:** 2026-02-22
**Status:** Decided

### Context

`complexity` appears once in the skill docs as a phase-level metadata field (`"complexity": "medium"`) but the server doesn't validate it, the plan template doesn't ask for it, and no code reads it. Meanwhile, complexity is the one piece of sizing information that's actually useful — unlike `estimated_hours` (which we're removing), complexity is a relative judgment the LLM can make reasonably well.

### Decision

Add `metadata.complexity` as a validated field on task nodes, with a corresponding section in the plan template. Not at phase or spec level — phases inherit complexity from their hardest task, they don't have independent complexity.

**Valid values:** `low`, `medium`, `high`

### What it enables

- **Task ordering.** The implementation skill can sequence high-complexity tasks earlier when context is fresh, or group low-complexity tasks for batch execution.
- **Review prioritization.** The fidelity gate can apply more scrutiny to high-complexity tasks.
- **Human visibility.** The plan reviewer sees which tasks the LLM considers hard — a useful signal for whether the plan is realistic.
- **Replaces estimated_hours.** Complexity is the useful core of what time estimates were trying to express, without the false precision.

### Plan Template Addition

```markdown
- **{Task title}** [{task_category}] [{complexity}]
  - Description: {What to do}
  - File: {repo-relative path, or "N/A" for investigation/research/decision}
  - Acceptance criteria:
    - {How to verify this task is done}
  - Depends on: {other task titles, or "none"}
```

### Action Items

- [ ] Add `metadata.complexity` to task node validation in `core/spec/rules.py` (valid values: `low`, `medium`, `high`)
- [ ] Add validation warning if complexity is missing on tasks
- [ ] Update plan template to include `[complexity]` tag per task
- [ ] Update spec review to check complexity alignment between plan and spec
