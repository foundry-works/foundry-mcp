# Spec Ingredients Research

## 1. The Template (what the server generates)

When you call `authoring(action='spec-create')`, the only template is `"empty"` — it produces this skeleton:

```json
{
  "spec_id": "<feature>-<YYYY-MM-DD>-<nnn>",
  "title": "<name>",
  "generated": "<ISO>",
  "last_updated": "<ISO>",
  "metadata": {
    "description": "",
    "mission": "<mission>",
    "objectives": [],
    "assumptions": []
  },
  "progress_percentage": 0,
  "status": "pending",
  "current_phase": null,
  "hierarchy": { "spec-root": { ... } },
  "journal": []
}
```

Structure is then added via **5 phase templates**: `planning`, `implementation`, `testing`, `security`, `documentation`. Each provides pre-configured tasks with titles, descriptions, acceptance criteria, estimated hours, and **auto-appended verification scaffolding** (run-tests + fidelity review).

| Template | Est. Hours | Tasks | Purpose |
|----------|-----------|-------|---------|
| planning | 4 | Define requirements, Design solution | Scope & requirements definition |
| implementation | 8 | Implement core, Add error handling | Primary functionality build |
| testing | 6 | Write unit tests, Write integration tests | Quality assurance |
| security | 6 | Security audit, Security remediation | Vulnerability identification/fix |
| documentation | 4 | Write API docs, Write user guide | Implementation documentation |

---

## 2. What the Server Enforces (validation in `core/spec/rules.py`)

### Hard errors (block progress)

- `spec_id`, `generated`, `last_updated`, `hierarchy` must exist
- `metadata.mission` must be non-empty
- `spec-root` must exist with `parent: null`
- Tree integrity: no orphans, no cycles, bidirectional parent-child consistency
- Every node needs: `type`, `title`, `status`, `parent`, `children`, `total_tasks`, `completed_tasks`, `metadata`
- Valid node types: `spec`, `phase`, `group`, `task`, `subtask`, `verify`, `research`
- Valid statuses: `pending`, `in_progress`, `completed`, `blocked`, `archived`
- Valid task categories: `investigation`, `implementation`, `refactoring`, `decision`, `research`
- **All tasks must have**: `task_category`, `description` (or details array), `acceptance_criteria` (non-empty array)
- **Implementation/refactoring tasks must have**: `file_path`
- **Verify nodes must have**: `verification_type` (`run-tests`, `fidelity`, `manual`)
- Task counts must propagate correctly up the tree
- Dependency targets must exist and be bidirectionally consistent
- Input size capped at 10MB

### Warnings

- Empty titles
- Non-ISO timestamps
- Missing estimates

### Auto-fixable

- Timestamp normalization
- Orphan repair
- Count mismatches
- Parent-child alignment
- Missing verify types

---

## 3. What the Skill Says to Include (claude-foundry `foundry-spec`)

The skill enforces a **9-step workflow** with mandatory gates:

```
Analyze → Markdown Plan → AI Review → HUMAN APPROVAL → JSON Spec → AI Review → Parse Mods → Apply → Validate
```

### The Markdown Plan (mandatory, before any JSON)

Must include:

- **Mission** — single sentence (becomes `metadata.mission`)
- **Objective** — one paragraph
- **Success Criteria** — measurable checkboxes
- **Constraints** — technical and business
- **Phases** — each with goal, files affected, tasks, verification
- **Risks** — table with likelihood/impact/mitigation
- **Dependencies** — external and internal
- **Open Questions** — unresolved items

### Task Requirements per the Skill

| Field | Required? |
|-------|-----------|
| `title` | Always |
| `description` | Always |
| `acceptance_criteria` | Always (min 1) |
| `task_category` | Always |
| `file_path` | Only for `implementation` / `refactoring` (must be real path, no placeholders) |
| `estimated_hours` | Always |
| `instructions` | Always (actionable steps) |

### Verification Coverage

Every phase must end with:

1. `run-tests` verify node
2. `fidelity` verify node (compares implementation to spec)

### Size Guidelines

- 1-6 phases recommended
- 3-50 tasks recommended
- 20-50% verification coverage

### AI Review Dimensions

Specs are reviewed across 6 dimensions:

1. **Completeness** — All necessary info present
2. **Clarity** — Unambiguous and actionable
3. **Feasibility** — Realistic estimates and achievable dependencies
4. **Architecture** — Sound design decisions
5. **Risk Management** — Edge cases and mitigations
6. **Verification** — Test plan and acceptance criteria testable

---

## 4. The Gap (Skill vs Server)

| Skill says... | Server enforces... | Status |
|---|---|---|
| Mandatory markdown plan first | No enforcement — plan is optional from server's perspective | Skill-only gate (by design) |
| Human approval gate before JSON | Not enforced server-side | Skill-only gate (by design) |
| `instructions` array on tasks | Not validated, not read | **Dropping** (see DECISIONS.md) |
| Risk table, open questions, constraints | Not stored in spec JSON at all | **Adding schema fields** (see DECISIONS.md) |
| Size guidelines (1-6 phases, 3-50 tasks) | No limits enforced | Skill-only guidance (by design) |
| `mission` is critical | Validated as required | **Aligned** — no gap |
| `file_path` for impl/refactor | Validated as required | **Aligned** — no gap |

The skill is the "should" layer (LLM guidance), the server is the "must" layer (hard validation). Some of the skill's richest requirements (the markdown plan, risks, constraints, open questions) live outside the spec JSON entirely — they're in `specs/.plans/` as markdown artifacts. Decisions on closing the real gaps are tracked in `DECISIONS.md`.
