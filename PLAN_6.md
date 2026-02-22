# Plan 6: Spec Review Enhancement — Diff Against the Plan

**Decision:** "Spec Review Should Diff Against the Plan"
**Dependencies:** Plan 4 (plan linkage — complete), Plan 5 (plan template alignment — complete)
**Risk:** Medium — changes how spec review works; the review becomes comparative rather than standalone

---

## Rationale

Currently, spec review evaluates a spec in isolation. With `metadata.plan_path` required (Plan 4) and the plan template aligned to the spec schema (Plan 5), the spec review can now compare the two. The plan represents what was agreed upon with the human. The spec is the JSON translation. Any gap is a translation error or undocumented deviation.

This closes the traceability loop: Plan → Plan Review → Spec → Spec Review (against plan) → Implementation → Fidelity Gate (against spec).

---

## Key Design Decisions

### Review type routing: auto-enhance, not new type

The existing `review_type` values are `["quick", "full", "security", "feasibility"]`. Plan comparison is **not** a new review type — it's an automatic enhancement of the `"full"` review when `plan_content` is available. This means:

- `review(action="spec", review_type="full")` on a spec with `plan_path` → uses `SPEC_REVIEW_VS_PLAN_V1`
- `review(action="spec", review_type="full")` on a spec without `plan_path` → uses `PLAN_REVIEW_FULL_V1` (unchanged)
- `review(action="spec", review_type="security|feasibility")` → unchanged regardless of plan_path

No new review_type string needed. No API surface change.

### Response format: JSON schema (like fidelity reviews)

Current spec reviews use **markdown-format** responses (via `_RESPONSE_SCHEMA` in `plan_review.py`). Fidelity reviews use **structured JSON** schemas. The plan-comparison review produces structured comparison data (coverage counts, alignment statuses) that maps naturally to JSON. We adopt the JSON pattern here.

When `plan_content` is absent and we fall back to `PLAN_REVIEW_FULL_V1`, the existing markdown response format is preserved — no change to the fallback path.

### ConsultationWorkflow: reuse `PLAN_REVIEW`

The `ConsultationRequest` currently uses `ConsultationWorkflow.PLAN_REVIEW` for spec reviews. We reuse this same workflow enum — the `prompt_id` field (`SPEC_REVIEW_VS_PLAN_V1` vs `PLAN_REVIEW_FULL_V1`) already differentiates the template. No new workflow enum needed. Caching keys include `prompt_id`, so cache isolation is automatic.

### Naming convention note

Existing naming is already inconsistent: `PLAN_REVIEW_FULL_V1` reviews **specs** (not plans), while `MARKDOWN_PLAN_REVIEW_FULL_V1` reviews **markdown plans**. The new `SPEC_REVIEW_VS_PLAN_V1` is more descriptive. Renaming existing prompts is out of scope for this plan.

---

## Scope

### 1. Reuse `_build_plan_context()` from Plan 4

Plan 4 already created `_build_plan_context(spec_data, workspace_root)` in `documentation_helpers.py` (lines 407-441). This function:
- Reads `metadata.plan_path` from spec data
- Resolves the path relative to workspace root
- Returns formatted plan content (or empty string if file missing/unreadable)

It's already wired into the **fidelity** review flow (review.py lines 824, 868). We reuse this same function for spec review — no new file-loading logic needed.

**File: `src/foundry_mcp/tools/unified/review.py`**
- `_handle_spec_review()` (lines 97-235): After loading the spec, call `_build_plan_context(spec_data, workspace_root)` to get plan content. Pass it through to `_run_ai_review()`.

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- `_run_ai_review()` (lines 115-320): Accept optional `plan_content` parameter. Add to `ConsultationRequest` context:
  ```python
  context = {
      "spec_content": json.dumps(spec_data, indent=2),
      "spec_id": spec_id,
      "title": context.title,
      "review_type": review_type,
      "plan_content": plan_content,  # NEW — None if no plan_path
  }
  ```
- When `review_type == "full"` and `plan_content` is truthy, override template selection to `SPEC_REVIEW_VS_PLAN_V1` instead of `PLAN_REVIEW_FULL_V1`.

### 2. Create spec-vs-plan review prompt

**File: `src/foundry_mcp/core/prompts/spec_review.py`** (new file)

Create `SPEC_REVIEW_VS_PLAN_V1` prompt template that:
1. Receives both `spec_content` (JSON) and `plan_content` (markdown)
2. Evaluates seven comparison dimensions:
   - **Coverage** — Every phase, task, and verification step in the plan has a corresponding spec node
   - **Fidelity** — Spec tasks match the plan's intent (semantic alignment, not string matching)
   - **Success criteria mapping** — Plan's success criteria reflected in `metadata.success_criteria` and/or task `acceptance_criteria`
   - **Constraints preserved** — Plan's constraints appear in `metadata.constraints`
   - **Risks preserved** — Plan's risk table reflected in `metadata.risks`
   - **Open questions preserved** — Plan's open questions in `metadata.open_questions`
   - **Undocumented additions** — Anything in spec not traceable to plan (flagged, not necessarily wrong)
3. Returns structured JSON response (see section 3 below)

The prompt should emphasize **semantic alignment** — markdown prose and JSON will never match syntactically. The LLM compares meaning, not strings.

Include a `SpecReviewPromptBuilder` class following the pattern in `plan_review.py` (`PlanReviewPromptBuilder`) and `fidelity_review.py` (`FidelityReviewPromptBuilder`).

### 3. Response schema for plan-comparison review

Structured JSON response (follows fidelity review pattern):

```json
{
  "verdict": "aligned|deviation|incomplete",
  "summary": "Overall alignment assessment",
  "coverage": {
    "plan_phases_covered": 3,
    "plan_phases_total": 3,
    "plan_tasks_covered": 12,
    "plan_tasks_total": 14,
    "missing_items": ["task title from plan that has no spec node"]
  },
  "fidelity": {
    "status": "aligned|diluted|diverged",
    "issues": [{"plan_item": "...", "spec_item": "...", "concern": "..."}]
  },
  "metadata_alignment": {
    "success_criteria": "aligned|missing|partial",
    "constraints": "aligned|missing|partial",
    "risks": "aligned|missing|partial",
    "open_questions": "aligned|missing|partial"
  },
  "undocumented_additions": [
    {"spec_item": "...", "justification_needed": true}
  ],
  "issues": ["..."],
  "recommendations": ["..."]
}
```

Define this as `SPEC_VS_PLAN_RESPONSE_SCHEMA` in the new `spec_review.py` prompt file, following the pattern of `FIDELITY_RESPONSE_SCHEMA` in `fidelity_review.py`.

### 4. Update template mapping

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- Add `"spec-vs-plan"` → `"SPEC_REVIEW_VS_PLAN_V1"` to `_REVIEW_TYPE_TO_TEMPLATE` mapping (used internally, not exposed as a user-facing review_type).
- Update `_run_ai_review()` to select the `"spec-vs-plan"` key when `review_type == "full"` and `plan_content` is truthy.

**File: `src/foundry_mcp/core/prompts/spec_review.py`**
- Register `SPEC_REVIEW_VS_PLAN_V1` in a `SPEC_REVIEW_TEMPLATES` dict.
- Export via the prompt builder class for discovery.

### 5. Store spec review results (new for spec reviews)

Currently, only fidelity reviews persist results to disk (`.fidelity-reviews/` directory). This plan adds persistence for spec reviews — a new behavior.

**File: `src/foundry_mcp/tools/unified/review.py`**
- After `_run_ai_review()` returns for a plan-comparison review, write the result to `specs/.spec-reviews/{spec_id}-spec-review.md`.
- Include: timestamp, review type, verdict, full comparison output.
- Return the review file path in the response `data.review_path`.

Follow the persistence pattern from fidelity review (`_handle_fidelity()` lines ~1020-1060) for consistency.

**Standalone reviews (no plan_path) do NOT get persisted** — this matches current behavior and avoids scope creep.

### 6. What the review should NOT do

Per the decision:
- Do NOT re-evaluate whether the plan itself was good (that's the plan review's job)
- Do NOT check implementation code (that's the fidelity gate's job)
- Do NOT block on minor wording differences — markdown prose vs JSON. Semantic alignment is the goal.

---

## Design Notes

- **Backward compatibility.** Old specs without `plan_path` get the existing standalone review. No degradation. The plan-comparison review is additive — it only activates when `plan_path` is present and the file exists.
- **Semantic vs syntactic comparison.** The LLM compares meaning, not strings. A plan task "implement OAuth2 with PKCE" should match a spec task with similar description, even if the wording differs. The prompt must emphasize semantic alignment.
- **"Undocumented additions" are not errors.** The spec may contain things not in the plan (e.g., LLM added error handling tasks). These should be flagged for review, not auto-rejected. The concern is silent omissions, not reasonable additions.
- **Review caching.** `ConsultationWorkflow.PLAN_REVIEW` is reused; cache keys include `prompt_id`, so `SPEC_REVIEW_VS_PLAN_V1` and `PLAN_REVIEW_FULL_V1` cache independently. If `consultation_cache` is True (default), the same spec+plan pair won't trigger a duplicate LLM call.
- **Persistence scope.** Only plan-comparison reviews (with plan_path) are persisted to disk. Standalone reviews retain current behavior (no disk write). This keeps the change minimal.
