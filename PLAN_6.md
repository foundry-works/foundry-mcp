# Plan 6: Spec Review Enhancement — Diff Against the Plan

**Decision:** "Spec Review Should Diff Against the Plan"
**Dependencies:** Plan 4 (plan linkage), Plan 5 (plan template alignment)
**Risk:** Medium — changes how spec review works; the review becomes comparative rather than standalone

---

## Rationale

Currently, spec review evaluates a spec in isolation. With `metadata.plan_path` required (Plan 4) and the plan template aligned to the spec schema (Plan 5), the spec review can now compare the two. The plan represents what was agreed upon with the human. The spec is the JSON translation. Any gap is a translation error or undocumented deviation.

This closes the traceability loop: Plan → Plan Review → Spec → Spec Review (against plan) → Implementation → Fidelity Gate (against spec).

---

## Scope

### 1. Load plan content during spec review

**File: `src/foundry_mcp/tools/unified/review.py`**
- `_handle_spec_review()` (lines 96-234): After loading the spec, check for `metadata.plan_path`. If present and file exists, read the plan content. If absent, fall back to standalone review (backward compat).

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- `_run_ai_review()` (lines 115-320): Update the context passed to `ConsultationRequest` to include `plan_content` when available:
  ```python
  context = {
      "spec_content": spec_content,
      "spec_id": spec_id,
      "title": title,
      "review_type": review_type,
      "plan_content": plan_content,  # NEW — None if no plan_path
  }
  ```

### 2. Update spec review prompt for plan comparison

**File: `src/foundry_mcp/core/prompts/plan_review.py`** (or a new `spec_review.py` prompt file)

The current spec review uses `PLAN_REVIEW_FULL_V1` template (mapped at lines 43-47 of review_helpers.py). This needs a new prompt variant or an updated prompt that handles both cases.

**Option A (recommended):** Create a new prompt template `SPEC_REVIEW_VS_PLAN_V1` that:
1. Receives both `spec_content` (JSON) and `plan_content` (markdown)
2. Evaluates the seven comparison dimensions from the decision:
   - **Coverage** — Every phase, task, and verification step in the plan has a corresponding spec node
   - **Fidelity** — Spec tasks match the plan's intent (semantic alignment, not string matching)
   - **Success criteria mapping** — Plan's success criteria reflected in `metadata.success_criteria` and/or task `acceptance_criteria`
   - **Constraints preserved** — Plan's constraints appear in `metadata.constraints`
   - **Risks preserved** — Plan's risk table reflected in `metadata.risks`
   - **Open questions preserved** — Plan's open questions in `metadata.open_questions`
   - **Undocumented additions** — Anything in spec not traceable to plan (flagged, not necessarily wrong)

**Option B:** Modify `PLAN_REVIEW_FULL_V1` to conditionally include plan comparison when `plan_content` is provided. This is messier — a dedicated prompt is cleaner.

**Prompt routing:**
- If `plan_content` is available → use `SPEC_REVIEW_VS_PLAN_V1`
- If no plan → fall back to existing `PLAN_REVIEW_FULL_V1` (standalone review)

### 3. Update response schema for plan-comparison review

The response should capture comparison-specific findings:

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

### 4. Store spec review results

**File: `src/foundry_mcp/tools/unified/review.py`**
- After spec review completes, write the review to `specs/.plan-reviews/{spec_id}-spec-review.md` (or similar)
- Store the path in a predictable location that the skill workflow can reference

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- Add review result persistence — write to disk with timestamp and review type

### 5. Update spec review invocation to pass plan

**File: `src/foundry_mcp/tools/unified/review_helpers.py`**
- `prepare_review_context()` (called around line 141-148): When reviewing a spec, resolve and load `plan_path` from spec metadata. Handle gracefully if file doesn't exist (warn, fall back to standalone review).

### 6. What the review should NOT do

Per the decision:
- Do NOT re-evaluate whether the plan itself was good (that's the plan review's job)
- Do NOT check implementation code (that's the fidelity gate's job)
- Do NOT block on minor wording differences — markdown prose vs JSON. Semantic alignment is the goal.

---

## Design Notes

- **Backward compatibility.** Old specs without `plan_path` get the existing standalone review. No degradation. The plan-comparison review is additive — it only activates when plan_path is present.
- **Semantic vs syntactic comparison.** The LLM compares meaning, not strings. A plan task "implement OAuth2 with PKCE" should match a spec task with similar description, even if the wording differs. The prompt must emphasize semantic alignment.
- **"Undocumented additions" are not errors.** The spec may contain things not in the plan (e.g., LLM added error handling tasks). These should be flagged for review, not auto-rejected. The concern is silent omissions, not reasonable additions.
- **Review caching.** If `consultation_cache` is True (default), the same spec+plan pair shouldn't trigger a new LLM call if already reviewed. The existing caching in the consultation orchestrator handles this.
