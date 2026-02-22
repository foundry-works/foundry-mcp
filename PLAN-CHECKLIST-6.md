# Plan 6 Checklist: Spec Review Enhancement — Diff Against the Plan

## Plan content loading (reuse Plan 4 infrastructure)

- [x] In `_handle_spec_review()` in `review.py`: call `_build_plan_context(spec_data, workspace_root)` after loading spec
- [x] Import `_build_plan_context` from `documentation_helpers` (already imported for fidelity — verify)
- [x] Pass `plan_content` string to `_run_ai_review()` (None/empty if unavailable)
- [x] In `_run_ai_review()` in `review_helpers.py`: accept optional `plan_content` parameter
- [x] Add `plan_content` to `ConsultationRequest` context dict

## Template selection (auto-enhance, not new review type)

- [x] Add `"spec-vs-plan": "SPEC_REVIEW_VS_PLAN_V1"` to `_REVIEW_TYPE_TO_TEMPLATE` in `review_helpers.py`
- [x] In `_run_ai_review()`: when `review_type == "full"` and `plan_content` is truthy, select `"spec-vs-plan"` template key
- [x] When `plan_content` is falsy, fall back to existing `PLAN_REVIEW_FULL_V1` (no behavior change)
- [x] `security` and `feasibility` review types are unaffected regardless of plan_path

## Spec-vs-plan prompt template

- [x] Create new file `src/foundry_mcp/core/prompts/spec_review.py`
- [x] Define `SPEC_VS_PLAN_RESPONSE_SCHEMA` (JSON format, following `FIDELITY_RESPONSE_SCHEMA` pattern)
- [x] Define system prompt for spec-vs-plan reviewer role
- [x] Create `SPEC_REVIEW_VS_PLAN_V1` PromptTemplate with 7 comparison dimensions:
  - [x] Coverage (phases, tasks, verification steps)
  - [x] Fidelity (semantic alignment of intent)
  - [x] Success criteria mapping
  - [x] Constraints preserved
  - [x] Risks preserved
  - [x] Open questions preserved
  - [x] Undocumented additions (flag, don't reject)
- [x] Prompt emphasizes semantic alignment, not syntactic matching
- [x] Create `SPEC_REVIEW_TEMPLATES` registry dict
- [x] Create `SpecReviewPromptBuilder` class (follows `PlanReviewPromptBuilder` pattern)
- [x] Register template in prompt discovery — `PlanReviewPromptBuilder.build()` delegates to `SpecReviewPromptBuilder` for `SPEC_REVIEW_*` prompt IDs

## Review result persistence (new for spec reviews)

- [x] After plan-comparison review completes, write result to `specs/.spec-reviews/{spec_id}-spec-review.md`
- [x] Include timestamp, review type, verdict, and full comparison output in persisted file
- [x] Follow persistence pattern from `_handle_fidelity()` in `review.py`
- [x] Return `review_path` in response `data` envelope
- [x] Standalone reviews (no plan_path) do NOT get persisted — no change to existing behavior

## Documentation

- [x] Document new spec review behavior in `docs/05-mcp-tool-reference.md`
- [x] Note that `full` review auto-enhances when spec has `plan_path`
- [x] Document the 7 comparison dimensions
- [x] Document backward compatibility: old specs without plan_path get standalone review
- [x] Document response schema differences (JSON for plan-comparison, markdown for standalone)

## Verification

- [x] Spec review with plan_path → uses `SPEC_REVIEW_VS_PLAN_V1`, returns JSON comparison results
- [x] Spec review without plan_path → falls back to `PLAN_REVIEW_FULL_V1` standalone review (backward compat)
- [x] Spec review when plan file missing/unreadable → warns, falls back to standalone (graceful via `_build_plan_context` returning empty string)
- [x] Response includes verdict, coverage counts, fidelity status, metadata alignment (structured in `SPEC_VS_PLAN_RESPONSE_SCHEMA`)
- [x] Undocumented additions are flagged but don't auto-fail the review (prompt instructs to flag, not reject)
- [x] Review results written to disk at `specs/.spec-reviews/{spec_id}-spec-review.md`
- [x] Review file path returned in response data
- [x] Caching: same spec+plan pair with `consultation_cache=True` returns cached result (reuses `PLAN_REVIEW` workflow; cache key includes `prompt_id`)
- [x] `security` and `feasibility` review types unchanged when plan_path present
- [x] Run full test suite — 3436 passed, 48 skipped, 0 regressions
