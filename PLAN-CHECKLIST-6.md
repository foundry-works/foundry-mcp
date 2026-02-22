# Plan 6 Checklist: Spec Review Enhancement — Diff Against the Plan

## Plan content loading

- [ ] In `_handle_spec_review()` in `review.py`: after loading spec, read `metadata.plan_path` if present
- [ ] Resolve plan_path relative to specs directory
- [ ] Read plan file content; handle gracefully if file not found (warn, fall back to standalone)
- [ ] Pass `plan_content` to review context (None if unavailable)

## Spec review prompt

- [ ] Create `SPEC_REVIEW_VS_PLAN_V1` prompt template (new file or in existing `plan_review.py`)
- [ ] Define 7 comparison dimensions: coverage, fidelity, success criteria mapping, constraints preserved, risks preserved, open questions preserved, undocumented additions
- [ ] Define response schema for plan-comparison review (verdict, coverage counts, fidelity status, metadata alignment, undocumented additions)
- [ ] Add prompt routing: use `SPEC_REVIEW_VS_PLAN_V1` when plan_content available, fall back to `PLAN_REVIEW_FULL_V1` otherwise

## Review context updates

- [ ] Update `prepare_review_context()` in `review_helpers.py` to include plan_content when reviewing specs
- [ ] Update `ConsultationRequest` context dict to accept `plan_content` field
- [ ] Update `_REVIEW_TYPE_TO_TEMPLATE` mapping to support the new prompt template

## Review result persistence

- [ ] Write spec review results to `specs/.plan-reviews/{spec_id}-spec-review.md` (or similar)
- [ ] Include timestamp, review type, and verdict in persisted review
- [ ] Return review file path in response data

## Documentation

- [ ] Document new spec review behavior in `docs/05-mcp-tool-reference.md`
- [ ] Note that spec review now compares against linked plan when available
- [ ] Document the 7 comparison dimensions
- [ ] Document backward compatibility: old specs get standalone review

## Verification

- [ ] Spec review with plan_path → uses plan-comparison prompt, returns comparison results
- [ ] Spec review without plan_path → falls back to standalone review (backward compat)
- [ ] Spec review when plan file missing → warns, falls back to standalone
- [ ] Review response includes coverage counts, fidelity status, metadata alignment
- [ ] Undocumented additions are flagged but don't auto-fail the review
- [ ] Review results written to disk at expected path
- [ ] Run full test suite
