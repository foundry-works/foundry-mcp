# Plan: Post-Synthesis Quality Improvements

**Branch:** alpha
**Scope:** 4 targeted fixes in the deep research synthesis and claim verification pipeline

## Context

Analysis of deep research session `deepres-f38ce17db061` revealed four actionable improvements to the post-synthesis pipeline introduced in 0.18.0a5–a7.

---

## Phase 1: Remove "Section #:" prefix from synthesis output

**Problem:** The synthesis LLM spontaneously generates numbered section headings like `## Section 1: Transfer Partner Landscape` instead of clean headings like `## Transfer Partner Landscape`. This happens because:
- The system prompt says "Use ## for each section title" but doesn't explicitly prohibit numbering
- The structure guidance templates don't number sections, but the LLM adds numbers anyway
- No post-processing strips the pattern

**Approach:** Two-pronged fix:
1. Add an explicit instruction in the `_build_synthesis_system_prompt` Section Writing Rules to not prefix section titles with numbering (e.g., "Section 1:", "Part 2:", etc.)
2. Add a lightweight post-processing regex strip in the synthesis phase that removes `Section \d+:\s*` from `## ` headings in the final report, as a safety net

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
  - Add instruction to Section Writing Rules (~line 1215)
  - Add `_strip_section_numbering()` helper and call it on the report text after LLM response (in `execute()` after report is captured)

**Tests:**
- Add test for `_strip_section_numbering` helper: verify patterns like `## Section 1: Foo` → `## Foo`, `## Part 2: Bar` → `## Bar`, and that non-matching headings are untouched
- Verify the prompt assertion: the system prompt must contain the no-numbering instruction

---

## Phase 2: Renumber citations in reading order

**Problem:** Citation numbers in the report reflect source **discovery order** during gathering, not the order they appear in the synthesized report. A reader encounters `[32][33]` in the opening paragraph, then `[5]` later — numbers feel arbitrary. The bibliography also has gaps (e.g., `[1], [2], [5], [6], [8]...`) because uncited sources are filtered out but remaining numbers aren't renumbered.

**Root cause:** Sources get `citation_number` assigned via a running counter in `DeepResearchState.add_source()` / `append_source()` at discovery time. The synthesis LLM uses these numbers in its output, and `postprocess_citations()` preserves them as-is. The bibliography is built from state using `cited_only=True`, which filters uncited sources but keeps their original numbers.

**Approach:** Add a renumbering pass inside `postprocess_citations()` between step 3 (dangling removal) and step 4 (bibliography append):

1. Scan the report left-to-right for all `[N]` citations in order of first appearance
2. Build a renumber map `{old_number: new_number}` where new numbers are 1, 2, 3... in first-appearance order
3. Replace all inline `[N]` citations in the report using the map
4. Update `source.citation_number` on all state sources to match the new mapping (so `state.get_citation_map()` returns renumbered entries)
5. Update `state.next_citation_number` to `max(new_numbers) + 1`
6. The existing `build_sources_section()` then produces a contiguous, reading-order bibliography

**Key constraint:** This must run *before* claim verification, since claim verification calls `state.get_citation_map()` and extracts `cited_sources` numbers from the report text. The current flow already has `postprocess_citations()` before claim verification in `workflow_execution.py`, so the ordering is correct.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`
  - Add `renumber_citations(report, state) -> tuple[str, dict[int, int]]` function
  - Integrate into `postprocess_citations()` between steps 3 and 4
  - Return renumber map in metadata for audit logging
- `src/foundry_mcp/core/research/models/deep_research.py`
  - No model changes needed — we mutate `source.citation_number` and `next_citation_number` in place

**Tests:**
- Test `renumber_citations` with a report containing out-of-order citations: `[5] foo [2] bar [5] baz` → `[1] foo [2] bar [1] baz`
- Test that state sources get their `citation_number` updated to match
- Test that gaps are eliminated: `[1], [3], [7]` → `[1], [2], [3]`
- Test that year references like `[2025]` are not renumbered (preserved by `max_citation` guard)
- Test that markdown links `[text](url)` are not affected
- Test idempotency: renumbering an already-ordered report is a no-op

---

## Phase 3: Fix `claims_extracted` reporting inconsistency

**Problem:** `claims_extracted` is set to 50 (post-filter count) but `extraction_claims_per_chunk` sums to 71 (raw extraction count). The field name implies it tracks extraction output, not filtered input to verification.

**Approach:** Track both values:
- Keep `claims_extracted` as the raw extraction count (sum of per-chunk claims)
- Add `claims_filtered` to `ClaimVerificationResult` for the count after `_filter_claims_for_verification` + `_apply_token_budget`
- Update `claims_verified` semantics to remain as-is (claims that actually went through Pass 2)

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py` — Add `claims_filtered: int = 0` to `ClaimVerificationResult`
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`
  - Move `result.claims_extracted = len(all_claims)` to before filtering (already correct location)
  - Add `result.claims_filtered = len(to_verify)` after `_apply_token_budget`
- `src/foundry_mcp/core/responses/builders.py` — Include `claims_filtered` in response if present

**Tests:**
- Update existing claim verification tests to assert `claims_extracted >= claims_filtered >= claims_verified`

---

## Phase 4: Remove dead `report_sections` and `content_fidelity` fields

**Problem:** `report_sections: dict[str, str]` and `content_fidelity: dict[str, ContentFidelityRecord]` are defined on `DeepResearchState` but never populated by any workflow phase. They add dead weight to the model, serialization, and state persistence.

**Approach:** Determine whether these are planned features or abandoned ones:
- `report_sections` — No write references exist anywhere in the codebase. Remove the field.
- `content_fidelity` — Has helper methods (`record_fidelity_level`, `get_fidelity_record`, etc.) but zero callers in the workflow. The synthesis phase stores fidelity info in `content_allocation_metadata` instead. Remove the field and its helper methods, plus the `ContentFidelityRecord` and `FidelityLevel` classes if they have no other consumers.

**Files:**
- `src/foundry_mcp/core/research/models/deep_research.py`
  - Remove `report_sections` field (~line 1492)
  - Remove `content_fidelity` field (~line 1531) and associated helper methods (lines ~2211–2391)
  - Remove `ContentFidelityRecord` and `FidelityLevel` if unused elsewhere
- `src/foundry_mcp/core/research/state_migrations.py` — Check for migration code referencing these fields
- `src/foundry_mcp/core/responses/builders.py` — Remove any serialization of these fields
- `src/foundry_mcp/core/responses/types.py` — Remove type references if any

**Risk mitigation:**
- Grep for all references before removing to ensure no consumer exists
- Add a state migration that silently drops these keys from persisted state files (backward compat for existing sessions)

**Tests:**
- Remove or update any tests that reference these fields
- Verify state deserialization of old sessions (with the fields) still works after removal

---

## Execution Order

Phase 2 (citation renumbering) must run before Phase 3 (claims_extracted fix) since Phase 3's tests may need to account for renumbered citations. Phase 1 is independent. Phase 4 is independent cleanup.

Recommended order: Phase 2 → Phase 1 → Phase 3 → Phase 4.
