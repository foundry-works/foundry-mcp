# Plan Checklist: Post-Synthesis Quality Improvements

## Phase 1: Remove "Section #:" prefix from synthesis output
- [ ] Add no-numbering instruction to Section Writing Rules in `_build_synthesis_system_prompt` (synthesis.py ~line 1215)
- [ ] Add `_strip_section_numbering(report: str) -> str` helper function with regex `r'^(#{1,6})\s*(?:Section|Part)\s+\d+\s*:\s*'` → `r'\1 '`
- [ ] Call `_strip_section_numbering()` on report text in `execute()` after LLM response is captured
- [ ] Add unit test: `_strip_section_numbering` correctly strips `## Section 1: Foo` → `## Foo`
- [ ] Add unit test: `_strip_section_numbering` correctly strips `## Part 2: Bar` → `## Bar`
- [ ] Add unit test: `_strip_section_numbering` leaves `## Executive Summary` untouched
- [ ] Add unit test: `_strip_section_numbering` leaves `## 2025 Market Overview` untouched (year not stripped)
- [ ] Add prompt assertion test: system prompt contains no-numbering instruction

## Phase 2: Renumber citations in reading order
- [x] Add `renumber_citations(report: str, state: DeepResearchState, *, max_citation: int | None) -> tuple[str, dict[int, int]]` to `_citation_postprocess.py`
  - [x] Scan report left-to-right with `_CITATION_RE.finditer()`, build `{old: new}` map in first-appearance order
  - [x] Skip numbers above `max_citation` (year references)
  - [x] Replace all `[old]` → `[new]` in report text
  - [x] Update `source.citation_number` on all `state.sources` using the map
  - [x] Update `state.next_citation_number` to `max(new_values) + 1`
  - [x] Return `(renumbered_report, renumber_map)`
- [x] Integrate into `postprocess_citations()` between step 3 (dangling removal) and step 4 (bibliography append)
- [x] Add renumber map to the returned metadata dict (e.g., `"renumbered_count": len(map)`)
- [x] Add test: out-of-order citations `[5] foo [2] bar [5]` → `[1] foo [2] bar [1]`
- [x] Add test: gaps eliminated `[1], [3], [7]` → `[1], [2], [3]`
- [x] Add test: state sources have updated `citation_number` values after renumbering
- [x] Add test: `state.next_citation_number` updated correctly
- [x] Add test: year references `[2025]` preserved (not renumbered)
- [x] Add test: markdown links `[text](url)` not affected
- [x] Add test: already-ordered report is a no-op (idempotent)
- [x] Add test: bibliography section uses renumbered citations in order

## Phase 3: Fix `claims_extracted` reporting inconsistency
- [ ] Add `claims_filtered: int = 0` field to `ClaimVerificationResult` in `models/deep_research.py`
- [ ] Set `result.claims_filtered = len(to_verify)` after `_apply_token_budget` in claim_verification.py
- [ ] Verify `result.claims_extracted` is set from raw extraction count (pre-filter) — confirm existing code is correct
- [ ] Include `claims_filtered` in response builder output (`builders.py`)
- [ ] Update audit event `claim_verification_complete` to include `claims_filtered`
- [ ] Add/update test asserting `claims_extracted >= claims_filtered >= claims_verified`

## Phase 4: Remove dead `report_sections` and `content_fidelity` fields
- [ ] Grep for all references to `report_sections` across codebase — confirm no write callers
- [ ] Grep for all references to `content_fidelity`, `ContentFidelityRecord`, `FidelityLevel` — confirm no workflow callers
- [ ] Remove `report_sections` field from `DeepResearchState`
- [ ] Remove `content_fidelity` field from `DeepResearchState`
- [ ] Remove `ContentFidelityRecord` class (if no other consumers)
- [ ] Remove `FidelityLevel` enum (if no other consumers)
- [ ] Remove helper methods: `record_fidelity_level`, `get_fidelity_record`, `get_items_at_fidelity_level`, `overall_fidelity_score`, `has_degraded_content`, and fidelity merge methods
- [ ] Update `state_migrations.py` — add migration to silently drop `report_sections` and `content_fidelity` keys from persisted state
- [ ] Update `builders.py` — remove serialization of removed fields
- [ ] Update `types.py` — remove type references if any
- [ ] Remove/update any tests referencing removed fields
- [ ] Verify old session state files deserialize without error (backward compat)
