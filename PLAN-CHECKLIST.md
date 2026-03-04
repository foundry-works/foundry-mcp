# Chunked Citation-Anchored Claim Extraction — Checklist

## Phase 1: Section Chunking Helper

- [ ] Add `_split_report_into_sections()` function after `_build_extraction_user_prompt()` (line ~203)
- [ ] Split on `\n## ` boundaries, preserve heading in each chunk
- [ ] Merge small sections (< 500 chars) with next section
- [ ] Exclude bibliography/sources section from extraction chunks
- [ ] Fall back to single chunk when no `##` headings found

## Phase 2: Citation-Anchored Extraction Prompt

- [ ] Replace `_EXTRACTION_SYSTEM_PROMPT` with citation-anchored version
- [ ] Anchor extraction to inline `[N]` citations ("For each inline citation [N]...")
- [ ] Add "ONLY extract claims that have an explicit [N] citation" hard rule
- [ ] Add "If no cited claims found, return empty array: []" guidance
- [ ] Add "If a single sentence cites multiple sources, extract ONE claim" dedup rule

## Phase 3: Per-Chunk Extraction Function

- [ ] Add `_extract_claims_from_chunk()` async function
- [ ] Build focused extraction prompt from chunk content only
- [ ] Set `max_tokens=4096` (reduced from 16384)
- [ ] Set `max_retries=1` (reduced from default 2)
- [ ] Parse response with existing `_parse_extracted_claims()`
- [ ] Tag extracted claims with chunk's section heading as `report_section`
- [ ] Graceful degradation: log warning and return empty list on failure

## Phase 4: Parallel Chunk Orchestrator

- [ ] Add `_extract_claims_chunked()` async function
- [ ] Use `asyncio.Semaphore(max_concurrent)` for bounded concurrency
- [ ] Gather all chunk tasks with `asyncio.gather(*tasks, return_exceptions=True)`
- [ ] Use `check_gather_cancellation()` for cancellation safety
- [ ] Merge claim lists from all chunks
- [ ] Deduplicate claims by exact claim text match
- [ ] Cap total claims at `max_claims`
- [ ] Fall back to single-call extraction when only 1 chunk exists
- [ ] Log per-chunk extraction results (success/fail, claim count)

## Phase 5: Update Orchestrator

- [ ] Replace monolithic extraction call in `extract_and_verify_claims()` (lines 791-813)
- [ ] Call `_extract_claims_chunked()` instead of direct execute_fn
- [ ] Preserve `extraction_failed` metadata when all chunks fail
- [ ] Keep existing 30K truncation guard (still useful to cap total input)

## Phase 6: Timeout and Retry Tuning

- [ ] Increase default `deep_research_claim_verification_timeout` from 180.0 to 300.0 in `config/research.py`
- [ ] Update TOML defaults dict to match new timeout value
- [ ] Per-chunk calls use `max_retries=1` (2 total attempts per chunk)
- [ ] Per-chunk calls use `max_tokens=4096`

## Phase 7: Post-Extraction Citation Filter

- [ ] Add `_filter_uncited_claims()` function
- [ ] Drop claims with empty or missing `cited_sources`
- [ ] Log count of dropped uncited claims
- [ ] Call from `_extract_claims_chunked()` after merge+dedup, before returning

## Phase 8: Multi-Window Source Truncation

- [ ] Replace `_keyword_proximity_truncate()` with `_multi_window_truncate()`
- [ ] Find all keyword positions (not just first match)
- [ ] Score candidate windows by keyword density
- [ ] Select top N non-overlapping windows (default max_windows=3)
- [ ] Concatenate windows in document order with `[...]` separators
- [ ] Allocate max_chars budget across windows
- [ ] Preserve prefix-truncate fallback when no keywords match
- [ ] Update `_build_verification_user_prompt()` to call `_multi_window_truncate()`

## Phase 9: Tighten CONTRADICTED Definition in Verification Prompt

- [ ] Replace `_VERIFICATION_SYSTEM_PROMPT` with version containing explicit verdict definitions
- [ ] Add "explicitly state something that DIRECTLY CONFLICTS" for CONTRADICTED
- [ ] Add "Absence of information does NOT mean the source contradicts"
- [ ] Add "When in doubt between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED"
- [ ] Add "You are seeing excerpts, not the full source"
- [ ] Mark evidence_quote as REQUIRED for CONTRADICTED verdicts

## Phase 10: Require Contradicting Evidence Quote (Structural Gate)

- [ ] Add post-parse check in `_verify_single_claim()`: CONTRADICTED + no evidence_quote → UNSUPPORTED
- [ ] Preserve original explanation in downgraded claim
- [ ] Log downgrade events at INFO level

## Phase 11: Tests

**Extraction tests:**
- [ ] `test_split_report_into_sections`: multiple headings → correct chunks
- [ ] `test_split_report_into_sections`: small sections merged
- [ ] `test_split_report_into_sections`: no headings → single chunk fallback
- [ ] `test_split_report_into_sections`: bibliography section excluded
- [ ] `test_extraction_prompt_is_citation_anchored`: prompt contains citation-anchor language
- [ ] `test_extraction_prompt_is_citation_anchored`: per-chunk prompt uses chunk content
- [ ] `test_extract_claims_from_chunk`: successful extraction returns claims
- [ ] `test_extract_claims_from_chunk`: failed extraction returns empty list
- [ ] `test_extract_claims_from_chunk`: claims tagged with correct report_section
- [ ] `test_extract_claims_from_chunk`: max_tokens=4096 and max_retries=1 passed
- [ ] `test_filter_uncited_claims`: claims with cited_sources kept
- [ ] `test_filter_uncited_claims`: claims with empty/missing cited_sources dropped
- [ ] `test_filter_uncited_claims`: all-uncited input returns empty list
- [ ] `test_filter_uncited_claims`: logs dropped count
- [ ] `test_extract_claims_chunked_parallel`: chunks extracted in parallel
- [ ] `test_extract_claims_chunked_parallel`: claims merged and deduplicated
- [ ] `test_extract_claims_chunked_parallel`: uncited claims filtered out
- [ ] `test_extract_claims_chunked_parallel`: total claims capped at max_claims
- [ ] `test_extract_claims_chunked_parallel`: partial failure preserves successful chunks
- [ ] `test_extract_claims_chunked_parallel`: all chunks fail → empty list
- [ ] `test_extract_claims_chunked_parallel`: single chunk → fallback behavior
- [ ] `test_extract_and_verify_claims_uses_chunked`: end-to-end integration
- [ ] `test_cancellation_safety`: check_gather_cancellation works in chunked extraction
- [ ] Update existing extraction tests to match new prompt and chunked behavior

**Verification improvement tests:**
- [ ] `test_multi_window_truncate`: multiple keyword matches → multiple windows
- [ ] `test_multi_window_truncate`: windows are non-overlapping
- [ ] `test_multi_window_truncate`: windows ordered by document position
- [ ] `test_multi_window_truncate`: total output within max_chars budget
- [ ] `test_multi_window_truncate`: no keyword matches → prefix-truncate fallback
- [ ] `test_multi_window_truncate`: source shorter than max_chars → returned unchanged
- [ ] `test_multi_window_truncate`: single keyword match → single window
- [ ] `test_verification_prompt_contradicted_definition`: prompt contains tightened definition
- [ ] `test_contradicted_without_quote_downgraded`: empty evidence_quote → UNSUPPORTED
- [ ] `test_contradicted_without_quote_downgraded`: null evidence_quote → UNSUPPORTED
- [ ] `test_contradicted_without_quote_downgraded`: valid evidence_quote → stays CONTRADICTED
- [ ] `test_contradicted_without_quote_downgraded`: SUPPORTED + empty quote → unchanged
- [ ] `test_contradicted_without_quote_downgraded`: original explanation preserved
- [ ] Verify existing verification and correction tests still pass

## Final Validation

- [ ] Run full test suite (`pytest tests/`)
- [ ] Run contract tests (`pytest tests/contract/`)
- [ ] Smoke test: run general-profile deep research session with large report
- [ ] Smoke test: confirm claim_verification object is populated with extracted claims
- [ ] Smoke test: verify wall time for claim verification phase < 120s (down from 550s)
- [ ] Smoke test: verify extracted claims all have non-empty cited_sources
- [ ] Smoke test: verify no CONTRADICTED verdicts with empty evidence_quote
