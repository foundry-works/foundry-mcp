# Chunked Citation-Anchored Claim Extraction — Checklist

## Phase 1: Section Chunking Helper

- [x] Add `_split_report_into_sections()` function after `_build_extraction_user_prompt()` (line ~203)
- [x] Split on `(?:^|\n)#{2,3} ` boundaries (handles start-of-string and mid-string headings)
- [x] Preserve heading text in each chunk's `"section"` key
- [x] Merge small sections (< 500 chars) with next section
- [x] Exclude bibliography/sources sections via anchored heading regex `(?i)^(bibliography|references|sources|works cited)$`
- [x] Fall back to single chunk when no `##`/`###` headings found
- [x] Discard final chunk if it lacks a heading (truncation boundary fragment), unless it's the only chunk

## Phase 2: Citation-Anchored Extraction Prompt

- [x] Replace `_EXTRACTION_SYSTEM_PROMPT` with citation-anchored version
- [x] Anchor extraction to inline `[N]` citations ("For each inline citation [N]...")
- [x] Add "ONLY extract claims that have an explicit [N] citation" hard rule
- [x] Add "If no cited claims found, return empty array: []" guidance
- [x] Add "If a single sentence cites multiple sources, extract ONE claim" dedup rule

## Phase 3: Per-Chunk Extraction Function

- [x] Add `_extract_claims_from_chunk()` async function
- [x] Build per-chunk user prompt: `"## Section\n\n{chunk_content}\n\n## Task\n\nExtract cited factual claims..."` (distinct from old full-report prompt)
- [x] Set `max_tokens=4096` (reduced from 16384)
- [x] Set `max_retries=1` (reduced from default 2)
- [x] Set `retry_delay=2.0` (reduced from default 5.0)
- [x] Parse response with existing `_parse_extracted_claims()`, passing `max_claims_per_chunk` as `max_claims`
- [x] Compute `max_claims_per_chunk` as `max(10, max_claims // len(chunks))` in caller
- [x] Tag extracted claims with chunk's section heading as `report_section`
- [x] Graceful degradation: log warning and return empty list on failure

## Phase 4: Parallel Chunk Orchestrator

- [x] Add `_extract_claims_chunked()` async function
- [x] Use `asyncio.Semaphore(max_concurrent)` for bounded concurrency
- [x] Gather all chunk tasks with `asyncio.gather(*tasks, return_exceptions=True)`
- [x] Use `check_gather_cancellation()` for cancellation safety
- [x] Merge claim lists from all chunks
- [x] Deduplicate claims by **normalized** claim text (lowercase, strip whitespace, remove `\[\d+\]` citation brackets only)
- [x] Cap total claims at `max_claims`
- [x] Single chunk processed via `_extract_claims_from_chunk()` — same code path, no special case
- [x] Call `_filter_uncited_claims()` after merge+dedup, before returning
- [x] Log per-chunk extraction results (chunk index, success/fail, claim count)
- [x] Accept optional `metadata` dict; populate `extraction_strategy`, `extraction_chunks_attempted`, `extraction_chunks_succeeded`, `extraction_claims_per_chunk`

## Phase 5: Update Orchestrator

- [ ] Replace monolithic extraction call in `extract_and_verify_claims()` (lines 791-813)
- [ ] Call `_extract_claims_chunked()` instead of direct execute_fn, passing `metadata=state.metadata`
- [ ] Caller (`extract_and_verify_claims()`) owns `extraction_failed` metadata — set in except block and empty-claims check
- [ ] Remove `_build_extraction_user_prompt()` function (dead code after this change)
- [ ] Keep existing 30K truncation guard (still useful to cap total input)
- [ ] No config changes needed — per-call timeout stays at 180s from config

## Phase 6: Post-Extraction Citation Filter

- [x] Add `_filter_uncited_claims()` function
- [x] Drop claims with empty or missing `cited_sources`
- [x] Log count of dropped uncited claims
- [x] Call from `_extract_claims_chunked()` after merge+dedup, before returning

## Phase 7: Multi-Window Source Truncation

- [ ] Replace `_keyword_proximity_truncate()` with `_multi_window_truncate()`
- [ ] Find all keyword positions (case-insensitive, not just first match)
- [ ] Cluster keyword positions by proximity (gap > `cluster_radius` starts new cluster, `cluster_radius = max_chars // max_windows`)
- [ ] Score clusters by distinct keyword count (not total occurrences)
- [ ] Select top N clusters by score (default max_windows=3)
- [ ] **Adaptive window sizing:** `window_size = max_chars // len(selected_clusters)` (full budget when 1 cluster, split when multiple)
- [ ] Extract `window_size` char window centered on each cluster's median position
- [ ] Ensure non-overlapping windows (shift or skip if overlap)
- [ ] Concatenate windows in document order with `\n[...]\n` separators
- [ ] Preserve prefix-truncate fallback when no keywords match
- [ ] Update `_build_verification_user_prompt()` to call `_multi_window_truncate()`

## Phase 8: Tighten CONTRADICTED Definition in Verification Prompt

- [ ] Replace `_VERIFICATION_SYSTEM_PROMPT` with version containing explicit verdict definitions
- [ ] Add "explicitly state something that DIRECTLY CONFLICTS" for CONTRADICTED
- [ ] Add "Absence of information does NOT mean the source contradicts"
- [ ] Add "When in doubt between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED"
- [ ] Add "You are seeing excerpts, not the full source"
- [ ] Mark evidence_quote as REQUIRED for CONTRADICTED verdicts

## Phase 9: Require Contradicting Evidence Quote (Structural Gate)

- [ ] Add post-parse check in `_verify_single_claim()`: CONTRADICTED + no evidence_quote → UNSUPPORTED
- [ ] Preserve original explanation in downgraded claim
- [ ] Log downgrade events at INFO level

## Phase 10: Tests

**Extraction tests:**
- [ ] `test_split_report_into_sections`: multiple headings (## and ###) → correct chunks
- [ ] `test_split_report_into_sections`: report starting with `##` (no preceding newline) → first heading captured
- [ ] `test_split_report_into_sections`: small sections merged
- [ ] `test_split_report_into_sections`: no headings → single chunk fallback
- [ ] `test_split_report_into_sections`: bibliography section excluded
- [ ] `test_split_report_into_sections`: heading "Data Sources and Methodology" NOT excluded (anchored regex)
- [ ] `test_split_report_into_sections`: truncated report (last chunk lacks heading) → fragment discarded
- [ ] `test_extraction_prompt_is_citation_anchored`: prompt contains citation-anchor language
- [ ] `test_extraction_prompt_is_citation_anchored`: per-chunk prompt uses chunk content (not full report)
- [ ] `test_extract_claims_from_chunk`: successful extraction returns claims
- [ ] `test_extract_claims_from_chunk`: failed extraction returns empty list
- [ ] `test_extract_claims_from_chunk`: claims tagged with correct report_section
- [ ] `test_extract_claims_from_chunk`: max_tokens=4096, max_retries=1, retry_delay=2.0 passed
- [ ] `test_filter_uncited_claims`: claims with cited_sources kept
- [ ] `test_filter_uncited_claims`: claims with empty/missing cited_sources dropped
- [ ] `test_filter_uncited_claims`: all-uncited input returns empty list
- [ ] `test_filter_uncited_claims`: logs dropped count
- [ ] `test_extract_claims_chunked_parallel`: chunks extracted in parallel
- [ ] `test_extract_claims_chunked_parallel`: claims merged and deduplicated (normalized text)
- [ ] `test_extract_claims_chunked_parallel`: uncited claims filtered out
- [ ] `test_extract_claims_chunked_parallel`: total claims capped at max_claims
- [ ] `test_extract_claims_chunked_parallel`: partial failure preserves successful chunks
- [ ] `test_extract_claims_chunked_parallel`: all chunks fail → empty list
- [ ] `test_extract_claims_chunked_parallel`: single chunk uses same code path
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
- [ ] `test_multi_window_truncate`: single keyword match → single window with full max_chars budget (adaptive)
- [ ] `test_multi_window_truncate`: two clusters → each gets max_chars // 2 (adaptive sizing)
- [ ] `test_verification_prompt_contradicted_definition`: prompt contains tightened definition
- [ ] `test_contradicted_without_quote_downgraded`: empty evidence_quote → UNSUPPORTED
- [ ] `test_contradicted_without_quote_downgraded`: null evidence_quote → UNSUPPORTED
- [ ] `test_contradicted_without_quote_downgraded`: valid evidence_quote → stays CONTRADICTED
- [ ] `test_contradicted_without_quote_downgraded`: SUPPORTED + empty quote → unchanged
- [ ] `test_contradicted_without_quote_downgraded`: original explanation preserved
- [ ] Verify existing verification and correction tests still pass

**Negative-behavior tests:**
- [ ] `test_monolithic_extraction_removed`: `_build_extraction_user_prompt` removed from module
- [ ] `test_monolithic_extraction_removed`: no direct execute_fn call with max_tokens=16384 in orchestrator
- [ ] `test_cancelled_error_during_gather`: CancelledError propagates correctly via check_gather_cancellation
- [ ] `test_many_small_sections`: 50+ tiny sections merged to reasonable chunk count (< 20), no excessive LLM calls

## Final Validation

- [ ] Run full test suite (`pytest tests/`)
- [ ] Run contract tests (`pytest tests/contract/`)
- [ ] Smoke test: run general-profile deep research session with large report
- [ ] Smoke test: confirm claim_verification object is populated with extracted claims
- [ ] Smoke test: verify wall time for claim verification phase < 120s (down from 550s)
- [ ] Smoke test: verify extracted claims all have non-empty cited_sources
- [ ] Smoke test: verify no CONTRADICTED verdicts with empty evidence_quote
- [ ] Smoke test: verify `extraction_strategy: "chunked"` in metadata
- [ ] Smoke test: verify `extraction_chunks_succeeded > 0` in metadata
