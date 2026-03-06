# Plan Checklist: Deep Research Post-Synthesis Resilience

## Phase 1: Finalize citations on cancellation/timeout

- [x] **1.1** Add `_citations_finalized` metadata flag to happy-path `finalize_citations` block (workflow_execution.py ~line 684)
- [x] **1.2** Add `finalize_citations` call in cancellation handler after rollback to `last_completed_iteration` (workflow_execution.py ~line 829)
  - [x] Guard with `_citations_finalized` check
  - [x] Re-save markdown file if `report_output_path` exists
  - [x] Audit event with `trigger: "cancellation_rollback"`
  - [x] Non-fatal try/except matching happy-path pattern
- [x] **1.3** Add `finalize_citations` call in cancellation handler for completed-iteration-at-cancel branch (workflow_execution.py ~line 843-849)
  - [x] Same guard, save, audit, and error handling pattern
  - [x] Audit event with `trigger: "cancellation_completed"`
- [x] **1.4** Test: `test_cancellation_rollback_finalizes_citations`
- [x] **1.5** Test: `test_cancellation_after_completed_iteration_finalizes_citations`
- [x] **1.6** Test: `test_cancellation_first_iteration_incomplete_skips_finalize`
- [x] **1.7** Test: `test_citation_finalize_failure_during_cancellation_is_nonfatal`
- [x] **1.8** Run existing cancellation tests to verify no regressions

## Phase 2: Fidelity convergence early-stop

- [x] **2.1** Add `deep_research_fidelity_min_improvement: float = 0.10` to `ResearchConfig` (config/research.py)
- [x] **2.2** Add `from_dict` parsing for `deep_research_fidelity_min_improvement`
- [x] **2.3** Add validation in `_validate_research_settings` (0.0 < value < 1.0)
- [x] **2.4** Add `fidelity_min_improvement` parameter to `decide_iteration` signature (orchestration.py)
- [x] **2.5** Add convergence stall check: if `len(fidelity_scores) >= 2` and `delta < min_improvement`, complete
- [x] **2.6** Pass `fidelity_min_improvement` from config in workflow_execution.py call site
- [x] **2.7** Test: `test_fidelity_convergence_stall_completes`
- [x] **2.8** Test: `test_fidelity_convergence_sufficient_improvement_iterates`
- [x] **2.9** Test: `test_fidelity_convergence_only_one_score_iterates`
- [x] **2.10** Test: `test_fidelity_convergence_decision_records_scores`
- [x] **2.11** Run existing fidelity iteration tests to verify no regressions

## Phase 3: Extract `_finalize_report` helper

- [x] **3.1** Rename concept from `_finalize_and_save_citations` to `_finalize_report` (will include confidence section in Phase 4)
- [x] **3.2** Extract `_finalize_report(self, state, *, trigger)` method on workflow executor
  - [x] Idempotency guard via `_report_finalized` metadata
  - [x] Non-fatal error handling with audit logging
  - [x] Markdown file re-save
- [x] **3.3** Replace happy-path finalize block (lines 673-709) with helper call
- [x] **3.4** Replace cancellation-rollback finalize block (Phase 1.2) with helper call
- [x] **3.5** Replace cancellation-completed finalize block (Phase 1.3) with helper call
- [x] **3.6** Verify all existing tests still pass after extraction

## Phase 4: LLM-interpreted research confidence section

### 4a: Context assembly (deterministic)

- [x] **4.1** Create `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py`
- [x] **4.2** Implement `build_confidence_context(state)` — assembles structured dict from:
  - [x] Verdict distribution (supported / partial / unsupported / contradicted counts)
  - [x] Per-section breakdown (which report sections have most unsupported claims)
  - [x] Claim-type breakdown (comparative/inferential vs. quantitative/factual unsupported claims)
  - [x] Failed sub-queries with query text
  - [x] Fidelity score trajectory across iterations
  - [x] Corrections applied count and summaries
  - [x] Source count and iteration count
- [x] **4.3** Handle edge case: `state.claim_verification is None` (verification didn't run or was wiped by iteration reset)

### 4b: LLM interpretation call

- [x] **4.4** Implement `generate_confidence_section(state, llm_call_fn, *, query_type)` async function
- [x] **4.5** Write system prompt — interpret verification data in context of the query type:
  - [x] Distinguish inferential claims (expected for this query) from evidence-gap claims
  - [x] Name specific failed sub-queries as gaps
  - [x] Mention corrections applied
  - [x] Note iteration history
  - [x] No raw scores, no "fidelity" terminology, no hedging
  - [x] 150-300 word target
- [x] **4.6** Write user prompt — JSON serialization of `build_confidence_context` output + `original_query` + `query_type`
- [x] **4.7** Model selection: use haiku-class model (compression-tier) via `resolve_phase_provider`
- [x] **4.8** Implement deterministic fallback: if LLM call fails/times out, produce bullet-point summary from raw data
- [x] **4.9** Return markdown string starting with `## Research Confidence\n\n`

### 4c: Integration into `_finalize_report`

- [x] **4.10** Add confidence section generation as step 2 in `_finalize_report` (after citation finalize, before markdown save)
- [x] **4.11** Section appended after `## Sources` — final section of report
- [x] **4.12** Handle async context: confidence call is async, verify it works in both happy-path and cancellation-handler contexts
- [x] **4.13** Non-fatal: confidence section failure does not block report delivery
- [x] **4.14** Audit event: `confidence_section_complete` or `confidence_section_failed`
- [x] **4.15** Skip confidence section entirely if `state.claim_verification is None` (no data to interpret)

### 4d: Tests

- [x] **4.16** Test: `test_build_confidence_context_basic` — known verdict distribution produces expected context structure
- [x] **4.17** Test: `test_build_confidence_context_no_verification` — None claim_verification handled gracefully
- [x] **4.18** Test: `test_build_confidence_context_failed_subqueries` — failed sub-queries appear in context output
- [x] **4.19** Test: `test_generate_confidence_section_success` — mock LLM returns valid markdown
- [x] **4.20** Test: `test_generate_confidence_section_llm_failure_falls_back` — mock LLM raises, deterministic fallback used
- [x] **4.21** Test: `test_generate_confidence_section_skipped_when_no_verification` — no verification data → section omitted
- [x] **4.22** Test: `test_confidence_section_integration` — end-to-end with realistic mock state

## Phase 5: Source-deepening verification strategy

### 5a: Classify UNSUPPORTED claims

- [x] **5.1** Create `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py`
- [x] **5.2** Implement `classify_unsupported_claims(verification_result, citation_map)` returning:
  - [x] `inferential`: comparative/recommendation claims (no action needed)
  - [x] `deepen_window`: factual claims where source has rich raw_content (>16K) but was truncated to 8K
  - [x] `deepen_extract`: factual claims where source is thin (snippet-only, <4K raw_content)
  - [x] `widen`: factual claims that genuinely need new sources
- [x] **5.3** Classification keys on `claim_type` (quantitative/negative → factual, comparative/positive → potentially inferential) plus `raw_content` length of cited source

### 5b: Expanded-window re-verification

- [x] **5.4** Implement `reverify_with_expanded_window(claims, citation_map, llm_call_fn, *, max_chars=24000)`
- [x] **5.5** Add `VERIFICATION_SOURCE_DEEPEN_MAX_CHARS = 24000` to `_constants.py`
- [x] **5.6** Re-uses existing `_VERIFICATION_SYSTEM_PROMPT` and `_parse_verification_response` — only the content window changes
- [x] **5.7** Updates `ClaimVerdict.verdict` in place when upgraded (UNSUPPORTED → SUPPORTED/PARTIALLY_SUPPORTED)
- [x] **5.8** Audit event: `source_deepening_complete` with counts of upgraded verdicts

### 5c: Re-extract thin sources

- [x] **5.9** Implement `deepen_thin_sources(claims, citation_map, state, extract_provider)`
- [N/A] **5.10** DOI resolution path: if source has `metadata.doi`, attempt Semantic Scholar paper details fetch (out of scope — requires partnership tier API access)
- [x] **5.11** URL re-extraction path: if source has URL, re-extract via TavilyExtractProvider
- [x] **5.12** Update `source.raw_content` with richer content (preserve original in `source.metadata["_pre_deepen_content"]`)
- [N/A] **5.13** After deepening, run standard verification on newly deepened claims (deferred — deepened sources feed into next iteration's verification)
- [x] **5.14** Audit event: `source_deepening_complete` with counts of sources deepened

### 5d: Integration into claim verification pipeline

- [x] **5.15** Wire expanded-window re-verification into `workflow_execution.py` between claim verification and fidelity scoring
- [N/A] **5.16** Wire source re-extraction before expanded-window step (deepen_extract is implemented but not wired inline — needs extract_provider plumbing; deepen_window runs standalone)
- [x] **5.17** Modify `build_gap_queries` to exclude inferential claims from gap queries
- [x] **5.18** Modify `build_gap_queries` to exclude claims already resolved by deepening
- [x] **5.19** Fidelity score now reflects post-deepening verdicts (aggregate counts recomputed after verdict upgrades)

### 5e: Tests

- [x] **5.20** Test: `test_classify_inferential_claims` — comparative claims classified correctly
- [x] **5.21** Test: `test_classify_deepen_window` — quantitative + rich source → deepen_window
- [x] **5.22** Test: `test_classify_deepen_extract` — quantitative + thin source → deepen_extract
- [x] **5.23** Test: `test_classify_widen` — factual + no existing source → widen
- [x] **5.24** Test: `test_reverify_expanded_window_upgrades_verdict`
- [x] **5.25** Test: `test_reverify_expanded_window_unchanged`
- [N/A] **5.26** Test: `test_deepen_thin_sources_with_doi` (DOI resolution deferred)
- [x] **5.27** Test: `test_deepen_thin_sources_with_url`
- [x] **5.28** Test: `test_build_gap_queries_excludes_inferential`
- [x] **5.29** Test: `test_fidelity_improves_after_deepening` — end-to-end

## Final Validation

- [x] **6.1** Run full deep research test suite: `pytest tests/core/research/workflows/deep_research/ -x` (1172 passed)
- [x] **6.2** Run citation postprocess tests: included in 6.1
- [x] **6.3** Run config tests: no new config fields in Phase 5
- [x] **6.4** Verify no import cycles introduced
- [x] **6.5** Run new confidence section tests: included in 6.1
- [x] **6.6** Run new source deepening tests: `pytest tests/core/research/workflows/deep_research/test_source_deepening.py -x` (19 passed)
