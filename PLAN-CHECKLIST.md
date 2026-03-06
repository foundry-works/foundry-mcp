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

- [ ] **2.1** Add `deep_research_fidelity_min_improvement: float = 0.10` to `ResearchConfig` (config/research.py)
- [ ] **2.2** Add `from_dict` parsing for `deep_research_fidelity_min_improvement`
- [ ] **2.3** Add validation in `_validate_research_settings` (0.0 < value < 1.0)
- [ ] **2.4** Add `fidelity_min_improvement` parameter to `decide_iteration` signature (orchestration.py)
- [ ] **2.5** Add convergence stall check: if `len(fidelity_scores) >= 2` and `delta < min_improvement`, complete
- [ ] **2.6** Pass `fidelity_min_improvement` from config in workflow_execution.py call site
- [ ] **2.7** Test: `test_fidelity_convergence_stall_completes`
- [ ] **2.8** Test: `test_fidelity_convergence_sufficient_improvement_iterates`
- [ ] **2.9** Test: `test_fidelity_convergence_only_one_score_iterates`
- [ ] **2.10** Test: `test_fidelity_convergence_decision_records_scores`
- [ ] **2.11** Run existing fidelity iteration tests to verify no regressions

## Phase 3: Extract `_finalize_report` helper

- [ ] **3.1** Rename concept from `_finalize_and_save_citations` to `_finalize_report` (will include confidence section in Phase 4)
- [ ] **3.2** Extract `_finalize_report(self, state, *, trigger)` method on workflow executor
  - [ ] Idempotency guard via `_report_finalized` metadata
  - [ ] Non-fatal error handling with audit logging
  - [ ] Markdown file re-save
- [ ] **3.3** Replace happy-path finalize block (lines 673-709) with helper call
- [ ] **3.4** Replace cancellation-rollback finalize block (Phase 1.2) with helper call
- [ ] **3.5** Replace cancellation-completed finalize block (Phase 1.3) with helper call
- [ ] **3.6** Verify all existing tests still pass after extraction

## Phase 4: LLM-interpreted research confidence section

### 4a: Context assembly (deterministic)

- [ ] **4.1** Create `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py`
- [ ] **4.2** Implement `build_confidence_context(state)` — assembles structured dict from:
  - [ ] Verdict distribution (supported / partial / unsupported / contradicted counts)
  - [ ] Per-section breakdown (which report sections have most unsupported claims)
  - [ ] Claim-type breakdown (comparative/inferential vs. quantitative/factual unsupported claims)
  - [ ] Failed sub-queries with query text
  - [ ] Fidelity score trajectory across iterations
  - [ ] Corrections applied count and summaries
  - [ ] Source count and iteration count
- [ ] **4.3** Handle edge case: `state.claim_verification is None` (verification didn't run or was wiped by iteration reset)

### 4b: LLM interpretation call

- [ ] **4.4** Implement `generate_confidence_section(state, llm_call_fn, *, query_type)` async function
- [ ] **4.5** Write system prompt — interpret verification data in context of the query type:
  - [ ] Distinguish inferential claims (expected for this query) from evidence-gap claims
  - [ ] Name specific failed sub-queries as gaps
  - [ ] Mention corrections applied
  - [ ] Note iteration history
  - [ ] No raw scores, no "fidelity" terminology, no hedging
  - [ ] 150-300 word target
- [ ] **4.6** Write user prompt — JSON serialization of `build_confidence_context` output + `original_query` + `query_type`
- [ ] **4.7** Model selection: use haiku-class model (compression-tier) via `resolve_phase_provider`
- [ ] **4.8** Implement deterministic fallback: if LLM call fails/times out, produce bullet-point summary from raw data
- [ ] **4.9** Return markdown string starting with `## Research Confidence\n\n`

### 4c: Integration into `_finalize_report`

- [ ] **4.10** Add confidence section generation as step 2 in `_finalize_report` (after citation finalize, before markdown save)
- [ ] **4.11** Section appended after `## Sources` — final section of report
- [ ] **4.12** Handle async context: confidence call is async, verify it works in both happy-path and cancellation-handler contexts
- [ ] **4.13** Non-fatal: confidence section failure does not block report delivery
- [ ] **4.14** Audit event: `confidence_section_complete` or `confidence_section_failed`
- [ ] **4.15** Skip confidence section entirely if `state.claim_verification is None` (no data to interpret)

### 4d: Tests

- [ ] **4.16** Test: `test_build_confidence_context_basic` — known verdict distribution produces expected context structure
- [ ] **4.17** Test: `test_build_confidence_context_no_verification` — None claim_verification handled gracefully
- [ ] **4.18** Test: `test_build_confidence_context_failed_subqueries` — failed sub-queries appear in context output
- [ ] **4.19** Test: `test_generate_confidence_section_success` — mock LLM returns valid markdown
- [ ] **4.20** Test: `test_generate_confidence_section_llm_failure_falls_back` — mock LLM raises, deterministic fallback used
- [ ] **4.21** Test: `test_generate_confidence_section_skipped_when_no_verification` — no verification data → section omitted
- [ ] **4.22** Test: `test_confidence_section_integration` — end-to-end with realistic mock state

## Phase 5: Source-deepening verification strategy

### 5a: Classify UNSUPPORTED claims

- [ ] **5.1** Create `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py`
- [ ] **5.2** Implement `classify_unsupported_claims(verification_result, citation_map)` returning:
  - [ ] `inferential`: comparative/recommendation claims (no action needed)
  - [ ] `deepen_window`: factual claims where source has rich raw_content (>16K) but was truncated to 8K
  - [ ] `deepen_extract`: factual claims where source is thin (snippet-only, <4K raw_content)
  - [ ] `widen`: factual claims that genuinely need new sources
- [ ] **5.3** Classification keys on `claim_type` (quantitative/negative → factual, comparative/positive → potentially inferential) plus `raw_content` length of cited source

### 5b: Expanded-window re-verification

- [ ] **5.4** Implement `reverify_with_expanded_window(claims, citation_map, llm_call_fn, *, max_chars=24000)`
- [ ] **5.5** Add `VERIFICATION_SOURCE_DEEPEN_MAX_CHARS = 24000` to `_constants.py`
- [ ] **5.6** Re-uses existing `_VERIFICATION_SYSTEM_PROMPT` and `_parse_verification_response` — only the content window changes
- [ ] **5.7** Updates `ClaimVerdict.verdict` in place when upgraded (UNSUPPORTED → SUPPORTED/PARTIALLY_SUPPORTED)
- [ ] **5.8** Audit event: `claim_reverification_expanded_window` with counts of upgraded verdicts

### 5c: Re-extract thin sources

- [ ] **5.9** Implement `deepen_thin_sources(claims, citation_map, state, extract_provider)`
- [ ] **5.10** DOI resolution path: if source has `metadata.doi`, attempt Semantic Scholar paper details fetch
- [ ] **5.11** URL re-extraction path: if source has URL, re-extract via TavilyExtractProvider
- [ ] **5.12** Update `source.raw_content` with richer content (preserve original in `source.metadata["_pre_deepen_content"]`)
- [ ] **5.13** After deepening, run standard verification on newly deepened claims
- [ ] **5.14** Audit event: `source_deepening_complete` with counts of sources deepened

### 5d: Integration into claim verification pipeline

- [ ] **5.15** Wire expanded-window re-verification into `workflow_execution.py` between claim verification and fidelity scoring
- [ ] **5.16** Wire source re-extraction before expanded-window step (so deepened content is available)
- [ ] **5.17** Modify `build_gap_queries` to exclude inferential claims from gap queries
- [ ] **5.18** Modify `build_gap_queries` to exclude claims already resolved by deepening
- [ ] **5.19** Fidelity score now reflects post-deepening verdicts (no code change needed — score is computed from ClaimVerificationResult which was mutated in place)

### 5e: Tests

- [ ] **5.20** Test: `test_classify_inferential_claims` — comparative claims classified correctly
- [ ] **5.21** Test: `test_classify_deepen_window` — quantitative + rich source → deepen_window
- [ ] **5.22** Test: `test_classify_deepen_extract` — quantitative + thin source → deepen_extract
- [ ] **5.23** Test: `test_classify_widen` — factual + no existing source → widen
- [ ] **5.24** Test: `test_reverify_expanded_window_upgrades_verdict`
- [ ] **5.25** Test: `test_reverify_expanded_window_unchanged`
- [ ] **5.26** Test: `test_deepen_thin_sources_with_doi`
- [ ] **5.27** Test: `test_deepen_thin_sources_with_url`
- [ ] **5.28** Test: `test_build_gap_queries_excludes_inferential`
- [ ] **5.29** Test: `test_fidelity_improves_after_deepening` — end-to-end

## Final Validation

- [ ] **6.1** Run full deep research test suite: `pytest tests/core/research/workflows/deep_research/ -x`
- [ ] **6.2** Run citation postprocess tests: `pytest tests/core/research/workflows/deep_research/test_citation_postprocess.py -x`
- [ ] **6.3** Run config tests to verify new field: `pytest tests/ -k "research_config" -x`
- [ ] **6.4** Verify no import cycles introduced
- [ ] **6.5** Run new confidence section tests: `pytest tests/core/research/workflows/deep_research/test_confidence_section.py -x`
- [ ] **6.6** Run new source deepening tests: `pytest tests/core/research/workflows/deep_research/test_source_deepening.py -x`
