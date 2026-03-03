# Implementation Checklist: Deep Research Claim Verification Pipeline

## Layer 2: Synthesis Prompt Grounding Enhancement

- [x] Add `## Factual Grounding` section to `_build_synthesis_system_prompt()` in `phases/synthesis.py`
  - [x] Insert into the **base prompt** between `## Writing Quality` and `## Citations` (NOT in conditional blocks)
  - [x] Negative claim guardrail (never claim absence without explicit source evidence)
  - [x] Comparative claim guardrail (verify both sides against sources)
  - [x] Conflict acknowledgment instruction
  - [x] Quantitative claim traceability instruction
- [ ] Manual regression test: re-run credit card research query, verify Aeroplan claim is correct

## Layer 1: Structured Data Preservation in Compression

- [x] Add `_detect_structured_blocks(text: str) -> list[str]` to `phases/compression.py`
  - [x] Detect markdown tables: consecutive `|...|...|` lines (2+ pipe-delimited rows, including `---` header separators)
  - [x] Detect definition-style bullet lists: `- **Term**: value` or `- Term — value` patterns where value contains a number/ratio/price/date (regex: `r'^[-*]\s+\*{0,2}.+?\*{0,2}\s*[-:—–]\s+.*\d.*'` — leading `-` in char class avoids range ambiguity, `\*{0,2}` matches 0-2 asterisks for optional bold)
  - [x] Return list of detected blocks as raw text strings (for validation, not extraction)
  - [x] Do NOT attempt general "proper noun" detection — focus on mechanically-detectable structures
  - [x] Accept that false positives are harmless — detection feeds into validation, which is the real correctness gate
- [x] Add structured data preservation instruction to the **per-topic compression system prompt** (inline in `_compress_single_topic_async()`, after `</Citation Rules>` and before the "Critical Reminder" paragraph)
  - [x] `<Structured Data Preservation>` section with verbatim table and list preservation rules
- [x] Add `_validate_structured_data_survival(original: str, compressed: str, blocks: list[str]) -> bool`
  - [x] Count markdown table rows (`|...|` lines) **within detected blocks only** (not the entire original) vs compressed — fail if compressed < block-sourced count. This avoids false positives from unrelated tables the model legitimately dropped.
  - [x] Extract numeric data tokens from detected blocks via regex `r'\d[\d,./:]+'` and check literal presence in compressed output
  - [x] Return False if either check fails (table row loss OR missing numeric tokens)
- [x] Wire structured block detection into `_compress_topic_findings_async()` (NOT `_compress_single_topic_async` — no return type change needed)
  - [x] Before the `compress_one()` coroutines are gathered, detect structured blocks for each topic by calling `_detect_structured_blocks()` on the concatenated `topic.message_history` content
  - [x] Store detected blocks per topic in a local `dict[str, list[str]]` keyed by `sub_query_id`
  - [x] After gather completes, pass the topic's blocks to `_compression_output_is_valid()` at the validation call site
  - [x] `_compress_single_topic_async` return type stays `tuple[int, int, bool]` — no interface change
- [x] Update **module-level** function `_compression_output_is_valid()` (top of `compression.py`, NOT a method on `CompressionMixin`)
  - [x] Current sig: `(compressed: str | None, message_history: list[dict[str, str]], topic_id: str) -> bool`
  - [x] New sig: `(compressed: str | None, message_history: list[dict[str, str]], topic_id: str, structured_blocks: list[str] | None = None) -> bool`
  - [x] Call `_validate_structured_data_survival()` when blocks are provided (additional check alongside existing length-ratio and source-reference checks)
  - [x] On validation failure, retain `message_history` (existing behavior — `topic.message_history.clear()` is skipped)
- [x] Write unit tests for `_detect_structured_blocks()`
  - [x] Test with markdown table input
  - [x] Test with bulleted list containing ratios (e.g., "1:1", "$95")
  - [x] Test with mixed content (tables + prose)
  - [x] Test with no structured data (returns empty list)
- [x] Write unit tests for `_validate_structured_data_survival()`
  - [x] Test with tables preserved (returns True)
  - [x] Test with tables paraphrased into prose (returns False)
  - [x] Test with partial survival (key tokens present but structure lost)
- [x] Mock-based integration test: detection → compression → validation → message_history retention
  - [x] Mock LLM to return compressed text that drops a table
  - [x] Verify `_detect_structured_blocks` detects the table in input
  - [x] Verify `_compression_output_is_valid` returns False (blocks not preserved)
  - [x] Verify `message_history` is NOT cleared (fallback behavior)
- [x] Wiring integration test: verify `_compress_topic_findings_async` calls `_detect_structured_blocks` and passes results to `_compression_output_is_valid`
  - [x] Patch both inner functions and assert they are called with correct arguments
  - [x] Verify blocks detected from `topic.message_history` are forwarded to validation
- [x] Integration test: structured table survives full compression pipeline
- [x] Test prompt assembly: verify `<Structured Data Preservation>` block appears after `</Citation Rules>` and before "Critical Reminder" in the assembled per-topic compression system prompt

## Layer 3: Post-Synthesis Claim Verification Phase

### Models

- [ ] Add `ClaimVerdict` model to `models/deep_research.py`
  - [ ] Fields: claim, claim_type, cited_sources, verdict, evidence_quote, explanation, correction_applied, corrected_text
- [ ] Add `ClaimVerificationResult` model to `models/deep_research.py`
  - [ ] Fields: claims_extracted, claims_verified, claims_supported, claims_contradicted, claims_unsupported, corrections_applied, details
- [ ] Add `claim_verification: Optional[ClaimVerificationResult] = None` field to `DeepResearchState`

### Configuration

- [ ] Add verification config fields to `config/research.py` (using `deep_research_` prefix)
  - [ ] `deep_research_claim_verification_enabled: bool = False` (opt-in until validated)
  - [ ] `deep_research_claim_verification_sample_rate: float = 0.3`
  - [ ] `deep_research_claim_verification_provider: Optional[str] = None`
  - [ ] `deep_research_claim_verification_model: Optional[str] = None`
  - [ ] `deep_research_claim_verification_timeout: int = 120`
  - [ ] `deep_research_claim_verification_max_claims: int = 50`
  - [ ] `deep_research_claim_verification_max_concurrent: int = 10`
  - [ ] `deep_research_claim_verification_max_corrections: int = 5`
  - [ ] `deep_research_claim_verification_annotate_unsupported: bool = False`
  - [ ] `deep_research_claim_verification_max_input_tokens: int = 200_000` (total token budget escape hatch — drops claims from tail of priority list when estimated input tokens exceed cap)
- [ ] Verify `resolve_phase_provider(config, "claim_verification", "synthesis")` works via dynamic `getattr()` lookup on variadic `*phase_names` (defined in `_model_resolution.py`; constructs `f"deep_research_{name}_provider"` for each name, falls back to `config.default_provider`)
- [ ] Unit test `resolve_phase_provider` fallback chain for claim verification:
  - [ ] Test with explicit `deep_research_claim_verification_provider` set (uses it directly)
  - [ ] Test with no `claim_verification_provider` but `deep_research_synthesis_provider` set (falls back to synthesis — second variadic arg)
  - [ ] Test with neither set (falls back to `config.default_provider`)
- [ ] Add verification budget constants to `_constants.py`
  - [ ] `VERIFICATION_MAX_CLAIMS_DEFAULT = 50`
  - [ ] `VERIFICATION_MAX_CONCURRENT_DEFAULT = 10`
  - [ ] `VERIFICATION_MAX_CORRECTIONS_DEFAULT = 5`
  - [ ] `VERIFICATION_SOURCE_MAX_CHARS = 8000` (per-source content truncation in verification prompts)
  - [ ] `VERIFICATION_OUTPUT_RESERVED = 2000`
  - [ ] `VERIFICATION_MAX_INPUT_TOKENS_DEFAULT = 200_000` (total token budget escape hatch)

### Claim Extraction (Pass 1)

- [ ] Create `phases/claim_verification.py` as a **standalone module with free async functions** (no mixin class)
- [ ] Implement `build_claim_extraction_prompt(report: str) -> tuple[str, str]`
  - [ ] System prompt: extract verifiable factual claims as structured JSON
  - [ ] Focus on factual assertions, not opinions/recommendations
  - [ ] Capture claim text, type (negative/quantitative/comparative/positive), cited source numbers, section, quote context
- [ ] Implement `parse_extracted_claims(response: str) -> list[ClaimVerdict]`
  - [ ] JSON parsing with fallback (similar to `_analysis_parsing.py` pattern)
  - [ ] Validate claim types against known set
  - [ ] Validate cited_sources against `state.get_citation_map()` keys
  - [ ] Log warning and skip sources where citation number not found in map (dangling ref)
  - [ ] **On total extraction failure** (invalid JSON, empty array, truncated): return empty list, log warning
- [ ] Implement `extract_and_verify_claims()` top-level orchestrator function
  - [ ] Calls extraction, then filtering, then batch verification
  - [ ] If extraction returns empty list → return empty `ClaimVerificationResult` with `claims_extracted=0`
  - [ ] Set `state.metadata["claim_verification_skipped"] = "extraction_failed"` on extraction failure
- [ ] Implement claim filtering/prioritization
  - [ ] All `negative` claims: always verify
  - [ ] All `quantitative` claims: always verify
  - [ ] `comparative` claims: verify when both sides have source citations
  - [ ] `positive` claims: **deterministic** sampling using `hashlib.sha256(claim.claim.encode()).hexdigest()` modulo 100 < `sample_rate * 100`
  - [ ] When filtered claims exceed `max_claims`, prioritize: negative → quantitative → comparative → positive
  - [ ] Within each type, prioritize claims with more cited sources

### Claim-Source Alignment (Pass 2)

- [ ] Materialize citation map once at start of `extract_and_verify_claims()` via `state.get_citation_map()` and pass as argument to all downstream functions (avoid redundant per-claim iteration over `state.sources`)
- [ ] Implement `_resolve_source_text(source: ResearchSource) -> Optional[str]` helper
  - [ ] Return `.content` if populated, else `.raw_content`, else `.snippet`
  - [ ] Return `None` if all three are `None` (source has no verifiable text)
- [ ] Implement `_keyword_proximity_truncate(text: str, claim_text: str, max_chars: int) -> str` helper
  - [ ] Extract keywords from claim text: split on whitespace, filter out words < 4 chars, then filter against `_STOPWORDS` — a `frozenset` of ~30 common function words (`{"this", "that", "with", "from", "have", "been", "will", "would", "could", "should", "their", "there", "which", "about", "where", "these", "those", "does", "into", "also", "more", "than", "only", "most", "each", "some", "when", "they", "were", "other"}`) defined as a module-level constant in `claim_verification.py`, no external dependency. The length filter handles most determiners/prepositions/conjunctions without maintaining a large curated list.
  - [ ] Search source text for first occurrence of any keyword
  - [ ] If found, extract window of `max_chars` centered on that position (clamped to content boundaries)
  - [ ] If no keyword match, fall back to prefix truncation (first `max_chars` characters)
- [ ] Implement `_build_verification_prompt(claim: ClaimVerdict, citation_map: dict[int, ResearchSource]) -> tuple[str, str]`
  - [ ] Look up source via `citation_map[citation_number]`, resolve text with `_resolve_source_text()`
  - [ ] Skip sources where resolved text is `None` with warning log ("source has no verifiable content")
  - [ ] Apply `_keyword_proximity_truncate()` to source text (using claim text as keyword source, `VERIFICATION_SOURCE_MAX_CHARS` as limit)
  - [ ] Include truncated source text, `.title`, `.url` for each resolved citation number
  - [ ] Skip unresolvable citation numbers with warning log
  - [ ] Include claim text and type
  - [ ] Request structured verdict: SUPPORTED / CONTRADICTED / UNSUPPORTED / PARTIALLY_SUPPORTED
  - [ ] Request evidence_quote and explanation
- [ ] Implement `_verify_single_claim(claim, citation_map, provider) -> ClaimVerdict`
  - [ ] Build prompt, call LLM, parse structured response
  - [ ] Handle parse failures gracefully (default to UNSUPPORTED)
  - [ ] Per-claim timeout handling
- [ ] Implement token budget check before dispatching verification batch
  - [ ] Estimate total input tokens: `sum(len(resolved_source_text) / 3.5 for each claim's resolved sources)` (conservative char-to-token ratio)
  - [ ] If estimate exceeds `config.deep_research_claim_verification_max_input_tokens`, drop claims from tail of priority list until under budget
  - [ ] Log number of claims dropped due to token budget
- [ ] Implement `_verify_claims_batch(claims, citation_map, provider) -> list[ClaimVerdict]`
  - [ ] Parallel execution of independent verification calls
  - [ ] Use `asyncio.Semaphore(config.deep_research_claim_verification_max_concurrent)` to limit parallel LLM calls (default: 10)
  - [ ] Aggregate results into `ClaimVerificationResult`

### Correction Application

- [ ] Implement `apply_corrections(state, config, verification_result, execute_fn)` as a free async function
  - [ ] **Corrections are applied sequentially** (not in parallel) — overlapping/adjacent context windows would race on `state.report` if corrected concurrently, producing garbled output. Iteration follows priority order: negative > quantitative > comparative > positive, then by cited source count.
  - [ ] **Correction budget**: Cap corrections at `config.deep_research_claim_verification_max_corrections` (default: 5). If CONTRADICTED claims exceed cap, prioritize: negative > quantitative > comparative > positive, then by cited source count. Uncorrected CONTRADICTED claims are logged with `correction_applied=False`.
  - [ ] **CONTRADICTED verdicts**: single-pass targeted re-synthesis using quote-context window
    - [ ] Locate `claim.quote_context` in `state.report` via substring match
    - [ ] Extract context window: ~500 chars before and after, clamped to paragraph boundaries by **expanding outward** — scan backward from `match_start - 500` to the nearest preceding `\n\n` (or start of string), scan forward from `match_end + 500` to the nearest following `\n\n` (or end of string). Window always contains complete paragraphs, never splits mid-sentence.
    - [ ] Send context window + contradicting source evidence to correction LLM, instructing rewrite of only the false claim portion
    - [ ] Replace original context window using `state.report.replace(original_window, corrected_window, 1)` — `count=1` limits to first occurrence
    - [ ] **Post-replacement sanity check**: verify corrected report still contains the paragraph boundaries (`\n\n`) that bracketed the original context window. If not, log warning and revert to pre-correction report for that claim.
    - [ ] Fallback: if `quote_context` not found, send full report with explicit single-claim correction instruction and `report_section` hint
    - [ ] Mark `correction_applied = True` and store `corrected_text` on the verdict
    - [ ] Corrections are NOT re-verified (single-pass only, no recursion)
  - [ ] **UNSUPPORTED verdicts**: logged in verification details (no inline annotation by default)
    - [ ] When `config.deep_research_claim_verification_annotate_unsupported` is True:
      - [ ] Locate `claim.quote_context` in `state.report` by substring match
      - [ ] Append ` (unverified)` immediately after the matched sentence
      - [ ] If quote context not found in report, skip annotation and log warning
  - [ ] Store `ClaimVerificationResult` on `state.claim_verification`

### Pipeline Integration

- [ ] Wire verification into `workflow_execution.py` (between orchestrator transition and `mark_completed()`)
  - [ ] Import `extract_and_verify_claims` and `apply_corrections` from `phases.claim_verification` (inline import inside the guard)
  - [ ] Guard with `config.deep_research_claim_verification_enabled`
  - [ ] Set `state.metadata["claim_verification_started"] = True` and `state.metadata["claim_verification_in_progress"] = True`, then persist state before verification
  - [ ] Snapshot `report_snapshot = state.report` before `try` block (for rollback on exception)
  - [ ] Clear `claim_verification_in_progress` in a `finally` block (so status polling shows correct state even on failure)
  - [ ] Wrap entire verification block in `try/except Exception` — on failure, **rollback** `state.report = report_snapshot`, log warning, set `state.metadata["claim_verification_skipped"]`, proceed to `mark_completed()`
  - [ ] After corrections succeed, call `self.memory.save_deep_research(state)` to persist corrected report + verification result (survives crash before `mark_completed`)
  - [ ] Add audit event logging for verification completion (claims_extracted, verified, contradicted, corrections_applied)
- [ ] Add resume guard at top of SYNTHESIS block (before `_run_phase` call)
  - [ ] If `state.report` exists and `claim_verification_started` is True but `state.claim_verification` is None → skip synthesis, run verification only
  - [ ] If `state.claim_verification` is already populated → skip verification entirely
- [ ] Re-save report markdown after corrections using `Path(state.report_output_path).write_text()` — do NOT re-call `_save_report_markdown()` (its collision logic would create a second file)
- [ ] **No changes needed to `core.py`** — verification is a standalone module, not a mixin

### Graceful Degradation

- [ ] Outer `try/except` in `workflow_execution.py` catches all verification errors
  - [ ] **Rollback** `state.report = report_snapshot` to avoid delivering a partially-corrected (garbled) report
  - [ ] Log a warning-level audit event: `claim_verification_failed` with error string
  - [ ] Set `state.metadata["claim_verification_skipped"]` to the error message string
  - [ ] Proceed to `mark_completed()` with the original (pre-correction) report
  - [ ] Do NOT fail the entire research session because of a verification error
- [ ] Claim extraction failure (invalid JSON / empty) returns empty `ClaimVerificationResult` without raising
  - [ ] Set `state.metadata["claim_verification_skipped"] = "extraction_failed"`

### Testing

**Mock strategy**: All unit tests for `claim_verification.py` must mock `execute_fn` (the LLM execution callable passed as a dependency). Because the module uses free functions (not mixins), mocking is straightforward — pass a mock/stub async callable directly. Create shared test fixtures for: (1) a minimal `DeepResearchState` with populated `sources` and `report`, (2) a `ResearchConfig` with verification enabled, (3) a mock `execute_fn` that returns canned JSON responses for extraction and verification prompts.

- [ ] Unit tests for claim extraction
  - [ ] Test extraction from report with mixed claim types
  - [ ] Test parsing with malformed JSON (fallback → empty list, no exception)
  - [ ] Test parsing with truncated JSON output (partial parse or empty list)
  - [ ] Test parsing with empty array `[]` (valid but no claims)
  - [ ] Test claim type classification
  - [ ] Test cited source number validation against citation map
  - [ ] Test filtering/prioritization with more claims than `max_claims`
  - [ ] Test deterministic positive claim sampling (same input → same sample set)
- [ ] Unit tests for claim-source alignment
  - [ ] Test SUPPORTED verdict (claim matches source)
  - [ ] Test CONTRADICTED verdict (source directly refutes claim)
  - [ ] Test UNSUPPORTED verdict (source has no relevant info)
  - [ ] Test with negative claim type (the Aeroplan case)
  - [ ] Test with dangling citation number (source not in citation map → skipped with warning)
- [ ] Unit tests for correction application
  - [ ] Test quote-context window extraction (found, correct paragraph-boundary clamping)
  - [ ] Test context window replacement uses `count=1` (duplicate text elsewhere not affected)
  - [ ] Test post-replacement sanity check (broken paragraph boundaries → revert to pre-correction report)
  - [ ] Test quote-context fallback (not found → full-report correction with `report_section` hint)
  - [ ] Test correction budget cap: >5 CONTRADICTED claims → only top 5 corrected, rest logged with `correction_applied=False`
  - [ ] Test correction priority ordering: negative > quantitative > comparative > positive
  - [ ] Test ` (unverified)` annotation insertion when `annotate_unsupported=True` (quote context found)
  - [ ] Test annotation skip when `annotate_unsupported=False` (default — no annotation, logged only)
  - [ ] Test annotation skip when quote context not found in report
  - [ ] Test single-pass correction (no re-verification)
  - [ ] Test sequential correction ordering: two CONTRADICTED claims with overlapping context windows — verify corrections are applied one at a time and the second operates on the already-corrected report (no garbled output from parallel mutation)
  - [ ] Test context drift after earlier correction: two adjacent CONTRADICTED claims where correction #1 alters text near claim #2's `quote_context`, causing substring match to fail — verify claim #2 falls back to full-report correction with `report_section` hint (not crash or skip)
  - [ ] Test paragraph boundary clamping expands outward (window grows to include full paragraphs, never shrinks)
- [ ] Unit tests for serialization round-trip
  - [ ] Test `ClaimVerificationResult` survives `DeepResearchState` serialize → deserialize
  - [ ] Test state with `claim_verification=None` (backward compat with existing saved states)
  - [ ] Test new-format state loaded by model without `claim_verification` field (forward compat — extra field silently ignored because `DeepResearchState` does not use `extra="forbid"`)
- [ ] Integration test: end-to-end verification with seeded hallucination
  - [ ] Create a report with a known-false negative claim
  - [ ] Provide source content that contradicts it (via populated citation map)
  - [ ] Verify detection and correction
- [ ] Integration test: resume after crash mid-verification
  - [ ] Save state with `claim_verification_started=True`, `claim_verification=None`, and populated `report`
  - [ ] Resume → verification runs without re-running synthesis
  - [ ] Save state with `claim_verification` already populated → verification skipped entirely
  - [ ] Save state with `claim_verification_in_progress=True` (crash during verification) — verify resume guard still triggers re-run correctly (stale `in_progress` flag does not interfere)
- [ ] Unit tests for report overwrite after corrections
  - [ ] Test `Path(state.report_output_path).write_text()` overwrites the synthesis-created file
  - [ ] Test that corrections are skipped when `report_output_path` is None (no crash)
- [ ] Unit tests for report snapshot/rollback
  - [ ] Test that `apply_corrections` exception mid-way rolls back `state.report` to pre-correction snapshot (no partial corrections delivered)
  - [ ] Test that original report markdown file is NOT overwritten when corrections fail (rollback means `write_text` is never reached)
- [ ] Unit tests for post-correction persistence
  - [ ] Test that `save_deep_research(state)` is called after corrections succeed (corrected report + ClaimVerificationResult survive crash before `mark_completed`)
- [ ] Unit tests for `_resolve_source_text`
  - [ ] Test with `.content` populated (returns `.content`)
  - [ ] Test with `.content=None`, `.raw_content` populated (returns `.raw_content`)
  - [ ] Test with `.content=None`, `.raw_content=None`, `.snippet` populated (returns `.snippet`)
  - [ ] Test with all three `None` (returns `None`)
- [ ] Unit tests for `_keyword_proximity_truncate`
  - [ ] Test keyword found mid-document (window centered on keyword)
  - [ ] Test keyword found near start (window clamped to start)
  - [ ] Test keyword found near end (window clamped to end)
  - [ ] Test no keyword match (falls back to prefix truncation)
  - [ ] Test source shorter than `max_chars` (returned as-is, no truncation)
- [ ] Unit tests for token budget escape hatch
  - [ ] Test that claims are dropped from tail of priority list when estimated input tokens exceed `max_input_tokens`
  - [ ] Test that estimation uses `sum(len(sources_text) / 3.5)` per claim (conservative char-to-token ratio)
  - [ ] Test that token budget check happens after prioritization/filtering, before dispatch
  - [ ] Test that Pass 1 (extraction) is unaffected by the token budget cap
- [ ] Unit tests for edge cases
  - [ ] Test verification when `state.report` is empty string or None (should short-circuit gracefully)
  - [ ] Test source content truncation at `VERIFICATION_SOURCE_MAX_CHARS` boundary
- [ ] Graceful degradation test
  - [ ] Simulate verification timeout → report delivered unverified, metadata records skip reason
  - [ ] Simulate LLM parse failure in extraction → empty result, report delivered
  - [ ] Simulate LLM parse failure in single claim verification → claim skipped, others still verified
  - [ ] Simulate concurrent verification with partial provider failures (some claims verified, some skipped)
- [ ] Performance test: verify latency stays within 15-45 second budget for typical report

### User-Facing Output

- [ ] Surface verification summary in deep-research status/report output
  - [ ] Include `claim_verification` field in the research result returned to the user
  - [ ] Summary line in report metadata: e.g., "Verified 23 claims: 20 supported, 1 corrected, 2 unsupported"
  - [ ] When verification was skipped, note it: "Claim verification: skipped (reason)"
- [ ] Surface in-progress status during verification
  - [ ] `deep-research-status` formatter checks `state.metadata.get("claim_verification_in_progress")` and shows "Verifying claims…" when True
- [ ] Document/expose config fields for user enablement
  - [ ] Add `claim_verification_enabled` as a supported key in research profile definitions (so profiles like `general` can opt in)
  - [ ] Ensure the `deep-research` tool description or help text mentions claim verification as an opt-in feature
