# Plan: Deep Research Claim Verification Pipeline

## Problem Statement

The deep research synthesis phase can hallucinate factual claims that are directly contradicted by its own source material. In a real-world case, the synthesis model asserted "Amex does not transfer to Air Canada Aeroplan" despite multiple fetched sources (including the report's own cited sources [15], [35], [63]) explicitly listing Aeroplan as an Amex transfer partner at 1:1.

**Root cause analysis** identified three contributing factors:

1. **Compression data loss**: Structured data (tables, comparison lists) in source material can be paraphrased during the compression phase (`compression.py`), losing precise factual content that would have prevented the error.
2. **No grounding guardrails in synthesis prompt**: The synthesis system prompt (`_build_synthesis_system_prompt()` in `synthesis.py`) instructs "write directly and authoritatively" and bans hedging, which encourages confident assertion — including of hallucinated claims. There are no instructions about negative claims requiring explicit source evidence.
3. **No post-synthesis verification**: The pipeline proceeds directly from synthesis to completion. Citation post-processing (`_citation_postprocess.py`) verifies citation format integrity (dangling refs, valid numbers) but never checks whether claims are actually supported by cited sources.

## Solution: Three-Layer Defense-in-Depth

### Layer 1: Structured Data Preservation in Compression

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/compression.py`

**Problem**: The compression prompt says "DO NOT summarize the information" and "preserve all relevant information", but structured data (markdown tables, bulleted lists of proper nouns with values) is still at risk of being paraphrased into lossy prose. A table like `| Amex: Aeroplan 1:1 | Chase: Aeroplan 1:1 |` could become "Chase offers access to Aeroplan" — losing the Amex column entirely.

**Change**: Prompt-based preservation with post-validation. Rather than extracting structured blocks into placeholders (which risks disconnecting the compressed prose from the data it references), we add explicit preservation instructions to the compression prompt and validate that structured data survives compression.

**Codebase context**: There is no single `_build_compression_prompt()` function. The compression module has two paths:
- **Per-topic compression** (`_compress_single_topic_async`): System prompt is built inline as a string concatenation. User prompt is built by `_build_message_history_prompt()` or `_build_structured_metadata_prompt()` as a fallback.
- **Global cross-topic compression** (`_execute_global_compression_async`): Separate system prompt built inline.

Structured data loss primarily occurs during **per-topic compression** (the raw source material flows through there), so changes target that path.

**Implementation**:

1. Add a `_detect_structured_blocks(text: str) -> list[str]` function that:
   - Detects markdown tables: consecutive lines matching `|...|...|` patterns (2+ pipe-delimited rows, including header separator lines with `---`)
   - Detects definition-style bullet lists: lines matching `- **Term**: value` or `- Term — value` patterns where value contains a number, ratio, price, or date (regex: `r'^[-*]\s+\*{0,2}.+?\*{0,2}\s*[-:—–]\s+.*\d.*'` — note the character class starts with `-` to avoid ambiguous range interpretation, and `\*{0,2}` correctly matches 0-2 asterisks for optional markdown bold)
   - Returns the list of detected blocks as raw text strings (for validation, not extraction)
   - Does NOT attempt general "proper noun" detection — focuses on mechanically-detectable structures
   - **False positives are acceptable**: Detection is used to feed the validation function, not to make pass/fail decisions directly. A false positive (e.g., a normal prose bullet matching the pattern) just means the validator checks an extra block — this is harmless. The correctness gate is `_validate_structured_data_survival`, not detection.

2. Add a structured data preservation instruction to the **per-topic compression system prompt** (the inline string in `_compress_single_topic_async()`), appended after the `</Citation Rules>` closing tag and before the "Critical Reminder" paragraph (the paragraph beginning "Critical Reminder: It is extremely important that any information…"):
   ```
   <Structured Data Preservation>
   - Markdown tables MUST be reproduced VERBATIM in your output. Do not paraphrase tables into prose.
   - Bulleted lists containing proper nouns with numeric values (prices, ratios, dates) must be
     preserved exactly as written. These contain precise factual data that cannot be safely rephrased.
   </Structured Data Preservation>
   ```

3. Add a `_validate_structured_data_survival(original: str, compressed: str, blocks: list[str]) -> bool` function that:
   - Counts markdown table rows (`|...|` lines) **within the detected blocks only** (not across the entire original document) vs the compressed output — fail if compressed count < block-sourced count. This scoping avoids false positives from unrelated tables in the source material that the compression model legitimately dropped as irrelevant to the research topic. Only tables that were detected as structured data blocks (and thus flagged for preservation) are checked.
   - Extracts numeric data tokens from detected blocks using regex `r'\d[\d,./:]+' ` and checks for their literal presence in the compressed output. This avoids the need for proper noun detection (which would require NLP) — numeric tokens are sufficient and mechanically detectable
   - Returns False if either check fails (table row loss OR missing numeric tokens)

4. Update `_compression_output_is_valid()` to accept an optional `structured_blocks: list[str] | None` parameter and call the new validation function as an additional check. When validation fails, the existing behavior applies: `message_history` is retained instead of being cleared.

   **Wiring (no return type change needed)**: Detect structured blocks in `_compress_topic_findings_async()`, not inside `_compress_single_topic_async()`. The caller already has access to each `TopicResearchResult`'s `message_history` before compression runs — call `_detect_structured_blocks()` on the concatenated message content there, then pass the result to `_compression_output_is_valid()` at the validation call site. This avoids widening `_compress_single_topic_async`'s `tuple[int, int, bool]` return type, which would be a needless interface change forcing all call sites to unpack a 4th value.

**Why not extract-and-reattach?** The placeholder approach (`[STRUCTURED_BLOCK_N]`) creates a risk where the compressed prose references or depends on data from an extracted table, producing incoherent output when blocks are reattached. Keeping structured data in-context lets the model reason over it while the explicit prompt instruction protects it from paraphrasing.

**Scope**: ~60-80 lines of new code in `compression.py`

### Layer 2: Synthesis Prompt Grounding Enhancement

**File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

**Problem**: The `## Writing Quality` section of the synthesis system prompt bans hedging and meta-commentary, which is good for writing quality but provides no counterbalance for factual grounding — especially for negative claims ("X does NOT do Y") which are the highest-risk hallucination type.

**Change**: Add a `## Factual Grounding` section to the **base prompt** in `_build_synthesis_system_prompt()`, between the existing `## Writing Quality` and `## Citations` sections. This must be in the base prompt (not in a conditional academic/literature-review block) because factual grounding applies to all research modes.

**Content**:

```
## Factual Grounding

- CRITICAL: Never claim that something is absent, excluded, or unavailable unless a source
  EXPLICITLY states that exclusion. If a source lists items (e.g., transfer partners, features,
  supported platforms), you may state what IS listed but must not infer what is NOT listed.
  Negative claims ("X does not support Y", "Y is not available through X") require direct
  source evidence — not inference from a list's absence of mention.
- When making comparative claims (X has Y but Z does not), verify BOTH sides against sources.
  Do not assume exclusivity from a source that only describes one side of the comparison.
- If sources conflict on a factual point, acknowledge the conflict rather than choosing one
  side and presenting it as definitive.
- Quantitative claims (prices, rates, dates, ratios) must be traceable to a specific source.
  Do not combine or extrapolate numbers from multiple sources without noting the synthesis.
```

**Scope**: ~15 lines added to the system prompt string in `_build_synthesis_system_prompt()`

### Layer 3: Post-Synthesis Claim Verification Phase

**New file**: `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

**Problem**: Even with improved compression and prompt grounding, synthesis hallucination can still occur. There is no safety net between report generation and delivery to the user.

**Design**: A two-pass LLM-based verification that runs after synthesis and before `mark_completed()`. Implemented as a **standalone module with free async functions** (`extract_claims`, `verify_claims`, `apply_corrections`) called from `WorkflowExecutionMixin`. This avoids adding a 14th mixin to `DeepResearchWorkflow`'s already-complex MRO. The functions receive explicit dependencies (`state`, `config`, `provider_id`, `execute_fn`) as arguments rather than accessing `self`.

#### Pass 1: Claim Extraction

A single LLM call extracts verifiable factual claims from the generated report as structured JSON:

```json
[
  {
    "claim": "Amex does not transfer to Air Canada Aeroplan",
    "claim_type": "negative",
    "cited_sources": [35, 63],
    "report_section": "Your Existing Portfolio",
    "quote_context": "Critically, Amex does not transfer to Air Canada Aeroplan..."
  },
  {
    "claim": "Aeroplan prices transatlantic business class at 60,000 miles one-way",
    "claim_type": "quantitative",
    "cited_sources": [35],
    "report_section": "Chase Ultimate Rewards",
    "quote_context": "...prices transatlantic business class at 60,000 miles one-way..."
  }
]
```

Claim types and their verification priority:
- **`negative`** ("X does NOT do Y") — ALWAYS verify (highest hallucination risk)
- **`quantitative`** (specific numbers, dates, prices) — ALWAYS verify
- **`comparative`** ("X is better/worse than Y") — verify when both sides are sourced
- **`positive`** ("X does Y") — verify by deterministic sampling (lower risk)

The extraction prompt instructs the model to focus on factual assertions (not opinions, recommendations, or subjective assessments) and to capture the exact cited source numbers.

**Positive claim sampling**: Uses a deterministic hash of the claim text (`hashlib.sha256(claim.encode()).hexdigest()`) modulo 100, compared against `sample_rate * 100`. This ensures the same report always produces the same verification set, making debugging reproducible. (SHA-256 over MD5 to avoid FIPS-mode warnings in restricted environments.)

#### Pass 2: Claim-Source Alignment

For each claim flagged for verification (all negative + quantitative, sampled positive/comparative), up to `claim_verification_max_claims`:

1. **Build the citation map once** by calling `state.get_citation_map()` at the start of the verification batch (not per-claim). This method iterates `state.sources` on every call, so materializing it once and passing the `dict[int, ResearchSource]` to per-claim functions avoids redundant iteration. Each `ResearchSource` has `.content`, `.title`, and `.url` fields. **Important**: `.content` is only populated when `follow_links` was enabled during research; when it is `None`, fall back to `.raw_content` (original unprocessed content), then `.snippet` (search result excerpt). If all three are `None`, skip the source with a warning — there is no text to verify against. If a cited number is not found in the citation map (dangling reference), skip that source and log a warning — the claim can still be verified against its remaining cited sources.

2. **Prioritization when claims exceed budget**: When more claims pass the type filter than `claim_verification_max_claims` allows, prioritize in this order: (1) all `negative` claims first, (2) `quantitative` claims, (3) `comparative` claims, (4) sampled `positive` claims. Within each type, prioritize claims with more cited sources (more material to verify against).

3. **Source content truncation**: Each source's content is truncated to `VERIFICATION_SOURCE_MAX_CHARS` (default: 8,000 characters) in verification prompts. Without truncation, a claim citing 3 large sources could produce a 30K+ token input per verification call. With 50 parallel calls, this risks 1.5M+ input tokens — far exceeding the "20-50K additional tokens" budget.

   **Keyword-proximity truncation** (not naive prefix): Extract keywords from the claim text by splitting on whitespace and filtering out short words (< 4 characters) plus a small explicit stopword set (`_STOPWORDS: frozenset` — ~20 common function words: `{"this", "that", "with", "from", "have", "been", "will", "would", "could", "should", "their", "there", "which", "about", "where", "these", "those", "does", "into", "also"}`, defined as a module-level constant in `claim_verification.py`, no external dependency). The length filter handles most determiners/prepositions/conjunctions ("the", "is", "of", "to", "and", "in", "for") without maintaining a large curated list. Search the source content for the first occurrence of any keyword. If found, extract a window of `VERIFICATION_SOURCE_MAX_CHARS` centered on that position (clamped to content boundaries). If no keywords match, fall back to prefix truncation (first `VERIFICATION_SOURCE_MAX_CHARS` characters). This is ~10 extra lines of code and dramatically improves verification accuracy for long sources where the relevant evidence appears deep in the document. The worst case (50 claims × 3 sources × 8K chars ≈ 1.2M chars ≈ 300K tokens) is unlikely in practice; the typical case (15-20 claims, 1-2 sources each) aligns with the 30-80K token estimate.

4. Build a verification prompt:

```
## Source Content
[Source [35] title and content...]
[Source [63] title and content...]

## Claim to Verify
"Amex does not transfer to Air Canada Aeroplan"
Claim type: negative
Cited sources: [35], [63]

## Task
Does the source content SUPPORT, CONTRADICT, or provide NO EVIDENCE for this claim?
Return a JSON object with: verdict, evidence_quote, explanation
```

5. Parse the structured response:

```json
{
  "verdict": "CONTRADICTED",
  "evidence_quote": "Air Canada Aeroplan | 1:1 ratio | listed under American Express column",
  "explanation": "Source [63] explicitly lists Aeroplan as an Amex transfer partner at 1:1"
}
```

#### Verdict Handling

| Verdict | Action |
|---------|--------|
| `SUPPORTED` | No action needed |
| `CONTRADICTED` | Flag the claim; attempt single-pass targeted re-synthesis of the affected section with contradicting evidence |
| `UNSUPPORTED` | Logged in verification details and summary metadata (no inline annotation by default) |
| `PARTIALLY_SUPPORTED` | No action; logged in verification details for transparency |

**UNSUPPORTED handling**: By default, UNSUPPORTED claims are recorded in `ClaimVerificationResult.details` and surfaced in the verification summary metadata — but no inline annotation is added to the report. This avoids noise from false positives (claims that are correct but whose source material happened to be weak or truncated). An optional config flag `deep_research_claim_verification_annotate_unsupported: bool = False` enables inline ` (unverified)` annotations for users who want them. When enabled, the annotation (parenthesized plain text — avoids `[text]` bracket syntax that markdown parsers interpret as links) is inserted immediately after the sentence containing the claim text (identified by substring match of `claim.quote_context` in `state.report`). If the quote context is not found (e.g., synthesis rephrased it), the annotation is skipped and logged.

**CONTRADICTED handling — targeted re-synthesis**: Re-prompt the synthesis model with a focused context window around the claim, the original findings, AND the contradicting source evidence, asking it to correct the claim. This is cheaper than re-running full synthesis and preserves the rest of the report.

**Context window identification for corrections**: Uses `quote_context` (captured during claim extraction) rather than heading-based section splitting, avoiding fragility with nested/ambiguous headings. The correction function:
1. Locates `claim.quote_context` in `state.report` via substring match
2. Extracts a context window of ~500 characters before and after the match, clamped to paragraph boundaries by **expanding outward** to the nearest `\n\n`. Specifically: scan backward from `match_start - 500` to find the closest preceding `\n\n` (or start of string), and scan forward from `match_end + 500` to find the closest following `\n\n` (or end of string). This ensures the window always contains complete paragraphs, never splits mid-sentence, and grows slightly rather than shrinking.
3. Sends the context window + contradicting source evidence to the correction LLM, instructing it to rewrite only the portion containing the false claim
4. Replaces the original context window in `state.report` with the corrected version using `state.report.replace(original_window, corrected_window, 1)` — the `count=1` argument limits replacement to the first occurrence, preventing accidental mutation of duplicate text elsewhere in the report
5. **Post-replacement sanity check**: After replacement, verify the corrected report still contains the paragraph boundaries (`\n\n` delimiters) that bracketed the original context window. If the correction LLM returned text that broke paragraph structure, log a warning and revert to the pre-correction `state.report` for that claim. This prevents garbled output from an overly aggressive correction.
6. If `quote_context` is not found in the report (e.g., synthesis rephrased it), falls back to full-report correction with explicit instruction to change only the contradicted claim and a hint about which section (`report_section`) to look in

**Corrections are applied sequentially, not in parallel.** Two CONTRADICTED claims may have overlapping or adjacent context windows. Parallel correction would race on `state.report`, potentially producing garbled output when both replacements land. Sequential application ensures each correction operates on the report as modified by all prior corrections. The iteration order follows the same priority used for budget selection (negative > quantitative > comparative > positive, then by cited source count).

**Single-pass correction only**: Corrections are NOT re-verified. A correction that itself introduces errors is an acceptable (and rare) tradeoff vs. the cost of recursive verification. The correction prompt explicitly instructs the model to only fix the specific contradicted claim without altering surrounding content, minimizing blast radius.

**Correction budget cap**: Corrections are capped at `deep_research_claim_verification_max_corrections` (default: 5). If more than 5 claims are CONTRADICTED, only the highest-priority ones are corrected, using the same priority order as verification: negative > quantitative > comparative > positive, with ties broken by number of cited sources (more sources = higher priority). Remaining CONTRADICTED claims beyond the cap are logged in `ClaimVerificationResult.details` with `correction_applied=False` but are not annotated or modified — their presence in the verification metadata serves as a signal that the report may have broader quality issues. This prevents unbounded correction LLM calls in pathologically bad synthesis outputs.

#### Claim Extraction Failure Handling

If Pass 1 (claim extraction) fails — LLM returns invalid JSON, empty array, or truncated output — the entire verification pipeline **returns an empty `ClaimVerificationResult`** (all counters at 0, empty details list) and sets `state.metadata["claim_verification_skipped"] = "extraction_failed"`. The report proceeds to `mark_completed()` unverified. This is logged as a warning-level audit event. Rationale: verification is a safety net, not a gate — extraction failure should never block report delivery.

#### Verification Model Selection

The verification model SHOULD differ from the synthesis model when possible, to avoid correlated failures (same model, same blind spots). The existing `resolve_phase_provider()` function supports this via dynamic attribute lookup — adding a `deep_research_claim_verification_provider` config field automatically makes `resolve_phase_provider(config, "claim_verification")` resolve it. The fallback chain is: `claim_verification` provider → `synthesis` provider → `default_provider`. Using the synthesis provider as the first fallback (rather than `report` provider) is intentional: the verification task is most analogous to synthesis (reading sources, evaluating claims), and reusing the same provider is an acceptable default when no explicit override is set.

#### Cost & Performance Budget

- **Claim extraction**: 1 LLM call (~2K output tokens, report as input ~5-20K tokens)
- **Verification**: N parallel LLM calls (capped at `claim_verification_max_concurrent`, default: 10) where N = min(claims_to_verify, `claim_verification_max_claims`). Typical report has ~10-30 negative/quantitative claims; comprehensive comparison reports may have 50-100+. The `max_claims` cap (default: 50) prevents unbounded cost. Per-claim input is bounded by `VERIFICATION_SOURCE_MAX_CHARS` truncation (default: 8,000 chars per source).
- **Targeted re-synthesis** (if needed): 0 to `max_corrections` (default: 5) LLM calls for contradicted sections
- **Expected latency addition**: 15-45 seconds (parallelized verification)
- **Expected token cost**: ~30-80K additional tokens per research session (dominated by verification inputs). Comprehensive comparison reports with many sources may reach ~100-150K tokens at `max_claims=50`. Source truncation is the primary cost control lever.
- **Total token budget escape hatch**: A `deep_research_claim_verification_max_input_tokens` config field (default: 200,000) caps the estimated total input tokens for the verification batch. Before dispatching verification calls, `extract_and_verify_claims` estimates the total input tokens as `sum(len(sources_text) / 4 for each claim)` (rough char-to-token ratio). If the estimate exceeds the cap, claims are dropped from the tail of the priority list until the estimate fits. This prevents pathological cases (100-source report, 200 claims) from producing unbounded LLM costs. The cap is checked after prioritization/filtering and before dispatch — it does not affect claim extraction (Pass 1), only verification (Pass 2).

#### Integration Point

In `workflow_execution.py`, between synthesis completion (after `_run_phase` returns) and `mark_completed()`. The verification runs **after** the orchestrator transition (`evaluate_phase_completion`, `decide_iteration`, `get_reflection_prompt`, `think_pause`, `record_to_state`) but **before** the completion block (`state.mark_completed(report=state.report)`):

```python
# SYNTHESIS
if state.phase == DeepResearchPhase.SYNTHESIS:
    err = await self._run_phase(...)
    if err:
        return err

    # Orchestrator transition (unchanged)...
    self.orchestrator.evaluate_phase_completion(state, DeepResearchPhase.SYNTHESIS)
    self.orchestrator.decide_iteration(state)
    prompt = self.orchestrator.get_reflection_prompt(state, DeepResearchPhase.SYNTHESIS)
    self.hooks.think_pause(state, prompt)
    self.orchestrator.record_to_state(state)

    # --- NEW: CLAIM VERIFICATION ---
    if self.config.deep_research_claim_verification_enabled:
        from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
            extract_and_verify_claims, apply_corrections,
        )

        state.metadata["claim_verification_started"] = True
        state.metadata["claim_verification_in_progress"] = True
        self.memory.save_deep_research(state)  # checkpoint before verification

        try:
            verification_result = await extract_and_verify_claims(
                state=state,
                config=self.config,
                provider_id=resolve_phase_provider(self.config, "claim_verification", "synthesis"),
                execute_fn=self._execute_provider_async,
                timeout=self.config.deep_research_claim_verification_timeout,
            )
            state.claim_verification = verification_result

            if verification_result.claims_contradicted > 0:
                await apply_corrections(
                    state=state,
                    config=self.config,
                    verification_result=verification_result,
                    execute_fn=self._execute_provider_async,
                )
                # Re-save report after corrections — use the SAME path
                # that synthesis wrote to, avoiding _save_report_markdown's
                # collision logic which would create a second file.
                if state.report_output_path:
                    Path(state.report_output_path).write_text(state.report, encoding="utf-8")

            self._write_audit_event(state, "claim_verification_complete", data={
                "claims_extracted": verification_result.claims_extracted,
                "claims_verified": verification_result.claims_verified,
                "claims_contradicted": verification_result.claims_contradicted,
                "corrections_applied": verification_result.corrections_applied,
            })
        except Exception as exc:
            logger.warning(
                "Claim verification failed for research %s, delivering unverified report: %s",
                state.id, exc,
            )
            state.metadata["claim_verification_skipped"] = str(exc)
            self._write_audit_event(state, "claim_verification_failed", data={
                "error": str(exc),
            }, level="warning")
        finally:
            state.metadata.pop("claim_verification_in_progress", None)
    # --- END CLAIM VERIFICATION ---

    # No refinement — workflow complete
    state.metadata["iteration_in_progress"] = False
    state.metadata["last_completed_iteration"] = state.iteration
    state.mark_completed(report=state.report)
```

Claim verification does NOT need its own `DeepResearchPhase` enum value — it runs as a sub-step within the SYNTHESIS phase, after report generation but before completion. This avoids state machine changes and simplifies resume logic.

**Note on graceful degradation**: The entire verification block is wrapped in a `try/except` in the integration snippet above. If verification crashes, times out, or encounters any unhandled error, the exception is caught, logged as a warning audit event, and the report proceeds to `mark_completed()` unverified. Verification failures must **never** fail the research session.

#### Resumability

If the process crashes mid-verification, resume will re-enter the SYNTHESIS phase. To avoid re-running full synthesis when the report already exists:

- Before verification starts, set `state.metadata["claim_verification_started"] = True` and persist state.
- On resume, if `state.report` is populated and `state.metadata.get("claim_verification_started")` is True but `state.claim_verification` is None, skip synthesis and re-run only verification.
- If `state.claim_verification` is already populated, skip verification entirely (already completed).

This requires a small guard at the top of the SYNTHESIS block in `workflow_execution.py`:

```python
if state.phase == DeepResearchPhase.SYNTHESIS:
    # Resume guard: skip synthesis if report exists and verification was in progress
    if state.report and state.metadata.get("claim_verification_started") and not state.claim_verification:
        logger.info("Resuming claim verification (synthesis already complete)")
        # Jump directly to verification
    elif not state.report:
        err = await self._run_phase(...)
        ...
```

#### State Model Extensions

Add to `models/deep_research.py`:

```python
class ClaimVerdict(BaseModel):
    """Verification result for a single claim."""
    claim: str
    claim_type: str  # negative, quantitative, comparative, positive
    cited_sources: list[int]
    verdict: str  # SUPPORTED, CONTRADICTED, UNSUPPORTED, PARTIALLY_SUPPORTED
    evidence_quote: Optional[str] = None
    explanation: Optional[str] = None
    correction_applied: bool = False
    corrected_text: Optional[str] = None

class ClaimVerificationResult(BaseModel):
    """Result of post-synthesis claim verification."""
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_supported: int = 0
    claims_contradicted: int = 0
    claims_unsupported: int = 0
    corrections_applied: int = 0
    details: list[ClaimVerdict] = Field(default_factory=list)
```

Add `claim_verification: Optional[ClaimVerificationResult] = None` to `DeepResearchState`.

#### Configuration

Add to `ResearchConfig` (following existing `deep_research_` prefix convention):

```python
deep_research_claim_verification_enabled: bool = False  # Opt-in until battle-tested
deep_research_claim_verification_sample_rate: float = 0.3  # Sample 30% of positive claims
deep_research_claim_verification_provider: Optional[str] = None  # Override verification provider
deep_research_claim_verification_model: Optional[str] = None  # Override verification model
deep_research_claim_verification_timeout: int = 120  # Seconds (overall verification phase)
deep_research_claim_verification_max_claims: int = 50  # Max claims to verify per report
deep_research_claim_verification_max_concurrent: int = 10  # Max parallel verification LLM calls
deep_research_claim_verification_max_corrections: int = 5  # Max correction LLM calls per report
deep_research_claim_verification_annotate_unsupported: bool = False  # Inline (unverified) annotations
deep_research_claim_verification_max_input_tokens: int = 200_000  # Total token budget escape hatch
```

**Default `enabled=False`**: This adds 15-45s latency and 30-80K tokens per session. Until the feature is validated through opt-in usage, defaulting to on is too aggressive. Enable via config or research profile (e.g., the `general` profile could set it to True once validated).

#### Status Visibility During Verification

During the 15-45s verification window, the workflow phase is still `SYNTHESIS` (verification is a sub-step, not a separate phase). To give users visibility, set `state.metadata["claim_verification_in_progress"] = True` before verification starts and clear it after. The `deep-research-status` formatter can surface this as "Verifying claims…" when the metadata flag is set. This is a display-only concern — no state machine changes needed.

#### Report Auto-Save Ordering

The current flow saves the report markdown inside `_finalize_synthesis_report()` (via `_save_report_markdown()`) before returning to `workflow_execution.py`, and stores the path in `state.report_output_path`. If verification modifies `state.report`, the saved file becomes stale.

**Important**: We cannot re-call `_save_report_markdown()` for the overwrite because its collision-handling logic appends a research-ID suffix when the file already exists — this would create a *second* file instead of overwriting the first. Instead, the integration code writes directly to `state.report_output_path` using `Path.write_text()`, which overwrites the exact file that synthesis created. This requires `state.report_output_path` to be populated, which is already the case in the current synthesis flow.

## File Change Summary

| File | Change Type | Scope |
|------|------------|-------|
| `phases/compression.py` | Modify | Add structured data detection, prompt instruction, validation (~60-80 lines). No return type changes to `_compress_single_topic_async`. |
| `phases/synthesis.py` | Modify | Add Factual Grounding section to base system prompt (~15 lines) |
| `phases/claim_verification.py` | **New file** | Standalone module: `extract_claims`, `verify_claims`, `apply_corrections` free functions, keyword-proximity truncation helper (~450-550 lines) |
| `models/deep_research.py` | Modify | Add `ClaimVerdict`, `ClaimVerificationResult` models, `claim_verification` field on state (~35 lines) |
| `workflow_execution.py` | Modify | Insert verification step + resume guard + graceful error handling (~40 lines) |
| `_constants.py` | Modify | Add verification budget constants (~6 lines) |
| `config/research.py` | Modify | Add `deep_research_claim_verification_*` config fields (~30 lines) |

**Total estimated new/modified code**: ~650-750 lines

**Notes**:
- `core.py` does NOT need changes — verification is a standalone module imported directly in `workflow_execution.py`, not a mixin added to the class hierarchy.
- `phases/__init__.py` does NOT need changes — `claim_verification.py` is not a mixin, so it is not re-exported from `__init__.py`. It is imported directly by full module path in `workflow_execution.py`.

#### Serialization Compatibility

- **Backward compat** (loading old state): `claim_verification: Optional[ClaimVerificationResult] = None` defaults cleanly — old persisted states without this field deserialize without error.
- **Forward compat** (new state loaded by old code): `DeepResearchState` does NOT use `model_config = {"extra": "forbid"}`, so Pydantic will silently ignore the unknown `claim_verification` field when loaded by a version that predates this change. No data loss.
- `ResearchExtensions` DOES use `extra = "forbid"` — the `claim_verification` field is intentionally placed on `DeepResearchState` directly, not on `ResearchExtensions`.

## Implementation Order

1. **Layer 2 first** (prompt fix) — smallest change, immediately deployable, zero risk of regression
2. **Layer 1 second** (compression fidelity) — moderate change, improves data quality for all downstream consumers
3. **Layer 3 third** (claim verification) — largest change, requires new file, LLM calls, config, and testing

## Testing Strategy

- **Layer 2**: Manual regression test — re-run the credit card research query and verify the Aeroplan claim is no longer hallucinated
- **Layer 1**: Unit test `_detect_structured_blocks()` with markdown tables and structured lists; unit test `_validate_structured_data_survival()` with matched/mismatched pairs; integration test showing tables survive compression
- **Layer 3**: Unit tests for claim extraction parsing, verdict classification, citation map resolution, and correction application; integration test with a known-hallucination scenario (seed a report with a contradicted claim and verify detection); graceful degradation test (verification timeout/error delivers unverified report)

## Future Work (out of scope for this change)

- **Metrics collection**: Define aggregation strategy for verification quality signals (verification rate, contradiction rate, correction rate, UNSUPPORTED false-positive rate) to inform graduation from opt-in (`enabled=False`) to default-on. The audit events emitted by this implementation provide the raw data; aggregation and dashboarding are a separate effort.
