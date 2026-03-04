# Chunked Claim Extraction for Deep Research

**Branch:** `alpha`
**Date:** 2026-03-04
**Triggered by:** Evaluation of session `deepres-b454ed345c8f` (credit card research, general profile)

## Problem Statement

Claim verification extraction consistently times out on large reports. The Phase 3 fix (commit `b2c7b1f`) raised the timeout to 180s, added `max_tokens=16384`, and truncated reports to 30K chars — but the extraction LLM call still fails all 3 attempts on real-world reports.

### Evidence from `deepres-b454ed345c8f`

| Metric | Value |
|--------|-------|
| Report size | 47,303 chars (truncated to 30K for extraction) |
| Claim verification started | `true` |
| Claim verification skipped | `extraction_failed` |
| Claims extracted | 0 |
| Wall time: synthesis complete → claim verification complete | **551 seconds (~9.2 min)** |
| Expected 3x timeout (180s × 3 + 2 × 5s delay) | **550 seconds** |

The timing match is exact: the extraction call timed out at 180s on all 3 attempts. No `llm.call.started` audit event appears between synthesis completion and claim verification completion, confirming the call never succeeded.

### Why 180s Is Insufficient

The extraction prompt sends ~30K chars of report text and asks the LLM to output a structured JSON array of all verifiable factual claims (claim text, type, cited sources, section, quote context) with `max_tokens=16384`. For dense, data-rich reports (dozens of specific numbers, partner lists, award chart pricing), the model needs to:
1. Read and parse the entire 30K input
2. Identify 50–100+ verifiable claims
3. Output each as a structured JSON object (~150–200 chars each)
4. Total output: 10K–20K tokens

This is an inherently slow task for a single LLM call. The output volume alone approaches the timeout ceiling even if the model processes efficiently.

### Other Post-Synthesis Fixes — Status

The other two fixes shipped in this cycle are **working correctly**:

1. **Provider leakage fix** (`86078ae`): `active_providers = ["tavily"]` in metadata, `search_provider_stats` shows only Tavily (37 queries), 0 academic sources. Working as intended.

2. **Bibliography cited-only filter** (`da272c6`): 67 sources have citation numbers, but only 50 appear in report body. Bibliography correctly lists exactly 50 entries matching in-text citations. 17 unreferenced sources filtered out. Working as intended.

---

## Root Cause Analysis

### The Single-Call Bottleneck

`extract_and_verify_claims()` (claim_verification.py:752-868) sends the entire (truncated) report to a single LLM call for extraction:

```python
# claim_verification.py:793-799
extraction_result = await execute_fn(
    prompt=extraction_prompt,       # ~30K chars of report
    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    provider_id=provider_id,
    timeout=timeout,                # 180s
    phase="claim_extraction",
    max_tokens=16384,               # ~10-20K tokens of JSON output
)
```

The base executor (`base.py:318`) applies `max_retries=2` (3 total attempts) with `retry_delay=5.0` between attempts. When the extraction consistently times out:

```
Attempt 0: 180s timeout → fail
5s delay
Attempt 1: 180s timeout → fail
5s delay
Attempt 2: 180s timeout → fail
Total: 550s → extraction_failed, 0 claims
```

### Why Divide-and-Conquer Works

Verification (Pass 2) already uses divide-and-conquer — each claim is verified independently against its sources, running in parallel with `asyncio.Semaphore(max_concurrent=10)`. This completes efficiently.

Extraction (Pass 1) should follow the same pattern:
- A 30K report has 5–10 `##` sections
- Each section is 2K–6K chars — small enough for a 60s extraction call
- Each section's claims are independent (no cross-section dependencies)
- Running 5–10 sections in parallel with bounded concurrency yields ~30–60s total wall time vs. 550s of failures

---

## Solution: Chunked Parallel Extraction

Split the report into section-level chunks, extract claims from each chunk independently in parallel, then merge results into the existing pipeline.

### Architecture

```
Before (monolithic, times out):
  [30K report] → single LLM call (180s timeout) → [all claims]

After (chunked, citation-anchored, parallelized):
  [30K report] → split by ## headings → [chunk1, chunk2, ..., chunkN]
                                              ↓         ↓           ↓
                                    "For each [N],  "For each [N],  ...  (bounded concurrency)
                                     what claim?"   what claim?"
                                              ↓         ↓           ↓
                                         [claims1] [claims2]  [claimsN]
                                              ↓         ↓           ↓
                                    merge + dedup + drop uncited → [all claims]
                                              ↓
                                   (existing pipeline: filter → budget → verify → correct)
```

### Phase 1: Add Section Chunking Helper

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

Add a function to split a report into section-level chunks:

```python
def _split_report_into_sections(report: str) -> list[dict[str, str]]:
    """Split report into section-level chunks for parallel extraction.

    Splits on ## headings. Each chunk includes its heading and body.
    Sections smaller than a minimum threshold are merged with the next section
    to avoid overly small extraction calls.

    Returns:
        List of dicts with "section" (heading) and "content" (full text including heading).
    """
```

**Rules:**
- Split on `\n## ` boundaries (level-2 headings)
- Keep each section's heading as metadata for `report_section` field
- Merge consecutive small sections (< 500 chars) with the next section to avoid trivially small chunks
- If the report has no `##` headings (unusual), fall back to single-chunk behavior
- The bibliography/sources section (if present in the truncated report) is excluded from extraction since it contains no verifiable claims

**Location:** Insert after `_build_extraction_user_prompt()` (after line 203).

### Phase 2: Citation-Anchored Extraction Prompt

Replace the current "find all claims" extraction prompt with a citation-anchored approach. Instead of asking the LLM to decide what's a verifiable claim (prone to false positives — opinions, recommendations, tautologies), anchor extraction to inline citations: "For each `[N]` citation in this section, extract the factual claim it supports."

**Why this reduces false positives:**
- Recommendations and subjective conclusions rarely have inline citations
- Tautologies ("X is a credit card") never cite a source
- Hedged opinions ("widely regarded as the best") have no citation anchor
- Every extracted claim is pre-linked to a source, making verification targeted

**Current prompt** (`_EXTRACTION_SYSTEM_PROMPT`, lines 179-198):
```
You are a factual claim extraction assistant. Your task is to extract verifiable
factual claims from the research report below.

Focus on:
- Factual assertions (NOT opinions, recommendations, or subjective assessments)
- Negative claims ... — label as "negative"
- Quantitative claims ... — label as "quantitative"
- Comparative claims ... — label as "comparative"
- Positive factual claims ... — label as "positive"

For each claim, capture:
- "claim": the exact factual assertion
- "claim_type": one of "negative", "quantitative", "comparative", "positive"
- "cited_sources": list of citation numbers (integers) referenced by or near this claim
- "report_section": the section heading where this claim appears
- "quote_context": the exact sentence or short passage containing this claim

Return a JSON array of claim objects. Return ONLY the JSON array, no other text.
```

**New prompt** (`_EXTRACTION_SYSTEM_PROMPT`):
```
You are a factual claim extraction assistant. Your task is to identify verifiable
factual claims that are backed by inline citations [N] in the report section below.

For each inline citation [N] you find, extract the specific factual claim it supports.

Rules:
- ONLY extract claims that have an explicit [N] citation adjacent to them
- Skip opinions, recommendations, subjective assessments, and uncited statements
- Classify each claim:
  - "negative" — "X does NOT do Y", "X is not available"
  - "quantitative" — specific numbers, dates, prices, ratios, percentages
  - "comparative" — "X is better/worse than Y", "X has Y but Z does not"
  - "positive" — "X does Y", "X supports Z"
- If a single sentence cites multiple sources [1][2], extract ONE claim for that sentence
  with all citation numbers in cited_sources

For each claim, return:
- "claim": the exact factual assertion (do not paraphrase)
- "claim_type": one of "negative", "quantitative", "comparative", "positive"
- "cited_sources": list of citation numbers (integers) from the inline [N] references
- "report_section": the section heading this claim appears under
- "quote_context": the exact sentence containing this claim and its citation(s)

Return a JSON array of claim objects. Return ONLY the JSON array, no other text.
If no cited claims are found in this section, return an empty array: []
```

**Key changes:**
- "For each inline citation [N]" flips the extraction direction — citations drive claim discovery, not free-text scanning
- "ONLY extract claims that have an explicit [N] citation" is the hard filter
- "If a single sentence cites multiple sources, extract ONE claim" prevents duplication
- "Return an empty array: []" gives explicit guidance for sections with no citations (e.g., Executive Summary)

### Phase 3: Add Per-Chunk Extraction Function

Add a function that extracts claims from a single section chunk:

```python
async def _extract_claims_from_chunk(
    chunk: dict[str, str],
    execute_fn: ExecuteFn,
    system_prompt: str,
    provider_id: str,
    timeout: float,
    max_claims_per_chunk: int,
) -> list[ClaimVerdict]:
    """Extract claims from a single report section chunk.

    Args:
        chunk: Dict with "section" and "content" keys.
        execute_fn: LLM execution callable.
        system_prompt: Extraction system prompt.
        provider_id: LLM provider to use.
        timeout: Per-call timeout in seconds.
        max_claims_per_chunk: Max claims to parse from this chunk.

    Returns:
        List of ClaimVerdict objects extracted from this chunk.
    """
```

**Details:**
- Builds a focused extraction prompt using the chunk content (not the full report)
- Sets `max_tokens=4096` (proportional to chunk size — much less output needed per chunk)
- Uses `max_retries=1` via execute_fn parameter to reduce total retry wall time
- Parses response with existing `_parse_extracted_claims()`
- On failure, logs warning and returns empty list (graceful per-chunk degradation)
- Tags each claim with the chunk's section heading as `report_section`

**Location:** Insert after `_split_report_into_sections()`.

### Phase 4: Add Parallel Chunk Orchestrator

Add a function that runs extraction across all chunks with bounded concurrency:

```python
async def _extract_claims_chunked(
    report: str,
    execute_fn: ExecuteFn,
    provider_id: str,
    timeout: float,
    max_claims: int,
    max_concurrent: int,
) -> list[ClaimVerdict]:
    """Extract claims from report using parallel section-level chunking.

    Splits the report into section chunks, extracts claims from each in
    parallel with bounded concurrency, then merges and deduplicates results.

    Args:
        report: The (possibly truncated) report text.
        execute_fn: LLM execution callable.
        provider_id: LLM provider to use.
        timeout: Per-call timeout in seconds.
        max_claims: Max total claims to return.
        max_concurrent: Max parallel extraction calls.

    Returns:
        Merged, deduplicated list of ClaimVerdict objects.
    """
```

**Details:**
- Calls `_split_report_into_sections()` to get chunks
- If only 1 chunk (no headings or very short report), falls back to existing single-call extraction to avoid overhead
- Creates `asyncio.Semaphore(max_concurrent)` for bounded concurrency
- Gathers all chunk extraction tasks with `asyncio.gather(*tasks, return_exceptions=True)`
- Uses existing `check_gather_cancellation()` for cancellation safety
- Merges all claim lists, deduplicates by claim text (exact match), and caps at `max_claims`
- Logs per-chunk extraction results for observability

**Location:** Insert after `_extract_claims_from_chunk()`.

### Phase 5: Update `extract_and_verify_claims()` Orchestrator

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

Replace the monolithic extraction call (lines 791-813) with the chunked orchestrator:

**Current code (lines 791-813):**
```python
extraction_prompt = _build_extraction_user_prompt(extraction_report)
try:
    extraction_result: "WorkflowResult" = await execute_fn(
        prompt=extraction_prompt,
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        provider_id=provider_id,
        timeout=timeout,
        phase="claim_extraction",
        max_tokens=16384,
    )
    if not extraction_result.success or not extraction_result.content:
        logger.warning("Claim extraction LLM call failed")
        state.metadata["claim_verification_skipped"] = "extraction_failed"
        return result

    all_claims = _parse_extracted_claims(
        extraction_result.content,
        max_claims=config.deep_research_claim_verification_max_claims,
    )
except Exception as exc:
    logger.warning("Claim extraction failed: %s", exc)
    state.metadata["claim_verification_skipped"] = "extraction_failed"
    return result
```

**New code:**
```python
try:
    all_claims = await _extract_claims_chunked(
        report=extraction_report,
        execute_fn=execute_fn,
        provider_id=provider_id,
        timeout=timeout,
        max_claims=config.deep_research_claim_verification_max_claims,
        max_concurrent=config.deep_research_claim_verification_max_concurrent,
    )
except Exception as exc:
    logger.warning("Claim extraction failed: %s", exc)
    state.metadata["claim_verification_skipped"] = "extraction_failed"
    return result
```

The `extraction_failed` metadata is now set inside `_extract_claims_chunked()` only when **all** chunks fail. If some chunks succeed and others fail, the successful claims still proceed through the pipeline.

### Phase 6: Reduce Per-Call Timeout and Retries for Extraction

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

In `_extract_claims_from_chunk()`, pass reduced timeout and retries to the execute_fn:

```python
extraction_result = await execute_fn(
    prompt=chunk_prompt,
    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    provider_id=provider_id,
    timeout=timeout,           # Same per-call timeout from config (180s)
    phase="claim_extraction",
    max_tokens=4096,           # Reduced from 16384 — chunks need less output
    max_retries=1,             # Reduced from default 2 — less total wait on failure
)
```

With `max_retries=1` (2 attempts total) and a chunk that's 3-6K chars instead of 30K, each chunk either succeeds quickly (~20-40s) or fails after at most 2 × 180s + 5s = 365s. But in practice, chunks are small enough that they'll either succeed well within timeout or fail fast.

**No config changes needed** — the `deep_research_claim_verification_timeout` still applies as the per-call timeout. The overall wall time drops dramatically because chunks are smaller and parallel.

### Phase 7: Post-Extraction Citation Filter

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

Add a hard filter after extraction and before the existing `_filter_claims_for_verification()`:

```python
def _filter_uncited_claims(claims: list[ClaimVerdict]) -> list[ClaimVerdict]:
    """Drop claims that have no explicit citation references.

    The citation-anchored extraction prompt should only produce claims with
    cited_sources, but this filter acts as a safety net — if the LLM
    hallucinated a claim with no [N] reference, it gets dropped here.

    Args:
        claims: Raw extracted claims.

    Returns:
        Claims with at least one entry in cited_sources.
    """
    filtered = [c for c in claims if c.cited_sources]
    dropped = len(claims) - len(filtered)
    if dropped:
        logger.info("Dropped %d claims with no citation references", dropped)
    return filtered
```

**Called from:** `_extract_claims_chunked()` after merging and deduplication, before returning. This ensures the downstream pipeline (filter → budget → verify → correct) only ever sees claims that are anchored to a source.

**Why a separate function (not just inline):**
- Testable independently
- Clear observability (logs dropped count)
- Defense in depth — even if the prompt changes, the structural filter remains

### Phase 8: Improve Verification Source Coverage (Multi-Window Truncation)

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

The current `_keyword_proximity_truncate()` (lines 122-158) finds the **first** keyword match in the source text and returns a single 8K-char window centered on it. If the relevant passage is elsewhere in the source (e.g., the claim mentions "annual fee" and the function matches on "annual" in the intro, but the actual fee table is 15K chars deeper), verification sees irrelevant content and returns CONTRADICTED or UNSUPPORTED.

**Replace** `_keyword_proximity_truncate()` with `_multi_window_truncate()`:

```python
def _multi_window_truncate(
    text: str,
    claim_text: str,
    max_chars: int,
    max_windows: int = 3,
) -> str:
    """Truncate source text to multiple windows centered on claim-relevant keywords.

    Instead of a single window around the first keyword match, finds all keyword
    positions, scores candidate windows by keyword density, and returns the top
    N non-overlapping windows concatenated with [...] separators.

    Args:
        text: Full source text.
        claim_text: The claim being verified (used to extract keywords).
        max_chars: Total character budget across all windows.
        max_windows: Maximum number of non-overlapping windows.

    Returns:
        Concatenated windows with [...] separators, within max_chars budget.
    """
```

**Algorithm:**
1. Extract keywords from claim (same as current: words >= 4 chars, not in stopwords)
2. Find **all** positions of each keyword in the source text (not just the first)
3. Score candidate windows by keyword density (how many distinct keywords appear in each window)
4. Select top `max_windows` non-overlapping windows by score (greedy — highest score first, skip if overlaps with already-selected window)
5. Concatenate selected windows in document order with `\n[...]\n` separators
6. Allocate `max_chars` budget across windows (equal split, or proportional to score)
7. **Fallback:** If no keywords match, prefix-truncate (same as current)

**Why this works:**
- A claim citing "annual fee is $550 with 3x dining" has keywords: annual, dining. Current code finds "annual" once. New code finds every occurrence of both "annual" and "dining," picks the 2-3 most keyword-dense windows, and the verification LLM sees the fee table AND the dining multiplier section.
- Budget stays the same (8K total) — just distributed across multiple windows instead of one contiguous block.

**Update `_build_verification_user_prompt()`** (line 413) to call `_multi_window_truncate()` instead of `_keyword_proximity_truncate()`.

### Phase 9: Tighten CONTRADICTED Definition in Verification Prompt

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

**Current prompt** (`_VERIFICATION_SYSTEM_PROMPT`, lines 377-388):
```
You are a claim verification assistant. You will be given a claim from a research
report and the source material it cites. Your task is to determine whether the
source material SUPPORTS, CONTRADICTS, or provides NO EVIDENCE for the claim.

Return a JSON object with exactly these fields:
- "verdict": one of "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"
- "evidence_quote": the most relevant quote from the source material (or null if no evidence)
- "explanation": a brief explanation of why you reached this verdict

Return ONLY the JSON object, no other text.
```

**New prompt** (`_VERIFICATION_SYSTEM_PROMPT`):
```
You are a claim verification assistant. You will be given a claim from a research
report and excerpts from the source material it cites. Your task is to determine
whether the source excerpts SUPPORT, CONTRADICT, or provide NO EVIDENCE for the claim.

Verdict definitions:
- SUPPORTED: The source excerpts explicitly confirm the claim or contain information
  fully consistent with it.
- CONTRADICTED: The source excerpts explicitly state something that DIRECTLY CONFLICTS
  with the claim. The source must contain a clear counter-statement — not merely the
  absence of confirming information.
- PARTIALLY_SUPPORTED: The source excerpts confirm part of the claim but not all of it,
  or confirm it with different specifics (e.g., different numbers, dates, or scope).
- UNSUPPORTED: The source excerpts do not contain enough information to confirm or deny
  the claim. This includes cases where the topic is not mentioned at all. When in doubt
  between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED.

IMPORTANT: You are seeing excerpts, not the full source. Absence of information in these
excerpts does NOT mean the source contradicts the claim.

Return a JSON object with exactly these fields:
- "verdict": one of "SUPPORTED", "CONTRADICTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"
- "evidence_quote": the exact quote from the source that supports your verdict (REQUIRED
  for CONTRADICTED — if you cannot quote a directly conflicting statement, use UNSUPPORTED)
- "explanation": a brief explanation of your verdict

Return ONLY the JSON object, no other text.
```

**Key changes:**
- Explicit verdict definitions — not just labels, but what each means
- "explicitly state something that DIRECTLY CONFLICTS" raises the bar for CONTRADICTED
- "Absence of information does NOT mean the source contradicts" — addresses the root cause directly
- "When in doubt between CONTRADICTED and UNSUPPORTED, choose UNSUPPORTED" — bias toward safety
- "You are seeing excerpts, not the full source" — reminds the model about truncation
- evidence_quote is REQUIRED for CONTRADICTED — structural coupling with Phase 10

### Phase 10: Require Contradicting Evidence Quote (Structural Gate)

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

Add a post-parse validation in `_verify_single_claim()` (after line 501):

```python
# After parsing the verification response:
if claim.verdict == "CONTRADICTED" and not claim.evidence_quote:
    logger.info(
        "Downgrading CONTRADICTED to UNSUPPORTED for claim %r: no evidence quote provided",
        claim.claim[:80],
    )
    claim.verdict = "UNSUPPORTED"
    claim.explanation = (
        f"Originally CONTRADICTED but no contradicting quote provided. "
        f"Original explanation: {claim.explanation}"
    )
```

**Why this matters:**
- This is defense in depth — even if the prompt improvements (Phase 9) aren't perfectly followed, the structural gate catches CONTRADICTED verdicts that lack evidence
- Only CONTRADICTED triggers corrections (rewrites). SUPPORTED, UNSUPPORTED, and PARTIALLY_SUPPORTED are informational only. So this gate specifically protects against the most consequential false positive: an unjustified rewrite.
- The original explanation is preserved in the downgraded explanation for observability

### Phase 11: Tests

**File:** `tests/core/research/workflows/deep_research/test_claim_verification.py`

Add tests for chunked extraction, citation-anchored prompt, citation filter, and verification improvements:

**Extraction tests:**

1. **`test_split_report_into_sections`** — Verify section splitting:
   - Report with multiple `##` headings → correct number of chunks
   - Each chunk contains its heading text
   - Small sections (< 500 chars) are merged with next
   - Report with no headings → single chunk (full report)
   - Bibliography/sources section excluded from extraction chunks

2. **`test_extraction_prompt_is_citation_anchored`** — Verify new prompt:
   - Extraction system prompt contains "For each inline citation [N]"
   - Extraction system prompt contains "ONLY extract claims that have an explicit [N] citation"
   - Per-chunk user prompt includes the chunk content (not the full report)

3. **`test_extract_claims_from_chunk`** — Verify per-chunk extraction:
   - Successful extraction returns parsed claims
   - Failed extraction returns empty list (graceful degradation)
   - Claims tagged with correct `report_section`
   - `max_tokens=4096` passed to execute_fn
   - `max_retries=1` passed to execute_fn

4. **`test_filter_uncited_claims`** — Verify citation filter:
   - Claims with `cited_sources=[1, 2]` are kept
   - Claims with `cited_sources=[]` are dropped
   - Claims with `cited_sources=None` / missing are dropped
   - All-uncited input → empty list returned
   - Logs dropped count when claims are filtered

5. **`test_extract_claims_chunked_parallel`** — Verify parallel orchestration:
   - Multiple chunks extracted in parallel
   - Claims from all chunks merged
   - Duplicate claims deduplicated by claim text
   - Uncited claims filtered out after merge
   - Total claims capped at `max_claims`
   - Partial failure: some chunks fail, successful ones still return claims
   - All chunks fail → empty list returned
   - Single-chunk fallback works correctly

6. **`test_extract_and_verify_claims_uses_chunked`** — Integration:
   - Verify `extract_and_verify_claims()` now uses chunked extraction
   - End-to-end: multi-section report → chunked extraction → citation filter → filtering → verification

7. **`test_cancellation_safety`** — Verify `check_gather_cancellation()` works in chunked extraction

8. **Update existing extraction tests** — Existing tests that mock the old extraction prompt or expect monolithic extraction behavior need to be updated to match the new chunked, citation-anchored approach

**Verification improvement tests:**

9. **`test_multi_window_truncate`** — Verify multi-window source truncation:
   - Multiple keyword matches → multiple windows returned
   - Windows are non-overlapping
   - Windows ordered by document position
   - Total output within `max_chars` budget
   - No keyword matches → prefix-truncate fallback
   - Source text shorter than `max_chars` → returned unchanged
   - Single keyword match → single window (same as before)

10. **`test_verification_prompt_contradicted_definition`** — Verify new prompt:
    - Prompt contains "explicitly state something that DIRECTLY CONFLICTS"
    - Prompt contains "Absence of information does NOT mean the source contradicts"
    - Prompt contains "REQUIRED for CONTRADICTED"

11. **`test_contradicted_without_quote_downgraded`** — Verify structural gate:
    - CONTRADICTED + empty evidence_quote → downgraded to UNSUPPORTED
    - CONTRADICTED + null evidence_quote → downgraded to UNSUPPORTED
    - CONTRADICTED + valid evidence_quote → remains CONTRADICTED
    - SUPPORTED + empty evidence_quote → remains SUPPORTED (gate only applies to CONTRADICTED)
    - Original explanation preserved in downgraded explanation

---

## Behavioral Changes

| Aspect | Before | After |
|--------|--------|-------|
| Extraction strategy | "Find all claims in this text" | "For each [N] citation, extract the claim it supports" |
| Extraction calls | 1 call on 30K chars, 16K output tokens | N calls on 2-6K chunks, 4K output tokens each |
| Timeout per call | 180s × 3 attempts = 550s max | 300s × 2 attempts per chunk, N chunks in parallel |
| Typical wall time | 550s (all timeouts) | 30-60s (parallel small chunks) |
| Failure mode | All-or-nothing (0 claims if extraction fails) | Graceful per-chunk (claims from successful chunks preserved) |
| Output token usage | 16K max (often truncated) | ~4K × N chunks (more headroom per chunk) |
| Concurrency | 1 serial call | Bounded by `max_concurrent` (default 10) |
| Deduplication | N/A | By exact claim text match |
| False positives | Opinions, recommendations, tautologies extracted | Only citation-anchored claims; uncited claims dropped |
| Claim-source linking | LLM guesses which sources are "near" the claim | Citations are explicit in the claim's `[N]` reference |
| Source truncation | Single 8K window around first keyword match | Up to 3 non-overlapping windows by keyword density, same 8K budget |
| CONTRADICTED threshold | Ambiguous — absence of info can trigger CONTRADICTED | Explicit definition: requires direct conflicting statement + evidence quote |
| False CONTRADICTED → correction | No guard — any CONTRADICTED triggers rewrite | CONTRADICTED without evidence quote downgraded to UNSUPPORTED |

## Docs Consulted

- `dev_docs/mcp_best_practices/12-timeout-resilience.md` (timeout/retry strategy)
- `dev_docs/mcp_best_practices/15-concurrency-patterns.md` (bounded concurrency with semaphores)
- `dev_docs/mcp_best_practices/05-observability-telemetry.md` (per-chunk logging)
- `dev_docs/mcp_best_practices/07-error-semantics.md` (graceful per-chunk degradation)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Report with no `##` headings | Falls back to single-chunk (current behavior) | Explicit fallback path tested |
| Duplicate claims across chunk boundaries | Over-counting claims | Exact-text dedup before filtering |
| Section splitting drops content between headings | Lost claims | Split regex preserves all text between headings |
| Higher total LLM token usage (N calls vs 1) | Cost increase | Chunks are small, output tokens reduced per-call; net token usage similar or lower since no wasted timeout retries |
| Claim `report_section` accuracy | Chunks carry their own heading, but cross-section claims could be misattributed | Claims are extracted from within their section context; edge case is acceptable |
| Citation-anchored extraction misses uncited true claims | Some verifiable facts without `[N]` won't be extracted | Acceptable tradeoff — uncited claims have no source to verify against anyway. The purpose of claim verification is to check source alignment, not exhaustive fact-checking |
| Prompt change alters extraction behavior | Existing tests expect the old prompt format | Update test mocks and assertions to match new prompt; add tests for citation-anchored behavior |
| Multi-window truncation picks low-quality windows | Windows with keyword matches but irrelevant content | Keyword density scoring favors windows with multiple distinct keywords; fallback to prefix truncation |
| Tighter CONTRADICTED definition reduces true CONTRADICTED count | Real contradictions might get classified as UNSUPPORTED | Acceptable — a missed correction is less harmful than a false correction. The prompt still allows CONTRADICTED when evidence is clear |
| Evidence quote gate is too aggressive | Model provides a verdict explanation but forgets the quote field | The prompt explicitly marks evidence_quote as REQUIRED for CONTRADICTED; if the model can't quote a conflicting passage, the contradiction is likely spurious |
