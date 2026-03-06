# Plan: Deep Research Post-Synthesis Resilience

## Problem Statement

When a deep research session times out or is cancelled mid-iteration, the report is delivered without a `## Sources` bibliography section — even though the report contains inline `[N]` citations and all 74 sources with citation numbers exist in state. This was observed in session `deepres-1e60452ebc50`, which completed 2 full iterations but timed out during iteration 3's synthesis. The `finalize_citations` call (which renumbers citations and appends the bibliography) only runs on the happy path after the fidelity gate decides not to iterate.

A secondary issue: the fidelity-gated iteration loop can burn full iteration cycles when fidelity improvement is marginal (0.33 → 0.42), eventually hitting the workflow timeout without converging.

## Changes

### Phase 1: Finalize citations on cancellation/timeout

**Goal:** When a workflow is cancelled or times out after at least one completed iteration, run `finalize_citations` on the last good report before saving state.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

**In the `except asyncio.CancelledError` handler (~line 786):**

After the rollback logic determines there is a `last_completed_iteration` (line 813-829), and after rolling back to that iteration's state, call `finalize_citations` on `state.report`. This mirrors the happy-path block at lines 673-709.

Specifically, insert a new block between line 829 (`state.phase = DeepResearchPhase.SYNTHESIS`) and line 830 (the `else` branch for first-iteration-incomplete):

```python
# Finalize citations on the rolled-back report
try:
    from foundry_mcp.core.research.workflows.deep_research.phases._citation_postprocess import (
        finalize_citations,
    )
    report, finalize_meta = finalize_citations(
        state.report or "",
        state,
        query_type=state.metadata.get("_query_type"),
    )
    state.report = report

    if state.report_output_path:
        validated = _validate_report_output_path(state.report_output_path)
        validated.write_text(state.report, encoding="utf-8")

    self._write_audit_event(
        state,
        "citation_finalize_complete",
        data={**finalize_meta, "trigger": "cancellation_rollback"},
    )
except Exception as exc:
    logger.warning(
        "Citation finalize failed during cancellation for research %s: %s",
        state.id, exc,
    )
    self._write_audit_event(
        state,
        "citation_finalize_failed",
        data={"error": str(exc), "trigger": "cancellation_rollback"},
        level="warning",
    )
```

Also apply the same pattern in the `else` branch (line 843-849) where the iteration was already completed at cancellation time — `finalize_citations` should run there too if it hasn't already (check for the absence of a `## Sources` section or a `citation_finalize_complete` audit event in metadata).

**Guard against double-finalization:** Add a state metadata flag `_citations_finalized` that is set to `True` after `finalize_citations` succeeds (both on the happy path and in cancellation). Check this flag before running finalize to avoid double-appending the bibliography.

#### File: `tests/core/research/workflows/deep_research/test_workflow_execution.py`

Add tests:
- `test_cancellation_rollback_finalizes_citations` — simulate cancellation after iteration 2 with a report containing `[N]` citations. Verify the saved report contains a `## Sources` section and citations are renumbered.
- `test_cancellation_after_completed_iteration_finalizes_citations` — cancellation fires after a fully completed iteration (not mid-iteration). Verify bibliography is appended.
- `test_cancellation_first_iteration_incomplete_skips_finalize` — when there's no completed iteration to roll back to, verify no finalize attempt is made.
- `test_citation_finalize_failure_during_cancellation_is_nonfatal` — mock `finalize_citations` to raise; verify the cancellation handler still completes and saves state.

### Phase 2: Fidelity convergence early-stop

**Goal:** Stop re-iterating when fidelity improvement is too small to justify another full cycle.

#### File: `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`

**In `decide_iteration` (~line 314):**

Add a convergence check before the `should_iterate = True` branch (line 375-381). When `state.fidelity_scores` has at least 2 entries, compute the delta between the last two scores. If the delta is below a configurable minimum improvement threshold, complete instead of iterating.

```python
elif len(state.fidelity_scores) >= 2:
    delta = state.fidelity_scores[-1] - state.fidelity_scores[-2]
    if delta < fidelity_min_improvement:
        # Convergence stall — stop iterating
        rationale = (
            f"Completing: fidelity improvement {delta:.3f} < min_improvement "
            f"{fidelity_min_improvement:.3f} (scores: {state.fidelity_scores})"
        )
        next_phase = "COMPLETED"
    else:
        should_iterate = True
        ...
```

The ordering of checks becomes:
1. Fidelity iteration disabled → complete
2. No fidelity score → complete
3. Fidelity >= threshold → complete
4. Max iterations reached → complete
5. **Fidelity improvement < min_improvement** → complete (NEW)
6. Otherwise → iterate

#### File: `src/foundry_mcp/config/research.py`

Add config field:
```python
deep_research_fidelity_min_improvement: float = 0.10
```

This means if fidelity improves by less than 0.10 between iterations, stop early. The default is deliberately conservative — the 0.33 → 0.42 case (delta=0.09) would have been caught.

Update `from_dict` to parse it and `_validate_research_settings` to validate range (0.0, 1.0).

#### File: `tests/core/research/workflows/deep_research/test_supervision.py`

Add tests in the existing `TestDecideIteration` class:
- `test_fidelity_convergence_stall_completes` — two scores with delta < min_improvement → completes
- `test_fidelity_convergence_sufficient_improvement_iterates` — two scores with delta >= min_improvement → iterates
- `test_fidelity_convergence_only_one_score_iterates` — single score below threshold still iterates (not enough history)
- `test_fidelity_convergence_decision_records_scores` — verify the decision rationale includes the score history

### Phase 3: Extract `_finalize_and_save_citations` helper

**Goal:** DRY up the citation finalize block that now appears in three places (happy path, cancellation-with-rollback, cancellation-after-completion).

#### File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

Extract a private method on the workflow executor:

```python
def _finalize_and_save_citations(self, state: DeepResearchState, *, trigger: str = "normal") -> None:
    """Run finalize_citations and re-save the markdown file.

    Non-fatal: logs and audits on failure but does not raise.
    No-op if citations were already finalized (idempotency guard).
    """
```

Replace all three call sites with `self._finalize_and_save_citations(state, trigger=...)`.

### Phase 4: LLM-interpreted research confidence section

**Goal:** Append a `## Research Confidence` section to the report that contextualizes the fidelity data for the reader. An LLM interprets the raw verification stats in relation to the query type and content, so the reader understands what the numbers mean for *their specific question* rather than seeing opaque metrics.

**Why LLM-interpreted, not deterministic:** A raw "fidelity: 0.42" is misleading without context. For a credit card comparison, many UNSUPPORTED claims are expected synthesis judgments ("Citi is better for dining-heavy spenders") — no single source will say that. For a medical dosage review, the same score would be alarming. The LLM can distinguish between these and explain which unsupported claims are inferential (expected) vs. which represent genuine evidence gaps.

#### Data available at finalize time

From `state.claim_verification` (a `ClaimVerificationResult`):
- `claims_extracted`, `claims_filtered`, `claims_verified` — pipeline throughput
- `claims_supported`, `claims_partially_supported`, `claims_unsupported`, `claims_contradicted` — verdict distribution
- `corrections_applied`, `citations_remapped` — mutations made to the report
- `details: list[ClaimVerdict]` — per-claim: `claim` text, `claim_type` (negative/quantitative/comparative/positive), `verdict`, `report_section`, `explanation`, `source_resolution` tier

From `state`:
- `original_query` — the user's question
- `metadata["_query_type"]` — classified type (e.g. "comparison", "explanation", "literature_review")
- `fidelity_scores: list[float]` — convergence history across iterations
- `failed_sub_queries()` — sub-queries that returned zero sources (the hard gaps)
- `iteration` / `max_iterations` — how many research cycles ran
- `sources` — total source count

#### New file: `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py`

This module builds the confidence section through two steps:

**Step 1 — Deterministic context assembly.** Build a structured prompt input from the claim verification data. No LLM call needed for this. Includes:

- Verdict distribution summary (counts by verdict type)
- Per-section breakdown: which report sections have the most unsupported claims
- Unsupported claims grouped by `claim_type` — comparative/inferential claims flagged as "expected synthesis" vs. quantitative/factual claims flagged as "evidence gaps"
- Failed sub-queries with their query text (the hard gaps)
- Fidelity score trajectory across iterations
- Corrections applied (contradictions caught and fixed)

```python
def build_confidence_context(state: DeepResearchState) -> dict[str, Any]:
    """Assemble structured verification data for the confidence LLM call.

    Pure data transformation — no LLM, no I/O.
    Returns a dict suitable for JSON serialization into the LLM prompt.
    """
```

**Step 2 — LLM interpretation.** A single, short LLM call that receives the structured context plus the original query and query type, and produces a 150-300 word markdown section. The prompt instructs the LLM to:

- Explain what was verified and what the scores mean *for this type of question*
- Distinguish between claims that are unsupported because they're synthesis/inference (expected, not a problem) vs. claims where source evidence is genuinely missing (reader should verify independently)
- Call out specific gaps by name (e.g., "Turkish Airlines transfer data could not be verified")
- Note corrections that were applied (builds trust — "we caught and fixed X")
- Note the iteration history if multiple cycles ran
- NOT use the word "fidelity" or expose raw scores — speak in reader-friendly terms
- Be concise — this is an appendix, not another report section

```python
async def generate_confidence_section(
    state: DeepResearchState,
    llm_call_fn: Callable,
    *,
    query_type: str | None = None,
) -> str:
    """Generate an LLM-interpreted Research Confidence section.

    Uses a fast model (haiku-class) for the interpretation call.
    Returns markdown string starting with '## Research Confidence'.
    Falls back to a deterministic summary on LLM failure.
    """
```

**Fallback:** If the LLM call fails or times out, fall back to a deterministic bullet-point summary (the raw numbers without interpretation). This ensures the section is always present.

#### Integration point: `_finalize_and_save_citations` helper (Phase 3)

The Phase 3 helper becomes `_finalize_report` and gains a second step after citation finalization:

```python
def _finalize_report(self, state: DeepResearchState, *, trigger: str = "normal") -> None:
    """Finalize citations and append confidence section.

    1. Run finalize_citations (renumber + append ## Sources)
    2. Run generate_confidence_section (append ## Research Confidence)
    Non-fatal for both steps. Idempotency-guarded.
    """
```

The confidence section is appended *after* `## Sources`, making it the final section of the report.

**Note on async:** `generate_confidence_section` is async (LLM call), but the cancellation handler is in a sync-ish `except` block. Two options:
- Use `asyncio.get_event_loop().run_until_complete()` if we're outside an async context
- More likely: the cancellation handler is already inside `_execute_workflow_async`, so we can `await` directly. If the event loop is shutting down, the fallback deterministic path handles it.

#### LLM prompt design

System prompt:
```
You are writing a brief Research Confidence section for a research report.
You will receive the original research question, the query classification,
and structured verification data showing how claims in the report were
checked against gathered sources.

Your job is to help the reader calibrate their trust in the report by
explaining what was verified, what wasn't, and why — in terms that make
sense for THIS specific type of question.

Rules:
- Write 150-300 words as a ## Research Confidence section in markdown
- Do NOT use the word "fidelity" or expose raw decimal scores
- Do NOT use hedging language ("it should be noted", "it is worth mentioning")
- Distinguish between claims that are naturally inferential for this query
  type (synthesis, recommendations, comparisons) vs. claims where source
  evidence is genuinely absent (factual assertions without backing)
- Name specific gaps if sub-queries failed (e.g., "Data on X could not be
  retrieved from any source")
- Mention corrections if any were made ("One claim about X was found to
  contradict source data and was corrected")
- Note iteration count if >1 ("This report was refined over N research cycles")
- Be direct and concise — this is an appendix, not a new analysis
```

User prompt: JSON blob with the assembled context from Step 1.

#### Model selection

Use the same model resolution as compression (haiku-class) — this is a short, structured interpretation task, not deep reasoning. The `resolve_phase_provider` function can be extended with a `"confidence"` phase, or we can hardcode the compression model.

#### File: `tests/core/research/workflows/deep_research/test_confidence_section.py`

New test file:
- `test_build_confidence_context_basic` — verify context assembly from a mock state with known verdict distribution
- `test_build_confidence_context_no_verification` — verify graceful handling when `claim_verification` is None
- `test_build_confidence_context_failed_subqueries` — verify failed sub-queries appear in context
- `test_generate_confidence_section_success` — mock LLM returns valid markdown; verify section starts with `## Research Confidence`
- `test_generate_confidence_section_llm_failure_falls_back` — mock LLM raises; verify deterministic fallback is returned
- `test_generate_confidence_section_skipped_when_no_verification` — when no claim verification ran, section is either omitted or says so explicitly
- `test_confidence_section_integration` — end-to-end: build context + generate section from a realistic mock state

### Phase 5: Source-deepening verification strategy

**Goal:** When claim verification produces UNSUPPORTED verdicts, distinguish between claims that need *new* sources (widening) and claims that need *better reading* of existing sources (deepening). For deepening candidates, re-extract content from the original source URL rather than searching the web for new sources.

**Why this matters:** The current re-iteration loop always widens — `build_gap_queries` generates "Find supporting evidence for..." queries that trigger new web searches. But in many cases (especially academic literature reviews), the evidence already exists in a source we gathered — it's just not in the 8K-char keyword window that verification checked. Or the source is a blog post *about* a paper, and the actual paper (reachable via DOI or direct link) has the specific figure. Widening produces more blog posts saying the same thing; deepening finds the actual evidence.

**Observed in session `deepres-1e60452ebc50`:** All 74 sources had `raw_content` (avg 32K chars), but verification truncated each to 8K via `_multi_window_truncate`. Claims about specific numbers ("60,000 Aeroplan points to Europe") may have been in the source text but outside the keyword windows. Meanwhile, re-iteration searched for *new* sources and found 25 more — but fidelity only improved from 0.33 to 0.42.

#### Classification: which UNSUPPORTED claims are deepening candidates?

Not all UNSUPPORTED claims should trigger deepening. The classification splits into three buckets:

1. **Inferential claims** (comparative, recommendation, synthesis) — these are UNSUPPORTED because no single source states them. They're expected products of synthesis. **Action:** none (these are correctly unsupported).

2. **Factual claims with a cited source that has rich `raw_content`** — the claim cites `[N]`, source `[N]` has 30K+ chars of raw_content, but verification only checked an 8K keyword window. The evidence may be elsewhere in the same document. **Action:** re-verify with a larger window or full content.

3. **Factual claims where the source is thin** (snippet-only, blog summary, or compressed) — the original source doesn't contain enough detail. The source URL may lead to a richer page, or the source may reference a paper via DOI that should be fetched directly. **Action:** re-extract the URL or resolve the DOI.

#### New file: `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py`

**Step 1 — Classify UNSUPPORTED claims into deepening candidates.**

```python
def classify_unsupported_claims(
    verification_result: ClaimVerificationResult,
    citation_map: dict[int, ResearchSource],
) -> DeepClassification:
    """Classify UNSUPPORTED claims into deepening vs. widening vs. inferential.

    Returns:
        DeepClassification with three lists:
        - inferential: claims that are synthesis/comparative (no action needed)
        - deepen_window: claims where source has rich raw_content (re-verify with larger window)
        - deepen_extract: claims where source is thin (re-extract URL or resolve DOI)
        - widen: claims that genuinely need new sources
    """
```

Classification logic:
- `claim_type in ("comparative", "positive")` AND claim text contains recommendation language → **inferential**
- `claim_type in ("quantitative", "negative")` AND cited source has `raw_content` with `len > 16000` → **deepen_window** (evidence likely exists but was outside the 8K keyword window)
- `claim_type in ("quantitative", "negative")` AND cited source has `raw_content` with `len < 4000` or only `snippet` → **deepen_extract** (source is too thin, needs re-fetch)
- Remaining → **widen** (needs new sources via web search)

**Step 2 — Re-verify with expanded windows (for `deepen_window` claims).**

```python
async def reverify_with_expanded_window(
    claims: list[ClaimVerdict],
    citation_map: dict[int, ResearchSource],
    llm_call_fn: Callable,
    *,
    max_chars: int = 24000,  # 3x the normal 8K window
) -> list[ClaimVerdict]:
    """Re-verify claims using a larger content window from the same source.

    Uses _multi_window_truncate with a 3x budget, or passes full raw_content
    if it fits. Does NOT fetch anything new — purely re-reads existing content.
    """
```

This is cheap — it's an LLM call with more context from content we already have in memory. No network I/O.

**Step 3 — Re-extract thin sources (for `deepen_extract` claims).**

```python
async def deepen_thin_sources(
    claims: list[ClaimVerdict],
    citation_map: dict[int, ResearchSource],
    state: DeepResearchState,
    extract_provider: TavilyExtractProvider,
) -> int:
    """Re-extract content for sources that were too thin for verification.

    For each thin source:
    1. If source has DOI in metadata → attempt Semantic Scholar full-text fetch
    2. If source has URL → re-extract via Tavily Extract with follow_links=True
    3. Update source.raw_content with the richer content

    Returns number of sources successfully deepened.
    """
```

This involves network I/O but targets specific URLs we already know, not broad web searches.

**Step 4 — Integrate into `build_gap_queries` and re-iteration decision.**

The current `build_gap_queries` in `claim_verification.py` only produces widening queries. Modify it to:

1. Run `classify_unsupported_claims` first
2. For `deepen_window` claims: queue them for expanded-window re-verification (runs immediately, no re-iteration needed)
3. For `deepen_extract` claims: queue the source URLs for re-extraction (runs before the next supervision round)
4. For `widen` claims: generate the existing "Find supporting evidence..." gap queries
5. For `inferential` claims: exclude from gap queries entirely (they inflate the perceived gap without adding value)

This changes the re-iteration flow:

```
Current:
  synthesis → claim verification → ALL unsupported → gap queries → web search → ...

Proposed:
  synthesis → claim verification → classify unsupported →
    inferential: skip (expected)
    deepen_window: re-verify with 3x window (immediate, no re-iteration)
    deepen_extract: re-fetch source URL (before next supervision round)
    widen: gap queries → web search (existing flow)
```

The `deepen_window` step runs inline after claim verification, before the fidelity score is computed. This means the fidelity score already reflects the expanded-window results, and fewer claims remain UNSUPPORTED by the time the iteration decision is made. The 0.33 → 0.42 session might have reached 0.55+ just from expanded windows, reducing or eliminating unnecessary re-iteration cycles.

#### File changes

| File | Change |
|------|--------|
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py` | **New** — claim classification, expanded-window re-verification, source re-extraction |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py` | Modify `build_gap_queries` to classify claims and exclude inferential; add deepening step before fidelity scoring |
| `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` | Wire deepening step between claim verification and fidelity decision |
| `src/foundry_mcp/core/research/workflows/deep_research/_constants.py` | Add `VERIFICATION_SOURCE_DEEPEN_MAX_CHARS = 24000` |

#### Tests

New file: `tests/core/research/workflows/deep_research/test_source_deepening.py`
- `test_classify_inferential_claims` — comparative claims classified as inferential
- `test_classify_deepen_window` — quantitative claim with rich raw_content classified as deepen_window
- `test_classify_deepen_extract` — quantitative claim with thin source classified as deepen_extract
- `test_classify_widen` — factual claim with no existing source classified as widen
- `test_reverify_expanded_window_upgrades_verdict` — claim goes from UNSUPPORTED to SUPPORTED with larger window
- `test_reverify_expanded_window_unchanged` — claim stays UNSUPPORTED with larger window (evidence genuinely absent)
- `test_deepen_thin_sources_with_doi` — source with DOI triggers Semantic Scholar fetch
- `test_deepen_thin_sources_with_url` — source with URL triggers Tavily Extract re-fetch
- `test_build_gap_queries_excludes_inferential` — inferential claims not included in gap queries
- `test_fidelity_improves_after_deepening` — end-to-end: deepening step improves fidelity score before iteration decision

## Files Modified

| File | Change |
|------|--------|
| `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` | Add citation finalize in cancellation handler; extract `_finalize_report` helper; wire deepening step |
| `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py` | Add convergence stall check in `decide_iteration` |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_confidence_section.py` | **New** — context assembly + LLM-interpreted confidence section |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/_source_deepening.py` | **New** — claim classification, expanded-window re-verification, source re-extraction |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py` | Modify `build_gap_queries` to classify and exclude inferential claims |
| `src/foundry_mcp/core/research/workflows/deep_research/_constants.py` | Add `VERIFICATION_SOURCE_DEEPEN_MAX_CHARS` |
| `src/foundry_mcp/config/research.py` | Add `deep_research_fidelity_min_improvement` config field |
| `tests/core/research/workflows/deep_research/test_workflow_execution.py` | 4 new cancellation+citation tests |
| `tests/core/research/workflows/deep_research/test_supervision.py` | 4 new convergence stall tests |
| `tests/core/research/workflows/deep_research/test_confidence_section.py` | **New** — 7 tests for confidence section |
| `tests/core/research/workflows/deep_research/test_source_deepening.py` | **New** — 10 tests for source deepening |

## Out of Scope

- Changing fidelity threshold default (0.70) — that's a tuning decision, not a code fix
- Retry logic for sub-queries that return zero sources (Tavily-specific)
- Adding `finalize_citations` to the `mark_failed` / `mark_interrupted` paths (different failure modes)
- Making the confidence section user-configurable (on/off toggle) — add later if needed
- Full PDF paper extraction (beyond what Tavily Extract provides) — future enhancement
- Semantic Scholar full-text API access (requires partnership tier) — deepen_extract falls back gracefully
