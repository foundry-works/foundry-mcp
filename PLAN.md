# Deep Research Post-Synthesis Remediation Plan

**Branch:** `alpha`
**Date:** 2026-03-03
**Triggered by:** Evaluation of session `deepres-eff86dea32c4` (credit card research, general profile)

## Problem Statement

Three post-synthesis features shipped in recent commits are either not working or partially effective:

1. **Academic providers still fire for general-profile sessions** despite Phase 1 removing them from the profile default. 49 wasted API calls (all returning 0 results), 8 garbage academic sources in the session.
2. **Bibliography includes uncited/irrelevant sources.** Phase 2 relevance filtering correctly excludes low-scoring sources from compression, but they still appear in the final report bibliography with citation numbers.
3. **Claim verification silently fails on large reports.** The extraction LLM call times out (3 attempts x 120s = 370s total) because the full 44K report is sent as a single prompt with no explicit `max_tokens` for the response.

---

## Phase 1: Fix provider leakage in gathering and supervision

### Root Cause

`gathering.py:340` and `supervision.py:1231` both read providers from `self.config.deep_research_providers` (global config, default `["tavily", "google", "semantic_scholar"]`). They completely ignore `state.metadata["active_providers"]` which the brief phase computes from the profile via `_apply_provider_hints()`.

The brief phase correctly builds an active provider list based on `profile.providers` plus any discipline-keyword hints, but this list is only stored in metadata for informational purposes — the actual search execution paths never consult it.

### Changes

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/gathering.py`

**Lines 340-344** — Replace:
```python
provider_names = getattr(
    self.config,
    "deep_research_providers",
    ["tavily", "google", "semantic_scholar"],
)
```
With:
```python
provider_names = state.metadata.get("active_providers") or getattr(
    self.config,
    "deep_research_providers",
    ["tavily"],
)
```

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

**Lines 1231-1235** — Same replacement pattern:
```python
provider_names = state.metadata.get("active_providers") or getattr(
    self.config,
    "deep_research_providers",
    ["tavily"],
)
```

### Behavioral Change

- General-profile sessions for non-academic topics will only query Tavily (or whatever providers the brief phase's adaptive hint system selected).
- Academic-profile sessions are unaffected (their profiles already list academic providers).
- The global config `deep_research_providers` remains the ultimate fallback if `active_providers` was never set (e.g., sessions started before the brief phase ran, or resumed from older state).

### Test Plan

- Unit test: mock `state.metadata["active_providers"] = ["tavily"]` and assert only Tavily providers are instantiated in gathering/supervision.
- Unit test: verify fallback to `self.config.deep_research_providers` when `active_providers` is absent from metadata (backward compat).
- Integration check: run a general-profile deep research session and confirm `search_provider_stats` shows only Tavily queries.

---

## Phase 2: Filter bibliography to cited-only sources

### Root Cause

`_citation_postprocess.py:337` calls `build_sources_section(state, format_style=format_style)` without passing `cited_only=True`. The function already supports `cited_only` and `cited_numbers` parameters (lines 132-176), and the `cited_numbers` set is already computed at line 333 — it just isn't passed through.

This means all sources with citation numbers appear in the bibliography, even if the synthesis model never referenced them in the body text. In the evaluated session, 17 of 55 sources were uncited, including all 8 irrelevant academic papers.

### Changes

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`

**Line 337** — Replace:
```python
sources_section = build_sources_section(state, format_style=format_style)
```
With:
```python
sources_section = build_sources_section(
    state,
    cited_only=True,
    cited_numbers=cited_numbers,
    format_style=format_style,
)
```

### Behavioral Change

- The `## Sources` section at the end of reports will only include sources that are actually cited `[N]` in the body text.
- Uncited sources remain in `state.sources` for provenance/export (no data loss).
- The existing `unreferenced` logging (lines 341-348) will still report how many sources were dropped.

### Test Plan

- Unit test: create a state with 5 sources (3 cited in body, 2 not), call `postprocess_citations()`, assert bibliography contains only the 3 cited sources.
- Regression test: confirm APA format mode still works with `cited_only=True`.
- Check that provenance export (`deep-research-provenance`) still lists all sources.

---

## Phase 3: Fix claim verification timeout on large reports

### Root Cause

The `extract_and_verify_claims()` function (claim_verification.py:781-787) sends the entire report as a single prompt to the extraction LLM call:
```python
extraction_result = await execute_fn(
    prompt=extraction_prompt,  # Full 44K report
    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    provider_id=provider_id,
    timeout=timeout,           # 120s
    phase="claim_extraction",
    # NOTE: no max_tokens specified
)
```

For large reports (44K+ chars), the model must output a structured JSON array of potentially 50-100+ claims. Without explicit `max_tokens`, it may hit default limits. With only 120s timeout and `max_retries=2` in the base executor, the call exhausts all 3 attempts (120+5+120+5+120 = 370s) and returns `success=False`.

### Changes

#### File: `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

**A. Add `max_tokens` to extraction call (line 781-787):**

```python
extraction_result = await execute_fn(
    prompt=extraction_prompt,
    system_prompt=_EXTRACTION_SYSTEM_PROMPT,
    provider_id=provider_id,
    timeout=timeout,
    phase="claim_extraction",
    max_tokens=16384,
)
```

**B. Add report truncation for extraction to cap input size:**

Before the extraction call, add a guard that truncates the report to a reasonable size for claim extraction (the synthesis sections and conclusion carry the most verifiable claims):

```python
# Cap report input to avoid timeout on very large reports.
_MAX_EXTRACTION_CHARS = 30_000
if len(state.report) > _MAX_EXTRACTION_CHARS:
    # Prefer the first portion (exec summary + body) over the bibliography.
    extraction_report = state.report[:_MAX_EXTRACTION_CHARS]
    logger.info(
        "Claim extraction: truncated report from %d to %d chars",
        len(state.report),
        _MAX_EXTRACTION_CHARS,
    )
else:
    extraction_report = state.report
extraction_prompt = _build_extraction_user_prompt(extraction_report)
```

**C. Increase default claim verification timeout:**

#### File: `src/foundry_mcp/config/research.py`

**Line 335** — Change default timeout from 120 to 180 seconds:
```python
deep_research_claim_verification_timeout: float = 180.0
```

Also update the TOML defaults dict (line 488):
```python
"deep_research_claim_verification_timeout": 180.0,
```

### Behavioral Change

- Extraction call gets explicit output token budget (16K tokens, enough for ~50-100 claims in JSON).
- Reports > 30K chars are truncated before extraction to avoid hitting timeout on very long reports. The bibliography (which contains no claims) is the first thing dropped.
- Default timeout raised from 120s to 180s to give the extraction call more headroom on moderately large reports.
- No change to the verification or correction sub-pipelines.

### Test Plan

- Unit test: mock `execute_fn` that returns a valid JSON claims array, verify extraction succeeds with `max_tokens=16384`.
- Unit test: verify report truncation fires for reports > 30K chars and doesn't fire for shorter reports.
- Unit test: verify the truncated report still produces parseable claims (mock extraction with truncated input).
- Integration check: re-run a general-profile session with a large report and confirm `claim_verification` object is populated.

---

## Docs Consulted

- `dev_docs/mcp_best_practices/12-timeout-resilience.md` (timeout strategy)
- `dev_docs/mcp_best_practices/05-observability-telemetry.md` (audit event coverage)
- `dev_docs/mcp_best_practices/04-validation-input-hygiene.md` (input bounds)

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 (providers) | Older sessions without `active_providers` metadata | Fallback to `self.config.deep_research_providers` preserves backward compat |
| 2 (bibliography) | Report appears to have fewer sources | Provenance/export still has all sources; body citations unchanged |
| 3 (claim verification) | Truncation drops claims in latter report sections | 30K chars covers exec summary + all analysis sections for typical reports; bibliography (no claims) is what gets dropped |
