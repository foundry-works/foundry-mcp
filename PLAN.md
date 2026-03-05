# PLAN: Deep Research Post-Synthesis Quality Improvements

## Context

Analysis of deep research session `deepres-db449219ed17` (credit card comparison, fidelity=0.43) revealed four quality issues in the post-synthesis pipeline.

### Observed Issues

1. **Broken heading `## Sign-`** — Synthesis LLM split "Sign-Up Bonuses" across a heading boundary, producing `## Sign-\n\nUp Bonuses and...`. Current repair code only handles same-line fusions (heading+body concatenated on one line), not mid-word heading truncations across lines.

2. **Table merged onto heading** — `## Annual Fee vs. Value Proposition| Card | Annual Fee |...`. A markdown table row got concatenated directly onto the heading line. The repair regexes require `[A-Z][a-z]` after the heading (body text start), so pipe-delimited table content is invisible.

3. **`[2026]` parsed as citation** — Year references like `[2026]` matched by `_CITATION_RE`. While `_compute_max_citation()` filters these during finalization, they appear as noise in intermediate stages (claim extraction, report-level citation counts).

4. **Low fidelity (0.43) with no re-iteration** — `max_iterations=3` was set but `decide_iteration()` always returns `should_iterate=False`. The fidelity score is computed but never triggers additional research. The state already tracks `iteration` and `max_iterations` — the wiring just needs to be connected.

### Non-issue (confirmed working)

- **Correction generation for CONTRADICTED claims** — All 3 corrections were applied successfully. The serialized field is `corrected_text` (not `correction`). No fix needed.

---

## Implementation Plan

### 1. Heading Truncation Repair

**Problem:** `## Sign-\n\nUp Bonuses and Business Class Valuation` — heading ends mid-word with a hyphen, continuation on later line after blank lines.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

**Changes:**
- New regex `_HEADING_TRUNCATED_RE = re.compile(r"^(#{1,6}\s+\S.*\w)-\s*$", re.MULTILINE)` — matches heading lines ending with `word-` at EOL
- New function `_repair_truncated_headings(text: str) -> str`:
  - Find headings matching the truncation pattern
  - Skip blank lines to find the continuation text (first non-empty line)
  - Merge continuation onto heading: `## Sign-\n\nUp Bonuses` → `## Sign-Up Bonuses`
  - Continuation line is removed from its original position
- Call as first pass in `_repair_heading_boundaries()` (before existing fusion repairs)
- Also called from `repair_heading_boundaries_global()`

**Edge cases:**
- Heading with legitimate trailing hyphen that is a complete title (e.g., `## X-Ray Analysis`) — regex requires `\w` before `-`, so `X-` would match, but `X-Ray` on one line wouldn't (no line break). Only triggers when `-` is at EOL.
- Multiple truncated headings — function loops over all matches
- Continuation that is itself a heading — skip merge if next non-blank line starts with `#`

**Tests** (`test_claim_verification.py`):
- `## Sign-\n\nUp Bonuses and Value` → `## Sign-Up Bonuses and Value`
- `## Sign-\nUp Bonuses` (no blank line gap) → merge
- `## Self-Hosted Solutions` (complete heading, no truncation) → unchanged
- Multiple truncated headings in one document → all repaired
- Continuation is another heading → no merge
- Truncated heading at end of document → left as-is

### 2. Table-on-Heading Repair

**Problem:** `## Annual Fee vs. Value Proposition| Card | Annual Fee | ...` — pipe-delimited table fused onto heading.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`

**Changes:**
- New regex `_HEADING_TABLE_FUSION_RE = re.compile(r"^(#{1,6}\s+[^|\n]+?)\s*(\|(?:[^|\n]*\|){2,}.*)$", re.MULTILINE)`:
  - Matches heading text (no pipes) followed by a pipe-delimited segment with 3+ cells
  - Requires `{2,}` pipe groups to avoid false positives on headings like `## A | B Comparison`
- Replacement: `\1\n\n\2` — split heading from table row, insert blank line
- Add as new pass in `_repair_heading_boundaries()` after existing fusion passes

**Tests** (`test_claim_verification.py`):
- `## Title| A | B | C |` → `## Title\n\n| A | B | C |`
- `## A | B Comparison` (single pipe, not table) → unchanged
- `## Title|---|---|---|` (separator row fused) → split
- `## Title\n\n| A | B |` (already separated) → unchanged

### 3. Citation Year-Reference Filtering

**Problem:** `_CITATION_RE` matches `[2026]` as a citation reference.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`

**Changes:**
- Update `_CITATION_RE` from:
  ```python
  _CITATION_RE = re.compile(r"\[(\d+)\](?!\()")
  ```
  to:
  ```python
  _CITATION_RE = re.compile(r"\[(?!(?:19|20)\d{2}\])(\d+)\](?!\()")
  ```
  Negative lookahead `(?!(?:19|20)\d{2}\])` skips `[1900]`–`[2099]`.

- Check `claim_verification.py` for any separate citation regex — apply same year filter if found.

**Tests** (`test_citation_postprocess.py`):
- `[1]`, `[2]`, `[99]` → matched
- `[100]`, `[500]` → matched (valid high citation numbers, not years)
- `[2026]`, `[2025]`, `[1999]` → NOT matched
- `[1899]`, `[2100]` → matched (outside year range)
- `"published in [2026] by researchers"` → `[2026]` not extracted by `extract_cited_numbers()`
- End-to-end `renumber_citations()` with `[2026]` in body text → year preserved, not renumbered

### 4. Fidelity-Gated Re-Iteration

**Problem:** Fidelity score is 0.43 ("low") but `decide_iteration()` always returns `should_iterate=False`. The state already has `iteration=1` and `max_iterations=3` — the mechanism exists but isn't wired up.

**Architecture:** The current flow is linear:
```
SUPERVISION → SYNTHESIS → claim_verification → citation_finalize → mark_completed
```

After claim verification, if fidelity is below threshold and iterations remain, loop back:
```
SUPERVISION → SYNTHESIS → claim_verification →
  if fidelity < threshold && iteration < max_iterations:
    → build gap queries from UNSUPPORTED/CONTRADICTED claims
    → increment iteration
    → SUPERVISION (with gap context) → SYNTHESIS → claim_verification → ...
  else:
    → citation_finalize → mark_completed
```

**Files:**

**a) Config** (`src/foundry_mcp/config/research.py`):
- Add `deep_research_fidelity_iteration_enabled: bool = True`
- Add `deep_research_fidelity_threshold: float = 0.7`
- Add parsing in `from_dict()` and validation in `validate_claim_verification_config()`

**b) Orchestrator** (`src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`):
- Update `decide_iteration()` to accept optional `fidelity_score: float | None`:
  - If `fidelity_iteration_enabled` is False → always complete (current behavior)
  - If `fidelity_score is None` (no claim verification) → complete
  - If `fidelity_score >= threshold` → complete
  - If `fidelity_score < threshold AND state.iteration < state.max_iterations` → iterate
  - If `fidelity_score < threshold AND state.iteration >= state.max_iterations` → complete (log warning)
- Remove the deprecation warning about `max_iterations > 1`

**c) Workflow execution** (`src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`):
- After claim verification (line ~546), check the iteration decision:
  - If `should_iterate=True`:
    - Build gap-focused directives from UNSUPPORTED/CONTRADICTED claims (extract topics/questions that need better sources)
    - Append to `state.directives` so supervision knows what to focus on
    - Increment `state.iteration`
    - Reset `state.phase = DeepResearchPhase.SUPERVISION`
    - Clear `state.claim_verification` (will be re-run after next synthesis)
    - **Don't** clear sources/findings — they accumulate across iterations
    - Loop continues naturally (the outer `if state.phase == SUPERVISION` block runs again)
  - If `should_iterate=False`: proceed to citation finalize as today
- Move citation finalize + mark_completed outside the iteration loop or guard with the decision
- Wrap the SUPERVISION → SYNTHESIS → CV block in a `while state.iteration <= state.max_iterations` loop

**d) State model** (`src/foundry_mcp/core/research/models/deep_research.py`):
- Add `fidelity_scores: list[float] = Field(default_factory=list)` to `DeepResearchState` to track convergence across iterations
- Add `iteration_gap_queries: list[str] = Field(default_factory=list)` for gap context passed to supervision

**e) Gap query generation** (new helper in `claim_verification.py` or `workflow_execution.py`):
- `build_gap_queries(verification_result: ClaimVerificationResult) -> list[str]`:
  - Group UNSUPPORTED claims by `report_section`
  - For each section with unsupported claims, generate a targeted research question
  - Group CONTRADICTED claims similarly
  - Return list of gap queries (typically 3-5)

**f) Supervision prompt update** (`phases/supervision_prompts.py`):
- When `state.iteration > 1`, prepend gap context to supervision prompt:
  - "This is iteration {N}. Previous synthesis had fidelity {score}. Focus on these gaps: ..."
  - Include the gap queries
  - Instruct supervisor to prioritize directives that address the gaps

**Tests:**
- `test_orchestration.py`:
  - Fidelity below threshold → `should_iterate=True`
  - Fidelity above threshold → `should_iterate=False`
  - Fidelity below threshold but max iterations reached → `should_iterate=False`
  - `fidelity_iteration_enabled=False` → always complete
  - Fidelity is None → complete
- `test_claim_verification.py` (or new file):
  - `build_gap_queries()` extracts meaningful queries from unsupported claims
  - Empty verification result → empty gap queries
- `test_deep_research.py` (integration):
  - Mock a low-fidelity first iteration → verify second iteration runs
  - Verify `fidelity_scores` accumulates across iterations
  - Verify sources accumulate (not cleared between iterations)

---

## Sequencing

| Order | Item | Risk | Scope |
|-------|------|------|-------|
| 1a | Heading truncation repair | Low | ~40 lines code + ~60 lines tests |
| 1b | Table-on-heading repair | Low | ~15 lines code + ~40 lines tests |
| 1c | Citation year filter | Low | ~5 lines code + ~30 lines tests |
| 2 | Fidelity-gated iteration | Medium | ~150 lines code + ~100 lines tests |

Items 1a/1b/1c are independent — implement in parallel. Item 2 depends on understanding the full pipeline (now documented above) but is independent of the heading/citation fixes.

## Risk Assessment

| Item | Risk | Mitigation |
|------|------|------------|
| Heading truncation repair | Low — additive pass | Careful regex to avoid false merges; skip if continuation is a heading |
| Table-on-heading repair | Low — additive pass | Require 3+ pipe cells to distinguish table from casual pipe |
| Citation year filter | Low — tightens existing regex | Negative lookahead is precise; only affects 1900–2099 |
| Fidelity-gated iteration | Medium — reactivates iteration loop, cost implications (more LLM calls per session) | Feature flag `fidelity_iteration_enabled` (default True); `max_iterations` cap (default 3); sources accumulate so re-iteration adds incrementally, not from scratch |
