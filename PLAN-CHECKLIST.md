# Implementation Checklist: Post-Synthesis Quality Improvements

## Item 1a: Heading Truncation Repair

### Code (`claim_verification.py`)
- [x] Add `_HEADING_TRUNCATED_RE = re.compile(r"^(#{1,6}\s+\S.*\w)-\s*$", re.MULTILINE)` near line 1056
- [x] Implement `_repair_truncated_headings(text: str) -> str`:
  - [x] Find all matches of `_HEADING_TRUNCATED_RE` with line positions
  - [x] For each match, scan forward past blank lines for first non-empty continuation line
  - [x] Skip merge if continuation starts with `#` (another heading)
  - [x] Merge: append continuation text onto heading (keeping the hyphen)
  - [x] Remove consumed continuation line from original position
- [x] Call `_repair_truncated_headings()` as final pass in `repair_heading_boundaries_global()` (after `_repair_heading_boundaries()`, not before — `_HEADING_RE` false-positives on hyphenated words like `Sign-Up`)
- [x] Note: NOT called inside `_repair_heading_boundaries()` — truncation is a synthesis artifact, not a correction artifact

### Tests (`test_claim_verification.py`)
- [x] `## Sign-\n\nUp Bonuses and Value` → `## Sign-Up Bonuses and Value`
- [x] `## Sign-\nUp Bonuses` (no blank line) → merge
- [x] `## Self-Hosted Solutions` (not truncated) → unchanged
- [x] Multiple truncated headings → all repaired
- [x] Continuation is another heading → no merge
- [x] Truncated heading at end of document → left as-is
- [x] `repair_heading_boundaries_global()` integration test with truncated heading

## Item 1b: Table-on-Heading Repair

### Code (`claim_verification.py`)
- [x] Add `_HEADING_TABLE_FUSION_RE = re.compile(r"^(#{1,6}\s+[^|\n]+?)\s*(\|(?:[^|\n]*\|){2,}.*)$", re.MULTILINE)`
- [x] Add `_HEADING_TABLE_FUSION_RE.sub(r"\1\n\n\2", ...)` pass in `_repair_heading_boundaries()`

### Tests (`test_claim_verification.py`)
- [x] `## Title| A | B | C |` → split
- [x] `## A | B Comparison` (single pipe) → unchanged
- [x] `## Title|---|---|---|` (separator fused) → split
- [x] Already separated → unchanged

## Item 1c: Citation Year-Reference Filtering

### Code (`_citation_postprocess.py`)
- [x] Update `_CITATION_RE` to `re.compile(r"\[(?!(?:19|20)\d{2}\])(\d+)\](?!\()")`
- [x] Check `claim_verification.py` for separate citation regex — apply same filter
- [x] Verify `extract_cited_numbers()` works with updated regex

### Tests (`test_citation_postprocess.py`)
- [x] `[1]`, `[99]` → matched
- [x] `[100]`, `[500]` → matched
- [x] `[2026]`, `[2025]`, `[1999]` → NOT matched
- [x] `[1899]`, `[2100]` → matched
- [x] `"in [2026] the market"` → not extracted
- [x] `renumber_citations()` with `[2026]` in body → year preserved

## Item 2: Fidelity-Gated Re-Iteration

### Config (`config/research.py`)
- [ ] Add `deep_research_fidelity_iteration_enabled: bool = True`
- [ ] Add `deep_research_fidelity_threshold: float = 0.7`
- [ ] Add parsing in `from_dict()`
- [ ] Add validation in `validate_claim_verification_config()`

### State model (`models/deep_research.py`)
- [ ] Add `fidelity_scores: list[float] = Field(default_factory=list)` to `DeepResearchState`
- [ ] Add `iteration_gap_queries: list[str] = Field(default_factory=list)` to `DeepResearchState`

### Orchestrator (`orchestration.py`)
- [ ] Update `decide_iteration()` signature to accept `fidelity_score: float | None = None`
- [ ] Implement threshold logic:
  - [ ] `fidelity_iteration_enabled=False` → complete
  - [ ] `fidelity_score is None` → complete
  - [ ] `fidelity_score >= threshold` → complete
  - [ ] `fidelity_score < threshold AND iteration < max_iterations` → iterate
  - [ ] `fidelity_score < threshold AND iteration >= max_iterations` → complete (log warning)
- [ ] Remove deprecation warning about `max_iterations > 1`

### Gap query generation (`claim_verification.py`)
- [ ] Add `build_gap_queries(verification_result: ClaimVerificationResult) -> list[str]`:
  - [ ] Group UNSUPPORTED claims by `report_section`
  - [ ] Group CONTRADICTED claims by `report_section`
  - [ ] Generate targeted research questions per section
  - [ ] Return 3-5 gap queries

### Workflow execution (`workflow_execution.py`)
- [ ] After claim verification, pass `fidelity_score` to `decide_iteration()`
- [ ] Append `fidelity_score` to `state.fidelity_scores`
- [ ] If `should_iterate=True`:
  - [ ] Call `build_gap_queries()` and store in `state.iteration_gap_queries`
  - [ ] Increment `state.iteration`
  - [ ] Reset `state.phase = DeepResearchPhase.SUPERVISION`
  - [ ] Clear `state.claim_verification = None`
  - [ ] Save state to disk
  - [ ] Continue loop (SUPERVISION runs again naturally)
- [ ] If `should_iterate=False`: proceed to citation finalize
- [ ] Wrap SUPERVISION→SYNTHESIS→CV in iteration loop (or restructure as `while`)

### Supervision prompt (`phases/supervision_prompts.py`)
- [ ] When `state.iteration > 1`, prepend gap context:
  - [ ] Previous fidelity score
  - [ ] Gap queries from `state.iteration_gap_queries`
  - [ ] Instruction to prioritize gap-filling directives

### Tests
- [ ] `test_orchestration.py`: threshold logic (5 cases above)
- [ ] `test_claim_verification.py`: `build_gap_queries()` with mixed verdicts
- [ ] `test_claim_verification.py`: empty verification → empty queries
- [ ] `test_deep_research.py` (integration): low fidelity triggers second iteration
- [ ] `test_deep_research.py`: `fidelity_scores` accumulates
- [ ] `test_deep_research.py`: sources accumulate across iterations

## Verification

- [ ] `pytest tests/ -k claim_verification` passes
- [ ] `pytest tests/ -k citation_postprocess` passes
- [ ] `pytest tests/ -k orchestration` passes
- [ ] `pytest tests/ -k deep_research` passes (full suite)
- [ ] Manual: `repair_heading_boundaries_global()` on `deepres-db449219ed17` report fixes both heading issues
