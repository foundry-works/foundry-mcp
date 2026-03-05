# Fix Heading-Body Fusion in Deep Research Reports

## Context

Deep research reports exhibit heading-body fusions where markdown headings get concatenated with body text (e.g., `## Section Title)The first sentence...`). Root causes: (1) `_repair_heading_boundaries` regex only matches `[a-z0-9]` at heading end, missing `)`, `?`, `—`, etc.; (2) context windows cross section boundaries during corrections; (3) no heading repair after synthesis.

## Checklist

### Phase 1: Regex & Same-Line Detection
- [x] Broaden `_HEADING_RE` character class at line 1012 in `claim_verification.py` to include `)\]"'\u2019\u201d!?.;:*\u2014\u2013-`
- [x] Add `_SAMELINE_FUSION_RE` module-level constant as fallback regex
- [x] Apply `_SAMELINE_FUSION_RE.sub()` in `_repair_heading_boundaries` after `_HEADING_RE.sub()` call
- [x] Add test: `test_heading_ending_with_parenthesis`
- [x] Add test: `test_heading_ending_with_question_mark`
- [x] Add test: `test_heading_ending_with_em_dash`
- [x] Add test: `test_sameline_fusion_with_terminal_punctuation`

### Phase 2: Context Window Clamping
- [x] Add `_HEADING_BOUNDARY_RE` module-level constant
- [x] Add backward clamp in `_extract_context_window` — stop at last heading before claim
- [x] Add forward clamp in `_extract_context_window` — stop at first heading after claim
- [x] Add test: `test_window_does_not_cross_heading_forward`
- [x] Add test: `test_window_does_not_cross_heading_backward`
- [x] Add test: `test_window_keeps_own_section_heading`

### Phase 3: Global Heading Repair
- [x] Add `repair_heading_boundaries_global()` public function in `claim_verification.py`
- [x] Call after `apply_corrections` in `claim_verification.py` (~line 1214)
- [x] Call after synthesis completes in `workflow_execution.py` (line 445, before claim verification)
- [x] Add test class `TestRepairHeadingBoundariesGlobal` with 3 tests

### Verification
- [x] All existing tests pass: `python -m pytest tests/core/research/workflows/deep_research/test_claim_verification.py -x -q`
- [x] All new tests pass
- [x] Manual validation: regex matches `## What You Have (and Don't)The Amex...` correctly
