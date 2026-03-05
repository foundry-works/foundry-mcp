# Plan: Fix Heading-Body Fusion in Deep Research Reports

## Context

Deep research reports exhibit heading-body fusions where markdown headings like `## Section Title` get concatenated with the following paragraph body (e.g., `## Your Current Portfolio: What You Already Have (and Don't)The Amex Platinum provides...`). A real session (deepres-106671273d6e on 0.18.0a11) showed 4 fusions:

- 2 from **correction LLM** stripping whitespace when rewriting context windows that bled across section boundaries
- 2 from **synthesis LLM** itself producing fused headings (no correction involved)

The existing `_repair_heading_boundaries()` function was designed to catch these but fails due to a narrow regex and limited scope.

## Root Causes

### 1. `_HEADING_RE` regex too narrow (line 1012)
```python
_HEADING_RE = re.compile(r"^(#{1,6}\s+[^\n]*?[a-z0-9])([A-Z][a-z])", re.MULTILINE)
```
The terminal `[a-z0-9]` only matches headings ending with lowercase letters or digits. Headings ending with `)`, `"`, `?`, `!`, `—` etc. are missed entirely. The real-world heading `(and Don't)` ends with `)`.

### 2. Line-based repair misses same-line fusions (lines 1037-1052)
When heading+body are fused on the same line and the regex fails (Problem 1), the line-based pass sees them as ONE line. There's no "next line" to check because the body is on the heading line itself.

### 3. Context windows cross section boundaries (lines 979-1005)
`_extract_context_window` expands to `\n\n` paragraph boundaries but doesn't stop at heading boundaries (`## `, `### `). A correction window for a claim in one section can include headings from adjacent sections. The correction LLM rewrites the entire window and may strip whitespace around those adjacent headings.

### 4. No heading repair after synthesis or after all corrections
Synthesis LLM can produce fusions directly. Heading repair only runs per-window during individual corrections, never on the full report.

## Files to Modify

1. **`src/foundry_mcp/core/research/workflows/deep_research/phases/claim_verification.py`** — regex fix, same-line detection, context window clamping, global repair function
2. **`src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`** — post-synthesis global repair call (line 445)
3. **`tests/core/research/workflows/deep_research/test_claim_verification.py`** — new test cases

## Implementation

### Phase 1: Broaden `_HEADING_RE` and Add Same-Line Fallback

**1a. Fix `_HEADING_RE` (line 1012)**

Expand terminal character class to include common heading-end punctuation:

```python
_HEADING_RE = re.compile(
    r"^(#{1,6}\s+[^\n]*?[a-z0-9)\]\"'\u2019\u201d!?.;:*\u2014\u2013\-])([A-Z][a-z])",
    re.MULTILINE,
)
```

Added characters: `)`, `]`, `"`, `'`, right-single-quote, right-double-quote, `!`, `?`, `.`, `;`, `:`, `*`, em-dash, en-dash, hyphen.

**1b. Add `_SAMELINE_FUSION_RE` (new module-level constant after line 1013)**

Second-pass safety net with greedy heading match and broader terminal set:

```python
_SAMELINE_FUSION_RE = re.compile(
    r"^(#{1,6}\s+.+?[.!?)\]\"'\u2019\u201d*\u2014\u2013:;\-])([A-Z][a-z])",
    re.MULTILINE,
)
```

**1c. Apply in `_repair_heading_boundaries` (after line 1033)**

```python
repaired = _HEADING_RE.sub(r"\1\n\n\2", corrected_text)
repaired = _SAMELINE_FUSION_RE.sub(r"\1\n\n\2", repaired)  # NEW: fallback pass
```

### Phase 2: Clamp Context Windows at Heading Boundaries

**2a. Add module-level constant:**
```python
_HEADING_BOUNDARY_RE = re.compile(r"\n(?=#{1,6}\s)")
```

**2b. Modify `_extract_context_window` (after line 1003)**

After paragraph-boundary expansion, clamp inward at heading boundaries:

- **Backward clamp:** Find the *last* heading between `window_start` and `match_start`. Keep it (the claim's own section heading) but drop anything before it.
- **Forward clamp:** Find the *first* heading between `match_end` and `window_end`. Stop there — don't include adjacent sections.

```python
# Backward: keep claim's own heading, drop earlier sections
backward_region = report[window_start:match_start]
heading_hits = list(_HEADING_BOUNDARY_RE.finditer(backward_region))
if heading_hits:
    last_hit = heading_hits[-1]
    clamped_start = window_start + last_hit.start()
    if clamped_start < match_start:
        window_start = clamped_start

# Forward: stop at next section heading
forward_region = report[match_end:window_end]
fwd_hit = _HEADING_BOUNDARY_RE.search(forward_region)
if fwd_hit:
    clamped_end = match_end + fwd_hit.start()
    if clamped_end > match_end:
        window_end = clamped_end
```

### Phase 3: Global Heading Repair

**3a. New public function in `claim_verification.py`:**

```python
def repair_heading_boundaries_global(report: str) -> str:
    """Run heading-boundary repair on the entire report."""
    if not report:
        return report
    # Pass a dummy original with a heading to force repair logic to activate.
    return _repair_heading_boundaries("# dummy heading\n\ntext", report)
```

**3b. Call after `apply_corrections` completes (~line 1214):**

```python
if corrections_applied > 0 and state.report:
    state.report = repair_heading_boundaries_global(state.report)
```

**3c. Call after synthesis, before claim verification (line 445 in `workflow_execution.py`):**

```python
# Repair heading-body fusions from synthesis.
if state.report:
    from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
        repair_heading_boundaries_global,
    )
    state.report = repair_heading_boundaries_global(state.report)
```

### Phase 4: Tests

**Add to `TestRepairHeadingBoundaries`:**
- `test_heading_ending_with_parenthesis` — the exact case from the real session: `(and Don't)The Amex...`
- `test_heading_ending_with_question_mark` — `Is It Worth It?The answer...`
- `test_heading_ending_with_em_dash` — `Overview —The program...`
- `test_sameline_fusion_with_terminal_punctuation` — verify no body text remains on heading line

**Add to `TestExtractContextWindow`:**
- `test_window_does_not_cross_heading_forward` — window stops at next `## Section`
- `test_window_does_not_cross_heading_backward` — window doesn't include prior section headings
- `test_window_keeps_own_section_heading` — claim's own `## Section` heading IS included

**New class `TestRepairHeadingBoundariesGlobal`:**
- `test_repairs_fusions_in_full_report` — multi-section report with fusions gets cleaned
- `test_noop_on_clean_report` — clean report is unchanged
- `test_empty_report` — empty string returns empty string

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Regex broadening | Low | Added chars are all legitimate heading-terminal chars. `[A-Z][a-z]` body-start requirement prevents false positives. |
| Same-line fallback | Low | Only fires on `#`-prefixed lines with sentence-terminal punctuation before `[A-Z][a-z]`. |
| Context window clamping | Medium | Reduces context for correction LLM. Mitigated by keeping claim's own section heading. |
| Global repair | Low | Pure text transform that only adds `\n\n` after headings. Triple-newline collapse prevents accumulation. |

## Verification

1. Run existing tests: `python -m pytest tests/core/research/workflows/deep_research/test_claim_verification.py -x -q`
2. All new tests pass
3. Manual regex validation: confirm `_HEADING_RE` matches `## What You Have (and Don't)The Amex...`
