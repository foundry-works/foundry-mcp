# PLAN-TIER2: Medium-Effort Academic Research Improvements

> **Goal**: Add capabilities that produce meaningfully richer academic outputs — influence-aware ranking, structured gap analysis, and cross-study comparison.
>
> **Estimated scope**: ~600-900 LOC across 6-8 files
>
> **Dependencies**: Tier 1 items 2 (APA citations) and 3 (Semantic Scholar citation graph) are recommended prerequisites

---

## 6. Influence-Aware Source Ranking

### Problem
The supervision coverage assessment (`phases/supervision.py:3137-3260`) treats all sources equally when evaluating adequacy. A literature review should weight highly-cited seminal papers more heavily than obscure low-citation sources. The citation count and influential citation count are already stored in `ResearchSource.metadata` (from Semantic Scholar) but are ignored during coverage assessment and source selection.

### Current Coverage Assessment Logic

In `_assess_coverage_heuristic()` (supervision.py:3184-3194):
```python
source_ratios = []
for sq in completed:
    sq_sources = [s for s in state.sources if s.sub_query_id == sq.id]
    count = len(sq_sources)
    ratio = min(1.0, count / min_sources) if min_sources > 0 else 1.0
    source_ratios.append(ratio)
source_adequacy = sum(source_ratios) / len(source_ratios)
```

This counts sources without regard to their impact. 3 blog posts score the same as 3 seminal papers with 500+ citations.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`**

#### 6a. Add influence-weighted source adequacy dimension

Add a new coverage dimension `source_influence` alongside the existing `source_adequacy`, `domain_diversity`, and `query_completion_rate`:

```python
def _compute_source_influence(self, state: DeepResearchState) -> float:
    """Compute influence score based on citation metadata.

    For ACADEMIC mode:
    - Sources with citation_count >= 100: weight 3x
    - Sources with citation_count >= 20: weight 2x
    - Sources with citation_count >= 5: weight 1x
    - Sources with citation_count < 5 or unknown: weight 0.5x
    - Bonus for influential_citation_count > 0

    Returns 0.0-1.0 score indicating how many high-impact sources are present.
    """
```

#### 6b. Integrate into coverage weights

Modify the default weights when `research_mode == ACADEMIC`:

```python
# Current defaults (for GENERAL):
{"source_adequacy": 0.5, "domain_diversity": 0.2, "query_completion_rate": 0.3}

# Academic mode defaults:
{"source_adequacy": 0.3, "domain_diversity": 0.15, "query_completion_rate": 0.2, "source_influence": 0.35}
```

#### 6c. Surface influence data in supervisor brief

In the compression phase (`phases/compression.py`), include citation count in the SUPERVISOR BRIEF section so the supervisor can make influence-aware gap decisions:

```
KEY FINDINGS:
- [1] Smith et al. (2021) [cited 342 times]: Found that...
- [2] Jones (2023) [cited 12 times]: Reported that...
```

**File: `src/foundry_mcp/config/research.py`**

#### 6d. Add influence scoring config

```python
deep_research_influence_scoring_enabled: bool = True
deep_research_influence_high_citation_threshold: int = 100
deep_research_influence_medium_citation_threshold: int = 20
deep_research_influence_low_citation_threshold: int = 5
```

### Testing
- Unit test: `_compute_source_influence()` with mix of high/low citation sources
- Unit test: coverage weights change when research_mode is ACADEMIC
- Unit test: GENERAL mode is unaffected (no influence dimension)
- Verify supervisor receives citation count data in compressed findings

---

## 7. Research Landscape Metadata Output

### Problem
The synthesis report is a flat markdown document. Academic researchers often need to understand how papers relate to each other — which papers cite which, how studies cluster by methodology or theme. The sources already have citation counts, venues, years, and fields of study, but this relational data is lost in the prose report.

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 7a. Add landscape metadata model

```python
class ResearchLandscape(BaseModel):
    """Structured metadata about the research landscape.

    Attached to DeepResearchState after synthesis, providing
    machine-readable data about paper relationships and clusters.
    """
    timeline: list[dict]  # [{year: int, count: int, key_papers: [str]}]
    methodology_breakdown: dict[str, int]  # {"RCT": 5, "qualitative": 3, ...}
    venue_distribution: dict[str, int]  # {"Journal of Ed. Psych.": 4, ...}
    field_distribution: dict[str, int]  # {"Education": 8, "Psychology": 5, ...}
    top_cited_papers: list[dict]  # [{title, authors, year, citation_count, doi}]
    author_frequency: dict[str, int]  # Most prolific authors in results
    source_type_breakdown: dict[str, int]  # {"academic": 15, "web": 3}
```

#### 7b. Add landscape field to state

```python
# In DeepResearchState:
research_landscape: Optional[ResearchLandscape] = Field(
    default=None,
    description="Structured metadata about the research landscape (populated after synthesis)",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 7c. Build landscape data after synthesis

Add `_build_research_landscape()` method to `SynthesisPhaseMixin`:

```python
def _build_research_landscape(self, state: DeepResearchState) -> ResearchLandscape:
    """Extract structured landscape metadata from research sources.

    Builds from source metadata — no LLM calls needed.
    """
```

This is a pure data transformation — iterate `state.sources`, aggregate by year, venue, field, citation count, and author. No additional API calls or LLM usage.

Call this at the end of `_execute_synthesis_async()`, after report generation, and store in `state.research_landscape`.

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 7d. Include landscape in report response

When returning the deep-research-report response, include the landscape data:

```python
response["landscape"] = state.research_landscape.model_dump() if state.research_landscape else None
```

### Testing
- Unit test: `_build_research_landscape()` with diverse source metadata
- Unit test: timeline aggregation (group by year, sort ascending)
- Unit test: venue distribution from Semantic Scholar metadata
- Unit test: handles sources with missing metadata gracefully
- Verify landscape data appears in deep-research-report response

---

## 8. Explicit Research Gaps Section in Report

### Problem
The supervisor identifies research gaps internally (`ResearchGap` model in `models/sources.py:466-501`) and uses them to drive iteration. But these gaps are consumed internally and never surface in the final report. For academic literature reviews, a "Future Research Directions" section based on identified gaps is one of the most valuable outputs.

### Current Gap Tracking

Gaps are stored in `state.gaps: list[ResearchGap]` with fields:
- `description`: What's missing
- `suggested_queries`: Queries that could fill the gap
- `priority`: Importance level
- `resolved`: Whether the gap was filled during iteration
- `resolution_notes`: How it was resolved (if resolved)

Unresolved gaps represent genuine research frontiers — exactly what a "Future Directions" section should contain.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 8a. Inject gaps into synthesis prompt

In `_build_synthesis_user_prompt()`, add a section listing unresolved gaps when query_type is `literature_review`:

```python
unresolved_gaps = [g for g in state.gaps if not g.resolved]
if unresolved_gaps and query_type == "literature_review":
    parts.append("\n## Identified Research Gaps (from iterative analysis)")
    for i, gap in enumerate(unresolved_gaps, 1):
        parts.append(f"{i}. {gap.description} (priority: {gap.priority})")
    parts.append(
        "\nIncorporate these gaps into a 'Research Gaps & Future Directions' "
        "section. Frame them constructively — what specific studies or "
        "methodologies would address each gap?"
    )
```

#### 8b. Include resolved gaps for context

Also include resolved gaps with their resolution notes, so the synthesis can note "Recent work by X has begun to address this gap":

```python
resolved_gaps = [g for g in state.gaps if g.resolved]
if resolved_gaps:
    parts.append("\n## Partially Addressed Gaps")
    for gap in resolved_gaps:
        parts.append(f"- {gap.description} — Addressed by: {gap.resolution_notes}")
```

#### 8c. Enhance structure guidance

The `literature_review` structure guidance (from Tier 1, item 1c) already includes "Research Gaps & Future Directions" as a section. Ensure the synthesis system prompt explicitly instructs the LLM to use the gap data:

```
For the "Research Gaps & Future Directions" section:
- Base this on the identified research gaps provided in the input
- Distinguish between completely unexplored areas and partially addressed topics
- For each gap, suggest specific research questions or methodological approaches
- Prioritize gaps by their potential impact on the field
```

### Testing
- Unit test: unresolved gaps are injected into synthesis user prompt for literature_review type
- Unit test: resolved gaps are included with resolution notes
- Unit test: gaps are NOT injected for non-literature_review query types
- Verify synthesis report includes a substantive "Research Gaps" section

---

## 9. Cross-Study Comparison Tables

### Problem
For empirical literature reviews, researchers need to compare studies side-by-side: sample size, methodology, key findings, effect sizes. The current synthesis produces only prose. A structured comparison table would be immediately useful for the researcher's own paper.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 9a. Add comparison table generation instructions

When `query_type == "literature_review"`, append to the synthesis system prompt:

```
## Study Comparison Table

If the research involves multiple empirical studies, include a markdown comparison table
in the "Methodological Approaches" section with columns:

| Study | Year | Method | Sample | Key Finding | Effect Size | Limitations |
|-------|------|--------|--------|-------------|-------------|-------------|

Only include this table when there are 3+ empirical studies with sufficient methodological
detail. Populate from the findings provided — do not invent data. Use "Not reported" for
missing values rather than omitting the study.
```

#### 9b. Add structured table data to landscape metadata

Extend `ResearchLandscape` (from item 7) with a `study_comparison` field:

```python
class StudyComparison(BaseModel):
    """Structured comparison of empirical studies."""
    study_title: str
    authors: str
    year: Optional[int]
    methodology: Optional[str]
    sample_description: Optional[str]
    key_finding: Optional[str]
    source_id: str

study_comparisons: list[StudyComparison] = Field(default_factory=list)
```

This is populated during synthesis by parsing the LLM's table output, or alternatively by a lightweight post-synthesis extraction step.

### Testing
- Unit test: synthesis prompt includes table instructions for literature_review type
- Unit test: synthesis prompt does NOT include table instructions for other types
- Verify generated reports contain comparison tables when sufficient empirical data exists
- Test graceful handling when fewer than 3 empirical studies are found

---

## File Impact Summary

| File | Change Type | Items |
|------|-------------|-------|
| `phases/supervision.py` | Modify | 6 (influence scoring in coverage assessment) |
| `phases/synthesis.py` | Modify | 8 (gap injection), 9 (comparison tables) |
| `phases/compression.py` | Modify | 6 (citation count in supervisor brief) |
| `models/deep_research.py` | Modify | 7 (ResearchLandscape model), 9 (StudyComparison) |
| `handlers_deep_research.py` | Modify | 7 (landscape in report response) |
| `config/research.py` | Modify | 6 (influence scoring config) |
