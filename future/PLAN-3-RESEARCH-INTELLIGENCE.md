# PLAN-3: Research Intelligence — Ranking, Landscape & Export

> **Goal**: Add capabilities that produce meaningfully richer academic outputs — influence-aware source ranking, structured landscape metadata, explicit research gaps sections, cross-study comparison tables, and reference export (BibTeX/RIS).
>
> **Estimated scope**: ~900-1300 LOC across 8-10 files
>
> **Dependencies**: PLAN-1 items 1 (profiles), 3 (literature_review type), 4 (APA citations); PLAN-2 item 1 (OpenAlex for richer metadata)

---

## Design Principles

1. **Use data already collected.** Most of these features transform metadata that the Semantic Scholar and OpenAlex providers already store — citation counts, venues, years, fields of study, authors. The cost is computation, not API calls.
2. **Academic features are profile-gated.** Influence scoring, landscape metadata, and comparison tables only activate when the profile enables them. General-mode behavior is completely unchanged.
3. **Structured data complements prose.** The landscape metadata, study comparisons, and export formats are part of the structured output (PLAN-1 item 6), not replacements for the report.

---

## 1. Influence-Aware Source Ranking

### Problem

The supervision coverage assessment (`phases/supervision.py`) treats all sources equally. A literature review should weight highly-cited seminal papers more heavily than obscure low-citation sources. The citation count and influential citation count are already stored in `ResearchSource.metadata` (from Semantic Scholar / OpenAlex) but are ignored during coverage assessment.

### Current Coverage Assessment

In `_assess_coverage_heuristic()`:
```python
source_ratios = []
for sq in completed:
    sq_sources = [s for s in state.sources if s.sub_query_id == sq.id]
    count = len(sq_sources)
    ratio = min(1.0, count / min_sources) if min_sources > 0 else 1.0
    source_ratios.append(ratio)
source_adequacy = sum(source_ratios) / len(source_ratios)
```

This counts sources without regard to impact. 3 blog posts score the same as 3 seminal papers with 500+ citations.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`**

#### 1a. Add influence-weighted source adequacy dimension

```python
def _compute_source_influence(self, state: DeepResearchState) -> float:
    """Compute influence score based on citation metadata.

    For profiles with source_quality_mode == ACADEMIC:
    - Sources with citation_count >= 100: weight 3x
    - Sources with citation_count >= 20: weight 2x
    - Sources with citation_count >= 5: weight 1x
    - Sources with citation_count < 5 or unknown: weight 0.5x
    - Bonus for influential_citation_count > 0

    Returns 0.0-1.0 score indicating how many high-impact sources are present.
    For GENERAL/TECHNICAL profiles, returns 1.0 (neutral, no impact on coverage).
    """
```

#### 1b. Integrate into coverage weights

When profile has `source_quality_mode == ACADEMIC`:
```python
# ACADEMIC weights (influence matters):
{"source_adequacy": 0.3, "domain_diversity": 0.15, "query_completion_rate": 0.2, "source_influence": 0.35}

# Default weights (unchanged):
{"source_adequacy": 0.5, "domain_diversity": 0.2, "query_completion_rate": 0.3}
```

#### 1c. Surface influence data in supervisor brief

In the compression phase, include citation count in the SUPERVISOR BRIEF so the supervisor can make influence-aware gap decisions:
```
KEY FINDINGS:
- [1] Smith et al. (2021) [cited 342 times]: Found that...
- [2] Jones (2023) [cited 12 times]: Reported that...
```

Only include citation count when available (omit for web sources).

**File: `src/foundry_mcp/config/research.py`**

#### 1d. Add influence scoring config

```python
deep_research_influence_high_citation_threshold: int = 100
deep_research_influence_medium_citation_threshold: int = 20
deep_research_influence_low_citation_threshold: int = 5
```

These provide sensible defaults that can be tuned per-domain (e.g., in humanities where citation counts are lower, thresholds should be lower).

### Testing

- Unit test: `_compute_source_influence()` with high-citation sources → high score
- Unit test: `_compute_source_influence()` with all unknown citations → low score
- Unit test: `_compute_source_influence()` with mixed citations → proportional score
- Unit test: ACADEMIC coverage weights include `source_influence`
- Unit test: GENERAL coverage weights exclude `source_influence`
- Unit test: supervisor brief includes citation counts for academic sources
- Verify no regression in coverage assessment for general-mode queries

---

## 2. Research Landscape Metadata

### Problem

The synthesis report is a flat markdown document. Academic researchers often need to understand how papers relate to each other — which venues dominate, how research output has changed over time, which authors are most prolific. The source metadata (years, venues, fields, authors, citation counts) supports all of this but the relational data is lost in the prose report.

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 2a. Add landscape metadata model

```python
class ResearchLandscape(BaseModel):
    """Structured metadata about the research landscape.

    Built from source metadata after synthesis — pure data transformation,
    no additional LLM or API calls. Included in structured output for
    downstream consumption by visualization or analysis tools.
    """
    timeline: list[dict] = Field(
        default_factory=list,
        description="[{year: int, count: int, key_papers: [{title, citation_count}]}]",
    )
    methodology_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description='{"RCT": 5, "qualitative": 3, "meta_analysis": 2, ...}',
    )
    venue_distribution: dict[str, int] = Field(
        default_factory=dict,
        description='{"Journal of Educational Psychology": 4, ...}',
    )
    field_distribution: dict[str, int] = Field(
        default_factory=dict,
        description='{"Education": 8, "Psychology": 5, ...}',
    )
    top_cited_papers: list[dict] = Field(
        default_factory=list,
        description="[{title, authors, year, citation_count, doi}] sorted by citation_count desc",
    )
    author_frequency: dict[str, int] = Field(
        default_factory=dict,
        description="Most prolific authors in results, by count",
    )
    source_type_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description='{"academic": 15, "web": 3}',
    )
```

#### 2b. Add landscape field to state

```python
# In DeepResearchState:
research_landscape: Optional[ResearchLandscape] = Field(
    default=None,
    description="Structured metadata about the research landscape (populated after synthesis)",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 2c. Build landscape data after synthesis

Add `_build_research_landscape()` method to `SynthesisPhaseMixin`:

```python
def _build_research_landscape(self, state: DeepResearchState) -> ResearchLandscape:
    """Extract structured landscape metadata from research sources.

    Pure data transformation — iterates state.sources, aggregates by year,
    venue, field, citation count, and author. No additional API or LLM calls.
    """
```

Call at the end of `_execute_synthesis_async()`, after report generation.

#### 2d. Include in structured output

The landscape data feeds into `StructuredResearchOutput` (PLAN-1 item 6):

```python
structured_output.landscape = state.research_landscape.model_dump() if state.research_landscape else None
```

### Testing

- Unit test: landscape with 10 academic sources → complete metadata
- Unit test: landscape with 0 academic sources → empty/default values
- Unit test: landscape with mixed academic + web sources → correct counts
- Unit test: timeline sorted ascending by year
- Unit test: top_cited_papers sorted descending by citation count
- Unit test: author_frequency counts correctly
- Unit test: JSON serialization roundtrip
- Verify landscape appears in structured output

---

## 3. Explicit Research Gaps Section

### Problem

The supervisor identifies research gaps internally (`ResearchGap` model) and uses them to drive iteration. But unresolved gaps are consumed internally and never surface in the final report. For academic literature reviews, a "Future Research Directions" section based on identified gaps is one of the most valuable outputs.

### Current Gap Tracking

Gaps are stored in `state.gaps: list[ResearchGap]` with fields:
- `description`: What's missing
- `suggested_queries`: Queries that could fill the gap
- `priority`: Importance level
- `resolved`: Whether the gap was filled during iteration
- `resolution_notes`: How it was resolved (if resolved)

Unresolved gaps represent genuine research frontiers.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 3a. Inject gaps into synthesis prompt

In `_build_synthesis_user_prompt()`, when `query_type == "literature_review"`:

```python
unresolved_gaps = [g for g in state.gaps if not g.resolved]
if unresolved_gaps:
    parts.append("\n## Identified Research Gaps (from iterative analysis)")
    for i, gap in enumerate(unresolved_gaps, 1):
        parts.append(f"{i}. {gap.description} (priority: {gap.priority})")
    parts.append(
        "\nIncorporate these gaps into a 'Research Gaps & Future Directions' "
        "section. Frame them constructively — what specific studies or "
        "methodologies would address each gap?"
    )
```

#### 3b. Include resolved gaps for context

Also include resolved gaps with their resolution notes, so the synthesis can note "Recent work by X has begun to address this gap":

```python
resolved_gaps = [g for g in state.gaps if g.resolved]
if resolved_gaps:
    parts.append("\n## Partially Addressed Gaps")
    for gap in resolved_gaps:
        parts.append(f"- {gap.description} — Addressed by: {gap.resolution_notes}")
```

#### 3c. Add explicit synthesis instructions for gaps section

```
For the "Research Gaps & Future Directions" section:
- Base this on the identified research gaps provided in the input
- Distinguish between completely unexplored areas and partially addressed topics
- For each gap, suggest specific research questions or methodological approaches
- Prioritize gaps by their potential impact on the field
```

#### 3d. Include gaps in structured output

Unresolved gaps are already part of `StructuredResearchOutput.gaps` (PLAN-1 item 6). Ensure resolved gaps are also included with their resolution status.

### Testing

- Unit test: unresolved gaps are injected into synthesis prompt for literature_review
- Unit test: resolved gaps are included with resolution notes
- Unit test: gaps are NOT injected for non-literature_review query types
- Unit test: empty gaps list → no gap section in prompt
- Verify synthesis report includes "Research Gaps" section

---

## 4. Cross-Study Comparison Tables

### Problem

For empirical literature reviews, researchers need to compare studies side-by-side: sample size, methodology, key findings, effect sizes. The current synthesis produces only prose. A structured comparison table would be immediately useful.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 4a. Add comparison table generation instructions

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

#### 4b. Add structured comparison data to landscape

Extend `ResearchLandscape` (item 2) with study comparisons:

```python
class StudyComparison(BaseModel):
    """Structured comparison of an empirical study."""
    study_title: str
    authors: str
    year: Optional[int] = None
    methodology: Optional[str] = None
    sample_description: Optional[str] = None
    key_finding: Optional[str] = None
    source_id: str

# In ResearchLandscape:
study_comparisons: list[StudyComparison] = Field(default_factory=list)
```

This is populated during synthesis by a lightweight post-processing step that parses the LLM's markdown table output. If no table was generated, the list remains empty.

### Testing

- Unit test: synthesis prompt includes table instructions for literature_review
- Unit test: synthesis prompt excludes table instructions for other types
- Verify generated reports contain comparison tables when sufficient data exists
- Test graceful handling when fewer than 3 empirical studies are found

---

## 5. BibTeX & RIS Export

### Problem

Academic researchers need to import discovered papers into reference managers (Zotero, Mendeley, EndNote). The DOIs, authors, years, and venues are already in `ResearchSource.metadata` but there's no export capability.

### Changes

**File: `src/foundry_mcp/core/research/export/__init__.py`** (NEW)
**File: `src/foundry_mcp/core/research/export/bibtex.py`** (NEW)

#### 5a. BibTeX generator

```python
def sources_to_bibtex(sources: list[ResearchSource]) -> str:
    """Convert research sources to BibTeX format.

    Generates one @article/@inproceedings/@misc entry per source.
    Uses DOI, authors, year, venue, title from source metadata.
    Generates stable citation keys from first author + year + title words.
    """

def source_to_bibtex_entry(source: ResearchSource, citation_key: str) -> str:
    """Convert a single ResearchSource to a BibTeX entry string."""
```

Entry type selection:
- `metadata.venue` contains "conference" or "proceedings" → `@inproceedings`
- `metadata.venue` is present → `@article`
- Otherwise → `@misc`

Special character escaping for BibTeX: `&`, `%`, `#`, `_`, `{`, `}`.

**File: `src/foundry_mcp/core/research/export/ris.py`** (NEW)

#### 5b. RIS generator

```python
def sources_to_ris(sources: list[ResearchSource]) -> str:
    """Convert research sources to RIS format.

    Generates TY/AU/TI/JO/PY/DO/UR/ER blocks per source.
    Compatible with Zotero, Mendeley, EndNote.
    """
```

Entry type selection:
- Academic with venue → `TY  - JOUR`
- Conference → `TY  - CONF`
- Web/other → `TY  - ELEC`

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 5c. Add export action

```python
def _handle_deep_research_export(
    research_id: str,
    format: str = "bibtex",  # "bibtex" | "ris"
    academic_only: bool = True,
) -> dict:
    """Export bibliography from completed research session."""
```

Loads the persisted `DeepResearchState`, filters sources (academic-only by default), generates the requested format.

Wire up `"deep-research-export"` action in the research action router.

#### 5d. Include exports in structured output

After synthesis, generate both BibTeX and RIS from `state.sources` and include in the structured output:

```python
structured_output.exports = {
    "bibtex": sources_to_bibtex(academic_sources),
    "ris": sources_to_ris(academic_sources),
}
```

### Testing

- Unit test: BibTeX with full metadata → valid entry
- Unit test: BibTeX with minimal metadata → valid @misc entry
- Unit test: BibTeX special character escaping
- Unit test: citation key uniqueness and stability
- Unit test: RIS with full metadata → valid block
- Unit test: RIS with minimal metadata → valid ELEC entry
- Unit test: export action handler with completed session
- Unit test: export action handler with non-existent session → error

---

## File Impact Summary

| File | Change Type | Items |
|------|-------------|-------|
| `phases/supervision.py` | Modify | 1 (influence scoring, supervisor brief enrichment) |
| `phases/synthesis.py` | Modify | 2 (landscape builder), 3 (gap injection), 4 (comparison tables) |
| `phases/compression.py` | Modify | 1 (citation count in supervisor brief) |
| `models/deep_research.py` | Modify | 2 (ResearchLandscape, StudyComparison) |
| `config/research.py` | Modify | 1 (influence thresholds) |
| `export/__init__.py` | **New** | 5 |
| `export/bibtex.py` | **New** | 5 |
| `export/ris.py` | **New** | 5 |
| `handlers_deep_research.py` | Modify | 5 (export action) |
| `research.py` (action router) | Modify | 5 (wire up export action) |

## Dependency Graph

```
[1. Influence-Aware Ranking] (independent — uses existing metadata)

[2. Research Landscape] (independent — pure data transformation)
       │
       └──▶ [4. Cross-Study Comparison Tables] (extends landscape model)

[3. Research Gaps Section] (independent — uses existing state.gaps)

[5. BibTeX/RIS Export] (independent — uses existing source metadata)
```

All items are largely independent and can be developed in any order. Item 4 extends item 2's model.
