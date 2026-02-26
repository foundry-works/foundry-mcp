# PLAN-TIER1: High-Impact Academic Research Improvements

> **Goal**: Make deep research immediately useful for academic literature reviews by building on existing infrastructure with targeted changes.
>
> **Estimated scope**: ~800-1200 LOC across 8-10 files
>
> **Dependencies**: None (all changes build on existing code)

---

## 1. Add `literature_review` Query Type to Synthesis

### Problem
The synthesis phase (`phases/synthesis.py:59-71`) classifies queries into four types: `comparison`, `enumeration`, `howto`, and `explanation`. Academic literature review queries fall through to the generic `explanation` type, producing a shallow "Key Findings / Conclusions" structure instead of the thematic, chronological, methodological organization researchers expect.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 1a. Add detection pattern (after line 56)

```python
_LITERATURE_REVIEW_PATTERNS = re.compile(
    r"\b(literature review|systematic review|meta.?analysis|survey of|"
    r"state of the art|research landscape|body of (research|literature|work)|"
    r"prior (work|research|studies)|existing (research|literature|studies)|"
    r"background (research|review)|review of (the )?(literature|research|studies)|"
    r"what (does|has) (the )?research (say|show|suggest|indicate|find)|"
    r"research (on|into|about|regarding) .{5,60} (in|for|across))\b",
    re.IGNORECASE,
)
```

#### 1b. Add classification check (in `_classify_query_type()`, before the `return "explanation"` fallback)

Insert a check for `_LITERATURE_REVIEW_PATTERNS` **before** the existing comparison/enumeration/howto checks, since academic queries like "compare assessment methods in educational research" should prefer `literature_review` over `comparison` when academic signals are present.

Also check `state.research_mode == ResearchMode.ACADEMIC` as a secondary signal — if the user has explicitly set academic mode, bias toward literature_review for ambiguous queries.

#### 1c. Add structure guidance entry in `_STRUCTURE_GUIDANCE` dict (after line 107)

```python
"literature_review": """\
For **literature review** queries, use this structure:
# Literature Review: [Topic]
## Executive Summary
Brief overview of the research landscape, key themes, and scope of the review.
## Introduction & Scope
Define the topic, its significance, and the boundaries of this review.
## Theoretical Foundations
Seminal works and foundational theories that define the field.
## Thematic Analysis
### [Theme 1]
### [Theme 2]
### [Theme N]
Organize findings by major themes or research questions. Within each theme,
discuss studies chronologically or by methodology as appropriate.
## Methodological Approaches
Overview of research methods used across the literature (quantitative,
qualitative, mixed methods, case studies, etc.) with strengths and limitations.
## Key Debates & Contradictions
Where studies disagree, present both sides with evidence and possible explanations.
## Research Gaps & Future Directions
Explicitly identify what remains unstudied or underexplored.
## Conclusions
Synthesize the overall state of knowledge and its implications.
## References
Full bibliographic entries in APA format (auto-generated from source metadata).""",
```

#### 1d. Modify `_build_synthesis_system_prompt()` (line 592+)

When `query_type == "literature_review"`, inject additional system prompt instructions:

```
Additional instructions for literature review synthesis:
- Organize studies thematically, not just as a list of summaries.
- For each study cited, include author(s), year, and a brief methodological note
  (e.g., "Smith & Jones (2021) conducted a randomized controlled trial with N=450...").
- Identify seminal/foundational works explicitly (high citation count, historical significance).
- Note methodological trends across the literature (shift from qualitative to mixed methods, etc.).
- When studies conflict, present both findings with context rather than choosing one.
- The References section MUST use APA 7th edition format using available metadata
  (authors, year, title, venue/journal, DOI).
```

### Testing
- Add test cases to `tests/core/research/workflows/test_deep_research.py` for `_classify_query_type()` with academic queries
- Verify the lit review structure guidance is selected for queries like:
  - "literature review on conversation-based assessment in education"
  - "what does the research say about formative assessment in K-12"
  - "survey of prior work on AI tutoring systems"
  - "existing research on adaptive learning technologies"

---

## 2. Format Citations with Full Bibliographic Metadata (APA Style)

### Problem
The citation postprocessing pipeline (`phases/_citation_postprocess.py:40-71`) formats sources as `[N] [Title](URL)`, discarding rich metadata already present in `ResearchSource.metadata` (authors, year, venue, DOI, citation_count). Academic researchers need proper bibliographic entries.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`**

#### 2a. Add APA formatting function (new helper)

```python
def format_source_apa(source: ResearchSource) -> str:
    """Format a ResearchSource as an APA 7th edition reference entry.

    Falls back gracefully when metadata is incomplete:
    - Full: Smith, J., & Jones, K. (2023). Title. *Journal*, 30(2), 145-168. https://doi.org/...
    - Minimal: Title. Retrieved from https://...
    """
```

Logic:
1. Extract `authors`, `year`, `venue` from `source.metadata`
2. If `source.source_type == SourceType.ACADEMIC` and metadata is rich:
   - Format as: `Authors (Year). Title. *Venue*. DOI_URL`
   - Handle "et al." for >5 authors (already formatted by Semantic Scholar provider)
3. If web source with limited metadata:
   - Format as: `Author/Organization (Year). Title. *Site Name*. URL`
4. Fallback: `Title. URL`

#### 2b. Modify `build_sources_section()` (line 40)

Add a `format_style: str = "default"` parameter. When `format_style == "apa"`:
- Use `format_source_apa()` instead of the current `[{cn}] [{title}]({url})` format
- Add section header "## References" instead of "## Sources"

#### 2c. Modify `postprocess_citations()` (line 137)

Accept `state.research_mode` and pass `format_style="apa"` to `build_sources_section()` when mode is ACADEMIC or query_type is `literature_review`.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 2d. Pass query_type through to postprocessing

The synthesis mixin calls `postprocess_citations(report, state)` — extend this to pass the classified `query_type` so the citation formatter can choose APA vs. default.

### Source Metadata Availability

The Semantic Scholar provider (`providers/semantic_scholar.py:394-441`) already stores:
- `metadata.authors` — comma-separated names (line 429)
- `metadata.year` — publication year (line 430)
- `metadata.venue` — journal/conference name (line 435)
- `metadata.doi` — DOI string (line 431)
- `metadata.citation_count` — for importance ranking (line 429)
- `metadata.fields_of_study` — disciplinary tags (line 438)

For web sources, Tavily provides `title` and `url` at minimum.

### Testing
- Unit test `format_source_apa()` with:
  - Full academic metadata (all fields present)
  - Partial metadata (missing venue, missing DOI)
  - Web source with no academic metadata
  - Source with >5 authors ("et al." handling)
- Integration test: run synthesis with ACADEMIC mode and verify References section format

---

## 3. Add Citation Graph and Related Papers Tools to Topic Researchers

### Problem
Topic researchers (`phases/topic_research.py`) only have `web_search`, `extract_content`, `think`, and `research_complete` tools. For academic research, two critical capabilities are missing:
- **Forward citation search**: "Find papers that cite this seminal paper" (snowball sampling)
- **Related papers**: "Find papers similar to this one" (lateral discovery)

Semantic Scholar's API supports both via endpoints not currently used.

### Changes

**File: `src/foundry_mcp/core/research/providers/semantic_scholar.py`**

#### 3a. Add citation search method

```python
async def get_citations(
    self,
    paper_id: str,
    max_results: int = 20,
    fields: str = DEFAULT_FIELDS,
    **kwargs: Any,
) -> list[ResearchSource]:
    """Get papers that cite a given paper.

    Uses: GET /paper/{paper_id}/citations
    See: https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_citations
    """
```

#### 3b. Add related papers method

```python
async def get_recommendations(
    self,
    paper_id: str,
    max_results: int = 10,
    **kwargs: Any,
) -> list[ResearchSource]:
    """Get recommended papers based on a given paper.

    Uses: POST /recommendations/v1/papers/
    See: https://api.semanticscholar.org/api-docs/recommendations
    """
```

#### 3c. Add paper lookup by DOI/ID

```python
async def get_paper(
    self,
    paper_id: str,
    fields: str = EXTENDED_FIELDS,
) -> Optional[ResearchSource]:
    """Look up a single paper by Semantic Scholar ID, DOI, or ArXiv ID.

    Uses: GET /paper/{paper_id}
    Supports ID formats: S2 paper ID, DOI:xxx, ArXiv:xxx, PMID:xxx
    """
```

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 3d. Add tool models

```python
class CitationSearchTool(BaseModel):
    """Search for papers citing a specific paper."""
    paper_id: str = Field(..., description="Semantic Scholar paper ID or DOI")
    max_results: int = Field(default=10, ge=1, le=50)

class RelatedPapersTool(BaseModel):
    """Find papers related to a specific paper."""
    paper_id: str = Field(..., description="Semantic Scholar paper ID or DOI")
    max_results: int = Field(default=10, ge=1, le=20)
```

Add to `RESEARCHER_TOOL_SCHEMAS` registry.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 3e. Add tool documentation to researcher system prompt

Add to `_RESEARCHER_SYSTEM_PROMPT` (after existing tool definitions, only when `research_mode == ACADEMIC`):

```
### citation_search (academic mode only)
Find papers that cite a specific paper. Useful for forward snowball sampling.
Arguments: {"paper_id": "DOI or Semantic Scholar ID", "max_results": 10}
Returns: List of citing papers with titles, authors, years, and abstracts.

### related_papers (academic mode only)
Find papers similar to a specific paper. Useful for lateral discovery.
Arguments: {"paper_id": "DOI or Semantic Scholar ID", "max_results": 10}
Returns: List of related papers with titles, authors, years, and abstracts.
```

#### 3f. Add dispatch handlers

Add `_handle_citation_search_tool()` and `_handle_related_papers_tool()` methods following the existing `_handle_web_search_tool()` pattern. These should:
1. Parse the tool arguments using the Pydantic models
2. Look up the Semantic Scholar provider from the provider registry
3. Call the appropriate provider method
4. Format results with novelty tracking
5. Count against the tool call budget

#### 3g. Conditional tool availability

Only inject academic tools into the system prompt when `state.research_mode == ResearchMode.ACADEMIC`. This keeps the general-purpose researcher lean.

### Testing
- Unit test each new Semantic Scholar method with mocked HTTP responses
- Unit test tool dispatch for `citation_search` and `related_papers`
- Integration test: verify topic researcher can chain `web_search` -> `citation_search` to find citing papers

---

## 4. Academic Brief Enrichment

### Problem
The brief phase system prompt (`phases/brief.py:203-244`) is generic. When `research_mode == ACADEMIC`, the brief should probe for discipline, education level, time period, and methodology preferences — dimensions that fundamentally shape a literature review.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`**

#### 4a. Add academic-specific brief system prompt

Modify `_build_brief_system_prompt()` to accept `research_mode` and append academic-specific instructions when mode is ACADEMIC:

```python
def _build_brief_system_prompt(self, research_mode: ResearchMode = ResearchMode.GENERAL) -> str:
```

Append when ACADEMIC:

```
Additional instructions for ACADEMIC research mode:
Your brief MUST additionally address these dimensions:
6. **Disciplinary scope**: Identify the primary discipline(s) and any interdisciplinary
   angles (e.g., "education + psychology + computer science").
7. **Time period**: Specify the temporal scope — foundational/seminal works (any era)
   plus recent literature (suggest last 5-10 years unless the user specifies otherwise).
8. **Methodology preferences**: Note whether the user seeks quantitative studies,
   qualitative research, mixed methods, meta-analyses, theoretical frameworks, or all types.
9. **Education level / population**: If applicable, specify the target population
   (K-12, higher education, professional development, specific age groups).
10. **Source type hierarchy**: For academic research, prioritize:
    (a) Peer-reviewed journal articles and conference papers
    (b) Systematic reviews and meta-analyses
    (c) Books and book chapters from academic publishers
    (d) Preprints from recognized repositories (arXiv, SSRN, EdArXiv)
    (e) Technical reports from research organizations
    Deprioritize: blog posts, news articles, Wikipedia, social media.
```

#### 4b. Pass research_mode from state

In `_execute_brief_async()`, pass `state.research_mode` to `_build_brief_system_prompt()`:

```python
system_prompt = self._build_brief_system_prompt(research_mode=state.research_mode)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`**

#### 4c. Academic-aware decomposition

In `_build_first_round_delegation_system_prompt()` (line 1795), append when ACADEMIC:

```
Additional decomposition guidelines for ACADEMIC research:
- Include a directive specifically targeting foundational/seminal works (sorted by citation count)
- Include a directive for recent empirical studies (last 3-5 years)
- If the topic spans disciplines, create separate directives per discipline
- Prefer directives that map to literature review sections: theoretical foundations,
  methodological approaches, empirical findings, and research gaps
- Each directive's "evidence_needed" should specify: peer-reviewed articles,
  sample sizes, effect sizes, or theoretical frameworks as appropriate
```

### Testing
- Verify academic brief enrichment with a mock LLM call
- Test that `research_mode=ACADEMIC` triggers the extended prompt
- Test that `research_mode=GENERAL` uses the original prompt unchanged

---

## 5. Expose `research_mode` as a Request-Time Parameter

### Problem
`research_mode` is currently only configurable at the global config level (`config/research.py:189`), not per-request. A user running the deep-research tool should be able to pass `research_mode: "academic"` without changing their config file.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/core.py`**

#### 5a. Add parameter to `execute()` (line 181)

```python
def execute(
    self,
    ...
    research_mode: Optional[str] = None,  # NEW: "general" | "academic" | "technical"
    ...
) -> WorkflowResult:
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py`**

#### 5b. Use request parameter with config fallback (line 124)

```python
research_mode=ResearchMode(research_mode or self.config.deep_research_mode),
```

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 5c. Add to handler signature (line 42)

```python
def _handle_deep_research(
    *,
    ...
    research_mode: Optional[str] = None,  # NEW
    ...
) -> dict:
```

**File: `src/foundry_mcp/tools/unified/research.py`**

#### 5d. Pass through from action router

Ensure `research_mode` flows from the MCP tool arguments through to the handler.

### Testing
- Test that `research_mode="academic"` at request time overrides config default
- Test that omitting `research_mode` falls back to config value
- Test invalid values raise appropriate errors

---

## 6. BibTeX/RIS Export

### Problem
Academic researchers need to import discovered papers into reference managers (Zotero, Mendeley, EndNote). The DOIs, authors, years, and venues are already in `ResearchSource.metadata` but there's no export capability.

### Changes

**File: `src/foundry_mcp/core/research/export/__init__.py`** (NEW)
**File: `src/foundry_mcp/core/research/export/bibtex.py`** (NEW)

#### 6a. BibTeX generator

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

**File: `src/foundry_mcp/core/research/export/ris.py`** (NEW)

#### 6b. RIS generator

```python
def sources_to_ris(sources: list[ResearchSource]) -> str:
    """Convert research sources to RIS (Research Information Systems) format.

    Generates TY/AU/TI/JO/PY/DO/UR/ER blocks per source.
    Compatible with Zotero, Mendeley, EndNote.
    """
```

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 6c. Add export action to deep-research handler

Add a `deep-research-export` action:

```python
def _handle_deep_research_export(
    research_id: str,
    format: str = "bibtex",  # "bibtex" | "ris"
) -> dict:
    """Export bibliography from completed research session."""
```

This loads the persisted `DeepResearchState`, filters to academic sources (or all sources), and returns the formatted bibliography string.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 6d. Append BibTeX to report metadata

After synthesis completes, generate BibTeX from `state.sources` and include it in the result metadata so the caller can access it without a separate export call:

```python
metadata["bibtex"] = sources_to_bibtex(state.sources)
```

### Testing
- Unit test BibTeX generation with various metadata completeness levels
- Unit test RIS generation
- Test citation key uniqueness and stability
- Test export action handler with a completed research session
- Verify Zotero import compatibility by validating BibTeX syntax

---

## Cross-Cutting: Configuration

**File: `src/foundry_mcp/config/research.py`**

Add new config fields:

```python
# Academic output format
deep_research_citation_style: str = "default"  # "default" | "apa" | "ieee" | "chicago"
deep_research_export_format: str = "bibtex"     # "bibtex" | "ris" | "csl-json"
```

These provide defaults that can be overridden per-request.

---

## File Impact Summary

| File | Change Type | Scope |
|------|-------------|-------|
| `phases/synthesis.py` | Modify | New query type, lit review structure, academic synthesis instructions |
| `phases/_citation_postprocess.py` | Modify | APA formatting, academic references section |
| `phases/brief.py` | Modify | Academic-aware brief enrichment |
| `phases/supervision.py` | Modify | Academic decomposition guidelines |
| `phases/topic_research.py` | Modify | New academic tools, conditional tool injection |
| `providers/semantic_scholar.py` | Modify | Citation graph, recommendations, paper lookup endpoints |
| `models/deep_research.py` | Modify | New tool models, tool registry update |
| `models/sources.py` | No change | Existing metadata sufficient |
| `config/research.py` | Modify | New config fields |
| `core.py` | Modify | New `research_mode` parameter |
| `action_handlers.py` | Modify | Pass `research_mode` from request |
| `handlers_deep_research.py` | Modify | Expose `research_mode`, add export action |
| `export/bibtex.py` | **New** | BibTeX export |
| `export/ris.py` | **New** | RIS export |
| `export/__init__.py` | **New** | Package init |
