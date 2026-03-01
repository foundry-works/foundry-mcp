# PLAN-4: Deep Analysis — Full-Text, Citation Networks & Quality Assessment

> **Branch**: `deep-academic`
>
> **Goal**: Capabilities that make deep research competitive with dedicated academic tools like Elicit, Research Rabbit, and Undermind — full-text PDF analysis, citation network graph construction, and methodology quality assessment.
>
> **Estimated scope**: ~650-900 LOC implementation + ~240-360 LOC tests
>
> **Dependencies**: PLAN-0 (prerequisites), PLAN-1 items 1 and 6 (profiles, structured output), PLAN-2 item 1 (OpenAlex)
>
> **Risk**: PDF parsing reliability, LLM accuracy for methodology extraction
>
> **Source**: [future/PLAN-4-DEEP-ANALYSIS.md](future/PLAN-4-DEEP-ANALYSIS.md)

---

## Design Principles

1. **Opt-in everything.** Every feature is gated behind profile flags. The `general` and basic `academic` profiles don't enable any of this. Only `systematic-review` and `bibliometric` profiles do.
2. **Degrade gracefully.** PDF extraction will fail for paywalled papers, scanned PDFs, and many edge cases. The pipeline must continue without full-text when extraction fails.
3. **Respect rate limits.** Citation network construction requires many API calls. All batched operations must respect per-provider rate limits and provide progress tracking.
4. **No new dependencies.** Uses existing `pypdf` + `pdfminer.six` already in the dependency tree.

---

## Current State — Existing Infrastructure

The existing `pdf_extractor.py` (833 lines) at `src/foundry_mcp/core/research/pdf_extractor.py` is production-ready and provides:

- **Libraries**: `pypdf` (primary) with `pdfminer.six` fallback
- **SSRF protection**: URL validation blocking localhost, internal IPs, cloud metadata endpoints
- **Security**: Magic byte validation (`%PDF-`), content-type checking, streaming download with size limits (default 10MB)
- **Page extraction**: Page-by-page text extraction with character offset tracking (`page_offsets` list)
- **Graceful fallback**: pypdf -> pdfminer.six (per-page) -> pdfminer.six (full document)
- **Observability**: Prometheus metrics, custom exceptions
- **DigestPayload** model supports page-based locators (`page:n:char:start-end`)
- Config field `deep_research_digest_fetch_pdfs: bool = False` exists but is disabled

---

## Item 1: Full-Text PDF Analysis

### Problem

Academic papers are primarily distributed as PDFs. The current pipeline fetches abstracts from Semantic Scholar and web page content via Tavily, but cannot read actual paper content. When a topic researcher discovers a paper, it only gets the abstract (~200-300 words). The full paper (methods, results, discussion) remains inaccessible.

### Architecture

```
Topic Researcher discovers paper via Semantic Scholar / OpenAlex
    |
Paper has DOI -> OpenAlex oa_url OR Unpaywall resolves to OA PDF URL
    | (or direct pdf_url from provider metadata)
Existing PDFExtractor.extract_from_url():
    1. SSRF-validated download (already implemented)
    2. Extract text via pypdf + pdfminer fallback (already implemented)
    3. Return PDFExtractionResult with page boundaries (already implemented)
    |
Standard summarization/digest pipeline processes extracted text
    |
Evidence snippets with page-based locators (page:3:char:150-320)
```

### Changes

**File: `src/foundry_mcp/core/research/pdf_extractor.py`** (EXTEND)

#### 1a. Add academic paper section detection

```python
def detect_sections(self, result: PDFExtractionResult) -> dict[str, tuple[int, int]]:
    """Detect standard academic paper sections from extracted text.

    Returns section name -> (start_char, end_char) mapping for:
    Abstract, Introduction, Methods/Methodology, Results, Discussion,
    Conclusion, References.

    Uses regex patterns for common section headers.
    Falls back gracefully — returns empty dict if no sections detected.
    """
```

#### 1b. Add prioritized extraction mode

```python
def extract_prioritized(
    self,
    result: PDFExtractionResult,
    max_chars: int = 50000,
    priority_sections: list[str] = ["methods", "results", "discussion"],
) -> str:
    """Extract text prioritizing specific sections for academic papers.

    When full text exceeds max_chars, prioritizes abstract + priority_sections
    over introduction/references. Useful for feeding into digest pipeline
    without exceeding context limits.
    """
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 1c. Integrate PDF extraction into extract_content tool

When `extract_content` is called with a URL that resolves to a PDF (detected by URL pattern `*.pdf`, `arxiv.org/pdf/*`, or Content-Type response header):
1. Route to existing `PDFExtractor.extract_from_url()` instead of Tavily Extract
2. Use `extract_prioritized()` to get section-aware content within context limits
3. Preserve page boundaries in source metadata for locator support

#### 1d. PDF-aware tool (optional, profile-gated)

When profile has `enable_pdf_extraction == True`, add an `extract_pdf` tool to the topic researcher:

```
### extract_pdf (systematic-review profile only)
Extract full text from an open-access academic paper PDF.
Arguments: {"url": "PDF URL or DOI", "max_pages": 30}
Returns: Full paper text with page numbers and section structure.
```

**File: `src/foundry_mcp/core/research/document_digest/digestor.py`**

#### 1e. Page-aware digest

When digesting PDF-extracted content:
- Use `page:N:char:start-end` locators for evidence snippets (existing `page_offsets` from `PDFExtractionResult`)
- Use `detect_sections()` to prioritize Methods, Results, and Discussion
- Handle academic paper structure

### Dependencies

- No new dependencies. Uses existing `pypdf` + `pdfminer.six`.
- PLAN-2 item 1 (OpenAlex `oa_url`) for DOI -> PDF URL resolution.

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Paywall PDFs | Only attempt OA PDFs (from OpenAlex oa_url / provider metadata) |
| Large PDFs | Existing `max_pages` config (override to 50 for deep research) |
| Scanned PDFs | Existing detection via text-per-page threshold in `PDFExtractor` |
| Publisher rate limiting | Respect robots.txt patterns, add 1s delay between downloads |
| Large file size | Existing streaming download with size limit (default 10MB) |
| Extraction failures | Log in provenance, continue with abstract-only |

### Testing

- Unit: section detection from synthetic academic PDF text
- Unit: prioritized extraction respects max_chars and section ordering
- Unit: PDF URL routing in extract_content tool
- Unit: graceful fallback when section detection finds nothing
- Integration: topic researcher extracts PDF and includes in findings
- ~80-120 LOC new tests

---

## Item 2: Citation Network / Connected Papers Graph (User-Triggered)

### Problem

Academic research involves understanding how papers relate through citation chains. A seminal paper can spawn multiple research threads. Understanding these threads is central to a good literature review. Currently, each source is treated as independent — there's no representation of inter-source relationships.

### Scope

**User-triggered, not automatic.** Building a citation network requires 2 API calls per source (references + citations). For 15 sources, that's 30+ API calls returning potentially 600+ nodes. Exposed as a dedicated `deep-research-network` action that users invoke on a completed research session.

### Architecture

```
User completes deep research session (produces report + sources)
    |
User explicitly requests: research action=deep-research-network research_id=<id>
    |
For each academic source with a paper ID:
    - Fetch references (papers it cites) via OpenAlex (100 req/s)
    - Fetch citations (papers that cite it) via OpenAlex
    |
Build citation adjacency graph:
    - Nodes: papers (discovered + referenced)
    - Edges: citation relationships (directed)
    |
Identify structure:
    - Papers cited by many discovered papers -> "foundational"
    - Papers that cite many discovered papers -> "recent extensions"
    - Connected components -> "research threads"
    |
Output: CitationNetwork in structured output
```

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 2a. Citation network model

```python
class CitationNode(BaseModel):
    paper_id: str
    title: str
    authors: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    is_discovered: bool  # True if in state.sources
    source_id: Optional[str] = None
    role: str = "peripheral"  # "foundational" | "discovered" | "extension" | "peripheral"

class CitationEdge(BaseModel):
    citing_paper_id: str
    cited_paper_id: str

class CitationNetwork(BaseModel):
    nodes: list[CitationNode]
    edges: list[CitationEdge]
    foundational_papers: list[str]  # paper_ids cited by many discovered papers
    research_threads: list[dict]    # [{name, paper_ids, description}]
    stats: dict                     # {total_nodes, total_edges, ...}
```

#### 2b. Add to ResearchExtensions

```python
citation_network: Optional[CitationNetwork] = None
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`** (NEW)

#### 2c. Network builder

`CitationNetworkBuilder` class with:
- `build_network()` — fetches refs/cites for each source, builds graph
- `_identify_foundational_papers()` — papers cited by 3+ discovered papers (or 30%)
- `_identify_research_threads()` — connected components via BFS/union-find (3+ nodes)
- `_classify_roles()` — foundational, discovered, extension, peripheral

Uses OpenAlex (primary, 100 req/s) or Semantic Scholar (fallback, 1 RPS).

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

#### 2d. Dedicated action handler

Wire up `"deep-research-network"` action in the research router:

```python
def _handle_deep_research_network(
    research_id: str,
    max_references_per_paper: int = 20,
    max_citations_per_paper: int = 20,
) -> dict:
    """Build citation network for a completed research session.
    User-triggered. Requires a completed session with 3+ academic sources.
    """
```

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| API rate limiting | OpenAlex (100 req/s) primary; Semantic Scholar (1 RPS) fallback |
| Graph explosion | Cap `max_references_per_paper` and `max_citations_per_paper` (default 20 each) |
| Missing paper IDs | Only build for sources with paper_id or DOI in metadata |
| Execution time | User-triggered, so latency expected. Progress tracking via action response. |

### Testing

- Unit: network builder with mocked provider responses
- Unit: foundational paper identification
- Unit: research thread detection with known graph
- Unit: role classification
- Unit: graceful handling when < 3 academic sources (returns skipped)
- Unit: action handler wiring and response format
- Integration: 5 academic sources -> network with edges
- ~100-140 LOC new tests

---

## Item 3: Methodology Quality Assessment (Experimental)

> **Status: Experimental.** Produces approximate heuristics, not validated instruments. No numeric rigor score — provides structured metadata to the synthesis LLM for qualitative judgment.

### Problem

Not all studies are equal. An RCT with N=1000 carries more weight than a case study with N=5. Currently, source quality assessment is purely domain-based. It doesn't assess methodological characteristics of individual studies.

### Architecture

```
After topic researchers gather findings:
    |
For each academic source with sufficient content (>200 chars):
    - LLM extracts methodology metadata (study design, sample, effect size)
    |
Structured methodology metadata (NO numeric score)
    |
Output: MethodologyAssessment per source, fed into synthesis prompt as context
```

### Changes

**File: `src/foundry_mcp/core/research/models/sources.py`**

#### 3a. Methodology assessment model

```python
class StudyDesign(str, Enum):
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "randomized_controlled_trial"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    COHORT = "cohort_study"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    QUALITATIVE = "qualitative"
    CASE_STUDY = "case_study"
    THEORETICAL = "theoretical"
    OPINION = "expert_opinion"
    UNKNOWN = "unknown"

class MethodologyAssessment(BaseModel):
    source_id: str
    study_design: StudyDesign = StudyDesign.UNKNOWN
    sample_size: Optional[int] = None
    sample_description: Optional[str] = None
    effect_size: Optional[str] = None
    statistical_significance: Optional[str] = None
    limitations_noted: list[str] = Field(default_factory=list)
    potential_biases: list[str] = Field(default_factory=list)
    confidence: str = "low"           # "high" | "medium" | "low"
    content_basis: str = "abstract"   # "abstract" | "full_text"
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`** (NEW)

#### 3b. Assessment engine

`MethodologyAssessor` class with `assess_sources()` — extracts methodology metadata via batched LLM call. Confidence forced to "low" for abstract-only content.

#### 3c. LLM extraction prompt

Structured JSON extraction prompt for study design, sample, effect size, statistical significance, limitations, and biases.

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 3d. Add to ResearchExtensions

```python
methodology_assessments: list[MethodologyAssessment] = Field(default_factory=list)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 3e. Feed assessments into synthesis

When assessments are available, inject methodology context into synthesis prompt:

```
## Methodology Context
[1] Smith et al. (2021) — Randomized controlled trial, N=450
    Effect: d=0.45 (p<0.001)
    Limitations: Single institution, self-report measures
    (extracted from: full text)

[2] Jones (2023) — Semi-structured interviews, N=12
    Limitations: Small sample, researcher bias possible
    (extracted from: abstract — treat with lower confidence)
```

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM accuracy from abstracts | Force `confidence="low"` for abstract-only |
| Cost (1 LLM call per source) | Use lightweight model, batch where possible |
| Over-reliance on metadata | Synthesis prompt frames as "context" not "ground truth" |
| Scope creep | Only assess sources with >200 chars of content |

### Testing

- Unit: LLM extraction prompt parsing with mocked responses
- Unit: StudyDesign classification from various abstracts
- Unit: graceful handling for sources without sufficient content
- Unit: assessment data correctly injected into synthesis prompt
- Unit: confidence forced to "low" for abstract-only content
- Integration: end-to-end assessment of 5 academic sources
- ~80-100 LOC new tests

---

## Removed Items (Feb 2026 Tool Evaluation)

- **Item 4 (MCP Bridge)**: Reframed as external documentation. Scite, Consensus, and PubMed MCP servers exist — users configure directly.
- **Item 5 (CORE Provider)**: Removed. Low throughput (5 req/10s), redundant with OpenAlex OA coverage.

---

## Configuration Surface

```python
# PDF extraction (extends existing PDFExtractor config)
deep_research_pdf_max_pages: int = 50
deep_research_pdf_priority_sections: list[str] = ["methods", "results", "discussion"]

# Citation network (user-triggered)
deep_research_citation_network_max_refs_per_paper: int = 20
deep_research_citation_network_max_cites_per_paper: int = 20

# Methodology assessment (experimental)
deep_research_methodology_assessment_provider: Optional[str] = None
deep_research_methodology_assessment_timeout: float = 60.0
```

---

## File Impact Summary

| File | Type | Items |
|------|------|-------|
| `pdf_extractor.py` | **Extend** | 1 (section detection, prioritized extraction) |
| `phases/topic_research.py` | Modify | 1 (PDF extraction routing) |
| `document_digest/digestor.py` | Modify | 1 (page-aware digest) |
| `phases/citation_network.py` | **New** | 2 |
| `models/deep_research.py` | Modify | 2, 3 (via ResearchExtensions) |
| `models/sources.py` | Modify | 3 (StudyDesign, MethodologyAssessment) |
| `phases/methodology_assessment.py` | **New** | 3 |
| `phases/synthesis.py` | Modify | 3 (assessment injection) |
| `handlers_deep_research.py` | Modify | 2 (network action) |
| `config/research.py` | Modify | All (new config fields) |

## Dependency Graph

```
[1. PDF Analysis] ── independent, extends existing pdf_extractor.py
    Enhanced by: PLAN-2 item 1 (OpenAlex oa_url)

[2. Citation Network] ── independent, user-triggered
    Depends on: PLAN-2 item 1 (OpenAlex)

[3. Methodology Assessment] ── independent, experimental
    Better with item 1 for full-text (but works with abstracts)
```

All three items can proceed independently. Item 2 is user-triggered (not automatic).

## Estimated Scope

| Item | Impl LOC | Test LOC |
|------|----------|----------|
| 1. PDF Analysis (extend) | ~150-200 | ~80-120 |
| 2. Citation Network | ~300-400 | ~100-140 |
| 3. Methodology (experimental) | ~200-300 | ~80-100 |
| **Total** | **~650-900** | **~240-360** |
