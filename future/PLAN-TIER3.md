# PLAN-TIER3: Aspirational Academic Research Improvements

> **Goal**: Capabilities that would make deep research competitive with dedicated academic research tools like Elicit, Consensus, or Research Rabbit.
>
> **Estimated scope**: ~1500-2500 LOC, significant new infrastructure
>
> **Dependencies**: Tier 1 (all items) and Tier 2 items 6-7 are strong prerequisites
>
> **Risk**: Higher implementation complexity, potential API rate limiting issues, PDF parsing reliability

---

## 10. Full-Text PDF Analysis

### Problem
Academic papers are primarily distributed as PDFs. The current implementation fetches abstracts from Semantic Scholar and web page content via Tavily, but cannot read the actual paper content. The config field `deep_research_digest_fetch_pdfs` exists but the pipeline for downloading, extracting text from, and analyzing PDFs is incomplete.

Key gap: When a topic researcher discovers a paper via Semantic Scholar, it only gets the abstract (~200-300 words). The full paper (methods, results, discussion) remains inaccessible, severely limiting the depth of findings.

### Current State

- Semantic Scholar provider stores `metadata.pdf_url` (open access PDF link) when available
- `SourceType` enum includes relevant types
- `DigestPayload` model supports page-based locators (`page:n:char:start-end`)
- Config field `deep_research_digest_fetch_pdfs: bool = False` exists but is disabled
- No PDF download or text extraction implementation exists

### Proposed Architecture

```
Topic Researcher discovers paper via Semantic Scholar
    ↓
extract_content tool called with PDF URL (or DOI redirect)
    ↓
New PDFExtractProvider:
    1. Download PDF (respect Content-Type, handle redirects)
    2. Extract text via pymupdf (fitz) or pdfplumber
    3. Preserve page boundaries for locator support
    4. Return structured text with page numbers
    ↓
Standard summarization/digest pipeline processes extracted text
    ↓
Evidence snippets with page-based locators (page:3:char:150-320)
```

### Changes

**File: `src/foundry_mcp/core/research/providers/pdf_extract.py`** (NEW)

#### 10a. PDF extraction provider

```python
class PDFExtractProvider:
    """Download and extract text from academic PDFs.

    Supports:
    - Direct PDF URLs (e.g., arxiv.org/pdf/2301.12345)
    - DOI redirects to publisher PDF pages
    - Open Access PDF links from Semantic Scholar metadata

    Text extraction via pymupdf (fitz) with page boundary preservation.
    """

    async def extract(
        self,
        url: str,
        max_pages: int = 50,
        timeout: float = 30.0,
    ) -> PDFExtractionResult:
        """Download PDF and extract text with page boundaries."""

    def _extract_text_with_pages(
        self,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> list[PDFPage]:
        """Extract text from PDF bytes, preserving page structure."""
```

#### 10b. PDF data models

```python
class PDFPage(BaseModel):
    """Single page of extracted PDF text."""
    page_number: int
    text: str
    char_offset: int  # Cumulative character offset from start of document

class PDFExtractionResult(BaseModel):
    """Result of PDF text extraction."""
    pages: list[PDFPage]
    total_pages: int
    total_chars: int
    metadata: dict  # PDF metadata (title, author, creation date)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 10c. Integrate PDF extraction into extract_content tool

When `extract_content` is called with a PDF URL (detected by URL pattern or Content-Type):
1. Route to `PDFExtractProvider` instead of `TavilyExtractProvider`
2. Convert `PDFExtractionResult` to the standard content format
3. Preserve page boundaries in source metadata for locator support

#### 10d. Add `extract_pdf` as a dedicated tool (optional)

Alternative: add a separate `extract_pdf` tool that topic researchers can explicitly invoke:

```
### extract_pdf (academic mode only)
Extract full text from an academic paper PDF.
Arguments: {"url": "PDF URL or DOI", "max_pages": 30}
Returns: Full paper text with page numbers.
```

**File: `src/foundry_mcp/core/research/document_digest/digestor.py`**

#### 10e. Page-aware digest

When digesting PDF-extracted content:
- Use `page:N:char:start-end` locators for evidence snippets
- Prioritize Methods, Results, and Discussion sections for evidence extraction
- Handle academic paper structure (Abstract, Introduction, Methods, Results, Discussion, References)

### Dependencies
- `pymupdf` (fitz) or `pdfplumber` for PDF text extraction
- Optional: `pdfminer.six` as fallback for complex PDFs

### Risks & Mitigations
- **Paywall PDFs**: Only attempt open-access PDFs (check `metadata.pdf_url` from Semantic Scholar)
- **Large PDFs**: Cap at 50 pages, skip appendices/supplementary material
- **Scanned PDFs**: pymupdf handles text PDFs well but not scanned images — detect and skip
- **Rate limiting**: Academic publishers may rate-limit PDF downloads — respect robots.txt and add delays
- **Size**: PDFs can be large (10-50MB) — stream download with size limit

### Testing
- Unit test: PDF text extraction from a test PDF file
- Unit test: page boundary preservation and char offset calculation
- Unit test: PDF URL detection (arxiv.org/pdf/*, *.pdf extension, etc.)
- Unit test: graceful failure for non-PDF URLs, scanned PDFs, corrupted files
- Integration test: topic researcher extracts PDF and includes in findings

---

## 11. Citation Network / Connected Papers Graph

### Problem
Academic research involves understanding how papers relate to each other through citation chains. A single seminal paper can spawn multiple research threads. Understanding these threads is central to writing a good literature review. Currently, each source is treated as independent — there's no representation of inter-source relationships.

### Proposed Architecture

```
After synthesis (or as a post-synthesis enrichment step):
    ↓
For each academic source in state.sources:
    - Fetch its references (papers it cites) via Semantic Scholar
    - Fetch its citations (papers that cite it) via Semantic Scholar
    ↓
Build citation adjacency graph:
    - Nodes: papers (both discovered and referenced)
    - Edges: citation relationships (directed)
    ↓
Identify clusters:
    - Papers that cite each other form research threads
    - Papers cited by many discovered papers are "foundational"
    - Papers that cite many discovered papers are "recent extensions"
    ↓
Output: CitationNetwork model in state
```

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 11a. Citation network model

```python
class CitationNode(BaseModel):
    """A paper in the citation network."""
    paper_id: str
    title: str
    authors: str
    year: Optional[int]
    citation_count: Optional[int]
    is_discovered: bool  # True if this paper is in state.sources
    source_id: Optional[str]  # Link to ResearchSource if discovered
    role: str  # "foundational" | "discovered" | "extension" | "peripheral"

class CitationEdge(BaseModel):
    """A citation relationship between two papers."""
    citing_paper_id: str
    cited_paper_id: str

class CitationNetwork(BaseModel):
    """Citation network built from discovered sources."""
    nodes: list[CitationNode]
    edges: list[CitationEdge]
    clusters: list[dict]  # [{name: str, paper_ids: [str], theme: str}]
    foundational_papers: list[str]  # paper_ids cited by many discovered papers
    research_threads: list[dict]  # [{name: str, paper_ids: [str], description: str}]
```

#### 11b. Add to state

```python
# In DeepResearchState:
citation_network: Optional[CitationNetwork] = Field(
    default=None,
    description="Citation network built from discovered academic sources",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`** (NEW)

#### 11c. Network builder

```python
class CitationNetworkBuilder:
    """Build citation network from discovered sources.

    Uses Semantic Scholar API to fetch references and citations
    for each discovered academic source, then builds a graph
    identifying clusters, foundational papers, and research threads.
    """

    async def build_network(
        self,
        sources: list[ResearchSource],
        provider: SemanticScholarProvider,
        max_references_per_paper: int = 20,
        max_citations_per_paper: int = 20,
        max_concurrent: int = 3,
    ) -> CitationNetwork:
        """Build citation network from discovered sources."""

    def _identify_foundational_papers(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
    ) -> list[str]:
        """Papers cited by many discovered papers are foundational."""

    def _identify_clusters(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
    ) -> list[dict]:
        """Use connected components / community detection on citation graph."""

    def _classify_roles(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
        discovered_ids: set[str],
    ) -> None:
        """Assign roles: foundational, discovered, extension, peripheral."""
```

#### 11d. Integration point

Call network builder as a post-synthesis enrichment step (optional, enabled by config):

```python
if state.research_mode == ResearchMode.ACADEMIC and config.deep_research_citation_network_enabled:
    network = await CitationNetworkBuilder().build_network(
        sources=[s for s in state.sources if s.source_type == SourceType.ACADEMIC],
        provider=semantic_scholar_provider,
    )
    state.citation_network = network
```

### Risks & Mitigations
- **API rate limiting**: Semantic Scholar has 1 RPS limit. With 15 sources x 2 API calls each = 30 calls = 30+ seconds minimum. Parallelize within rate limits.
- **Graph explosion**: Each paper may have hundreds of citations. Cap at `max_references_per_paper` and `max_citations_per_paper`.
- **Missing paper IDs**: Not all sources have Semantic Scholar paper IDs. Only build network for sources with `metadata.paper_id`.

### Testing
- Unit test: network builder with mocked Semantic Scholar responses
- Unit test: foundational paper identification (paper cited by 5+ discovered papers)
- Unit test: cluster detection with known graph structure
- Unit test: role classification
- Unit test: graceful handling when no academic sources exist
- Integration test: end-to-end with 5 academic sources -> network with edges

---

## 12. Methodology Quality Assessment

### Problem
Not all studies are created equal. A randomized controlled trial with N=1000 carries more weight than a case study with N=5. Currently, source quality assessment (`source_quality.py`) is purely domain-based (is this from a journal? from social media?). It doesn't assess the methodological rigor of individual studies.

### Proposed Architecture

```
After topic researchers gather findings:
    ↓
For each academic finding with sufficient detail:
    - LLM extracts methodology metadata:
        - Study design (RCT, quasi-experimental, observational, qualitative, case study, etc.)
        - Sample size
        - Effect size (if reported)
        - Confidence intervals / p-values
        - Potential biases noted by authors
        - Limitations acknowledged
    ↓
Methodology quality scoring:
    - Study design rigor (RCT > quasi-experimental > observational > case study)
    - Sample size adequacy
    - Statistical reporting completeness
    - Bias acknowledgment
    ↓
Output: MethodologyAssessment per finding/source
```

### Changes

**File: `src/foundry_mcp/core/research/models/sources.py`**

#### 12a. Methodology assessment model

```python
class StudyDesign(str, Enum):
    """Research study design types, ordered by typical rigor."""
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
    """Assessment of a study's methodological quality."""
    source_id: str
    study_design: StudyDesign
    sample_size: Optional[int]
    sample_description: Optional[str]
    effect_size: Optional[str]  # Reported as string (e.g., "d=0.45", "r=0.32")
    statistical_significance: Optional[str]  # e.g., "p<0.001"
    limitations_noted: list[str]  # Limitations acknowledged by authors
    potential_biases: list[str]  # Biases identified during assessment
    rigor_score: float  # 0.0-1.0 composite quality score
    confidence: str  # "high" | "medium" | "low" — confidence in this assessment
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`** (NEW)

#### 12b. Assessment engine

```python
class MethodologyAssessor:
    """Assess methodological quality of academic sources.

    Uses LLM to extract study design, sample characteristics,
    and statistical reporting from source content, then computes
    a composite rigor score.
    """

    async def assess_sources(
        self,
        sources: list[ResearchSource],
        findings: list[ResearchFinding],
        provider_id: Optional[str] = None,
        timeout: float = 60.0,
    ) -> list[MethodologyAssessment]:
        """Assess methodology for all academic sources with sufficient content."""

    def _compute_rigor_score(self, assessment: MethodologyAssessment) -> float:
        """Compute composite rigor score from study characteristics.

        Weights:
        - Study design hierarchy: 40%
        - Sample size adequacy: 20%
        - Statistical reporting: 20%
        - Limitation acknowledgment: 10%
        - Bias awareness: 10%
        """
```

#### 12c. LLM extraction prompt

```
You are a research methodology analyst. Given a research source's content,
extract the following methodological details:

1. Study design: What type of study is this? (RCT, quasi-experimental, qualitative, etc.)
2. Sample size: How many participants/subjects/data points?
3. Sample description: Who/what was studied?
4. Effect size: Was an effect size reported? If so, what?
5. Statistical significance: Were p-values or confidence intervals reported?
6. Limitations: What limitations did the authors acknowledge?
7. Potential biases: What biases do you identify (selection bias, confirmation bias, etc.)?

Respond as JSON:
{
    "study_design": "randomized_controlled_trial",
    "sample_size": 450,
    "sample_description": "Undergraduate students at a US university",
    "effect_size": "d=0.45, 95% CI [0.32, 0.58]",
    "statistical_significance": "p<0.001",
    "limitations": ["Self-selected sample", "Single institution"],
    "potential_biases": ["WEIRD sample bias", "Self-report measures"],
    "confidence": "high"
}
```

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 12d. Add to state

```python
# In DeepResearchState:
methodology_assessments: list[MethodologyAssessment] = Field(
    default_factory=list,
    description="Methodology quality assessments for academic sources",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 12e. Feed assessments into synthesis

When assessments are available, include them in the synthesis prompt:

```
## Methodology Quality Assessments
The following sources have been assessed for methodological rigor:

[1] Smith et al. (2021) — RCT, N=450, rigor: 0.85/1.0
    Design: Randomized controlled trial
    Effect: d=0.45 (p<0.001)
    Limitations: Single institution, self-report measures

[2] Jones (2023) — Qualitative, N=12, rigor: 0.55/1.0
    Design: Semi-structured interviews
    Limitations: Small sample, researcher bias possible

When synthesizing findings, weight higher-rigor studies more heavily.
Note methodological limitations when reporting findings from lower-rigor studies.
```

### Risks & Mitigations
- **LLM accuracy**: Methodology extraction from abstracts is imperfect. Full-text access (Tier 3, item 10) significantly improves accuracy.
- **Cost**: One LLM call per source. Use cheap model (gemini-2.5-flash). Batch where possible.
- **Scope creep**: Only assess sources with `source_type == ACADEMIC` and sufficient content (>200 chars of abstract/content).
- **False precision**: The rigor score is heuristic, not a validated instrument. Present it as "approximate" in outputs.

### Testing
- Unit test: `_compute_rigor_score()` with various study designs and characteristics
- Unit test: LLM extraction prompt parsing with mocked responses
- Unit test: methodology assessment model validation
- Unit test: graceful handling for sources without sufficient content
- Unit test: assessment data correctly injected into synthesis prompt
- Integration test: end-to-end assessment of 5 academic sources

---

## Cross-Cutting Considerations for Tier 3

### New Dependencies
- `pymupdf` (fitz) — for PDF text extraction (item 10)
- No other new dependencies required

### Configuration Surface

```python
# PDF extraction
deep_research_pdf_extraction_enabled: bool = False  # Opt-in
deep_research_pdf_max_pages: int = 50
deep_research_pdf_max_size_mb: float = 50.0
deep_research_pdf_timeout: float = 30.0

# Citation network
deep_research_citation_network_enabled: bool = False  # Opt-in
deep_research_citation_network_max_refs_per_paper: int = 20
deep_research_citation_network_max_cites_per_paper: int = 20

# Methodology assessment
deep_research_methodology_assessment_enabled: bool = False  # Opt-in
deep_research_methodology_assessment_provider: Optional[str] = None  # Cheap model
deep_research_methodology_assessment_model: Optional[str] = None
deep_research_methodology_assessment_timeout: float = 60.0
```

All Tier 3 features are **opt-in** (disabled by default) to avoid impacting existing behavior or cost.

### Performance Impact

| Feature | API Calls | LLM Calls | Added Latency |
|---------|-----------|-----------|---------------|
| PDF extraction | 1 download per PDF | 0 (text extraction only) | 5-15s per paper |
| Citation network | 2 S2 calls per source | 0 (graph analysis only) | 30-60s total |
| Methodology assessment | 0 | 1 per academic source | 15-30s total |

### File Impact Summary

| File | Change Type | Items |
|------|-------------|-------|
| `providers/pdf_extract.py` | **New** | 10 |
| `phases/topic_research.py` | Modify | 10 (PDF extraction routing) |
| `document_digest/digestor.py` | Modify | 10 (page-aware digest) |
| `phases/citation_network.py` | **New** | 11 |
| `models/deep_research.py` | Modify | 11 (CitationNetwork), 12 (MethodologyAssessment) |
| `models/sources.py` | Modify | 12 (StudyDesign, MethodologyAssessment) |
| `phases/methodology_assessment.py` | **New** | 12 |
| `phases/synthesis.py` | Modify | 12 (assessment injection) |
| `workflow_execution.py` | Modify | 11, 12 (post-synthesis enrichment steps) |
| `config/research.py` | Modify | 10, 11, 12 (new config fields) |
