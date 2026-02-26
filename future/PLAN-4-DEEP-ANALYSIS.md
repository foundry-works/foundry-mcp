# PLAN-4: Deep Analysis — Full-Text, Citation Networks & Quality Assessment

> **Goal**: Capabilities that make deep research competitive with dedicated academic tools like Elicit, Research Rabbit, and Undermind — full-text PDF analysis, citation network graph construction, methodology quality assessment, and integration with remote academic MCP servers (Scite, Consensus, PubMed).
>
> **Estimated scope**: ~2000-3000 LOC, significant new infrastructure
>
> **Dependencies**: PLAN-1 (all items), PLAN-2 items 1-5 (providers and citation tools), PLAN-3 items 1-2 (influence ranking and landscape)
>
> **Risk**: Higher implementation complexity, API rate limiting, PDF parsing reliability, LLM accuracy for methodology extraction
>
> **All features in this plan are opt-in** (disabled by default in profiles) to avoid impacting existing behavior.

---

## Design Principles

1. **Opt-in everything.** Every feature in this plan is gated behind profile flags. The `general` and basic `academic` profiles don't enable any of this. The `systematic-review` and `bibliometric` profiles do.
2. **Degrade gracefully.** PDF extraction will fail for paywalled papers, scanned PDFs, and many edge cases. The pipeline must continue without full-text when extraction fails.
3. **Respect rate limits.** Citation network construction requires many API calls. All batched operations must respect per-provider rate limits and provide progress tracking.
4. **MCP bridge pattern.** Remote MCP servers (Scite, Consensus, PubMed) are integrated via a generic MCP-to-provider bridge, not bespoke integrations per server.

---

## 1. Full-Text PDF Analysis

### Problem

Academic papers are primarily distributed as PDFs. The current pipeline fetches abstracts from Semantic Scholar and web page content via Tavily, but cannot read actual paper content. When a topic researcher discovers a paper, it only gets the abstract (~200-300 words). The full paper (methods, results, discussion) remains inaccessible.

### Current State

- `pdf_extractor.py` exists at `src/foundry_mcp/core/research/pdf_extractor.py` but its capabilities are unclear
- Semantic Scholar stores `metadata.pdf_url` (OA PDF link) when available
- PLAN-2's Unpaywall provider can resolve DOIs to OA PDF URLs
- `DigestPayload` model supports page-based locators (`page:n:char:start-end`)
- Config field `deep_research_digest_fetch_pdfs: bool = False` exists but is disabled

### Architecture

```
Topic Researcher discovers paper via Semantic Scholar / OpenAlex
    ↓
Paper has DOI → Unpaywall resolves to OA PDF URL
    ↓ (or direct pdf_url from provider metadata)
PDFExtractProvider:
    1. Download PDF (respect Content-Type, handle redirects, size limit)
    2. Extract text via pymupdf (fitz) with page boundary preservation
    3. Return structured text with page numbers
    ↓
Standard summarization/digest pipeline processes extracted text
    ↓
Evidence snippets with page-based locators (page:3:char:150-320)
```

### Changes

**File: `src/foundry_mcp/core/research/providers/pdf_extract.py`** (NEW)

#### 1a. PDF extraction provider

```python
class PDFExtractProvider:
    """Download and extract text from academic PDFs.

    Supports:
    - Direct PDF URLs (e.g., arxiv.org/pdf/2301.12345)
    - DOI redirects to publisher PDF pages
    - Open Access PDF links from Semantic Scholar / Unpaywall metadata

    Text extraction via pymupdf (fitz) with page boundary preservation.
    """

    async def extract(
        self,
        url: str,
        max_pages: int = 50,
        timeout: float = 30.0,
        max_size_mb: float = 50.0,
    ) -> PDFExtractionResult:
        """Download PDF and extract text with page boundaries.

        Steps:
        1. Stream-download with size limit check
        2. Verify Content-Type is application/pdf
        3. Extract text via pymupdf, preserving page structure
        4. Return structured result with page boundaries
        """

    def _extract_text_with_pages(
        self,
        pdf_bytes: bytes,
        max_pages: int,
    ) -> list[PDFPage]:
        """Extract text from PDF bytes, preserving page structure.

        Detects scanned PDFs (pages with no extractable text) and skips them.
        """
```

#### 1b. PDF data models

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
    metadata: dict  # PDF metadata (title, author, creation date from PDF properties)
    is_scanned: bool  # True if most pages had no extractable text
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 1c. Integrate PDF extraction into extract_content tool

When `extract_content` is called with a URL that resolves to a PDF (detected by URL pattern `*.pdf`, `arxiv.org/pdf/*`, or Content-Type response header):
1. Route to `PDFExtractProvider` instead of Tavily Extract
2. Convert `PDFExtractionResult` to the standard content format
3. Preserve page boundaries in source metadata for locator support

#### 1d. PDF-aware Unpaywall enrichment (optional dedicated tool)

When profile has `enable_pdf_extraction == True`, add an `extract_pdf` tool:

```
### extract_pdf (systematic-review profile only)
Extract full text from an open-access academic paper PDF.
Arguments: {"url": "PDF URL or DOI", "max_pages": 30}
Returns: Full paper text with page numbers and section structure.

Use when you need the full methods/results/discussion of a paper,
not just the abstract. Only works for open-access papers.
```

**File: `src/foundry_mcp/core/research/document_digest/digestor.py`**

#### 1e. Page-aware digest

When digesting PDF-extracted content:
- Use `page:N:char:start-end` locators for evidence snippets
- Prioritize Methods, Results, and Discussion sections
- Handle academic paper structure (Abstract, Introduction, Methods, Results, Discussion, References)

### Dependencies

- `pymupdf` (fitz) for PDF text extraction
- PLAN-2 item 2 (Unpaywall) for DOI → PDF URL resolution

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Paywall PDFs | Only attempt OA PDFs (from Unpaywall/provider metadata) |
| Large PDFs | Cap at `max_pages` (default 50), skip appendices/supplementary |
| Scanned PDFs | Detect via text-per-page threshold, skip gracefully |
| Publisher rate limiting | Respect robots.txt patterns, add 1s delay between downloads |
| Large file size | Stream download with size limit (default 50MB) |
| Extraction failures | Log in provenance, continue with abstract-only |

### Testing

- Unit test: PDF text extraction from test PDF file
- Unit test: page boundary preservation and char offset calculation
- Unit test: PDF URL detection (arxiv.org/pdf/*, *.pdf extension)
- Unit test: scanned PDF detection (empty text pages)
- Unit test: graceful failure for non-PDF URLs, corrupted files
- Unit test: size limit enforcement
- Integration test: topic researcher extracts PDF and includes in findings

---

## 2. Citation Network / Connected Papers Graph

### Problem

Academic research involves understanding how papers relate to each other through citation chains. A seminal paper can spawn multiple research threads. Understanding these threads is central to a good literature review. Currently, each source is treated as independent — there's no representation of inter-source relationships.

### Architecture

```
After synthesis (post-synthesis enrichment step):
    ↓
For each academic source in state.sources with a paper ID:
    - Fetch references (papers it cites) via OpenAlex / Semantic Scholar
    - Fetch citations (papers that cite it) via OpenAlex / OpenCitations
    ↓
Build citation adjacency graph:
    - Nodes: papers (discovered + referenced)
    - Edges: citation relationships (directed)
    ↓
Identify structure:
    - Papers cited by many discovered papers → "foundational"
    - Papers that cite many discovered papers → "recent extensions"
    - Connected components → "research threads"
    ↓
Output: CitationNetwork in state + structured output
```

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 2a. Citation network model

```python
class CitationNode(BaseModel):
    """A paper in the citation network."""
    paper_id: str
    title: str
    authors: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    is_discovered: bool  # True if in state.sources
    source_id: Optional[str] = None  # Link to ResearchSource if discovered
    role: str = "peripheral"  # "foundational" | "discovered" | "extension" | "peripheral"

class CitationEdge(BaseModel):
    """A directed citation relationship between two papers."""
    citing_paper_id: str
    cited_paper_id: str
    is_self_citation: bool = False  # From OpenCitations metadata

class CitationNetwork(BaseModel):
    """Citation network built from discovered sources."""
    nodes: list[CitationNode]
    edges: list[CitationEdge]
    foundational_papers: list[str]  # paper_ids cited by many discovered papers
    research_threads: list[dict]  # [{name: str, paper_ids: [str], description: str}]
    stats: dict  # {total_nodes, total_edges, discovered_count, foundational_count, ...}
```

#### 2b. Add to state

```python
# In DeepResearchState:
citation_network: Optional[CitationNetwork] = Field(
    default=None,
    description="Citation network built from discovered academic sources",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`** (NEW)

#### 2c. Network builder

```python
class CitationNetworkBuilder:
    """Build citation network from discovered sources.

    Uses OpenAlex (preferred, higher rate limit) or Semantic Scholar
    to fetch references and citations for each discovered academic source.
    Uses OpenCitations for self-citation metadata when available.
    """

    async def build_network(
        self,
        sources: list[ResearchSource],
        providers: dict,  # Available provider instances
        max_references_per_paper: int = 20,
        max_citations_per_paper: int = 20,
        max_concurrent: int = 3,
    ) -> CitationNetwork:
        """Build citation network from discovered sources.

        Steps:
        1. For each source with paper_id/DOI, fetch refs and cites
        2. Build node and edge lists
        3. Identify foundational papers (cited by 3+ discovered papers)
        4. Identify research threads via connected components
        5. Classify roles (foundational, discovered, extension, peripheral)
        """

    def _identify_foundational_papers(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
        discovered_ids: set[str],
    ) -> list[str]:
        """Papers cited by many discovered papers are foundational.

        Threshold: cited by >= 3 discovered papers, or >= 30% of discovered papers
        (whichever is lower).
        """

    def _identify_research_threads(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
    ) -> list[dict]:
        """Find connected components in the citation graph.

        Uses simple BFS/union-find. Each component with >= 3 nodes
        is a research thread.
        """

    def _classify_roles(
        self,
        nodes: list[CitationNode],
        edges: list[CitationEdge],
        discovered_ids: set[str],
    ) -> None:
        """Assign roles based on graph position.

        - foundational: cited by many discovered papers, typically older
        - discovered: in state.sources
        - extension: cites many discovered papers, typically newer
        - peripheral: weakly connected (1 edge to discovered set)
        """
```

#### 2d. Integration point

Call network builder as a post-synthesis enrichment step:

```python
if state.research_profile.enable_citation_network:
    academic_sources = [s for s in state.sources if s.source_type == SourceType.ACADEMIC]
    if len(academic_sources) >= 3:  # Not worth building for tiny sets
        network = await CitationNetworkBuilder().build_network(
            sources=academic_sources,
            providers=available_providers,
        )
        state.citation_network = network
```

Include in structured output for downstream visualization tools.

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| API rate limiting | Use OpenAlex (10 RPS) as primary; Semantic Scholar as fallback. Parallelize within limits. |
| Graph explosion | Cap `max_references_per_paper` and `max_citations_per_paper` |
| Missing paper IDs | Only build for sources with paper_id or DOI in metadata |
| Slow execution | Run as post-synthesis (non-blocking). 15 sources × 2 calls × 10 RPS = ~3s with OpenAlex |

### Testing

- Unit test: network builder with mocked provider responses
- Unit test: foundational paper identification (paper cited by 5+ discovered papers)
- Unit test: research thread detection with known graph
- Unit test: role classification
- Unit test: graceful handling when no academic sources exist
- Unit test: graceful handling when < 3 academic sources (skip network)
- Integration test: 5 academic sources → network with edges

---

## 3. Methodology Quality Assessment

### Problem

Not all studies are equal. An RCT with N=1000 carries more weight than a case study with N=5. Currently, source quality assessment is purely domain-based (is this from a journal or social media?). It doesn't assess the methodological rigor of individual studies.

### Architecture

```
After topic researchers gather findings (during compression or post-synthesis):
    ↓
For each academic source with sufficient content (abstract + any extracted text):
    - LLM extracts methodology metadata
    ↓
Methodology quality scoring (heuristic, not a validated instrument)
    ↓
Output: MethodologyAssessment per source, fed into synthesis prompt
```

### Changes

**File: `src/foundry_mcp/core/research/models/sources.py`**

#### 3a. Methodology assessment model

```python
class StudyDesign(str, Enum):
    """Research study design types, roughly ordered by typical rigor."""
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
    """Assessment of a study's methodological quality.

    Generated by LLM extraction from source content. The rigor_score
    is an approximate heuristic, not a validated psychometric instrument.
    """
    source_id: str
    study_design: StudyDesign = StudyDesign.UNKNOWN
    sample_size: Optional[int] = None
    sample_description: Optional[str] = None
    effect_size: Optional[str] = None        # e.g., "d=0.45", "r=0.32"
    statistical_significance: Optional[str] = None  # e.g., "p<0.001"
    limitations_noted: list[str] = Field(default_factory=list)
    potential_biases: list[str] = Field(default_factory=list)
    rigor_score: float = 0.0                 # 0.0-1.0 composite
    confidence: str = "low"                  # "high" | "medium" | "low"
    content_basis: str = "abstract"          # "abstract" | "full_text"
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`** (NEW)

#### 3b. Assessment engine

```python
class MethodologyAssessor:
    """Assess methodological quality of academic sources.

    Uses LLM to extract study design, sample characteristics,
    and statistical reporting from source content. Computes a
    composite heuristic rigor score.

    Important: confidence is forced to "low" when working from
    abstracts only. Full-text analysis (PLAN-4 item 1) significantly
    improves assessment accuracy.
    """

    async def assess_sources(
        self,
        sources: list[ResearchSource],
        findings: list[ResearchFinding],
        provider_id: Optional[str] = None,
        timeout: float = 60.0,
    ) -> list[MethodologyAssessment]:
        """Assess methodology for academic sources with sufficient content.

        Filters to sources with source_type == ACADEMIC and content > 200 chars.
        Uses a single batched LLM call where possible.
        """

    def _compute_rigor_score(self, assessment: MethodologyAssessment) -> float:
        """Compute composite rigor score from study characteristics.

        Weights:
        - Study design hierarchy: 40% (meta_analysis=1.0 → opinion=0.1)
        - Sample size adequacy: 20% (log-scaled, domain-dependent)
        - Statistical reporting: 20% (effect size + significance reported)
        - Limitation acknowledgment: 10%
        - Bias awareness: 10%

        Score is capped at 0.6 when content_basis == "abstract" to reflect
        the inherent uncertainty of assessing methodology from limited text.
        """
```

#### 3c. LLM extraction prompt

```
You are a research methodology analyst. Given a research source's content,
extract the following methodological details. If information is not present,
use null/empty values — do not guess.

Source content:
{content}

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

#### 3d. Add to state

```python
# In DeepResearchState:
methodology_assessments: list[MethodologyAssessment] = Field(
    default_factory=list,
    description="Methodology quality assessments for academic sources",
)
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 3e. Feed assessments into synthesis

When assessments are available, include in synthesis prompt:

```
## Methodology Quality Assessments
The following sources have been assessed for methodological rigor.
Rigor scores are approximate heuristics, not validated instruments.

[1] Smith et al. (2021) — RCT, N=450, rigor: 0.85/1.0 (from full text)
    Design: Randomized controlled trial
    Effect: d=0.45 (p<0.001)
    Limitations: Single institution, self-report measures

[2] Jones (2023) — Qualitative, N=12, rigor: 0.45/1.0 (from abstract)
    Design: Semi-structured interviews
    Limitations: Small sample, researcher bias possible

When synthesizing findings, consider study rigor when findings conflict.
Note methodological limitations when reporting findings from lower-rigor studies.
Do NOT present rigor scores to the reader — use them to inform your weighting,
not as displayed numbers.
```

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM accuracy from abstracts | Force `confidence="low"` and cap rigor_score at 0.6 for abstract-only |
| Cost (1 LLM call per source) | Use lightweight model. Batch where possible. Only assess sources with ACADEMIC type. |
| False precision | Present scores as "approximate heuristic" in all outputs. Don't display scores in report — use for internal weighting only. |
| Scope creep | Only assess sources with >200 chars of content |

### Testing

- Unit test: `_compute_rigor_score()` with various study designs
- Unit test: rigor_score capped at 0.6 for abstract-only content
- Unit test: LLM extraction prompt parsing with mocked responses
- Unit test: graceful handling for sources without sufficient content
- Unit test: assessment data correctly injected into synthesis prompt
- Integration test: end-to-end assessment of 5 academic sources

---

## 4. Remote MCP Server Bridge

### Problem

Remote academic MCP servers (Scite.ai, Consensus, PubMed) offer unique capabilities — citation sentiment, cross-study agreement, biomedical full-text — but integrating each one as a bespoke provider doesn't scale. A generic MCP-to-provider bridge lets the pipeline call any MCP server uniformly.

### Design

The bridge is a provider adapter that:
1. Connects to a remote MCP server URL
2. Discovers available tools via MCP protocol
3. Exposes them as methods callable from the deep research pipeline
4. Handles serialization, timeouts, and error mapping

This is a foundational pattern — once the bridge exists, adding new MCP servers is configuration, not code.

### Changes

**File: `src/foundry_mcp/core/research/providers/mcp_bridge.py`** (NEW)

#### 4a. Generic MCP bridge

```python
class MCPBridgeProvider:
    """Bridge between deep research pipeline and remote MCP servers.

    Connects to any MCP server URL, discovers tools, and exposes them
    as async methods. Handles MCP protocol details (SSE transport,
    tool discovery, argument serialization, response parsing).
    """

    def __init__(
        self,
        server_url: str,
        server_name: str,
        timeout: float = 30.0,
    ):
        """Initialize bridge with MCP server URL."""

    async def connect(self) -> None:
        """Connect to MCP server and discover available tools."""

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        timeout: Optional[float] = None,
    ) -> dict:
        """Call a tool on the remote MCP server.

        Returns parsed tool response as dict.
        Raises MCPBridgeError on timeout, connection failure, or tool error.
        """

    async def list_tools(self) -> list[dict]:
        """List available tools on the remote server."""

    async def disconnect(self) -> None:
        """Disconnect from MCP server."""
```

#### 4b. Scite.ai integration via bridge

```python
class SciteBridgeProvider:
    """Scite.ai integration via MCP bridge.

    Provides citation sentiment analysis — for each paper, how it has been
    cited: supporting, contrasting, or mentioning. Unique capability
    not available from any other provider.
    """

    def __init__(self, bridge: MCPBridgeProvider):
        self.bridge = bridge

    async def get_citation_sentiment(
        self,
        doi: str,
        max_results: int = 10,
    ) -> dict:
        """Get citation sentiment for a paper.

        Returns: {
            supporting: int,
            contrasting: int,
            mentioning: int,
            total: int,
            citations: [{text, sentiment, citing_doi}]
        }
        """

    async def enrich_source(self, source: ResearchSource) -> ResearchSource:
        """Enrich source metadata with Scite citation sentiment."""
```

#### 4c. Consensus integration via bridge

```python
class ConsensusBridgeProvider:
    """Consensus integration via MCP bridge.

    Provides cross-study agreement analysis on yes/no research questions.
    Most useful for specific sub-questions during the supervision phase.
    """

    async def query(self, question: str) -> dict:
        """Query Consensus for cross-study agreement.

        Returns: {
            agreement: str,  # e.g., "87% yes (23 studies)"
            papers: [{title, authors, year, doi, finding}]
        }
        """
```

#### 4d. PubMed integration note

PubMed's remote MCP server (`https://pubmed.mcp.claude.com/mcp`) is recommended as a parallel MCP server in the user's Claude Code config rather than embedded in the deep research pipeline. For users who want PubMed results in deep research reports, the MCP bridge can connect to it:

```python
# In config:
pubmed_mcp_url: str = "https://pubmed.mcp.claude.com/mcp"
pubmed_mcp_enabled: bool = False  # Opt-in, biomedical only
```

### Integration Points

**Post-gathering enrichment (Scite)**: After topic researchers gather sources, batch-query Scite for citation sentiment on all academic sources with DOIs. Store in `source.metadata`:
```python
metadata["scite_supporting"] = 12
metadata["scite_contrasting"] = 3
metadata["scite_mentioning"] = 45
```

**Supervision phase (Consensus)**: When the supervisor generates directives that can be phrased as yes/no questions, optionally route to Consensus for agreement analysis. Store in `state` metadata for synthesis injection.

### Configuration

```python
# In ResearchConfig:
scite_mcp_url: str = "https://api.scite.ai/mcp"
scite_mcp_enabled: bool = False

consensus_mcp_url: str = "https://mcp.consensus.app/mcp"
consensus_mcp_enabled: bool = False

pubmed_mcp_url: str = "https://pubmed.mcp.claude.com/mcp"
pubmed_mcp_enabled: bool = False
```

### Testing

- Unit test: MCPBridgeProvider connects to mock MCP server
- Unit test: tool discovery returns tool list
- Unit test: tool call with valid arguments returns response
- Unit test: timeout handling for slow MCP servers
- Unit test: SciteBridgeProvider parses sentiment response
- Unit test: ConsensusBridgeProvider parses agreement response
- Integration test: Scite enrichment adds sentiment to source metadata

---

## 5. CORE Open Access Provider

### Why

Full-text fallback for papers not available via Unpaywall or PubMed Central. CORE aggregates 37M+ full-text articles from institutional repositories worldwide — particularly strong for grey literature and institutional outputs that Semantic Scholar/OpenAlex may index as metadata-only.

### API Details

- **Base URL**: `https://api.core.ac.uk/v3`
- **Key endpoints**:
  - `GET /search/works?q=<query>` — full-text search across paper bodies
  - `GET /works/{id}` — single work with full text
- **Auth**: Free (registration for better rate limits)
- **Rate limit**: 5 req/10s (free), higher with registration
- **Returns**: Full text content, metadata, repository info

### Implementation

**File: `src/foundry_mcp/core/research/providers/core_oa.py`** (NEW)

```python
class COREProvider(SearchProvider):
    """CORE API provider for open-access full-text search.

    Searches across 37M+ full-text articles from institutional and subject
    repositories. Particularly valuable for grey literature and institutional
    publications not indexed by Semantic Scholar or OpenAlex.
    """

    BASE_URL = "https://api.core.ac.uk/v3"

    def get_provider_name(self) -> str:
        return "core"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Full-text search across open-access papers."""

    async def get_full_text(self, core_id: str) -> Optional[str]:
        """Retrieve full text for a specific work."""
```

### Configuration

```python
core_api_key: Optional[str] = None   # Optional, for higher rate limits
core_enabled: bool = False           # Opt-in (rate-limited without key)
```

### Testing

- Unit test: search with mocked response
- Unit test: full text retrieval
- Unit test: rate limiting respected (5 req/10s)
- Unit test: graceful handling when CORE is unavailable

---

## Cross-Cutting Considerations

### New Dependencies

- `pymupdf` (fitz) — for PDF text extraction (item 1)
- `mcp` — MCP client library for bridge (item 4) — check if already a dependency

### Configuration Surface

```python
# PDF extraction
deep_research_pdf_max_pages: int = 50
deep_research_pdf_max_size_mb: float = 50.0
deep_research_pdf_timeout: float = 30.0

# Citation network
deep_research_citation_network_max_refs_per_paper: int = 20
deep_research_citation_network_max_cites_per_paper: int = 20

# Methodology assessment
deep_research_methodology_assessment_provider: Optional[str] = None  # Lightweight model
deep_research_methodology_assessment_timeout: float = 60.0

# Remote MCP servers
scite_mcp_url: str = "https://api.scite.ai/mcp"
scite_mcp_enabled: bool = False
consensus_mcp_url: str = "https://mcp.consensus.app/mcp"
consensus_mcp_enabled: bool = False
pubmed_mcp_url: str = "https://pubmed.mcp.claude.com/mcp"
pubmed_mcp_enabled: bool = False

# CORE
core_api_key: Optional[str] = None
core_enabled: bool = False
```

### Performance Impact

| Feature | API Calls | LLM Calls | Added Latency |
|---------|-----------|-----------|---------------|
| PDF extraction | 1 download per PDF | 0 (text extraction only) | 5-15s per paper |
| Citation network | 2 calls per source (OpenAlex) | 0 (graph analysis) | 3-10s total |
| Methodology assessment | 0 | 1 per academic source (batched) | 15-30s total |
| Scite enrichment | 1 MCP call per source with DOI | 0 | 5-15s total |
| Consensus queries | 1 MCP call per yes/no question | 0 | 3-10s per query |
| CORE search | 1 per search query | 0 | 1-3s per query |

### File Impact Summary

| File | Type | Items |
|------|------|-------|
| `providers/pdf_extract.py` | **New** | 1 |
| `phases/topic_research.py` | Modify | 1 (PDF extraction routing) |
| `document_digest/digestor.py` | Modify | 1 (page-aware digest) |
| `phases/citation_network.py` | **New** | 2 |
| `models/deep_research.py` | Modify | 2 (CitationNetwork), 3 (MethodologyAssessment in state) |
| `models/sources.py` | Modify | 3 (StudyDesign, MethodologyAssessment) |
| `phases/methodology_assessment.py` | **New** | 3 |
| `phases/synthesis.py` | Modify | 3 (assessment injection into prompt) |
| `providers/mcp_bridge.py` | **New** | 4 |
| `providers/core_oa.py` | **New** | 5 |
| `config/research.py` | Modify | All (new config fields) |

### Dependency Graph

```
[1. PDF Extraction]─────────────────────────────────────────┐
    (depends on PLAN-2 Unpaywall for DOI→PDF resolution)     │
                                                              │
[2. Citation Network]───────────────────────────────────────┤
    (depends on PLAN-2 OpenAlex/OpenCitations providers)     │
                                                              │
[3. Methodology Assessment]─────────────────────────────────┤
    (independent, but better with item 1 for full-text)      │
                                                              │
[4. MCP Bridge + Scite/Consensus/PubMed]────────────────────┤
    (independent infrastructure)                              │
                                                              │
[5. CORE Provider]──────────────────────────────────────────┘
    (independent)
```

All items are largely independent. Item 3 produces better results when item 1 (PDF extraction) is available (full text vs. abstract-only assessment).
