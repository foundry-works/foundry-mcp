# PLAN-4: Deep Analysis — Full-Text, Citation Networks & Quality Assessment

> **Goal**: Capabilities that make deep research competitive with dedicated academic tools like Elicit, Research Rabbit, and Undermind — full-text PDF analysis, citation network graph construction, and methodology quality assessment.
>
> **Estimated scope**: ~650-900 LOC implementation + ~240-360 LOC tests (reduced from original ~1500-2200 after Feb 2026 tool evaluation)
>
> **Dependencies**: PLAN-0 (prerequisites), PLAN-1 items 1 and 6 (profiles, structured output), PLAN-2 item 1 (OpenAlex)
>
> **Risk**: PDF parsing reliability, LLM accuracy for methodology extraction
>
> **All features in this plan are opt-in** (disabled by default in profiles) to avoid impacting existing behavior.
>
> **Scope revisions from review**:
> - Item 1 (PDF): **Extends existing `pdf_extractor.py`** instead of creating a duplicate module. No new `pymupdf` dependency.
> - Item 2 (Citation Network): **Deferred to explicit user-triggered tool** — not an automatic pipeline step. Uses OpenAlex exclusively (not OpenCitations).
> - Item 3 (Methodology): **Demoted to experimental/opt-in** with strong caveats about reliability.
>
> **Removed in Feb 2026 tool evaluation**:
> - Item 4 (MCP Bridge): Reframed as external documentation. Scite MCP (paid), Consensus MCP (free limited), and PubMed MCP (free) all exist — users configure these directly rather than through built-in bridge wrappers.
> - Item 5 (CORE Provider): Removed. Low throughput (5 req/10s), largely redundant with OpenAlex OA coverage.

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

### Current State — Existing Infrastructure

The existing `pdf_extractor.py` (833 lines) at `src/foundry_mcp/core/research/pdf_extractor.py` is **production-ready** and provides:

- **Libraries**: `pypdf` (primary) with `pdfminer.six` fallback — no new dependencies needed
- **SSRF protection**: URL validation blocking localhost, internal IPs, cloud metadata endpoints, with redirect loop detection (max 5)
- **Security**: Magic byte validation (`%PDF-`), content-type checking, streaming download with size limits (default 10MB)
- **Page extraction**: Page-by-page text extraction with character offset tracking (`page_offsets` list)
- **Graceful fallback**: pypdf → pdfminer.six (per-page) → pdfminer.six (full document)
- **Observability**: Prometheus metrics (`foundry_mcp_pdf_extraction_duration_seconds`, `foundry_mcp_pdf_extraction_pages_total`)
- **Error handling**: Custom exceptions (`InvalidPDFError`, `PDFSecurityError`, `PDFSizeError`, `SSRFError`)

Additional existing infrastructure:
- `DigestPayload` model supports page-based locators (`page:n:char:start-end`)
- Config field `deep_research_digest_fetch_pdfs: bool = False` exists but is disabled
- Semantic Scholar stores `metadata.pdf_url` (OA PDF link) when available

**Decision: Extend the existing `pdf_extractor.py` rather than creating a parallel module.** No new `pymupdf` dependency.

### Architecture

```
Topic Researcher discovers paper via Semantic Scholar / OpenAlex
    ↓
Paper has DOI → OpenAlex oa_url OR Unpaywall resolves to OA PDF URL
    ↓ (or direct pdf_url from provider metadata)
Existing PDFExtractor.extract_from_url():
    1. SSRF-validated download (already implemented)
    2. Extract text via pypdf + pdfminer fallback (already implemented)
    3. Return PDFExtractionResult with page boundaries (already implemented)
    ↓
Standard summarization/digest pipeline processes extracted text
    ↓
Evidence snippets with page-based locators (page:3:char:150-320)
```

### Changes

**File: `src/foundry_mcp/core/research/pdf_extractor.py`** (EXTEND — not new)

#### 1a. Add academic paper section detection

```python
def detect_sections(self, result: PDFExtractionResult) -> dict[str, tuple[int, int]]:
    """Detect standard academic paper sections from extracted text.

    Returns section name → (start_char, end_char) mapping for:
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
- Use `page:N:char:start-end` locators for evidence snippets (existing `page_offsets` from `PDFExtractionResult`)
- Use `detect_sections()` to prioritize Methods, Results, and Discussion
- Handle academic paper structure (Abstract, Introduction, Methods, Results, Discussion, References)

### Dependencies

- **No new dependencies.** Uses existing `pypdf` + `pdfminer.six` already in the dependency tree.
- PLAN-2 item 1 (OpenAlex `oa_url`) for DOI → PDF URL resolution. Optionally enhanced by Unpaywall (Tier 2).

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Paywall PDFs | Only attempt OA PDFs (from OpenAlex oa_url / Unpaywall / provider metadata) |
| Large PDFs | Existing `max_pages` config (default 500, override to 50 for deep research) |
| Scanned PDFs | Existing detection via text-per-page threshold in `PDFExtractor` |
| Publisher rate limiting | Respect robots.txt patterns, add 1s delay between downloads |
| Large file size | Existing streaming download with size limit (default 10MB, configurable) |
| Extraction failures | Log in provenance, continue with abstract-only |

### Testing

- Unit test: section detection from synthetic academic PDF text
- Unit test: prioritized extraction respects max_chars and section ordering
- Unit test: PDF URL routing in extract_content tool
- Unit test: graceful fallback when section detection finds nothing
- Existing tests for PDFExtractor core functionality remain unchanged
- Integration test: topic researcher extracts PDF and includes in findings
- ~80-120 LOC new tests

---

## 2. Citation Network / Connected Papers Graph (User-Triggered)

### Problem

Academic research involves understanding how papers relate to each other through citation chains. A seminal paper can spawn multiple research threads. Understanding these threads is central to a good literature review. Currently, each source is treated as independent — there's no representation of inter-source relationships.

### Scope Revision

**Changed from automatic post-synthesis step to explicit user-triggered tool.** Building a citation network requires 2 API calls per source (references + citations). For 15 sources, that's 30+ API calls returning potentially 600+ nodes. This is expensive and slow — not appropriate as an automatic pipeline step.

Instead, citation network building is exposed as a dedicated `deep-research-network` action that users invoke on a completed research session when they specifically need citation graph analysis.

### Architecture

```
User completes deep research session (produces report + sources)
    ↓
User explicitly requests: research action=deep-research-network research_id=<id>
    ↓
For each academic source in state.sources with a paper ID:
    - Fetch references (papers it cites) via OpenAlex (100 req/s)
    - Fetch citations (papers that cite it) via OpenAlex
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
Output: CitationNetwork in structured output
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

class CitationNetwork(BaseModel):
    """Citation network built from discovered sources."""
    nodes: list[CitationNode]
    edges: list[CitationEdge]
    foundational_papers: list[str]  # paper_ids cited by many discovered papers
    research_threads: list[dict]  # [{name: str, paper_ids: [str], description: str}]
    stats: dict  # {total_nodes, total_edges, discovered_count, foundational_count, ...}
```

#### 2b. Add to ResearchExtensions

```python
# In ResearchExtensions (PLAN-0 item 2):
citation_network: Optional[CitationNetwork] = None

# Convenience accessor on DeepResearchState:
@property
def citation_network(self) -> Optional[CitationNetwork]:
    return self.extensions.citation_network
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`** (NEW)

#### 2c. Network builder

```python
class CitationNetworkBuilder:
    """Build citation network from discovered sources.

    Uses OpenAlex (primary, 100 req/s) or Semantic Scholar (fallback, 1 RPS)
    to fetch references and citations for each discovered academic source.
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

#### 2d. Integration point — dedicated action handler

**File: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`**

Expose as a dedicated action, not an automatic pipeline step:

```python
def _handle_deep_research_network(
    research_id: str,
    max_references_per_paper: int = 20,
    max_citations_per_paper: int = 20,
) -> dict:
    """Build citation network for a completed research session.

    User-triggered — not automatic. Requires a completed session.
    Returns network graph data for visualization tools.
    """
    state = load_state(research_id)
    academic_sources = [s for s in state.sources if s.source_type == SourceType.ACADEMIC]
    if len(academic_sources) < 3:
        return {"status": "skipped", "reason": "fewer than 3 academic sources"}

    network = await CitationNetworkBuilder().build_network(
        sources=academic_sources,
        providers=available_providers,
        max_references_per_paper=max_references_per_paper,
        max_citations_per_paper=max_citations_per_paper,
    )
    state.extensions.citation_network = network
    save_state(state)
    return network.model_dump()
```

Wire up `"deep-research-network"` action in the research action router.

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| API rate limiting | Use OpenAlex (100 req/s) as primary; Semantic Scholar (1 RPS) as fallback. Parallelize within limits. |
| Graph explosion | Cap `max_references_per_paper` and `max_citations_per_paper` (default 20 each) |
| Missing paper IDs | Only build for sources with paper_id or DOI in metadata |
| Execution time | User-triggered, so latency is expected. 15 sources × 2 calls at 100 req/s = <1s with OpenAlex. Progress tracking via action response. |
| OpenAlex budget | 30 API calls cost ~$0.003 (list queries) — negligible against $1/day free budget |
| Cost visibility | Action response includes API call count and elapsed time |

### Testing

- Unit test: network builder with mocked provider responses
- Unit test: foundational paper identification (paper cited by 5+ discovered papers)
- Unit test: research thread detection with known graph
- Unit test: role classification
- Unit test: graceful handling when no academic sources exist
- Unit test: graceful handling when < 3 academic sources (returns skipped)
- Unit test: action handler wiring and response format
- Integration test: 5 academic sources → network with edges
- ~100-140 LOC new tests

---

## 3. Methodology Quality Assessment (Experimental — Strong Caveats)

> **Status: Experimental.** This feature produces approximate heuristics, not validated psychometric instruments. The numeric rigor score is fundamentally unreliable when derived from abstracts alone (~80% of cases). It should be treated as a rough sorting signal for internal synthesis weighting, never as a displayed metric or a basis for excluding studies.
>
> **Recommendation**: Implement the study design classification and metadata extraction (useful, relatively reliable) but **do not compute or store a numeric rigor score**. Instead, provide the extracted metadata to the synthesis LLM and let it make qualitative judgments.

### Problem

Not all studies are equal. An RCT with N=1000 carries more weight than a case study with N=5. Currently, source quality assessment is purely domain-based (is this from a journal or social media?). It doesn't assess the methodological characteristics of individual studies.

### Architecture (Revised — No Numeric Score)

```
After topic researchers gather findings (during compression or post-synthesis):
    ↓
For each academic source with sufficient content (abstract + any extracted text):
    - LLM extracts methodology metadata (study design, sample, effect size)
    ↓
Structured methodology metadata (NO numeric score)
    ↓
Output: MethodologyAssessment per source, fed into synthesis prompt as context
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
    """Structured methodology metadata extracted from source content.

    Generated by LLM extraction. Provides qualitative metadata for
    synthesis weighting — NO numeric rigor score. The synthesis LLM
    uses this context to make informed qualitative judgments.
    """
    source_id: str
    study_design: StudyDesign = StudyDesign.UNKNOWN
    sample_size: Optional[int] = None
    sample_description: Optional[str] = None
    effect_size: Optional[str] = None        # e.g., "d=0.45", "r=0.32"
    statistical_significance: Optional[str] = None  # e.g., "p<0.001"
    limitations_noted: list[str] = Field(default_factory=list)
    potential_biases: list[str] = Field(default_factory=list)
    # NO rigor_score — removed per review. Numeric scoring from abstracts
    # is unreliable and invites misuse. Qualitative metadata is sufficient.
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
        """Extract methodology metadata for academic sources with sufficient content.

        Filters to sources with source_type == ACADEMIC and content > 200 chars.
        Uses a single batched LLM call where possible.
        Returns structured metadata — no numeric scoring.
        """
```

**Removed**: `_compute_rigor_score()` — numeric scoring from abstracts is unreliable. The synthesis LLM receives the structured metadata and makes qualitative weighting decisions.

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

#### 3d. Add to ResearchExtensions

```python
# In ResearchExtensions (PLAN-0 item 2):
methodology_assessments: list[MethodologyAssessment] = Field(default_factory=list)

# Convenience accessor on DeepResearchState:
@property
def methodology_assessments(self) -> list[MethodologyAssessment]:
    return self.extensions.methodology_assessments
```

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`**

#### 3e. Feed assessments into synthesis

When assessments are available, include in synthesis prompt:

```
## Methodology Context
The following sources have methodology metadata extracted.
Use this context to inform your qualitative weighting when findings conflict.

[1] Smith et al. (2021) — Randomized controlled trial, N=450
    Effect: d=0.45 (p<0.001)
    Limitations: Single institution, self-report measures
    (extracted from: full text)

[2] Jones (2023) — Semi-structured interviews, N=12
    Limitations: Small sample, researcher bias possible
    (extracted from: abstract — treat with lower confidence)

When synthesizing findings, consider study design and sample size when
findings conflict. Note methodological limitations when appropriate.
Present the methodology context naturally in your synthesis — do not
create a separate "methodology scores" section.
```

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM accuracy from abstracts | Force `confidence="low"` for abstract-only. Metadata extraction is more reliable than numeric scoring. |
| Cost (1 LLM call per source) | Use lightweight model. Batch where possible. Only assess sources with ACADEMIC type. |
| Over-reliance on extracted metadata | Synthesis prompt frames metadata as "context" not "ground truth". LLM makes final judgment. |
| Scope creep | Only assess sources with >200 chars of content |

### Testing

- Unit test: LLM extraction prompt parsing with mocked responses
- Unit test: StudyDesign classification from various abstracts
- Unit test: graceful handling for sources without sufficient content
- Unit test: assessment data correctly injected into synthesis prompt
- Unit test: confidence forced to "low" for abstract-only content
- Integration test: end-to-end assessment of 5 academic sources
- ~80-100 LOC new tests

---

## ~~4. Remote MCP Server Bridge~~ — REMOVED (Reframed as External Documentation)

> **Status: Removed from implementation scope (Feb 2026 tool evaluation)**
>
> **Rationale**: All three target MCP servers now exist and have been validated:
> - **Scite.ai MCP** (scite.ai/mcp) — launched Feb 26, 2026. Official, OAuth 2.0 auth. **Requires paid Scite Premium subscription (~16 EUR/mo).** Not viable as a default built-in dependency.
> - **Consensus MCP** (consensus.app/home/mcp/) — production-ready, official. Free tier with 3 results/query. Worth recommending but not embedding.
> - **PubMed MCP** — multiple community implementations exist. Best: cyanheads/pubmed-mcp-server (TypeScript, production-grade, dual transport). Fully free with NCBI API key.
>
> **Decision**: Rather than building bespoke bridge wrappers (SciteBridgeProvider, ConsensusBridgeProvider), document these as **external MCP servers users configure directly** in their Claude Code / MCP client settings. This avoids:
> 1. Coupling to a paid service (Scite) as a default dependency
> 2. Maintaining wrapper code for rapidly evolving external APIs
> 3. Adding ~300-400 LOC of bridge infrastructure with limited value over direct MCP configuration
>
> **Action**: Add a "Recommended External MCP Servers" section to the deep research documentation with configuration examples for each server.
>
> **Also notable**: AI2's Asta project provides an official "Scientific Corpus Tool" — an MCP extension of the Semantic Scholar API with sparse + dense full-text semantic search across OA papers. This is another strong external MCP recommendation.

---

## ~~5. CORE Open Access Provider~~ — REMOVED

> **Status: Removed (Feb 2026 tool evaluation)**
>
> **Rationale**: CORE has grown to 46M hosted full texts (323M linked), but its rate limit of **5 req/10s** makes it impractical for interactive pipeline use. OpenAlex now provides OA PDF URLs (via its integrated Unpaywall engine) for a comparable set of papers at 100 req/s. The unique value of CORE (grey literature, institutional repository content) is real but niche, and doesn't justify another API dependency with severe rate constraints.
>
> **Alternative for users who need CORE**: Multiple community-built CORE MCP servers exist and can be configured as external MCP servers.

---

## Cross-Cutting Considerations

### New Dependencies

- **No new dependencies.** Item 1 (PDF) uses existing `pypdf` + `pdfminer.six`. Items 4 (MCP Bridge) and 5 (CORE) have been removed.

### Configuration Surface

```python
# PDF extraction (extends existing PDFExtractor config)
deep_research_pdf_max_pages: int = 50          # Override default 500 for deep research
deep_research_pdf_priority_sections: list[str] = ["methods", "results", "discussion"]

# Citation network (user-triggered, not automatic)
deep_research_citation_network_max_refs_per_paper: int = 20
deep_research_citation_network_max_cites_per_paper: int = 20

# Methodology assessment (experimental)
deep_research_methodology_assessment_provider: Optional[str] = None  # Lightweight model
deep_research_methodology_assessment_timeout: float = 60.0
```

### Performance Impact

| Feature | API Calls | LLM Calls | Added Latency | Trigger |
|---------|-----------|-----------|---------------|---------|
| PDF extraction | 1 download per PDF | 0 (text extraction only) | 5-15s per paper | Automatic (profile-gated) |
| Citation network | 2 calls per source (OpenAlex, 100 req/s) | 0 (graph analysis) | <1s for 15 sources | **User-triggered** |
| Methodology assessment | 0 | 1 per academic source (batched) | 15-30s total | Automatic (experimental, profile-gated) |

### Testing Budget

| Item | Impl LOC | Test LOC | Test Focus |
|------|----------|----------|------------|
| 1. PDF Analysis (extend) | ~150-200 | ~80-120 | Section detection, prioritization, routing |
| 2. Citation Network | ~300-400 | ~80-120 | Graph building, role classification, action handler |
| 3. Methodology (experimental) | ~200-300 | ~80-120 | LLM extraction parsing, synthesis injection |
| **Total** | **~650-900** | **~240-360** | |

### File Impact Summary

| File | Type | Items |
|------|------|-------|
| `pdf_extractor.py` | **Extend** (not new!) | 1 (section detection, prioritized extraction) |
| `phases/topic_research.py` | Modify | 1 (PDF extraction routing) |
| `document_digest/digestor.py` | Modify | 1 (page-aware digest) |
| `phases/citation_network.py` | **New** | 2 |
| `models/deep_research.py` | Modify | 2 (CitationNetwork), 3 (MethodologyAssessment) — via ResearchExtensions |
| `models/sources.py` | Modify | 3 (StudyDesign, MethodologyAssessment) |
| `phases/methodology_assessment.py` | **New** | 3 |
| `phases/synthesis.py` | Modify | 3 (assessment injection into prompt) |
| `handlers_deep_research.py` | Modify | 2 (network action) |
| `config/research.py` | Modify | All (new config fields) |

### Dependency Graph (Revised)

```
[1. PDF Analysis]───────────────────────────────────────────┐
    Extends existing pdf_extractor.py                        │
    Enhanced by: PLAN-2 item 1 (OpenAlex oa_url)             │
                                                              │
[2. Citation Network] (USER-TRIGGERED)──────────────────────┤
    Depends on: PLAN-2 item 1 (OpenAlex)                     │
    Uses OpenAlex exclusively (100 req/s)                    │
                                                              │
[3. Methodology Assessment] (EXPERIMENTAL)──────────────────┘
    Independent. Better with item 1 for full-text.
    No numeric scoring — qualitative metadata only.
```

All three items can proceed independently. Item 2 is user-triggered (not automatic).
