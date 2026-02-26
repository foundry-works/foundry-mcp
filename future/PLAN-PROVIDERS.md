# PLAN-PROVIDERS: Academic Research Provider Expansion

> **Goal**: Expand deep research's academic capabilities by integrating free academic APIs and remote MCP servers that complement the existing Semantic Scholar provider.
>
> **Relationship to other plans**: This is a cross-cutting concern — providers feed into Tier 1 (citation tools, APA formatting), Tier 2 (influence ranking, landscape metadata), and Tier 3 (PDF analysis, citation networks).

---

## Current State

The deep research pipeline currently uses these providers (`src/foundry_mcp/core/research/providers/`):

| Provider | Type | Purpose |
|----------|------|---------|
| Tavily Search | Web search | General web discovery |
| Tavily Extract | Web extraction | Full page content |
| Google Search | Web search | General web discovery |
| Semantic Scholar | Academic API | Paper search, abstracts, citation counts |
| Perplexity | AI search | Synthesized web answers |

Semantic Scholar is the only academic-specific provider. It covers paper search and basic metadata but lacks full-text access, citation graph traversal, cross-study synthesis, and citation sentiment analysis.

---

## Proposed Provider Architecture

### Layer 1: Discovery (find papers)

| Provider | Type | Index Size | Key Strength | Auth | Rate Limit |
|----------|------|-----------|--------------|------|------------|
| Semantic Scholar | Raw API (existing) | 200M+ | AI TLDRs, influence scores | Optional key | 1 RPS |
| **OpenAlex** | **Raw API (new)** | **450M+** | **Broadest coverage, topics, institutions, funders, snowball search** | **Free key (email)** | **100K/day** |
| **Consensus** | **Remote MCP (new)** | **200M+** | **Cross-study agreement meter** | **Free, no auth** | **3 results default** |
| **PubMed** | **Remote MCP (new)** | **36M+** | **Biomedical full-text, MeSH terms** | **Free, no auth** | **3 RPS (NCBI)** |
| Tavily/Google | Web search (existing) | Web-scale | General fallback | Key required | 60/min |

### Layer 2: Citation Intelligence (understand relationships)

| Provider | Type | Index Size | Key Strength | Auth | Rate Limit |
|----------|------|-----------|--------------|------|------------|
| **Scite.ai** | **Remote MCP (new)** | **1.6B+ citations** | **Smart Citations: supporting/contrasting/mentioning** | **Free (OA data)** | **Unknown** |
| **OpenCitations** | **Raw API (new)** | **2B+ DOI links** | **Complete open citation graph, self-citation detection** | **Free, no key** | **Unlimited** |
| Semantic Scholar | Raw API (existing) | 200M+ | Citation counts, influential citations | Optional key | 1 RPS |
| **OpenAlex** | **Raw API (new)** | **2B+ edges** | **`referenced_works`, `cited_by`, `related_works` on every record** | **Free key** | **100K/day** |

### Layer 3: Full-Text Access (get paper content)

| Provider | Type | Coverage | Key Strength | Auth | Rate Limit |
|----------|------|----------|--------------|------|------------|
| **Unpaywall** | **Raw API (new)** | **DOI→PDF mapping** | **Resolve DOI to free PDF URL instantly** | **Free (email)** | **100K/day** |
| **PubMed Central** | **Via PubMed MCP** | **8M+ full-text** | **Authoritative biomedical full-text** | **Free** | **3 RPS** |
| **CORE** | **Raw API (new)** | **37M+ full-text** | **Largest OA full-text aggregator, pre-extracted text** | **Free (registration)** | **5 req/10s** |
| **OpenAlex** | **Raw API (new)** | **60M linked PDFs** | **PDF download links on records** | **Free key** | **100K/day** |
| Tavily Extract | Web extraction (existing) | Web pages | HTML→markdown conversion | Key required | 60/min |

### Layer 4: Metadata Enrichment (improve citation data)

| Provider | Type | Coverage | Key Strength | Auth | Rate Limit |
|----------|------|----------|--------------|------|------------|
| **Crossref** | **Raw API (new)** | **165M+ DOIs** | **Authoritative bibliographic metadata (journal, volume, pages)** | **Free (polite pool w/ email)** | **50 RPS polite** |
| Semantic Scholar | Raw API (existing) | 200M+ | Authors, venue, year, DOI | Optional key | 1 RPS |

---

## Priority 1: OpenAlex Provider

### Why First

OpenAlex is the single most impactful addition. It covers all four layers simultaneously (discovery, citations, full-text links, metadata), has the broadest index (450M+ works), and its API is designed for agent consumption.

### Capabilities

- **Work search**: Full-text search with filters by concept, institution, funder, author, year, OA status, citation count, type
- **Topic taxonomy**: Hierarchical topic classification (Level 0-3) — "find all papers on conversation-based assessment" semantically
- **Citation graph**: Every work object includes `referenced_works` (outgoing) and `cited_by_api_url` (incoming) — no separate endpoint needed
- **Related works**: `related_works` field on every record
- **Snowball search**: Forward/backward traversal from a seed set (find citing AND cited papers)
- **Aboutness endpoint**: `POST /text` — send arbitrary text, get back topic classifications
- **Institution data**: Filter by institution, get affiliation networks
- **Funder data**: Filter by funder, understand research funding landscape
- **Full-text PDFs**: 60M records have direct PDF links via Unpaywall integration
- **Author disambiguation**: Robust ORCID-based dedup with profile pages

### API Details

- **Base URL**: `https://api.openalex.org`
- **Auth**: Free API key (email-based, 100K credits/day). Polite pool (faster) with email parameter.
- **Format**: JSON, clean schema
- **License**: CC0 (completely open)
- **Key endpoints**:
  - `GET /works?search=<query>&filter=<filters>` — search with 50+ filter dimensions
  - `GET /works/<id>` — single work with full metadata
  - `GET /works?filter=cited_by:<id>` — papers citing a given work
  - `GET /works?filter=cites:<id>` — papers cited by a given work
  - `POST /text` — topic classification from arbitrary text
  - `GET /authors/<id>` — author profile with works
  - `GET /institutions/<id>` — institution profile

### Implementation

**File: `src/foundry_mcp/core/research/providers/openalex.py`** (NEW)

```python
class OpenAlexProvider(SearchProvider):
    """OpenAlex API provider for comprehensive academic search.

    Wraps the OpenAlex REST API to provide:
    - Paper search with rich filtering (topic, institution, funder, year, OA status)
    - Citation graph traversal (cited_by, cites, related_works)
    - Topic classification via /text endpoint
    - Author and institution lookup
    """

    BASE_URL = "https://api.openalex.org"

    async def search(self, query: str, max_results: int = 10, **kwargs) -> list[ResearchSource]:
        """Search works with optional filters."""

    async def get_citations(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get papers that cite a given work."""

    async def get_references(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get papers cited by a given work."""

    async def get_related(self, work_id: str, max_results: int = 10) -> list[ResearchSource]:
        """Get related works."""

    async def classify_text(self, text: str) -> list[dict]:
        """Classify arbitrary text into OpenAlex topics."""

    async def search_by_topic(self, topic_id: str, max_results: int = 20, **kwargs) -> list[ResearchSource]:
        """Search works by OpenAlex topic ID."""
```

**Metadata mapping** (OpenAlex → ResearchSource.metadata):
- `id` → `openalex_id`
- `doi` → `doi`
- `title` → `title`
- `authorships[].author.display_name` → `authors`
- `cited_by_count` → `citation_count`
- `publication_year` → `year`
- `primary_location.source.display_name` → `venue`
- `topics[0].display_name` → `primary_topic`
- `open_access.oa_url` → `pdf_url`
- `type` → `publication_type` (article, book, dataset, etc.)
- `grants[].funder.display_name` → `funders`
- `authorships[].institutions[].display_name` → `institutions`

---

## Priority 2: Unpaywall Provider

### Why Second

Trivially simple (single endpoint), directly unlocks PDF access for the Tier 3 "Full-Text PDF Analysis" feature. One DOI in, PDF URL out.

### API Details

- **Endpoint**: `GET https://api.unpaywall.org/v2/{DOI}?email=<email>`
- **Auth**: Free, just needs email address
- **Rate limit**: 100K/day
- **Returns**: `best_oa_location.url_for_pdf`, all OA locations, license info

### Implementation

**File: `src/foundry_mcp/core/research/providers/unpaywall.py`** (NEW)

```python
class UnpaywallProvider:
    """Unpaywall API provider for resolving DOIs to open-access PDF URLs.

    Given a DOI, returns the best available open-access PDF URL.
    Part of the OpenAlex/OurResearch ecosystem.
    """

    async def resolve_pdf(self, doi: str) -> Optional[str]:
        """Resolve a DOI to an open-access PDF URL. Returns None if no OA copy found."""

    async def get_oa_status(self, doi: str) -> dict:
        """Get full OA status including all locations, license, and OA color."""
```

---

## Priority 3: Scite.ai (Remote MCP)

### Why Third

Smart Citations (supporting/contrasting/mentioning) is a unique capability no other provider offers. This directly enables the "Key Debates & Contradictions" section in literature reviews.

### Integration Approach: MCP-to-MCP Delegation

Rather than wrapping Scite's internal API, connect to their remote MCP server at `https://api.scite.ai/mcp` and invoke it from within the deep research pipeline.

**Option A — Topic researcher tool**: Add a `check_citation_sentiment` tool to the topic researcher's toolkit that calls Scite's MCP endpoint:

```
### check_citation_sentiment (academic mode only)
Check how a paper has been cited — supporting, contrasting, or mentioning.
Arguments: {"doi": "10.1234/example", "max_results": 10}
Returns: Citation statements classified as supporting/contrasting/mentioning.
```

**Option B — Post-gathering enrichment**: After topic researchers gather sources, batch-query Scite for citation sentiment on all academic sources with DOIs. Store sentiment data in `ResearchSource.metadata`:

```python
metadata["scite_supporting"] = 12
metadata["scite_contrasting"] = 3
metadata["scite_mentioning"] = 45
metadata["scite_total"] = 60
```

**Recommendation**: Option B — it's less disruptive to the topic researcher loop and provides systematic coverage.

### Remote MCP Configuration

```python
# In ResearchConfig:
scite_mcp_url: str = "https://api.scite.ai/mcp"
scite_mcp_enabled: bool = False  # Opt-in
```

---

## Priority 4: OpenCitations (COCI)

### Why Fourth

Complete open citation graph with 2B+ DOI-to-DOI links. Unlike Semantic Scholar (which gives citation counts and limited lists), OpenCitations provides the full wiring plus metadata about each citation link (date, self-citation flags).

### API Details

- **Base URL**: `https://opencitations.net/index/api/v2`
- **Key endpoints**:
  - `GET /citations/{doi}` — papers citing a given DOI
  - `GET /references/{doi}` — papers cited by a given DOI
  - `GET /citation/{oci}` — single citation metadata by Open Citation Identifier
- **Returns per citation**: citing DOI, cited DOI, creation date, timespan, journal self-citation flag, author self-citation flag
- **Auth**: Free, no key required
- **Rate limit**: No documented limit
- **License**: CC0

### Implementation

**File: `src/foundry_mcp/core/research/providers/opencitations.py`** (NEW)

```python
class OpenCitationsProvider:
    """OpenCitations COCI provider for DOI-to-DOI citation graph.

    Provides complete citation traversal with metadata about
    each citation link (date, timespan, self-citation flags).
    """

    async def get_citations(self, doi: str) -> list[dict]:
        """Get all papers citing a given DOI."""

    async def get_references(self, doi: str) -> list[dict]:
        """Get all papers referenced by a given DOI."""

    async def get_citation_metadata(self, citing_doi: str, cited_doi: str) -> Optional[dict]:
        """Get metadata about a specific citation link."""
```

---

## Priority 5: Consensus (Remote MCP)

### Why Fifth

The Consensus Meter (cross-study agreement on yes/no questions) is unique and valuable for lit reviews. However, it returns limited results (3 by default) and its index overlaps with Semantic Scholar.

### Integration Approach

Best used for specific sub-questions during the supervision phase, not as a primary discovery tool.

**Integration point**: When the supervisor generates directives that can be phrased as yes/no questions ("Does conversation-based assessment improve learning outcomes?"), route those to Consensus for agreement analysis.

```python
# In ResearchConfig:
consensus_mcp_url: str = "https://mcp.consensus.app/mcp"
consensus_mcp_enabled: bool = False  # Opt-in
```

Store Consensus Meter results in state metadata for synthesis:

```python
state.metadata["consensus_results"] = [
    {
        "question": "Does CBA improve formative assessment?",
        "agreement": "87% yes (23 studies)",
        "top_papers": [...]
    }
]
```

Inject into synthesis prompt for the "Key Findings" section.

---

## Priority 6: Crossref

### Why Sixth

Authoritative metadata enrichment. When Semantic Scholar or OpenAlex metadata is incomplete (missing volume, issue, pages for APA formatting), Crossref is the definitive fallback.

### API Details

- **Base URL**: `https://api.crossref.org`
- **Key endpoint**: `GET /works/{doi}` — full bibliographic metadata
- **Auth**: Free (polite pool with `mailto:` header)
- **Rate limit**: 50 RPS in polite pool
- **Returns**: Title, authors, journal, volume, issue, pages, publisher, license, funder, references, type

### Implementation

**File: `src/foundry_mcp/core/research/providers/crossref.py`** (NEW)

```python
class CrossrefProvider:
    """Crossref API provider for authoritative bibliographic metadata.

    Used as a metadata enrichment fallback when primary providers
    (Semantic Scholar, OpenAlex) have incomplete records.
    """

    async def get_work(self, doi: str) -> Optional[dict]:
        """Fetch full bibliographic metadata for a DOI."""

    async def enrich_source(self, source: ResearchSource) -> ResearchSource:
        """Enrich a ResearchSource with Crossref metadata (fills missing fields)."""
```

---

## Priority 7: CORE

### Why Seventh

Full-text fallback for papers not available via Unpaywall or PubMed Central. Particularly strong for institutional repository content.

### API Details

- **Base URL**: `https://api.core.ac.uk/v3`
- **Key endpoints**:
  - `GET /search/works?q=<query>` — full-text search across paper bodies
  - `GET /works/{id}` — single work with full text
- **Auth**: Free (registration recommended for better rate limits)
- **Rate limit**: 5 req/10s (free), higher with registration
- **Returns**: Full text content, metadata, repository info

### Implementation

**File: `src/foundry_mcp/core/research/providers/core_oa.py`** (NEW)

```python
class COREProvider(SearchProvider):
    """CORE API provider for open-access full-text search.

    Searches across 37M+ full-text articles from institutional
    and subject repositories worldwide.
    """

    async def search(self, query: str, max_results: int = 10, **kwargs) -> list[ResearchSource]:
        """Full-text search across open-access papers."""

    async def get_full_text(self, core_id: str) -> Optional[str]:
        """Retrieve full text for a specific work."""
```

---

## Priority 8: PubMed (Remote MCP)

### Why Last

Domain-specific (biomedical/life sciences). Extremely valuable when the research topic intersects health/medicine, but not a general academic tool. Anthropic already hosts the MCP server, so no implementation needed — just configuration guidance.

### Integration Approach

Recommend as a **parallel MCP server** configured alongside foundry-mcp, not embedded in the pipeline. Users doing biomedical research add it to their Claude Code config:

```json
{
  "mcpServers": {
    "pubmed": {
      "url": "https://pubmed.mcp.claude.com/mcp"
    }
  }
}
```

For deep research integration, add optional PubMed provider wrapping NCBI E-utilities for users who want PubMed results in their deep research reports:

```python
# In ResearchConfig:
pubmed_enabled: bool = False  # Opt-in, biomedical research only
pubmed_mcp_url: str = "https://pubmed.mcp.claude.com/mcp"
```

---

## Configuration Surface

### New config fields in `src/foundry_mcp/config/research.py`:

```python
# OpenAlex
openalex_api_key: Optional[str] = None  # Free key from openalex.org
openalex_email: Optional[str] = None     # For polite pool (faster responses)
openalex_enabled: bool = True            # On by default when key present

# Unpaywall
unpaywall_email: Optional[str] = None    # Required (their only auth)
unpaywall_enabled: bool = True           # On by default when email present

# OpenCitations
opencitations_enabled: bool = True       # Free, no key needed

# Crossref
crossref_email: Optional[str] = None     # For polite pool
crossref_enabled: bool = True            # Free, on by default

# CORE
core_api_key: Optional[str] = None       # Optional, for higher rate limits
core_enabled: bool = False               # Opt-in (rate-limited without key)

# Remote MCP servers
scite_mcp_enabled: bool = False          # Opt-in
scite_mcp_url: str = "https://api.scite.ai/mcp"

consensus_mcp_enabled: bool = False      # Opt-in
consensus_mcp_url: str = "https://mcp.consensus.app/mcp"

pubmed_mcp_enabled: bool = False         # Opt-in, biomedical only
pubmed_mcp_url: str = "https://pubmed.mcp.claude.com/mcp"
```

### Provider selection for ACADEMIC mode

When `research_mode == "academic"`, the default provider chain becomes:

```python
deep_research_providers = [
    "semantic_scholar",  # Existing — TLDRs, influence scores
    "openalex",          # NEW — broadest coverage, topics, institutions
    "tavily",            # Existing — web fallback
]
```

Optional additions enabled per-config:
- `"scite"` — citation sentiment enrichment
- `"consensus"` — cross-study agreement
- `"core"` — full-text fallback
- `"pubmed"` — biomedical research

---

## File Impact Summary

| File | Type | Priority |
|------|------|----------|
| `providers/openalex.py` | **New** | P1 |
| `providers/unpaywall.py` | **New** | P2 |
| `providers/opencitations.py` | **New** | P4 |
| `providers/crossref.py` | **New** | P6 |
| `providers/core_oa.py` | **New** | P7 |
| `providers/mcp_bridge.py` | **New** | P3/P5/P8 (generic MCP-to-provider bridge) |
| `config/research.py` | Modify | All (new config fields) |
| `config/research_sub_configs.py` | Modify | All (new sub-config dataclasses) |
| `phases/topic_research.py` | Modify | P1, P3 (new tools for academic mode) |
| `phases/supervision.py` | Modify | P5 (Consensus integration for yes/no questions) |
| `phases/_citation_postprocess.py` | Modify | P2, P6 (metadata enrichment for APA) |
| `models/sources.py` | Modify | P3 (scite sentiment fields in metadata) |

---

## Relationship to Tier Plans

| Provider | Tier 1 | Tier 2 | Tier 3 |
|----------|--------|--------|--------|
| **OpenAlex** | Citation graph tools (#3), APA metadata (#2) | Landscape metadata (#7), Influence ranking (#6) | Citation network (#11) |
| **Unpaywall** | — | — | PDF analysis (#10) |
| **Scite.ai** | — | Contradictions section (#8) | Methodology assessment context (#12) |
| **OpenCitations** | — | — | Citation network (#11) |
| **Consensus** | — | Cross-study tables (#9) | — |
| **Crossref** | APA formatting (#2) | — | — |
| **CORE** | — | — | PDF analysis fallback (#10) |
| **PubMed** | — | — | Biomedical full-text (#10) |
