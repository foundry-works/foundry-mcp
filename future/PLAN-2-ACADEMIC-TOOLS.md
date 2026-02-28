# PLAN-2: Academic Tools & Provider Expansion

> **Goal**: Expand the research pipeline's academic capabilities with citation graph tools, new academic providers (OpenAlex, Unpaywall, Crossref, OpenCitations), adaptive provider selection from the brief phase, and strategic research primitives that improve how topic researchers navigate the literature.
>
> **Estimated scope**: ~960-1350 LOC implementation + ~600-800 LOC tests (including mock fixtures) across 12-16 files
>
> **Dependencies**: PLAN-0 (supervision refactoring), PLAN-1 item 1 (Research Profiles) for profile-driven provider selection
>
> **Provider tiers**: Providers are split into **Tier 1** (OpenAlex — highest value, free, broad index) and **Tier 2** (Unpaywall, Crossref, OpenCitations — optional enrichment). Tier 1 is implemented first and is the only hard dependency for items 5-7. Tier 2 providers enhance but are not required.

---

## Design Principles

1. **Providers are plug-and-play.** Every new provider implements `SearchProvider` and registers in the provider chain. The pipeline doesn't need to know provider internals.
2. **The brief drives the pipeline.** Instead of hardcoded provider chains per mode, the brief phase output selects which providers and tools to activate. The brief already analyzes the query — now it also configures the machinery.
3. **Researchers have strategy, not just tools.** Adding `citation_search` and `related_papers` as raw tools isn't enough. The topic researcher needs strategic primitives that tell it *when* to broaden, deepen, or validate — not just *how*.
4. **Rate limiting is per-provider.** With multiple academic APIs in play, each with different limits, the resilience layer must manage them independently.

---

## 1. OpenAlex Provider (Tier 1 — Required)

### Why First

OpenAlex is the single most impactful provider addition. It covers discovery, citations, full-text PDF links, and metadata enrichment in one API. Its index (450M+ works) is the broadest available, it's completely free (CC0), and its API is designed for programmatic consumption. **This is the only provider that items 5-7 depend on.** Tier 2 providers enhance results but are not blocking.

### API Details

- **Base URL**: `https://api.openalex.org`
- **Auth**: Free. Polite pool (faster responses) with email parameter.
- **Rate limit**: 100K requests/day (polite pool), 10 RPS sustained
- **License**: CC0 (completely open)

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

    def get_provider_name(self) -> str:
        return "openalex"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        *,
        filters: Optional[dict] = None,
        sort: Optional[str] = None,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Search works with optional filters.

        Supported filters (partial list):
        - publication_year: int or range (e.g., "2020-2024")
        - type: "article" | "book" | "dataset" | ...
        - open_access.is_oa: bool
        - cited_by_count: ">100" (comparative filters)
        - concepts.id: OpenAlex concept ID
        - authorships.institutions.id: institution filter
        """

    async def get_work(self, work_id: str) -> Optional[ResearchSource]:
        """Fetch a single work by OpenAlex ID, DOI, or PMID.

        Supports ID formats: W1234567890, https://doi.org/..., doi:..., pmid:...
        """

    async def get_citations(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get papers that cite a given work.

        Uses: GET /works?filter=cites:{work_id}
        """

    async def get_references(self, work_id: str, max_results: int = 20) -> list[ResearchSource]:
        """Get papers cited by a given work.

        Uses referenced_works field from the work record.
        """

    async def get_related(self, work_id: str, max_results: int = 10) -> list[ResearchSource]:
        """Get related works.

        Uses related_works field from the work record.
        """

    async def classify_text(self, text: str) -> list[dict]:
        """Classify arbitrary text into OpenAlex topics.

        Uses: POST /text
        Returns: [{topic_id, display_name, score, subfield, field, domain}]
        """

    async def search_by_topic(
        self,
        topic_id: str,
        max_results: int = 20,
        **kwargs: Any,
    ) -> list[ResearchSource]:
        """Search works by OpenAlex topic ID.

        Uses: GET /works?filter=topics.id:{topic_id}
        """
```

**Metadata mapping** (OpenAlex → `ResearchSource.metadata`):

| OpenAlex field | Metadata key | Notes |
|---------------|-------------|-------|
| `id` | `openalex_id` | e.g. `W1234567890` |
| `doi` | `doi` | Normalized DOI |
| `title` | (source.title) | |
| `authorships[].author.display_name` | `authors` | Comma-separated |
| `cited_by_count` | `citation_count` | |
| `publication_year` | `year` | |
| `primary_location.source.display_name` | `venue` | Journal/conference |
| `topics[0].display_name` | `primary_topic` | |
| `open_access.oa_url` | `pdf_url` | If available |
| `type` | `publication_type` | article, book, etc. |
| `grants[].funder.display_name` | `funders` | Comma-separated |
| `authorships[].institutions[].display_name` | `institutions` | Unique, comma-separated |
| `abstract_inverted_index` | (source.snippet) | Reconstruct from inverted index |

**Resilience config**:
```python
ERROR_CLASSIFIERS = {
    429: ErrorType.RATE_LIMIT,
    503: ErrorType.UNAVAILABLE,
}
# Rate limit: 10 RPS (polite pool)
```

### Testing

- Unit test: `search()` with mocked response, verify metadata mapping
- Unit test: `get_citations()` with mocked response
- Unit test: `get_references()` with mocked response
- Unit test: `get_work()` with DOI input format
- Unit test: `classify_text()` with mocked response
- Unit test: abstract reconstruction from inverted index
- Unit test: rate limiting respected
- Unit test: graceful handling of empty results

---

## 2. Unpaywall Provider (Tier 2 — Optional Enrichment)

### Why Second

Single-endpoint, zero-complexity addition that directly unlocks PDF access for PLAN-4's full-text analysis. One DOI in, best open-access PDF URL out. **Optional**: OpenAlex also provides `open_access.oa_url` for many papers, so Unpaywall is a complementary fallback for broader OA coverage.

### API Details

- **Endpoint**: `GET https://api.unpaywall.org/v2/{DOI}?email={email}`
- **Auth**: Free, requires email address
- **Rate limit**: 100K/day
- **Returns**: Best OA location with PDF URL, all OA locations, license info

### Implementation

**File: `src/foundry_mcp/core/research/providers/unpaywall.py`** (NEW)

```python
class UnpaywallProvider:
    """Unpaywall API provider for resolving DOIs to open-access PDF URLs.

    Not a SearchProvider — this is an enrichment provider used by the
    pipeline to locate full-text PDFs for sources with DOIs.
    """

    BASE_URL = "https://api.unpaywall.org/v2"

    async def resolve_pdf(self, doi: str) -> Optional[str]:
        """Resolve a DOI to the best available open-access PDF URL.

        Returns None if no OA copy is found.
        """

    async def get_oa_status(self, doi: str) -> dict:
        """Get full OA status including all locations, license, and OA color.

        Returns: {
            is_oa: bool,
            best_oa_location: {url, url_for_pdf, license, version},
            oa_locations: [...],
            oa_status: "gold" | "green" | "hybrid" | "bronze" | "closed"
        }
        """

    async def enrich_source(self, source: ResearchSource) -> ResearchSource:
        """Enrich a ResearchSource with OA PDF URL if available.

        Reads DOI from source.metadata, resolves via Unpaywall,
        stores pdf_url and oa_status in metadata.
        """
```

### Testing

- Unit test: `resolve_pdf()` with known OA DOI → PDF URL
- Unit test: `resolve_pdf()` with closed-access DOI → None
- Unit test: `enrich_source()` adds `pdf_url` to source metadata
- Unit test: graceful handling of invalid DOIs

---

## 3. Crossref Provider (Tier 2 — Optional Enrichment)

### Why Third

Authoritative metadata enrichment. When Semantic Scholar or OpenAlex metadata is incomplete (missing volume, issue, pages for APA formatting), Crossref is the definitive fallback. This directly supports PLAN-1's APA citation formatting. **Optional**: OpenAlex provides sufficient metadata for most APA formatting; Crossref fills gaps for edge cases.

### API Details

- **Base URL**: `https://api.crossref.org`
- **Key endpoint**: `GET /works/{doi}`
- **Auth**: Free (polite pool with `mailto:` header for 50 RPS)
- **Returns**: Title, authors, journal, volume, issue, pages, publisher, license, funder, references, type

### Implementation

**File: `src/foundry_mcp/core/research/providers/crossref.py`** (NEW)

```python
class CrossrefProvider:
    """Crossref API provider for authoritative bibliographic metadata.

    Used as a metadata enrichment fallback when primary providers
    (Semantic Scholar, OpenAlex) have incomplete records.
    Not a SearchProvider — this is an enrichment provider.
    """

    BASE_URL = "https://api.crossref.org"

    async def get_work(self, doi: str) -> Optional[dict]:
        """Fetch full bibliographic metadata for a DOI.

        Returns normalized dict with: title, authors (list of {given, family}),
        journal, volume, issue, pages, publisher, doi, type, issued (date).
        """

    async def enrich_source(self, source: ResearchSource) -> ResearchSource:
        """Enrich a ResearchSource with Crossref metadata (fills missing fields only).

        Only overwrites fields that are currently None/empty in source.metadata.
        """
```

### Testing

- Unit test: `get_work()` with mocked response, verify field extraction
- Unit test: `enrich_source()` fills missing venue but doesn't overwrite existing
- Unit test: graceful handling of DOIs not in Crossref

---

## 4. OpenCitations Provider (Tier 2 — Optional Enrichment)

### Why Fourth

Complete open citation graph with 2B+ DOI-to-DOI links. Unlike Semantic Scholar (citation counts and limited lists), OpenCitations provides the full wiring with metadata about each citation link (date, self-citation flags). This enriches PLAN-4's citation network feature. **Optional**: OpenAlex's `cites` filter provides citation traversal; OpenCitations adds self-citation metadata and more complete graph edges.

### API Details

- **Base URL**: `https://opencitations.net/index/api/v2`
- **Key endpoints**:
  - `GET /citations/{doi}` — papers citing a given DOI
  - `GET /references/{doi}` — papers cited by a given DOI
- **Auth**: Free, no key required
- **Rate limit**: No documented limit
- **License**: CC0
- **Returns per citation**: citing DOI, cited DOI, creation date, timespan, journal self-citation flag, author self-citation flag

### Implementation

**File: `src/foundry_mcp/core/research/providers/opencitations.py`** (NEW)

```python
class OpenCitationsProvider:
    """OpenCitations COCI provider for DOI-to-DOI citation graph traversal.

    Provides complete citation graph with metadata about each citation link.
    Not a SearchProvider — this is a graph traversal provider.
    """

    BASE_URL = "https://opencitations.net/index/api/v2"

    async def get_citations(self, doi: str) -> list[dict]:
        """Get all papers citing a given DOI.

        Returns: [{citing_doi, cited_doi, creation_date, timespan,
                   journal_self_citation, author_self_citation}]
        """

    async def get_references(self, doi: str) -> list[dict]:
        """Get all papers referenced by a given DOI.

        Returns: same structure as get_citations()
        """

    async def get_citation_count(self, doi: str) -> int:
        """Get total citation count for a DOI (len of get_citations)."""
```

### Testing

- Unit test: `get_citations()` with mocked response
- Unit test: `get_references()` with mocked response
- Unit test: self-citation flag parsing
- Unit test: graceful handling of DOIs with no citations

---

## 5. Citation Graph & Related Papers Tools for Topic Researchers

### Problem

Topic researchers (`phases/topic_research.py`) only have `web_search`, `extract_content`, `think`, and `research_complete` tools. For academic research, two critical capabilities are missing:
- **Forward citation search**: "Find papers that cite this seminal paper" (snowball sampling)
- **Related papers**: "Find papers similar to this one" (lateral discovery)

### Changes

**File: `src/foundry_mcp/core/research/providers/semantic_scholar.py`**

#### 5a. Add citation search method

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
    """
```

#### 5b. Add related papers method

```python
async def get_recommendations(
    self,
    paper_id: str,
    max_results: int = 10,
    **kwargs: Any,
) -> list[ResearchSource]:
    """Get recommended papers based on a given paper.

    Uses: POST /recommendations/v1/papers/
    """
```

#### 5c. Add paper lookup by DOI/ID

```python
async def get_paper(
    self,
    paper_id: str,
    fields: str = EXTENDED_FIELDS,
) -> Optional[ResearchSource]:
    """Look up a single paper by Semantic Scholar ID, DOI, or ArXiv ID.

    Supports ID formats: S2 paper ID, DOI:xxx, ArXiv:xxx, PMID:xxx
    """
```

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 5d. Add tool models

```python
class CitationSearchTool(BaseModel):
    """Search for papers citing a specific paper."""
    paper_id: str = Field(..., description="Semantic Scholar paper ID, DOI, or ArXiv ID")
    max_results: int = Field(default=10, ge=1, le=50)

class RelatedPapersTool(BaseModel):
    """Find papers related to a specific paper."""
    paper_id: str = Field(..., description="Semantic Scholar paper ID, DOI, or ArXiv ID")
    max_results: int = Field(default=10, ge=1, le=20)
```

Register in `RESEARCHER_TOOL_SCHEMAS`.

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 5e. Conditional tool injection

Only add academic tools to the system prompt when the profile has `enable_citation_tools == True`:

```
### citation_search (academic profile only)
Find papers that cite a specific paper. Useful for forward snowball sampling —
when you've found a seminal paper, use this to discover the research it spawned.
Arguments: {"paper_id": "DOI or Semantic Scholar ID", "max_results": 10}
Returns: List of citing papers with titles, authors, years, and abstracts.

### related_papers (academic profile only)
Find papers similar to a specific paper. Useful for lateral discovery —
when you've found a relevant paper, use this to find others in the same space.
Arguments: {"paper_id": "DOI or Semantic Scholar ID", "max_results": 10}
Returns: List of related papers with titles, authors, years, and abstracts.
```

#### 5f. Add dispatch handlers

Add `_handle_citation_search_tool()` and `_handle_related_papers_tool()` following the existing `_handle_web_search_tool()` pattern:
1. Parse tool arguments via Pydantic models
2. Look up provider (Semantic Scholar first; OpenAlex as fallback if available)
3. Call the appropriate method
4. Format results with novelty tracking (reuse existing dedup logic)
5. Count against tool call budget
6. Log `provider_query` provenance event

### Testing

- Unit test: each new Semantic Scholar method with mocked HTTP responses
- Unit test: tool dispatch routes correctly for `citation_search` and `related_papers`
- Unit test: tools are NOT available when profile has `enable_citation_tools == False`
- Unit test: novelty tracking deduplicates sources across tools
- Integration test: topic researcher chains `web_search` → `citation_search`

---

## 6. Strategic Research Primitives

### Problem

Adding tools (citation search, related papers) is necessary but not sufficient. The topic researcher's LLM needs to understand *when* to use different strategies. Currently, the system prompt describes tools mechanically. It doesn't teach the researcher how to navigate a literature.

### Design

Strategic primitives are not new API calls — they're framing in the system prompt that help the LLM make better decisions about which tools to use and in what order. They map to the strategies real researchers use.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`**

#### 6a. Add strategic guidance to academic researcher prompt

When the profile has `enable_citation_tools == True`, append to the researcher system prompt:

```
## Research Strategies

You have access to academic tools. Use these strategies deliberately:

### Strategy: BROADEN
When your initial results are too narrow or you keep finding the same papers.
- Use `web_search` with broader/alternative terminology
- Use `related_papers` on a relevant paper to discover adjacent work
- Reformulate your query using different disciplinary vocabulary

### Strategy: DEEPEN
When you've found a seminal or highly relevant paper and need to trace its impact.
- Use `citation_search` on seminal papers to find the research they spawned
- Use `extract_content` on the most promising citing papers
- Look for methodological variations on the original study

### Strategy: VALIDATE
When you've found a surprising or important claim and need corroboration.
- Use `web_search` for the specific claim or finding
- Use `citation_search` to see if later papers confirmed or contradicted it
- Use `related_papers` to find parallel studies with independent evidence

### Strategy: SATURATE
When you keep discovering papers you've already seen — signal that coverage is likely sufficient.
- Call `research_complete` when >50% of new results are duplicates
- Note the saturation point in your summary — this is useful provenance data

Choose your strategy based on the current state of your research, not a fixed sequence.
A typical academic search interleaves: search broadly → identify seminal papers →
deepen via citations → validate key claims → check for saturation.
```

#### 6b. Log strategy usage in provenance

When the topic researcher's `think` tool output contains strategy keywords (BROADEN, DEEPEN, VALIDATE, SATURATE), log the strategy choice in provenance. This is lightweight — just pattern-match the think output.

### Testing

- Unit test: academic researcher prompt includes strategy guidance
- Unit test: general researcher prompt does NOT include strategy guidance
- Unit test: strategy keywords in think output are captured in provenance

---

## 7. Adaptive Provider Selection

### Problem

PLAN-PROVIDERS defined a fixed provider chain for academic mode. But different queries benefit from different providers. A biomedical query should preference PubMed. An education query benefits from ERIC. A CS query benefits from DBLP. Hardcoding these mappings doesn't scale.

### Design

Use the brief phase output — which already identifies discipline, scope, and source preferences — to select providers dynamically. The brief becomes a pipeline configurator.

### Changes

**File: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`**

#### 7a. Extract provider hints from brief

After the brief LLM call, extract discipline and topic signals. These are already in the brief output (or can be parsed from it) — no additional LLM call needed.

```python
def _extract_provider_hints(self, brief: ResearchBriefOutput, profile: ResearchProfile) -> list[str]:
    """Extract provider hints from the research brief.

    Rules:
    - Brief mentions "biomedical", "clinical", "health" → suggest pubmed
    - Brief mentions "computer science", "machine learning" → suggest semantic_scholar (already default)
    - Brief mentions "education" → suggest openalex (broad ERIC coverage)
    - Brief mentions "social science", "economics" → suggest openalex
    - Profile already specifies providers → use those (no override)

    Returns list of suggested provider names to add to the active set.
    """
```

#### 7b. Apply hints to session state

If the profile doesn't explicitly specify providers (i.e., using the default chain), augment the provider list with hints:

```python
if not profile.providers_explicitly_set:
    state.active_providers = list(set(profile.providers + hints))
```

If the profile explicitly specifies providers, respect that — hints are only for auto-configuration.

**File: `phases/supervision.py`** (orchestration) and **`phases/supervision_prompts.py`** (prompt builders)

#### 7c. Use active providers in delegation

When creating topic researcher tasks, pass `state.active_providers` so the researcher knows which search tools are available. Modify `build_delegation_user_prompt()` in `supervision_prompts.py` to include:

```python
# In delegation prompt:
Available search providers for this session: {', '.join(state.active_providers)}
```

### Testing

- Unit test: biomedical brief triggers PubMed hint
- Unit test: education brief triggers OpenAlex hint
- Unit test: explicit profile providers are not overridden by hints
- Unit test: hints are additive (don't remove existing providers)
- Unit test: unknown/unavailable provider hints are silently dropped

---

## 8. Per-Provider Rate Limiting

### Problem

With multiple academic APIs in play, each with vastly different rate limits:
- Semantic Scholar: 1 RPS (without key), 10 RPS (with key)
- OpenAlex: 10 RPS (polite pool)
- Unpaywall: ~2 RPS (100K/day)
- OpenCitations: no documented limit
- Crossref: 50 RPS (polite pool)

The existing resilience layer (`providers/resilience/`) supports per-provider rate limiting but needs configuration for the new providers.

### Changes

**File: `src/foundry_mcp/core/research/providers/resilience/config.py`**

#### 8a. Add resilience configs for new providers

The existing `PROVIDER_CONFIGS` dict maps provider names to `ProviderResilienceConfig` instances (with `requests_per_second`, `burst_limit`, `max_retries`, `base_delay`, `max_delay`, `jitter`, `circuit_failure_threshold`, `circuit_recovery_timeout`). Add entries for new providers following this pattern:

```python
# Add to PROVIDER_CONFIGS dict:
"openalex": ProviderResilienceConfig(
    requests_per_second=10.0,   # 10 RPS (polite pool)
    burst_limit=5,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    jitter=0.5,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
),
"unpaywall": ProviderResilienceConfig(
    requests_per_second=2.0,    # Conservative (100K/day)
    burst_limit=2,
    max_retries=3,
    base_delay=1.5,
    max_delay=60.0,
    jitter=0.5,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
),
"crossref": ProviderResilienceConfig(
    requests_per_second=10.0,   # 50 RPS polite pool, conservative default
    burst_limit=5,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    jitter=0.5,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
),
"opencitations": ProviderResilienceConfig(
    requests_per_second=5.0,    # Conservative (no documented limit)
    burst_limit=3,
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    jitter=0.5,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
),
```

These are the defaults used by `get_provider_config()`. Per-provider overrides can be specified in `ResearchConfig.per_provider_rate_limits`.

**File: `src/foundry_mcp/config/research.py`**

#### 8b. Add provider config fields

```python
# OpenAlex
openalex_email: Optional[str] = None     # For polite pool (faster responses)
openalex_enabled: bool = True             # On by default (free, no key needed)

# Unpaywall
unpaywall_email: Optional[str] = None     # Required (their only auth)
unpaywall_enabled: bool = True            # On by default when email present

# OpenCitations
opencitations_enabled: bool = True        # Free, no key needed

# Crossref
crossref_email: Optional[str] = None      # For polite pool
crossref_enabled: bool = True             # Free, on by default
```

### Testing

- Unit test: new providers get correct default rate limits
- Unit test: config overrides default rate limits
- Unit test: unavailable providers (missing email for Unpaywall) are excluded from provider chain

---

## Testing Budget

| Item | Impl LOC | Test LOC | Test Focus |
|------|----------|----------|------------|
| 1. OpenAlex (Tier 1) | ~250-350 | ~150-200 | Mock API responses, metadata mapping, abstract reconstruction, rate limiting |
| 2. Unpaywall (Tier 2) | ~100-150 | ~60-80 | Mock responses, DOI resolution, enrichment |
| 3. Crossref (Tier 2) | ~100-150 | ~50-70 | Mock responses, field extraction, enrichment |
| 4. OpenCitations (Tier 2) | ~80-120 | ~40-60 | Mock responses, self-citation parsing |
| 5. Citation Graph Tools | ~200-250 | ~100-130 | Tool dispatch, dedup, provider fallback |
| 6. Strategic Primitives | ~80-100 | ~30-40 | Prompt inclusion, profile gating |
| 7. Adaptive Selection | ~100-150 | ~60-80 | Hint extraction, provider augmentation |
| 8. Rate Limiting | ~50-80 | ~30-40 | Config loading, override behavior |
| **Shared fixtures** | — | ~80-100 | Mock HTTP responses for all 4 APIs |
| **Total** | **~960-1350** | **~600-800** | |

**Note**: Each new provider requires dedicated mock fixtures (~20-25 LOC each) simulating successful responses, empty results, rate limit errors, and unavailability. These are shared across provider tests and integration tests.

## File Impact Summary

| File | Type | Items |
|------|------|-------|
| `providers/openalex.py` | **New** (Tier 1) | 1 |
| `providers/unpaywall.py` | **New** (Tier 2) | 2 |
| `providers/crossref.py` | **New** (Tier 2) | 3 |
| `providers/opencitations.py` | **New** (Tier 2) | 4 |
| `providers/semantic_scholar.py` | Modify | 5 (citation search, recommendations, paper lookup) |
| `models/deep_research.py` | Modify | 5 (CitationSearchTool, RelatedPapersTool) |
| `phases/topic_research.py` | Modify | 5 (tool injection, dispatch), 6 (strategic primitives) |
| `phases/brief.py` | Modify | 7 (provider hint extraction) |
| `phases/supervision.py` | Modify | 7 (active providers in delegation orchestration) |
| `phases/supervision_prompts.py` | Modify | 7 (active providers in delegation prompts) |
| `providers/resilience/config.py` | Modify | 8 (rate limits for new providers) |
| `config/research.py` | Modify | 8 (new provider config fields) |

## Dependency Graph

```
Tier 1 (required):
[1. OpenAlex Provider]──────────────────────────────────────┐
                                                             │
Tier 2 (optional enrichment, can be deferred):               │
[2. Unpaywall Provider]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
[3. Crossref Provider]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
[4. OpenCitations Provider]─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
                                                             │
[5. Citation Graph Tools] (needs Semantic Scholar + OpenAlex)┤
                                                             │
[6. Strategic Research Primitives] (needs item 5 tools)      │
                                                             │
[7. Adaptive Provider Selection] (needs item 1; enhanced by 2-4)
                                                             │
[8. Per-Provider Rate Limiting] (parallel with all above)────┘
```

**Tier 1 critical path**: Item 1 (OpenAlex) → Items 5, 7 → Item 6. This is the minimum viable academic pipeline.

**Tier 2 providers** (items 2-4) can be added independently at any point after item 8 (rate limiting config). They enhance results but don't block core functionality. Item 7 (adaptive selection) works with just OpenAlex; Tier 2 providers expand the hint vocabulary.
