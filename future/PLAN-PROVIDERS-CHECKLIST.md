# PLAN-PROVIDERS Implementation Checklist

> Track progress for the academic provider expansion. Ordered by priority.

---

## P1: OpenAlex Provider

### Core Implementation
- [ ] Create `src/foundry_mcp/core/research/providers/openalex.py`
- [ ] Implement `OpenAlexProvider(SearchProvider)` class
- [ ] Implement `search(query, max_results, **kwargs)` with filter support
- [ ] Support filters: year, topic, institution, funder, OA status, citation count, type
- [ ] Implement `get_citations(work_id, max_results)` via `filter=cited_by:<id>`
- [ ] Implement `get_references(work_id, max_results)` via `filter=cites:<id>`
- [ ] Implement `get_related(work_id, max_results)` via `related_works` field
- [ ] Implement `classify_text(text)` via POST `/text` endpoint
- [ ] Map OpenAlex fields to `ResearchSource` with `SourceType.ACADEMIC`
- [ ] Include metadata: openalex_id, doi, authors, citation_count, year, venue, primary_topic, pdf_url, publication_type, funders, institutions
- [ ] Handle pagination (cursor-based, OpenAlex style)
- [ ] Add polite pool support (email parameter for faster responses)
- [ ] Add API key support (free key from openalex.org)
- [ ] Apply resilience patterns (retry, circuit breaker) matching existing providers

### Configuration
- [ ] Add `openalex_api_key`, `openalex_email`, `openalex_enabled` to ResearchConfig
- [ ] Parse in `from_toml_dict()`
- [ ] Add to `deep_research_providers` default list for ACADEMIC mode
- [ ] Add rate limit config: `openalex_rate_limit` (default: reasonable for 100K/day)

### Testing
- [ ] Unit test: search with mocked HTTP response
- [ ] Unit test: get_citations with mocked response
- [ ] Unit test: get_references with mocked response
- [ ] Unit test: metadata mapping completeness
- [ ] Unit test: filter parameter construction
- [ ] Unit test: pagination handling
- [ ] Unit test: resilience (retry on 429, circuit breaker on 5xx)

---

## P2: Unpaywall Provider

### Core Implementation
- [ ] Create `src/foundry_mcp/core/research/providers/unpaywall.py`
- [ ] Implement `UnpaywallProvider` class
- [ ] Implement `resolve_pdf(doi) -> Optional[str]` (DOI → PDF URL)
- [ ] Implement `get_oa_status(doi) -> dict` (full OA metadata)
- [ ] Handle: no OA copy available (return None, not error)
- [ ] Handle: multiple OA locations (return best, store all)
- [ ] Extract: license info, OA color (gold/green/bronze/hybrid)

### Configuration
- [ ] Add `unpaywall_email`, `unpaywall_enabled` to ResearchConfig
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: resolve_pdf with OA paper -> returns URL
- [ ] Unit test: resolve_pdf with paywalled paper -> returns None
- [ ] Unit test: get_oa_status returns license and OA color
- [ ] Unit test: invalid DOI handling

---

## P3: Scite.ai (Remote MCP)

### MCP Bridge
- [ ] Create `src/foundry_mcp/core/research/providers/mcp_bridge.py`
- [ ] Implement generic `RemoteMCPBridge` class for calling remote MCP servers
- [ ] Handle MCP protocol: tool discovery, tool invocation, response parsing
- [ ] Add timeout, retry, and error handling

### Scite Integration
- [ ] Implement post-gathering enrichment step for citation sentiment
- [ ] For each academic source with DOI: query Scite for Smart Citation data
- [ ] Store in `source.metadata`: `scite_supporting`, `scite_contrasting`, `scite_mentioning`, `scite_total`
- [ ] Batch queries where possible to minimize round-trips
- [ ] Only run when `scite_mcp_enabled == True`

### Synthesis Integration
- [ ] Include citation sentiment in synthesis prompt for literature_review type
- [ ] Format: "Smith (2021) — 12 supporting, 3 contrasting citations"
- [ ] Feed into "Key Debates & Contradictions" section

### Configuration
- [ ] Add `scite_mcp_enabled`, `scite_mcp_url` to ResearchConfig
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: MCP bridge tool invocation with mocked server
- [ ] Unit test: citation sentiment parsing
- [ ] Unit test: metadata enrichment on ResearchSource
- [ ] Unit test: graceful handling when Scite MCP unavailable
- [ ] Unit test: synthesis prompt includes sentiment data

---

## P4: OpenCitations (COCI)

### Core Implementation
- [ ] Create `src/foundry_mcp/core/research/providers/opencitations.py`
- [ ] Implement `OpenCitationsProvider` class
- [ ] Implement `get_citations(doi) -> list[dict]` via `/citations/{doi}`
- [ ] Implement `get_references(doi) -> list[dict]` via `/references/{doi}`
- [ ] Parse response: citing DOI, cited DOI, creation date, timespan, self-citation flags
- [ ] Handle DOI URL encoding (slashes in DOIs)
- [ ] Handle empty results (paper not in COCI index)

### Configuration
- [ ] Add `opencitations_enabled` to ResearchConfig
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: get_citations with mocked response
- [ ] Unit test: get_references with mocked response
- [ ] Unit test: self-citation flag extraction
- [ ] Unit test: DOI encoding for API calls
- [ ] Unit test: empty result handling

---

## P5: Consensus (Remote MCP)

### Integration
- [ ] Reuse `RemoteMCPBridge` from P3
- [ ] Implement Consensus-specific query formatting
- [ ] Route yes/no sub-questions from supervisor to Consensus
- [ ] Parse Consensus Meter results (agreement percentage, study count)
- [ ] Store in `state.metadata["consensus_results"]`
- [ ] Inject into synthesis prompt for "Key Findings" section
- [ ] Only run when `consensus_mcp_enabled == True`

### Configuration
- [ ] Add `consensus_mcp_enabled`, `consensus_mcp_url` to ResearchConfig
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: MCP bridge invocation with mocked Consensus server
- [ ] Unit test: Consensus Meter result parsing
- [ ] Unit test: synthesis prompt includes agreement data
- [ ] Unit test: graceful handling when Consensus MCP unavailable

---

## P6: Crossref

### Core Implementation
- [ ] Create `src/foundry_mcp/core/research/providers/crossref.py`
- [ ] Implement `CrossrefProvider` class
- [ ] Implement `get_work(doi) -> Optional[dict]` for full metadata
- [ ] Implement `enrich_source(source) -> ResearchSource` for filling missing fields
- [ ] Map: title, authors (given/family), journal, volume, issue, pages, publisher, license, funder, type
- [ ] Add polite pool support (`mailto:` header)

### APA Formatting Integration
- [ ] Use Crossref as fallback when Semantic Scholar/OpenAlex metadata is incomplete
- [ ] Fill missing: volume, issue, pages, publisher for APA citations
- [ ] Call `enrich_source()` in citation postprocessing when fields are missing

### Configuration
- [ ] Add `crossref_email`, `crossref_enabled` to ResearchConfig
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: get_work with mocked response
- [ ] Unit test: enrich_source fills missing volume/issue/pages
- [ ] Unit test: polite pool header sent when email configured
- [ ] Unit test: handles 404 for unknown DOI

---

## P7: CORE

### Core Implementation
- [ ] Create `src/foundry_mcp/core/research/providers/core_oa.py`
- [ ] Implement `COREProvider(SearchProvider)` class
- [ ] Implement `search(query, max_results, **kwargs)` for full-text search
- [ ] Implement `get_full_text(core_id) -> Optional[str]`
- [ ] Map CORE fields to `ResearchSource`
- [ ] Handle rate limiting (5 req/10s without key)

### Configuration
- [ ] Add `core_api_key`, `core_enabled` to ResearchConfig
- [ ] Parse in `from_toml_dict()`
- [ ] Default: disabled (rate-limited without key)

### Testing
- [ ] Unit test: search with mocked response
- [ ] Unit test: full-text retrieval
- [ ] Unit test: rate limit handling

---

## P8: PubMed (Remote MCP)

### Documentation & Configuration
- [ ] Add PubMed MCP configuration guidance to docs
- [ ] Add `pubmed_mcp_enabled`, `pubmed_mcp_url` to ResearchConfig
- [ ] Document as parallel MCP server (not embedded in pipeline)
- [ ] Optional: implement NCBI E-utilities wrapper for deep research integration

### Testing
- [ ] Verify PubMed MCP server is reachable
- [ ] Document example configuration for Claude Code users

---

## Cross-Cutting

### Provider Chain
- [ ] Update default `deep_research_providers` for ACADEMIC mode
- [ ] Implement provider chain fallback logic (try OpenAlex, fall back to Semantic Scholar, then Tavily)
- [ ] Add provider health checks for new providers
- [ ] Register new providers in provider registry

### Metadata Enrichment Pipeline
- [ ] Implement post-gathering metadata enrichment step
- [ ] Chain: Unpaywall (PDF URLs) → Crossref (missing bib fields) → Scite (citation sentiment)
- [ ] Only enrich sources with DOIs
- [ ] Respect rate limits across all providers

### Testing
- [ ] All new providers pass unit tests independently
- [ ] Provider chain fallback works correctly
- [ ] GENERAL mode is unaffected (no new providers in default chain)
- [ ] ACADEMIC mode uses expanded provider chain
- [ ] `pytest tests/core/research/` passes with all providers disabled
- [ ] `pytest tests/core/research/` passes with all providers enabled (mocked)
