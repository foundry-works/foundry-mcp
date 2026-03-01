# PLAN-2 Checklist: Academic Tools & Provider Expansion

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed. Sub-items can be worked in parallel where noted.

---

## Item 1: OpenAlex Provider (Tier 1)
> **New file**: `providers/openalex.py`
> **Parallel with**: Item 6 (rate limiting config)

### 1a. Core provider class
- [x] Create `src/foundry_mcp/core/research/providers/openalex.py`
- [x] Implement `OpenAlexProvider(SearchProvider)` with `BASE_URL`, `get_provider_name()`
- [x] Implement API key auth (`?api_key=` param or `x-api-key` header)
- [x] Implement HTTP client setup with resilience integration

### 1b. Search method
- [x] Implement `search()` with query, max_results, filters, sort
- [x] Support filters: `publication_year`, `type`, `open_access.is_oa`, `cited_by_count`, `topics.id`, `authorships.institutions.id`
- [x] Map OpenAlex response fields to `ResearchSource` (see metadata mapping in PLAN.md)

### 1c. Abstract reconstruction
- [x] Implement `_reconstruct_abstract(abstract_inverted_index: dict) -> str`
- [x] Handle edge cases: empty index, None input

### 1d. Single work lookup
- [x] Implement `get_work(work_id)` supporting formats: OpenAlex ID, DOI, PMID

### 1e. Citation graph methods
- [x] Implement `get_citations(work_id, max_results)` via `?filter=cites:{work_id}`
- [x] Implement `get_references(work_id, max_results)` via `referenced_works` field
- [x] Implement `get_related(work_id, max_results)` via `related_works` field

### 1f. Topic classification
- [x] Implement `classify_text(text)` via `POST /text`
- [x] Implement `search_by_topic(topic_id, max_results)` via `?filter=topics.id:{topic_id}`

### 1g. Error handling
- [x] Map 429 → RATE_LIMIT, 503 → UNAVAILABLE
- [x] Handle budget exhaustion gracefully (429 with specific message)
- [x] Handle missing API key with clear error message

### Item 1 Testing
- [x] Unit test: `search()` with mocked response, verify metadata mapping
- [x] Unit test: `get_citations()` with mocked response
- [x] Unit test: `get_references()` with mocked response
- [x] Unit test: `get_work()` with DOI input format
- [x] Unit test: `classify_text()` with mocked response
- [x] Unit test: abstract reconstruction from inverted index
- [x] Unit test: rate limiting respected
- [x] Unit test: graceful handling of empty results
- [x] Mock fixture: successful search response
- [x] Mock fixture: empty results response
- [x] Mock fixture: rate limit error response

---

## Item 2: Crossref Provider (Tier 2)
> **New file**: `providers/crossref.py`
> **Parallel with**: Items 1, 3, 6

### 2a. Core provider class
- [x] Create `src/foundry_mcp/core/research/providers/crossref.py`
- [x] Implement `CrossrefProvider` (not a SearchProvider — enrichment only)
- [x] Set up `mailto:` in User-Agent header for polite pool

### 2b. Work lookup
- [x] Implement `get_work(doi)` via `GET /works/{doi}`
- [x] Normalize response fields: title, authors, journal, volume, issue, pages, publisher, doi, type, issued

### 2c. Source enrichment
- [x] Implement `enrich_source(source)` — fills missing fields only, never overwrites
- [x] Handle DOIs not in Crossref gracefully (return source unchanged)

### Item 2 Testing
- [x] Unit test: `get_work()` with mocked response, verify field extraction
- [x] Unit test: `enrich_source()` fills missing venue but doesn't overwrite existing
- [x] Unit test: graceful handling of DOIs not in Crossref
- [x] Mock fixture: successful work response
- [x] Mock fixture: 404 response

---

## Item 3: Citation Graph & Related Papers Tools
> **Depends on**: Item 1 (OpenAlex as fallback provider)
> **Parallel with**: Items 2, 6

### 3a. Semantic Scholar methods
> **File**: `providers/semantic_scholar.py`

- [x] Add `get_citations(paper_id, max_results, fields)` — `GET /paper/{paper_id}/citations`
- [x] Add `get_recommendations(paper_id, max_results)` — `POST /recommendations/v1/papers/`
- [x] Add `get_paper(paper_id, fields)` — `GET /paper/{paper_id}` (supports DOI, ArXiv, PMID formats)

### 3b. Tool models
> **File**: `models/deep_research.py`

- [x] Add `CitationSearchTool(BaseModel)` with `paper_id`, `max_results`
- [x] Add `RelatedPapersTool(BaseModel)` with `paper_id`, `max_results`
- [x] Register both in `RESEARCHER_TOOL_SCHEMAS`

### 3c. Conditional tool injection
> **File**: `phases/topic_research.py`

- [x] Add `citation_search` and `related_papers` tool descriptions to system prompt
- [x] Gate on `profile.enable_citation_tools == True`
- [x] Include tool usage guidance (forward snowball sampling, lateral discovery)

### 3d. Dispatch handlers
> **File**: `phases/topic_research.py`

- [x] Add `_handle_citation_search_tool()` following `_handle_web_search_tool()` pattern
- [x] Add `_handle_related_papers_tool()` following same pattern
- [x] Provider fallback: Semantic Scholar first, OpenAlex if available
- [x] Integrate with novelty tracking (reuse existing dedup logic)
- [x] Count against tool call budget
- [x] Log `provider_query` provenance event

### Item 3 Testing
- [x] Unit test: each new Semantic Scholar method with mocked HTTP responses
- [x] Unit test: tool dispatch routes correctly for `citation_search`
- [x] Unit test: tool dispatch routes correctly for `related_papers`
- [x] Unit test: tools NOT available when `enable_citation_tools == False`
- [x] Unit test: novelty tracking deduplicates sources across tools

---

## Item 4: Strategic Research Primitives
> **Depends on**: Item 3 (tools must exist for strategies to reference)

### 4a. Strategic guidance in researcher prompt
> **File**: `phases/topic_research.py`

- [x] Add "Research Strategies" section to academic researcher system prompt
- [x] Include BROADEN strategy (alternative terminology, related_papers, reformulation)
- [x] Include DEEPEN strategy (citation_search on seminal papers, methodological variations)
- [x] Include VALIDATE strategy (corroboration via search, citations, parallel studies)
- [x] Include SATURATE strategy (>50% duplicates = coverage sufficient, call research_complete)
- [x] Gate on `profile.enable_citation_tools == True`

### 4b. Strategy provenance logging
- [x] Pattern-match think output for strategy keywords (BROADEN, DEEPEN, VALIDATE, SATURATE)
- [x] Log detected strategy choice in provenance

### Item 4 Testing
- [x] Unit test: academic researcher prompt includes strategy guidance
- [x] Unit test: general researcher prompt does NOT include strategy guidance
- [x] Unit test: strategy keywords in think output captured in provenance

---

## Item 5: Adaptive Provider Selection
> **Depends on**: Item 1 (OpenAlex must be available as a hint target)

### 5a. Extract provider hints from brief
> **File**: `phases/brief.py`

- [x] Add `_extract_provider_hints(brief, profile)` method
- [x] Map discipline signals: biomedical/clinical/health → pubmed
- [x] Map: computer science/machine learning → semantic_scholar
- [x] Map: education → openalex
- [x] Map: social science/economics → openalex
- [x] Respect explicit profile providers (no override)

### 5b. Apply hints to session state
- [x] If `not profile.providers_explicitly_set`: augment provider list with hints
- [x] Deduplicate provider list
- [x] Silently drop unknown/unavailable provider hints

### 5c. Use active providers in delegation
> **Files**: `phases/supervision.py`, `phases/supervision_prompts.py`

- [x] Pass `state.active_providers` when creating topic researcher tasks
- [x] Modify `build_delegation_user_prompt()` to include available providers list

### Item 5 Testing
- [x] Unit test: biomedical brief triggers PubMed hint
- [x] Unit test: education brief triggers OpenAlex hint
- [x] Unit test: explicit profile providers not overridden by hints
- [x] Unit test: hints are additive (don't remove existing providers)
- [x] Unit test: unknown/unavailable provider hints silently dropped

---

## Item 6: Per-Provider Rate Limiting
> **Parallel with**: All items (independent infrastructure)

### 6a. Resilience configs for new providers
> **File**: `providers/resilience/config.py`

- [ ] Add `openalex` entry to `PROVIDER_CONFIGS` (50 req/s, burst 10, 3 retries)
- [ ] Add `crossref` entry to `PROVIDER_CONFIGS` (10 req/s, burst 5, 3 retries)

### 6b. Provider config fields
> **File**: `config/research.py`

- [ ] Add `openalex_api_key: Optional[str] = None`
- [ ] Add `openalex_enabled: bool = True`
- [ ] Add `crossref_email: Optional[str] = None`
- [ ] Add `crossref_enabled: bool = True`

### Item 6 Testing
- [ ] Unit test: new providers get correct default rate limits
- [ ] Unit test: config overrides default rate limits
- [ ] Unit test: unavailable providers excluded from provider chain

---

## Final Validation

- [ ] All new providers registered and importable
- [ ] OpenAlex search returns correctly mapped `ResearchSource` objects
- [ ] Crossref enrichment fills gaps without overwriting
- [ ] Citation tools gated by profile flag
- [ ] Strategic primitives appear only in academic prompts
- [ ] Adaptive selection augments (not replaces) provider chains
- [ ] Rate limiting configured for all new providers
- [ ] Full existing test suite passes with zero modifications
- [ ] All new tests pass
