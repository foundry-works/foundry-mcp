# PLAN-TIER3 Implementation Checklist

> Track progress for each Tier 3 improvement. Check items as completed.
>
> **Prerequisites**: Tier 1 (all items) and Tier 2 items 6-7 are strong prerequisites.
>
> **Note**: All Tier 3 features are opt-in (disabled by default).

---

## 10. Full-Text PDF Analysis

### PDF Extract Provider
- [ ] Create `src/foundry_mcp/core/research/providers/pdf_extract.py`
- [ ] Add `PDFPage` model (page_number, text, char_offset)
- [ ] Add `PDFExtractionResult` model (pages, total_pages, total_chars, metadata)
- [ ] Implement `PDFExtractProvider` class
- [ ] Implement `extract(url, max_pages, timeout)` method
- [ ] Download PDF via httpx with size limit (50MB default)
- [ ] Detect Content-Type to confirm PDF (reject non-PDFs)
- [ ] Follow redirects (DOI -> publisher -> PDF)
- [ ] Handle 403/paywall responses gracefully (return empty result, not error)
- [ ] Extract text via pymupdf (fitz) with page boundaries
- [ ] Calculate cumulative char_offset per page
- [ ] Extract PDF metadata (title, author, creation date) from document properties
- [ ] Detect scanned PDFs (pages with no extractable text) and skip
- [ ] Cap at max_pages (default 50)
- [ ] Add timeout handling for slow downloads

### Topic Researcher Integration
- [ ] Detect PDF URLs in `_handle_extract_tool()`: arxiv.org/pdf/*, *.pdf, metadata.pdf_url
- [ ] Route PDF URLs to `PDFExtractProvider` instead of `TavilyExtractProvider`
- [ ] Convert `PDFExtractionResult` to standard content format (concatenated text with page markers)
- [ ] Preserve page boundaries in source metadata: `metadata.pages = [{page: N, char_offset: M}]`
- [ ] Only attempt PDF extraction when `config.deep_research_pdf_extraction_enabled`
- [ ] Fallback to Tavily extract if PDF extraction fails

### Alternative: Dedicated extract_pdf tool
- [ ] (Optional) Add `ExtractPDFTool` Pydantic model
- [ ] (Optional) Add tool documentation to researcher system prompt (ACADEMIC mode only)
- [ ] (Optional) Add `_handle_extract_pdf_tool()` dispatch handler
- [ ] (Optional) Route to `PDFExtractProvider`

### Page-Aware Digest
- [ ] Modify `document_digest/digestor.py` to use page-based locators for PDF sources
- [ ] Generate `page:N:char:start-end` locators instead of `char:start-end`
- [ ] Detect PDF sources by checking `metadata.pages` or `metadata.pdf_extracted`
- [ ] Prioritize Methods, Results, Discussion sections for evidence extraction
- [ ] Handle academic paper structure awareness (optional heuristic section detection)

### Configuration
- [ ] Add `deep_research_pdf_extraction_enabled: bool = False` to config
- [ ] Add `deep_research_pdf_max_pages: int = 50`
- [ ] Add `deep_research_pdf_max_size_mb: float = 50.0`
- [ ] Add `deep_research_pdf_timeout: float = 30.0`
- [ ] Parse in `from_toml_dict()`
- [ ] Validate: max_pages > 0, max_size_mb > 0, timeout > 0

### Dependencies
- [ ] Add `pymupdf` to project dependencies (pyproject.toml / requirements)
- [ ] Make it optional: `pip install foundry-mcp[pdf]` or graceful ImportError handling
- [ ] Test with pymupdf unavailable -> feature disabled, clear error message

### Testing
- [ ] Create test PDF fixture (simple academic paper format, 3 pages)
- [ ] Unit test: extract text from test PDF -> correct page count, text content
- [ ] Unit test: page boundary preservation -> char_offset values correct
- [ ] Unit test: PDF metadata extraction -> title, author present
- [ ] Unit test: max_pages cap -> only first N pages extracted
- [ ] Unit test: non-PDF URL -> graceful rejection
- [ ] Unit test: scanned PDF detection -> returns empty with warning
- [ ] Unit test: oversized PDF -> size limit enforced
- [ ] Unit test: timeout handling -> raises within configured timeout
- [ ] Unit test: paywall 403 response -> returns None, not exception
- [ ] Unit test: PDF URL detection regex (arxiv.org/pdf/*, *.pdf extension)
- [ ] Integration test: topic researcher extracts PDF and includes content in findings
- [ ] Integration test: digest produces page-based locators for PDF sources

---

## 11. Citation Network / Connected Papers Graph

### Citation Network Model
- [ ] Add `CitationNode` model to `models/deep_research.py`
  - [ ] Fields: paper_id, title, authors, year, citation_count, is_discovered, source_id, role
  - [ ] Role enum: "foundational" | "discovered" | "extension" | "peripheral"
- [ ] Add `CitationEdge` model (citing_paper_id, cited_paper_id)
- [ ] Add `CitationNetwork` model
  - [ ] Fields: nodes, edges, clusters, foundational_papers, research_threads
- [ ] Add `citation_network: Optional[CitationNetwork]` to `DeepResearchState`

### Semantic Scholar API Extension
- [ ] Verify Tier 1 item 3 is complete (get_citations, get_recommendations methods)
- [ ] Add `get_references(paper_id, max_results)` method if not already present
  - [ ] Endpoint: GET /paper/{paper_id}/references
  - [ ] Returns papers cited BY the given paper
- [ ] Handle rate limiting: batch requests with 1 RPS cap
- [ ] Handle missing papers gracefully (404 -> skip, don't fail)

### Network Builder
- [ ] Create `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py`
- [ ] Implement `CitationNetworkBuilder` class
- [ ] Implement `build_network(sources, provider, max_refs, max_cites, max_concurrent)`
  - [ ] Filter to academic sources with `metadata.paper_id`
  - [ ] For each source: fetch references and citations in parallel (within rate limits)
  - [ ] Build node list (discovered + referenced papers)
  - [ ] Build edge list (citing -> cited relationships)
  - [ ] Deduplicate nodes by paper_id
- [ ] Implement `_identify_foundational_papers(nodes, edges)`
  - [ ] Papers cited by 3+ discovered papers
  - [ ] Sorted by in-degree among discovered papers
- [ ] Implement `_identify_clusters(nodes, edges)`
  - [ ] Connected components on undirected citation graph
  - [ ] Filter to components with 2+ discovered papers
  - [ ] Assign cluster names based on common keywords in titles
- [ ] Implement `_classify_roles(nodes, edges, discovered_ids)`
  - [ ] "foundational": cited by many discovered, not discovered itself
  - [ ] "discovered": in state.sources
  - [ ] "extension": cites many discovered papers, published recently
  - [ ] "peripheral": weakly connected
- [ ] Implement `_identify_research_threads(clusters, nodes, edges)`
  - [ ] Sequences of papers linked by citation chains
  - [ ] Named by common theme/methodology

### Workflow Integration
- [ ] Add post-synthesis enrichment step in `workflow_execution.py`
- [ ] Condition: `research_mode == ACADEMIC and config.deep_research_citation_network_enabled`
- [ ] Call `CitationNetworkBuilder.build_network()`
- [ ] Store result in `state.citation_network`
- [ ] Persist updated state
- [ ] Include network in deep-research-report response
- [ ] Emit audit event for citation network phase

### Configuration
- [ ] Add `deep_research_citation_network_enabled: bool = False`
- [ ] Add `deep_research_citation_network_max_refs_per_paper: int = 20`
- [ ] Add `deep_research_citation_network_max_cites_per_paper: int = 20`
- [ ] Parse in `from_toml_dict()`

### Testing
- [ ] Unit test: build_network with 5 mock sources -> nodes and edges created
- [ ] Unit test: foundational paper identification (paper cited by 4/5 discovered)
- [ ] Unit test: cluster identification with 2 distinct groups of papers
- [ ] Unit test: role classification (foundational, discovered, extension, peripheral)
- [ ] Unit test: handles 0 academic sources -> empty network
- [ ] Unit test: handles sources without paper_id -> skipped gracefully
- [ ] Unit test: rate limiting respected (mock provider tracks call timing)
- [ ] Unit test: 404 for missing paper -> node skipped, no failure
- [ ] Unit test: JSON serialization roundtrip for CitationNetwork
- [ ] Integration test: 5 academic sources -> network with foundational papers identified

---

## 12. Methodology Quality Assessment

### Study Design Model
- [ ] Add `StudyDesign` enum to `models/sources.py`
  - [ ] Values: META_ANALYSIS, SYSTEMATIC_REVIEW, RCT, QUASI_EXPERIMENTAL, COHORT, CASE_CONTROL, CROSS_SECTIONAL, QUALITATIVE, CASE_STUDY, THEORETICAL, OPINION, UNKNOWN
  - [ ] Order reflects typical evidence hierarchy
- [ ] Add `MethodologyAssessment` model to `models/sources.py`
  - [ ] Fields: source_id, study_design, sample_size, sample_description, effect_size, statistical_significance, limitations_noted, potential_biases, rigor_score, confidence
- [ ] Add `methodology_assessments: list[MethodologyAssessment]` to `DeepResearchState`

### Assessment Engine
- [ ] Create `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py`
- [ ] Implement `MethodologyAssessor` class
- [ ] Implement `assess_sources(sources, findings, provider_id, timeout)`
  - [ ] Filter to academic sources with content > 200 chars
  - [ ] Batch LLM calls (max_concurrent from config)
  - [ ] Parse structured JSON responses
  - [ ] Handle parse failures gracefully (mark assessment confidence as "low")
- [ ] Implement LLM extraction prompt
  - [ ] Input: source title, abstract/content, any extracted findings
  - [ ] Output: structured JSON with study design, sample, effect size, limitations, biases
  - [ ] Use cheap model (summarization-tier) for cost efficiency
- [ ] Implement `_compute_rigor_score(assessment) -> float`
  - [ ] Study design weight: 40% (hierarchy: meta_analysis=1.0 down to opinion=0.1)
  - [ ] Sample size weight: 20% (log-scaled: N>=1000=1.0, N>=100=0.7, N>=30=0.5, N<30=0.3)
  - [ ] Statistical reporting: 20% (has effect_size + p-value = 1.0, partial = 0.5, none = 0.0)
  - [ ] Limitation acknowledgment: 10% (1+ limitations = 0.8, 2+ = 1.0, 0 = 0.3)
  - [ ] Bias awareness: 10% (1+ biases noted = 0.8, 0 = 0.3)
- [ ] Handle edge cases: theoretical papers (no sample), opinion pieces (no methodology)

### Synthesis Integration
- [ ] Feed methodology assessments into synthesis user prompt (ACADEMIC mode)
- [ ] Format: table or bullet list with source citation, design, sample, rigor score
- [ ] Include in synthesis system prompt: "Weight higher-rigor studies more heavily"
- [ ] Add assessment data to ResearchLandscape (Tier 2, item 7) if available:
  - [ ] `methodology_breakdown` populated from assessments
  - [ ] Average rigor score as a quality metric

### Workflow Integration
- [ ] Add post-gathering, pre-synthesis assessment step in `workflow_execution.py`
- [ ] Condition: `research_mode == ACADEMIC and config.deep_research_methodology_assessment_enabled`
- [ ] Run assessor with `state.sources` and `state.findings`
- [ ] Store in `state.methodology_assessments`
- [ ] Persist updated state
- [ ] Emit audit event

### Configuration
- [ ] Add `deep_research_methodology_assessment_enabled: bool = False`
- [ ] Add `deep_research_methodology_assessment_provider: Optional[str] = None`
- [ ] Add `deep_research_methodology_assessment_model: Optional[str] = None`
- [ ] Add `deep_research_methodology_assessment_timeout: float = 60.0`
- [ ] Parse in `from_toml_dict()`
- [ ] Route to cheap model by default (summarization-tier in role resolution)

### Testing
- [ ] Unit test: `_compute_rigor_score()` for RCT with full stats -> high score (>0.8)
- [ ] Unit test: `_compute_rigor_score()` for case study with no stats -> low score (<0.4)
- [ ] Unit test: `_compute_rigor_score()` for meta-analysis -> highest score (>0.9)
- [ ] Unit test: `_compute_rigor_score()` for theoretical paper -> moderate score (~0.5)
- [ ] Unit test: assessment extraction with mocked LLM response
- [ ] Unit test: parse failure handling (malformed JSON -> confidence "low", design UNKNOWN)
- [ ] Unit test: filter to academic sources only (web sources skipped)
- [ ] Unit test: filter by content length (sources < 200 chars skipped)
- [ ] Unit test: assessment data correctly formatted in synthesis prompt
- [ ] Unit test: MethodologyAssessment model validation
- [ ] Integration test: 5 academic sources -> assessments with rigor scores

---

## Cross-Cutting

### Dependencies
- [ ] Add `pymupdf` as optional dependency for PDF extraction
- [ ] Handle ImportError gracefully when pymupdf not installed
- [ ] Document dependency in README or dev_docs

### Configuration Validation
- [ ] All new config fields have sensible defaults (all opt-in, disabled)
- [ ] Validate boolean fields, numeric ranges, optional string fields
- [ ] New features are completely inert when disabled

### Performance
- [ ] PDF extraction: verify 50-page PDF processes within 30s timeout
- [ ] Citation network: verify 15-source network builds within 60s (rate-limited)
- [ ] Methodology assessment: verify 10-source batch completes within 60s
- [ ] All new features respect cancellation (`_check_cancellation()` calls)

### Regression
- [ ] `pytest tests/core/research/` passes with all Tier 3 features disabled
- [ ] `pytest tests/core/research/` passes with all Tier 3 features enabled (on test data)
- [ ] `pytest tests/integration/test_deep_research_*.py` passes unchanged
- [ ] GENERAL mode behavior completely unchanged by Tier 3 additions
- [ ] TECHNICAL mode behavior unchanged
- [ ] ACADEMIC mode without Tier 3 features matches Tier 1+2 behavior exactly
