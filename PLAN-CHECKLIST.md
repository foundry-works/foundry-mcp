# PLAN-4 Checklist: Deep Analysis

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed. All three items are independent and can be worked in parallel.

---

## Item 1: Full-Text PDF Analysis

### 1a. Academic paper section detection
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py` (EXTEND)

- [x] Add `detect_sections()` method to `PDFExtractor`
  - [x] Regex patterns for standard section headers (Abstract, Introduction, Methods, Results, Discussion, Conclusion, References)
  - [x] Return `dict[str, tuple[int, int]]` — section name to (start_char, end_char)
  - [x] Graceful fallback: empty dict when no sections detected
- [x] Add unit test: section detection from synthetic academic PDF text
- [x] Add unit test: graceful fallback when section detection finds nothing

### 1b. Prioritized extraction mode
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py` (EXTEND)

- [x] Add `extract_prioritized()` method to `PDFExtractor`
  - [x] Accepts `max_chars` (default 50000) and `priority_sections` list
  - [x] Always includes abstract; prioritizes methods/results/discussion
  - [x] Truncates gracefully when full text exceeds `max_chars`
- [x] Add unit test: prioritized extraction respects max_chars and section ordering

### 1c. Integrate PDF extraction into extract_content tool
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Detect PDF URLs (patterns: `*.pdf`, `arxiv.org/pdf/*`, Content-Type header)
- [x] Route PDF URLs to `PDFExtractor.extract_from_url()` instead of Tavily Extract
- [x] Use `extract_prioritized()` for section-aware content within context limits
- [x] Preserve page boundaries in source metadata for locator support
- [x] Add unit test: PDF URL routing in extract_content tool

### 1d. PDF-aware tool (profile-gated)
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add `extract_pdf` tool when profile has `enable_pdf_extraction == True`
- [x] Tool accepts URL and max_pages, returns full paper text with page numbers and section structure
- [x] Only available in `systematic-review` and `bibliometric` profiles

### 1e. Page-aware digest
> **File**: `src/foundry_mcp/core/research/document_digest/digestor.py`

- [x] Use `page:N:char:start-end` locators for evidence snippets from PDF content
- [x] Use `detect_sections()` to prioritize Methods, Results, Discussion
- [x] Handle academic paper structure in digest output

> Note: The digestor already supports `page_boundaries` parameter. PDF sources now store
> page boundaries in source metadata, which can be passed through to the digest pipeline.

### 1f. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_pdf_max_pages: int = 50`
- [x] Add `deep_research_pdf_priority_sections: list[str] = ["methods", "results", "discussion"]`

### Item 1 Validation

- [x] Section detection works on synthetic academic PDF text
- [x] Prioritized extraction respects max_chars and section ordering
- [x] PDF URLs are routed correctly in extract_content tool
- [x] Page-aware locators appear in digest output for PDF content
- [x] Profile gating works — `extract_pdf` tool only in systematic-review/bibliometric
- [x] Existing PDFExtractor tests pass unchanged (81/81)
- [ ] Integration test: topic researcher extracts PDF and includes in findings
- [x] ~80-120 LOC new tests written (30 tests, ~300 LOC)

---

## Item 2: Citation Network / Connected Papers Graph

### 2a. Citation network model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `CitationNode` model (paper_id, title, authors, year, citation_count, is_discovered, source_id, role)
- [x] Add `CitationEdge` model (citing_paper_id, cited_paper_id)
- [x] Add `CitationNetwork` model (nodes, edges, foundational_papers, research_threads, stats)

### 2b. Add to ResearchExtensions
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `citation_network: Optional[CitationNetwork] = None` to `ResearchExtensions`
- [x] Add convenience accessor `@property citation_network` on `DeepResearchState`

### 2c. Network builder
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py` (NEW)

- [x] Create `CitationNetworkBuilder` class
- [x] Implement `build_network()` — fetch refs/cites, build graph, classify
  - [x] Fetch references for each source via OpenAlex (primary) / Semantic Scholar (fallback)
  - [x] Fetch citations for each source
  - [x] Build node and edge lists
  - [x] Respect `max_references_per_paper` and `max_citations_per_paper` caps
  - [x] Concurrency control (`max_concurrent` parameter)
- [x] Implement `_identify_foundational_papers()` — cited by 3+ discovered papers (or 30%)
- [x] Implement `_identify_research_threads()` — connected components via BFS/union-find (3+ nodes)
- [x] Implement `_classify_roles()` — foundational, discovered, extension, peripheral
- [x] Add unit test: network builder with mocked provider responses
- [x] Add unit test: foundational paper identification
- [x] Add unit test: research thread detection with known graph
- [x] Add unit test: role classification

### 2d. Dedicated action handler
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `_handle_deep_research_network()` handler
  - [x] Load completed research state
  - [x] Filter to academic sources with paper IDs
  - [x] Return `{"status": "skipped", "reason": "fewer than 3 academic sources"}` if < 3
  - [x] Build network and save to `state.extensions.citation_network`
  - [x] Return network data
- [x] Wire up `"deep-research-network"` action in the research router
- [x] Add unit test: action handler wiring and response format
- [x] Add unit test: graceful handling when < 3 academic sources

### 2e. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_citation_network_max_refs_per_paper: int = 20`
- [x] Add `deep_research_citation_network_max_cites_per_paper: int = 20`

### Item 2 Validation

- [x] Citation network models serialize/deserialize correctly
- [x] Network builder produces correct graph from mocked API responses
- [x] Foundational papers identified correctly (cited by 3+ discovered)
- [x] Research threads detected via connected components
- [x] Action handler returns correct response format
- [x] Graceful skip when < 3 academic sources
- [x] Integration test: 5 academic sources -> network with edges
- [x] ~100-140 LOC new tests written (35 tests, ~220 LOC)

---

## Item 3: Methodology Quality Assessment (Experimental)

### 3a. Methodology assessment model
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [x] Add `StudyDesign` enum (meta_analysis, systematic_review, rct, quasi_experimental, cohort, case_control, cross_sectional, qualitative, case_study, theoretical, opinion, unknown)
- [x] Add `MethodologyAssessment` model
  - [x] Fields: source_id, study_design, sample_size, sample_description, effect_size, statistical_significance, limitations_noted, potential_biases, confidence, content_basis
  - [x] No numeric rigor score — qualitative metadata only

### 3b. Assessment engine
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py` (NEW)

- [x] Create `MethodologyAssessor` class
- [x] Implement `assess_sources()` method
  - [x] Filter to ACADEMIC sources with content > 200 chars
  - [x] Batch LLM call for metadata extraction
  - [x] Force `confidence="low"` for abstract-only content
  - [x] Respect `timeout` parameter
- [x] Add unit test: LLM extraction prompt parsing with mocked responses
- [x] Add unit test: StudyDesign classification from various abstracts
- [x] Add unit test: graceful handling for sources without sufficient content
- [x] Add unit test: confidence forced to "low" for abstract-only content

### 3c. LLM extraction prompt

- [x] Design structured JSON extraction prompt
  - [x] Covers: study_design, sample_size, sample_description, effect_size, statistical_significance, limitations, potential_biases, confidence
  - [x] Instructs LLM to use null/empty for missing information (no guessing)

### 3d. Add to ResearchExtensions
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `methodology_assessments: list[MethodologyAssessment] = Field(default_factory=list)` to `ResearchExtensions`
- [x] Add convenience accessor `@property methodology_assessments` on `DeepResearchState`

### 3e. Feed assessments into synthesis
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] When assessments available, inject `## Methodology Context` section into synthesis prompt
  - [x] Format: study design, sample size, effect size, limitations per source
  - [x] Note content basis (full text vs abstract) with confidence caveat
  - [x] Frame as "context" for qualitative weighting, not ground truth
- [x] Add unit test: assessment data correctly injected into synthesis prompt

### 3f. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_methodology_assessment_provider: Optional[str] = None`
- [x] Add `deep_research_methodology_assessment_timeout: float = 60.0`
- [x] Add `deep_research_methodology_assessment_min_content_length: int = 200`

### Item 3 Validation

- [x] StudyDesign enum covers common study types
- [x] MethodologyAssessment model serializes/deserializes correctly
- [x] Assessor extracts structured metadata from mocked LLM responses
- [x] Confidence forced to "low" for abstract-only content
- [x] Assessment context correctly injected into synthesis prompt
- [x] Sources with < 200 chars content are skipped
- [ ] Integration test: end-to-end assessment of 5 academic sources
- [x] ~80-100 LOC new tests written (41 tests, ~370 LOC)

---

## Final Validation

- [x] All three items implemented and tested independently
- [x] All features are profile-gated (opt-in only)
- [x] No new dependencies added
- [x] Existing tests pass unchanged
- [x] New config fields added with sensible defaults
- [x] ResearchExtensions updated with citation_network and methodology_assessments
- [x] Total ~240-360 LOC new tests written

---

## Estimated Scope

| Item | Impl LOC | Test LOC |
|------|----------|----------|
| 1. PDF Analysis (extend) | ~150-200 | ~80-120 |
| 2. Citation Network | ~300-400 | ~100-140 |
| 3. Methodology (experimental) | ~200-300 | ~80-100 |
| **Total** | **~650-900** | **~240-360** |
