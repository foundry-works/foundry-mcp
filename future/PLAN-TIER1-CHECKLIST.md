# PLAN-TIER1 Implementation Checklist

> Track progress for each Tier 1 improvement. Check items as completed.

---

## 1. Add `literature_review` Query Type to Synthesis

- [ ] Add `_LITERATURE_REVIEW_PATTERNS` regex to `phases/synthesis.py`
- [ ] Add `literature_review` check in `_classify_query_type()` (before `explanation` fallback)
- [ ] Add ResearchMode-aware override: if ACADEMIC mode + ambiguous query -> prefer literature_review
- [ ] Add `"literature_review"` entry to `_STRUCTURE_GUIDANCE` dict with full section template
- [ ] Modify `_build_synthesis_system_prompt()` to inject academic-specific instructions when query_type is `literature_review`
- [ ] Add instructions for: thematic organization, author+year inline citations, seminal work identification, methodology trends, APA references
- [ ] Unit test: `_classify_query_type("literature review on X")` -> `"literature_review"`
- [ ] Unit test: `_classify_query_type("what does the research say about X")` -> `"literature_review"`
- [ ] Unit test: `_classify_query_type("survey of prior work on X")` -> `"literature_review"`
- [ ] Unit test: `_classify_query_type("existing research on X in Y contexts")` -> `"literature_review"`
- [ ] Unit test: generic query still returns `"explanation"` (no regression)
- [ ] Unit test: comparison query still returns `"comparison"` (no regression)
- [ ] Verify structure guidance is correctly injected into synthesis system prompt

## 2. Format Citations with Full Bibliographic Metadata (APA Style)

- [ ] Add `format_source_apa(source: ResearchSource) -> str` to `_citation_postprocess.py`
- [ ] Handle full metadata case: Authors (Year). Title. *Venue*. DOI_URL
- [ ] Handle partial metadata: Title. URL (graceful degradation)
- [ ] Handle >5 authors with "et al." formatting
- [ ] Handle missing year (use "n.d." per APA convention)
- [ ] Handle missing venue (omit journal/conference)
- [ ] Add `format_style` parameter to `build_sources_section()`
- [ ] When `format_style="apa"`: use `format_source_apa()`, header "## References"
- [ ] When `format_style="default"`: preserve existing `[N] [Title](URL)` behavior
- [ ] Modify `postprocess_citations()` to accept and pass `format_style`
- [ ] Auto-select `format_style="apa"` when research_mode is ACADEMIC or query_type is `literature_review`
- [ ] Pass query_type from synthesis phase through to `postprocess_citations()`
- [ ] Also handle `strip_llm_sources_section()` recognizing "## References" header (already handled, verify)
- [ ] Unit test: `format_source_apa()` with full Semantic Scholar metadata
- [ ] Unit test: `format_source_apa()` with web-only source (title + URL)
- [ ] Unit test: `format_source_apa()` with partial metadata (missing venue)
- [ ] Unit test: `format_source_apa()` with >5 authors
- [ ] Unit test: `build_sources_section()` with `format_style="apa"` produces "## References" header
- [ ] Integration test: end-to-end synthesis in ACADEMIC mode produces APA references

## 3. Add Citation Graph and Related Papers Tools

### Semantic Scholar Provider
- [ ] Add `get_citations(paper_id, max_results)` method to `SemanticScholarProvider`
- [ ] Implement `/paper/{paper_id}/citations` endpoint call
- [ ] Parse citation response into `list[ResearchSource]` with `SourceType.ACADEMIC`
- [ ] Add `get_recommendations(paper_id, max_results)` method
- [ ] Implement `/recommendations/v1/papers/` POST endpoint call
- [ ] Parse recommendations response into `list[ResearchSource]`
- [ ] Add `get_paper(paper_id)` single-paper lookup method
- [ ] Support DOI, ArXiv ID, and S2 paper ID formats
- [ ] Apply resilience patterns (retry, circuit breaker) to new endpoints
- [ ] Unit test: `get_citations()` with mocked HTTP response
- [ ] Unit test: `get_recommendations()` with mocked HTTP response
- [ ] Unit test: `get_paper()` with DOI format input
- [ ] Unit test: rate limiting respected for new endpoints

### Tool Models
- [ ] Add `CitationSearchTool` Pydantic model to `models/deep_research.py`
- [ ] Add `RelatedPapersTool` Pydantic model
- [ ] Register both in `RESEARCHER_TOOL_SCHEMAS` dict
- [ ] Decide budget treatment: count against tool call limit (recommended: yes)

### Topic Researcher Integration
- [ ] Add `citation_search` and `related_papers` tool documentation to system prompt
- [ ] Make tool injection conditional on `state.research_mode == ResearchMode.ACADEMIC`
- [ ] Add `_handle_citation_search_tool()` dispatch handler
- [ ] Add `_handle_related_papers_tool()` dispatch handler
- [ ] Add elif clauses in main tool dispatch loop (after `extract_content`, before `research_complete`)
- [ ] Format tool results with novelty tracking (reuse existing dedup logic)
- [ ] Assign `SourceType.ACADEMIC` to all sources from these tools
- [ ] Unit test: tool dispatch correctly routes to citation_search handler
- [ ] Unit test: tool dispatch correctly routes to related_papers handler
- [ ] Unit test: tools are NOT available when research_mode is GENERAL
- [ ] Integration test: topic researcher chains web_search -> citation_search

## 4. Academic Brief Enrichment

- [ ] Modify `_build_brief_system_prompt()` signature to accept `research_mode` parameter
- [ ] Add academic-specific dimensions: disciplinary scope, time period, methodology, population, source hierarchy
- [ ] Ensure GENERAL mode prompt is unchanged (no regression)
- [ ] Update `_execute_brief_async()` to pass `state.research_mode` to prompt builder
- [ ] Modify supervision `_build_first_round_delegation_system_prompt()` for ACADEMIC mode
- [ ] Add academic decomposition guidelines: foundational works directive, recent studies directive, per-discipline directives
- [ ] Ensure supervision GENERAL mode prompt is unchanged (no regression)
- [ ] Unit test: brief system prompt includes academic dimensions when ACADEMIC
- [ ] Unit test: brief system prompt is original when GENERAL
- [ ] Unit test: supervision prompt includes academic guidelines when ACADEMIC

## 5. Expose `research_mode` as Request-Time Parameter

- [ ] Add `research_mode: Optional[str] = None` to `DeepResearchWorkflow.execute()` in `core.py`
- [ ] Add `research_mode` to `_handle_deep_research()` signature in `handlers_deep_research.py`
- [ ] Pass `research_mode` through action router in `research.py`
- [ ] Update state initialization in `action_handlers.py` to use request param with config fallback
- [ ] Validate `research_mode` value against `ResearchMode` enum values
- [ ] Raise clear error for invalid research_mode values
- [ ] Unit test: request `research_mode="academic"` overrides config default "general"
- [ ] Unit test: omitting `research_mode` uses config fallback
- [ ] Unit test: invalid `research_mode` value raises ValueError
- [ ] Verify research_mode flows through to state and affects brief/supervision prompts

## 6. BibTeX/RIS Export

### BibTeX Generator
- [ ] Create `src/foundry_mcp/core/research/export/__init__.py`
- [ ] Create `src/foundry_mcp/core/research/export/bibtex.py`
- [ ] Implement `sources_to_bibtex(sources: list[ResearchSource]) -> str`
- [ ] Implement `source_to_bibtex_entry(source, citation_key) -> str`
- [ ] Generate stable citation keys: `firstauthor_year_firsttitleword`
- [ ] Handle entry types: `@article` (journal), `@inproceedings` (conference), `@misc` (web/other)
- [ ] Map metadata fields: author, title, journal/booktitle, year, doi, url, abstract
- [ ] Handle special characters in BibTeX (escape `&`, `%`, `#`, `_`, `{`, `}`)
- [ ] Handle missing fields gracefully (omit rather than empty)
- [ ] Unit test: full metadata -> valid BibTeX entry
- [ ] Unit test: minimal metadata -> valid @misc entry
- [ ] Unit test: special characters escaped
- [ ] Unit test: citation keys are unique across sources
- [ ] Unit test: multiple sources produce valid concatenated BibTeX

### RIS Generator
- [ ] Create `src/foundry_mcp/core/research/export/ris.py`
- [ ] Implement `sources_to_ris(sources: list[ResearchSource]) -> str`
- [ ] Map fields: TY, AU, TI, JO, PY, DO, UR, AB, ER
- [ ] Handle entry types: JOUR (journal), CONF (conference), ELEC (web/electronic)
- [ ] Unit test: full metadata -> valid RIS block
- [ ] Unit test: minimal metadata -> valid ELEC entry
- [ ] Unit test: multiple sources produce valid concatenated RIS

### Export Action Handler
- [ ] Add `_handle_deep_research_export()` to `handlers_deep_research.py`
- [ ] Load persisted `DeepResearchState` by research_id
- [ ] Filter sources (all vs. academic-only, configurable)
- [ ] Generate export in requested format (bibtex/ris)
- [ ] Return export string in response envelope
- [ ] Wire up "deep-research-export" action in research action router
- [ ] Unit test: export handler returns BibTeX for completed session
- [ ] Unit test: export handler returns error for non-existent session
- [ ] Unit test: export handler returns error for incomplete session

### Synthesis Integration
- [ ] After synthesis completes, generate BibTeX from `state.sources`
- [ ] Include BibTeX string in result metadata: `metadata["bibtex"]`
- [ ] Include RIS string in result metadata: `metadata["ris"]`

## Cross-Cutting

- [ ] Add `deep_research_citation_style` config field (default: "default")
- [ ] Add `deep_research_export_format` config field (default: "bibtex")
- [ ] Update `from_toml_dict()` to parse new config fields
- [ ] Add validation for citation_style and export_format values
- [ ] Update dev_docs if applicable (reference new academic features)
- [ ] Run full test suite: `pytest tests/core/research/` — all pass
- [ ] Run integration tests: `pytest tests/integration/test_deep_research_*.py` — all pass
