# Academic Deep Research Platform — Checklist

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed. See [future/](future/) for full design details.

---

## Phase 0: Prerequisites

### Item 0.1: Complete Supervision Refactoring

#### 0.1a. Extract first-round decomposition pipeline
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_first_round.py` (NEW)

- [x] Extract `_first_round_decompose_critique_revise()` as standalone async function
- [x] Extract `_run_first_round_generate()` as standalone async function
- [x] Extract `_run_first_round_critique()` as standalone async function
- [x] Extract `_run_first_round_revise()` as standalone async function
- [x] Functions take explicit parameters (state, config, memory) — no `self`
- [x] `supervision.py` calls into extracted module
- [x] All existing supervision tests pass with zero changes
- [x] Add unit test: import `supervision_first_round` module independently
- [x] Add unit test: first-round pipeline produces same results from extracted module

#### 0.1b. Evaluate further delegation extraction (optional)
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_helpers.py` (optional NEW)

- [ ] Evaluate whether extracting these helpers reduces complexity enough to justify:
  - [ ] `_compress_directive_results_inline()`
  - [ ] `_build_directive_fallback_summary()`
  - [ ] `_build_evidence_inventory()`
- [x] Decision: skip — first-round extraction provided sufficient decomposition

#### 0.1c. Remove thin wrapper methods
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

- [ ] Inline thin wrapper methods (lines ~2107-2174) at their call sites
- [ ] Verify `supervision.py` is under 1,800 lines (down from 2,174)

#### Item 0.1 Validation

- [ ] `supervision.py` is under 1,800 lines (currently ~1,880)
- [x] All existing tests pass with zero changes (zero-diff verification)
- [x] `from phases.supervision import SupervisionPhaseMixin` still works
- [x] Imports from `supervision_coverage` and `supervision_prompts` unchanged
- [x] ~30-50 LOC new structural tests written

---

### Item 0.2: ResearchExtensions Container Model

#### 0.2a. Add ResearchExtensions model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `ResearchExtensions(BaseModel)` with `exclude_none = True`
- [x] Initially empty placeholder fields (populated by subsequent plans)

#### 0.2b. Add extensions field to DeepResearchState
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `extensions: ResearchExtensions = Field(default_factory=ResearchExtensions)`

#### 0.2c. Add convenience accessors
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `@property research_profile` accessor
- [x] Add `@property provenance` accessor
- [x] (Other accessors added as subsequent plans populate fields)

#### Item 0.2 Validation

- [x] `ResearchExtensions()` default serializes to `{}`
- [x] Extensions with one field populated serializes only that field
- [x] `DeepResearchState` with default extensions is backward-compatible with existing serialized states
- [x] Convenience property accessors work correctly
- [x] ~40-60 LOC new tests written

---

## Phase 1: Foundations

### Item 1.1: Research Profiles

#### 1.1a. Add ResearchProfile model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `ResearchProfile(BaseModel)` with fields: name, providers, source_quality_mode, citation_style, export_formats, synthesis_template, enable_citation_tools, enable_methodology_assessment, enable_citation_network, enable_pdf_extraction, source_type_hierarchy, disciplinary_scope, time_period, methodology_preferences
- [x] Define built-in profiles: `general`, `academic`, `systematic-review`, `bibliometric`, `technical`

#### 1.1b. Add profile to ResearchExtensions
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `research_profile: Optional[ResearchProfile] = None` to `ResearchExtensions`
- [x] Add/update convenience accessor on `DeepResearchState`

#### 1.1c. Add profile registry to config
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_profiles: dict[str, dict]` field
- [x] Add `deep_research_default_profile: str = "general"` field

#### 1.1d. Profile resolution logic
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `resolve_profile()` function
  - [x] Look up built-in profiles by name
  - [x] Look up custom profiles from config
  - [x] Map legacy `research_mode` to profile for backward compat
  - [x] Apply per-request overrides on top of resolved profile
  - [x] Default to `deep_research_default_profile` from config

#### 1.1e. Add profile parameter to deep-research handler
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `research_profile: Optional[str]` parameter
- [x] Add `profile_overrides: Optional[dict]` parameter
- [x] Resolution order: `research_profile` > `research_mode` (legacy) > config default

#### 1.1f. Pass profile through action router
> **File**: `src/foundry_mcp/tools/unified/research.py`

- [x] Ensure `research_profile` and `profile_overrides` flow through to handler

#### Item 1.1 Validation

- [x] Profile resolution with built-in names
- [x] Profile resolution with legacy `research_mode` mapping
- [x] Per-request overrides applied on top of profile
- [x] Unknown profile name raises validation error
- [x] Backward compat: `research_mode="academic"` produces same behavior as `research_profile="academic"`
- [x] Config-defined custom profiles load correctly

---

### Item 1.2: Research Provenance Audit Trail

#### 1.2a. Add provenance models
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `ProvenanceEntry(BaseModel)` with: timestamp, phase, event_type, summary, details
- [x] Add `ProvenanceLog(BaseModel)` with: session_id, query, profile, profile_config, started_at, completed_at, entries, `append()` method

#### 1.2b. Add provenance to ResearchExtensions
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `provenance: Optional[ProvenanceLog] = None` to `ResearchExtensions`
- [x] Add/update convenience accessor on `DeepResearchState`

#### 1.2c. Initialize provenance at session creation

- [x] Populate session_id, query, profile, profile_config, started_at at creation time

#### 1.2d. Log brief generation
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [x] Append `brief_generated` event after brief is generated

#### 1.2e. Log supervision events
> **Files**: `phases/supervision.py`, `phases/supervision_coverage.py`, `phases/topic_research.py`

- [x] Log `decomposition` event after delegation response
- [x] Log `provider_query` event after each provider search
- [x] Log `source_discovered` event when sources added to state
- [x] Log `source_deduplicated` event when dedup occurs
- [x] Log `coverage_assessment` event after heuristic assessment
- [x] Log `gap_identified` event when gaps added
- [x] Log `gap_resolved` event when gaps resolved
- [x] Log `iteration_complete` event at end of each supervision round

#### 1.2f. Log synthesis events
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Log `synthesis_query_type` event after classification
- [x] Log `synthesis_completed` event after synthesis

#### 1.2g. Persist provenance separately
> **File**: `src/foundry_mcp/core/research/memory.py`

- [x] Save provenance as `deepres-{id}.provenance.json` alongside state
- [x] Load provenance when loading state

#### 1.2h. Expose provenance in response
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add provenance to `deep-research-report` response (when `include_provenance=True`)
- [x] Add dedicated `deep-research-provenance` action

#### Item 1.2 Validation

- [x] `ProvenanceLog.append()` creates timestamped entries
- [x] Provenance populated after brief phase
- [x] Provenance populated after supervision round
- [x] Provenance persisted and loadable
- [x] Provenance included in report response
- [x] Provenance serialization roundtrip

---

### Item 1.3: `literature_review` Query Type

#### 1.3a. Add detection pattern
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Add `_LITERATURE_REVIEW_PATTERNS` regex

#### 1.3b. Add classification check

- [x] Insert `literature_review` check in `_classify_query_type()` before existing patterns
- [x] Check profile's `synthesis_template` for forced type
- [x] Bias toward `literature_review` for ACADEMIC mode with ambiguous queries

#### 1.3c. Add structure guidance

- [x] Add `"literature_review"` entry to `_STRUCTURE_GUIDANCE` dict with sections: Executive Summary, Theoretical Foundations, Thematic Analysis, Methodological Approaches, Key Debates, Research Gaps, Conclusions, References

#### 1.3d. Add academic synthesis instructions

- [x] Inject additional instructions when `query_type == "literature_review"` (thematic organization, author/year/method notes, seminal works, methodological trends, conflicting studies, citation style formatting)

#### Item 1.3 Validation

- [x] `"literature review on X"` -> `"literature_review"`
- [x] `"what does the research say about X"` -> `"literature_review"`
- [x] `"survey of prior work on X"` -> `"literature_review"`
- [x] Profile with `synthesis_template="literature_review"` forces the type
- [x] Generic query still returns `"explanation"` (no regression)
- [x] Comparison query still returns `"comparison"` (no regression)
- [x] Structure guidance correctly injected

---

### Item 1.4: APA Citation Formatting

#### 1.4a. Add APA formatting function
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/_citation_postprocess.py`

- [x] Add `format_source_apa(source: ResearchSource) -> str`
  - [x] Full format: `Authors (Year). Title. *Venue*. DOI_URL`
  - [x] Handle >5 authors ("et al.")
  - [x] Handle missing year ("n.d.")
  - [x] Web source fallback: `Author/Organization (Year). Title. *Site Name*. URL`
  - [x] Minimal fallback: `Title. URL`

#### 1.4b. Add format_style parameter

- [x] Add `format_style` param to `build_sources_section()`
- [x] `"apa"` -> use `format_source_apa()`, section header "## References"
- [x] `"default"` -> preserve existing behavior exactly

#### 1.4c. Connect to profile

- [x] Read `state.research_profile.citation_style` in `postprocess_citations()`
- [x] Use `format_style="apa"` for `literature_review` query type regardless of profile

#### Item 1.4 Validation

- [x] Full academic metadata -> correct APA entry
- [x] Partial metadata (missing venue, DOI) -> graceful format
- [x] Web source (no academic metadata) -> fallback format
- [x] >5 authors -> "et al." handling
- [x] Missing year -> "n.d."
- [x] `format_style="apa"` produces "## References" header
- [x] `format_style="default"` preserves existing format
- [x] Integration: academic profile produces APA references

---

### Item 1.5: Academic Brief Enrichment

#### 1.5a. Profile-aware brief system prompt
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [x] Modify `_build_brief_system_prompt()` to accept profile
- [x] Append academic dimensions for ACADEMIC mode: disciplinary scope, time period, methodology preferences, education level, source type hierarchy
- [x] Inject profile-specified constraints as pre-filled values

#### 1.5b. Profile-aware decomposition
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_prompts.py`

- [x] Append academic decomposition guidelines in `build_first_round_delegation_system_prompt()` when academic
- [x] Include directives for seminal works, recent empirical studies, per-discipline separation, literature review section mapping

#### Item 1.5 Validation

- [x] Brief prompt includes academic dimensions when profile is academic
- [x] Brief prompt unchanged when profile is general
- [x] Profile-specified constraints injected into brief prompt
- [x] Supervision prompt includes academic guidelines when profile is academic
- [x] Supervision prompt unchanged when profile is general

---

### Item 1.6: Structured Output Mode

#### 1.6a. Add structured output model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `StructuredResearchOutput(BaseModel)` with: sources (denormalized), findings, gaps, contradictions, query_type, profile

#### 1.6b. Build structured output after synthesis
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Add `_build_structured_output()` method
  - [x] Sources: flat, denormalized, ready for consumption
  - [x] Findings: with confidence levels and source IDs
  - [x] Gaps: unresolved only
  - [x] Contradictions: with source IDs

#### 1.6c. Include in report response
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `structured` field to report response

#### Item 1.6 Validation

- [x] `_build_structured_output()` with diverse sources
- [x] Sources include full denormalized metadata
- [x] Only unresolved gaps included
- [x] Structured output appears in report response
- [x] Serialization roundtrip

---

## Phase 2: Academic Tools

### Item 2.1: OpenAlex Provider (Tier 1)

> **File**: `src/foundry_mcp/core/research/providers/openalex.py` (NEW)

- [x] Create `OpenAlexProvider(SearchProvider)` class
- [x] Implement `search()` with filter support (year, type, OA status, topic, institution)
- [x] Implement `get_work()` — single work by OpenAlex ID, DOI, or PMID
- [x] Implement `get_citations()` — papers that cite a given work
- [x] Implement `get_references()` — papers cited by a given work
- [x] Implement `get_related()` — related works
- [x] Implement `classify_text()` — topic classification via POST /text
- [x] Implement `search_by_topic()` — works by topic ID
- [x] Implement abstract reconstruction from `abstract_inverted_index`
- [x] Metadata mapping: OpenAlex fields -> `ResearchSource.metadata`
- [x] API key auth via `?api_key=` param or `x-api-key` header
- [x] Handle Walden rewrite field changes (topics not concepts, awards not grants)
- [x] Error classification: 429 -> RATE_LIMIT, 503 -> UNAVAILABLE

#### Item 2.1 Validation

- [x] `search()` with mocked response -> correct metadata mapping
- [x] `get_citations()` with mocked response
- [x] `get_references()` with mocked response
- [x] `get_work()` with DOI input format
- [x] `classify_text()` with mocked response
- [x] Abstract reconstruction from inverted index
- [x] Rate limiting respected
- [x] Graceful handling of empty results

---

### Item 2.2: Crossref Provider (Tier 2)

> **File**: `src/foundry_mcp/core/research/providers/crossref.py` (NEW)

- [x] Create `CrossrefProvider` class (enrichment provider, not `SearchProvider`)
- [x] Implement `get_work(doi)` — full bibliographic metadata by DOI
- [x] Implement `enrich_source()` — fill missing metadata fields only
- [x] Polite pool support via `mailto:` in User-Agent header

#### Item 2.2 Validation

- [x] `get_work()` with mocked response -> correct field extraction
- [x] `enrich_source()` fills missing venue but doesn't overwrite existing
- [x] Graceful handling of DOIs not in Crossref

---

### Item 2.3: Citation Graph & Related Papers Tools

#### 2.3a. Semantic Scholar methods
> **File**: `src/foundry_mcp/core/research/providers/semantic_scholar.py`

- [x] Add `get_citations()` method (GET /paper/{id}/citations)
- [x] Add `get_recommendations()` method (POST /recommendations/v1/papers/)
- [x] Add `get_paper()` method (lookup by S2 ID, DOI, or ArXiv ID)

#### 2.3b. Tool models
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `CitationSearchTool(BaseModel)` with paper_id, max_results
- [x] Add `RelatedPapersTool(BaseModel)` with paper_id, max_results
- [x] Register in `RESEARCHER_TOOL_SCHEMAS`

#### 2.3c. Conditional tool injection
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add `citation_search` tool description when profile `enable_citation_tools == True`
- [x] Add `related_papers` tool description when profile `enable_citation_tools == True`
- [x] Add `_handle_citation_search_tool()` dispatch handler
- [x] Add `_handle_related_papers_tool()` dispatch handler
- [x] Provider fallback: Semantic Scholar first, OpenAlex if available
- [x] Novelty tracking deduplication across tools
- [x] Count against tool call budget
- [x] Log `provider_query` provenance event

#### Item 2.3 Validation

- [x] Each new Semantic Scholar method with mocked HTTP responses
- [x] Tool dispatch routes correctly for `citation_search` and `related_papers`
- [x] Tools NOT available when profile has `enable_citation_tools == False`
- [x] Novelty tracking deduplicates sources across tools
- [x] Integration: topic researcher chains `web_search` -> `citation_search`

---

### Item 2.4: Strategic Research Primitives

#### 2.4a. Add strategic guidance
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add BROADEN strategy guidance (broader terms, related_papers, alternate vocabulary)
- [x] Add DEEPEN strategy guidance (citation_search on seminal papers, extract content)
- [x] Add VALIDATE strategy guidance (corroborate claims, check contradictions)
- [x] Add SATURATE strategy guidance (duplicate detection -> research_complete)
- [x] Only inject when profile has `enable_citation_tools == True`

#### 2.4b. Log strategy usage
- [x] Pattern-match think output for strategy keywords
- [x] Log strategy choice in provenance

#### Item 2.4 Validation

- [x] Academic researcher prompt includes strategy guidance
- [x] General researcher prompt does NOT include strategy guidance
- [x] Strategy keywords in think output captured in provenance

---

### Item 2.5: Adaptive Provider Selection

#### 2.5a. Extract provider hints from brief
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/brief.py`

- [x] Add `_extract_provider_hints()` function
  - [x] Biomedical/clinical/health -> suggest pubmed
  - [x] CS/ML -> suggest semantic_scholar
  - [x] Education/social science -> suggest openalex
  - [x] Profile-specified providers not overridden

#### 2.5b. Apply hints to session state
- [x] If profile doesn't explicitly specify providers, augment with hints
- [x] If profile explicitly specifies, respect that (hints only for auto-config)

#### 2.5c. Use active providers in delegation
> **Files**: `phases/supervision.py`, `phases/supervision_prompts.py`

- [x] Pass `state.active_providers` to topic researcher tasks
- [x] Include available providers in delegation prompt

#### Item 2.5 Validation

- [x] Biomedical brief triggers PubMed hint
- [x] Education brief triggers OpenAlex hint
- [x] Explicit profile providers not overridden by hints
- [x] Hints are additive (don't remove existing providers)
- [x] Unknown/unavailable provider hints silently dropped

---

### Item 2.6: Per-Provider Rate Limiting

#### 2.6a. Add resilience configs
> **File**: `src/foundry_mcp/core/research/providers/resilience/config.py`

- [x] Add `openalex` entry to `PROVIDER_CONFIGS` (50 RPS, burst 10, 3 retries)
- [x] Add `crossref` entry to `PROVIDER_CONFIGS` (10 RPS, burst 5, 3 retries)

#### 2.6b. Add provider config fields
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `openalex_api_key: Optional[str] = None`
- [x] Add `openalex_enabled: bool = True`
- [x] Add `crossref_email: Optional[str] = None`
- [x] Add `crossref_enabled: bool = True`

#### Item 2.6 Validation

- [x] New providers get correct default rate limits
- [x] Config overrides default rate limits
- [x] Unavailable providers excluded from provider chain

---

## Phase 3: Research Intelligence

### Item 3.1: Influence-Aware Source Ranking

#### 3.1a. Add influence-weighted source adequacy
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision_coverage.py`

- [x] Add `_compute_source_influence()` function
  - [x] citation_count >= 100: weight 3x
  - [x] citation_count >= 20: weight 2x
  - [x] citation_count >= 5: weight 1x
  - [x] citation_count < 5 or unknown: weight 0.5x
  - [x] Returns 0.0-1.0 score
  - [x] Returns 1.0 (neutral) for GENERAL/TECHNICAL mode

#### 3.1b. Integrate into coverage weights
- [x] ACADEMIC weights: `{source_adequacy: 0.3, domain_diversity: 0.15, query_completion: 0.2, source_influence: 0.35}`
- [x] Default weights unchanged
- [x] Detection: profile `source_quality_mode == ACADEMIC` OR legacy `research_mode == ACADEMIC`

#### 3.1c. Surface influence in supervisor brief
> **File**: compression phase

- [x] Include citation count in supervisor brief for academic sources
- [x] Omit for web sources

#### 3.1d. Add influence scoring config
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_influence_high_citation_threshold: int = 100`
- [x] Add `deep_research_influence_medium_citation_threshold: int = 20`
- [x] Add `deep_research_influence_low_citation_threshold: int = 5`

#### Item 3.1 Validation

- [x] High-citation sources -> high score
- [x] All unknown citations -> low score
- [x] Mixed citations -> proportional score
- [x] ACADEMIC coverage weights include `source_influence`
- [x] GENERAL coverage weights exclude `source_influence`
- [x] Supervisor brief includes citation counts for academic sources
- [x] No regression in general-mode coverage assessment

---

### Item 3.2: Research Landscape Metadata

#### 3.2a. Add landscape model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `ResearchLandscape(BaseModel)` with: timeline, methodology_breakdown, venue_distribution, field_distribution, top_cited_papers, author_frequency, source_type_breakdown

#### 3.2b. Add to ResearchExtensions
- [x] Add `research_landscape: Optional[ResearchLandscape] = None`
- [x] Add convenience accessor on `DeepResearchState`

#### 3.2c. Build landscape data
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Add `_build_research_landscape()` — pure data transformation from `state.sources`
- [x] Call at end of `_execute_synthesis_async()` after report generation

#### 3.2d. Include in structured output
- [x] Feed landscape into `StructuredResearchOutput`

#### Item 3.2 Validation

- [x] 10 academic sources -> complete metadata
- [x] 0 academic sources -> empty/default values
- [x] Mixed academic + web sources -> correct counts
- [x] Timeline sorted ascending by year
- [x] Top cited papers sorted descending by citation count
- [x] Author frequency counts correctly
- [x] JSON serialization roundtrip
- [x] Landscape appears in structured output

---

### Item 3.3: Explicit Research Gaps Section

#### 3.3a. Inject unresolved gaps
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Inject unresolved gaps into synthesis prompt for `literature_review` query type
- [x] Also inject for ACADEMIC `research_mode` (pre-PLAN-1 fallback)
- [x] Frame constructively: what studies/methodologies would address each gap

#### 3.3b. Include resolved gaps
- [x] Include resolved gaps with resolution notes
- [x] Enable synthesis to note "Recent work by X has begun to address this gap"

#### 3.3c. Add synthesis instructions for gaps section
- [x] Distinguish unexplored areas vs partially addressed topics
- [x] Suggest specific research questions per gap
- [x] Prioritize by impact

#### 3.3d. Include in structured output
- [x] Both resolved and unresolved gaps in structured output with resolution status

#### Item 3.3 Validation

- [x] Unresolved gaps injected for literature_review query type
- [x] Unresolved gaps injected for ACADEMIC research_mode
- [x] Resolved gaps included with resolution notes
- [x] Gaps NOT injected for non-academic general queries
- [x] Empty gaps list -> no gap section in prompt

---

### Item 3.4: Cross-Study Comparison Tables

#### 3.4a. Add comparison table instructions
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] Append comparison table generation instructions for `literature_review` query type
- [x] Table columns: Study, Year, Method, Sample, Key Finding, Effect Size, Limitations
- [x] Only when 3+ empirical studies with sufficient detail

#### 3.4b. Add structured comparison data
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `StudyComparison(BaseModel)` with: study_title, authors, year, methodology, sample_description, key_finding, source_id
- [x] Add `study_comparisons: list[StudyComparison]` to `ResearchLandscape`

#### Item 3.4 Validation

- [x] Synthesis prompt includes table instructions for literature_review
- [x] Synthesis prompt excludes table instructions for other types
- [x] Graceful handling when fewer than 3 empirical studies

---

### Item 3.5: BibTeX & RIS Export

#### 3.5a. BibTeX generator
> **File**: `src/foundry_mcp/core/research/export/bibtex.py` (NEW)

- [x] Create `src/foundry_mcp/core/research/export/__init__.py`
- [x] Implement `sources_to_bibtex(sources)` -> BibTeX string
- [x] Implement `source_to_bibtex_entry(source, citation_key)` -> single entry
- [x] Entry type selection: `@inproceedings` / `@article` / `@misc`
- [x] Stable citation key generation (first author + year + title words)
- [x] Special character escaping: `&`, `%`, `#`, `_`, `{`, `}`

#### 3.5b. RIS generator
> **File**: `src/foundry_mcp/core/research/export/ris.py` (NEW)

- [x] Implement `sources_to_ris(sources)` -> RIS string
- [x] Entry type selection: `TY - JOUR` / `TY - CONF` / `TY - ELEC`
- [x] Compatible with Zotero, Mendeley, EndNote

#### 3.5c. Add export action
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `_handle_deep_research_export()` handler (format: bibtex|ris, academic_only flag)
- [x] Wire up `"deep-research-export"` action in research router

#### 3.5d. Include in structured output
- [x] Generate both BibTeX and RIS after synthesis
- [x] Include in `structured_output.exports`

#### Item 3.5 Validation

- [x] BibTeX with full metadata -> valid entry
- [x] BibTeX with minimal metadata -> valid @misc entry
- [x] BibTeX special character escaping
- [x] Citation key uniqueness and stability
- [x] RIS with full metadata -> valid block
- [x] RIS with minimal metadata -> valid ELEC entry
- [x] Export action handler with completed session
- [x] Export action handler with non-existent session -> error

---

## Phase 4: Deep Analysis

### Item 4.1: Full-Text PDF Analysis

#### 4.1a. Academic paper section detection
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py` (EXTEND)

- [x] Add `detect_sections()` method to `PDFExtractor`
  - [x] Regex patterns for standard section headers (Abstract, Introduction, Methods, Results, Discussion, Conclusion, References)
  - [x] Return `dict[str, tuple[int, int]]` — section name to (start_char, end_char)
  - [x] Graceful fallback: empty dict when no sections detected

#### 4.1b. Prioritized extraction mode
> **File**: `src/foundry_mcp/core/research/pdf_extractor.py` (EXTEND)

- [x] Add `extract_prioritized()` method
  - [x] Accepts `max_chars` (default 50000) and `priority_sections` list
  - [x] Always includes abstract; prioritizes methods/results/discussion
  - [x] Truncates gracefully when full text exceeds `max_chars`

#### 4.1c. Integrate PDF extraction into extract_content tool
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Detect PDF URLs (patterns: `*.pdf`, `arxiv.org/pdf/*`, Content-Type header)
- [x] Route PDF URLs to `PDFExtractor.extract_from_url()` instead of Tavily Extract
- [x] Use `extract_prioritized()` for section-aware content within context limits
- [x] Preserve page boundaries in source metadata for locator support

#### 4.1d. PDF-aware tool (profile-gated)
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/topic_research.py`

- [x] Add `extract_pdf` tool when profile has `enable_pdf_extraction == True`
- [x] Tool accepts URL and max_pages, returns full paper text with section structure
- [x] Only available in `systematic-review` and `bibliometric` profiles

#### 4.1e. Page-aware digest
> **File**: `src/foundry_mcp/core/research/document_digest/digestor.py`

- [x] Use `page:N:char:start-end` locators for evidence snippets from PDF content
- [x] Use `detect_sections()` to prioritize Methods, Results, Discussion
- [x] Handle academic paper structure in digest output

#### 4.1f. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_pdf_max_pages: int = 50`
- [x] Add `deep_research_pdf_priority_sections: list[str] = ["methods", "results", "discussion"]`

#### Item 4.1 Validation

- [x] Section detection works on synthetic academic PDF text
- [x] Prioritized extraction respects max_chars and section ordering
- [x] PDF URLs routed correctly in extract_content tool
- [x] Page-aware locators appear in digest output
- [x] Profile gating works — `extract_pdf` only in systematic-review/bibliometric
- [x] Existing PDFExtractor tests pass unchanged
- [x] Integration: topic researcher extracts PDF and includes in findings
- [x] ~80-120 LOC new tests written

---

### Item 4.2: Citation Network / Connected Papers Graph (User-Triggered)

#### 4.2a. Citation network model
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `CitationNode` model (paper_id, title, authors, year, citation_count, is_discovered, source_id, role)
- [x] Add `CitationEdge` model (citing_paper_id, cited_paper_id)
- [x] Add `CitationNetwork` model (nodes, edges, foundational_papers, research_threads, stats)

#### 4.2b. Add to ResearchExtensions
- [x] Add `citation_network: Optional[CitationNetwork] = None`
- [x] Add convenience accessor on `DeepResearchState`

#### 4.2c. Network builder
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/citation_network.py` (NEW)

- [x] Create `CitationNetworkBuilder` class
- [x] Implement `build_network()` — fetch refs/cites, build graph, classify
  - [x] Fetch references via OpenAlex (primary) / Semantic Scholar (fallback)
  - [x] Fetch citations for each source
  - [x] Build node and edge lists
  - [x] Respect max_references/max_citations caps
  - [x] Concurrency control (`max_concurrent` parameter)
- [x] Implement `_identify_foundational_papers()` — cited by 3+ discovered (or 30%)
- [x] Implement `_identify_research_threads()` — connected components via BFS/union-find (3+ nodes)
- [x] Implement `_classify_roles()` — foundational, discovered, extension, peripheral

#### 4.2d. Dedicated action handler
> **File**: `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

- [x] Add `_handle_deep_research_network()` handler
  - [x] Load completed research state
  - [x] Filter to academic sources with paper IDs
  - [x] Skip if < 3 academic sources
  - [x] Build network and save to `state.extensions.citation_network`
- [x] Wire up `"deep-research-network"` action in research router

#### 4.2e. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_citation_network_max_refs_per_paper: int = 20`
- [x] Add `deep_research_citation_network_max_cites_per_paper: int = 20`

#### Item 4.2 Validation

- [x] Citation network models serialize/deserialize correctly
- [x] Network builder produces correct graph from mocked API responses
- [x] Foundational papers identified correctly (cited by 3+ discovered)
- [x] Research threads detected via connected components
- [x] Action handler returns correct response format
- [x] Graceful skip when < 3 academic sources
- [x] Integration: 5 academic sources -> network with edges
- [x] ~100-140 LOC new tests written

---

### Item 4.3: Methodology Quality Assessment (Experimental)

#### 4.3a. Methodology assessment model
> **File**: `src/foundry_mcp/core/research/models/sources.py`

- [x] Add `StudyDesign` enum (meta_analysis, systematic_review, rct, quasi_experimental, cohort, case_control, cross_sectional, qualitative, case_study, theoretical, opinion, unknown)
- [x] Add `MethodologyAssessment` model
  - [x] Fields: source_id, study_design, sample_size, sample_description, effect_size, statistical_significance, limitations_noted, potential_biases, confidence, content_basis
  - [x] No numeric rigor score — qualitative metadata only

#### 4.3b. Assessment engine
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/methodology_assessment.py` (NEW)

- [x] Create `MethodologyAssessor` class
- [x] Implement `assess_sources()` method
  - [x] Filter to ACADEMIC sources with content > 200 chars
  - [x] Batch LLM call for metadata extraction
  - [x] Force `confidence="low"` for abstract-only content
  - [x] Respect `timeout` parameter

#### 4.3c. LLM extraction prompt
- [x] Design structured JSON extraction prompt
  - [x] Covers: study_design, sample_size, sample_description, effect_size, statistical_significance, limitations, potential_biases, confidence
  - [x] Instructs LLM to use null/empty for missing info (no guessing)

#### 4.3d. Add to ResearchExtensions
> **File**: `src/foundry_mcp/core/research/models/deep_research.py`

- [x] Add `methodology_assessments: list[MethodologyAssessment] = Field(default_factory=list)`
- [x] Add convenience accessor on `DeepResearchState`

#### 4.3e. Feed assessments into synthesis
> **File**: `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`

- [x] When assessments available, inject `## Methodology Context` section
  - [x] Format: study design, sample size, effect size, limitations per source
  - [x] Note content basis (full text vs abstract) with confidence caveat
  - [x] Frame as "context" for qualitative weighting, not ground truth

#### 4.3f. Configuration
> **File**: `src/foundry_mcp/config/research.py`

- [x] Add `deep_research_methodology_assessment_provider: Optional[str] = None`
- [x] Add `deep_research_methodology_assessment_timeout: float = 60.0`
- [x] Add `deep_research_methodology_assessment_min_content_length: int = 200`

#### Item 4.3 Validation

- [x] StudyDesign enum covers common study types
- [x] MethodologyAssessment model serializes/deserializes correctly
- [x] Assessor extracts structured metadata from mocked LLM responses
- [x] Confidence forced to "low" for abstract-only content
- [x] Assessment context correctly injected into synthesis prompt
- [x] Sources with < 200 chars content are skipped
- [x] Integration: end-to-end assessment of 5 academic sources
- [x] ~80-100 LOC new tests written

---

## Final Validation

- [x] All phases implemented and tested
- [x] All new features are profile-gated or backward-compatible
- [x] No new dependencies added (except OpenAlex/Crossref HTTP clients)
- [x] Existing tests pass unchanged
- [x] GENERAL-mode behavior completely unchanged
- [x] `research_mode` backward compat maintained
- [x] ResearchExtensions used for all new state fields
- [x] Provenance logging active across all phases
- [x] Structured output includes all data from PLAN-1/3

---

## Estimated Scope

| Phase | Impl LOC | Test LOC |
|-------|----------|----------|
| 0. Prerequisites | ~200-400 | ~200-300 |
| 1. Foundations | ~1000-1350 | ~450-580 |
| 2. Academic Tools | ~580-850 | ~370-500 |
| 3. Research Intelligence | ~660-940 | ~350-460 |
| 4. Deep Analysis | ~650-900 | ~240-360 |
| **Total** | **~3,090-4,440** | **~1,610-2,200** |
