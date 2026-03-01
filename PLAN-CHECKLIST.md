# PLAN-1 Checklist: Foundations — Profiles, Provenance & Academic Output

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed. Sub-items can be worked in parallel where noted.

---

## Item 1: Research Profiles
> **Parallel with**: Items 2 and 6 (independent foundations)

### 1a. Add ResearchProfile model
> **File**: `models/deep_research.py`

- [x] Define `ResearchProfile(BaseModel)` with all fields (name, providers, source_quality_mode, citation_style, export_formats, synthesis_template, enable_* flags, source_type_hierarchy, disciplinary_scope, time_period, methodology_preferences)
- [x] Define built-in profile constants (GENERAL, ACADEMIC, SYSTEMATIC_REVIEW, BIBLIOMETRIC, TECHNICAL)
- [x] Add `BUILTIN_PROFILES: dict[str, ResearchProfile]` registry

### 1b. Wire into ResearchExtensions
> **File**: `models/deep_research.py`

- [x] Add `research_profile: Optional[ResearchProfile] = None` to `ResearchExtensions`
- [x] Add `@property research_profile` convenience accessor on `DeepResearchState` (returns `self.extensions.research_profile or PROFILE_GENERAL`)
- [x] Verify existing serialized states are backward-compatible

### 1c. Add profile config to research config
> **File**: `config/research.py`

- [x] Add `deep_research_profiles: dict[str, dict]` field (custom profiles)
- [x] Add `deep_research_default_profile: str = "general"` field

### 1d. Profile resolution logic
> **File**: `config/research.py`

- [x] Implement `resolve_profile()` function
  - [x] Explicit `research_profile` name → look up built-in or config
  - [x] Legacy `research_mode` mapping (general→general, academic→academic, technical→technical)
  - [x] Config default fallback
  - [x] Per-request `profile_overrides` applied on top
  - [x] Deprecation warning when both `research_mode` and `research_profile` specified

### 1e. Add profile parameters to deep-research handler
> **File**: `handlers_deep_research.py`

- [x] Add `research_profile: Optional[str]` parameter to `_handle_deep_research()`
- [x] Add `profile_overrides: Optional[dict]` parameter
- [x] Resolution order: `research_profile` > `research_mode` (legacy) > config default

### 1f. Pass profile through action router
> **File**: `research.py`

- [x] Ensure `research_profile` and `profile_overrides` flow from MCP tool arguments through to handler

### Item 1 Testing

- [x] Profile resolution with built-in names (general, academic, systematic-review, bibliometric, technical)
- [x] Profile resolution with legacy `research_mode` mapping
- [x] Per-request overrides applied on top of profile
- [x] Unknown profile name raises validation error
- [x] Backward compat: `research_mode="academic"` ≡ `research_profile="academic"`
- [x] Config-defined custom profiles load correctly

---

## Item 2: Research Provenance Audit Trail
> **Parallel with**: Items 1 and 6 (independent foundations)

### 2a. Add provenance models
> **File**: `models/deep_research.py`

- [ ] Define `ProvenanceEntry(BaseModel)` with fields: timestamp, phase, event_type, summary, details
- [ ] Define `ProvenanceLog(BaseModel)` with fields: session_id, query, profile, profile_config, started_at, completed_at, entries
- [ ] Implement `ProvenanceLog.append()` method with auto-timestamp

### 2b. Wire into ResearchExtensions
> **File**: `models/deep_research.py`

- [ ] Add `provenance: Optional[ProvenanceLog] = None` to `ResearchExtensions`
- [ ] Add `@property provenance` convenience accessor on `DeepResearchState`

### 2c. Initialize provenance at session creation

- [ ] Populate provenance at session start: session_id, query, profile name, profile_config (frozen), started_at
- [ ] Set `completed_at` when session finishes

### 2d. Log brief generation
> **File**: `phases/brief.py`

- [ ] Append `brief_generated` event after brief is generated (brief_text, scope_boundaries, source_preferences)

### 2e. Log supervision events
> **Files**: `phases/supervision.py`, `phases/supervision_coverage.py`

- [ ] `decomposition` — after delegation response in `_run_think_delegate_step()` with directives
- [ ] `provider_query` — after each provider search in topic researcher (provider, query, result_count, source_ids)
- [ ] `source_discovered` — when sources added to state in `_execute_and_merge_directives()` (source_id, title, provider, source_type, url)
- [ ] `source_deduplicated` — when deduplication occurs (source_id, duplicate_of, reason)
- [ ] `coverage_assessment` — after `assess_coverage_heuristic()` (scores, decision)
- [ ] `gap_identified` — when gaps added (gap_id, description, priority, suggested_queries)
- [ ] `gap_resolved` — when gaps resolved (gap_id, resolution_notes)
- [ ] `iteration_complete` — at end of each supervision round in `_post_round_bookkeeping()` (iteration, round, total_sources, total_findings)

### 2f. Log synthesis events
> **File**: `phases/synthesis.py`

- [ ] `synthesis_query_type` — after `_classify_query_type()` (query_type, detection_reason)
- [ ] `synthesis_completed` — after synthesis completes (report_length, source_count, citation_count)

### 2g. Persist provenance separately
> **File**: `memory.py`

- [ ] Save provenance as `deepres-{id}.provenance.json` alongside state file
- [ ] Load provenance when loading state
- [ ] Keep main state file compact (provenance excluded from state serialization)

### 2h. Expose provenance in API
> **File**: `handlers_deep_research.py`

- [ ] Add `include_provenance` to `deep-research-report` response
- [ ] Add dedicated `deep-research-provenance` action handler
- [ ] Wire new action into research router

### Item 2 Testing

- [ ] `ProvenanceLog.append()` creates timestamped entries
- [ ] Provenance populated after brief phase
- [ ] Provenance populated after supervision round
- [ ] Provenance persisted and loadable from disk
- [ ] Provenance included in report response
- [ ] Serialization roundtrip (model_dump → model_validate)

---

## Item 3: `literature_review` Query Type
> **Depends on**: Item 1 (profiles) for `synthesis_template` and academic bias

### 3a. Add detection pattern
> **File**: `phases/synthesis.py`

- [ ] Add `_LITERATURE_REVIEW_PATTERNS` regex after existing patterns
  - [ ] Matches: "literature review", "systematic review", "meta-analysis", "survey of", "state of the art", "body of research/literature/work", "prior work/research/studies", "existing research/literature/studies", "review of the literature/research", "what does the research say/show/suggest", etc.

### 3b. Add classification check
> **File**: `phases/synthesis.py`

- [ ] Insert `_LITERATURE_REVIEW_PATTERNS` check in `_classify_query_type()` **before** existing comparison/enumeration/howto checks
- [ ] Honor `profile.synthesis_template == "literature_review"` (direct override)
- [ ] Bias toward `literature_review` when `source_quality_mode == ACADEMIC` and query is ambiguous

### 3c. Add structure guidance
> **File**: `phases/synthesis.py`

- [ ] Add `"literature_review"` entry to `_STRUCTURE_GUIDANCE` dict with sections:
  - [ ] Executive Summary
  - [ ] Introduction & Scope
  - [ ] Theoretical Foundations
  - [ ] Thematic Analysis (with sub-themes)
  - [ ] Methodological Approaches
  - [ ] Key Debates & Contradictions
  - [ ] Research Gaps & Future Directions
  - [ ] Conclusions
  - [ ] References

### 3d. Add academic synthesis instructions
> **File**: `phases/synthesis.py`

- [ ] Inject additional instructions into `_build_synthesis_system_prompt()` when `query_type == "literature_review"`:
  - [ ] Thematic organization over listing
  - [ ] Author/year/method notes per study
  - [ ] Seminal works identification
  - [ ] Methodological trend tracking
  - [ ] Balanced conflict presentation
  - [ ] Citation style-formatted References section

### Item 3 Testing

- [ ] `"literature review on X"` → `"literature_review"`
- [ ] `"what does the research say about X"` → `"literature_review"`
- [ ] `"survey of prior work on X"` → `"literature_review"`
- [ ] Profile `synthesis_template="literature_review"` forces the type
- [ ] Generic query → `"explanation"` (no regression)
- [ ] Comparison query → `"comparison"` (no regression)
- [ ] Structure guidance correctly injected for literature_review type

---

## Item 4: APA Citation Formatting
> **Depends on**: Item 1 (profiles) for `citation_style`

### 4a. Add APA formatting function
> **File**: `phases/_citation_postprocess.py`

- [ ] Implement `format_source_apa(source: ResearchSource) -> str`
  - [ ] Full academic: `Authors (Year). Title. *Venue*. DOI_URL`
  - [ ] Handle "et al." for >5 authors
  - [ ] Web source: `Author/Organization (Year). Title. *Site Name*. URL`
  - [ ] Minimal fallback: `Title. URL`
  - [ ] Missing year → "n.d." per APA convention

### 4b. Add format_style parameter
> **File**: `phases/_citation_postprocess.py`

- [ ] Add `format_style` parameter to `build_sources_section()`
- [ ] When `"apa"`: use `format_source_apa()`, header "## References"
- [ ] When `"default"`: preserve existing `[{cn}] [{title}]({url})` format exactly

### 4c. Connect to profile
> **File**: `phases/_citation_postprocess.py`

- [ ] Read `state.research_profile.citation_style` in `postprocess_citations()`
- [ ] Force `format_style="apa"` when `query_type == "literature_review"` regardless of profile

### Item 4 Testing

- [ ] `format_source_apa()` with full academic metadata (all fields present)
- [ ] `format_source_apa()` with partial metadata (missing venue, DOI)
- [ ] `format_source_apa()` with web source (no academic metadata)
- [ ] `format_source_apa()` with >5 authors ("et al.")
- [ ] `format_source_apa()` with missing year ("n.d.")
- [ ] `build_sources_section(format_style="apa")` → "## References" header
- [ ] `build_sources_section(format_style="default")` → existing format preserved
- [ ] Integration: synthesis in academic profile produces APA references

---

## Item 5: Academic Brief Enrichment
> **Depends on**: Item 1 (profiles) for profile fields

### 5a. Profile-aware brief system prompt
> **File**: `phases/brief.py`

- [ ] Modify `_build_brief_system_prompt()` to accept research profile
- [ ] When `source_quality_mode == ACADEMIC`, append instructions for:
  - [ ] Disciplinary scope (primary + interdisciplinary)
  - [ ] Time period (foundational + recent literature)
  - [ ] Methodology preferences (quantitative, qualitative, mixed, meta-analysis, theoretical)
  - [ ] Education level / population (if applicable)
  - [ ] Source type hierarchy (peer-reviewed > meta-analyses > books > preprints > reports; deprioritize blogs, news, Wikipedia)
- [ ] Inject profile-specified constraints (source_type_hierarchy, disciplinary_scope, time_period, methodology_preferences) as pre-filled values

### 5b. Profile-aware decomposition
> **File**: `phases/supervision_prompts.py`

- [ ] Append academic decomposition guidelines to `build_first_round_delegation_system_prompt()` when academic:
  - [ ] Foundational/seminal works directive (sorted by citation count)
  - [ ] Recent empirical studies directive (last 3-5 years)
  - [ ] Per-discipline directives for cross-disciplinary topics
  - [ ] Map directives to literature review sections
  - [ ] Evidence needed: peer-reviewed articles, sample sizes, effect sizes, theoretical frameworks

### Item 5 Testing

- [ ] Brief prompt includes academic dimensions when profile is academic
- [ ] Brief prompt unchanged when profile is general
- [ ] Profile-specified constraints injected into brief prompt
- [ ] Supervision prompt includes academic guidelines when profile is academic
- [ ] Supervision prompt unchanged when profile is general

---

## Item 6: Structured Output Mode
> **Parallel with**: Items 1 and 2 (independent foundations)

### 6a. Add StructuredResearchOutput model
> **File**: `models/deep_research.py`

- [ ] Define `StructuredResearchOutput(BaseModel)` with fields:
  - [ ] `sources: list[dict]` — full metadata, reference-manager ready
  - [ ] `findings: list[dict]` — with confidence + source IDs
  - [ ] `gaps: list[dict]` — unresolved only
  - [ ] `contradictions: list[dict]` — cross-source conflicts
  - [ ] `query_type: str = "explanation"`
  - [ ] `profile: str = "general"`

### 6b. Build structured output after synthesis
> **File**: `phases/synthesis.py`

- [ ] Add `_build_structured_output()` method
  - [ ] Sources: `{id, title, url, source_type, authors, year, venue, doi, citation_count, ...}` — flat, denormalized
  - [ ] Findings: `{id, content, confidence, source_ids, category}`
  - [ ] Gaps: `{id, description, priority, suggested_queries}` — unresolved only
  - [ ] Contradictions: `{id, description, source_ids}`
- [ ] Call from synthesis finalization, store on `state.extensions.structured_output`

### 6c. Include in report response
> **File**: `handlers_deep_research.py`

- [ ] Add `structured` field to `deep-research-report` response
- [ ] Always present: `state.extensions.structured_output.model_dump() if structured else None`

### Item 6 Testing

- [ ] `_build_structured_output()` with diverse sources
- [ ] Sources include full denormalized metadata
- [ ] Only unresolved gaps included
- [ ] Structured output appears in report response
- [ ] Serialization roundtrip

---

## Final Validation

- [ ] All new PLAN-1 tests pass
- [ ] All existing tests pass with zero regressions
- [ ] No behavioral changes to GENERAL-mode queries
- [ ] `research_mode` parameter still works (backward compat)
- [ ] Provenance log produced for every deep research session
- [ ] Structured output included in every report response
- [ ] Literature review queries produce thematic structure
- [ ] Academic profile produces APA references
