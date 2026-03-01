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

- [x] Define `ProvenanceEntry(BaseModel)` with fields: timestamp, phase, event_type, summary, details
- [x] Define `ProvenanceLog(BaseModel)` with fields: session_id, query, profile, profile_config, started_at, completed_at, entries
- [x] Implement `ProvenanceLog.append()` method with auto-timestamp

### 2b. Wire into ResearchExtensions
> **File**: `models/deep_research.py`

- [x] Add `provenance: Optional[ProvenanceLog] = None` to `ResearchExtensions`
- [x] Add `@property provenance` convenience accessor on `DeepResearchState`

### 2c. Initialize provenance at session creation

- [x] Populate provenance at session start: session_id, query, profile name, profile_config (frozen), started_at
- [x] Set `completed_at` when session finishes (via mark_completed, mark_failed, mark_cancelled)

### 2d. Log brief generation
> **File**: `phases/brief.py`

- [x] Append `brief_generated` event after brief is generated (brief_length, provider_id, model_used, tokens_used)

### 2e. Log supervision events
> **Files**: `phases/supervision.py`, `phases/supervision_coverage.py`

- [x] `decomposition` — after delegation response in `_run_think_delegate_step()` with directives
- [ ] `provider_query` — deferred (requires threading provenance through topic researcher stack)
- [x] `source_discovered` — after directive execution with per-directive source counts and IDs
- [ ] `source_deduplicated` — deferred (requires changes to topic researcher dedup internals)
- [x] `coverage_assessment` — after `assess_coverage_heuristic()` (confidence, dimensions, decision)
- [ ] `gap_identified` — deferred (gaps managed by analysis/refinement phases, not supervision)
- [ ] `gap_resolved` — deferred (gaps managed by analysis/refinement phases, not supervision)
- [x] `iteration_complete` — at end of each supervision round in `_post_round_bookkeeping()` (round, new_sources, total_sources, total_findings)

### 2f. Log synthesis events
> **File**: `phases/synthesis.py`

- [x] `synthesis_query_type` — after `_classify_query_type()` (query_type)
- [x] `synthesis_completed` — after synthesis completes (report_length, source_count, finding_count, provider_id, model_used)

### 2g. Persist provenance separately
> **File**: `memory.py`

- [x] Save provenance as `deepres-{id}.provenance.json` alongside state file
- [x] Load provenance when loading state
- [x] Keep main state file compact (provenance excluded from state serialization)

### 2h. Expose provenance in API
> **File**: `handlers_deep_research.py`

- [x] Add `provenance_summary` to `deep-research-report` response
- [x] Add dedicated `deep-research-provenance` action handler
- [x] Wire new action into research router

### Item 2 Testing

- [x] `ProvenanceLog.append()` creates timestamped entries
- [x] Provenance populated after brief phase
- [x] Provenance populated after supervision round
- [x] Provenance persisted and loadable from disk
- [x] Provenance included in report response
- [x] Serialization roundtrip (model_dump → model_validate)

---

## Item 3: `literature_review` Query Type
> **Depends on**: Item 1 (profiles) for `synthesis_template` and academic bias

### 3a. Add detection pattern
> **File**: `phases/synthesis.py`

- [x] Add `_LITERATURE_REVIEW_PATTERNS` regex after existing patterns
  - [x] Matches: "literature review", "systematic review", "meta-analysis", "survey of", "state of the art", "body of research/literature/work", "prior work/research/studies", "existing research/literature/studies", "review of the literature/research", "what does the research say/show/suggest", etc.

### 3b. Add classification check
> **File**: `phases/synthesis.py`

- [x] Insert `_LITERATURE_REVIEW_PATTERNS` check in `_classify_query_type()` **before** existing comparison/enumeration/howto checks
- [x] Honor `profile.synthesis_template == "literature_review"` (direct override)
- [x] Bias toward `literature_review` when `source_quality_mode == ACADEMIC` and query is ambiguous

### 3c. Add structure guidance
> **File**: `phases/synthesis.py`

- [x] Add `"literature_review"` entry to `_STRUCTURE_GUIDANCE` dict with sections:
  - [x] Executive Summary
  - [x] Introduction & Scope
  - [x] Theoretical Foundations
  - [x] Thematic Analysis (with sub-themes)
  - [x] Methodological Approaches
  - [x] Key Debates & Contradictions
  - [x] Research Gaps & Future Directions
  - [x] Conclusions
  - [x] References

### 3d. Add academic synthesis instructions
> **File**: `phases/synthesis.py`

- [x] Inject additional instructions into `_build_synthesis_system_prompt()` when `query_type == "literature_review"`:
  - [x] Thematic organization over listing
  - [x] Author/year/method notes per study
  - [x] Seminal works identification
  - [x] Methodological trend tracking
  - [x] Balanced conflict presentation
  - [x] Citation style-formatted References section

### Item 3 Testing

- [x] `"literature review on X"` → `"literature_review"`
- [x] `"what does the research say about X"` → `"literature_review"`
- [x] `"survey of prior work on X"` → `"literature_review"`
- [x] Profile `synthesis_template="literature_review"` forces the type
- [x] Generic query → `"explanation"` (no regression)
- [x] Comparison query → `"comparison"` (no regression)
- [x] Structure guidance correctly injected for literature_review type

---

## Item 4: APA Citation Formatting
> **Depends on**: Item 1 (profiles) for `citation_style`

### 4a. Add APA formatting function
> **File**: `phases/_citation_postprocess.py`

- [x] Implement `format_source_apa(source: ResearchSource) -> str`
  - [x] Full academic: `Authors (Year). Title. *Venue*. DOI_URL`
  - [x] Handle "et al." for >5 authors
  - [x] Web source: `Author/Organization (Year). Title. *Site Name*. URL`
  - [x] Minimal fallback: `Title. URL`
  - [x] Missing year → "n.d." per APA convention

### 4b. Add format_style parameter
> **File**: `phases/_citation_postprocess.py`

- [x] Add `format_style` parameter to `build_sources_section()`
- [x] When `"apa"`: use `format_source_apa()`, header "## References"
- [x] When `"default"`: preserve existing `[{cn}] [{title}]({url})` format exactly

### 4c. Connect to profile
> **File**: `phases/_citation_postprocess.py`

- [x] Read `state.research_profile.citation_style` in `postprocess_citations()`
- [x] Force `format_style="apa"` when `query_type == "literature_review"` regardless of profile

### Item 4 Testing

- [x] `format_source_apa()` with full academic metadata (all fields present)
- [x] `format_source_apa()` with partial metadata (missing venue, DOI)
- [x] `format_source_apa()` with web source (no academic metadata)
- [x] `format_source_apa()` with >5 authors ("et al.")
- [x] `format_source_apa()` with missing year ("n.d.")
- [x] `build_sources_section(format_style="apa")` → "## References" header
- [x] `build_sources_section(format_style="default")` → existing format preserved
- [x] Integration: synthesis in academic profile produces APA references

---

## Item 5: Academic Brief Enrichment
> **Depends on**: Item 1 (profiles) for profile fields

### 5a. Profile-aware brief system prompt
> **File**: `phases/brief.py`

- [x] Modify `_build_brief_system_prompt()` to accept research profile
- [x] When `source_quality_mode == ACADEMIC`, append instructions for:
  - [x] Disciplinary scope (primary + interdisciplinary)
  - [x] Time period (foundational + recent literature)
  - [x] Methodology preferences (quantitative, qualitative, mixed, meta-analysis, theoretical)
  - [x] Education level / population (if applicable)
  - [x] Source type hierarchy (peer-reviewed > meta-analyses > books > preprints > reports; deprioritize blogs, news, Wikipedia)
- [x] Inject profile-specified constraints (source_type_hierarchy, disciplinary_scope, time_period, methodology_preferences) as pre-filled values

### 5b. Profile-aware decomposition
> **File**: `phases/supervision_prompts.py`

- [x] Append academic decomposition guidelines to `build_first_round_delegation_system_prompt()` when academic:
  - [x] Foundational/seminal works directive (sorted by citation count)
  - [x] Recent empirical studies directive (last 3-5 years)
  - [x] Per-discipline directives for cross-disciplinary topics
  - [x] Map directives to literature review sections
  - [x] Evidence needed: peer-reviewed articles, sample sizes, effect sizes, theoretical frameworks

### Item 5 Testing

- [x] Brief prompt includes academic dimensions when profile is academic
- [x] Brief prompt unchanged when profile is general
- [x] Profile-specified constraints injected into brief prompt
- [x] Supervision prompt includes academic guidelines when profile is academic
- [x] Supervision prompt unchanged when profile is general

---

## Item 6: Structured Output Mode
> **Parallel with**: Items 1 and 2 (independent foundations)

### 6a. Add StructuredResearchOutput model
> **File**: `models/deep_research.py`

- [x] Define `StructuredResearchOutput(BaseModel)` with fields:
  - [x] `sources: list[dict]` — full metadata, reference-manager ready
  - [x] `findings: list[dict]` — with confidence + source IDs
  - [x] `gaps: list[dict]` — unresolved only
  - [x] `contradictions: list[dict]` — cross-source conflicts
  - [x] `query_type: str = "explanation"`
  - [x] `profile: str = "general"`

### 6b. Build structured output after synthesis
> **File**: `phases/synthesis.py`

- [x] Add `_build_structured_output()` method
  - [x] Sources: `{id, title, url, source_type, authors, year, venue, doi, citation_count, ...}` — flat, denormalized
  - [x] Findings: `{id, content, confidence, source_ids, category}`
  - [x] Gaps: `{id, description, priority, suggested_queries}` — unresolved only
  - [x] Contradictions: `{id, description, source_ids}`
- [x] Call from synthesis finalization, store on `state.extensions.structured_output`

### 6c. Include in report response
> **File**: `handlers_deep_research.py`

- [x] Add `structured` field to `deep-research-report` response
- [x] Always present: `state.extensions.structured_output.model_dump() if structured else None`

### Item 6 Testing

- [x] `_build_structured_output()` with diverse sources
- [x] Sources include full denormalized metadata
- [x] Only unresolved gaps included
- [x] Structured output appears in report response
- [x] Serialization roundtrip

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
