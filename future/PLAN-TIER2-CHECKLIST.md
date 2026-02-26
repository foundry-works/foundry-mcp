# PLAN-TIER2 Implementation Checklist

> Track progress for each Tier 2 improvement. Check items as completed.
>
> **Prerequisites**: Tier 1 items 2 (APA citations) and 3 (Semantic Scholar citation graph) recommended before starting.

---

## 6. Influence-Aware Source Ranking

### Influence Scoring
- [ ] Add `_compute_source_influence(state) -> float` to supervision.py
- [ ] Extract `citation_count` from `source.metadata` for each source
- [ ] Extract `influential_citation_count` from `source.metadata` (bonus)
- [ ] Apply tiered weighting: >=100 (3x), >=20 (2x), >=5 (1x), <5 (0.5x)
- [ ] Normalize to 0.0-1.0 scale
- [ ] Handle sources with missing citation metadata (treat as unknown, weight 0.5x)
- [ ] Only compute for ACADEMIC mode (return 1.0 for GENERAL/TECHNICAL)

### Coverage Integration
- [ ] Add `source_influence` dimension to `_assess_coverage_heuristic()` return dict
- [ ] Modify default coverage weights for ACADEMIC mode: source_adequacy=0.3, domain_diversity=0.15, query_completion_rate=0.2, source_influence=0.35
- [ ] Ensure GENERAL mode weights are unchanged (no regression)
- [ ] TECHNICAL mode: optional influence dimension with lower weight (0.15)

### Supervisor Brief Enhancement
- [ ] Modify compression system prompt to include citation count per source
- [ ] Format: `[N] Author (Year) [cited X times]: Finding...`
- [ ] Only include citation count when available (omit for web sources)
- [ ] Verify supervisor receives enriched brief in subsequent rounds

### Configuration
- [ ] Add `deep_research_influence_scoring_enabled` (default: True) to config
- [ ] Add `deep_research_influence_high_citation_threshold` (default: 100)
- [ ] Add `deep_research_influence_medium_citation_threshold` (default: 20)
- [ ] Add `deep_research_influence_low_citation_threshold` (default: 5)
- [ ] Parse new config fields in `from_toml_dict()`
- [ ] Validate thresholds are positive integers in order high > medium > low

### Testing
- [ ] Unit test: `_compute_source_influence()` with 5 high-citation sources -> high score
- [ ] Unit test: `_compute_source_influence()` with all unknown citations -> low score
- [ ] Unit test: `_compute_source_influence()` with mixed citations -> proportional score
- [ ] Unit test: ACADEMIC coverage weights include `source_influence`
- [ ] Unit test: GENERAL coverage weights exclude `source_influence`
- [ ] Unit test: influence scoring disabled via config -> dimension returns 1.0
- [ ] Verify no regression in coverage assessment for GENERAL mode queries

---

## 7. Research Landscape Metadata Output

### Model Definition
- [ ] Add `ResearchLandscape` Pydantic model to `models/deep_research.py`
- [ ] Fields: `timeline`, `methodology_breakdown`, `venue_distribution`, `field_distribution`
- [ ] Fields: `top_cited_papers`, `author_frequency`, `source_type_breakdown`
- [ ] Add `research_landscape: Optional[ResearchLandscape]` to `DeepResearchState`

### Landscape Builder
- [ ] Add `_build_research_landscape(state) -> ResearchLandscape` to SynthesisPhaseMixin
- [ ] Timeline: group sources by year, count per year, identify key papers per year
- [ ] Methodology breakdown: extract from source metadata or findings categories
- [ ] Venue distribution: aggregate `metadata.venue` across academic sources
- [ ] Field distribution: aggregate `metadata.fields_of_study` across sources
- [ ] Top cited papers: sort by `metadata.citation_count`, take top 10
- [ ] Author frequency: count appearances of each author across all sources
- [ ] Source type breakdown: count by `source.source_type` (academic vs web)
- [ ] Handle missing metadata gracefully (skip sources without relevant fields)

### Integration
- [ ] Call `_build_research_landscape()` at end of `_execute_synthesis_async()`
- [ ] Store result in `state.research_landscape`
- [ ] Persist landscape with state (verify JSON serialization)
- [ ] Include landscape in deep-research-report handler response
- [ ] Include landscape in synthesis `WorkflowResult.metadata`

### Testing
- [ ] Unit test: landscape with 10 academic sources -> complete metadata
- [ ] Unit test: landscape with 0 academic sources -> empty/default values
- [ ] Unit test: landscape with mixed academic + web sources -> correct counts
- [ ] Unit test: timeline sorted ascending by year
- [ ] Unit test: top_cited_papers sorted descending by citation count
- [ ] Unit test: author_frequency counts correctly with "et al." handling
- [ ] Unit test: JSON serialization roundtrip
- [ ] Verify landscape appears in deep-research-report response

---

## 8. Explicit Research Gaps Section in Report

### Gap Injection
- [ ] Collect unresolved gaps: `[g for g in state.gaps if not g.resolved]`
- [ ] Collect resolved gaps: `[g for g in state.gaps if g.resolved]`
- [ ] Add unresolved gaps section to synthesis user prompt (for literature_review type only)
- [ ] Format: numbered list with description and priority
- [ ] Add resolved gaps section with resolution notes
- [ ] Include instruction: "Incorporate into Research Gaps & Future Directions section"
- [ ] Include instruction: frame constructively with specific research questions

### Structure Guidance Enhancement
- [ ] Verify `literature_review` structure guidance includes "Research Gaps & Future Directions"
- [ ] Add explicit instructions for the gaps section in synthesis system prompt:
  - [ ] Base on identified gaps from input
  - [ ] Distinguish unexplored vs. partially addressed
  - [ ] Suggest specific research questions per gap
  - [ ] Prioritize by potential impact

### Conditional Application
- [ ] Only inject gaps for `query_type == "literature_review"`
- [ ] Also inject when `research_mode == ACADEMIC` regardless of query_type (optional, decide)
- [ ] Skip gap injection when `state.gaps` is empty

### Testing
- [ ] Unit test: unresolved gaps appear in synthesis prompt for literature_review
- [ ] Unit test: resolved gaps appear with resolution notes
- [ ] Unit test: gaps NOT injected for query_type "explanation"
- [ ] Unit test: empty gaps list -> no gap section in prompt
- [ ] Unit test: mix of resolved and unresolved gaps -> both sections present
- [ ] Verify synthesis output contains "Research Gaps" section with substantive content

---

## 9. Cross-Study Comparison Tables

### Synthesis Prompt Enhancement
- [ ] Add comparison table instructions to synthesis system prompt for literature_review
- [ ] Specify markdown table format: Study | Year | Method | Sample | Key Finding | Effect Size | Limitations
- [ ] Condition: "only when 3+ empirical studies with sufficient methodological detail"
- [ ] Instruct: "Use 'Not reported' for missing values"
- [ ] Place table in "Methodological Approaches" section

### Structured Data Model (Optional, depends on item 7)
- [ ] Add `StudyComparison` Pydantic model to `models/deep_research.py`
- [ ] Fields: study_title, authors, year, methodology, sample_description, key_finding, source_id
- [ ] Add `study_comparisons: list[StudyComparison]` to `ResearchLandscape`
- [ ] Populate from synthesis output parsing (post-processing step)
- [ ] Handle case where LLM doesn't generate a table (empty list)

### Conditional Application
- [ ] Only include table instructions for `query_type == "literature_review"`
- [ ] Do NOT include for comparison, enumeration, howto, or explanation types
- [ ] Graceful degradation: if LLM omits table, report is still valid

### Testing
- [ ] Unit test: synthesis prompt includes table instructions for literature_review
- [ ] Unit test: synthesis prompt excludes table instructions for other types
- [ ] Verify generated report contains markdown table when sufficient empirical data
- [ ] Verify report is valid when fewer than 3 empirical studies found (no table required)
- [ ] Test `StudyComparison` model serialization

---

## Cross-Cutting

- [ ] All Tier 2 changes pass `pytest tests/core/research/` with no regressions
- [ ] All Tier 2 changes pass `pytest tests/integration/test_deep_research_*.py`
- [ ] GENERAL mode behavior is completely unchanged by all Tier 2 changes
- [ ] TECHNICAL mode behavior is minimally affected (only influence scoring, if enabled)
- [ ] New config fields have sensible defaults that match current behavior
