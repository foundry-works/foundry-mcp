# PLAN-3 Checklist: Research Intelligence

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed.

---

## Item 1: Influence-Aware Source Ranking

- [x] Add influence scoring config to `config/research.py`
  - [x] `deep_research_influence_high_citation_threshold` (default: 100)
  - [x] `deep_research_influence_medium_citation_threshold` (default: 20)
  - [x] `deep_research_influence_low_citation_threshold` (default: 5)
  - [x] `deep_research_academic_coverage_weights` (default: source_adequacy 0.3, domain_diversity 0.15, query_completion_rate 0.2, source_influence 0.35)
- [x] Add `_is_academic_mode()` helper to `supervision_coverage.py`
- [x] Add `compute_source_influence()` to `supervision_coverage.py`
- [x] Integrate `source_influence` dimension into `assess_coverage_heuristic()`
- [x] Use academic weights when `research_mode == ACADEMIC` and no explicit weights set
- [x] Update compression supervisor brief to request citation counts
- [x] Tests: general mode returns neutral, high/mixed/unknown citations scored correctly
- [x] Tests: academic weights include source_influence, general weights neutral

## Item 2: Research Landscape Metadata

- [x] Add `ResearchLandscape` model to `models/deep_research.py`
  - [x] timeline, methodology_breakdown, venue_distribution, field_distribution
  - [x] top_cited_papers, author_frequency, source_type_breakdown
  - [x] study_comparisons (shared with item 4)
- [x] Update `ResearchExtensions.research_landscape` to concrete `ResearchLandscape` type
- [x] Add `DeepResearchState.research_landscape` convenience accessor
- [x] Add `_build_research_landscape()` method to `SynthesisPhaseMixin`
- [x] Call landscape builder from `_finalize_synthesis_report()`
- [x] Tests: empty landscape, populated landscape, extensions field, state accessor

## Item 3: Explicit Research Gaps Section

- [x] Enhance `_append_contradictions_and_gaps()` in `synthesis.py`
- [x] Inject structured unresolved gaps with priority for academic mode
- [x] Include resolved gaps with resolution notes for academic mode
- [x] Add constructive framing instructions for "Research Gaps & Future Directions" section
- [x] Preserve original compact format for non-academic queries
- [x] Tests: unresolved gaps available, resolution notes preserved

## Item 4: Cross-Study Comparison Tables

- [x] Add `StudyComparison` model to `models/deep_research.py`
- [x] Add `study_comparisons` field to `ResearchLandscape`
- [x] Add comparison table instructions to synthesis system prompt for academic mode
- [x] Add "Research Gaps & Future Directions" instructions for academic mode

## Item 5: BibTeX & RIS Export

- [x] Create `core/research/export/__init__.py`
- [x] Create `core/research/export/bibtex.py`
  - [x] `sources_to_bibtex()` with stable unique citation keys
  - [x] `source_to_bibtex_entry()` with entry type detection
  - [x] BibTeX special character escaping
- [x] Create `core/research/export/ris.py`
  - [x] `sources_to_ris()` with TY-ER blocks
  - [x] `source_to_ris_entry()` with entry type mapping
- [x] Add `_handle_deep_research_export()` handler
- [x] Wire `deep-research-export` action into router
- [x] Tests: BibTeX full/minimal/conference/escaping/uniqueness/empty
- [x] Tests: RIS full/minimal/conference/empty/multi-author

---

## Final Validation

- [x] All 27 new PLAN-3 tests pass
- [x] All 192 existing tests pass with zero regressions
- [x] No behavioral changes to general-mode queries
- [x] Academic features activate only when `research_mode == ACADEMIC`
- [x] Export action wired into research router
