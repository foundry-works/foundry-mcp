# PLAN-3: Research Intelligence — Ranking, Landscape & Export

> **Branch**: `deep-academic`
>
> **Goal**: Add capabilities that produce meaningfully richer academic outputs — influence-aware source ranking, structured landscape metadata, explicit research gaps sections, cross-study comparison tables, and reference export (BibTeX/RIS).
>
> **Status**: **Implemented**
>
> **Source**: [future/PLAN-3-RESEARCH-INTELLIGENCE.md](future/PLAN-3-RESEARCH-INTELLIGENCE.md)

---

## Design Principles

1. **Use data already collected.** Most features transform metadata that Semantic Scholar already stores — citation counts, venues, years, fields of study, authors. The cost is computation, not API calls.
2. **Work with whatever metadata exists.** Every item gracefully handles missing metadata fields. BibTeX export works with just title+URL; influence scoring falls back to equal weighting when citation counts are absent.
3. **Academic features activate conditionally.** Features activate based on `research_mode == ACADEMIC` (existing enum). General-mode behavior is completely unchanged.
4. **Structured data complements prose.** Landscape metadata, study comparisons, and export formats are stored on `ResearchExtensions.research_landscape` for downstream consumption.

---

## Implementation Summary

### 1. Influence-Aware Source Ranking

**Files modified:**
- `config/research.py` — Added influence scoring config thresholds and academic coverage weights
- `phases/supervision_coverage.py` — Added `compute_source_influence()`, `_is_academic_mode()`, integrated `source_influence` dimension into `assess_coverage_heuristic()`
- `phases/compression.py` — Updated supervisor brief to include citation counts

### 2. Research Landscape Metadata

**Files modified:**
- `models/deep_research.py` — Added `ResearchLandscape` and `StudyComparison` models, updated `ResearchExtensions.research_landscape` to use concrete type, added `DeepResearchState.research_landscape` convenience accessor
- `phases/synthesis.py` — Added `_build_research_landscape()` method, called from `_finalize_synthesis_report()`

### 3. Explicit Research Gaps Section

**Files modified:**
- `phases/synthesis.py` — Enhanced `_append_contradictions_and_gaps()` to inject structured gap information (unresolved + resolved with resolution notes) for academic mode queries

### 4. Cross-Study Comparison Tables

**Files modified:**
- `models/deep_research.py` — `StudyComparison` model (shared with item 2)
- `phases/synthesis.py` — Added academic-specific synthesis system prompt instructions for comparison tables and research gaps sections

### 5. BibTeX & RIS Export

**Files created:**
- `core/research/export/__init__.py` — Package init
- `core/research/export/bibtex.py` — BibTeX generator with citation key generation, entry type detection, special character escaping
- `core/research/export/ris.py` — RIS generator with entry type mapping

**Files modified:**
- `tools/unified/research_handlers/handlers_deep_research.py` — Added `_handle_deep_research_export()`
- `tools/unified/research_handlers/__init__.py` — Wired `deep-research-export` action
- `tools/unified/research_handlers/_helpers.py` — Added action summary
- `tools/unified/research.py` — Re-exported new handler

---

## File Impact Summary

| File | Change Type | Items |
|------|-------------|-------|
| `phases/supervision_coverage.py` | Modify | 1 (influence scoring + academic mode detection) |
| `phases/synthesis.py` | Modify | 2 (landscape builder), 3 (gap injection), 4 (comparison tables) |
| `phases/compression.py` | Modify | 1 (citation count in supervisor brief) |
| `models/deep_research.py` | Modify | 2 (ResearchLandscape, StudyComparison, accessor) |
| `config/research.py` | Modify | 1 (influence thresholds, academic weights) |
| `export/__init__.py` | **New** | 5 |
| `export/bibtex.py` | **New** | 5 |
| `export/ris.py` | **New** | 5 |
| `handlers_deep_research.py` | Modify | 5 (export action) |
| `research_handlers/__init__.py` | Modify | 5 (wire up export action) |
| `research_handlers/_helpers.py` | Modify | 5 (action summary) |
| `research.py` (shim) | Modify | 5 (re-export) |

## Testing

- 27 new tests in `tests/unit/test_plan3_research_intelligence.py`
- 192 existing tests pass with zero regressions

## What Comes Next

| Next Plan | Theme | Hard Dep |
|-----------|-------|----------|
| [PLAN-1](future/PLAN-1-FOUNDATIONS.md) | Profiles, Provenance, Academic Output | PLAN-0 ResearchExtensions |
| [PLAN-2](future/PLAN-2-ACADEMIC-TOOLS.md) | OpenAlex, Crossref, Citation Tools | PLAN-0 Supervision refactoring |
| [PLAN-4](future/PLAN-4-DEEP-ANALYSIS.md) | PDF, Citation Networks, Methodology | PLAN-0 + PLAN-3 |
