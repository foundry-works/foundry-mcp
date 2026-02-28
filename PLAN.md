# PLAN-0: Prerequisites — Supervision Refactoring & State Model Architecture

> **Branch**: `deep-academic`
>
> **Goal**: Prepare the codebase for PLAN-1 through PLAN-4 by completing the supervision refactoring (partially done) and introducing the `ResearchExtensions` container to prevent `DeepResearchState` bloat.
>
> **Risk**: Low. Pure refactoring with no behavioral changes.
>
> **Source**: [future/PLAN-0-PREREQUISITES.md](future/PLAN-0-PREREQUISITES.md)

---

## Design Principles

1. **Extract, don't rewrite.** Move existing code into focused modules without changing behavior. Every refactored method must pass its existing tests unchanged.
2. **State model discipline.** New feature state goes into a `ResearchExtensions` container, not directly onto `DeepResearchState`. One field on state, structured sub-models within.
3. **No functional changes.** This plan is purely structural. If a test changes, something went wrong.

---

## Current State

The original 3,445-line supervision module has already been split into three files:

| File | Lines | Contents |
|------|-------|----------|
| `supervision.py` | 2,174 | Orchestration, delegation loop, first-round decomposition |
| `supervision_coverage.py` | 424 | Coverage assessment, delta computation, verdict parsing |
| `supervision_prompts.py` | 847 | 16+ pure prompt builder functions |

---

## Phase 1: Complete Supervision Refactoring

### 1a. Extract first-round decomposition pipeline

**New file: `phases/supervision_first_round.py`**

Move the first-round decompose-critique-revise pipeline into a standalone module:

- `_first_round_decompose_critique_revise()` — orchestrates the full decompose->critique->revise cycle
- `_run_first_round_generate()` — initial decomposition LLM call
- `_run_first_round_critique()` — critique of generated sub-queries
- `_run_first_round_revise()` — revision based on critique feedback

These methods are self-contained — they call prompt builders from `supervision_prompts.py` and `critique_has_issues()` from `supervision_coverage.py`, but don't share state mutation patterns with the main delegation loop.

Implement as standalone async functions that take explicit parameters (state, config, memory), consistent with the existing extraction pattern used for `supervision_coverage.py` and `supervision_prompts.py`.

### 1b. Evaluate further delegation extraction (optional, lower priority)

The delegation loop (`_execute_supervision_delegation_async()` and its helpers) is tightly coupled — splitting it further would create fragmentation. However, these self-contained helpers could optionally move to `supervision_helpers.py`:

- `_compress_directive_results_inline()`
- `_build_directive_fallback_summary()`
- `_build_evidence_inventory()`

### 1c. Remove thin wrapper methods

After 1a, the remaining thin wrapper methods at the bottom of `supervision.py` (~lines 2107-2174) that simply delegate to `supervision_coverage` and `supervision_prompts` can be inlined at their call sites, reducing the file by ~60 lines.

**Target: `supervision.py` drops from 2,174 to ~1,700-1,800 lines.**

### Backward Compatibility

- `SupervisionPhaseMixin` public interface is unchanged
- All existing tests pass without modification
- `from phases.supervision import SupervisionPhaseMixin` still works
- Imports from `supervision_coverage` and `supervision_prompts` are unchanged

### Testing

- Existing supervision tests pass with zero changes (zero-diff verification)
- New test: import `supervision_first_round` module independently
- New test: verify first-round pipeline produces same results when called from extracted module
- ~30-50 LOC of new structural tests

---

## Phase 2: Introduce `ResearchExtensions` Container Model

### Problem

Across PLAN-1 through PLAN-4, `DeepResearchState` (1,659 lines, 52 fields, 47 methods) gains ~7 new field groups:

| Plan | New Fields |
|------|-----------|
| PLAN-1 | `research_profile`, `provenance`, `structured_output` |
| PLAN-3 | `research_landscape`, `study_comparisons` |
| PLAN-4 | `citation_network`, `methodology_assessments` |

### Design

```python
class ResearchExtensions(BaseModel):
    """Container for extended research capabilities.

    All fields from PLAN-1 through PLAN-4 live here rather than
    directly on DeepResearchState. This keeps the core state model
    stable and serialization cost proportional to features used.
    """

    # PLAN-1: Foundations
    research_profile: Optional["ResearchProfile"] = None
    provenance: Optional["ProvenanceLog"] = None
    structured_output: Optional["StructuredResearchOutput"] = None

    # PLAN-3: Intelligence
    research_landscape: Optional["ResearchLandscape"] = None

    # PLAN-4: Deep Analysis
    citation_network: Optional["CitationNetwork"] = None
    methodology_assessments: list["MethodologyAssessment"] = Field(default_factory=list)

    class Config:
        exclude_none = True
```

### 2a. Add ResearchExtensions model

Add the container model definition to `models/deep_research.py`. Initially empty — each subsequent plan populates it.

### 2b. Add extensions field to DeepResearchState

```python
extensions: ResearchExtensions = Field(
    default_factory=ResearchExtensions,
    description="Extended capabilities from PLAN-1 through PLAN-4",
)
```

### 2c. Add convenience accessors

```python
@property
def research_profile(self) -> Optional["ResearchProfile"]:
    return self.extensions.research_profile

@property
def provenance(self) -> Optional["ProvenanceLog"]:
    return self.extensions.provenance
```

### Serialization Behavior

- `exclude_none=True` means empty extensions add zero overhead to state serialization
- Provenance (which can grow large) is persisted separately (as specified in PLAN-1 item 2g)
- The extensions field itself is always present but lightweight when unused

### Testing

- Unit test: `ResearchExtensions()` default is empty, serializes to `{}`
- Unit test: extensions with one field populated serializes only that field
- Unit test: `DeepResearchState` with default extensions is backward-compatible with existing serialized states
- Unit test: convenience property accessors work correctly
- ~40-60 LOC of new tests

---

## File Impact Summary

| File | Change Type | Item |
|------|-------------|------|
| `phases/supervision.py` | Refactor (shrink) | 1 |
| `phases/supervision_first_round.py` | **New** (extracted) | 1a |
| `phases/supervision_coverage.py` | Already exists (424 lines) | — |
| `phases/supervision_prompts.py` | Already exists (847 lines) | — |
| `phases/__init__.py` | Modify (exports) | 1 |
| `models/deep_research.py` | Modify | 2 |

## Execution Order

```
[1a. Extract first-round decomposition] --+
[1c. Remove thin wrappers] ---------------+ (sequential — 1c depends on 1a)
                                           |
[2. ResearchExtensions container] ---------  (independent of item 1)
```

Item 1a is the primary extraction. Item 1c is cleanup after 1a. Item 2 is fully independent.

## Success Criteria

- [ ] `supervision.py` is under 1,800 lines (down from 2,174)
- [ ] All existing tests pass with zero changes
- [ ] `DeepResearchState` has a single `extensions` field for all new capabilities
- [ ] No behavioral changes — purely structural

## What Comes Next

After PLAN-0 completes, significant parallel work opens up:

| Next Plan | Theme | Hard Dep on PLAN-0 |
|-----------|-------|--------------------|
| [PLAN-1](future/PLAN-1-FOUNDATIONS.md) | Profiles, Provenance, Academic Output | ResearchExtensions container |
| [PLAN-2](future/PLAN-2-ACADEMIC-TOOLS.md) | OpenAlex, Crossref, Citation Tools | Supervision refactoring |
| [PLAN-3](future/PLAN-3-RESEARCH-INTELLIGENCE.md) | Ranking, Landscape, Export | ResearchExtensions container |
| [PLAN-4](future/PLAN-4-DEEP-ANALYSIS.md) | PDF, Citation Networks, Methodology | ResearchExtensions + Supervision |
