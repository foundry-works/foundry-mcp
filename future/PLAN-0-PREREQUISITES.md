# PLAN-0: Prerequisites — Supervision Refactoring & State Model Architecture

> **Goal**: Prepare the codebase for PLAN-1 through PLAN-4 by addressing two structural bottlenecks: `supervision.py` (3,340 lines, modified by 3 of 4 plans) and `DeepResearchState` (1,517 lines, gaining ~7 new field groups across all plans).
>
> **Estimated scope**: ~400-600 LOC changes (refactoring, not new features) + ~200-300 LOC tests
>
> **Dependencies**: None — this is the foundation for everything else
>
> **Risk**: Low. Pure refactoring with no behavioral changes.

---

## Design Principles

1. **Extract, don't rewrite.** Move existing code into focused modules without changing behavior. Every refactored method must pass its existing tests unchanged.
2. **State model discipline.** New feature state goes into a `ResearchExtensions` container, not directly onto `DeepResearchState`. One field on state, structured sub-models within.
3. **No functional changes.** This plan is purely structural. If a test changes, something went wrong.

---

## 1. Refactor `supervision.py` into Focused Modules

### Problem

`supervision.py` is 3,340 lines containing 6 distinct responsibilities:

| Section | Lines | Responsibility |
|---------|-------|---------------|
| Main entry point | 92-121 | Routes between delegation and legacy models |
| Delegation model | 119-1777 | Parallel directive execution, compression, evidence inventory |
| First-round decomposition | 1777-1971 | Initial query decomposition prompts |
| Decompose-critique-revise | 1971-2382 | Multi-step first-round pipeline |
| Legacy query-generation | 2382-2637 | Fallback supervision model |
| Shared helpers | 2637-3340 | Coverage assessment, prompt builders, response parsers |

Three of four subsequent plans modify this file (PLAN-1 adds provenance, PLAN-2 adds provider delegation, PLAN-3 adds influence scoring). Without refactoring, merge conflicts and cognitive load will compound.

### Changes

#### 1a. Extract delegation model

**File: `phases/supervision_delegation.py`** (NEW — extracted from supervision.py)

Move lines 119-1777 into a new module containing:
- `_execute_supervision_delegation_async()`
- `_supervision_think_step()` / `_supervision_delegate_step()` / `_supervision_combined_think_delegate_step()`
- `_build_combined_think_delegate_system_prompt()` / `_build_combined_think_delegate_user_prompt()`
- `_parse_combined_response()` / `_extract_gap_analysis_section()`
- `_execute_directives_async()` / `_compress_directive_results_inline()`
- `_build_directive_fallback_summary()` / `_build_evidence_inventory()`
- `_classify_query_complexity()`
- `_build_delegation_system_prompt()` / `_build_delegation_user_prompt()`
- `_apply_directive_caps()` / `_parse_delegation_response()`

Implement as a mixin class `DelegationMixin` that `SupervisionPhaseMixin` inherits from.

#### 1b. Extract first-round decomposition

**File: `phases/supervision_first_round.py`** (NEW — extracted from supervision.py)

Move lines 1777-2382 into a new module containing:
- `_build_first_round_think_system_prompt()` / `_build_first_round_think_prompt()`
- `_build_first_round_delegation_system_prompt()` / `_build_first_round_delegation_user_prompt()`
- `_first_round_decompose_critique_revise()`
- `_build_critique_system_prompt()` / `_build_critique_user_prompt()`
- `_build_revision_system_prompt()` / `_build_revision_user_prompt()`
- `_critique_has_issues()`

Implement as a mixin class `FirstRoundMixin` that `SupervisionPhaseMixin` inherits from.

#### 1c. Extract coverage assessment

**File: `phases/supervision_coverage.py`** (NEW — extracted from supervision.py)

Move `_assess_coverage_heuristic()` and its helpers into a focused module. This is the method that PLAN-3 will modify for influence-aware scoring — isolating it prevents conflicts with delegation and first-round changes.

Implement as a mixin class `CoverageMixin` or as standalone functions that take `DeepResearchState` as input.

#### 1d. Slim down supervision.py

After extraction, `supervision.py` becomes:
- Imports from the three new modules
- `SupervisionPhaseMixin` class definition (inheriting `DelegationMixin`, `FirstRoundMixin`, `CoverageMixin`)
- `_execute_supervision_async()` entry point (~30 lines)
- Legacy query-generation model (~250 lines — small enough to keep inline)
- Remaining shared prompt/parse helpers (~400 lines)

**Target: supervision.py drops from 3,340 to ~700-800 lines.**

### Backward Compatibility

- `SupervisionPhaseMixin` public interface is unchanged
- All existing tests pass without modification
- `from phases.supervision import SupervisionPhaseMixin` still works
- Internal method names are preserved (just moved to mixin base classes)

### Testing

- Existing supervision tests pass without changes (zero-diff verification)
- New test: import each extracted module independently
- New test: verify mixin composition produces same method resolution order
- ~50-80 LOC of new structural tests

---

## 2. Introduce `ResearchExtensions` Container Model

### Problem

Across PLAN-1 through PLAN-4, `DeepResearchState` (1,517 lines) gains approximately 7 new field groups:

| Plan | New Fields |
|------|-----------|
| PLAN-1 | `research_profile`, `provenance`, `structured_output` |
| PLAN-3 | `research_landscape`, `study_comparisons` |
| PLAN-4 | `citation_network`, `methodology_assessments` |

Adding these directly to `DeepResearchState` would push it past 2,000 lines and increase serialization/deserialization cost for every state save (which happens after each supervision round).

### Design

A `ResearchExtensions` container groups all plan-added state under a single field:

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
        # Only serialize non-None fields to keep state compact
        exclude_none = True
```

### Changes

**File: `src/foundry_mcp/core/research/models/deep_research.py`**

#### 2a. Add ResearchExtensions model

Add the container model definition. Initially empty — each subsequent plan populates it.

#### 2b. Add extensions field to DeepResearchState

```python
# In DeepResearchState:
extensions: ResearchExtensions = Field(
    default_factory=ResearchExtensions,
    description="Extended capabilities from PLAN-1 through PLAN-4",
)
```

#### 2c. Add convenience accessors

```python
@property
def research_profile(self) -> Optional["ResearchProfile"]:
    return self.extensions.research_profile

@property
def provenance(self) -> Optional["ProvenanceLog"]:
    return self.extensions.provenance
```

This lets downstream code use `state.research_profile` naturally while the storage is organized.

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

| File | Change Type | Items |
|------|-------------|-------|
| `phases/supervision.py` | Refactor (shrink) | 1 |
| `phases/supervision_delegation.py` | **New** (extracted) | 1a |
| `phases/supervision_first_round.py` | **New** (extracted) | 1b |
| `phases/supervision_coverage.py` | **New** (extracted) | 1c |
| `phases/__init__.py` | Modify (exports) | 1 |
| `models/deep_research.py` | Modify | 2 (ResearchExtensions) |

## Execution Order

```
[1a. Extract delegation model]──────────┐
[1b. Extract first-round decomposition]─┤ (parallel — independent extractions)
[1c. Extract coverage assessment]────────┤
                                          │
[1d. Slim down supervision.py]───────────┘ (after all extractions)
                                          │
[2. ResearchExtensions container]─────────  (independent of item 1)
```

Items 1a-1c can be done in parallel. Item 1d is their integration point. Item 2 is fully independent.

## Success Criteria

- `supervision.py` is under 800 lines
- All existing tests pass with zero changes
- `DeepResearchState` has a single `extensions` field for all new capabilities
- No behavioral changes — purely structural
