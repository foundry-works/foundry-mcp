# PLAN-0: Prerequisites — Remaining Supervision Refactoring & State Model Architecture

> **Goal**: Prepare the codebase for PLAN-1 through PLAN-4 by completing the supervision refactoring (partially done) and introducing the `ResearchExtensions` container to prevent `DeepResearchState` bloat.
>
> **Current state**: The original 3,445-line supervision module has already been split into three files:
> - `supervision.py` (2,174 lines) — orchestration, delegation loop, first-round decomposition
> - `supervision_coverage.py` (424 lines) — coverage assessment, delta computation, verdict parsing
> - `supervision_prompts.py` (847 lines) — 16+ pure prompt builder functions
>
> **Remaining scope**: ~200-400 LOC changes (further refactoring of the orchestration core) + ~200-300 LOC tests
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

## 1. Complete Supervision Refactoring

### Current State (Already Done)

The original supervision module has already been split along two boundaries:

**`supervision_coverage.py` (424 lines)** — standalone functions, no `self`:
- `critique_has_issues()` — verdict/issue parsing from critique text
- `build_per_query_coverage()` — per-sub-query coverage metrics
- `store_coverage_snapshot()` — persist coverage state
- `compute_coverage_delta()` — round-over-round coverage changes
- `assess_coverage_heuristic()` — 3-dimensional confidence scoring (source adequacy, domain diversity, completion rate)

**`supervision_prompts.py` (847 lines)** — pure prompt builder functions:
- `render_supervision_conversation_history()` — history rendering with sanitization
- `classify_query_complexity()` — simple/moderate/complex classification
- `build_combined_think_delegate_system_prompt()` / `build_combined_think_delegate_user_prompt()`
- `build_delegation_system_prompt()` / `build_delegation_user_prompt()`
- `build_first_round_think_system_prompt()` / `build_first_round_think_prompt()`
- `build_first_round_delegation_system_prompt()` / `build_first_round_delegation_user_prompt()`
- `build_critique_system_prompt()` / `build_critique_user_prompt()`
- `build_revision_system_prompt()` / `build_revision_user_prompt()`
- `build_think_prompt()` / `build_think_system_prompt()`

**`supervision.py` (2,174 lines)** — the remaining orchestration mixin with:
- `_execute_supervision_async()` entry point and delegation loop
- `_execute_supervision_delegation_async()` — main round loop (think→delegate→execute→assess)
- `_run_think_delegate_step()` — combined think+delegate LLM orchestration
- `_execute_and_merge_directives()` — parallel directive execution and source merging
- `_compress_directive_results_inline()` / `_build_directive_fallback_summary()` / `_build_evidence_inventory()`
- `_post_round_bookkeeping()` — coverage delta, state serialization, history trim
- `_should_exit_wall_clock()` / `_should_exit_heuristic()` — exit conditions
- `_first_round_decompose_critique_revise()` — decompose→critique→revise pipeline
- `_run_first_round_generate()` / `_run_first_round_critique()` / `_run_first_round_revise()`
- Thin wrapper methods delegating to `supervision_coverage` and `supervision_prompts`

### What Remains

The remaining `supervision.py` (2,174 lines) contains two logically distinct responsibility groups that could benefit from further separation, primarily to reduce merge conflicts when PLAN-1 (provenance logging) and PLAN-3 (influence scoring) modify these areas:

#### 1a. Extract first-round decomposition pipeline

**File: `phases/supervision_first_round.py`** (NEW — extracted from supervision.py)

Move the first-round decompose-critique-revise pipeline into a standalone module:
- `_first_round_decompose_critique_revise()` — orchestrates the full decompose→critique→revise cycle
- `_run_first_round_generate()` — initial decomposition LLM call
- `_run_first_round_critique()` — critique of generated sub-queries
- `_run_first_round_revise()` — revision based on critique feedback

These methods are self-contained — they call prompt builders from `supervision_prompts.py` and `critique_has_issues()` from `supervision_coverage.py`, but don't share state mutation patterns with the main delegation loop.

Implement as standalone async functions that take explicit parameters (state, config, memory), consistent with the existing extraction pattern used for `supervision_coverage.py` and `supervision_prompts.py`.

#### 1b. Evaluate further delegation extraction (optional)

The delegation loop (`_execute_supervision_delegation_async()` and its helpers) is the core orchestration logic and is tightly coupled — splitting it further would create fragmentation without clarity gain. However, the following self-contained helpers could optionally be extracted to a `supervision_helpers.py`:
- `_compress_directive_results_inline()`
- `_build_directive_fallback_summary()`
- `_build_evidence_inventory()`

This is lower priority than item 1a and may not be worth the indirection.

#### 1c. Remove thin wrapper methods

After item 1a, the remaining thin wrapper methods at the bottom of `supervision.py` (lines ~2107-2174) that simply delegate to `supervision_coverage` and `supervision_prompts` can be inlined at their call sites, reducing the file by ~60 lines.

**Target: supervision.py drops from 2,174 to ~1,700-1,800 lines** (more modest than the original target, reflecting that the high-value extractions are already done).

### Backward Compatibility

- `SupervisionPhaseMixin` public interface is unchanged
- All existing tests pass without modification
- `from phases.supervision import SupervisionPhaseMixin` still works
- Imports from `supervision_coverage` and `supervision_prompts` are unchanged

### Testing

- Existing supervision tests pass without changes (zero-diff verification)
- New test: import `supervision_first_round` module independently
- New test: verify first-round pipeline produces same results when called from extracted module
- ~30-50 LOC of new structural tests

---

## 2. Introduce `ResearchExtensions` Container Model

### Problem

Across PLAN-1 through PLAN-4, `DeepResearchState` (1,659 lines, 52 fields, 47 methods) gains approximately 7 new field groups:

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
| `phases/supervision_first_round.py` | **New** (extracted) | 1a |
| `phases/supervision_coverage.py` | Already exists (424 lines) | — |
| `phases/supervision_prompts.py` | Already exists (847 lines) | — |
| `phases/__init__.py` | Modify (exports) | 1 |
| `models/deep_research.py` | Modify | 2 (ResearchExtensions) |

## Execution Order

```
[1a. Extract first-round decomposition]──┐
[1c. Remove thin wrappers]───────────────┤ (sequential — 1c depends on 1a)
                                          │
[2. ResearchExtensions container]─────────  (independent of item 1)
```

Item 1a is the primary extraction. Item 1c is cleanup after 1a. Item 2 is fully independent.

## Success Criteria

- `supervision.py` is under 1,800 lines (down from 2,174)
- All existing tests pass with zero changes
- `DeepResearchState` has a single `extensions` field for all new capabilities
- No behavioral changes — purely structural
