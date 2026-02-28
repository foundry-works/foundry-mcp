# PLAN-0 Checklist: Prerequisites

> Track implementation progress for [PLAN.md](PLAN.md).
> Mark items `[x]` as completed. Sub-items can be worked in parallel where noted.

---

## Phase 1: Complete Supervision Refactoring

### 1a. Extract first-round decomposition pipeline
> **New file**: `phases/supervision_first_round.py`
> **Parallel with**: Phase 2 (independent)

- [ ] Identify exact line ranges for the 4 methods to extract from `supervision.py`
  - [ ] `_first_round_decompose_critique_revise()`
  - [ ] `_run_first_round_generate()`
  - [ ] `_run_first_round_critique()`
  - [ ] `_run_first_round_revise()`
- [ ] Determine function signatures (state, config, memory params) matching existing extraction pattern
- [ ] Create `phases/supervision_first_round.py` with extracted functions
- [ ] Update `supervision.py` to import and delegate to extracted functions
- [ ] Update `phases/__init__.py` exports if needed
- [ ] Verify all existing supervision tests pass unchanged (zero-diff)
- [ ] Add structural test: import `supervision_first_round` module independently
- [ ] Add test: first-round pipeline produces same results via extracted module

### 1b. Evaluate further delegation extraction (optional)
> **Priority**: Low — skip if 1a achieves target line count

- [ ] Assess whether extracting these helpers to `supervision_helpers.py` is worth it:
  - [ ] `_compress_directive_results_inline()`
  - [ ] `_build_directive_fallback_summary()`
  - [ ] `_build_evidence_inventory()`
- [ ] If proceeding: create `supervision_helpers.py`, update imports, verify tests

### 1c. Remove thin wrapper methods
> **Depends on**: 1a complete

- [ ] Identify thin wrapper methods at bottom of `supervision.py` (~lines 2107-2174)
- [ ] Inline each wrapper at its call site(s)
- [ ] Remove ~60 lines of wrapper methods
- [ ] Verify all existing tests pass unchanged

### Phase 1 Validation

- [ ] `supervision.py` is under 1,800 lines
- [ ] `supervision_first_round.py` exists and is importable
- [ ] Full test suite passes with zero test modifications
- [ ] `from phases.supervision import SupervisionPhaseMixin` still works

---

## Phase 2: ResearchExtensions Container Model
> **Parallel with**: Phase 1 (independent)

### 2a. Add ResearchExtensions model
> **File**: `models/deep_research.py`

- [ ] Define `ResearchExtensions(BaseModel)` with `Config: exclude_none = True`
- [ ] Add placeholder fields with forward references (all `Optional`, all `None` default):
  - [ ] `research_profile: Optional["ResearchProfile"] = None`
  - [ ] `provenance: Optional["ProvenanceLog"] = None`
  - [ ] `structured_output: Optional["StructuredResearchOutput"] = None`
  - [ ] `research_landscape: Optional["ResearchLandscape"] = None`
  - [ ] `citation_network: Optional["CitationNetwork"] = None`
  - [ ] `methodology_assessments: list["MethodologyAssessment"] = Field(default_factory=list)`

### 2b. Add extensions field to DeepResearchState

- [ ] Add `extensions: ResearchExtensions = Field(default_factory=ResearchExtensions)` to `DeepResearchState`
- [ ] Verify existing serialized states are backward-compatible (default empty extensions)

### 2c. Add convenience accessors

- [ ] Add `@property research_profile` on `DeepResearchState`
- [ ] Add `@property provenance` on `DeepResearchState`
- [ ] Verify accessor behavior with both None and populated extensions

### Phase 2 Testing

- [ ] Unit test: `ResearchExtensions()` default serializes to `{}`
- [ ] Unit test: extensions with one field populated serializes only that field
- [ ] Unit test: `DeepResearchState` with default extensions is backward-compatible
- [ ] Unit test: convenience property accessors work correctly
- [ ] Full test suite passes with zero test modifications

---

## Final Validation

- [ ] `supervision.py` < 1,800 lines
- [ ] All existing tests pass with zero changes
- [ ] `DeepResearchState` has a single `extensions` field
- [ ] No behavioral changes — purely structural
- [ ] New structural tests all pass (~70-110 LOC total)

---

## Estimated Scope

| Item | Impl LOC | Test LOC |
|------|----------|----------|
| 1a. First-round extraction | ~100-200 | ~30-50 |
| 1c. Wrapper inlining | ~-60 (removal) | 0 |
| 2a-c. ResearchExtensions | ~100-200 | ~40-60 |
| **Total** | **~200-400** | **~70-110** |
