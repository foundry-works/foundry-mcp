# PLAN-CHECKLIST: Pre-Merge Hardening — Review Findings Remediation

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-25

---

## Phase 1: Runtime Crash Fixes (Critical Path)

- [ ] **1A.1** Add `deep_research_reflection_timeout: float = 60.0` to `ResearchConfig` dataclass
- [ ] **1A.2** Add `deep_research_reflection_timeout` entry in `ResearchConfig.from_toml_dict()`
- [ ] **1A.3** Audit `getattr` call sites — confirm all use 60.0 as fallback default
- [ ] **1B.1** Replace broad regex pattern in `_CONTEXT_WINDOW_ERROR_PATTERNS` with tighter token+limit patterns
- [ ] **1B.2** Remove `"InvalidArgument"` from `_CONTEXT_WINDOW_ERROR_CLASSES`
- [ ] **1B.3** Add combined class+message check for `InvalidArgument` with token-related messages
- [ ] **1B.4** Add false-positive regression tests (auth token, context param, length positive)
- [ ] **1C.1** Replace `ThreadPoolExecutor` + `asyncio.run` with proper async dispatch in evaluation handler
- [ ] **1C.2** Remove the `RuntimeError` fallback masking the fragility
- [ ] **1C.3** Verify evaluation still works via `deep-research-evaluate` action end-to-end

---

## Phase 2: Data Quality & Correctness

- [x] **2A.1** Delete `_extract_domain` from `_helpers.py` (lines 784-803)
- [x] **2A.2** Import `_extract_domain` from `source_quality` in `_helpers.py`
- [x] **2A.3** Update `compute_novelty_tag` to use imported version
- [x] **2A.4** Grep for any other references to deleted function — update if found
- [x] **2B.1** Define `DEFAULT_MAX_SUPERVISION_ROUNDS = 6` constant in `models/deep_research.py`
- [x] **2B.2** Update `DeepResearchState.max_supervision_rounds` Field default to use constant
- [x] **2B.3** Update `action_handlers.py` `getattr` fallback to use constant
- [x] **2C.1** Add `imputed: bool = False` field to `DimensionScore` dataclass
- [x] **2C.2** Set `imputed=True` in fallback score path (`evaluator.py:199-201`)
- [x] **2C.3** Add `imputed_count` and `warnings` to `EvaluationResult.metadata`
- [x] **2C.4** Exclude imputed dimensions from variance calculation
- [x] **2D.1** Compute weighted variance using effective weights and weighted mean
- [x] **2D.2** Update variance tests to reflect weighted computation

---

## Phase 3: Concurrency & Error Handling

- [ ] **3A.1** Change `return_exceptions=False` to `True` in directive execution gather (`supervision.py:1034`)
- [ ] **3A.2** Add post-gather `CancelledError` detection and re-raise
- [ ] **3A.3** Log other `BaseException` results as non-fatal directive failures
- [ ] **3B.1** Add `CancelledError` detection after compression gather (`compression.py:757`)
- [ ] **3B.2** Re-raise `CancelledError` to propagate cancellation upward
- [ ] **3C.1** Add cumulative content length tracking in `_build_structured_metadata_prompt`
- [ ] **3C.2** Stop adding source content at global cap (200k chars)
- [ ] **3C.3** Add truncation marker for omitted sources
- [ ] **3D.1** Track `total_remaining` during rebalance phase in `truncate_supervision_messages`
- [ ] **3D.2** Add guard to break when `total_remaining >= budget_chars`

---

## Phase 4: State Model Hygiene

- [ ] **4A.1** Rename `ReflectionDecision` → `PhaseReflectionDecision` in `orchestration.py`
- [ ] **4A.2** Update all references in `orchestration.py` (type annotations, instantiation)
- [ ] **4A.3** Search for external imports and update
- [ ] **4B.1** Add `_SKIP_PHASES` set to `advance_phase()` containing `GATHERING`
- [ ] **4B.2** Implement skip logic: advance again if landed on a skip phase
- [ ] **4B.3** Remove direct `state.phase = DeepResearchPhase.SUPERVISION` workaround in `workflow_execution.py`
- [ ] **4B.4** Verify existing phase lifecycle tests pass with skip logic
- [ ] **4C.1** Add `_MAX_STORED_DIRECTIVES = 30` constant to `supervision.py`
- [ ] **4C.2** Cap `state.directives` after each `extend()` call

---

## Phase 5: Prompt Injection Surface Reduction

- [ ] **5A.1** Add `sanitize_external_content()` function to `_helpers.py`
  - Strip XML-like instruction tags (`<system>`, `<instructions>`, `<tool_use>`, `<human>`, `<assistant>`)
  - Strip markdown heading injection patterns (`# SYSTEM`, `## INSTRUCTIONS`)
- [ ] **5A.2** Apply sanitizer to source titles/snippets in supervision message rendering
- [ ] **5A.3** Apply sanitizer in `_build_directive_fallback_summary` for source-derived content
- [ ] **5A.4** Add unit tests for sanitizer (confirm tags stripped, normal content preserved)

---

## Phase 6: Test Gaps & Quality

- [ ] **6A.1** Fix `test_evaluator.py:746` — capture user prompt at `execute_llm_call` layer
- [ ] **6A.2** Remove `or True` fallback from assertion
- [ ] **6A.3** Assert raw notes content appears in captured prompt
- [ ] **6B.1** Add test: cancellation during directive execution propagates upward
- [ ] **6B.2** Add test: cancellation during compression gather propagates upward
- [ ] **6B.3** Add test: partial results preserved on mid-batch cancellation
- [ ] **6C.1** Test: "invalid authentication token" NOT classified as context-window error
- [ ] **6C.2** Test: "context parameter is required" NOT classified as context-window error
- [ ] **6C.3** Test: "maximum context length exceeded" IS classified as context-window error
- [ ] **6C.4** Test: `InvalidArgument` with non-token message NOT classified as context-window error

---

## Final Validation

- [ ] Full deep research test suite passes: `python -m pytest tests/core/research/ -x -q`
- [ ] Evaluation tests pass: `python -m pytest tests/core/research/evaluation/ -x -q`
- [ ] Config smoke test: `ResearchConfig().deep_research_reflection_timeout == 60.0`
- [ ] Renamed class import: `from ...orchestration import PhaseReflectionDecision`
- [ ] No regressions in existing workflow tests
