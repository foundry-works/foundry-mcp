# Supervisor-as-Sole-Orchestrator — Checklist

Cross-reference: [PLAN.md](PLAN.md) for detailed rationale, ODR patterns, and code references.

**Note:** BRIEF phase already only does enrichment (no decomposition). Supervision
round 0 already performs initial decomposition. This plan completes the alignment
by adding self-critique, removing the GATHERING re-entry loop, and deprecating
legacy phases.

---

## Phase 1: Self-Critique in Supervision Round 0

- [x] **1a** Implement 3-call decompose → critique → revise pipeline for first-round delegation
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - **Call 1 (Generate):** Existing first-round delegation call produces initial directives JSON (no prompt changes needed)
  - **Call 2 (Critique):** New LLM call evaluates initial directives against: redundancy, coverage, proportionality, specificity
    - Add: `_build_critique_system_prompt()`, `_build_critique_user_prompt(directives)`
    - Returns structured critique feedback (not revised directives)
  - **Call 3 (Revise):** New LLM call receives original directives + critique → produces final revised JSON
    - Add: `_build_revision_system_prompt()`, `_build_revision_user_prompt(directives, critique)`
    - Optimization: skip call 3 if critique finds no issues
  - Add: `_first_round_decompose_critique_revise()` orchestrator method
  - Modify: `_supervision_delegate_step()` to branch to new pipeline when `is_first_round`
  - Remove: Self-Critique Checklist from `_build_first_round_delegation_system_prompt()` (now handled by separate call)
  - Test: Existing supervision tests still pass; first-round produces critique audit events

- [x] **1b** Add audit events for critique pipeline stages
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: Inside `_first_round_decompose_critique_revise()` after each call
  - Events: `first_round_generate`, `first_round_critique`, `first_round_revise`, `first_round_decomposition` (summary)
  - Summary event fields: `initial_directive_count`, `final_directive_count`, `critique_triggered_revision`, `query_complexity`
  - Test: Verify all audit events appear in supervision audit trail

---

## Phase 2: Remove GATHERING from Active Workflow Loop

Depends on: None (independent of Phase 1).

- [x] **2a** Simplify `_execute_workflow_async()` iteration loop
  - File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
  - Replace: `while True` loop (lines 221-317) with linear SUPERVISION → SYNTHESIS
  - Keep: Legacy GATHERING entry (lines 198-214) for saved-state resume, but transition to SUPERVISION after it completes (not loop)
  - Test: Full workflow test — verify phases are CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS
  - Test: No GATHERING phase appears in phase_metrics for new workflows

- [x] **2b** Remove `should_continue_gathering` from supervision WorkflowResult metadata
  - File: `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
  - Location: `_execute_supervision_delegation_async()` return (~line 456-471)
  - Remove: `"should_continue_gathering": False` from metadata dict
  - Keep: `should_continue_gathering` in supervision_history audit entries (observability)
  - Test: Verify WorkflowResult metadata no longer contains the field

- [x] **2c** Remove GATHERING re-entry logic from orchestrator
  - File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
  - Remove: `should_gather` / `has_pending` / `within_limit` check block (lines 258-266)
  - Remove: `state.phase = DeepResearchPhase.GATHERING; continue` branch
  - Test: Verify orchestrator never sets phase to GATHERING during normal execution

---

## Phase 3: Deprecate PLANNING and GATHERING Phase Enum Values

Depends on: Phase 2 (GATHERING loop removed from active path).

- [x] **3a** Add deprecation comments to `DeepResearchPhase` enum
  - File: `src/foundry_mcp/core/research/models/deep_research.py`
  - Location: `DeepResearchPhase` enum (~line 644)
  - Add: `# DEPRECATED: legacy-resume-only` comment on GATHERING value
  - Test: Enum still serializes/deserializes correctly (no functional change)

- [x] **3b** Add `deprecated_phase: true` to GATHERING resume audit event
  - File: `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
  - Location: Legacy resume audit event (lines 206-214)
  - Add: `"deprecated_phase": True` to audit event data dict
  - Test: Verify audit event includes the deprecation flag

- [x] **3c** Remove unused PLANNING imports from workflow execution
  - Files: `workflow_execution.py`, any other active-path modules
  - Check: `grep -r "_execute_planning_async" src/` and remove unused imports
  - Test: No import errors; existing tests pass
