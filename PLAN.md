# Supervisor-as-Sole-Orchestrator — Alignment Plan

Completes the transition to a supervisor-owned orchestration model where
SUPERVISION is the single phase responsible for both initial query decomposition
and iterative gap-filling. Eliminates the vestigial GATHERING re-entry loop and
folds PLANNING's self-critique quality gate into supervision round 0.

**Context:** The codebase is already ~70% aligned. BRIEF only does enrichment
(no decomposition). New workflows skip PLANNING/GATHERING and jump to
SUPERVISION after BRIEF. Supervision round 0 already performs initial
decomposition into directives. The delegation model always returns
`should_continue_gathering: False`. What remains is cleanup: removing the
GATHERING loop from the orchestrator, adding the critique mechanism that
PLANNING had, and marking legacy phases as deprecated.

**Comparison with open_deep_research (ODR):**
ODR has no separate planning or gathering phases. The supervisor is the sole
orchestrator: think → delegate → research → compress → assess → iterate. This
plan brings foundry-mcp to the same model.

---

## Phase 1: Self-Critique in Supervision Round 0

**Rationale:** PLANNING had a self-critique step (`_apply_critique_adjustments`
in planning.py:656-741) that caught redundant sub-queries, missing perspectives,
and scope issues. Supervision round 0 currently lacks this. Without it, the
first-round decomposition may produce overlapping directives.

**ODR pattern:** ODR's supervisor uses natural heuristic judgment — the LLM
reasons about redundancy and proportionality inline during delegation. No
separate critique pass needed because the instructions embed it.

**Changes:**

### 1a. Three-call decompose → critique → revise pipeline for first-round delegation

Replace the single first-round delegation LLM call with a 3-call pipeline that
mirrors PLANNING's separate critique step but keeps everything inside supervision
round 0. This gives the LLM a genuine opportunity to reflect on its own output
and revise, producing higher-quality directives than inline self-critique alone.

**Call 1 — Generate:** The existing first-round delegation call produces initial
directives (JSON). No prompt changes needed to the existing system/user prompts
for this call.

**Call 2 — Critique:** A new LLM call receives the initial directives and
evaluates them against four criteria:
1. **Redundancy:** Are any directives investigating the same topic from the same angle? Identify which to merge.
2. **Coverage:** Is there a major dimension of the query that no directive addresses? Identify what's missing.
3. **Proportionality:** For simple queries, could fewer directives suffice? Identify excess.
4. **Specificity:** Are directives specific enough, or too broad/vague? Identify which need sharpening.

The critique call returns structured feedback (not revised directives).

**Call 3 — Revise:** A new LLM call receives the original directives + the
critique feedback and produces the final revised directive set as JSON. If the
critique found no issues, this call may return the directives unchanged.

**Optimization:** If the critique indicates no issues (all four criteria pass),
skip call 3 and use the original directives directly.

**Implementation:** Add a new method `_first_round_decompose_critique_revise()`
that orchestrates the 3-call pipeline, called from `_supervision_delegate_step()`
when `is_first_round` is True. New helper methods:
- `_build_critique_system_prompt()` — system prompt for call 2
- `_build_critique_user_prompt(directives)` — user prompt with initial directives
- `_build_revision_system_prompt()` — system prompt for call 3
- `_build_revision_user_prompt(directives, critique)` — user prompt with directives + critique feedback

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- `_supervision_delegate_step()` (~line 563) — branch to new pipeline when first round
- New methods for critique and revision prompts and orchestration

### 1b. Log critique pipeline in audit trail

Add audit events for each stage of the 3-call pipeline to provide observability
into how the critique step affects directive quality:

- **`first_round_generate`** — after call 1, records initial directive count and topics
- **`first_round_critique`** — after call 2, records the critique findings (which criteria flagged issues)
- **`first_round_revise`** — after call 3, records final directive count, directives added/removed/merged, and whether revision was skipped (no issues found)
- **`first_round_decomposition`** — summary event with `initial_directive_count`, `final_directive_count`, `critique_triggered_revision: bool`, `query_complexity` fields

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- Inside `_first_round_decompose_critique_revise()` after each call

---

## Phase 2: Remove GATHERING from Active Workflow Loop

**Rationale:** The delegation model (`_execute_supervision_delegation_async`)
handles all research execution internally via `_execute_directives_async()` and
always returns `should_continue_gathering: False` (supervision.py:466). The
GATHERING ↔ SUPERVISION loop in the workflow orchestrator (workflow_execution.py
lines 221-317) is dead code for the active path.

**ODR pattern:** ODR's supervisor loop is self-contained — delegate, execute,
assess, repeat. No external phase re-entry.

**Changes:**

### 2a. Simplify the iteration loop in workflow_execution.py

Replace the `while True` GATHERING ↔ SUPERVISION loop (lines 221-317) with
a linear SUPERVISION → SYNTHESIS progression. The supervision phase's internal
multi-round loop (bounded by `max_supervision_rounds`) is the sole iteration
mechanism.

Before:
```python
while True:
    if state.phase == DeepResearchPhase.GATHERING:
        err = await self._run_phase(state, GATHERING, ...)
    if state.phase == DeepResearchPhase.SUPERVISION:
        err = await self._run_phase(state, SUPERVISION, ...)
        if should_gather and has_pending and within_limit:
            state.phase = GATHERING; continue
        else:
            state.phase = SYNTHESIS
    if state.phase == DeepResearchPhase.SYNTHESIS:
        ...; break
    break
```

After:
```python
# Legacy resume: if saved state is at GATHERING, run it once then proceed
if state.phase == DeepResearchPhase.GATHERING:
    err = await self._run_phase(state, GATHERING, ...)
    if err: return err
    state.phase = DeepResearchPhase.SUPERVISION

# SUPERVISION (handles decomposition + research + gap-fill internally)
if state.phase == DeepResearchPhase.SUPERVISION:
    state.metadata["iteration_in_progress"] = True
    err = await self._run_phase(state, SUPERVISION, ..., skip_transition=True)
    if err: return err
    state.phase = DeepResearchPhase.SYNTHESIS

# SYNTHESIS
if state.phase == DeepResearchPhase.SYNTHESIS:
    err = await self._run_phase(state, SYNTHESIS, ..., skip_transition=True)
    if err: return err
    # ... orchestrator transition, mark_completed, break
```

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
- `_execute_workflow_async()` (lines 218-317)

### 2b. Remove `should_continue_gathering` from supervision return metadata

The delegation model's WorkflowResult always sets `should_continue_gathering: False`.
Remove this field from the return metadata since it's no longer consumed by the
orchestrator. Keep it in the `supervision_history` audit entries for observability.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`
- `_execute_supervision_delegation_async()` return value (~line 456-471)

### 2c. Remove `pending_sub_queries()` / `should_continue_gathering` check from orchestrator

The `has_pending = len(state.pending_sub_queries()) > 0` check (line 261) and
`should_gather = last_hist.get("should_continue_gathering", False)` (line 260)
are part of the GATHERING re-entry logic. Remove them from the orchestrator.
`pending_sub_queries()` itself stays on the state model (used by coverage
assessment internally).

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
- Lines 258-266 (the should_gather / has_pending / within_limit block)

---

## Phase 3: Deprecate PLANNING and GATHERING Phase Enum Values

**Rationale:** New workflows never enter these phases. Marking them explicitly
as deprecated prevents future code from accidentally routing through them and
makes the architectural intent clear.

**Changes:**

### 3a. Add deprecation comments to phase enum

Add inline comments to `DeepResearchPhase.GATHERING` marking it as
legacy-resume-only. Note: there is no `PLANNING` value in the enum (it was
already removed). Only GATHERING remains as a legacy phase.

**File:** `src/foundry_mcp/core/research/models/deep_research.py`
- `DeepResearchPhase` enum (~line 644)

### 3b. Strengthen deprecation logging for GATHERING resume

The legacy resume path (workflow_execution.py lines 198-214) already logs a
warning. Ensure the audit event includes `deprecated_phase: true` for
monitoring dashboards.

**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

### 3c. Remove unused PLANNING imports from workflow execution

If `_execute_planning_async` or other planning-specific functions are imported
in workflow_execution.py or the main workflow module, remove them from the
active path. Planning.py itself stays for potential legacy resume use.

**Files:**
- `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`
- Any other files with unused planning imports

---

## Dependency Graph

```
Phase 1 (supervision critique)  ← independent
Phase 2 (remove GATHERING loop) ← independent
Phase 3 (deprecate enum values) ← depends on Phase 2
```

Phases 1 and 2 can be implemented in parallel. Phase 3 is cleanup after Phase 2.

---

## Files Modified

| File | Phases | Description |
|------|--------|-------------|
| `phases/supervision.py` | 1a, 1b, 2b | 3-call decompose→critique→revise pipeline; audit events; remove should_continue_gathering from return |
| `workflow_execution.py` | 2a, 2c, 3b, 3c | Simplify loop; remove GATHERING re-entry; deprecation audit; clean imports |
| `models/deep_research.py` | 3a | Deprecation comments on phase enum |

All paths relative to `src/foundry_mcp/core/research/workflows/deep_research/`.

---

## What Is NOT Changed

- **TopicResearchMixin** — unchanged, still the ReAct loop engine
- **SubQuery model** — stays as the internal execution unit (directives convert to SubQuery via `state.add_sub_query()`)
- **Compression pipeline** — unchanged
- **Coverage assessment** — unchanged (operates on completed sub-queries internally)
- **Synthesis** — unchanged
- **Evaluation** — unchanged
- **PLANNING phase code** (planning.py) — kept for legacy compat, not deleted
- **GATHERING phase code** (gathering.py) — kept for legacy compat, not deleted
- **Config parameters** — no new parameters needed

---

## Verification

1. **Unit tests**: Run existing deep research test suite for regressions
2. **Integration test**: Execute a full deep research workflow and verify:
   - BRIEF → SUPERVISION → SYNTHESIS (no GATHERING phase entered)
   - Supervision round 0 produces directives with critique evidence in audit log
   - Supervision rounds 1+ operate as before (gap-fill via directives)
3. **Legacy resume**: Create a saved state at GATHERING phase, resume it, verify
   it still works through the legacy path with deprecation warning logged
4. **Audit trail**: Verify supervision_history entries no longer contain
   `should_continue_gathering` in the WorkflowResult metadata

```bash
python -m pytest tests/ -x -q --timeout=120
```

---

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1 | Inline critique adds tokens to first-round delegation prompt | Minimal — ~100 tokens of instruction; first-round prompt is not token-constrained |
| 2 | Legacy saved states expect GATHERING in the loop | Legacy resume path preserved; only the active-path loop changes |
| 3 | Code referencing `DeepResearchPhase.GATHERING` breaks | Enum values stay; only comments and routing change |
