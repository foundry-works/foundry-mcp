# PLAN: Phase 6 — Iterative Supervisor Architecture

**Branch:** `tyler/foundry-mcp-20260223-0747`
**Date:** 2026-02-23
**Depends on:** Phases 1-5 (all complete)
**Status:** Draft

---

## Context

Phases 1-5 of the deep research alignment plan are complete. Phase 6 adds the most impactful architectural change: an iterative supervisor that can run multiple assess-delegate rounds within the gathering phase, filling coverage gaps before proceeding to analysis/synthesis. This is modeled after open_deep_research's supervisor loop but adapted to foundry-mcp's single-prompt (not multi-turn) architecture.

**Current flow:** CLARIFICATION → PLANNING → [GATHERING → ANALYSIS → SYNTHESIS → REFINEMENT]*
**New flow:** CLARIFICATION → PLANNING → [GATHERING → SUPERVISION → (loop back if gaps) → ANALYSIS → SYNTHESIS → REFINEMENT]*

### Key Architectural Insight

Gathering already only processes `pending_sub_queries()`, so supervision just needs to add new pending sub-queries and set the phase back to GATHERING. URL dedup via `seen_urls` (initialized from `state.sources`) prevents re-fetching. The `advance_phase()` method uses enum definition order, so inserting `SUPERVISION` between `GATHERING` and `ANALYSIS` in the enum makes phase progression automatic.

### How `_safe_orchestrator_transition` Enables This

After GATHERING completes, `_run_phase()` calls `_safe_orchestrator_transition()` which calls `state.advance_phase()`. With SUPERVISION in the enum, this naturally advances to SUPERVISION. The workflow loop's `if state.phase == DeepResearchPhase.SUPERVISION:` block then handles the supervision logic, including looping back to GATHERING or advancing to ANALYSIS.

---

## Sub-Phase 6.1: State Model Extensions

**Effort:** Low | **Impact:** Foundation for all other sub-phases
**File:** `src/foundry_mcp/core/research/models/deep_research.py`

### Changes

1. **Add `SUPERVISION` to `DeepResearchPhase` enum** between GATHERING and ANALYSIS (line 193):
   ```python
   GATHERING = "gathering"
   SUPERVISION = "supervision"  # NEW
   ANALYSIS = "analysis"
   ```
   This makes `advance_phase()` naturally go GATHERING → SUPERVISION → ANALYSIS.

2. **Add supervision tracking fields to `DeepResearchState`** (after `refinement_model`, line ~335):
   ```python
   # Supervision tracking (iterative coverage assessment)
   supervision_round: int = Field(default=0, description="Current supervision round within this iteration (0-based, resets each refinement iteration)")
   max_supervision_rounds: int = Field(default=3, description="Maximum supervisor assess-delegate rounds per iteration")
   supervision_provider: Optional[str] = Field(default=None)
   supervision_model: Optional[str] = Field(default=None)
   ```

3. **Update `start_new_iteration()`** (line 641) — add `self.supervision_round = 0` to reset supervision rounds for the new refinement iteration

4. **Update `advance_phase()` docstring** (line 607) — include SUPERVISION in the phase progression list

5. **Add `should_continue_supervision()` method** after `should_continue_refinement()` (line 639):
   ```python
   def should_continue_supervision(self) -> bool:
       """Check if another supervision round should occur."""
       if self.supervision_round >= self.max_supervision_rounds:
           return False
       return len(self.pending_sub_queries()) > 0
   ```

### Validation

- Unit test: SUPERVISION exists in enum between GATHERING and ANALYSIS
- Unit test: `advance_phase()` from GATHERING → SUPERVISION → ANALYSIS
- Unit test: `start_new_iteration()` resets `supervision_round` to 0
- Unit test: `should_continue_supervision()` returns correct values

---

## Sub-Phase 6.2: Configuration

**Effort:** Low | **Impact:** Enables supervision tuning
**File:** `src/foundry_mcp/config/research.py`

### Changes

1. **Add config fields** (in the deep research config section):
   - `deep_research_enable_supervision: bool = True` — master switch
   - `deep_research_max_supervision_rounds: int = 3` — max assess-delegate rounds per iteration
   - `deep_research_supervision_min_sources_per_query: int = 2` — minimum sources for "sufficient" coverage

2. **Add `"supervision"` to `_ROLE_RESOLUTION_CHAIN`** (line 916):
   ```python
   "supervision": ["supervision", "reflection"],
   ```
   Falls back to reflection provider (cheap model) when no supervision-specific provider configured.

3. **Add `"supervision"` to `get_phase_timeout()`** (line 729):
   ```python
   "supervision": self.deep_research_planning_timeout,  # Reuse planning timeout (lightweight LLM call)
   ```

4. **Wire `max_supervision_rounds` into `DeepResearchConfig`** — add field to dataclass, parse from TOML overrides, pass to state initialization

### Validation

- Config loads with new fields
- Role resolution chain for "supervision" falls back to "reflection"
- `get_phase_timeout("supervision")` returns expected value

---

## Sub-Phase 6.3: Orchestration Updates

**Effort:** Low | **Impact:** Integrates supervision into orchestrator framework
**File:** `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py`

### Changes

1. **Add `SUPERVISION` to `PHASE_TO_AGENT` mapping** (line 59):
   ```python
   DeepResearchPhase.SUPERVISION: AgentRole.SUPERVISOR,
   ```

2. **Add SUPERVISION to `_build_agent_inputs()`** (after GATHERING block, line ~310):
   ```python
   elif phase == DeepResearchPhase.SUPERVISION:
       return {
           **base_inputs,
           "completed_sub_queries": len(state.completed_sub_queries()),
           "total_sources": len(state.sources),
           "supervision_round": state.supervision_round,
           "max_supervision_rounds": state.max_supervision_rounds,
       }
   ```

3. **Add SUPERVISION to `_evaluate_phase_quality()`** (after GATHERING block, line ~409):
   ```python
   elif phase == DeepResearchPhase.SUPERVISION:
       pending = len(state.pending_sub_queries())
       return {
           "supervision_round": state.supervision_round,
           "pending_follow_ups": pending,
           "quality_ok": True,  # Supervision controls its own loop
           "rationale": f"Supervision round {state.supervision_round}: {pending} follow-up queries queued.",
       }
   ```

4. **Add SUPERVISION to `get_reflection_prompt()`** and **`_build_reflection_llm_prompt()`**

### Validation

- `PHASE_TO_AGENT[SUPERVISION]` returns `SUPERVISOR`
- `_build_agent_inputs` returns correct dict for SUPERVISION
- `_evaluate_phase_quality` returns correct structure for SUPERVISION

---

## Sub-Phase 6.4: Supervision Phase Mixin

**Effort:** High | **Impact:** Core supervisor logic
**New file:** `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py`

Follow the pattern of `refinement.py` for structure, prompts, and lifecycle.

### Methods

#### `_execute_supervision_async(state, provider_id, timeout) → WorkflowResult`
Main entry point:
1. Build per-sub-query coverage data (source count, quality distribution, domain diversity)
2. If heuristic says all queries are sufficiently covered AND round > 0, skip LLM call → return `should_continue_gathering=False`
3. Build system + user prompts for coverage assessment
4. Call `execute_llm_call()` with `role="supervision"`, `temperature=0.3`
5. Parse JSON response → coverage assessment + follow-up queries
6. Dedup follow-up queries against existing sub-queries (case-insensitive)
7. Cap new sub-queries at `max_sub_queries - len(state.sub_queries)` to prevent unbounded growth
8. Add new sub-queries via `state.add_sub_query()` with priority 2 (lower than originals)
9. Increment `state.supervision_round`
10. Record supervision history in `state.metadata["supervision_history"]`
11. Return WorkflowResult with `metadata["should_continue_gathering"]` flag

#### `_build_per_query_coverage(state) → list[dict]`
For each completed/failed sub-query: source count, quality distribution (HIGH/MEDIUM/LOW counts), unique domain count, findings summary from topic research results.

#### `_build_supervision_system_prompt(state) → str`
Instruct LLM to evaluate coverage and return JSON:
```json
{
  "overall_coverage": "sufficient|partial|insufficient",
  "per_query_assessment": [{"sub_query_id": "...", "coverage": "...", "rationale": "..."}],
  "follow_up_queries": [{"query": "...", "rationale": "...", "priority": 1}],
  "should_continue_gathering": true/false,
  "rationale": "..."
}
```
Rules:
- "sufficient" = 2+ quality sources from diverse domains
- Follow-ups must be MORE SPECIFIC than originals (drill down, not repeat)
- Max 3 follow-ups per round
- Do NOT generate queries duplicating existing sub-queries

#### `_build_supervision_user_prompt(state, coverage_data) → str`
Research query + brief + per-query coverage table + existing sub-queries (for dedup) + round N/max.

#### `_parse_supervision_response(content, state) → dict`
Extract JSON, validate schema, dedup follow-up queries against existing sub-queries (case-insensitive exact match).

#### `_assess_coverage_heuristic(state, min_sources) → dict`
Fallback when LLM fails: check if each query has `min_sources` sources. Returns `should_continue_gathering=False` (conservative — won't loop without LLM guidance).

### Edge Cases

- **Sub-query dedup**: Case-insensitive exact match in `_parse_supervision_response()` against all existing sub-queries
- **URL dedup in gathering**: Already handled — `seen_urls` initialized from `state.sources` at gathering start
- **Max sub-queries budget**: New sub-queries capped at `max_sub_queries - len(state.sub_queries)`
- **Empty follow-ups**: If supervision returns 0 follow-ups, `should_continue_gathering` → False
- **All rounds exhausted**: `supervision_round >= max_supervision_rounds` → advance to ANALYSIS

### Validation

- Test: `_build_per_query_coverage()` produces correct source counts and quality
- Test: `_parse_supervision_response()` with valid/invalid JSON
- Test: duplicate follow-up queries are stripped
- Test: heuristic fallback returns correct assessment

---

## Sub-Phase 6.5: Workflow Execution Integration

**Effort:** Medium | **Impact:** Wires supervisor into the main loop
**File:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py`

### Changes

1. **Add TYPE_CHECKING stub** (line 72):
   ```python
   async def _execute_supervision_async(self, *args: Any, **kwargs: Any) -> Any: ...
   ```

2. **Add SUPERVISION block** in the `while True` loop, after the GATHERING block and its extract/digest follow-ups (after line 334). Because `_safe_orchestrator_transition` after GATHERING calls `advance_phase()` → SUPERVISION, the new block activates naturally:

   ```python
   if state.phase == DeepResearchPhase.SUPERVISION:
       if not getattr(self.config, "deep_research_enable_supervision", True):
           state.advance_phase()  # Skip directly to ANALYSIS
       else:
           err = await self._run_phase(
               state,
               DeepResearchPhase.SUPERVISION,
               self._execute_supervision_async(
                   state=state,
                   provider_id=resolve_phase_provider(self.config, "supervision", "reflection"),
                   timeout=self.config.get_phase_timeout("supervision"),
               ),
               skip_transition=True,  # We handle transition manually
           )
           if err:
               return err
           await self._maybe_reflect(state, DeepResearchPhase.SUPERVISION)

           # Check if supervisor wants more gathering
           last_hist = (state.metadata.get("supervision_history") or [{}])[-1]
           should_gather = last_hist.get("should_continue_gathering", False)
           has_pending = len(state.pending_sub_queries()) > 0
           within_limit = state.supervision_round < state.max_supervision_rounds

           if should_gather and has_pending and within_limit:
               state.phase = DeepResearchPhase.GATHERING
               continue  # Inner loop: re-enter GATHERING
           else:
               state.advance_phase()  # SUPERVISION → ANALYSIS
   ```

3. **Cancellation safety**: Already handled — `iteration_in_progress` is True during GATHERING, `_run_phase` calls `_check_cancellation()` before each phase. No additional handling needed.

### Validation

- Integration test: GATHERING → SUPERVISION → GATHERING → SUPERVISION → ANALYSIS loop
- Integration test: supervision disabled → SUPERVISION phase auto-skipped
- Cancellation during supervision handled correctly

---

## Sub-Phase 6.6: Core Class Wiring

**Effort:** Low | **Impact:** Activates the mixin

### Changes

**File:** `src/foundry_mcp/core/research/workflows/deep_research/core.py`
1. Import `SupervisionPhaseMixin`
2. Add `SupervisionPhaseMixin` to `DeepResearchWorkflow` class inheritance (after `TopicResearchMixin`)

**File:** `src/foundry_mcp/core/research/workflows/deep_research/phases/__init__.py`
1. Add import and export for `SupervisionPhaseMixin`

### Validation

- `DeepResearchWorkflow` has `_execute_supervision_async` method

---

## Sub-Phase 6.7: Tests

**Effort:** Medium | **Impact:** Ensures correctness
**New file:** `tests/core/research/workflows/deep_research/test_supervision.py`

### Test Categories

1. **State model tests** (6 tests):
   - `test_supervision_phase_in_enum` — SUPERVISION between GATHERING and ANALYSIS
   - `test_advance_phase_gathering_to_supervision` — advance from GATHERING → SUPERVISION
   - `test_advance_phase_supervision_to_analysis` — advance from SUPERVISION → ANALYSIS
   - `test_start_new_iteration_resets_supervision_round` — resets to 0
   - `test_should_continue_supervision_within_limit` — True when pending + within limit
   - `test_should_continue_supervision_at_limit` — False when round >= max

2. **Prompt/coverage tests** (3 tests):
   - `test_build_per_query_coverage` — correct source counts, quality, domains
   - `test_supervision_system_prompt_has_json_schema` — prompt contains expected JSON structure
   - `test_supervision_user_prompt_includes_coverage` — prompt includes per-query data

3. **Response parsing tests** (4 tests):
   - `test_parse_valid_supervision_response` — valid JSON parsed correctly
   - `test_parse_with_duplicate_queries_deduped` — duplicate follow-ups stripped
   - `test_parse_invalid_json_returns_fallback` — graceful degradation
   - `test_heuristic_fallback` — correct assessment

4. **Integration tests** (4 tests):
   - `test_supervision_loop_adds_follow_up_queries` — new sub-queries in state
   - `test_supervision_loop_terminates_at_max_rounds` — max round cap respected
   - `test_supervision_skipped_when_disabled` — `enable_supervision=False` → skips
   - `test_supervision_proceeds_when_all_covered` — sufficient coverage → ANALYSIS

5. **Full regression**: `pytest tests/core/research/ -x` — confirm no regressions

---

## Sub-Phase 6.8: Update PLAN-CHECKLIST.md

Update the Phase 6 section in `PLAN-CHECKLIST.md` with the granular checklist items (6.1-6.7) replacing the deferred placeholders.

---

## Implementation Order

```
6.1 (State Model) ──┐
6.2 (Config)     ──┤── foundation, no interdependencies
6.3 (Orchestration)─┘
         │
         ▼
6.4 (Supervision Mixin) ── depends on 6.1, 6.2
         │
         ▼
6.6 (Core Wiring) ── depends on 6.4
         │
         ▼
6.5 (Workflow Execution) ── depends on 6.1, 6.3, 6.4, 6.6
         │
         ▼
6.7 (Tests) ── depends on all above
         │
         ▼
6.8 (Checklist Update) ── last
```

Sub-phases 6.1, 6.2, 6.3 can be implemented in parallel. Sub-phase 6.4 is the heaviest lift. Sub-phase 6.5 is the trickiest integration point.

---

## Files Modified

| File | Change |
|------|--------|
| `src/foundry_mcp/core/research/models/deep_research.py` | Add SUPERVISION enum, state fields, methods |
| `src/foundry_mcp/config/research.py` | Add supervision config, role chain, timeout |
| `src/foundry_mcp/core/research/workflows/deep_research/orchestration.py` | Add SUPERVISION to mappings |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/supervision.py` | **NEW** — SupervisionPhaseMixin |
| `src/foundry_mcp/core/research/workflows/deep_research/phases/__init__.py` | Export SupervisionPhaseMixin |
| `src/foundry_mcp/core/research/workflows/deep_research/core.py` | Add mixin to class |
| `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` | Add SUPERVISION block to loop |
| `tests/core/research/workflows/deep_research/test_supervision.py` | **NEW** — test suite |
| `PLAN-CHECKLIST.md` | Update Phase 6 items |
