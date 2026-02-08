# Deep Research Workflow Decomposition Plan

## 1. Current Structure

### File Metadata
- **Path**: `src/foundry_mcp/core/research/workflows/deep_research.py`
- **Size**: 6,994 lines
- **Classes**: 5
- **Module-level functions**: 8
- **Class methods**: 87

### Top-Level Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_persist_active_sessions()` | 133-158 | Crash recovery: persists active research state |
| `_crash_handler()` | 161-207 | Uncaught exception handler with session persistence |
| `_cleanup_on_exit()` | 215-224 | atexit handler for graceful shutdown |
| `_extract_domain()` | 232-256 | URL domain extraction utility |
| `_extract_hostname()` | 259-278 | Full hostname extraction (preserves subdomains) |
| `_domain_matches_pattern()` | 281-314 | Wildcard domain pattern matching |
| `get_domain_quality()` | 317-344 | Source quality assessment by domain tier |
| `_normalize_title()` | 347-365 | Title normalization for deduplication |

### Classes

| Class | Lines | Purpose |
|-------|-------|---------|
| `AgentRole` (Enum) | 373-407 | Specialist agent roles in multi-agent orchestration |
| `AgentDecision` (Dataclass) | 411-444 | Record decisions for traceability |
| `SupervisorHooks` | 452-523 | Callback hooks for workflow event injection |
| `SupervisorOrchestrator` | 531-855 | Agent dispatch, phase evaluation, iteration decisions |
| `DeepResearchWorkflow` | 863-6994 | Main workflow engine (60+ methods) |

### DeepResearchWorkflow Method Categories

| Category | Approx Lines | Description |
|----------|-------------|-------------|
| Initialization & Configuration | ~45 | `__init__`, audit config, persistence sync |
| State Persistence | ~150 | Throttled persistence, terminal state detection |
| Audit & Observability | ~100 | JSONL audit events, payload preparation |
| Search Provider Configuration | ~140 | Per-provider parameter builders |
| Error Handling & Recovery | ~40 | Error recording, orchestrator error wrapping |
| Core Workflow API | ~500 | `execute()`, start/continue/status/report/cancel |
| Background Task Management | ~60 | Task launch, retrieval, cleanup, GC |
| Session Management | ~200 | List/delete/resume sessions |
| Phase Implementations | ~4000 | Planning, gathering, analysis, synthesis, refinement |
| Budget Management | ~600 | Source/synthesis/refinement allocation, validation |
| Digest & Archive Management | ~150 | Source digestion, archive writing, cleanup |

---

## 2. Phase Extraction Pattern: Mixins

### Why Mixins (Not Delegates)

The orchestration loop in `_execute_workflow_async` calls phases as `self._execute_*_async()`.
Every phase method accesses cross-cutting `self.*` state: `self.config`, `self.memory`,
`self.hooks`, `self.orchestrator`, `self._search_providers`, and inherited methods like
`self._execute_provider_async()`. A delegate pattern would require either passing 6+ objects
to each delegate or defining a facade interface — both add indirection without improving
testability since tests already patch methods directly on the workflow instance.

Mixins let us split the code across files while preserving `self.*` access with zero wiring
changes. The orchestration loop and all tests remain identical. Each mixin is a plain class
that contributes methods; `DeepResearchWorkflow` inherits from all of them plus
`ResearchWorkflowBase`.

### Concrete Design

```python
# deep_research/core.py
from .phases.planning import PlanningPhaseMixin
from .phases.gathering import GatheringPhaseMixin
from .phases.analysis import AnalysisPhaseMixin
from .phases.synthesis import SynthesisPhaseMixin
from .phases.refinement import RefinementPhaseMixin
from .session_management import SessionManagementMixin
from .background_tasks import BackgroundTaskMixin

class DeepResearchWorkflow(
    PlanningPhaseMixin,
    GatheringPhaseMixin,
    AnalysisPhaseMixin,
    SynthesisPhaseMixin,
    RefinementPhaseMixin,
    SessionManagementMixin,
    BackgroundTaskMixin,
    ResearchWorkflowBase,
):
    """Multi-phase deep research workflow with background execution."""

    def __init__(self, config, memory=None, hooks=None):
        super().__init__(config, memory)
        # ... existing init code unchanged ...

    # _execute_workflow_async stays here (orchestration loop)
    # execute(), _start_research(), _continue_research(),
    # _get_status(), _get_report(), _cancel_research() stay here
```

```python
# deep_research/phases/planning.py
class PlanningPhaseMixin:
    """Planning phase methods. Mixed into DeepResearchWorkflow."""

    async def _execute_planning_async(self, state, provider_id, timeout):
        # ... moved verbatim from monolith ...
        # self._write_audit_event() etc. still work via MRO

    def _build_planning_system_prompt(self, state): ...
    def _build_planning_user_prompt(self, state): ...
    def _parse_planning_response(self, content, state): ...
```

Each phase mixin follows this same pattern. Methods move verbatim; no signature changes.

### Why This Doesn't Create Diamond Inheritance Problems

The mixins contribute disjoint method sets (no overlapping method names). They don't define
`__init__`. Only `DeepResearchWorkflow` and `ResearchWorkflowBase` have `__init__`, and
Python's MRO handles this cleanly since `ResearchWorkflowBase` is listed last.

---

## 3. Shared Utilities Inventory

Methods used by 2+ phases that must live in a shared location (not inside any single phase mixin).

### Pure Utilities → `deep_research/_helpers.py`

These access no `self.*` state — they're stateless functions:

| Method | Used By | Destination |
|--------|---------|-------------|
| `_extract_json` | Planning, Analysis, Refinement | `_helpers.py` (standalone function) |
| `_fidelity_level_from_score` | Analysis, Synthesis | `_helpers.py` (standalone function) |
| `_truncate_at_boundary` | Analysis, Synthesis, Refinement (via `_final_fit_validate`) | `_helpers.py` (standalone function) |

These become module-level functions. Callers change from `self._extract_json(x)` to
`_extract_json(x)` (imported at top of each phase mixin file).

### Cross-Cutting Instance Methods → Stay on `DeepResearchWorkflow` in `core.py`

These access instance state (`self.config`, `self.memory`, `self._last_persisted_*`) and
are called pervasively. Moving them to a separate mixin gains nothing — they'd still be
accessed via `self.*`:

| Method | Call Sites | `self.*` Accessed |
|--------|-----------|-------------------|
| `_write_audit_event` | ~35 sites across all phases + orchestration | `self.config`, `self.memory.base_path` |
| `_persist_state_if_needed` | Orchestration loop (all phases) | `self._last_persisted_*`, `self.config`, `self.memory` |
| `_persist_state` | Via `_persist_state_if_needed`, `_flush_state` | `self.memory`, `self._last_persisted_*` |
| `_flush_state` | Orchestration loop (all phases on failure/completion) | Via `_persist_state` |
| `_check_cancellation` | All 5 phases + orchestration loop | `self._tasks`, `self._tasks_lock` |
| `_record_workflow_error` | Exception handlers in orchestration + background tasks | `state` param only |
| `_safe_orchestrator_transition` | Orchestration loop (after Planning, Gathering, Analysis) | `self.orchestrator`, `self.hooks` |

Supporting methods that stay with their parent:
- `_audit_enabled`, `_audit_path`, `_prepare_audit_payload` → stay with `_write_audit_event`
- `_should_persist_status`, `_is_terminal_state`, `_sync_persistence_tracking_from_state` → stay with persistence
- `_cleanup_completed_task` → stays with background task management

### Budget/Validation Methods → `deep_research/budgeting/`

| Method | Used By | Destination |
|--------|---------|-------------|
| `_final_fit_validate` | Analysis, Synthesis, Refinement | `budgeting/validation.py` (standalone, takes all deps as params) |
| `_allocate_source_budget` | Analysis only | `budgeting/allocators.py` |
| `_allocate_synthesis_budget` | Synthesis only | `budgeting/allocators.py` |
| `_compute_refinement_budget` | Refinement only | `budgeting/refinement.py` |
| `_summarize_report_for_refinement` | Refinement only | `budgeting/refinement.py` |
| `_extract_report_summary` | Refinement only | `budgeting/refinement.py` |

Note: `_final_fit_validate` already takes all dependencies as parameters (no `self.*` access),
so it extracts cleanly as a standalone function. The allocator methods access `self.config` and
token management — they become methods on the respective phase mixins that call shared budget
helpers.

---

## 4. Proposed Package Layout

```
src/foundry_mcp/core/research/workflows/
├── __init__.py                    # (existing, unchanged)
├── base.py                        # (existing, unchanged)
├── deep_research/
│   ├── __init__.py                # Public API re-exports (backward compat)
│   ├── _constants.py              # Budget allocation constants (ANALYSIS_PHASE_BUDGET_FRACTION, etc.)
│   ├── _helpers.py                # Shared pure utilities (_extract_json, _fidelity_level_from_score, _truncate_at_boundary)
│   ├── core.py                    # DeepResearchWorkflow class (init, orchestration loop, public API, cross-cutting methods)
│   ├── orchestration.py           # SupervisorOrchestrator, SupervisorHooks, AgentRole, AgentDecision
│   ├── infrastructure.py          # Crash handler, _active_research_sessions, cleanup hooks
│   ├── source_quality.py          # Domain assessment, dedup, title normalization
│   ├── background_tasks.py        # BackgroundTaskMixin (task launch, retrieval, cleanup, GC)
│   ├── session_management.py      # SessionManagementMixin (list/delete/resume sessions)
│   ├── phases/
│   │   ├── __init__.py            # Re-exports all phase mixins
│   │   ├── planning.py            # PlanningPhaseMixin (~400 lines)
│   │   ├── gathering.py           # GatheringPhaseMixin + follow-up extraction + provider config (~600 lines)
│   │   ├── analysis.py            # AnalysisPhaseMixin + digest pipeline (~1400 lines)
│   │   ├── synthesis.py           # SynthesisPhaseMixin (~530 lines)
│   │   └── refinement.py          # RefinementPhaseMixin (~590 lines)
│   └── budgeting/
│       ├── __init__.py            # Re-exports
│       ├── allocators.py          # Source and synthesis budget allocation (~400 lines)
│       ├── refinement.py          # Refinement budget computation (~200 lines)
│       ├── validation.py          # final_fit_validate, truncate_at_boundary (~150 lines)
│       └── archives.py            # Digest archive management (~120 lines)
```

**Removed from original plan**: `persistence.py`, `audit.py`, `errors.py`, `config_builders.py`,
`phases/base.py`. These were over-decomposed — the methods they'd contain are cross-cutting
instance methods that stay on `DeepResearchWorkflow` in `core.py`, or are single-phase helpers
that belong in their phase mixin (e.g., `config_builders` → `gathering.py`).

---

## 5. Dependency Graph

```
                         infrastructure.py  (crash handler, imported first)
                              ↑
                              |
core.py (DeepResearchWorkflow)
├── _constants.py             (budget constants)
├── _helpers.py               (pure utility functions)
├── orchestration.py          (SupervisorOrchestrator, SupervisorHooks)
├── source_quality.py         (domain assessment — used by gathering)
├── background_tasks.py       (BackgroundTaskMixin)
├── session_management.py     (SessionManagementMixin)
└── phases/
    ├── planning.py           → _helpers
    ├── gathering.py          → _helpers, source_quality
    ├── analysis.py           → _helpers, budgeting/allocators, budgeting/archives
    ├── synthesis.py          → _helpers, budgeting/allocators
    └── refinement.py         → _helpers, budgeting/refinement, budgeting/validation
                                   ↑
                              budgeting/
                              ├── allocators.py  → _constants
                              ├── refinement.py  → _constants
                              ├── validation.py  → _constants
                              └── archives.py
```

**Key invariant**: All imports are unidirectional. No phase imports another phase.
No budgeting module imports a phase. `core.py` imports phases (for MRO); phases never import `core.py`.

---

## 6. Nested Async Closures

Three methods contain nested async functions that capture enclosing scope variables:

| Method | Nested Function | Captures |
|--------|----------------|----------|
| `_execute_gathering_async` (line 3173) | `execute_sub_query()` (line 3329) | `state`, `provider`, `search_kwargs`, `resilience_mgr`, local counters |
| `_execute_digest_step_async` (line 4190) | `_digest_source()` (line 4399) | `state`, `digestor`, `summarizer`, `pdf_extractor`, config values |
| `_execute_digest_step_async` (line 4190) | `_tracked_digest_source()` (line 4644) | `state`, wraps `_digest_source` with tracking |

**Strategy**: These closures move with their enclosing method into the respective phase mixin.
The closures stay as nested functions — no change. The enclosing method accesses `self.*` for
the captured variables, which works identically via the mixin MRO.

No special handling required. This is a non-issue with the mixin approach.

---

## 7. Test File Impact Analysis

### Test Files and Their Dependencies

| Test File | Imports from deep_research | Patches on Internals | Impact |
|-----------|---------------------------|---------------------|--------|
| `tests/core/research/workflows/test_deep_research.py` | `DeepResearchWorkflow` | `workflow._execute_{planning,gathering,analysis,synthesis,refinement}_async`, `workflow._get_search_provider`, `workflow._check_cancellation`, `workflow._execute_provider_async` | **None** — patches target instance methods, which work regardless of which file defines them |
| `tests/core/research/workflows/test_timeout_resilience.py` | `DeepResearchWorkflow` | None (inspects `_execute_provider_async` signature) | **None** |
| `tests/core/research/test_deep_research_digest.py` | `DeepResearchWorkflow` | Patches `DocumentDigestor`, `ContentSummarizer`, `PDFExtractor`, `ContextBudgetManager` at module path | **BREAKS** — patches use `foundry_mcp.core.research.workflows.deep_research.DocumentDigestor` but after extraction these imports live in `phases/analysis.py` |
| `tests/core/research/test_deep_research_token_integration.py` | `DeepResearchWorkflow`, `ANALYSIS_PHASE_BUDGET_FRACTION`, `ANALYSIS_OUTPUT_RESERVED`, `SYNTHESIS_PHASE_BUDGET_FRACTION`, `SYNTHESIS_OUTPUT_RESERVED`, `REFINEMENT_PHASE_BUDGET_FRACTION`, `REFINEMENT_OUTPUT_RESERVED` | None | **None** — constants re-exported via `__init__.py` |
| `tests/integration/test_deep_research_resilience.py` | `DeepResearchWorkflow` | `workflow.get_background_task`, `workflow.memory.*` | **None** — instance method patches |
| `tests/integration/test_deep_research_tavily.py` | `DeepResearchWorkflow` | `workflow._get_tavily_search_kwargs` | **None** — instance method patch |
| `tests/unit/test_core/research/test_workflows.py` | `DeepResearchWorkflow`, `_active_sessions_lock`, `_active_research_sessions`, `AgentDecision`, `AgentRole` | None | **None** — all re-exported via `__init__.py` |
| `tests/unit/test_core/research/test_heartbeat_timing.py` | `DeepResearchWorkflow` | `workflow._execute_provider_async`, `workflow._check_cancellation`, `workflow._get_search_provider` | **None** — instance method patches |
| `tests/unit/test_core/research/test_partial_results.py` | `DeepResearchWorkflow` (via patch) | Class-level patch | **None** |
| `tests/unit/test_core/research/test_status_timeout_metadata.py` | `DeepResearchWorkflow` | `workflow.get_background_task`, `workflow.memory.*` | **None** |
| `tests/tools/unified/test_research.py` | None (patches via `tools.unified.research`) | `foundry_mcp.tools.unified.research.DeepResearchWorkflow` | **None** — patches at consumer site |

### Summary

**Only 1 test file requires changes**: `test_deep_research_digest.py`.

It patches classes at the monolith's module path (`foundry_mcp.core.research.workflows.deep_research.DocumentDigestor`).
After extraction, those imports live in `phases/analysis.py`. Fix: update the patch targets to
the new module path, **or** re-import `DocumentDigestor` etc. in `deep_research/__init__.py`
so the old patch path still resolves.

All other test files either:
- Patch instance methods on `workflow.*` (works regardless of which file defines the method)
- Import public symbols that will be re-exported via `__init__.py`

---

## 8. Migration Strategy

### Stage 1: Scaffolding (1 commit)

1. Create `deep_research/` directory
2. Create `__init__.py` that re-exports everything from the monolith:
   ```python
   # Backward-compat shim — re-export all public symbols
   from foundry_mcp.core.research.workflows.deep_research._monolith import *  # noqa: F401,F403
   from foundry_mcp.core.research.workflows.deep_research._monolith import (
       _active_research_sessions,
       _active_sessions_lock,
       _crash_handler,
       _cleanup_on_exit,
   )
   ```
3. Rename `deep_research.py` → `deep_research/_monolith.py`
4. Run full test suite — must pass with zero changes

### Stage 2: Extract Non-Class Code (3 commits)

**Commit 2a**: Extract `infrastructure.py`
- `_persist_active_sessions`, `_crash_handler`, `_cleanup_on_exit`
- `_active_research_sessions`, `_active_sessions_lock`, `_active_research_memory`
- Replace `sys.excepthook` assignment with explicit `install_crash_handler()` function
- Call `install_crash_handler()` from `DeepResearchWorkflow.__init__` with a `once` guard
- Update `__init__.py` re-exports

**Commit 2b**: Extract `orchestration.py`
- `AgentRole`, `PHASE_TO_AGENT`, `AgentDecision`, `SupervisorHooks`, `SupervisorOrchestrator`
- These are self-contained classes with no dependency on `DeepResearchWorkflow`
- Update `__init__.py` re-exports

**Commit 2c**: Extract `source_quality.py` + `_constants.py` + `_helpers.py`
- Module-level functions: `_extract_domain`, `_extract_hostname`, `_domain_matches_pattern`, `get_domain_quality`, `_normalize_title`
- Constants: `ANALYSIS_PHASE_BUDGET_FRACTION`, etc.
- Pure utilities: `_extract_json`, `_fidelity_level_from_score`, `_truncate_at_boundary` (extracted from class as standalone functions)
- Update `__init__.py` re-exports (constants needed for `test_deep_research_token_integration.py`)

### **CHECKPOINT 1**: Run full test suite. All tests must pass. If not, stop and reassess.

### Stage 3: Extract Budgeting Package (1 commit)

**Commit 3**: Extract `budgeting/`
- `allocators.py`: `_allocate_source_budget`, `_allocate_synthesis_budget` (become standalone functions taking config + state params)
- `refinement.py`: `_compute_refinement_budget`, `_summarize_report_for_refinement`, `_extract_report_summary`
- `validation.py`: `_final_fit_validate` (already takes all deps as params — clean extraction)
- `archives.py`: `_archive_digest_source`, `_write_digest_archive`, `_validate_archive_source_id`, `_cleanup_digest_archives`, `_ensure_private_dir`

### **CHECKPOINT 2**: Run full test suite. Especially `test_deep_research_token_integration.py` and `test_deep_research_digest.py`.

### Stage 4: Extract Phase Mixins (5 commits)

Extract one phase per commit, in dependency order:

**Commit 4a**: `phases/planning.py` → `PlanningPhaseMixin`
- `_execute_planning_async`, `_build_planning_system_prompt`, `_build_planning_user_prompt`, `_parse_planning_response`
- Import `_extract_json` from `_helpers`

**Commit 4b**: `phases/gathering.py` → `GatheringPhaseMixin`
- `_execute_gathering_async`, `_execute_extract_followup_async`, `_get_search_provider`
- `_get_tavily_search_kwargs`, `_get_perplexity_search_kwargs`, `_get_semantic_scholar_search_kwargs`
- Nested `execute_sub_query()` moves with its parent — no changes needed
- Import `get_domain_quality`, `_normalize_title` from `source_quality`

**Commit 4c**: `phases/analysis.py` → `AnalysisPhaseMixin`
- `_execute_analysis_async`, `_execute_digest_step_async` (with nested closures), `_build_analysis_system_prompt`, `_build_analysis_user_prompt`, `_parse_analysis_response`
- Budget helpers called from `budgeting/allocators` and `budgeting/archives`
- **Fix `test_deep_research_digest.py`**: Update patch targets from `...deep_research.DocumentDigestor` to `...deep_research.phases.analysis.DocumentDigestor` (or add re-imports in `__init__.py`)

**Commit 4d**: `phases/synthesis.py` → `SynthesisPhaseMixin`
- `_execute_synthesis_async`, `_build_synthesis_system_prompt`, `_build_synthesis_user_prompt`, `_extract_markdown_report`, `_generate_empty_report`

**Commit 4e**: `phases/refinement.py` → `RefinementPhaseMixin`
- `_execute_refinement_async`, `_build_refinement_system_prompt`, `_build_refinement_user_prompt`, `_parse_refinement_response`, `_extract_fallback_queries`

### Stage 5: Extract Remaining Mixins (1 commit)

**Commit 5**: `background_tasks.py` → `BackgroundTaskMixin`, `session_management.py` → `SessionManagementMixin`
- `_start_background_task`, `get_background_task`, `_cleanup_completed_task`, `cleanup_stale_tasks`
- `list_sessions`, `delete_session`, `resume_research`, `_validate_state_for_resume`, `list_resumable_sessions`

### Stage 6: Finalize (1 commit)

**Commit 6**: Rename `_monolith.py` → `core.py`, clean up
- `core.py` now contains: `DeepResearchWorkflow` class with `__init__`, `execute()`, `_execute_workflow_async`, action handlers (`_start_research`, `_continue_research`, `_get_status`, `_get_report`, `_cancel_research`), and cross-cutting methods (audit, persistence, cancellation, error recording)
- Remove `_monolith.py`
- Verify `__init__.py` re-exports are complete
- Add public API contract test
- Add circular import detection test

### **CHECKPOINT 3 (Final)**: Run full test suite. Run import-time benchmark. Verify no regressions.

---

## 9. Backward Compatibility

### Re-export Contract

`deep_research/__init__.py` must re-export all symbols that are currently importable from
`foundry_mcp.core.research.workflows.deep_research`:

```python
# deep_research/__init__.py
from .core import DeepResearchWorkflow
from .orchestration import AgentRole, AgentDecision, SupervisorHooks, SupervisorOrchestrator
from .infrastructure import (
    _active_research_sessions,
    _active_sessions_lock,
    _crash_handler,
    _cleanup_on_exit,
)
from .source_quality import get_domain_quality
from ._constants import (
    ANALYSIS_PHASE_BUDGET_FRACTION,
    ANALYSIS_OUTPUT_RESERVED,
    SYNTHESIS_PHASE_BUDGET_FRACTION,
    SYNTHESIS_OUTPUT_RESERVED,
    REFINEMENT_PHASE_BUDGET_FRACTION,
    REFINEMENT_OUTPUT_RESERVED,
    REFINEMENT_REPORT_BUDGET_FRACTION,
    FINAL_FIT_MAX_ITERATIONS,
    FINAL_FIT_COMPRESSION_FACTOR,
    FINAL_FIT_SAFETY_MARGIN,
)

__all__ = [
    "DeepResearchWorkflow",
    "AgentRole",
    "AgentDecision",
    "SupervisorHooks",
    "SupervisorOrchestrator",
    "get_domain_quality",
    # Constants
    "ANALYSIS_PHASE_BUDGET_FRACTION",
    "ANALYSIS_OUTPUT_RESERVED",
    "SYNTHESIS_PHASE_BUDGET_FRACTION",
    "SYNTHESIS_OUTPUT_RESERVED",
    "REFINEMENT_PHASE_BUDGET_FRACTION",
    "REFINEMENT_OUTPUT_RESERVED",
    "REFINEMENT_REPORT_BUDGET_FRACTION",
    "FINAL_FIT_MAX_ITERATIONS",
    "FINAL_FIT_COMPRESSION_FACTOR",
    "FINAL_FIT_SAFETY_MARGIN",
]
```

### Contract Test

```python
# tests/unit/test_core/research/test_deep_research_public_api.py

def test_all_public_symbols_importable_from_original_path():
    """Verify backward-compat re-exports from the original import path."""
    from foundry_mcp.core.research.workflows.deep_research import (
        DeepResearchWorkflow,
        SupervisorOrchestrator,
        SupervisorHooks,
        AgentRole,
        AgentDecision,
        get_domain_quality,
        ANALYSIS_PHASE_BUDGET_FRACTION,
        ANALYSIS_OUTPUT_RESERVED,
        SYNTHESIS_PHASE_BUDGET_FRACTION,
        SYNTHESIS_OUTPUT_RESERVED,
        REFINEMENT_PHASE_BUDGET_FRACTION,
        REFINEMENT_OUTPUT_RESERVED,
        _active_research_sessions,
        _active_sessions_lock,
    )

def test_no_circular_imports():
    """Verify no circular import errors in the deep_research package."""
    import importlib
    import foundry_mcp.core.research.workflows.deep_research
    importlib.reload(foundry_mcp.core.research.workflows.deep_research)

def test_workflow_inherits_all_phase_methods():
    """Verify the mixin MRO provides all expected phase methods."""
    from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow
    expected_methods = [
        "_execute_planning_async",
        "_execute_gathering_async",
        "_execute_analysis_async",
        "_execute_synthesis_async",
        "_execute_refinement_async",
        "list_sessions",
        "delete_session",
        "resume_research",
    ]
    for method in expected_methods:
        assert hasattr(DeepResearchWorkflow, method), f"Missing method: {method}"
```

---

## 10. Risk Assessment

### Critical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Module initialization order** | Crash handler not installed if `infrastructure.py` imported late | Replace side-effectful `sys.excepthook =` with explicit `install_crash_handler()` guarded by a `once` flag, called from `__init__` |
| **Circular imports** | Import failure at runtime | Enforce unidirectional imports; add cycle detection test; phases never import `core.py` |
| **Threading state** | `_active_research_sessions` accessed from multiple threads | Keep lock + dict in `infrastructure.py` as single source of truth |
| **Mixin MRO conflicts** | `__init__` called incorrectly or methods shadowed | Mixins define no `__init__`; method names are disjoint across mixins; add assertion test |
| **Backward compat drift** | Re-exports get out of sync with submodule exports | Explicit `__all__` + contract test; test runs on every CI build |
| **Patch path breakage** | `test_deep_research_digest.py` patches fail | Update 1 test file's patch targets; add `DocumentDigestor` etc. re-import in `__init__.py` as belt-and-suspenders |
| **Import-time regression** | Splitting into 15+ files slows cold import | Measure before/after with `python -X importtime`; set a 200ms budget for the package |

### Verification Checklist

- [ ] All public symbols importable via both old and new paths
- [ ] No circular import errors (automated test)
- [ ] Crash handler fires on uncaught exceptions
- [ ] State persistence throttling works correctly
- [ ] Background task spawn/poll/cancel lifecycle works
- [ ] Audit JSONL events written correctly
- [ ] Budget validation passes/fails at correct thresholds
- [ ] All 11 existing test files pass (1 with updated patch targets)
- [ ] No increase in memory footprint
- [ ] Import time within 200ms budget
- [ ] Mixin MRO provides all expected phase methods (automated test)

---

## 11. Rollback Plan

### Per-Commit Rollback

Each commit in the migration is independently revertable. The `_monolith.py` shim pattern
ensures that at any point between Stage 1 and Stage 6, the codebase is in a working state:

- **Stages 1-5**: `_monolith.py` exists and `__init__.py` re-exports from both `_monolith` and extracted modules. Reverting any single commit restores the previous working state.
- **Stage 6**: Removes `_monolith.py`. This is the only commit that can't be reverted in isolation — it requires reverting the full Stage 6 commit.

### Checkpoint-Level Rollback

If a checkpoint fails:

| Checkpoint | Failure Mode | Rollback Action |
|-----------|-------------|-----------------|
| **Checkpoint 1** (after Stage 2) | Import errors, crash handler not firing, test failures | `git revert` the Stage 2 commits. The `_monolith.py` + `__init__.py` shim from Stage 1 still works. |
| **Checkpoint 2** (after Stage 3) | Budget tests fail, token integration tests fail | `git revert` the Stage 3 commit. Budgeting methods remain in `_monolith.py`. |
| **Checkpoint 3** (after Stage 6) | Mixin MRO issues, performance regression, unforeseen test failures | Revert Stages 4-6. The non-class extractions (Stages 1-3) are stable and can remain. |

### Full Rollback

If the entire approach proves unworkable (e.g., mixin pattern causes unforeseen issues with
`super()` chains or testing frameworks):

```bash
git revert --no-commit HEAD~N..HEAD  # revert all decomposition commits
git commit -m "revert: deep_research decomposition — <reason>"
```

The original `deep_research.py` monolith is restored. No other files in the repo depend on
the internal package structure — only the re-export path in `workflows/__init__.py` matters.

### Decision Point: Abandon vs. Redesign

If Checkpoint 1 passes but Checkpoint 3 fails due to mixin issues specifically, consider
pivoting to a **"file-split-only" approach**: keep all methods on `DeepResearchWorkflow` but
split the class definition across files using a simpler pattern:

```python
# core.py defines the class
class DeepResearchWorkflow(ResearchWorkflowBase): ...

# phases/planning.py adds methods to it
from .core import DeepResearchWorkflow

async def _execute_planning_async(self, ...): ...
DeepResearchWorkflow._execute_planning_async = _execute_planning_async
```

This is uglier but has zero MRO risk. It's a fallback, not the primary plan.

---

## 12. Estimated Effort

| Stage | Work | Estimate |
|-------|------|----------|
| Stage 1: Scaffolding | Create package, rename monolith, verify | 1-2 hours |
| Stage 2: Non-class extraction | 3 commits (infrastructure, orchestration, utilities) | 3-4 hours |
| Stage 3: Budgeting extraction | 1 commit | 2-3 hours |
| Stage 4: Phase mixin extraction | 5 commits (one per phase) | 8-10 hours |
| Stage 5: Remaining mixins | 1 commit (background tasks, sessions) | 2-3 hours |
| Stage 6: Finalize | Rename, cleanup, contract tests, import benchmark | 2-3 hours |
| Test migration | Update `test_deep_research_digest.py` patch targets | 1-2 hours |
| Integration validation | End-to-end workflow test, performance check | 2-3 hours |
| **Total** | | **21-30 hours** |

The previous estimate of 13-19 hours was optimistic. The additional effort comes from:
- Mixin wiring and MRO validation
- Converting `self._extract_json` etc. to standalone functions across all call sites
- Budget method signature changes (adding config/state params)
- Careful checkpoint validation at each stage
