# Deep Research Workflow Decomposition Plan

## 0. Progress Tracker

| Stage | Status | Commit | Notes |
|-------|--------|--------|-------|
| Pre-work: Baselines | DONE | — | 3993 passed, 12 skipped. Import: 1734μs self / 51818μs cumulative |
| Stage 1: Scaffolding | DONE | f8f5e99 | Package created, _monolith.py shim, test patch targets updated |
| Stage 2a: infrastructure.py | DONE | 0d41315 | Crash handler, active sessions, install_crash_handler() |
| Stage 2b: orchestration.py | DONE | d538f9e | AgentRole, AgentDecision, SupervisorHooks, SupervisorOrchestrator |
| Stage 2c: source_quality + _constants + _helpers | DONE | a61ccc4 | Domain assessment, budget constants, pure utilities |
| **CHECKPOINT 1** | **PASSED** | — | 3993 passed, 12 skipped — identical to baseline |
| Stage 3: _budgeting.py | DONE | 9ed8013 | 11 methods extracted as standalone functions, thin delegates remain |
| **CHECKPOINT 2** | **PASSED** | — | 56/56 budget+digest tests pass. Monolith: 6994→5692 lines |
| Stage 4a: PlanningPhaseMixin | DONE | dff4d3c | 4 methods → phases/planning.py (421 lines) |
| Stage 4b: GatheringPhaseMixin | DONE | f843f5e | 6 methods → phases/gathering.py (824 lines). __init__.py imports from canonical modules |
| Stage 4c: AnalysisPhaseMixin | DONE | — | 5 methods + digest pipeline → phases/analysis.py (1203 lines). Test patches updated to phases.analysis/budgeting |
| Stage 4d: SynthesisPhaseMixin | DONE | ef925fa | 5 methods → phases/synthesis.py (619 lines). Removed thin delegates for _fidelity_level_from_score, _allocate_synthesis_budget |
| Stage 4e: RefinementPhaseMixin | DONE | a55f8a6 | 5 methods → phases/refinement.py (660 lines). Removed ALL remaining thin delegates + unused imports. All 5 phases extracted |
| Stage 5: Remaining mixins | DONE | 00dc2a0 | 4 methods → background_tasks.py (268 lines), 5 methods → session_management.py (296 lines). Monolith: 2130→1639 lines |
| Stage 6: Finalize | DONE | — | Renamed _monolith→core.py, updated 7 TYPE_CHECKING imports, 18 contract tests added |
| **CHECKPOINT 3 (Final)** | **PASSED** | — | 2767 unit/tools passed, 1250 core/research passed (6 skipped — tiktoken). Decomposition complete |

### Final Package Layout
```
src/foundry_mcp/core/research/workflows/deep_research/
├── __init__.py            # Re-exports all public symbols
├── core.py                # 1639 lines — DeepResearchWorkflow class, orchestration loop, cross-cutting methods
├── _constants.py          # Budget allocation constants
├── _helpers.py            # extract_json, fidelity_level_from_score, truncate_at_boundary
├── _budgeting.py          # Budget allocation, validation, digest archives (~540 lines)
├── infrastructure.py      # Crash handler, active sessions, cleanup hooks
├── orchestration.py       # AgentRole, AgentDecision, SupervisorHooks, SupervisorOrchestrator
├── source_quality.py      # Domain assessment, title normalization
├── background_tasks.py    # BackgroundTaskMixin (268 lines)
├── session_management.py  # SessionManagementMixin (296 lines)
└── phases/
    ├── __init__.py        # Re-exports all phase mixins
    ├── planning.py        # PlanningPhaseMixin (421 lines)
    ├── gathering.py       # GatheringPhaseMixin (824 lines)
    ├── analysis.py        # AnalysisPhaseMixin (1203 lines)
    ├── synthesis.py       # SynthesisPhaseMixin (619 lines)
    └── refinement.py      # RefinementPhaseMixin (660 lines)
```

---

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

### Alternatives Considered

**Context-delegate pattern** — Each phase becomes a standalone class receiving a single
`WorkflowContext` object that bundles `config`, `memory`, `hooks`, `orchestrator`, and
cross-cutting methods (`_write_audit_event`, `_check_cancellation`, etc.):

```python
class WorkflowContext:
    config: ResearchConfig
    memory: ResearchMemory
    hooks: SupervisorHooks | None
    def write_audit_event(self, state, event_type, ...): ...
    def check_cancellation(self, research_id): ...
    async def execute_provider_async(self, prompt, **kw): ...

class PlanningPhase:
    async def execute(self, ctx: WorkflowContext, state, provider_id, timeout):
        ctx.write_audit_event(state, "planning_start")
        result = await ctx.execute_provider_async(...)
```

**Strengths**: Explicit dependencies, real testability (mock one object), full IDE/type-checker
support, no MRO concerns. **Weaknesses**: Requires rewriting the orchestration loop from
`self._execute_planning_async(state, ...)` to `self._planning_phase.execute(self._ctx, state, ...)`,
updating every test that patches `workflow._execute_*_async`, and refactoring ~35 `self._write_audit_event`
call sites to `self._ctx.write_audit_event`. The cross-cutting methods also need extraction from
`DeepResearchWorkflow` into the context object, which is a larger scope change.

**Chosen: Mixins** — for pragmatic reasons specific to this codebase:

1. **Zero orchestration changes** — `_execute_workflow_async` calls `self._execute_*_async()`.
   Mixins preserve this. Delegates require rewriting the orchestration loop.
2. **Zero test changes** (except 1 file) — Tests patch `workflow._execute_planning_async` etc.
   on the instance. Mixins preserve this. Delegates change the object graph.
3. **Incremental migration** — Each phase can be extracted in a single commit with a
   checkpoint. The monolith remains functional at every intermediate state.
4. **Scope control** — This is a file decomposition, not an architecture redesign. The delegate
   pattern is architecturally superior but 2-3x the scope. It's the right second step if the
   mixin decomposition proves stable.

The delegate pattern is documented as the fallback/evolution path in Section 11.

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
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import DeepResearchWorkflow

class PlanningPhaseMixin:
    """Planning phase methods. Mixed into DeepResearchWorkflow.

    At runtime, `self` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods on core.py)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    async def _execute_planning_async(
        self: DeepResearchWorkflow, state, provider_id, timeout
    ):
        # ... moved verbatim from monolith ...
        # self._write_audit_event() etc. resolves via MRO + type annotation

    def _build_planning_system_prompt(self: DeepResearchWorkflow, state): ...
    def _build_planning_user_prompt(self: DeepResearchWorkflow, state): ...
    def _parse_planning_response(self: DeepResearchWorkflow, content, state): ...
```

Each phase mixin follows this same pattern. Methods move verbatim; the only change is adding
`self: DeepResearchWorkflow` annotations for IDE/type-checker support (see Typing Strategy for Mixins (Section 2)).

### Typing Strategy for Mixins

Mixins have a well-known typing problem: `PlanningPhaseMixin._execute_planning_async` calls
`self._write_audit_event()`, but no type checker can verify that `self` provides that method
by looking at `PlanningPhaseMixin` alone.

**Solution**: Annotate `self` with the composed class type using `TYPE_CHECKING` lazy imports.

```python
# Every mixin file uses this pattern:
from __future__ import annotations      # PEP 563: all annotations are strings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import DeepResearchWorkflow  # only imported during type checking
```

This works because:
- `from __future__ import annotations` makes all type annotations lazy strings (already used in the monolith)
- The `TYPE_CHECKING` guard prevents circular imports at runtime (`core.py` imports `phases/planning.py`
  which would import `core.py` back)
- Pyright (configured in basic mode) resolves `self: DeepResearchWorkflow` and provides
  autocomplete + attribute access validation in mixin files
- At runtime, the annotation is never evaluated — zero import cost

**Convention**: Annotate `self: DeepResearchWorkflow` on every public-facing mixin method
(the `_execute_*_async` entry points). Private helper methods within the same mixin can omit
the annotation since Pyright infers from the calling context.

**Not needed**: A standalone `Protocol` class. The codebase has zero `Protocol` usage, Pyright is
in basic mode, and the `self` annotation approach provides the same IDE benefits with less ceremony.

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

### Budget/Validation Methods → `deep_research/_budgeting.py`

| Method | Used By | Destination |
|--------|---------|-------------|
| `_final_fit_validate` | Analysis, Synthesis, Refinement | `_budgeting.py` (standalone, takes all deps as params) |
| `_allocate_source_budget` | Analysis only | `_budgeting.py` |
| `_allocate_synthesis_budget` | Synthesis only | `_budgeting.py` |
| `_compute_refinement_budget` | Refinement only | `_budgeting.py` |
| `_summarize_report_for_refinement` | Refinement only | `_budgeting.py` |
| `_extract_report_summary` | Refinement only | `_budgeting.py` |
| `_archive_digest_source` | Analysis only | `_budgeting.py` |
| `_write_digest_archive` | Analysis only | `_budgeting.py` |
| `_validate_archive_source_id` | Analysis only | `_budgeting.py` |
| `_cleanup_digest_archives` | Analysis only | `_budgeting.py` |
| `_ensure_private_dir` | Analysis only | `_budgeting.py` |

~870 lines total. A single file is appropriate — a nested `budgeting/` package with 4 files
averaging ~220 lines each adds import indirection without meaningful separation of concerns.

Note: `_final_fit_validate` already takes all dependencies as parameters (one `self._truncate_at_boundary()`
call is resolved by importing the shared utility from `_helpers.py`), so it extracts cleanly as a standalone
function. The allocator methods access `self.config` and token management — they become standalone functions
taking config + state params, called from the respective phase mixins.

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
│   ├── _budgeting.py              # Budget allocation, validation, digest archives (~870 lines)
│   ├── core.py                    # DeepResearchWorkflow class (init, orchestration loop, public API, cross-cutting methods)
│   ├── orchestration.py           # SupervisorOrchestrator, SupervisorHooks, AgentRole, AgentDecision
│   ├── infrastructure.py          # Crash handler, _active_research_sessions, cleanup hooks
│   ├── source_quality.py          # Domain assessment, dedup, title normalization
│   ├── background_tasks.py        # BackgroundTaskMixin (task launch, retrieval, cleanup, GC)
│   ├── session_management.py      # SessionManagementMixin (list/delete/resume sessions)
│   └── phases/
│       ├── __init__.py            # Re-exports all phase mixins
│       ├── planning.py            # PlanningPhaseMixin (~400 lines)
│       ├── gathering.py           # GatheringPhaseMixin + follow-up extraction + provider config (~600 lines)
│       ├── analysis.py            # AnalysisPhaseMixin + digest pipeline (~1400 lines)
│       ├── synthesis.py           # SynthesisPhaseMixin (~530 lines)
│       └── refinement.py          # RefinementPhaseMixin (~590 lines)
```

**12 files** in the package (down from 17 in the original plan). Removed:
- `budgeting/` subdirectory (4 files + `__init__.py`) → collapsed into single `_budgeting.py`
- `persistence.py`, `audit.py`, `errors.py`, `config_builders.py`, `phases/base.py` — these were
  over-decomposed; the methods they'd contain are cross-cutting instance methods that stay on
  `DeepResearchWorkflow` in `core.py`, or are single-phase helpers that belong in their phase
  mixin (e.g., `config_builders` → `gathering.py`).

---

## 5. Dependency Graph

```
                         infrastructure.py  (crash handler, imported first)
                              ↑
                              |
core.py (DeepResearchWorkflow)
├── _constants.py             (budget constants)
├── _helpers.py               (pure utility functions)
├── _budgeting.py             (allocation, validation, archives → _constants, _helpers)
├── orchestration.py          (SupervisorOrchestrator, SupervisorHooks)
├── source_quality.py         (domain assessment — used by gathering)
├── background_tasks.py       (BackgroundTaskMixin)
├── session_management.py     (SessionManagementMixin)
└── phases/
    ├── planning.py           → _helpers
    ├── gathering.py          → _helpers, source_quality
    ├── analysis.py           → _helpers, _budgeting
    ├── synthesis.py          → _helpers, _budgeting
    └── refinement.py         → _helpers, _budgeting
```

**Key invariant**: All imports are unidirectional. No phase imports another phase.
`_budgeting.py` never imports a phase. `core.py` imports phases (for MRO); phases never
import `core.py` (except under `TYPE_CHECKING` for `self` annotations — see Typing Strategy for Mixins (Section 2)).

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

### Pre-work: Baseline Measurements

Before starting, capture baselines for regression detection:

```bash
# Import-time baseline (record the real number, not the 200ms budget)
python -X importtime -c "from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow" 2>&1 | head -20

# Pyright baseline (record current warning count)
pyright src/foundry_mcp/core/research/workflows/deep_research.py --outputjson | jq '.summary'

# Test suite baseline
pytest tests/unit/ tests/tools/ tests/core/research/ -q --tb=no 2>&1 | tail -3
```

### Branch Strategy

Work on a dedicated feature branch (`refactor/deep-research-decomposition`). The monolith
`deep_research.py` is touched by this branch only — coordinate with other contributors to
avoid concurrent edits during migration. Keep individual commits (don't squash) for
`git bisect` support. Rebase on `main` before each Stage to minimize drift.

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

### **CHECKPOINT 1**:
```bash
pytest tests/unit/ tests/tools/ tests/core/research/ -q        # all tests pass
pyright src/foundry_mcp/core/research/workflows/deep_research/  # no new errors vs. baseline
python -c "from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow"  # imports clean
```
If any fail, stop and reassess before proceeding.

### Stage 3: Extract Budgeting (1 commit)

**Commit 3**: Extract `_budgeting.py`
- `_allocate_source_budget`, `_allocate_synthesis_budget` (become standalone functions taking config + state params)
- `_compute_refinement_budget`, `_summarize_report_for_refinement`, `_extract_report_summary`
- `_final_fit_validate` (already takes all deps as params — clean extraction)
- `_archive_digest_source`, `_write_digest_archive`, `_validate_archive_source_id`, `_cleanup_digest_archives`, `_ensure_private_dir`

### **CHECKPOINT 2**:
```bash
pytest tests/core/research/test_deep_research_token_integration.py tests/core/research/test_deep_research_digest.py -v
pyright src/foundry_mcp/core/research/workflows/deep_research/  # no new errors
```

### Stage 4: Extract Phase Mixins (5 commits)

Extract one phase per commit, in dependency order. Each mixin file follows the typing
convention from Typing Strategy for Mixins (Section 2) (`from __future__ import annotations` + `TYPE_CHECKING` import).

**Commit 4a**: `phases/planning.py` → `PlanningPhaseMixin`
- `_execute_planning_async`, `_build_planning_system_prompt`, `_build_planning_user_prompt`, `_parse_planning_response`
- Import `_extract_json` from `_helpers`
- Add `self: DeepResearchWorkflow` annotations on entry-point methods

**Commit 4b**: `phases/gathering.py` → `GatheringPhaseMixin`
- `_execute_gathering_async`, `_execute_extract_followup_async`, `_get_search_provider`
- `_get_tavily_search_kwargs`, `_get_perplexity_search_kwargs`, `_get_semantic_scholar_search_kwargs`
- Nested `execute_sub_query()` moves with its parent — no changes needed
- Import `get_domain_quality`, `_normalize_title` from `source_quality`

**Commit 4c**: `phases/analysis.py` → `AnalysisPhaseMixin`
- `_execute_analysis_async`, `_execute_digest_step_async` (with nested closures), `_build_analysis_system_prompt`, `_build_analysis_user_prompt`, `_parse_analysis_response`
- Budget helpers called from `_budgeting`
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

### **CHECKPOINT 3 (Final)**:
```bash
pytest tests/unit/ tests/tools/ tests/core/research/ -q                    # all tests pass
pyright src/foundry_mcp/core/research/workflows/deep_research/ --outputjson  # compare to baseline
python -X importtime -c "from foundry_mcp.core.research.workflows.deep_research import DeepResearchWorkflow" 2>&1 | head -20  # compare to baseline
```

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

@pytest.mark.parametrize("module", [
    "foundry_mcp.core.research.workflows.deep_research.core",
    "foundry_mcp.core.research.workflows.deep_research.orchestration",
    "foundry_mcp.core.research.workflows.deep_research.infrastructure",
    "foundry_mcp.core.research.workflows.deep_research.source_quality",
    "foundry_mcp.core.research.workflows.deep_research._constants",
    "foundry_mcp.core.research.workflows.deep_research._helpers",
    "foundry_mcp.core.research.workflows.deep_research._budgeting",
    "foundry_mcp.core.research.workflows.deep_research.background_tasks",
    "foundry_mcp.core.research.workflows.deep_research.session_management",
    "foundry_mcp.core.research.workflows.deep_research.phases.planning",
    "foundry_mcp.core.research.workflows.deep_research.phases.gathering",
    "foundry_mcp.core.research.workflows.deep_research.phases.analysis",
    "foundry_mcp.core.research.workflows.deep_research.phases.synthesis",
    "foundry_mcp.core.research.workflows.deep_research.phases.refinement",
])
def test_submodule_importable(module):
    """Verify every submodule imports without errors."""
    import importlib
    importlib.import_module(module)
```

---

## 10. Risk Assessment

### Critical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Module initialization order** | Crash handler not installed if `infrastructure.py` imported late | Replace side-effectful `sys.excepthook =` with explicit `install_crash_handler()` guarded by a `once` flag, called from `__init__` |
| **Circular imports** | Import failure at runtime | Enforce unidirectional imports; add cycle detection test; phases import `core.py` only under `TYPE_CHECKING` |
| **Threading state** | `_active_research_sessions` accessed from multiple threads | Keep lock + dict in `infrastructure.py` as single source of truth |
| **Mixin MRO conflicts** | `__init__` called incorrectly or methods shadowed | Mixins define no `__init__`; method names are disjoint across mixins; add assertion test |
| **Backward compat drift** | Re-exports get out of sync with submodule exports | Explicit `__all__` + contract test; test runs on every CI build |
| **Patch path breakage** | `test_deep_research_digest.py` patches fail | Update 1 test file's patch targets; add `DocumentDigestor` etc. re-import in `__init__.py` as belt-and-suspenders |
| **Import-time regression** | Splitting into 12 files slows cold import | Measure before/after with `python -X importtime`; baseline captured in pre-work step |
| **Pyright regressions** | Mixin `self` access produces new warnings | `self: DeepResearchWorkflow` annotations via `TYPE_CHECKING` import (Typing Strategy for Mixins (Section 2)); compare pyright warning count to baseline at every checkpoint |
| **Untested code paths break** | Error-handling branches reference methods that didn't land in the right file | Run pyright at every checkpoint (catches attribute access errors statically); add smoke-import test for every submodule |

### Verification Checklist

- [ ] All public symbols importable via both old and new paths
- [ ] No circular import errors (automated test)
- [ ] Pyright warning count ≤ baseline (no new errors introduced)
- [ ] Crash handler fires on uncaught exceptions
- [ ] State persistence throttling works correctly
- [ ] Background task spawn/poll/cancel lifecycle works
- [ ] Audit JSONL events written correctly
- [ ] Budget validation passes/fails at correct thresholds
- [ ] All 11 existing test files pass (1 with updated patch targets)
- [ ] No increase in memory footprint
- [ ] Import time ≤ baseline + 50ms
- [ ] Mixin MRO provides all expected phase methods (automated test)
- [ ] Every submodule importable in isolation (`python -c "from ...phases.planning import PlanningPhaseMixin"`)

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
| **Checkpoint 2** (after Stage 3) | Budget tests fail, token integration tests fail | `git revert` the Stage 3 commit. Budgeting methods remain in `_monolith.py`. Single `_budgeting.py` file makes this a clean revert. |
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

If Checkpoint 1 passes but Checkpoint 3 fails due to mixin issues specifically (MRO conflicts,
`super()` chain problems, or testing framework incompatibilities), pivot to the **context-delegate
pattern** described in Section 2:

1. Keep the non-class extractions from Stages 1-3 (they're stable and mixin-independent)
2. Revert Stages 4-5 (phase mixin extraction)
3. Introduce a `WorkflowContext` class on `core.py` that bundles cross-cutting state and methods
4. Convert each phase mixin to a standalone class receiving `ctx: WorkflowContext`
5. Update the orchestration loop from `self._execute_planning_async(...)` to
   `self._planning.execute(self._ctx, ...)`

This is a larger scope change (~2x the remaining effort) but produces a cleaner architecture
with explicit dependencies and full type-checker support. It's the right evolution path if the
mixin decomposition proves stable and a future refactoring pass is warranted.

---

## 12. Estimated Effort

Two estimates are provided: AI-assisted (with Claude Code or similar) and manual.

| Stage | Work | AI-Assisted | Manual |
|-------|------|-------------|--------|
| Pre-work | Baseline measurements, branch setup | 15 min | 30 min |
| Stage 1: Scaffolding | Create package, rename monolith, verify | 30 min | 1-2 hours |
| Stage 2: Non-class extraction | 3 commits (infrastructure, orchestration, utilities) | 1-2 hours | 3-4 hours |
| Stage 3: Budgeting extraction | 1 commit (`_budgeting.py`) | 30-60 min | 2-3 hours |
| Stage 4: Phase mixin extraction | 5 commits (one per phase) + typing annotations | 3-5 hours | 8-10 hours |
| Stage 5: Remaining mixins | 1 commit (background tasks, sessions) | 30-60 min | 2-3 hours |
| Stage 6: Finalize | Rename, cleanup, contract tests | 1-2 hours | 2-3 hours |
| Test migration | Update `test_deep_research_digest.py` patch targets | 30 min | 1-2 hours |
| Checkpoint validation | 3 checkpoints × (tests + pyright + import benchmark) | 1-2 hours | 2-3 hours |
| **Total** | | **8-14 hours** | **21-30 hours** |

The bulk of the work (Stage 4) is mechanical: move methods verbatim, add imports, add `self`
type annotations, verify. This is well-suited to AI-assisted extraction. The real time cost
is checkpoint validation — tests and pyright must pass at each stage regardless of who writes
the code.
