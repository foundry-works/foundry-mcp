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

## 2. Proposed Package Layout

```
src/foundry_mcp/core/research/workflows/
├── __init__.py                    # (existing, unchanged)
├── base.py                        # (existing, unchanged)
├── deep_research/
│   ├── __init__.py                # Public API re-exports (backward compat)
│   ├── core.py                    # DeepResearchWorkflow main class + _execute_workflow_async
│   ├── orchestration.py           # SupervisorOrchestrator, SupervisorHooks, AgentRole, AgentDecision
│   ├── infrastructure.py          # Crash handler, session persistence, cleanup hooks
│   ├── source_quality.py          # Domain assessment, dedup, title normalization
│   ├── persistence.py             # State persistence, throttling, flushing
│   ├── audit.py                   # Audit event logging, payload preparation
│   ├── config_builders.py         # Search provider parameter builders
│   ├── errors.py                  # Error recording, orchestrator error handling
│   ├── background_tasks.py        # Background task management, task registry
│   ├── session_management.py      # List/delete/resume sessions
│   ├── phases/
│   │   ├── __init__.py            # Re-exports all phase executors
│   │   ├── base.py                # PhaseExecutor abstract base
│   │   ├── planning.py            # Planning phase + prompts (~400 lines)
│   │   ├── gathering.py           # Gathering phase + follow-up extraction (~450 lines)
│   │   ├── analysis.py            # Analysis phase + digest management (~1000 lines)
│   │   ├── synthesis.py           # Synthesis phase + prompts (~700 lines)
│   │   └── refinement.py          # Refinement phase + prompts (~550 lines)
│   └── budgeting/
│       ├── __init__.py            # Re-exports allocators and validators
│       ├── allocators.py          # Source and synthesis budget allocation (~500 lines)
│       ├── refinement.py          # Refinement budget computation (~200 lines)
│       ├── validation.py          # Final fit validation, truncation (~200 lines)
│       └── archives.py            # Digest archive management (~150 lines)
```

---

## 3. Dependency Graph

```
                         infrastructure.py
                              ↑ (imported first for crash handler)
                              |
core.py (DeepResearchWorkflow)
├── orchestration.py (SupervisorOrchestrator, SupervisorHooks)
├── persistence.py (state persistence)
├── audit.py (audit logging)
├── config_builders.py (search provider builders)
├── errors.py (error recording)
├── background_tasks.py (task management)
├── session_management.py (resume/list/delete)
├── source_quality.py (domain assessment)
└── phases/*.py
    ├── phases/base.py
    ├── phases/planning.py
    ├── phases/gathering.py
    ├── phases/analysis.py ─────→ budgeting/allocators.py, budgeting/archives.py
    ├── phases/synthesis.py ────→ budgeting/allocators.py
    └── phases/refinement.py ──→ budgeting/refinement.py → budgeting/validation.py
```

**Key constraint**: All imports are unidirectional. Phases → Budgeting only; never reverse.

External dependencies (unchanged):
- `foundry_mcp.config.ResearchConfig`
- `foundry_mcp.core.background_task.BackgroundTask`
- `foundry_mcp.core.research.models.*`
- `foundry_mcp.core.research.memory.ResearchMemory`
- `foundry_mcp.core.research.providers.*`
- `foundry_mcp.core.research.token_management.*`
- `foundry_mcp.core.research.context_budget.*`
- `foundry_mcp.core.research.document_digest.*`

---

## 4. Migration Strategy

### Phase 1: Create Package, Extract Incrementally

Each extraction should be a separate commit:

1. Create `deep_research/` directory with `__init__.py` re-exporting everything from the monolith
2. Extract `infrastructure.py` (crash handler, session persistence)
3. Extract `orchestration.py` (AgentRole, AgentDecision, SupervisorHooks, SupervisorOrchestrator)
4. Extract `source_quality.py` (domain assessment, title normalization)
5. Extract `persistence.py`, `audit.py`, `errors.py` (workflow support)
6. Extract `config_builders.py` (search provider parameter builders)
7. Extract `background_tasks.py`, `session_management.py`
8. Extract `budgeting/` package (allocators, refinement, validation, archives)
9. Extract `phases/` package (base, planning, gathering, analysis, synthesis, refinement)
10. Final: `core.py` contains only DeepResearchWorkflow class + orchestration loop

### Phase 2: Verify Backward Compatibility

Add a public API contract test (like `test_spec_public_api.py`):

```python
def test_all_public_symbols_importable():
    from foundry_mcp.core.research.workflows.deep_research import (
        DeepResearchWorkflow,
        SupervisorOrchestrator,
        SupervisorHooks,
        AgentRole,
        AgentDecision,
        get_domain_quality,
    )
```

### Phase 3: (Optional, future) Update Internal Import Paths

Migrate internal callers to import from submodules directly. The backward-compatibility re-exports allow gradual migration.

---

## 5. Risk Assessment

### Critical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Module initialization order** | Crash handler not installed if `infrastructure.py` imported late | Import it first in `__init__.py` |
| **Circular imports** | Import failure at runtime | Enforce unidirectional imports; add cycle detection test |
| **Threading state** | `_active_research_sessions` accessed from multiple threads | Keep lock + dict in `infrastructure.py` as single source of truth |
| **Backward compat drift** | Re-exports get out of sync with submodule exports | Explicit `__all__` + contract test |
| **Search provider caching** | Stale state across phases | Document caching semantics in `config_builders.py` |

### Verification Checklist

- [ ] All public symbols importable via both old and new paths
- [ ] No circular import errors
- [ ] Crash handler fires on uncaught exceptions
- [ ] State persistence throttling works correctly
- [ ] Background task spawn/poll/cancel lifecycle works
- [ ] Audit JSONL events written correctly
- [ ] Budget validation passes/fails at correct thresholds
- [ ] All existing tests pass without modification
- [ ] No increase in memory footprint

### Estimated Effort

- Code extraction: ~3-4 hours (mechanical)
- Import reorganization: ~2-3 hours
- Test updates: ~4-6 hours
- Integration testing: ~3-4 hours
- Documentation: ~1-2 hours
- **Total: ~13-19 hours**
