# Import Consumer Audit

Captured: 2026-02-06
Spec: unified-refactoring-2026-02-06-001

Target modules: `foundry_mcp.core.spec`, `foundry_mcp.core.task`, `foundry_mcp.tools.unified.*`

---

## 1. `foundry_mcp.core.spec` Consumers

### Source Code (24 files)

| File | Symbols | Classification |
|------|---------|---------------|
| `core/__init__.py:3-13` | `find_specs_directory`, `find_spec_file`, `resolve_spec_file`, `load_spec`, `save_spec`, `backup_spec`, `list_specs`, `get_node`, `update_node` | **Internal** — re-export facade |
| `core/task/_helpers.py`, `core/task/queries.py`, `core/task/mutations.py`, `core/task/batch.py` | `CATEGORIES`, `load_spec`, `save_spec`, `find_spec_file`, `find_specs_directory`, `get_node` | **Internal** — sibling core module (split into package) |
| `core/batch_operations.py:16` | `load_spec`, `find_specs_directory`, `save_spec` | **Internal** — core utility |
| `core/review.py:12` | `load_spec`, `find_specs_directory` | **Internal** — core utility |
| `prompts/workflows.py:14-18` | `load_spec`, `list_specs`, `find_specs_directory` | **Internal** — MCP prompts |
| `resources/specs.py:15-20` | `load_spec`, `list_specs`, `find_specs_directory`, `find_spec_file` | **Internal** — MCP resources |
| `tools/unified/spec.py:32-45` | `TEMPLATES`, `TEMPLATE_DESCRIPTIONS`, `check_spec_completeness`, `detect_duplicate_tasks`, `diff_specs`, `find_spec_file`, `find_specs_directory`, `list_spec_backups`, `list_specs`, `load_spec`, `recalculate_actual_hours`, `recalculate_estimated_hours` | **Internal** — tool router |
| `tools/unified/authoring.py:26-47` | `CATEGORIES`, `PHASE_TEMPLATES`, `TEMPLATES`, `add_assumption`, `add_phase`, `add_phase_bulk`, `add_revision`, `apply_phase_template`, `create_spec`, `find_replace_in_spec`, `find_specs_directory`, `generate_spec_data`, `get_phase_template_structure`, `list_assumptions`, `load_spec`, `move_phase`, `remove_phase`, `rollback_spec`, `update_frontmatter`, `update_phase_metadata` | **Internal** — tool router (heaviest consumer) |
| `tools/unified/task.py:37` | `find_specs_directory`, `load_spec`, `save_spec` | **Internal** — tool router |
| `tools/unified/journal.py:31` | `find_specs_directory`, `load_spec`, `save_spec` | **Internal** — tool router |
| `tools/unified/lifecycle.py:24` | `find_specs_directory` | **Internal** — tool router |
| `tools/unified/plan.py:35` | `find_specs_directory` | **Internal** — tool router |
| `tools/unified/review.py:41` | `find_spec_file`, `find_specs_directory`, `load_spec` | **Internal** — tool router |
| `tools/unified/verification.py:23` | `find_specs_directory`, `load_spec`, `save_spec` | **Internal** — tool router |
| `tools/unified/research.py:743,778,884` | `load_spec`, `find_specs_directory`, `save_spec` | **Internal** — tool router (function-scope imports) |
| `cli/commands/modify.py:26-31` | `add_assumption`, `add_phase`, `add_revision`, `update_frontmatter` | **Internal** — CLI (deprecated) |
| `cli/commands/validate.py:21` | `load_spec`, `find_spec_file` | **Internal** — CLI (deprecated) |
| `cli/commands/specs.py:25` | `list_specs`, `load_spec` | **Internal** — CLI (deprecated) |
| `cli/commands/tasks.py:20` | `load_spec`, `find_spec_file`, `get_node` | **Internal** — CLI (deprecated) |
| `cli/commands/review.py:509` | `load_spec`, `find_spec_file` | **Internal** — CLI (function-scope) |
| `cli/commands/journal.py:19` | `load_spec`, `find_spec_file` | **Internal** — CLI (deprecated) |
| `cli/commands/plan.py:22` | `find_specs_directory` | **Internal** — CLI (deprecated) |
| `cli/config.py:11` | `find_specs_directory` | **Internal** — CLI config |

### Test Files (10 files)

| File | Symbols | Classification |
|------|---------|---------------|
| `tests/unit/test_core/test_spec.py:9-25` | `PHASE_TEMPLATES`, `apply_phase_template`, `find_specs_directory`, `find_spec_file`, `get_node`, `get_phase_template_structure`, `list_specs`, `load_spec`, `update_node`, `add_revision`, `update_frontmatter`, `add_phase`, `remove_phase`, `move_phase`, `recalculate_actual_hours`, `recalculate_estimated_hours` | **Internal** — unit tests |
| `tests/unit/test_core/test_task.py:25` | `load_spec` | **Internal** — unit tests |
| `tests/unit/test_core/test_spec_history.py:12-22` | `DEFAULT_MAX_BACKUPS`, `DEFAULT_BACKUP_PAGE_SIZE`, `DEFAULT_DIFF_MAX_RESULTS`, `backup_spec`, `diff_specs`, `list_spec_backups`, `load_spec`, `rollback_spec`, `_apply_backup_retention` | **Internal** — unit tests |
| `tests/unit/test_core/test_spec_validation.py:9` | `check_spec_completeness`, `detect_duplicate_tasks` | **Internal** — unit tests |
| `tests/unit/test_core/test_spec_find_replace.py:9` | `find_replace_in_spec`, `load_spec` | **Internal** — unit tests |
| `tests/unit/test_core/test_phase_metadata_update.py:9-12` | `load_spec`, `update_phase_metadata` | **Internal** — unit tests |
| `tests/unit/test_batch_operations.py` | `load_spec` (14 function-scope imports) | **Internal** — unit tests |
| `tests/unit/test_contracts/test_phase6_contracts.py:849` | `load_spec` | **Internal** — contract tests |
| `tests/unit/test_core/test_task_batch_update.py:10` | `load_spec` | **Internal** — unit tests |
| `tests/unit/test_sdd_cli_runtime.py:18` | `find_specs_directory` | **Internal** — runtime tests |

### Most Imported Symbols

| Symbol | Import count |
|--------|------------:|
| `load_spec` | 23 |
| `find_specs_directory` | 15 |
| `find_spec_file` | 9 |
| `save_spec` | 7 |
| `list_specs` | 4 |
| `get_node` | 3 |

---

## 2. `foundry_mcp.core.task` Consumers

> **Note:** `core.task` was split from a single `task.py` into a `task/` package with sub-modules
> (`_helpers.py`, `queries.py`, `mutations.py`, `batch.py`). The `__init__.py` re-exports all
> public symbols, so consumer imports remain unchanged.

### Source Code (7 files)

| File | Symbols | Classification |
|------|---------|---------------|
| `core/__init__.py:15-25` | `is_unblocked`, `is_in_current_phase`, `get_next_task`, `check_dependencies`, `get_previous_sibling`, `get_parent_context`, `get_phase_context`, `get_task_journal_summary`, `prepare_task` | **Internal** — re-export facade |
| `core/batch_operations.py:17` | `is_unblocked` | **Internal** — core utility |
| `tools/unified/task.py:46-59` | `add_task`, `batch_update_tasks`, `check_dependencies`, `get_next_task`, `manage_task_dependency`, `move_task`, `prepare_task` (as `core_prepare_task`), `remove_task`, `REQUIREMENT_TYPES`, `update_estimate`, `update_task_metadata`, `update_task_requirements` | **Internal** — tool router (heaviest consumer) |
| `tools/unified/authoring.py:48` | `TASK_TYPES` | **Internal** — tool router |
| `cli/commands/tasks.py:28-36` | `check_dependencies`, `get_next_task`, `get_parent_context`, `get_phase_context`, `get_previous_sibling`, `get_task_journal_summary`, `prepare_task` | **Internal** — CLI (deprecated) |
| `cli/commands/modify.py:25` | `add_task`, `remove_task` | **Internal** — CLI (deprecated) |
| `core/research/workflows/deep_research.py:46` | `task_registry` (module) | **Internal** — research workflow |

### Test Files (3 files)

| File | Symbols | Classification |
|------|---------|---------------|
| `tests/unit/test_core/test_task.py:9-24` | `is_unblocked`, `is_in_current_phase`, `get_next_task`, `check_dependencies`, `get_previous_sibling`, `get_parent_context`, `get_phase_context`, `get_task_journal_summary`, `prepare_task`, `add_task`, `update_task_metadata`, `move_task`, `manage_task_dependency`, `update_task_requirements` | **Internal** — unit tests |
| `tests/unit/test_core/test_task_batch_update.py:9` | `batch_update_tasks` | **Internal** — unit tests |
| `tests/unit/test_core/test_task_registry.py:20-33` | 12 symbols from `foundry_mcp.core.task_registry` (separate module) | **Internal** — unit tests |

### Unique Symbols from `core.task`

`add_task`, `batch_update_tasks`, `check_dependencies`, `get_next_task`, `get_parent_context`, `get_phase_context`, `get_previous_sibling`, `get_task_journal_summary`, `is_in_current_phase`, `is_unblocked`, `manage_task_dependency`, `move_task`, `prepare_task`, `remove_task`, `REQUIREMENT_TYPES`, `TASK_TYPES`, `update_estimate`, `update_task_metadata`, `update_task_requirements`

---

## 3. `foundry_mcp.tools.unified.*` Consumers

### Source Code (3 external files + 14 internal cross-imports)

| File | Symbols | Classification |
|------|---------|---------------|
| `server.py:23` | `register_unified_tools` | **Internal** — server entry point |
| `tools/__init__.py:6` | `register_unified_tools` | **Internal** — package init |
| `cli/commands/review.py:33-45` | `_build_implementation_artifacts`, `_build_journal_entries`, `_build_spec_requirements`, `_build_test_results` (from `documentation_helpers`); `DEFAULT_AI_TIMEOUT`, `REVIEW_TYPES`, `_get_llm_status`, `_run_ai_review`, `_run_quick_review` (from `review_helpers`) | **Internal** — CLI review |

### Internal Cross-Module Imports (within `tools/unified/`)

All 14 tool routers import from `tools/unified/router`:
- `ActionDefinition`, `ActionRouter`, `DispatchError`, `error_response`, `success_response`

`tools/unified/server.py` manifest builder imports all 14 router singletons:
- `_AUTHORING_ROUTER`, `_ENVIRONMENT_ROUTER`, `_ERROR_ROUTER`, `_HEALTH_ROUTER`, `_JOURNAL_ROUTER`, `_LIFECYCLE_ROUTER`, `_PLAN_ROUTER`, `_PROVIDER_ROUTER`, `_RESEARCH_ROUTER`, `_REVIEW_ROUTER`, `_SERVER_ROUTER`, `_SPEC_ROUTER`, `_TASK_ROUTER`, `_VERIFICATION_ROUTER`

`tools/unified/__init__.py:53` — Dynamic import via `importlib.import_module("foundry_mcp.tools.unified.task")` for conditional tool registration.

### Test Files (50+ files)

Tests extensively import dispatch functions (`_dispatch_*_action`) and handler functions (`_handle_*`) from individual tool modules. All classified as **internal**.

---

## 4. Pickling / Reflection Usage

### Pickling: NONE

No `pickle`, `pickle.dump`, `pickle.dumps`, `pickle.load`, `pickle.loads`, or `import pickle` found anywhere in the codebase.

### Reflection on Target Modules

| Type | Occurrences | Details |
|------|:-----------:|---------|
| `importlib.import_module` on target | 1 | `tools/unified/__init__.py:53` — dynamic load of `foundry_mcp.tools.unified.task` for conditional registration |
| `getattr`/`setattr`/`hasattr` on target | 0 | No dynamic attribute access on the three target modules |
| `inspect.*` on target | 0 | None |
| `sys.modules` on target | 0 | None |
| `eval`/`exec` on target | 0 | None |
| `__import__` on target | 0 | None |

### Reflection on Other Modules (not target, for reference)

- `core/providers/registry.py:313,352` — `importlib.import_module` for provider factory loading
- `tools/unified/environment.py:880` — `__import__` for package validation
- `core/health.py:693` — `__import__("threading")` for lock initialization

---

## 5. Compatibility Mode Decision

### Assessment

All consumers of the three target modules are **strictly internal**:

- **No external/third-party packages** import from `foundry_mcp.core.spec`, `foundry_mcp.core.task`, or `foundry_mcp.tools.unified`
- **No documented public API** exposes these modules to external consumers
- **No pickling** means no serialized references to module paths
- **One dynamic import** (`tools/unified/__init__.py`) uses `importlib.import_module` for conditional registration — must be updated if the module path changes
- The `core/__init__.py` re-export facade provides the only abstraction layer

### Decision: Strict-Internal Mode

**Compatibility mode: NONE REQUIRED**

Rationale:
1. All 34+ source consumers and 60+ test consumers are within the `foundry_mcp` package
2. No external documentation advertises these import paths as public API
3. No pickling or dynamic attribute access creates hidden dependencies
4. The `core/__init__.py` re-export facade can absorb internal restructuring
5. The single `importlib.import_module` call is trivially updatable

**Migration approach:** Direct rename/move with search-and-replace across `src/` and `tests/`. No deprecation shims, compatibility re-exports, or transition period needed.
