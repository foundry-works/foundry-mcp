# Unified Refactoring Plan for foundry-mcp

## Objective

Execute high-leverage refactors that reduce maintenance cost and defect risk without changing external behavior. The codebase (~34,400 lines across 38 core modules and 21 unified tool files) is architecturally sound but has accumulated significant duplication and god-module bloat that makes cross-cutting changes expensive and error-prone.

## Primary Outcomes

- Reduce coupling and duplication across unified tool routers (~500+ lines of repeated helpers, reducing cross-cutting changes from 10+ file edits to 1-2 file edits).
- Split oversized god modules into composable, independently testable units.
- Consolidate search provider boilerplate into shared utility functions.
- Unify tool registration and discovery metadata from a single source of truth.
- Improve test maintainability while preserving response-v2 contract guarantees.
- Preserve concurrency, timeout, and resilience behavior during all structural changes.

## Scope

### In Scope
- Internal module decomposition and helper extraction.
- Unified router/dispatch framework consolidation.
- Search provider utility extraction.
- Tool registry + manifest source unification.
- God module splits (spec.py, task.py, deep_research.py).
- Test-suite deduplication for repeated dispatcher behavior.

### Out of Scope
- New end-user features.
- Response envelope schema changes.
- Provider algorithm changes (quality/behavior).

## Required Constraints

- Preserve `response-v2` contract and metadata semantics.
- Use shared response helpers (`success_response`, `error_response`) consistently.
- Maintain early input validation and actionable errors.
- Preserve security boundaries, sanitization, and redaction behavior.
- Keep timeout/cancellation/concurrency safeguards unchanged or improved.
- Keep specs, docs, and tests synchronized in each changeset.

## Refactoring Principles

- **Behavior-preserving by default.** When behavior changes are unavoidable, gate by explicit versioning and updated contract tests.
- **Extract only when duplication appears in 3+ places.** Don't create abstractions for hypothetical reuse.
- **Incremental changesets with passing tests after each phase.** Never leave the codebase in a broken state.
- **Define split boundaries first, then move code.** Update all import paths in the same changeset — no compatibility facades needed since the refactoring ships as a single release.
- **Mechanical before design.** Do safe, obvious extractions first to reduce noise before tackling changes that require architectural decisions.
- **Cohesion over line count.** No module should contain functions from distinct responsibility domains. This is the split criterion — not an arbitrary line-count cap.

---

## Phase 0: Baseline & Green Suite (Prerequisite)

**Goal:** Confirm green test suite and capture baseline measurements before moving any code.

### Existing Test Infrastructure
Strong test infrastructure already exists and should be leveraged, not recreated:
- `tests/contract/test_response_schema.py` (510 lines) — comprehensive contract coverage for response-v2 envelope.
- `tests/core/research/workflows/test_deep_research.py` (2,403 lines) — extensive deep research workflow coverage including status/cancel/resume/persistence.
- Root `tests/conftest.py` (304 lines) — fixture versioning (`FIXTURE_SCHEMA_VERSION`), freshness validation (`validate_fixture_freshness`), and response envelope validation (`validate_response_envelope`).

### Tasks
1. **Run full test suite** and confirm green: `pytest tests/ -x --timeout=60`.
2. **Capture baseline metrics** and commit as `dev_docs/refactoring/baseline_metrics.json`:
   - Module line counts for all files listed in Key Files Reference.
   - Per-tool-file helper function counts (`_validation_error`, `_request_id`, `_metric_name`, `_resolve_specs_dir`).
   - Duplicated helper pattern inventory with exact file locations.

### Deferred Audits
These audits happen lazily when their dependent phases begin, not upfront:
- **Contract test gap audit** (before Phase 1): verify coverage for all 16 unified tool response shapes.
- **Deep research characterization audit** (before Phase 6): confirm existing tests cover status/report/cancel/resume behavior.

### Exit Criteria
- Full test suite green.
- `baseline_metrics.json` committed with line counts and duplicate helper inventory.

---

## Phase 1: Extract Shared Tool Helpers + Dispatch Test Dedup

**Goal:** Remove ~500 lines of near-identical private helpers duplicated across unified tool files, and deduplicate the dispatch-exception test boilerplate that becomes consolidatable via the new `dispatch_with_standard_errors` wrapper.

### Problem
The `tools/unified/` directory contains 21 files (16 tool routers + 3 helper files + 2 infrastructure files). The 16 tool routers each contain their own copies of:
- `_validation_error()` — 10 files, differing only by tool name
- `_request_id()` — 8 files, differing only by prefix
- `_metric_name()` — 5 files, differing only by prefix
- `_resolve_specs_dir()` — 5 files, identical logic
- `_dispatch_*_action()` try/except boilerplate — all 16 routers (13 identical, 3 with minor signature variations), ~25 lines each

### Dispatch Function Audit (Resolved)
All 16 dispatch functions were audited. **13 of 16 routers** follow an identical two-level try/except pattern (ActionRouterError → Exception) with only cosmetic differences (tool name in messages, presence/absence of `request_id` or `details` fields). The 3 non-standard routers:

- **health.py** — unique signature (`include_details` kwarg instead of `payload`/`config`), but error handling logic is standard. Pre-processes kwargs before dispatch.
- **plan.py / pr.py** — simplified signatures (no `config` parameter), otherwise identical error handling.
- **research.py** — standard dispatch error handling; complex error recovery exists in individual action handlers (provider-specific `AuthenticationError`, `RateLimitError`, `UrlValidationError`), not in the dispatch wrapper.

**Decision: Proceed with `dispatch_with_standard_errors`.** All meaningful behavioral divergence lives in action handlers, not dispatch functions. The wrapper handles the mechanical ActionRouterError/Exception pattern; per-action error logic is untouched.

### Solution
Create `src/foundry_mcp/tools/unified/common.py` with parameterized versions:
```python
build_request_id(tool_name: str) -> str
validation_error(tool_name, *, field, action, message, request_id, code, remediation) -> dict
make_metric_name(prefix: str, action: str) -> str
resolve_specs_dir(config, path_or_workspace, mode=...) -> Optional[Path]
dispatch_with_standard_errors(router, tool_name, action, **kwargs) -> dict
```

`dispatch_with_standard_errors` parameterizes: `tool_name` (error messages/logging), `router` (ActionRouter instance), `action` string, `include_request_id` (bool, default True), `include_details_in_router_error` (bool, default False). health.py pre-processes its kwargs before calling the wrapper. research.py's action-level error handling is unaffected — it lives inside handlers, not dispatch.

### Adoption Strategy
Migrate 4 routers first (`task.py`, `authoring.py`, `spec.py`, `server.py`), then the remaining routers in a second pass.

### Dispatch Test Deduplication
With `dispatch_with_standard_errors` extracted, the per-router dispatch-exception tests become consolidatable. Create shared parametrized test fixtures that exercise the standard dispatch error paths (ActionRouterError, unexpected Exception) across all routers using the common wrapper. Per-tool tests retain only tool-specific behavior assertions.

This is the natural time for this dedup — the shared wrapper makes the tests structurally identical and consolidatable in a single pass.

### Risks & Mitigations
- **Risk:** Hidden differences in error wording between routers.
- **Mitigation:** Golden/contract assertions for `error_code`, `error_type`, `remediation`, and `meta.version`.
- **Risk (resolved):** Dispatch wrapper hides per-router behavioral nuance.
- **Finding:** Audit confirmed all meaningful behavioral divergence lives in action handlers, not dispatch functions. The 9/16 routers that include `request_id` and the 2/16 that include `details` in the ActionRouterError handler are parameterized via the wrapper's `include_request_id` and `include_details_in_router_error` flags.

### Exit Criteria
- At least 4 routers migrated with no behavior regressions.
- Measurable boilerplate reduction (target: 15-25% across migrated modules).
- Dispatch-exception test dedup: baseline count of near-duplicate dispatch tests recorded, with ≥50% consolidated into parametrized shared tests.

**Nature:** Mechanical extraction. No behavior change.

---

## Phase 1b: Standardize Singleton Accessors (Stretch/Optional)

**Goal:** Standardize the 24+ singleton/accessor patterns and remove boilerplate module-level singleton assignments from tool files.

**Status: Optional.** All three singleton patterns work correctly and the inconsistency is an implementation detail invisible to consumers — as long as `get_X()` is the public API (which it already is for 19 of 22 singletons), the internal mechanism doesn't matter. This phase touches sensitive initialization code for zero external ROI and risks subtle startup-order regressions. Demoted to stretch alongside Phase 7.

### Problem
The codebase uses three distinct singleton mechanisms:
1. **Class-level `_instance` with `__new__` + lock** (1 class): `ObservabilityManager`.
2. **Module-level lazy-init `get_X()` accessor** (~19 accessors): `get_resilience_manager()`, `get_config()`, `get_error_collector()`, `get_provider_executor()`, `get_error_store()`, `get_watchdog()`, `get_rate_limit_manager()`, `get_metrics_collector()`, `get_prometheus_exporter()`, `get_tool_registry()`, `get_health_manager()`, `get_llm_config()`, `get_workflow_config()`, `get_consultation_config()`, `get_intake_store()`, `get_metrics_store()`, `get_global_registry()`, `get_capabilities_registry()`, etc.
3. **Module-level eager instantiation** (2 instances): `_metrics = MetricsCollector()` and `_audit = AuditLogger()` in `observability.py`.

Note: `ProviderResilienceManager` uses pattern 2 (module-level `get_resilience_manager()` accessor), not pattern 1.

Additionally, 12 tool files import `get_metrics()` and cache it as `_metrics = get_metrics()` at module level.

### Solution
- Standardize `ObservabilityManager` to use the module-level `get_X()` lazy-init pattern (consistent with the majority of the codebase, including `ProviderResilienceManager` which already uses this pattern).
- Convert `MetricsCollector` and `AuditLogger` from eager to lazy instantiation.
- Remove module-level `_metrics = get_metrics()` from all 12 tool files — call `get_metrics()` directly at each use site.

### Scope Limitation
This phase does NOT attempt to reduce the total number of singletons or consolidate them — only to make the existing ones consistent.

**Nature:** Mechanical. Low risk.

---

## Phase 2: Split God Modules into Packages

**Goal:** Break the largest core modules into focused, independently testable sub-modules.

**Import risk ordering:** Split in ascending order of import fan-out to reduce blast radius during early phases. Since the refactoring ships as a single release, all import paths are updated in-place — no re-export facades needed except for the permanent public API surface in `core/__init__.py`.

### Internal Dependency Analysis (Completed)
Call graphs for Phases 2a and 2b have been fully traced. Key findings:

- **2a (task.py):** No mutation→query or query→mutation calls. One shared helper (`_get_phase_for_node`) extracted to `_helpers.py`. Split boundaries are validated.
- **2b (spec.py):** Dependency graph is a clean DAG: `_constants ← io ← hierarchy ← templates`, with `analysis ← io` only. No circular dependencies. One cross-boundary call: `templates.apply_phase_template()` → `hierarchy.add_phase_bulk()` (not circular).

**If circular dependencies are discovered during implementation**, apply these strategies in priority order:
1. **Merge coupled functions into the same sub-module.** Adjust the split boundary — a slightly uneven split is better than a circular dependency.
2. **Extract shared types into a `_types.py` sub-module.** If the cycle is caused by shared data structures.
3. **Use runtime imports as a last resort.** Move the import inside the function body that needs it.

### Re-export Chain Policy

When a god module is split into a package, the re-export chain ensures consumers can import from any level without needing to know the internal structure:

```
core/__init__.py  →  core/task/__init__.py  →  core/task/queries.py
                                             →  core/task/mutations.py
                                             →  core/task/batch.py

core/__init__.py  →  core/spec/__init__.py  →  core/spec/io.py
                                             →  core/spec/hierarchy.py
                                             →  core/spec/templates.py
                                             →  core/spec/analysis.py
```

**Rules:**
1. Each package `__init__.py` (`core/task/__init__.py`, `core/spec/__init__.py`) re-exports the **full public API** so that `from foundry_mcp.core.task import add_task` continues to work.
2. `core/__init__.py` re-exports the same symbols it does today — no changes to the top-level public surface.
3. Internal consumers (tool handlers, tests) are updated to import from sub-modules directly for clarity, but importing from the package root remains valid.
4. All new `__init__.py` files define `__all__` listing every re-exported symbol.

### 2a: Split `core/task.py` (2,463 lines) → `core/task/` Package

**Import fan-out: 7 files** (lowest risk — split this first)

#### Validated Call Graph

Internal dependency analysis confirms the queries/mutations boundary is clean:
- **No mutation→query calls.** `add_task()` does NOT call `is_unblocked()` or any query function.
- **No query→mutation calls.** Query functions are pure readers.
- **One shared private helper:** `_get_phase_for_node()` is used by both `get_phase_context()` (queries) and `move_task()` (mutations). Extracted to `_helpers.py`.
- **batch.py is fully independent.** `batch_update_tasks()` and `_match_tasks_for_batch()` have no calls to queries or mutations.

Private helper clustering (validated):
- **Query-only:** `_get_sibling_ids`, `_get_latest_journal_excerpt`, `_find_phase_node`, `_compute_auto_mode_hints`
- **Mutation-only:** `_generate_task_id`, `_update_ancestor_counts`, `_decrement_ancestor_counts`, `_collect_descendants`, `_count_tasks_in_subtree`, `_remove_dependency_references`, `_would_create_circular_dependency`, `_can_reach_via_dependency`, `_is_descendant`, `_check_cross_phase_dependencies`, `_generate_requirement_id`
- **Shared:** `_get_phase_for_node` → `_helpers.py`

| Module | Contents | ~Lines |
|---|---|---|
| `task/_helpers.py` | `_get_phase_for_node()`, shared constants (`TASK_TYPES`, `REQUIREMENT_TYPES`) | ~80 |
| `task/queries.py` | is_unblocked, is_in_current_phase, get_next_task, check_dependencies, get_*_context, prepare_task + query-only private helpers | ~820 |
| `task/mutations.py` | add_task, remove_task, move_task, update_estimate, manage_task_dependency, update_task_metadata, update_task_requirements + mutation-only private helpers | ~1,260 |
| `task/batch.py` | batch_update_tasks, _match_tasks_for_batch | ~300 |

**Public API surface:** 9 symbols + 2 constants (`TASK_TYPES`, `REQUIREMENT_TYPES`) currently re-exported via `core/__init__.py` — these permanent re-exports are maintained. All internal consumers are updated to import from the new sub-modules directly.

### 2b: Split `core/spec.py` (4,116 lines) → `core/spec/` Package

**Import fan-out: 24 files** (moderate risk)

#### Validated Call Graph

Internal dependency analysis reveals a clean DAG with no circular dependencies:

```
_constants.py  (no deps)
     ↑
   io.py  (imports _constants)
     ↑
hierarchy.py  (imports io, _constants)
     ↑
templates.py  (imports io, hierarchy, _constants)

analysis.py   (imports io only — no hierarchy/templates deps)
```

**Cross-boundary dependency details:**
- **hierarchy → io (heavy):** `add_phase`, `add_phase_bulk`, `remove_phase`, `move_phase`, `update_phase_metadata`, `recalculate_estimated_hours`, `recalculate_actual_hours` all call `find_specs_directory()`, `find_spec_file()`, `load_spec()`, `save_spec()`. This is expected — hierarchy operations are spec mutations that need I/O.
- **templates → io (heavy):** `add_assumption`, `add_revision`, `list_assumptions`, `update_frontmatter`, `find_replace_in_spec` all call I/O functions. `create_spec()` writes directly (calls `find_specs_directory()` but not `save_spec()`).
- **templates → hierarchy (single call):** `apply_phase_template()` calls `add_phase_bulk()`. Not circular — hierarchy does not call templates.
- **analysis → io (read-only):** All analysis functions call `find_specs_directory()`, `find_spec_file()`, `load_spec()` — never `save_spec()`.
- **No circular dependencies.** The graph is strictly layered.

**Shared private helpers** (used across proposed module boundaries):
- `_requires_rich_task_fields()` and `_normalize_acceptance_criteria()` — used by `add_phase_bulk()` in hierarchy. These are validation helpers that belong with hierarchy since that's their only call site.

**Misplacement fix:** `rollback_spec()` is a backup/restore operation. The original plan listed it ambiguously — it belongs in `io.py` with `backup_spec()` and `list_spec_backups()`.

| Module | Contents | ~Lines |
|---|---|---|
| `spec/_constants.py` | `TEMPLATES`, `TEMPLATE_DESCRIPTIONS`, `CATEGORIES`, `VERIFICATION_TYPES`, `PHASE_TEMPLATES`, `DEFAULT_MAX_BACKUPS`, `DEFAULT_BACKUP_PAGE_SIZE`, `MAX_BACKUP_PAGE_SIZE` | ~40 |
| `spec/io.py` | find_specs_directory, find_spec_file, resolve_spec_file, find_git_root, load_spec, save_spec, backup_spec, rollback_spec, list_spec_backups, _migrate_spec_fields, _validate_spec_structure, _apply_backup_retention | ~800 |
| `spec/hierarchy.py` | get_node, update_node, add_phase, add_phase_bulk, remove_phase, move_phase, update_phase_metadata, recalculate_estimated_hours, recalculate_actual_hours, _collect_descendants, _count_tasks_in_subtree, _remove_dependency_references, _add_phase_verification, _generate_phase_id, _requires_rich_task_fields, _normalize_acceptance_criteria | ~1,500 |
| `spec/templates.py` | generate_spec_data, create_spec, get_template_structure, get_phase_template_structure, apply_phase_template, generate_spec_id, add_assumption, add_revision, list_assumptions, update_frontmatter, find_replace_in_spec | ~780 |
| `spec/analysis.py` | check_spec_completeness, detect_duplicate_tasks, diff_specs, list_specs, _load_spec_source, _diff_node + analysis-specific constants (_CC_WEIGHT_*, _DD_*, _FR_*) | ~800 |

**Note on hierarchy.py size:** At ~1,500 lines this is the densest sub-module. It contains functions from a single responsibility domain (spec hierarchy operations), so it does not violate the cohesion rule. If navigability becomes an issue, consider splitting phase operations (`add_phase`, `remove_phase`, `move_phase`, `add_phase_bulk`) into `spec/phases.py` — assess during implementation based on developer experience, not line count alone.

**Public API surface:** 13 symbols + 3 constants (`CATEGORIES`, `PHASE_TEMPLATES`, `TEMPLATES`) currently re-exported via `core/__init__.py` — these permanent re-exports are maintained. Critical path: `load_spec` is imported by all 24 dependents; all 24 import statements are updated to the new sub-module paths in the same changeset.

### 2c: Split `config.py` (2,012 lines) → `config/` Package (Stretch)

**Status: Stretch.** config.py is 2,012 lines of mostly dataclass definitions with 35 dependents. The risk-to-reward ratio is unfavorable: the module is internally cohesive (dataclass definitions with minimal cross-calling), has the highest import fan-out in the codebase, and the maintenance cost of a single flat file of dataclasses is low. Demoted to stretch alongside Phases 1b, 5b, and 7.

Split into: `config/server.py`, `config/observability.py`, `config/research.py`, `config/runners.py`, `config/git.py`, `config/helpers.py`. Re-export from `__init__.py`.

**Critical path:** `ServerConfig` is imported by 23 files, `ResearchConfig` by 7. All import statements are updated to the new sub-module paths in the same changeset. A `config/__init__.py` re-exports the public API symbols for external consumers.

### Exit Criteria (all splits)
- All imports updated to new sub-module paths (no stale paths remain).
- No caller behavior changes.
- Import-cycle-free: `python -c "import foundry_mcp.server"` succeeds.
- Full test suite passes.
- **Cohesion check:** No sub-module mixes functions from distinct responsibility domains. Total line count across split modules is net-zero or lower (restructuring should not add lines).

### Test Migration Strategy
- Update all test imports to use the new sub-module paths directly.
- Audit `tests/unit/test_core/test_task.py` and `tests/unit/test_core/test_task_batch_update.py` for imports of internal functions that move between `queries.py`/`mutations.py`/`batch.py`.
- For spec split: audit all test files importing from `foundry_mcp.core.spec` and update to the new sub-module paths.

### Naming Note
Phase 2a produces `core/task/` (domain logic) while Phase 3 produces `tools/unified/task_handlers/` (tool routing). The `_handlers` suffix in Phase 3 disambiguates the two packages.

---

## Phase 3: Split Large Tool Handler Files

**Goal:** Break the two largest unified tool files into domain-focused handler modules with thin registration surfaces.

### Proposed Structure

**`tools/unified/task_handlers/`** (from `task.py`, 3,887 lines)
| Module | Contents |
|---|---|
| `__init__.py` | Router build + registration entry point |
| `handlers_lifecycle.py` | start, complete, block, unblock, etc. |
| `handlers_batch.py` | prepare-batch, start-batch, etc. |
| `handlers_query.py` | list, query, hierarchy, progress |
| `handlers_mutation.py` | add, remove, move, metadata/deps |

**`tools/unified/authoring_handlers/`** (from `authoring.py`, 3,645 lines)
| Module | Contents |
|---|---|
| `__init__.py` | Router build + registration entry point |
| `handlers_spec.py` | Spec CRUD operations |
| `handlers_phase.py` | Phase management |
| `handlers_metadata.py` | Metadata operations |
| `handlers_intake.py` | Intake queue operations |

### Key Rules
- Preserve public tool name/signature and action names.
- Preserve telemetry and audit hooks.
- Do not introduce manual envelope construction.
- ActionRouter registration stays in `__init__.py`.

### Dependencies
- **Requires** Phase 1 (shared helpers) being done first.
- **Requires** Phase 2a (task split) and Phase 2b (spec split). Investigation confirmed:
  - `tools/unified/task.py` imports **12 symbols directly** from `foundry_mcp.core.task` (not via `core/__init__.py`): `add_task`, `batch_update_tasks`, `check_dependencies`, `get_next_task`, `manage_task_dependency`, `move_task`, `prepare_task`, `remove_task`, `REQUIREMENT_TYPES`, `update_estimate`, `update_task_metadata`, `update_task_requirements`.
  - `tools/unified/authoring.py` imports **20 symbols directly** from `foundry_mcp.core.spec`: all major functions plus `CATEGORIES`, `PHASE_TEMPLATES`, `TEMPLATES`.
  - Splitting tool handlers before splitting core modules would require updating these imports twice. Phase 2a/2b must land first so Phase 3 can target the final import paths in one pass.
- **Does not require** Phase 2c (config split is stretch and independent).

### Test Migration Strategy
- Tool tests in `tests/tools/unified/` (16 test files, 2,159 lines) should use public tool interfaces (the registered MCP tool entry points), not internal handler functions.
- Update `test_task.py` and `test_authoring.py` imports to use the new handler module paths.

### Exit Criteria
- Existing tests pass with updated import paths.
- Registration remains via existing entrypoints.
- **Cohesion check:** Each handler module contains functions from a single responsibility domain. Each `__init__.py` is ≤50 lines (thin registration surface).

---

## Phase 4: Search Provider Utility Extraction

**Goal:** Eliminate duplicated boilerplate across 5 search providers using shared utility functions.

### Problem
5 providers (`tavily.py` 675 LOC, `perplexity.py` 730 LOC, `google.py` 651 LOC, `semantic_scholar.py` 673 LOC, `tavily_extract.py` 734 LOC — 3,463 lines total) each duplicate:
- Initialization boilerplate (~40 lines): API key from env, httpx client, rate limit config, resilience config setup
- Error classification (~74 lines): 96% identical HTTP status → ErrorClassification mapping (only Google adds quota-specific handling)
- Resilience wrapping (~91 lines): `_execute_with_retry()` — nearly identical across all 5
- Parsing helpers (~48 lines): `_parse_retry_after()`, `_extract_error_message()`, `_parse_date()`, `_extract_domain()` — identical
- Health check (~18 lines): structurally identical across all 5

Validated boilerplate breakdown: Tavily 48%, Perplexity 48%, Google 55%, Semantic Scholar 29%, Tavily Extract 25%. Total: **~1,416 lines of boilerplate** (41% of provider code).

### Solution: Utility Functions (Not Base Class)
The existing `SearchProvider` ABC provides only ~27 lines of actual code reuse (the `SearchResult.to_research_source()` method). Adding another inheritance layer (`HttpSearchProvider`) would add coupling and learning curve for modest additional benefit — the shared logic is better expressed as composable utility functions.

Create `providers/shared.py` with:
```python
# Pure functions — identical across all 5 providers
parse_retry_after(response: httpx.Response) -> Optional[float]
extract_error_message(response: httpx.Response) -> str
parse_iso_date(date_str: str) -> Optional[datetime]
extract_domain(url: str) -> Optional[str]

# Parameterized — same pattern, different config
classify_http_error(error: Exception, provider_name: str,
                    custom_classifier: Optional[Callable] = None) -> ErrorClassification
# custom_classifier callback handles provider-specific codes (e.g., Google quota)

create_resilience_executor(provider_name: str, config: ProviderResilienceConfig,
                           client: httpx.AsyncClient) -> Callable
# Returns an async callable wrapping the _execute_with_retry pattern

check_provider_health(provider_name: str, api_key: str, base_url: str) -> dict

# Provider client factory — eliminates __init__ boilerplate across all 5 providers
create_provider_client(provider_name: str, env_key: str,
                       base_url: str) -> tuple[httpx.AsyncClient, RateLimitConfig]
# Handles: env var lookup, API key validation, httpx client creation,
# rate limit config from settings, resilience config setup.
# Each provider's __init__ reduces to: client, rate_config = create_provider_client(...)
```

Each provider keeps its own `search()` and `_parse_response()` — the provider-specific logic that genuinely differs. Providers call shared utilities instead of duplicating them.

### Companion: Parametrize Provider Tests
Create shared parametrized test fixtures for the common behaviors (error classification, retry logic, health checks). Per-provider tests cover only provider-specific search/parse logic.

### Exit Criteria
- ~800 lines removed from provider implementations (the 4 pure-function families + resilience wrapper + client factory).
- Error classification consolidated with callback extension point for Google's quota handling.
- All provider tests pass with parametrized base.
- Adding a new provider requires: `search()`, `_parse_response()`, `get_provider_name()` + a `create_provider_client()` call — calling shared utilities for everything else.

---

## Phase 5a: Tool Registration Parity Test (Required)

**Goal:** Prevent drift between runtime registration and manifest output with an automated check.

### Tasks
- Add parity test asserting manifest tool count/names match registered routers.
- Remove stale wording/comments (e.g., fixed tool counts like "15-tool").

### Exit Criteria
- Drift detection test added and passing.
- Stale counts/wording cleaned up.

---

## Phase 5b: Tool Catalog Single Source of Truth (Stretch)

**Goal:** Eliminate divergence between runtime registration and `server(action="tools")` manifest output via a centralized catalog.

**Status: Stretch.** Defer until there is evidence of actual drift-caused issues. The parity test from Phase 5a provides the safety net.

### Solution
New registry descriptor: `src/foundry_mcp/tools/unified/catalog.py`
- Canonical tool IDs, categories, descriptions, version, tags
- Action summaries/aliases
- Registration callable mapping

Used by:
- `tools/unified/__init__.py` (registration)
- `tools/unified/server.py` (tools manifest)

### Exit Criteria
- Manifest output and runtime registration generated from one data model.

---

## Phase 6: Decompose `deep_research.py` (6,994 lines) → Phase Modules

**Goal:** Make each research phase independently testable and navigable.

### Prerequisites
- Characterization tests from Phase 0 must cover status/report/cancel/resume behavior.
- **Recommended:** Phase 2a should be complete before starting Phase 6. While there is no technical dependency, Phase 2a establishes the split-and-update-imports pattern on a simpler module (2,463 lines, 7 dependents). Applying the same pattern to the largest file (6,994 lines) without that confidence is higher risk.

### Proposed Structure: `research/workflows/deep_research/`

| Module | Contents | ~Lines |
|---|---|---|
| `orchestrator.py` | Main class, phase sequencing, state transitions, crash recovery | ~950 |
| `_constants.py` | Budget constants, phase config defaults | ~200 |
| `planning.py` | Planning phase logic, query decomposition, planning prompts | ~650 |
| `gathering.py` | Gathering phase logic, extract followup, gathering prompts | ~850 |
| `analysis.py` | Analysis phase logic, analysis prompts (without digest) | ~950 |
| `digest.py` | Digest step extraction from analysis phase | ~600 |
| `synthesis.py` | Synthesis phase logic, report generation, synthesis prompts | ~850 |
| `refinement.py` | Refinement phase logic, gap analysis, refinement prompts | ~650 |

### Design Decision: Extract Functions to Sub-modules

Phase functions are extracted as **standalone functions in sub-modules** that the orchestrator imports and calls directly — the same pattern used in Phase 2a for task.py. No Protocol classes, no PhaseContext dataclass, no additional abstraction layers.

Phase methods already receive their dependencies as parameters (config, memory, state, hooks, etc.). The extraction is mechanical: move functions to their respective sub-modules, update imports in the orchestrator. If bundling frequently-passed parameters becomes unwieldy, a simple dataclass can group them — but this is a convenience decision made during implementation, not an upfront architectural commitment.

```python
# orchestrator.py — imports and calls phase functions directly
from .planning import execute_planning_phase
from .gathering import execute_gathering_phase
from .analysis import execute_analysis_phase
from .synthesis import execute_synthesis_phase
from .refinement import execute_refinement_phase

class DeepResearchOrchestrator:
    async def _run_planning(self, state, timeout):
        return await execute_planning_phase(
            state, timeout, self.config, self.memory, self.hooks, ...
        )
```

**Rationale:** This is the simplest approach that achieves the goal (independent testability and navigability). It matches the codebase's existing patterns (Phase 2a uses the same approach), avoids introducing novel abstractions (no Protocols or phase classes exist elsewhere in the codebase), and each phase function is independently testable by calling it directly with mock arguments.

**Module count: 8.** Budget constants are extracted to `_constants.py` to keep `orchestrator.py` focused on sequencing. The analysis phase's digest step is extracted to `digest.py` to keep `analysis.py` focused. Prompt templates fold into their respective phase modules to co-locate templates with usage.

### Decomposition Order
Split in chunks to keep each changeset reviewable:
1. `orchestrator.py` + `_constants.py` (infrastructure + budget + crash recovery)
2. `planning.py` + `gathering.py`
3. `analysis.py` + `digest.py` + `synthesis.py`
4. `refinement.py`

### Critical Safeguards
- Preserve cancellation propagation (`CancelledError`) and timeout budgets.
- Preserve state persistence semantics and crash recovery hooks.
- Preserve concurrency patterns per `15-concurrency-patterns.md`.

### Test Migration Strategy
- Deep research tests span 2,403 lines (`test_deep_research.py`) plus 2,927 lines total across the workflows test directory.
- Update all test imports to use the new sub-module paths directly.
- Identify any tests that import internal methods (e.g., `_execute_planning_phase`, `_check_cancellation`) and update them to import from the new phase modules.
- Phase-specific tests may be added to validate individual phase function behavior, but existing integration tests remain the primary regression gate.

### Exit Criteria
- No regression in deep-research workflow tests.
- Each phase module is independently importable and testable.
- **Cohesion check:** Each module in the `deep_research/` package contains functions from a single responsibility domain.
- **Quantitative target:** Total line count across the `deep_research/` package is net-zero or lower vs. the original 6,994-line file.

---

## Phase 7: Typed Action Payloads (Stretch)

**Goal:** Reduce large function signatures and ad-hoc payload dict assembly.

### Approach
- Introduce typed payload models per action family (dataclass or Pydantic).
- Parse/validate once at router edge, pass validated models to handlers.
- Standardize validation order and error details.

### Ordering Note
If this phase is pursued, consider piloting it on a router *before* Phase 3 splits that router's handlers. Splitting already-typed handlers is cleaner than retrofitting types onto split handlers. However, this phase is stretch — if it's deferred indefinitely, Phase 3 should not wait for it.

### Exit Criteria
- For migrated routers, validation paths are centrally testable.
- Error semantics remain aligned with existing `ErrorCode` and `ErrorType` usage.

---

## Phase 8: Test Refactor and Fixture Hygiene

**Goal:** Remove remaining repetitive tests and refresh stale fixtures after structural changes.

### Tasks
- Consolidate remaining per-tool test duplication not addressed in Phase 1's dispatch dedup.
- Keep per-tool tests for tool-specific behavior only.
- Refresh stale fixtures when metadata/schema behavior changes.

### Enforcement
To prevent this from being perpetually deferred, each phase's changeset must include:
- **Deduplication of tests that are structurally modified during that phase.** "Structurally modified" means: import paths changed, function signatures changed, or test assertions updated due to the refactoring. Cosmetic import-path updates (e.g., `from core.task import X` → `from core.task.queries import X`) do NOT trigger dedup — only changes that require re-understanding the test logic.
- Updated fixtures for any modified response shapes.
- **Per-phase dedup checkpoint:** Before merging each phase, count the number of near-duplicate test functions (same assertion structure, differing only by tool name/action). Record this count in the PR description. If the count increased, explain why.

A final cleanup pass covers any remaining duplication after Phase 6.

### Exit Criteria
- Fewer duplicated test bodies with equal or better coverage.
- Target: ≥70% reduction in duplicate dispatch-exception test code (bulk achieved in Phase 1, remainder here).

---

## Execution Schedule

```
Foundation (low risk, high fan-out):
  Phase 0    Stabilization harness + baseline metrics
  Phase 1    Extract shared tool helpers + dispatch test dedup

Core module decomposition (ascending import risk):
  Phase 2a   Split task.py -> task/ package          (7 dependents)
  Phase 2b   Split spec.py -> spec/ package          (24 dependents)

Tool layer restructuring:
  Phase 3    Split task.py + authoring.py handlers
  Phase 4    Search provider utility extraction + test parametrize
  Phase 5a   Tool registration parity test

Deep research decomposition:
  Phase 6    Decompose deep_research.py -> phase modules

Polish:
  Phase 8    Final per-tool test dedup + fixture refresh

Stretch / Optional:
  Phase 1b   Standardize singleton accessors (optional)
  Phase 2c   Split config.py -> config/ package (stretch — high risk/low reward)
  Phase 5b   Tool catalog single source of truth (stretch)
  Phase 7    Typed action payloads (stretch)
```

### Why This Order
1. **Phase 0** confirms green suite and captures baselines before any changes.
2. **Phase 1** reduces noise across the entire codebase — every subsequent diff is cleaner. Dispatch test dedup happens here because `dispatch_with_standard_errors` makes it natural.
3. **Phase 2a** splits the lowest-risk god module (task, 7 dependents) first, establishing the split-and-update pattern. **2b** follows with spec (24 dependents). **2c** is demoted to stretch — 2,012 lines of mostly dataclasses with 35 dependents is high-risk/low-reward.
4. **Phase 3** requires Phase 1 helpers AND Phase 2a + 2b core splits. Confirmed via import analysis: tool handlers import directly from core modules, not via the public API surface.
5. **Phase 4** cleans providers independently (can run in parallel with Phases 2 and 3).
6. **Phase 5a** can run in parallel with Phase 4.
7. **Phase 6** tackles the largest file last. Should start after Phase 2a to benefit from the proven split pattern. Does not depend on Phase 4 — the gathering phase calls providers through the existing `SearchProvider` interface. Characterization test audit happens at the start of Phase 6, not upfront.
8. **Phases 1b, 2c, 5b, 7** are stretch/optional work deferred until core refactoring proves them necessary.
9. **Phase 8** is polish work that builds on the cleaner structure. Bulk dispatch dedup is already done in Phase 1.

---

## Dependency Graph

```
                       ┌──→ Phase 2a ─→ Phase 2b ──→ Phase 3
                       │
Phase 0 ──→ Phase 1 ──┤
                       │                  : (soft) ──→ Phase 6
                       ├──→ Phase 4
                       │
                       └──→ Phase 5a

Phase 3 + Phase 4 + Phase 5a ──→ Phase 7 (stretch)

Phase 8 runs incrementally with each phase; final pass after Phase 6.

Stretch (no ordering constraints among themselves):
  Phase 1b, Phase 2c, Phase 5b — can be done at any point after Phase 0

Legend: ──→ = hard dependency, : ··→ = soft/recommended dependency
```

**Key dependency notes:**
- **2a → 2b** are technically independent (different modules, no shared split boundaries) but executed sequentially in ascending risk order to build confidence.
- **Phase 3** hard-depends on Phase 1 (shared helpers) AND Phase 2a + 2b (core module splits). Does NOT depend on Phase 2c (stretch). Confirmed: `tools/unified/task.py` imports 12 symbols directly from `core.task`; `tools/unified/authoring.py` imports 20 symbols directly from `core.spec`. Splitting handlers before splitting core modules would require updating these imports twice.
- **Phase 6** has no hard dependencies beyond Phase 0, but should wait for Phase 2a to complete so the split pattern is proven on a simpler module first.
- **Phases 4 and 5a** can start immediately after Phase 1 and run in parallel with everything else.

---

## Branching & Parallel Execution Strategy

Phases that share no dependency edge can run concurrently on separate feature branches:

| Track | Phases | Base |
|---|---|---|
| **Core decomposition** | 2a → 2b → 3 | main (after Phase 1 merges) |
| **Providers** | 4 | main (after Phase 1 merges) |
| **Registry** | 5a | main (after Phase 1 merges) |
| **Deep research** | 6 | main (after Phase 2a merges, soft dep) |

### Rules
- Each phase is a **single feature branch** off the latest main that includes its prerequisites.
- Merge to main only after the phase's full verification protocol passes.
- Parallel tracks (e.g., Phases 3 and 4) branch from the same main state and should not touch overlapping files. If they do, the second to merge resolves conflicts.
- If a phase spans multiple PRs (e.g., Phase 2's two sub-phases), each sub-phase merges to main before the next branches off.

### Handling In-Flight Feature Work
- Refactoring branches should rebase onto latest `main` before merging.
- If new tool routers or providers are added to `main` during the refactoring, the next refactoring branch to merge is responsible for adopting them into the new patterns (e.g., migrating a new router to use Phase 1's shared helpers).
- Feature branches that are in-flight when a refactoring phase merges should rebase onto the updated `main` and adopt the new import paths.

---

## Verification Protocol

After each changeset:
1. `pytest tests/ -x --timeout=60` — full test suite.
2. Import smoke test: `python -c "from foundry_mcp.core.spec import load_spec, save_spec, get_node"` (etc. for each split).
3. Circular import check: `python -c "import foundry_mcp.server"`.
4. Contract tests: `tests/contract/test_response_schema.py`.
5. Targeted integration tests for touched modules.
6. For deep research changes: `tests/core/research/workflows/` suite.

After reverting a failed changeset, re-run the full verification protocol to confirm clean rollback.

---

## Success Metrics

Line counts and file splits are proxies. The actual goals are reduced maintenance cost and defect risk. Measure these directly:

### Primary Metrics (measured in Phase 0 baseline, re-measured after Phase 6)

| Metric | How to Measure | Target |
|---|---|---|
| **Cross-cutting edit cost** | Count files touched to add a new unified tool router (enumerate: common.py, router registration, manifest, test file). | Before: 10+ files. After: 3-4 files. |
| **Provider onboarding cost** | Count lines/functions required for a new provider (search + parse + client factory call). | Before: ~700 lines. After: ~300 lines (search + parse only, shared utilities handle the rest). |
| **Module cohesion** | Audit: does any module contain functions from distinct responsibility domains? | No module mixes distinct responsibility domains. |
| **Duplicate helper count** | Count of `_validation_error` / `_request_id` / `_metric_name` / `_resolve_specs_dir` definitions across `tools/unified/`. | Before: 28 definitions. After: 4 (in `common.py`). |
| **Test dedup ratio** | Count near-duplicate dispatch-exception test functions (same structure, differing only by tool name). | Before: baseline count. After: ≥70% reduction. |

### Secondary Metrics (tracked per-phase, warning-only)

| Metric | Threshold |
|---|---|
| Test suite runtime | >10% regression triggers investigation |
| Import time (`python -X importtime`) | >20% regression triggers investigation (run only if regressions are noticed) |
| Total source line count (src/) | Net increase >2% triggers investigation |

---

## Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| 1 | Behavioral drift in error payloads | Medium | High | Snapshot/contract assertions for error fields |
| 2 | Hidden coupling during module extraction | Medium | Medium | Add `__all__` exports to split modules; run `python -c "import foundry_mcp.server"` with `-Werror`; update all import paths in the same changeset |
| 3 | Concurrency regressions in deep research | Low | High | Characterization tests + timeout/cancellation checks |
| 4 | Tool discovery metadata drift | Medium | Medium | Parity tests between registered routers and manifest |
| 5 | Circular imports after splits | Medium | Low | Verify with import smoke tests after each split |
| 6 | Startup-time regression from eager re-exports | Low | Medium | Use lazy imports in `__init__.py` if regression noticed; run `python -X importtime` to diagnose |
| 7 | Performance regression (latency/memory) | Low | Medium | Track test suite runtime as proxy; flag >10% regression |
| 8 | Type-checker / IDE breakage from import path changes | Medium | Low | Add `__all__` to all new `__init__.py` files; verify `pyright`/`mypy` passes if configured; run IDE import resolution spot-checks on high-fan-out symbols like `load_spec` and `ServerConfig` |

## Rollback Strategy

- Each phase ships in isolated changesets; rollback at changeset granularity.
- If regression occurs, revert only the most recent changeset and retain earlier scaffolding.
- After any revert, re-run the full verification protocol to confirm clean state.

### Rollback Cascade Table

Rolling back a phase may require cascading rollbacks of dependent phases. Use this table to determine what must revert together:

| If rolling back... | Must also revert... | Reason |
|---|---|---|
| Phase 1 | Phases 3, 5a | Phase 3 handlers import shared helpers; Phase 5a parity test may reference helper-based registration |
| Phase 2a | Phase 3 (task handlers) | Task handler split depends on `core/task/` package structure |
| Phase 2b | Phase 3 (authoring handlers) | Authoring handler split depends on `core/spec/` package structure |
| Phase 3 | — | Handler splits are leaf changes |
| Phase 4 | — | Provider utilities are independent |
| Phase 5a | — | Parity test is a standalone addition |
| Phase 6 | — | Deep research decomposition is self-contained |

**Rule:** If a phase's rollback would break a dependent phase's imports or assumptions, roll back the dependent phase first (top-down), then the dependency.

---

## Import Path Policy

Since the refactoring ships as a single release, there is no transition period with dual import paths.

### Rules
1. **Each phase updates all import paths** (source + tests) to the new sub-module paths in the same changeset. No stale paths should remain after a phase merges.
2. **`core/__init__.py` re-exports** for the public API surface (`load_spec`, `save_spec`, `add_task`, etc.) are permanent — they define the package's public interface.
3. **All new `__init__.py` files** must define `__all__` for type-checker compatibility and to make the public surface explicit.

---

## Key Files Reference

| File | Lines | Role |
|---|---|---|
| `src/foundry_mcp/tools/unified/lifecycle.py` | 652 | Canonical template for tool helper pattern |
| `src/foundry_mcp/tools/unified/router.py` | 102 | ActionRouter infrastructure |
| `src/foundry_mcp/core/spec.py` | 4,116 | Largest god module to decompose (24 dependents) |
| `src/foundry_mcp/core/task.py` | 2,463 | Second god module to decompose (7 dependents) |
| `src/foundry_mcp/core/research/workflows/deep_research.py` | 6,994 | Largest single file |
| `src/foundry_mcp/core/research/providers/base.py` | 356 | SearchProvider ABC to extend |
| `src/foundry_mcp/config.py` | 2,012 | Monolithic config (stretch split — 35 dependents) |
| `src/foundry_mcp/core/observability.py` | 1,218 | Non-standard singleton to standardize |

### Unified Tool Files (21 total)

**16 tool routers:** authoring.py (3,645), environment.py (1,409), error.py (491), health.py (237), journal.py (853), lifecycle.py (652), plan.py (888), provider.py (601), pr.py (306), research.py (1,732), review.py (1,054), server.py (573), spec.py (1,295), task.py (3,887), test.py (443), verification.py (532).

**3 helper files:** context_helpers.py (97), documentation_helpers.py (268), review_helpers.py (314).

**2 infrastructure files:** `__init__.py` (88), router.py (102).

## Best-Practice Alignment (Consulted)

All changes must comply with:
- `dev_docs/codebase_standards/mcp_response_schema.md` (response contract)
- `dev_docs/mcp_best_practices/02-envelopes-metadata.md` through `15-concurrency-patterns.md`

Per CLAUDE.md triage rules, re-read relevant sections before editing any surface area.
