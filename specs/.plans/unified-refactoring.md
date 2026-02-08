# unified-refactoring

## Mission

Execute high-leverage refactors that reduce maintenance cost and defect risk without changing external behavior.

## Objective

The codebase (~34,400 lines across 38 core modules and 21 unified tool files) is architecturally sound but has accumulated duplication and god-module bloat that makes cross-cutting changes expensive. This plan eliminates duplicated helpers across 16 tool routers, splits god modules (spec.py, task.py) into focused packages, extracts shared provider utilities, and adds tool registration parity testing.

## Scope

### In Scope
- Internal module decomposition and helper extraction
- Unified router/dispatch framework consolidation
- Search provider utility extraction
- Tool registration parity testing
- God module splits (spec.py, task.py)
- Large tool handler file splits (task.py, authoring.py handlers)

### Out of Scope
- New end-user features
- Response envelope schema changes
- Provider algorithm changes (quality/behavior)
- config.py split (stretch, 35 dependents, high risk/low reward)
- deep_research.py split (stretch, evaluate after Phase 2a)

## Constraints
- Preserve `response-v2` contract and metadata semantics
- Use shared response helpers (`success_response`, `error_response`) consistently
- Maintain early input validation and actionable errors
- Preserve security boundaries, sanitization, and redaction behavior
- Keep timeout/cancellation/concurrency safeguards unchanged or improved
- Keep specs, docs, and tests synchronized in each changeset

## Compatibility Contract

**Compatibility mode: `strict-internal`** (default). All import paths are internal. No compatibility shims or deprecation windows.

**Explicit compatibility surface for module splits (Phase 2a/2b):**
- **Import paths:** `core/__init__.py` re-exports all public symbols from sub-packages unchanged. `from foundry_mcp.core.task import add_task` continues to work.
- **`__all__`:** Every new `__init__.py` defines `__all__` listing the full public API surface.
- **`__module__` stability:** Not guaranteed. No code in this codebase uses pickling/reflection on these modules. The Phase 0 consumer audit confirms this.
- **`core/__init__.py` behavior:** Preserved — same symbols re-exported to callers.
- **Consumer audit scope:** Includes `src/`, `tests/`, `docs/`, and `samples/`. A grep check (`"no direct sub-module imports remain in tests/"`) is added as a Phase 2 verification step.

## Layering Rules for Shared Modules

New shared modules must follow strict dependency rules to avoid circular imports and "god helper" sprawl:

- **`tools/unified/common.py`**: May import from `core/responses.py` and stdlib only. Must NOT import from any tool router or core domain module.
- **`providers/shared.py`**: May import from stdlib and `httpx` types only. Must NOT import from provider implementations or core domain modules. Organized by cohesion: pure parsing helpers (retry-after, error messages, dates, domains) are separate functions from parameterized patterns (error classification, resilience executor, health check, settings resolution).

## Canonical Execution Sequence

```
Phase 1: Baseline & Green Suite
Phase 2: Extract Shared Tool Helpers + Dispatch Test Dedup
Phase 3: Split task.py into task/ Package
Phase 4: Split spec.py into spec/ Package
Phase 5: Provider Utility Extraction
Phase 6: Split Large Tool Handler Files
Phase 7: Tool Registration Parity Test
```

**Dependency graph:**
```
Phase 1 ──→ Phase 2 ──┬──→ Phase 3 ──→ Phase 4
                       ├──→ Phase 5 (parallel with 3)
                       ├──→ Phase 6 (soft dep on 3/4 for cleaner imports)
                       └──→ Phase 7 (parallel with others)
```

**Parallelism after Phase 2:** Phases 3, 5, 6, and 7 can run on separate branches. Phase 4 starts after Phase 3 (ascending risk order). When parallel branches touch overlapping files, the second to merge resolves conflicts and re-runs verification.

**MVR (Minimum Viable Refactoring):** Phases 1-5 complete with green tests.
**Full completion:** MVR plus Phases 6 and 7.

## Phases

### Phase 1: Baseline & Green Suite

**Purpose**: Confirm green test suite and capture baseline metrics before moving any code. Establishes the foundation all subsequent phases depend on.

**Tasks**:
1. Run full test suite and confirm green: `pytest tests/ -x` [category: verification]
2. Capture baseline metrics in `dev_docs/refactoring/baseline.md`: module line counts for key files, duplicate helper inventory (count of `_validation_error`, `_request_id`, `_metric_name`, `_resolve_specs_dir` definitions per file with exact locations), metric name inventory per router, key log fields per dispatch function [category: investigation, file: dev_docs/refactoring/baseline.md]
3. Import consumer audit in `dev_docs/refactoring/import_consumer_audit.md`: search for internal-path imports of `foundry_mcp.core.spec`, `foundry_mcp.core.task`, `foundry_mcp.tools.unified.*` across src/, tests/, docs/, and samples/; classify as internal-only or externally documented; confirm no pickling/reflection usage on target modules; record `strict-internal` compatibility mode [category: investigation, file: dev_docs/refactoring/import_consumer_audit.md]
4. Verify phase completion: full test suite green, baseline.md committed, import_consumer_audit.md committed [category: verification]

**Verification**: Full test suite green. `baseline.md` committed with line counts, duplicate helper inventory, metric name inventory, and log field inventory. `import_consumer_audit.md` committed with compatibility mode decision and tests/ coverage.

### Phase 2: Extract Shared Tool Helpers + Dispatch Test Dedup

**Purpose**: Remove ~500 lines of duplicated helpers across 16 tool routers and consolidate dispatch-exception test boilerplate. Highest-leverage mechanical extraction.

**Tasks**:
1. Create `src/foundry_mcp/tools/unified/common.py` with shared helpers: `build_request_id()`, `make_metric_name()`, `resolve_specs_dir()`, `dispatch_with_standard_errors()`, `make_validation_error_fn()`. Layering: imports only from `core/responses.py` and stdlib [category: implementation, file: src/foundry_mcp/tools/unified/common.py]
2. Contract test gap audit: verify coverage for all 16 tool response shapes before migrating [category: investigation]
3. Changeset A (pilot): Migrate 4 routers (`task.py`, `authoring.py`, `spec.py`, `server.py`) to shared helpers — prove parity. Assert metric names and key log fields match Phase 1 baseline per router [category: refactoring, files: src/foundry_mcp/tools/unified/task.py, src/foundry_mcp/tools/unified/authoring.py, src/foundry_mcp/tools/unified/spec.py, src/foundry_mcp/tools/unified/server.py]
4. Changeset B (completion): Migrate remaining 12 routers to shared helpers; remove all duplicated definitions [category: refactoring]
5. Add telemetry invariant tests: assert metric names and `request_id`/`details` inclusion per router match baseline [category: implementation]
6. Create shared parametrized test fixtures for dispatch error paths (ActionRouterError, unexpected Exception). Keep representative full-envelope snapshots. Per-tool tests retain only tool-specific assertions [category: implementation, file: tests/tools/unified/test_dispatch_common.py]
7. Verify: all 16 routers on shared helper + dispatch wrapper paths, duplicated helper definitions removed, >=50% dispatch-exception tests consolidated, 15-25% boilerplate reduction, telemetry invariants green [category: verification]

**Verification**: All 16 routers on shared helper + dispatch wrapper paths. Duplicated helper definitions removed. >=50% dispatch-exception tests consolidated. 15-25% boilerplate reduction. Telemetry invariant tests (metric names, request_id/details inclusion) pass. Golden/contract assertions for full error envelopes on representative routers.

### Phase 3: Split task.py into task/ Package

**Purpose**: Break `core/task.py` (2,463 lines, 7 dependents) into focused, independently testable sub-modules. Lowest-risk god module split — do first to establish the pattern.

**Tasks**:
1. Public API parity audit: add a contract test validating package-level imports for task module against Phase 1 baseline [category: investigation]
2. Create `src/foundry_mcp/core/task/` package with `__init__.py` re-exporting full public API and defining `__all__` [category: refactoring, file: src/foundry_mcp/core/task/__init__.py]
3. Extract `_helpers.py`: `_get_phase_for_node()`, shared constants (`TASK_TYPES`, `REQUIREMENT_TYPES`) (~80 lines) [category: refactoring, file: src/foundry_mcp/core/task/_helpers.py]
4. Extract `queries.py`: is_unblocked, is_in_current_phase, get_next_task, check_dependencies, get_*_context, prepare_task (~820 lines) [category: refactoring, file: src/foundry_mcp/core/task/queries.py]
5. Extract `mutations.py`: add_task, remove_task, move_task, update_estimate, manage_task_dependency, update_task_metadata, update_task_requirements (~1,260 lines) [category: refactoring, file: src/foundry_mcp/core/task/mutations.py]
6. Extract `batch.py`: batch_update_tasks, _match_tasks_for_batch (~300 lines) [category: refactoring, file: src/foundry_mcp/core/task/batch.py]
7. Update all 7 consumer imports to new sub-module paths in same changeset [category: refactoring]
8. Grep check: confirm no direct `core.task` (old monolith) imports remain in src/ or tests/ [category: verification]
9. Verify: `python -c "import foundry_mcp.server"` succeeds, full test suite passes, package-level import parity against baseline [category: verification]

**Verification**: All source imports updated. No caller behavior changes. `python -c "import foundry_mcp.server"` succeeds. Full test suite passes. Package-level import parity. Grep confirms no stale direct imports.

### Phase 4: Split spec.py into spec/ Package

**Purpose**: Break `core/spec.py` (4,116 lines, 24 dependents) into focused sub-modules. Higher risk due to fan-out — do after Phase 3 establishes the pattern.

**Tasks**:
1. Public API parity audit: add contract test for spec module package-level imports against baseline [category: investigation]
2. Identify shared hierarchy/traversal helpers that overlap between spec and task modules; extract to shared location or explicitly defer with rationale [category: investigation]
3. Create `src/foundry_mcp/core/spec/` package with `__init__.py` re-exporting full public API and defining `__all__` [category: refactoring, file: src/foundry_mcp/core/spec/__init__.py]
4. Extract `_constants.py`: `TEMPLATES`, `CATEGORIES`, `VERIFICATION_TYPES`, `PHASE_TEMPLATES`, backup constants (~40 lines) [category: refactoring, file: src/foundry_mcp/core/spec/_constants.py]
5. Extract `io.py`: find_specs_directory, find_spec_file, load_spec, save_spec, backup ops, schema helpers (~800 lines) [category: refactoring, file: src/foundry_mcp/core/spec/io.py]
6. Extract `hierarchy.py`: get_node, update_node, add/remove/move phase, recalculate hours (~1,500 lines) [category: refactoring, file: src/foundry_mcp/core/spec/hierarchy.py]
7. Extract `templates.py`: create_spec, apply_phase_template, generate_spec_id, assumptions, frontmatter (~780 lines) [category: refactoring, file: src/foundry_mcp/core/spec/templates.py]
8. Extract `analysis.py`: check_spec_completeness, detect_duplicate_tasks, diff_specs, list_specs (~800 lines) [category: refactoring, file: src/foundry_mcp/core/spec/analysis.py]
9. Update all 24 consumer imports to new sub-module paths in same changeset [category: refactoring]
10. Grep check: confirm no direct `core.spec` (old monolith) imports remain in src/ or tests/ [category: verification]
11. Verify circular dependency freedom: DAG must be `_constants <- io <- hierarchy <- templates`, `analysis <- io` [category: verification]
12. Verify: `python -c "import foundry_mcp.server"` succeeds, full test suite passes, package-level import parity [category: verification]

**Verification**: All source imports updated. No caller behavior changes. Clean DAG (no circular imports). Full test suite passes. Package-level import parity. Grep confirms no stale direct imports.

### Phase 5: Provider Utility Extraction

**Purpose**: Eliminate ~1,416 lines of duplicated boilerplate across 5 HTTP-backed research providers (41% of provider code).

**Tasks**:
1. Add characterization tests/snapshots per provider for: error classification behavior (all HTTP status codes), Retry-After parsing, timeout/cancellation propagation, API key resolution, client lifecycle invariants [category: implementation]
2. Create `src/foundry_mcp/core/research/providers/shared.py` with shared utilities organized by cohesion — pure parsing helpers: `parse_retry_after()`, `extract_error_message()`, `parse_iso_date()`, `extract_domain()`; parameterized patterns: `classify_http_error()`, `create_resilience_executor()`, `check_provider_health()`, `resolve_provider_settings()`. Layering: imports only from stdlib and `httpx` types [category: implementation, file: src/foundry_mcp/core/research/providers/shared.py]
3. Migrate tavily.py to shared utilities [category: refactoring, file: src/foundry_mcp/core/research/providers/tavily.py]
4. Migrate perplexity.py to shared utilities [category: refactoring, file: src/foundry_mcp/core/research/providers/perplexity.py]
5. Migrate google.py to shared utilities (note: has custom quota handling via `custom_classifier` parameter) [category: refactoring, file: src/foundry_mcp/core/research/providers/google.py]
6. Migrate semantic_scholar.py to shared utilities [category: refactoring, file: src/foundry_mcp/core/research/providers/semantic_scholar.py]
7. Migrate tavily_extract.py to shared utilities [category: refactoring, file: src/foundry_mcp/core/research/providers/tavily_extract.py]
8. Assert characterization test parity: all pre-refactor snapshots match post-migration behavior [category: verification]
9. Create parametrized shared test fixtures for providers [category: implementation]
10. Verify: ~800 lines removed, all provider tests pass, characterization parity confirmed, client lifecycle preserved [category: verification]

**Verification**: ~800 lines removed from provider implementations. All provider tests pass. Characterization tests confirm parity for error classification, retry, timeout/cancellation, API key resolution. Shared helpers do not create/cache httpx.AsyncClient instances.

### Phase 6: Split Large Tool Handler Files

**Purpose**: Break the two largest unified tool files into domain-focused handler modules for better navigability and testability.

**Tasks**:
1. Create `src/foundry_mcp/tools/unified/task_handlers/` package: `__init__.py` (router build + registration, <=50 lines), `handlers_lifecycle.py` (start, complete, block, unblock), `handlers_batch.py` (prepare-batch, start-batch), `handlers_query.py` (list, query, hierarchy, progress), `handlers_mutation.py` (add, remove, move, metadata/deps) [category: refactoring, file: src/foundry_mcp/tools/unified/task_handlers/__init__.py]
2. Create `src/foundry_mcp/tools/unified/authoring_handlers/` package: `__init__.py` (router build + registration, <=50 lines), `handlers_spec.py` (spec CRUD), `handlers_phase.py` (phase management), `handlers_metadata.py` (metadata ops), `handlers_intake.py` (intake queue ops) [category: refactoring, file: src/foundry_mcp/tools/unified/authoring_handlers/__init__.py]
3. Update imports and registrations [category: refactoring]
4. Verify: existing tests pass, public tool names/signatures preserved, action names preserved, telemetry/audit hooks preserved, each `__init__.py` <=50 lines [category: verification]

**Verification**: Existing tests pass. Public tool names/signatures and action names preserved. Telemetry and audit hooks preserved. Each `__init__.py` <=50 lines.

### Phase 7: Tool Registration Parity Test

**Purpose**: Prevent drift between runtime tool registration and manifest output.

**Tasks**:
1. Add parity test asserting manifest output matches registered routers for: tool count + names, version, category, tags, description, action summaries/aliases [category: implementation, file: tests/tools/unified/test_tool_registration_parity.py]
2. Remove stale fixed counts in comments (e.g., "15-tool") [category: refactoring]
3. Verify: parity test added and passing with metadata coverage (not just count/names) [category: verification]

**Verification**: Parity test added and passing with metadata coverage (not just count/names).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Error payload drift after helper extraction | High | Contract assertions for full error envelopes; telemetry invariant tests for metric names and log fields |
| Hidden coupling in module splits | Medium | `__all__` exports; import smoke tests; same-changeset path updates; grep checks for stale imports |
| Tool metadata drift | Medium | Parity test (Phase 7) |
| Circular imports from splits | Low | `python -c "import foundry_mcp.server"` after each split |
| Startup-time regression | Medium | Lazy imports in `__init__.py` if noticed |
| Telemetry regressions from dispatch consolidation | Medium | Metric name + log field invariant tests per router before/after |
| Provider reliability regression | High | Pre-refactor characterization tests/snapshots, post-migration parity assertions |
| God-module split hard to revert | Medium | Each phase merges as discrete commits; revert merge commit(s) if regressions found; Phase 3 before Phase 4 establishes pattern at lower risk |
| Cross-module duplicated helpers spread during splits | Low | Phase 4 task 2 explicitly audits shared hierarchy utilities; extract or defer with rationale |

## Dependencies

- Phase 1 -> Phase 2 (hard)
- Phase 2 -> Phase 3 (hard)
- Phase 3 -> Phase 4 (hard, ascending risk order)
- Phase 2 -> Phase 5 (hard, can run parallel with 3)
- Phase 2 -> Phase 6 (hard, soft dep on 3/4 for cleaner imports)
- Phase 2 -> Phase 7 (hard, can run parallel with others)

## Success Criteria

- [ ] Duplicate helper definitions reduced from 28 to 4 (in common.py)
- [ ] Provider onboarding cost reduced from ~700 lines to ~300 lines per provider
- [ ] No module mixes responsibility domains (spec.py, task.py split)
- [ ] >=50% dispatch test duplication reduction
- [ ] Telemetry invariants (metric names, request_id/details) preserved across all routers
- [ ] Provider characterization tests confirm parity post-migration
- [ ] Test suite runtime regression <10%
- [ ] Total source lines (src/) net increase <2%
- [ ] All phases pass verification protocol: pytest, import smoke test, contract tests, fixture freshness
