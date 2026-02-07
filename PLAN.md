# Unified Refactoring Plan for foundry-mcp

## Objective

Execute high-leverage refactors that reduce maintenance cost and defect risk without changing external behavior. The codebase (~34,400 lines across 38 core modules and 21 unified tool files) is architecturally sound but has accumulated duplication and god-module bloat that makes cross-cutting changes expensive.

## Scope

### In Scope
- Internal module decomposition and helper extraction.
- Unified router/dispatch framework consolidation.
- Search provider utility extraction.
- Tool registration parity testing.
- God module splits (spec.py, task.py).

### Out of Scope
- New end-user features.
- Response envelope schema changes.
- Provider algorithm changes (quality/behavior).

## Constraints

- Preserve `response-v2` contract and metadata semantics.
- Use shared response helpers (`success_response`, `error_response`) consistently.
- Maintain early input validation and actionable errors.
- Preserve security boundaries, sanitization, and redaction behavior.
- Keep timeout/cancellation/concurrency safeguards unchanged or improved.
- Keep specs, docs, and tests synchronized in each changeset.

## Principles

- **Behavior-preserving by default.** When behavior changes are unavoidable, gate by explicit versioning and updated contract tests.
- **Extract only when duplication appears in 3+ places.** Don't create abstractions for hypothetical reuse.
- **Incremental changesets with passing tests.** Never leave the codebase in a broken state.
- **Mechanical before design.** Do safe, obvious extractions first to reduce noise before architectural decisions.
- **Cohesion over line count.** Split criterion is mixed responsibility domains, not line count.

## Done Definition

**Minimum viable refactoring (MVR):** Phases 0, 1, 2a, 2b, and 4 complete with green tests. This delivers the two highest-value outcomes: eliminating router duplication (28 → 4 helper definitions) and splitting the god modules into testable units.

**Full completion:** MVR plus Phases 3 and 5a, and any stretch phases justified by observed friction.

The effort can stop at MVR and still be worthwhile. Phases beyond MVR are pursued based on demonstrated need, not plan momentum.

---

## Phase 0: Baseline & Green Suite

**Goal:** Confirm green test suite and capture baseline before moving any code.

### Existing Test Infrastructure
Leverage, don't recreate:
- `tests/contract/test_response_schema.py` (510 lines) — response-v2 contract coverage.
- `tests/core/research/workflows/test_deep_research.py` (2,403 lines) — deep research workflow coverage.
- `tests/conftest.py` (304 lines) — fixture versioning, freshness validation, envelope validation.

### Tasks
1. **Run full test suite** and confirm green: `pytest tests/ -x`.
2. **Capture baseline metrics** in `dev_docs/refactoring/baseline.md`:
   - Module line counts for key files (see Key Files Reference).
   - Duplicate helper inventory: count of `_validation_error`, `_request_id`, `_metric_name`, `_resolve_specs_dir` definitions per file with exact locations.
3. **Import consumer audit** in `dev_docs/refactoring/import_consumer_audit.md`:
   - Search for internal-path imports of `foundry_mcp.core.spec`, `foundry_mcp.core.task`, `foundry_mcp.tools.unified.*` across docs, samples, and tests.
   - Classify each as internal-only or externally documented.
   - Record compatibility mode decision: `strict-internal` (default) or `compatibility-window`.

### Deferred Audits
These happen when their dependent phases begin, not upfront:
- **Contract test gap audit** (before Phase 1): verify coverage for all 16 tool response shapes.
- **Public API parity audit** (before Phase 2): add a contract test validating package-level imports for split modules against baseline.
- **Deep research characterization audit** (before Phase 6, if pursued): confirm tests cover status/report/cancel/resume.

### Exit Criteria
- Full test suite green.
- `baseline.md` committed with line counts and duplicate helper inventory.
- `import_consumer_audit.md` committed with compatibility mode decision.

---

## Phase 1: Extract Shared Tool Helpers + Dispatch Test Dedup

**Goal:** Remove ~500 lines of duplicated helpers across 16 tool routers and consolidate dispatch-exception test boilerplate.

### Problem
16 tool routers each duplicate:
- `_validation_error()` — 10 files, differing only by tool name
- `_request_id()` — 8 files, differing only by prefix
- `_metric_name()` — 5 files, differing only by prefix
- `_resolve_specs_dir()` — 5 files, two return-shape variants
- `_dispatch_*_action()` try/except — all 16 routers, ~25 lines each

### Solution
Create `src/foundry_mcp/tools/unified/common.py`:

```python
build_request_id(tool_name: str) -> str
make_metric_name(prefix: str, action: str) -> str

resolve_specs_dir(config, path_or_workspace) -> Tuple[Optional[Path], Optional[dict]]
# Single function. Callers that only need the path: `path, _ = resolve_specs_dir(...)`

dispatch_with_standard_errors(
    router, tool_name, action, *,
    include_request_id: bool,
    include_details_in_router_error: bool = False,
    **kwargs,
) -> dict
```

**Per-tool validation factories** to reduce call-site verbosity:
```python
def make_validation_error_fn(tool_name: str, *, include_request_id: bool) -> Callable:
    """Returns a tool-specific validation_error with baked-in tool_name and request_id policy."""
```
Each router calls `make_validation_error_fn` once at module level. This bakes in tool identity and request-id policy while keeping call sites clean (`validation_error(field=..., action=..., message=...)` instead of repeating `tool_name` and `include_request_id` at every call site).

**Dispatch audit summary:** 13/16 routers follow an identical two-level try/except (ActionRouterError → Exception). 9/16 include `request_id`; 2/16 include `details` in ActionRouterError handler. All behavioral divergence lives in action handlers, not dispatch. See Appendix A for details.

### Adoption Strategy
- **Changeset A (pilot):** 4 routers (`task.py`, `authoring.py`, `spec.py`, `server.py`) — prove parity.
- **Changeset B (completion):** remaining 12 routers; remove all duplicated definitions.

### Dispatch Test Dedup
With `dispatch_with_standard_errors` extracted, create shared parametrized test fixtures for dispatch error paths (ActionRouterError, unexpected Exception). Keep representative full-envelope snapshots so message/detail regressions are caught. Per-tool tests retain only tool-specific assertions.

### Risks & Mitigations
- **Error wording drift:** Golden/contract assertions for full error envelopes on representative routers.
- **Observability drift:** Parity assertions for metric naming and `request_id`/`details` inclusion per router.

### Exit Criteria
- All 16 routers on shared helper + dispatch wrapper paths.
- Duplicated helper definitions removed from all routers.
- ≥50% dispatch-exception tests consolidated into parametrized shared tests.
- Target: 15-25% boilerplate reduction across migrated routers.

**Nature:** Mechanical extraction. No behavior change.

---

## Phase 2: Split God Modules into Packages

**Goal:** Break the two largest core modules into focused, independently testable sub-modules.

### Shared Policies

**Re-exports:**
1. Each package `__init__.py` re-exports the **full public API** so existing imports work.
2. `core/__init__.py` re-exports the same symbols it does today — no public surface changes.
3. All new `__init__.py` files define `__all__`.

**Circular dependency resolution** (priority order):
1. Merge coupled functions into the same sub-module.
2. Extract shared types into `_types.py`.
3. Runtime imports as last resort.

**Import path policy:** Default `strict-internal` (no compatibility shims). If Phase 0 audit selects `compatibility-window`, keep temporary re-exports for one release with deprecation notice.

**Test updates:** Black-box/contract tests keep stable public imports. Internal-behavior tests update to sub-module imports when internals move.

### 2a: Split `core/task.py` (2,463 lines) → `core/task/` Package

**Import fan-out: 7 files** — lowest risk, split first.

Call graph validated: no mutation→query or query→mutation calls. One shared helper (`_get_phase_for_node`) extracted to `_helpers.py`. batch.py fully independent. See Appendix B.

| Module | Contents | ~Lines |
|---|---|---|
| `task/_helpers.py` | `_get_phase_for_node()`, shared constants (`TASK_TYPES`, `REQUIREMENT_TYPES`) | ~80 |
| `task/queries.py` | is_unblocked, is_in_current_phase, get_next_task, check_dependencies, get_*_context, prepare_task | ~820 |
| `task/mutations.py` | add_task, remove_task, move_task, update_estimate, manage_task_dependency, update_task_metadata, update_task_requirements | ~1,260 |
| `task/batch.py` | batch_update_tasks, _match_tasks_for_batch | ~300 |

### 2b: Split `core/spec.py` (4,116 lines) → `core/spec/` Package

**Import fan-out: 24 files** — moderate risk.

Dependency graph validated as clean DAG: `_constants ← io ← hierarchy ← templates`, `analysis ← io`. No circular dependencies. See Appendix B.

| Module | Contents | ~Lines |
|---|---|---|
| `spec/_constants.py` | `TEMPLATES`, `CATEGORIES`, `VERIFICATION_TYPES`, `PHASE_TEMPLATES`, backup constants | ~40 |
| `spec/io.py` | find_specs_directory, find_spec_file, load_spec, save_spec, backup ops, schema helpers | ~800 |
| `spec/hierarchy.py` | get_node, update_node, add/remove/move phase, recalculate hours | ~1,500 |
| `spec/templates.py` | create_spec, apply_phase_template, generate_spec_id, assumptions, frontmatter | ~780 |
| `spec/analysis.py` | check_spec_completeness, detect_duplicate_tasks, diff_specs, list_specs | ~800 |

**Note:** hierarchy.py at ~1,500 lines is dense but cohesive (single responsibility domain). Further splitting into `spec/phases.py` is an implementation-time judgment, not a plan decision.

### Naming Note
Phase 2a produces `core/task/` (domain logic). Phase 3 produces `tools/unified/task_handlers/` (tool routing). The `_handlers` suffix disambiguates.

### Exit Criteria
- All source imports updated to new sub-module paths in the same changeset.
- No caller behavior changes.
- `python -c "import foundry_mcp.server"` succeeds (no circular imports).
- Full test suite passes.

---

## Phase 4: Provider Utility Extraction

**Goal:** Eliminate duplicated boilerplate across 5 HTTP-backed research providers.

### Problem
5 providers (tavily.py 675, perplexity.py 730, google.py 651, semantic_scholar.py 673, tavily_extract.py 734 — 3,463 lines total) each duplicate:
- Init boilerplate (~40 lines): API key, base URL, resilience config
- Error classification (~74 lines): 96% identical HTTP status mapping (Google adds quota handling)
- Resilience wrapping (~91 lines): `_execute_with_retry()` nearly identical
- Parsing helpers (~48 lines): `_parse_retry_after()`, `_extract_error_message()`, etc. — identical
- Health check (~18 lines): structurally identical

Total: ~1,416 lines of boilerplate (41% of provider code).

### Solution
Create `providers/shared.py` with shared utilities:

```python
# Pure functions (identical across providers)
parse_retry_after(response) -> Optional[float]
extract_error_message(response) -> str
parse_iso_date(date_str) -> Optional[datetime]
extract_domain(url) -> Optional[str]

# Parameterized patterns
classify_http_error(error, provider_name, custom_classifier=None) -> ErrorClassification
create_resilience_executor(provider_name, config, classify_error) -> Callable
check_provider_health(provider_name, api_key, base_url) -> dict
resolve_provider_settings(provider_name, env_key, api_key, base_url, ...) -> ProviderSettings
```

**Implementation flexibility:** Utility functions are the recommended default. However, if during implementation the utilities are always called in the same sequence with the same lifecycle, a base class or mixin that enforces correct ordering is an acceptable alternative. The goal is eliminating duplication; the mechanism is an implementation decision.

Each provider keeps its own `search()`/`extract()`, request construction, and `_parse_response()`.

### Async Client Lifecycle (Non-Negotiable)
Shared helpers must not create or cache `httpx.AsyncClient` instances. Providers retain client lifecycle ownership.

### Reliability Parity
- Preserve retryability classification, `Retry-After` parsing, timeout/cancellation propagation.
- Preserve API key resolution semantics and error codes/messages.
- Preserve client lifecycle (no leaked resources).

### Exit Criteria
- ~800 lines removed from provider implementations.
- All provider tests pass with parametrized shared fixtures.
- New provider requires only: `search()`/`extract()`, `_parse_response()`, `get_provider_name()` + shared utilities.

---

## Phase 3: Split Large Tool Handler Files (Recommended)

**Goal:** Break the two largest unified tool files into domain-focused handler modules.

**Dependencies:** Requires Phase 1 (shared helpers). Soft dependency on Phase 2a/2b — the tool handlers import ~32 symbols directly from `core.task` and `core.spec`. If Phase 2 hasn't landed, those imports get updated twice (once for the handler split, once when core splits). That's minor mechanical work, not a schedule blocker.

### Proposed Structure

**`tools/unified/task_handlers/`** (from `task.py`, 3,887 lines)
| Module | Contents |
|---|---|
| `__init__.py` | Router build + registration entry point |
| `handlers_lifecycle.py` | start, complete, block, unblock |
| `handlers_batch.py` | prepare-batch, start-batch |
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

### Exit Criteria
- Existing tests pass.
- Public tool names/signatures and action names preserved.
- Telemetry and audit hooks preserved.
- Each `__init__.py` ≤50 lines.

---

## Phase 5a: Tool Registration Parity Test (Recommended)

**Goal:** Prevent drift between runtime tool registration and manifest output.

### Tasks
- Add parity test asserting manifest output matches registered routers for: tool count + names, version, category, tags, description, action summaries/aliases.
- Remove stale fixed counts in comments (e.g., "15-tool").

### Exit Criteria
- Parity test added and passing with metadata coverage (not just count/names).

---

## Stretch Phases

Pursued only if observed maintenance friction justifies them after core + recommended phases land.

### Phase 5b: Tool Catalog Single Source of Truth
Centralize tool metadata into `tools/unified/catalog.py` used by both registration and manifest generation. Requires Phase 5a. **Low priority:** the parity test from 5a already catches drift; the catalog adds a layer of indirection for marginal benefit.

### Phase 6: Decompose `deep_research.py` (6,994 lines)
Split into `research/workflows/deep_research/` package with 6 modules:

| Module | ~Lines |
|---|---|
| `orchestrator.py` | ~1,150 (class, phase sequencing, state, crash recovery, budget constants) |
| `planning.py` | ~650 (planning phase, query decomposition, planning prompts) |
| `gathering.py` | ~850 (gathering phase, extract followup, gathering prompts) |
| `analysis.py` | ~1,550 (analysis phase + digest step, analysis prompts) |
| `synthesis.py` | ~850 (synthesis phase, report generation, synthesis prompts) |
| `refinement.py` | ~650 (refinement phase, gap analysis, refinement prompts) |

Extract as standalone functions the orchestrator calls directly — same pattern as Phase 2a. No Protocol classes or PhaseContext abstractions.

**Critical safeguards:** Preserve cancellation propagation (`CancelledError`), timeout budgets, state persistence, and concurrency patterns.

**Evaluate after Phase 2a** establishes the split pattern on a simpler module. Consider whether the file's maintenance friction actually warrants splitting — if the phases are tightly coupled and the file is rarely edited by multiple people, better in-file organization (region markers, consistent ordering) may suffice.

### Phase 1b: Standardize Singleton Accessors
Standardize 3 singleton mechanisms to `get_X()` lazy-init pattern. Touches startup-order-sensitive code for zero external ROI.

### Phase 2c: Split `config.py` (2,012 lines)
Mostly dataclasses with 35 dependents. High risk, low reward.

### Phase 7: Typed Action Payloads
Typed payload models per action family (dataclass or Pydantic). Optional early pilot after Phase 1.

### Phase 8: Test Cleanup
Final pass to consolidate remaining test duplication and refresh stale fixtures. Target: ≥70% reduction in duplicate dispatch-exception test code (bulk achieved in Phase 1).

---

## Execution Schedule

```
Core (minimum viable refactoring):
  Phase 0    Baseline & green suite
  Phase 1    Extract shared tool helpers + dispatch test dedup
  Phase 2a   Split task.py → task/ package        (7 dependents)
  Phase 2b   Split spec.py → spec/ package        (24 dependents)
  Phase 4    Provider utility extraction

Recommended (after core):
  Phase 3    Split task.py + authoring.py handlers
  Phase 5a   Tool registration parity test

Stretch (evaluate based on observed friction):
  Phase 5b, 6, 1b, 2c, 7, 8
```

## Dependency Graph

```
Phase 0 ──→ Phase 1 ──┬──→ Phase 2a ──→ Phase 2b
                       ├──→ Phase 4
                       ├──→ Phase 3  (soft dep on 2a/2b)
                       └──→ Phase 5a ──→ Phase 5b (stretch)

Phase 2a ··→ Phase 6 (stretch, soft dep)

Phase 8 runs as final cleanup after active phases complete.

Legend: ──→ hard dependency, ··→ soft/recommended
```

**Parallelism:** After Phase 1 merges, Phases 2a, 4, 3, and 5a can run on separate branches. Phase 2b starts after 2a (ascending risk order). When parallel branches touch overlapping files, the second to merge resolves conflicts and re-runs verification.

## Branching Strategy

| Track | Phases | Base |
|---|---|---|
| **Core decomposition** | 2a → 2b | main (after Phase 1) |
| **Providers** | 4 | main (after Phase 1) |
| **Tool handlers** | 3 | main (after Phase 1; cleaner if after 2a/2b) |
| **Registry** | 5a | main (after Phase 1) |

- Each phase is a single feature branch off latest main with prerequisites merged.
- Merge only after verification passes on the branch tip.
- If rebased or conflict-resolved, re-run verification before merge.
- New routers/providers added to main during refactoring are adopted by the next phase to merge.

---

## Verification Protocol

After each phase, before merge:

1. `pytest tests/ -x` — full test suite.
2. `python -c "import foundry_mcp.server"` — circular import / startup check.
3. `tests/contract/test_response_schema.py` — contract tests.
4. Import smoke tests for touched modules (e.g., `from foundry_mcp.core.spec import load_spec`).
5. Fixture freshness check — stale warnings are blockers.

**Phase-specific checks:**
- **Phase 1:** Dispatch envelope parity for representative routers (message format, `data.details`, error codes).
- **Phase 2:** Package-level import parity against Phase 0 baseline.
- **Phase 4:** Provider reliability parity (retry classification, timeout/cancellation, auth/init errors, client lifecycle).
- **Phase 6:** `tests/core/research/workflows/` suite.

Record verification results (commit SHA, pass/fail) in the PR description.

## Rollback Strategy

- Each phase merges to main as a discrete set of commits.
- If a phase introduces regressions not caught by tests, revert the merge commit(s).
- Parallel branches that already branched from post-regression main rebase onto reverted main.
- Partial rollback (reverting sub-changesets within a phase) is acceptable when the phase's intermediate changesets are independently valid.

---

## Success Metrics

| Metric | Before | After (Target) | Measured At |
|---|---|---|---|
| **Duplicate helper definitions** | 28 across unified/ | 4 (in common.py) | Phase 1 exit |
| **Provider onboarding cost** | ~700 lines per provider | ~300 lines | Phase 4 exit |
| **Module cohesion** | spec.py, task.py mix domains | No module mixes domains | Phase 2 exit |
| **Dispatch test duplication** | Baseline (Phase 0) | ≥50% reduction | Phase 1 exit |

### Secondary (warning-only)

| Metric | Threshold |
|---|---|
| Test suite runtime | >10% regression triggers investigation |
| Total source lines (src/) | Net increase >2% triggers investigation |

---

## Risk Register

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| 1 | Error payload drift | M | H | Contract assertions for full error envelopes |
| 2 | Hidden coupling in splits | M | M | `__all__` exports; import smoke tests; same-changeset path updates |
| 3 | Concurrency regressions (Phase 6) | L | H | Characterization tests + cancellation checks |
| 4 | Tool metadata drift | M | M | Parity test (Phase 5a) |
| 5 | Circular imports | M | L | `python -c "import foundry_mcp.server"` after each split |
| 6 | Startup-time regression | L | M | Lazy imports in `__init__.py` if noticed |

---

## Key Files Reference

All paths relative to `src/foundry_mcp/`.

| File | Lines | Role |
|---|---|---|
| `tools/unified/lifecycle.py` | 652 | Canonical template for tool helper pattern |
| `tools/unified/router.py` | 102 | ActionRouter infrastructure |
| `core/spec.py` | 4,116 | God module (24 dependents) |
| `core/task.py` | 2,463 | God module (7 dependents) |
| `core/research/workflows/deep_research.py` | 6,994 | Largest single file |
| `core/research/providers/base.py` | 356 | SearchProvider ABC |
| `config.py` | 2,012 | Monolithic config (stretch — 35 dependents) |

### Unified Tool Routers (16)
authoring.py (3,645), task.py (3,887), research.py (1,732), environment.py (1,409), spec.py (1,295), review.py (1,054), plan.py (888), journal.py (853), lifecycle.py (652), provider.py (601), server.py (573), verification.py (532), error.py (491), test.py (443), pr.py (306), health.py (237).

---

## Appendix A: Dispatch Function Audit

All 16 dispatch functions audited. 13/16 follow identical two-level try/except (ActionRouterError → Exception). Non-standard routers:
- **health.py** — unique signature (`include_details` kwarg instead of `payload`/`config`), standard error handling. Pre-processes kwargs before dispatch.
- **plan.py / pr.py** — simplified signatures (no `config`), otherwise identical.
- **research.py** — standard dispatch; complex recovery lives in action handlers (provider-specific `AuthenticationError`, `RateLimitError`, `UrlValidationError`), not in dispatch.

Request-id inclusion: 9/16. Details in ActionRouterError: 2/16. Both parameterized via `dispatch_with_standard_errors`.

## Appendix B: Call Graph Summaries

### task.py
- No mutation→query or query→mutation calls.
- One shared helper: `_get_phase_for_node` (queries + mutations) → `_helpers.py`.
- batch.py fully independent (no calls to queries or mutations).
- Private helper clustering validated — see Phase 2a table for assignment.

### spec.py
Clean DAG with no circular dependencies:
```
_constants  (no deps)
     ↑
   io       (imports _constants)
     ↑
hierarchy   (imports io, _constants)
     ↑
templates   (imports io, hierarchy, _constants)

analysis    (imports io only)
```
- hierarchy → io: phase operations need I/O (expected).
- templates → hierarchy: single call `apply_phase_template()` → `add_phase_bulk()` (not circular).
- analysis → io: read-only (never calls `save_spec()`).

## Best-Practice Alignment

All changes must comply with:
- `dev_docs/codebase_standards/mcp_response_schema.md` (response contract)
- `dev_docs/mcp_best_practices/02-envelopes-metadata.md` through `15-concurrency-patterns.md`

Per CLAUDE.md triage rules, re-read relevant sections before editing any surface area.
