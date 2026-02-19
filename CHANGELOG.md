# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0b14] - 2026-02-19

### Changed

- **Rebrand SDD to Foundry**: Renamed all SDD references to Foundry across the codebase, including CLI commands (`cast run`, `cast stop`, `cast watch`), skill prefixes, and documentation.
- **Type safety overhaul**: Added ruff and pyright tooling; fixed all 573 type errors to zero across the codebase.

### Fixed

- **Agent-facing validation UX**: Aggregate validation errors into a single response instead of failing on the first error. Protect step proofs from being consumed on validation failure. Verification command fallback when no command is configured.

### Added

- **Specs directory**: Added spec definitions, autonomy session records, and plan review artifacts.

## [0.12.0b13] - 2026-02-18

### Fixed

- **Verification gate ignoring pending gate evidence**: `_should_run_fidelity_gate()` could re-emit a fidelity gate step while `pending_gate_evidence` was still waiting for Step 13 (`_handle_gate_evidence`) to process it. Now defers when evidence is pending, allowing auto-retry routing to `address_fidelity_feedback` to proceed correctly.
- **Spec-status fallback polluting session verification evidence**: `_find_next_verification()` and `_should_run_fidelity_gate()` fell back to `task.get("status") == "completed"` from the spec when a verification wasn't in `session.completed_task_ids`. This let stale completions from prior sessions count as evidence for the current session's fidelity gate. Now only `session.completed_task_ids` is authoritative.

### Added

- **`session-step-heartbeat` in autonomy runner allowlist**: Enables autonomous agents to send heartbeats during long operations (e.g., fidelity gate reviews ~90s each), preventing `heartbeat_stale` pauses.

### Changed

- **Step handler docs**: Unified all step types to always use the extended `last_step_result` envelope (removed simple `command="report"` transport). Added concrete JSON examples with required fields for every step type. Documented heartbeat protocol.

## [0.12.0b12] - 2026-02-18

### Fixed

- **Proof deadlock on gate invariant errors**: When the orchestrator consumed a step proof (Step 3) but a later pipeline stage (e.g. `REQUIRED_GATE_UNSATISFIED` at Step 17) returned an error with `should_persist=True`, the session was saved with `last_step_issued` still pointing to the consumed proof. The next call required a proof that no longer existed — unrecoverable deadlock. Now clears `last_step_issued` and advances `state_version` after step consumption (Step 6b), before the next-step computation pipeline.
- **Fidelity gate skipped for spec-completed verifications**: `_should_run_fidelity_gate()` only checked `session.completed_task_ids` for verification status, but when a prior session already completed verifications in the spec, the new session's `completed_task_ids` didn't include them. The gate checker said "verifications pending" while the verification finder said "nothing to issue" — deadlock between Steps 12 and 16. Now also checks `task.get("status") == "completed"` from the spec.
- **`update_task_status()` blind to phases-format specs**: Only checked `hierarchy` dict. Now falls back to `phases` array for specs using the denormalized format.
- **`_persist_task_completion_to_spec()` silently failed on phases-format specs**: `save_spec()` with default `validate=True` rejected specs without a `hierarchy` key. Now passes `validate=False` since this is a targeted status update.
- **Session reuse ignores fidelity cycle limit**: `foundry-implement-auto` would reuse a session paused at `fidelity_cycle_limit` (gate retries exhausted), then immediately hit the same wall. The skill now detects this pause reason and raises `SESSION_REUSE_PAUSED_GATE_LIMIT` with clear remediation guidance instead of entering a futile retry loop.

## [0.12.0b10] - 2026-02-18

### Fixed

- **Fidelity gate blind to verification evidence**: The fidelity gate reviewer received no meaningful verification evidence, causing gates to fail in a retry loop. Four bugs fixed:
  - `_build_journal_entries()` only rendered entry titles (not content) — the reviewer saw one-liners like "Step execute_verification: success" with zero detail
  - `_build_journal_entries()` ignored `phase_id` parameter — when the gate passed `phase_id` without `task_id`, no filtering occurred; now filters to entries from the phase's child tasks
  - `_build_test_results()` checked for `"verify"` substring but journal titles contain `"verification"` (not a match) — added `"verification"` to the keyword filter
  - `_write_step_journal()` wrote terse content for verification steps without receipt evidence — now includes exit_code, command_hash, and output_digest when a verification receipt is present

## [0.12.0b9] - 2026-02-18

### Fixed

- **Write-lock task completion deadlock**: When `write_lock_enforced=true`, the autonomous runner cannot call `task(action="complete")` (blocked by write lock), but the orchestrator also never persisted completions to the spec file. The phantom reconciler (Step 4b) would then see the spec task as still "pending", revoke the session-level completion, and re-issue the same task indefinitely. The orchestrator now calls `_persist_task_completion_to_spec()` server-side during Step 3 when write lock is enforced.
- **Phantom reconciler blind to hierarchy format**: `_get_spec_task_status()` only checked the `phases` array, but production specs use the `hierarchy` dict (no `phases` key). This caused every completion to be treated as phantom and revoked. Now checks `hierarchy` first, falling back to `phases` for test fixtures.

## [0.12.0b8] - 2026-02-18

### Fixed

- **Phantom task completions**: Added Step 4b reconciliation (`_reconcile_completed_tasks`) that cross-checks `session.completed_task_ids` against actual spec task status after spec data is loaded. When the runner self-reports success but the spec task remains pending (e.g., `task(action="complete")` was never called), the phantom completion is revoked and the task is re-issued on the next step cycle. Self-healing — no manual intervention required.
- **Proof deadlock on crash recovery**: Enhanced `session-step-replay` to detect consumed proof tokens and reissue fresh ones. Previously, if a crash occurred between proof consumption and session state update, recovery agents received a dead proof from replay and were permanently stuck (`STEP_PROOF_CONFLICT`). Now replay checks `get_proof_record()`, generates a replacement proof, and persists the updated session state.

## [0.12.0b7] - 2026-02-18

### Fixed

- **Commandless verify node deadlock**: Verify nodes without a command in metadata no longer cause unrecoverable deadlock. The orchestrator now distinguishes "no receipt was ever pending" from "receipt missing", skipping receipt validation when inapplicable while preserving the hard boundary for verify nodes that do have commands.
- **Payload key collision in session-step-report**: Consumed fields (`step_id`, `step_type`, `outcome`, etc.) are now stripped from `**payload` before delegation to `session-step-next`, preventing `TypeError: got multiple values for argument` when callers pass redundant keys.

### Added

- **`prepare` in autonomy runner allowlist**: Added read-only `prepare` action to `AUTONOMY_RUNNER_ALLOWLIST`, enabling autonomous agents to fetch task context via server-mediated authorization instead of soft prompt-based file reading.

### Changed

- **Step-handlers reference**: Added explicit outcome enum callout (`success | failure | skipped`) and clarified that `next_step.instruction` is the primary context source for `implement_task`, with `task(action="prepare")` as optional.

## [0.12.0b6] - 2026-02-18

### Fixed

- **Stale session recovery authorization**: Added `session-step-replay` to `AUTONOMY_RUNNER_ALLOWLIST`, enabling autonomous agents to probe for unreported pending steps during stale session recovery. Previously, replay was documented in the skill but blocked at the server with `AUTHORIZATION` error.

## [0.12.0b5] - 2026-02-18

### Added

- **Stale session recovery**: Replay-based recovery for autonomous sessions where a prior agent died mid-step. New agents probe for unreported pending steps via `session-step replay`, recovering the step proof without counter resets, session cycling, or gate bypass. Covers all `loop_signal` states with deterministic escalation or continuation.
- **Skill documentation**: Updated `foundry-implement-auto` flow diagram, session management reference, and step loop docs to document the full stale session recovery sequence, safety guarantees, state-specific behavior, and edge cases.

## [0.12.0b4] - 2026-02-18

### Added

- **Verification-execute authorization**: Added `verification-execute` to `AUTONOMY_RUNNER_ALLOWLIST`, enabling autonomous sessions to execute verification commands for proof-carrying receipts.
- **Hierarchy spec integration tests**: Comprehensive test suite (`TestHierarchySpecIntegration`) verifying the full orchestrator pipeline works with hierarchy-format specs (the production format), covering task discovery, phase advancement, fidelity gates, hash equivalence, subtask expansion, and multi-step progression.
- **Hierarchy spec test factory**: New `make_hierarchy_spec_data()` factory and `hierarchy_spec_factory` fixture in autonomy test conftest for building realistic production-format spec data.

## [0.12.0b3] - 2026-02-18

### Added

- **Spec adapter** (`core/autonomy/spec_adapter.py`): New module that bridges hierarchy-based spec data to the phases-array view expected by the orchestrator. Provides `load_spec_file()` as the single entry point for loading spec files in the autonomy layer, and `ensure_phases_view()` for converting hierarchy format to phases format non-destructively. Handles nested subtasks, group nodes, and is idempotent.

### Changed

- **Spec loading centralized**: All spec file loading in the autonomy layer (`orchestrator.py`, `handlers_session_lifecycle.py`, `handlers_session_rebase.py`) now uses `load_spec_file()` from the spec adapter instead of inline `json.loads()` calls. This ensures hierarchy-format specs are automatically converted to the phases view.

## [0.12.0b2] - 2026-02-18

### Removed

- **Feature flag system**: Removed the entire feature flag infrastructure (parsing, env vars, config loading, validation, dependency resolution). Autonomy features (`autonomy_sessions`, `autonomy_fidelity_gates`) are now always enabled — no opt-in required.
- **`FEATURE_DISABLED` error code**: Removed `ErrorCode.FEATURE_DISABLED` and `ErrorType.FEATURE_FLAG` from the response schema and autonomy signal handling.
- **Feature flag discovery metadata**: Removed `AUTONOMY_FEATURE_FLAGS` registry, `get_autonomy_capabilities()`, `is_autonomy_feature_flag()`, and `get_autonomy_feature_flag()` from discovery module.
- **Stale files**: Removed `REFACTOR-RESEARCH.md` and `skills/foundry-implement-v2/SKILL.md`.

### Changed

- **Default role**: Changed default server role from `observer` (fail-closed read-only) to `maintainer` (full interactive access). Autonomous sessions continue to use posture-driven role overrides.
- **Capabilities endpoint**: `get_capabilities()` no longer accepts `feature_flags` parameter; always reports autonomy as enabled.
- **Config loader**: Removed `[feature_flags]` TOML section parsing, `FOUNDRY_MCP_FEATURE_FLAGS` env var, and per-flag `FOUNDRY_MCP_FEATURE_FLAG_<NAME>` overrides.
- **Startup validation**: Removed feature flag dependency checks and autonomy-only security toggle warnings.

## [0.12.0b1] - 2026-02-18

### Added

- **Autonomous Spec Execution (ADR-002)**: Durable session management for autonomous spec-driven task execution with replay-safe semantics
  - Session lifecycle: `session-start`, `session-pause`, `session-resume`, `session-end`, `session-reset`, `session-rebase`, `session-heartbeat`, `session-status`, `session-list`
  - Step orchestration: 18-step priority sequence (`session-step-next`, `session-step-report`, `session-step-replay`) driving task progression with exactly-once semantics
  - Fidelity gates: per-phase quality gates with strict/lenient/manual policies, auto-retry with cycle caps, and human acknowledgment flow
  - Write-lock enforcement: prevents concurrent mutations to specs under active autonomous sessions, with bypass mechanism for emergency overrides
  - File-backed persistence (`AutonomyStorage`): atomic writes, per-spec pointer files, cursor-based pagination, TTL-based garbage collection
  - Spec integrity validation: structure hashing with mtime fast-path optimization, rebase recovery for mid-session spec edits
  - Resume context: provides completed/pending task summaries, journal hints, and phase progress on session resume
  - State migrations infrastructure for schema versioning
- **Capabilities manifest**: `autonomy_sessions` and `autonomy_fidelity_gates` registered as experimental feature flags
- **Discovery module**: Autonomy capabilities exposed for tool discovery

### Fixed

- **Terminal states check**: Removed `FAILED` from terminal states in orchestrator — only `COMPLETED` and `ENDED` are truly terminal per ADR state transition table; `FAILED` can transition to `running` via resume or rebase
- **Auto-retry cycle cap**: Added cycle cap check before scheduling `address_fidelity_feedback` to prevent exceeding `max_fidelity_review_cycles_per_phase`
- **State version increments**: Added missing `state_version` increments in rebase handler (both no-change and success paths), heartbeat handler, and force-end on start
- **Reset handler safety**: Reset now requires explicit `session_id` parameter — no active-session lookup allowed, per ADR safety constraint
- **Session override contract enforcement**: `session-end` and `session-reset` now require `reason_code` (`OverrideReasonCode`) from callers/tests; validation errors are returned when missing or invalid
- **GC spec-lock cleanup**: `cleanup_expired()` now removes orphaned spec-lock files in addition to session locks and pointer files
- **`AutonomyStorage` export**: Added `AutonomyStorage` to `core/autonomy/__init__.py` public API exports
- **`LastStepResult` validation**: Added cross-field model validator ensuring required fields per step type (`task_id` for implement/verify, `phase_id`+`gate_attempt_id` for gate steps)

### Changed

- **Autonomy posture profiles**: Added fixed posture profiles (`unattended`, `supervised`, `debug`) with runtime defaults for role, escape-hatch policy, and session-start defaults.
- **Capability response enhancements**: `server(action="capabilities")` now includes runtime posture/security/session-default details and a role-preflight convention contract for consumers.
- **Session-start default expansion**: `task(action="session", command="start")` now uses config-driven session defaults (`gate_policy`, stop/retry behavior, bounded limits) when request fields are omitted.
- **Declarative parameter validation framework** (`tools/unified/param_schema.py`): Migrated 69 handlers across 5 waves from imperative `_validation_error()` calls to declarative schemas. ~370 validation calls reduced to 4 (irreducibly runtime-dependent). ~15-20% handler file size reduction.
- **Unified error hierarchy** (`core/errors/`): Consolidated all 37 error classes from 17 source files into a structured package (9 domain modules). Added `ERROR_MAPPINGS` registry with 24 error-to-code mappings and `error_to_response()` helper.
- **God object decomposition**: Split 3 core monoliths into sub-module packages:
  - `core/validation/` (2,342 lines → 10 sub-modules)
  - `core/ai_consultation/` (1,774 lines → 4 sub-modules)
  - `core/observability/` (1,218 lines → 6 sub-modules)
- **Research tool decomposition**: Extracted 17 handlers from monolithic `research.py` into `research_handlers/` package with 5 domain-focused modules.
- **Tool registration signature cleanup**: Replaced manual payload dict construction (~470 lines) with `locals()` filter pattern in task, authoring, and research tool registrations.
- **Observability/metrics consolidation** (`core/metrics/`, `core/observability/`): Merged 6 standalone files into 2 bounded packages with deprecation shims.
- **Deep research phase execution framework** (`phases/_lifecycle.py`, `_run_phase()`): Extracted shared LLM call lifecycle and orchestrator dispatch boilerplate from 4 phase mixins. ~350 lines of phase boilerplate and ~250 lines of dispatch boilerplate eliminated.
- **Config module decomposition** (`config/`): Split `config.py` (2,771 lines, 24 dataclasses) into 7 focused sub-modules (server, loader, research, autonomy, domains, parsing, decorators).
- **Research models decomposition** (`core/research/models/`): Split 2,176-line models file (33 classes) into 8 sub-modules (enums, digest, conversations, thinkdeep, ideation, consensus, fidelity, sources, deep_research).
- **Autonomy models decomposition** (`core/autonomy/models/`): Split 1,154-line models file (36 classes) into 8 sub-modules (enums, verification, gates, steps, session_config, responses, state, signals).
- **LLM config decomposition** (`core/llm_config/`): Split 1,419-line module into 5 sub-modules (paths, provider_spec, llm, workflow, consultation).
- **Discovery module decomposition** (`core/discovery/`): Split 1,811-line module (55% metadata dicts) into 10 sub-modules with metadata separated from infrastructure.
- **Research infrastructure decomposition**: Split 7 monolithic modules (~9,936 lines total) into focused sub-packages: `document_digest/`, `context_budget/`, `summarization/`, `token_management/`, `providers/resilience/`, plus deep research core/analysis mixin extraction.
- **Response module decomposition** (`core/responses/`): Split 1,697-line module (62 import sites) into 7 sub-modules (types, builders, errors_generic, errors_spec, errors_ai, sanitization, batch_schemas).
- **Tool router handler decomposition**: Split 3 remaining large routers (spec 1,180 lines, environment 1,373 lines, review 1,268 lines) into handler packages with 29 handlers across 11 files.
- **Test fixture consolidation**: Created 5 `conftest.py` files centralizing duplicate fixtures across the test suite. ~820 lines deduplicated.
- **Validation fix dispatch dict**: Replaced 12 sequential `if code ==` branches with two dispatch dicts in `core/validation/fixes.py`.
- All refactoring completed with zero test regressions (5,208 tests passing). Backward-compatible `__init__.py` re-exports preserved for all decomposed packages.

### Migration

- **Legacy session action names remain supported** but are now on an explicit removal window: **3 months or 2 minor releases (whichever is later)**.
- Legacy action responses include machine-readable deprecation metadata:
  - `meta.deprecated.action`
  - `meta.deprecated.replacement`
  - `meta.deprecated.removal_target`
- Server logs now emit `WARN` entries for legacy action invocations to support migration tracking.

## [0.11.1] - 2026-02-15

### Changed

- **Unified config search paths**: Centralized config file discovery into `_default_config_search_paths()` with priority-ordered search (bundled defaults → XDG → home dotfile → project dotfile → project config). "Last match wins" replaces "first match breaks" for proper layered config.
- **Bundled default config**: Package wheel now includes `samples/foundry-mcp.toml` as a fallback default, ensuring sensible defaults even without user config files.
- **AI status reporting**: `get_llm_status()` now reports consultation orchestrator provider availability first, falling back to legacy LLM config only when the consultation layer is unavailable.

### Fixed

- **Empty AI consultation guard**: Review helpers now validate that AI consultation returned non-empty content before processing, returning a structured error instead of silently proceeding with empty results.
- **Review status test**: Fixed `test_get_llm_status_handles_import_error` to patch both consultation and legacy config paths.

## [0.11.0] - 2026-02-14

### Removed

- **MCP tools: `pr`, `code`, `test`** — Removed three MCP tools that duplicated capabilities already available natively in Claude Code and other AI coding agents. This reduces the tool surface from 16 to 13 unified tools, lowering token overhead and eliminating maintenance burden for tools that provided no unique value over the host environment.
  - `pr` tool (PR creation/description generation) — replaced by native `gh` CLI usage
  - `test` tool (test execution/debugging) — replaced by native test runner invocation
  - `code` tool (code generation) — removed as redundant with agent capabilities
- **CLI commands: `pr`, `test`** — Removed corresponding CLI subcommands (`cli/commands/pr.py`, `cli/commands/testing.py`) and their registry entries
- **Core testing module** (`core/testing.py`, 839 lines) — Removed the test execution engine backing the test tool

### Changed

- Updated capabilities manifest to reflect 13 tools (was 16)
- Updated tool registration parity tests, dispatch contract tests, telemetry invariant tests, and golden fixture tests
- Updated documentation (MCP tool reference, CLI command reference) to remove references to deleted tools
- Added `research` to capabilities manifest (was missing)

## [0.10.1] - 2026-02-08

### Fixed

- **Error tool completely broken**: All error tool actions (`list`, `get`, `stats`, `patterns`, `cleanup`) crashed with `dispatch_with_standard_errors() got multiple values for argument 'tool_name'`. The `tool_name` filter parameter in the error tool payload collided with the positional `tool_name` parameter in the shared dispatch function when unpacked via `**payload`.
- **Hardened dispatch signature**: Made `router`, `tool_name`, and `action` positional-only (`/`) in `dispatch_with_standard_errors()` to prevent any future parameter name collisions from tool payloads.

## [0.10.0] - 2026-02-08

### Changed

- **Deep Research Decomposition**: Refactored `core/research/workflows/deep_research.py` (6,994 lines) into a `deep_research/` package with 12 focused modules:
  - `constants.py` — shared constants, enums, and configuration defaults
  - `helpers.py` — utility functions (formatting, token estimation, ID generation)
  - `source_quality.py` — source quality scoring and deduplication
  - `budgeting.py` — token and source budget allocation
  - `crash_handler.py` — crash recovery infrastructure and partial result persistence
  - `orchestration.py` — `PhaseOrchestrator` and `PhaseContext` for phase lifecycle management
  - Phase mixins: `PlanningPhaseMixin`, `GatheringPhaseMixin`, `AnalysisPhaseMixin`, `SynthesisPhaseMixin`, `RefinementPhaseMixin`
  - `BackgroundTaskMixin` and `SessionManagementMixin` for async task and session lifecycle
  - `core.py` — main `DeepResearchWorkflow` class composing all mixins (orchestration + cross-cutting concerns)
  - 18 contract tests in `test_deep_research_public_api.py` verifying backward compatibility

- **Spec Module Split**: Refactored `core/spec.py` (4,116 lines) into a `core/spec/` package with focused sub-modules:
  - `_constants.py` — shared constants (templates, categories, verification types)
  - `io.py` — I/O functions (find, load, save, backup, list, diff, rollback)
  - `hierarchy.py` — hierarchy operations (get/update node, phase CRUD, recalculate hours)
  - `templates.py` — spec creation, phase templates, assumptions, revisions, frontmatter
  - `analysis.py` — read-only analysis (completeness checks, duplicate detection)
  - `_monolith.py` — remaining operations (find-replace)
  - `__init__.py` re-exports all public symbols for backward compatibility

- **Task Module Split**: Refactored `core/task.py` (2,463 lines) into a `core/task/` package with focused sub-modules:
  - `_helpers.py` — shared constants and utilities
  - `queries.py` — read-only query and context functions
  - `mutations.py` — task mutation operations (add, remove, update, move)
  - `batch.py` — batch update operations
  - `__init__.py` re-exports all public symbols for backward compatibility

- **Shared Dispatch Helpers**: Extracted `ActionRouter` and `dispatch_with_standard_errors` into reusable pattern across all tool routers
- **Claude CLI Base Class**: Extracted shared `_claude_base.py` for `claude` and `claude-zai` providers, reducing duplication
- **Unified Handler Signatures**: Task handlers now use `**payload: Any` convention matching authoring handlers
- **pypdf moved to optional deps**: `pypdf` is no longer a hard dependency; imported on demand with graceful fallback

## [0.9.0b12] - 2026-02-01

### Added

- **Claude ZAI Provider**: New CLI provider for users with custom `claude-zai` alias configurations
  - Separate provider from `claude` for custom alias settings (e.g., custom MCP servers, system prompts)
  - Read-only tool restrictions for security: allows Read, Grep, Glob, read-only git commands
  - Blocks write operations, web operations (data exfiltration risk), and destructive commands
  - Detector registration with `FOUNDRY_CLAUDE_ZAI_AVAILABLE_OVERRIDE` and `FOUNDRY_CLAUDE_ZAI_BINARY` env vars
  - Sample config updated with claude-zai provider examples

## [0.9.0b11] - 2026-01-28

### Added

- **Document Digest for Deep Research**: Integrated document digest pipeline into deep research workflows
  - PDF extraction with security hardening (file size limits, page limits, malformed PDF handling)
  - Foundation models and configuration for document digest processing
  - Deep research integration for digest-based source analysis
  - Comprehensive test coverage for document digest, deep research digest integration, and PDF extraction

### Changed

- Updated MCP response schema documentation with expanded contract details

## [0.9.0b10] - 2026-01-27

### Added

- **Provider Rate Limit Resilience**: Unified resilience layer for all search providers with circuit breakers, rate limiting, and retry with backoff
  - **Circuit Breakers**: Per-provider circuit breakers with CLOSED → OPEN → HALF_OPEN state transitions
    - Opens after 5 consecutive failures, 30s recovery timeout
    - HALF_OPEN allows recovery probes before fully closing
    - Deep research skips providers with OPEN circuits, enabling graceful failover
  - **Rate Limiting**: Token bucket rate limiting per provider
    - Default: 1 RPS with burst limit of 3
    - Semantic Scholar: 0.9 RPS with burst limit of 2 (conservative for academic API)
    - Configurable `max_wait_seconds` for rate limit queue timeout
  - **Retry with Jitter**: Exponential backoff with deterministic jitter (seeded RNG for testing)
    - 50-150% jitter range prevents thundering herd
    - Respects time budget constraints during retry delays
  - **Error Classification**: Provider-specific error handling via `classify_error()` hook
    - 429 errors: Retryable, does NOT trip circuit breaker
    - 401/403 errors: Not retryable, does NOT trip circuit breaker
    - 5xx errors: Retryable, trips circuit breaker
    - Timeouts/network errors: Retryable, trips circuit breaker
  - **Time Budget Enforcement**: Operations cancelled when exceeding time budget
  - **Graceful Degradation**: Providers that trip mid-gathering are skipped for remaining sub-queries
  - **Observability**: Circuit breaker state changes and rate limit waits logged via audit_log
  - **Testing Support**: `reset_resilience_manager_for_testing()` for test isolation

- **New Exceptions**:
  - `RateLimitWaitError`: Raised when rate limit wait would exceed `max_wait_seconds`
  - `TimeBudgetExceededError`: Raised when operation exceeds time budget

### Changed

- Deep research gathering phase now filters providers by circuit breaker state before execution
- Provider docstrings updated with resilience configuration details

## [0.9.0b9] - 2026-01-27

### Added

- **Semantic Scholar API Enhancements**: Extended Semantic Scholar search provider with TLDR support and new filtering/sorting capabilities
  - **TLDR Summaries**: Auto-generated paper summaries now used as snippet when available, with abstract fallback
  - **Extended Metadata**: New fields in search results: `venue`, `influential_citation_count`, `reference_count`, `fields_of_study`, `tldr`
  - **Publication Type Filtering**: `publication_types` parameter to filter by paper type (JournalArticle, Conference, Review, etc.)
  - **Sorting**: `sort_by` parameter (citationCount, publicationDate, paperId) with `sort_order` (asc/desc, default: desc)
  - **Extended Fields Toggle**: `use_extended_fields` parameter (default: True) to control metadata verbosity
  - **Parameter Validation**: Comprehensive validation with clear error messages for all new parameters
  - **Configuration Fields**: New `[research]` TOML config options:
    - `semantic_scholar_publication_types`, `semantic_scholar_sort_by`, `semantic_scholar_sort_order`
    - `semantic_scholar_use_extended_fields`
  - **Deep Research Integration**: Parameters automatically propagated from config to Semantic Scholar search calls

### Changed

- **Semantic Scholar Endpoint**: Switched from `/paper/search/bulk` to `/paper/search` (relevance search) to enable TLDR support
  - **Breaking**: Maximum results per query reduced from 1000 to 100 (API limit for /paper/search endpoint)
  - Results are now relevance-ranked by default
- **Semantic Scholar Sorting**: `sort_order` without `sort_by` now defaults to `publicationDate`

## [0.9.0b8] - 2026-01-27

### Added

- **Perplexity API Enhancements**: Extended Perplexity search provider with all parameters from the Perplexity Sonar API spec
  - **Search Context Size**: `search_context_size` parameter (`"low"`, `"medium"`, `"high"`) controls result context depth
  - **Token Limits**: `max_tokens` (default: 50000) and `max_tokens_per_page` (default: 2048) for response control
  - **Recency Filtering**: `recency_filter` parameter (`"day"`, `"week"`, `"month"`, `"year"`) for time-based filtering
  - **Date Range Filters**: `search_after_date` and `search_before_date` for precise date range queries (MM/DD/YYYY format)
  - **Content Modification Filters**: `last_updated_after_filter` and `last_updated_before_filter` for filtering by modification date
  - **Geographic Filtering**: `country` parameter (ISO 3166-1 alpha-2 code, e.g., `"US"`)
  - **Configuration Fields**: New `[research]` TOML config options:
    - `perplexity_search_context_size`, `perplexity_max_tokens`, `perplexity_max_tokens_per_page`
    - `perplexity_recency_filter`, `perplexity_country`
  - **Deep Research Integration**: Parameters automatically propagated from config to Perplexity search calls
  - **Parameter Validation**: Comprehensive validation with clear error messages for all new parameters

### Migration

**Default behavior unchanged**: All new config fields have defaults that preserve existing behavior. Users not configuring Perplexity options will see no change.

### Fixed

- **Environment get-config Lookup**: Fixed `environment(action="get-config")` to use the full config lookup hierarchy (matching `ServerConfig.from_env()`), instead of only checking the current working directory
  - Added `path` parameter for explicit config file specification
  - Now searches: explicit path → `FOUNDRY_MCP_CONFIG_FILE` env var → project dir → `~/.foundry-mcp.toml` → XDG config
  - Error response includes `searched_paths` for debugging

## [0.9.0b7] - 2026-01-26

### Added

- **Deep Research CPU Optimization**: Performance improvements to reduce CPU and I/O overhead in deep research workflows
  - **Status Persistence Throttling**: New `status_persistence_throttle_seconds` config (default: 5) limits disk writes during research, persisting immediately only on phase/iteration changes or terminal states
  - **Token Count Caching**: Content hash-based caching in `ResearchSource` eliminates redundant token estimation calls on workflow resume
  - **Audit Verbosity Modes**: New `audit_verbosity` config option (`"full"` or `"minimal"`) controls JSONL audit payload size - minimal mode nulls large text fields while preserving schema shape for downstream compatibility

### Changed

- Status persistence now uses smart throttling instead of persisting after every operation
- Token estimation skips already-cached sources, reducing CPU on resume operations

## [0.9.0b6] - 2026-01-26

### Added

- **Deep Research Resilience**: Comprehensive resilience improvements for long-running research workflows
  - **Task Registry & Cancellation**: Centralized task registry with two-phase cancellation (cooperative then forced)
  - **Timeout Watchdog**: Background monitor that detects timeout and staleness conditions
  - **Provider Timeout Enforcement**: Per-provider timeout handling with `ProviderTimeoutError`
  - **Progress Heartbeats**: `last_heartbeat_at` timestamp updated before each provider call for visibility
  - **Default Timeout Configuration**: Configurable `deep_research_timeout` with precedence rules (explicit > config > 600s fallback)
  - **Executor Isolation**: `ProviderExecutor` thread pool for isolating blocking CLI provider operations
  - **Status Metadata**: New fields in status response: `last_heartbeat_at`, `is_timed_out`, `is_stale`, `effective_timeout`

- **Documentation**: Added resilience configuration and troubleshooting guides
  - Timeout behavior and configuration in `docs/06-configuration.md`
  - Troubleshooting for timeout, staleness, and cancellation issues in `docs/07-troubleshooting.md`

## [0.9.0b5] - 2026-01-26

### Fixed

- **TOCTOU Race Conditions**: Fixed time-of-check-time-of-use race conditions in `FileStorageBackend` by moving file existence and expiry checks inside file locks
- **Thread Safety**: Added `_active_sessions_lock` for thread-safe session tracking in deep research, preventing race conditions during concurrent access
- **Task Registry GC**: Replaced `WeakValueDictionary` with regular dict for background task registry to prevent premature garbage collection of running tasks
- **Timeout Flag Accuracy**: Improved timeout metadata accuracy - only sets `timeout: true` when all failures were timeouts, not when mixed with other error types
- **State Persistence**: Save thread/consensus state BEFORE provider calls to ensure user messages and responses persist even if provider fails
- **Deep Research Timeout Handling**: Mark deep research state as failed on timeout instead of leaving in intermediate state
- **Orphaned Lock Files**: Clean up orphaned lock files when data files are missing

### Changed

- Use timezone-aware `datetime.now(timezone.utc)` throughout instead of deprecated `datetime.utcnow()`
- Added task cleanup methods (`cleanup_stale_tasks`, `_cleanup_completed_task`) for memory management in long-running processes

## [0.9.0b4] - 2026-01-25

### Added

- **Tavily API Enhancements**: Extended Tavily search provider with all parameters from the Tavily API spec and added new Extract endpoint support
  - **New Search Parameters**:
    - `search_depth`: Search mode - `"basic"` (default), `"advanced"` (2x credits), `"fast"`, `"ultra_fast"`
    - `topic`: Search topic - `"general"` (default), `"news"`
    - `days`: News recency limit (1-365 days, only when `topic="news"`)
    - `include_images`: Include image results (default: false)
    - `include_raw_content`: Get raw page content - `false`, `true`/`"markdown"`, or `"text"`
    - `country`: ISO 3166-1 alpha-2 country code to boost results (e.g., `"US"`)
    - `chunks_per_source`: Chunk count for advanced search (1-5, default: 3)
    - `auto_parameters`: Let Tavily auto-configure based on query intent
  - **Tavily Extract Provider** (`TavilyExtractProvider`): New provider for URL content extraction
    - Extracts structured content from up to 10 URLs per request
    - Parameters: `extract_depth` (`"basic"`/`"advanced"`), `include_images`, `format` (`"markdown"`/`"text"`), `query`, `chunks_per_source`
    - SSRF protection: blocks private IPs, localhost, reserved ranges, suspicious schemes
    - Partial failure handling: returns successful extractions even when some URLs fail
    - Retry logic with exponential backoff matching search provider
  - **Configuration Fields**: New `[research]` TOML config options
    - `tavily_search_depth`, `tavily_topic`, `tavily_news_days`, `tavily_include_images`
    - `tavily_country`, `tavily_chunks_per_source`, `tavily_auto_parameters`
    - `tavily_extract_depth`, `tavily_extract_include_images`
    - `tavily_extract_in_deep_research`: Enable extract as follow-up step in deep research
    - `tavily_extract_max_urls`: Max URLs to extract per deep research run (default: 5)
  - **Deep Research Integration**:
    - `_get_tavily_search_kwargs()` method propagates config to search calls
    - Research mode smart defaults: academic/technical modes prefer `"advanced"` depth
    - Optional extract follow-up step when `tavily_extract_in_deep_research=true`
  - **Security**: URL validation for extract with SSRF protection
    - Blocks localhost, private IPs (10.x, 172.16.x, 192.168.x), link-local, loopback
    - Blocks dangerous schemes (file://, gopher://, dict://, data://)
    - DNS resolution with timeout to prevent DNS rebinding
    - Max URL length (2048 chars), max content size (50KB per source)

### Migration

**Default behavior unchanged**: All new config fields have defaults that preserve existing behavior. Users not configuring Tavily options will see no change.

**Credit cost awareness**: Using `search_depth="advanced"` doubles Tavily API credit usage. Consider `"basic"` for most queries, `"advanced"` only when deeper analysis is needed.

**Configuration example**:
```toml
[research]
# Search configuration
tavily_search_depth = "basic"  # or "advanced" (2x credits), "fast", "ultra_fast"
tavily_topic = "general"       # or "news"
tavily_news_days = 7           # only when topic = "news"
tavily_include_images = false
tavily_country = "US"          # boost results from country
tavily_chunks_per_source = 3   # 1-5, for advanced search
tavily_auto_parameters = false # let Tavily auto-configure

# Extract configuration
tavily_extract_depth = "basic"           # or "advanced"
tavily_extract_include_images = false
tavily_extract_in_deep_research = false  # enable extract follow-up
tavily_extract_max_urls = 5              # max URLs per deep research run
```

## [0.9.0b3] - 2026-01-25

### Added

- **XDG Base Directory support**: Config loading now checks `~/.config/foundry-mcp/config.toml` (or `$XDG_CONFIG_HOME/foundry-mcp/config.toml`) as the lowest-priority user config location, following the XDG Base Directory Specification common on Linux systems

### Changed

- **Config loading priority** (highest to lowest):
  1. Environment variables
  2. Project config (`./foundry-mcp.toml`)
  3. User config (`~/.foundry-mcp.toml`)
  4. XDG config (`~/.config/foundry-mcp/config.toml`) - NEW
  5. Built-in defaults

## [0.9.0b2] - 2026-01-25

### Fixed

- **Deep research timezone crash**: Fixed `TypeError: can't subtract offset-naive and offset-aware datetimes` in `_allocate_source_budget` when Tavily returns timezone-naive `discovered_at` timestamps

## [0.9.0b1] - 2026-01-25

### Added

- **Deep research token management**: Intelligent content compression and graceful degradation for large research workflows
  - New `ContextBudgetAllocator` manages token budget allocation with priority-based content selection
  - New `ContentArchive` enables file-based storage of dropped content with TTL cleanup
  - New `GracefulDegradationManager` applies progressive content compression (FULL → CONDENSED → COMPRESSED → KEY_POINTS → HEADLINE → DROPPED)
  - Fidelity tracking with per-item compression ratios and overall fidelity score
  - Top-5 source protection guardrail ensures minimum 30% fidelity for high-priority items
  - Content summarization via configurable LLM providers with fallback chain

- **Token management configuration** (`ResearchConfig`):
  - `token_management_enabled` - Master switch for token management (default: true)
  - `token_safety_margin` - Budget safety buffer fraction (default: 0.15)
  - `runtime_overhead` - Tokens reserved for CLI/IDE context (default: 60000 for Claude Code)
  - `model_context_overrides` - Per-model context/output limit overrides
  - `summarization_provider` / `summarization_providers` - LLM providers for content summarization
  - `summarization_timeout` / `summarization_cache_enabled` - Summarization performance settings
  - `allow_content_dropping` - Allow dropping low-priority content (default: false)
  - `content_archive_enabled` / `content_archive_ttl_hours` / `research_archive_dir` - Archive settings

- **Response schema extensions** for content fidelity:
  - `meta.content_fidelity` - Response completeness level (full|partial|summary|reference_only)
  - `meta.content_fidelity_schema_version` - Schema version for fidelity metadata ("1.0")
  - `meta.dropped_content_ids` - IDs of content items omitted from response
  - `meta.content_archive_hashes` - Map of archive IDs to content hashes for retrieval
  - `meta.warning_details` - Structured warnings with code, severity, message, and context

- **Token management warning codes**:
  - `CONTENT_TRUNCATED` - Content summarized/compressed to fit budget
  - `CONTENT_DROPPED` - Low-priority content removed
  - `TOKEN_BUDGET_FLOORED` - Item preserved due to min items guardrail
  - `ARCHIVE_WRITE_FAILED` / `ARCHIVE_DISABLED` / `ARCHIVE_READ_CORRUPT` - Archive status
  - `TOKEN_COUNT_ESTIMATE_USED` - Character-based heuristic fallback (tiktoken unavailable)

- **Documentation**:
  - Token Management section in `docs/concepts/deep_research_workflow.md`
  - Token Management Warnings section (17.4) in `dev_docs/codebase_standards/cli-output.md`
  - Sample TOML configuration with runtime overhead values for different CLIs

### Changed

- **`success_response` helper** now accepts content fidelity parameters (`content_fidelity`, `dropped_content_ids`, `content_archive_hashes`, `warning_details`)

## [0.8.35] - 2026-01-24

### Added

- **Deep research polling mitigation**: Track status check counts and provide guidance to prevent aggressive polling
  - New `status_check_count` and `last_status_check_at` fields in `DeepResearchState`
  - Start response now includes `polling_guidance` with max checks, typical duration, and behavioral rules
  - Status response includes `next_action` guidance based on check count (warns at limit)
  - Instructs callers not to use WebSearch/WebFetch while deep research is running

## [0.8.34] - 2026-01-24

### Added

- **Deep research timeout resilience**: Async provider execution with timeout protection and retry/fallback logic
  - New `_execute_provider_async()` method in base workflow with `asyncio.wait_for()` timeout protection
  - All 4 deep research phase methods now use async execution with timeout handling
  - Per-phase fallback provider lists: `deep_research_planning_providers`, `deep_research_analysis_providers`, `deep_research_synthesis_providers`, `deep_research_refinement_providers`
  - Retry settings: `deep_research_max_retries` (default: 2) and `deep_research_retry_delay` (default: 5.0s)
  - `get_phase_fallback_providers()` helper method in `ResearchConfig`
  - Enhanced timeout logging with phase context and provider history
  - Sample TOML documentation for new retry/fallback settings

## [0.8.33] - 2026-01-24

### Added

- **Research workflow exception handling**: Improved resilience to prevent MCP server crashes
  - `DeepResearchWorkflow.execute()` now catches all exceptions and returns error result
  - `_dispatch_research_action()` catches exceptions and returns structured error response
  - Added warning logging when provider resolution fails in base workflow
  - Added comprehensive tests for exception handling behavior

## [0.8.32] - 2026-01-22

### Changed

- **Improved lifecycle action descriptions**: More descriptive summaries for `activate`, `complete`, and `archive` actions clarifying behavior and folder destinations

### Removed

- **Removed intake tools feature flag**: Intake tools (`intake-add`, `intake-list`, `intake-dismiss`) are now always enabled
  - Removed `intake_tools` feature flag from capabilities manifest
  - Removed feature flag check code from authoring tool handlers
  - Removed feature flag documentation from intake guide

## [0.8.31] - 2026-01-22

### Added

- **Batch mode for autonomous sessions**: New workflow mode that executes a fixed number of tasks before pausing for approval
  - `AutonomousSession` now tracks `batch_mode`, `batch_size`, and `batch_remaining`
  - Task `complete` action decrements batch counter and auto-pauses when limit reached
  - Session config `session-config` applies `WorkflowConfig` defaults for batch settings
  - Supports `WorkflowMode.BATCH` and `WorkflowMode.AUTONOMOUS` from config

### Changed

- **Research state storage relocation**: Default storage moved from `~/.foundry-mcp/research` to project-local `specs/.research`
  - New `research_dir` config option replaces old global storage path
  - New `FOUNDRY_MCP_RESEARCH_DIR` environment variable
- **Sample config updates**: Increased default research timeouts from 360s to 600s, updated provider examples
- **Documentation updates**: Various fixes across guides and tool reference

### Removed

- **Simplified ResearchConfig**: Removed `storage_path` and `storage_backend` fields (storage is now always project-local)
- **Removed `journals_path`**: Replaced with `research_dir` for research state storage
- **Removed root config file**: Deleted `foundry-mcp.toml` from repository root (use `samples/foundry-mcp.toml` as template)
- **Removed home config sample**: Deleted `samples/home-foundry-mcp.toml` (layered config still works but sample removed)

## [0.8.30] - 2026-01-22

### Added

- **Layered configuration loading**: Configuration now loads in layers for better separation of concerns
  - Home config (`~/.foundry-mcp.toml`) for user-wide defaults (API keys, preferred providers, logging preferences)
  - Project config (`./foundry-mcp.toml`) for project-specific overrides
  - Environment variables for runtime overrides (highest priority)
  - Legacy `.foundry-mcp.toml` fallback maintained for backwards compatibility
- **User config sample**: New `samples/home-foundry-mcp.toml` example for user-level configuration
- **Config hierarchy tests**: Comprehensive test coverage for layered configuration loading

### Changed

- **Updated configuration documentation**: `docs/06-configuration.md` explains the new config hierarchy with table of file locations and use cases
- **Updated project config sample**: `samples/foundry-mcp.toml` updated with layered config priority comments

## [0.8.29] - 2026-01-19

### Changed

- **Renamed bikelane to notes**: Intake queue terminology updated across tools, docs, and specs
  - Config key `notes_dir` replaces `bikelane_dir` (no backward compatibility)
  - Env var `FOUNDRY_MCP_NOTES_DIR` replaces `FOUNDRY_MCP_BIKELANE_DIR`
  - Default storage path is now `specs/.notes/`

## [0.8.28] - 2026-01-18

### Changed

- **Perplexity provider enhancements**: Improved deep research content extraction
  - Added `include_raw_content` option to map snippets to content field for source aggregation
  - Added `max_tokens` (50000) and `max_tokens_per_page` (2048) parameters for better content retrieval

### Removed

- **Google search provider**: Removed Google Custom Search from deep research providers
  - Removed `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` environment variables from documentation
  - Removed `google` from default `deep_research_providers` list
  - Removed Google rate limit configuration

## [0.8.27] - 2026-01-18

### Fixed

- **Restored sample config sections**: Added back `[workflow]` and `[consultation]` sections to `samples/foundry-mcp.toml` that are required by `llm_config.py` and `ai_consultation.py`

### Removed

- **Removed `[metrics_persistence]` from sample config**: Advanced feature not needed in sample configuration

## [0.8.26] - 2026-01-18

### Changed

- **Documentation reorganization**: Separated internal developer docs from user-facing documentation
  - Moved MCP best practices, CLI best practices, codebase standards, and developer guides to `dev_docs/`
  - Created new user-facing documentation structure in `docs/` with quick start, workflow guides, CLI/MCP references, configuration, and troubleshooting
  - Updated `AGENTS.md` and `CLAUDE.md` with corrected paths to `dev_docs/`
- **README overhaul**: Rewrote README to be user-focused with clearer value proposition, streamlined installation, and practical quick start guide

### Added

- **New user documentation**: Comprehensive docs for end users
  - `docs/01-quick-start.md` - Getting started guide
  - `docs/02-core-concepts.md` - Specs, phases, tasks explained
  - `docs/03-workflow-guide.md` - Step-by-step workflow
  - `docs/04-cli-command-reference.md` - Full CLI documentation
  - `docs/05-mcp-tool-reference.md` - MCP tool documentation
  - `docs/06-configuration.md` - Configuration options
  - `docs/07-troubleshooting.md` - Common issues and solutions
  - `docs/concepts/spec-schema.md` - Spec JSON schema reference
  - `docs/concepts/response-envelope.md` - Response format reference
  - `docs/reference/error-codes.md` - Error code reference

## [0.8.25] - 2026-01-18

### Removed

- **Dashboard feature**: Removed entire Streamlit-based dashboard package (`src/foundry_mcp/dashboard/`)
- **Dashboard CLI command**: Removed `dashboard` subcommand (`src/foundry_mcp/cli/commands/dashboard.py`)
- **Dashboard dependencies**: Removed streamlit, plotly, pandas from pyproject.toml
- **Dashboard documentation**: Removed `docs/dashboards/` directory and Prometheus rules
- **Metrics unified tool**: Removed `src/foundry_mcp/tools/unified/metrics.py` (tool count 17→16)
- **Legacy verification type mapping**: Removed `"test"→"run-tests"` and `"auto"→"run-tests"` auto-conversion
- **Legacy `phase_id` parameter aliasing**: Removed `phase_id` as alias for `parent` in task actions
- **Legacy metadata category fallback**: Removed category fallback logic from metadata validation

### Changed

- Updated branding from "legacy claude-sdd-toolkit CLI" to "Foundry CLI"
- Simplified `fix-verification-types` action (sets unknown types to `"manual"` only, no legacy mapping)
- Updated tool count in documentation (17→16)

## [0.8.24] - 2026-01-17

### Changed

- **Updated skill references for claude-foundry v1.6.8**: Renamed all skill references to match the new unified naming convention
  - `sdd-toolkit:sdd-pr` → `foundry:foundry-pr`
  - `sdd-toolkit:sdd-modify` → `foundry:foundry-spec`
  - `sdd-toolkit:sdd-plan-review` → `foundry:foundry-spec`
  - Updated 15 occurrences across 4 files: `tools/unified/pr.py`, `tools/unified/review.py`, `core/review.py`, `cli/commands/review.py`
  - Plugin name changed from `sdd-toolkit` to `foundry`

## [0.8.23] - 2026-01-11

### Fixed

- **Deep research per-phase timeout config parsing**: Fixed config parsing to use correct timeout defaults
  - Main timeout: 120s → 600s
  - Planning timeout: 60s → 360s
  - Analysis timeout: 90s → 360s
  - Synthesis timeout: 180s → 600s
  - Refinement timeout: 60s → 360s
  - Config parsing now matches the class default values

### Enhanced

- **Bulk phase task validation for all task types**: Extended `authoring(action="phase-add-bulk")` validation
  - Now supports all task types: task, subtask, verify, research (previously only task/verify)
  - Uses `TASK_TYPES` constant from task module for consistency
  - Added validation for research-specific parameters: `blocking_mode`, `research_type`, `query`
  - Improved error messages to show all valid task types

## [0.8.22] - 2026-01-07

### Changed

- **Research default timeout increased**: Changed `default_timeout` from 60s to 360s in research config
  - Provides more time for complex research operations that involve multiple LLM calls and web fetches

### Fixed

- **Flexible assumption types**: Removed strict validation on `assumption_type` parameter in assumption operations
  - Previously only accepted "constraint" or "requirement"
  - Now accepts any string value (e.g., "architectural", "security", "performance")
  - Affects `authoring(action="assumption-add")` and `authoring(action="assumption-list")`

## [0.8.21] - 2026-01-07

### Added

- **Review tool "spec-review" alias**: Added `"spec-review"` as an alias for the `"spec"` action in the review tool
  - Users can now call `review(action="spec-review", ...)` or `review(action="spec", ...)` interchangeably

- **MCP task tool batch parameters**: Added missing parameters to task tool MCP schema
  - `completions: List[Dict]` for `complete-batch` action with per-task completion notes
  - `threshold_hours: float` for `reset-batch` action to filter tasks by estimated hours

## [0.8.20] - 2026-01-06

### Added

- **Hours hierarchy recalculation actions**: New spec actions to aggregate hours from tasks up through hierarchy
  - `spec(action="recalculate-hours")`: Sums `estimated_hours` from task/subtask/verify nodes to phases, then to spec total
  - `spec(action="recalculate-actual-hours")`: Sums `actual_hours` from task/subtask/verify nodes to phases, then to spec total
  - Both support `dry_run` mode to preview changes without saving
  - Returns detailed breakdown per phase with previous/calculated/delta values

- **Auto-calculate actual_hours on task completion**: When completing tasks with `started_at` but no manual `actual_hours`
  - Calculates elapsed time from `started_at` to `completed_at` in hours
  - Applies to both single task completion via `update_task_status()` and batch completion via `complete_batch()`
  - Preserves manually set values - only auto-calculates if `actual_hours` not already present

## [0.8.19] - 2026-01-06

### Fixed

- **Deep research workflow failure state handling**: Fixed issue where failed research sessions appeared "stalled" instead of properly marked as failed
  - Added `mark_failed(error: str)` method to `DeepResearchState` to explicitly mark sessions as failed
  - `_execute_gathering_async` now returns descriptive error message when all sub-queries fail to find sources
  - All phase failure blocks (planning, gathering, analysis, synthesis) now call `state.mark_failed()` before returning
  - Status reporting now includes `is_failed` and `failure_error` fields
  - Status display shows "Failed" instead of "In Progress" for failed sessions with error details

## [0.8.18] - 2026-01-06

### Fixed

- **Deep research `continue`/`resume` event loop conflict**: Fixed `RuntimeError: Cannot run the event loop while another loop is running`
  - `_continue_research()` now uses same event loop handling pattern as `_start_research()`
  - Detects running event loop with `asyncio.get_event_loop().is_running()`
  - Uses `ThreadPoolExecutor` to run async code in separate thread when already in async context
  - Fixes issue when MCP server calls deep research continue/resume actions

## [0.8.17] - 2026-01-06

### Fixed

- **MCP task tool missing `task_ids` parameter**: Added `task_ids: Optional[List[str]]` parameter to the task tool schema
  - The `start-batch`, `complete-batch`, and `reset-batch` actions require a list of task IDs
  - Previously, only `task_id` (singular string) was exposed in the MCP schema
  - Now both `task_id` and `task_ids` are available for single vs batch operations

## [0.8.16] - 2026-01-06

### Changed

- **Timeout standardization for AI CLI providers**: Increased default timeouts across all workflows
  - Per-provider/per-operation minimum: 360s (6 minutes) for Claude, Codex, Gemini, CursorAgent, OpenCode
  - Whole workflow timeouts: 600s (10 minutes) for plan_review, markdown_plan_review, deep_research
  - Deep research phase timeouts increased: planning/analysis/refinement 360s, synthesis 600s
  - Updated defaults in: `config.py`, `research.py`, `consensus.py`, `plan.py`, `review_helpers.py`
  - Updated sample and default TOML configs with new timeout standards and documentation

### Fixed

- **Deep research `continue` action background execution**: Now properly supports `background=True` parameter
  - Previously, continuing a research session ignored the `background` flag
  - Added `background` and `task_timeout` parameters to `_continue_research()` method
  - Continued research can now run in background thread like initial `start` action

- **Deep research status check crash**: Fixed `'NoneType' object has no attribute 'done'` error
  - `BackgroundTask.is_done` property now correctly handles thread-based execution
  - Previously used `task.done()` which only works for asyncio tasks
  - Now checks `thread.is_alive()` for thread-based execution (daemon threads with `asyncio.run()`)
  - Added comprehensive tests for background task state checking

## [0.8.15] - 2026-01-05

### Fixed

- **OpenCodeProvider server health check**: Now verifies server is actually responding, not just port open
  - Added `_is_opencode_server_healthy()` to verify HTTP response from `/session` endpoint
  - Prevents using stale processes or other services on port 4096
  - Clear error message when port is in use by another process

- **Missing `_intake_feature_flag_blocked` function**: Added missing function to authoring.py
  - Caused `NameError` when using intake-add, intake-list, intake-dismiss actions

## [0.8.14] - 2026-01-05

### Fixed

- **OpenCodeProvider silent server failure**: Empty responses no longer masked as SUCCESS
  - `opencode_wrapper.js` now exits with error (code 1) when server returns empty response
  - `opencode.py` validates non-empty content before returning SUCCESS status
  - Added logging to `_ensure_server_running()` for better visibility into server startup

## [0.8.13] - 2026-01-05

### Fixed

- **All CLI providers use stdin for prompts**: ClaudeProvider, GeminiProvider, and CursorAgentProvider now pass prompts via stdin
  - Avoids CLI argument length limits for long prompts
  - Updated `RunnerProtocol`, `_build_command()`, and `_run()` signatures to support `input_data`
  - Consistent pattern across all CLI-based providers (Claude, Gemini, Cursor, Codex)

- **Full spec review collects all task files**: `_build_implementation_artifacts()` now collects `file_path` from all task/subtask/verify nodes
  - Previously only worked when task_id or phase_id was specified
  - Full spec reviews now include implementation artifacts from entire spec hierarchy

### Changed

- **Provider tests updated**: Test assertions updated to reflect stdin-based prompt handling
  - `_build_command()` no longer includes prompt in returned command
  - Tests verify prompt is passed separately for stdin input

## [0.8.12] - 2026-01-04

### Fixed

- **CodexProvider stdin prompt handling**: Prompts are now passed via stdin instead of CLI arguments
  - Avoids CLI argument length limits for long prompts
  - Uses `-` marker to read prompt from stdin
  - Updated `_build_command()` and `_run()` signatures to support `input_data`

## [0.8.11] - 2026-01-04

### Fixed

- **Deep research background task execution**: Sub-queries now execute correctly
  - Root cause: `asyncio.create_task()` was called from sync MCP handlers without a running event loop
  - Solution: Use daemon threads with `asyncio.run()` for reliable background execution
  - `BackgroundTask` class now supports both thread-based and asyncio-based execution

- **Intake-add priority validation UX**: Improved error messages and usability
  - Handle explicit `null` from JSON as "use default p2" (previously caused confusing error)
  - Error messages now include valid values: `p0, p1, p2, p3, p4`
  - Added human-readable priority aliases: `critical`, `high`, `medium`, `low`, `lowest`

## [0.8.10] - 2026-01-04

### Added

- **Implement command configuration**: New `[implement]` section in TOML config
  - `auto`: Skip prompts between tasks (default: false)
  - `delegate`: Use subagents for implementation (default: **true**)
  - `parallel`: Run subagents concurrently (default: false)
- `model`: Size for delegated tasks - small, medium, large (default: small)

## [0.8.9] - 2026-01-04

### Fixed

- **Improved `phase-remove` discoverability**: Error message when attempting to remove a phase via `task action="remove"` now hints to use `authoring action="phase-remove"` instead

## [0.8.8] - 2026-01-04

### Added

- **ErrorCode.OPERATION_FAILED**: Added missing error code to `ErrorCode` enum
  - Fixes AttributeError when batch operations fail (e.g., `prepare-batch` on completed spec)

### Fixed

- **Graceful `prepare-batch` on Complete Specs**: No longer throws error when spec is complete
  - Returns `{tasks: [], spec_complete: true}` instead of erroring
  - Enables clean detection of spec completion in parallel workflows

### Changed

- **Test Updates**: Updated batch operation tests to verify graceful completion handling
  - `test_no_active_phases` now expects `None` error (not error string)
  - `test_prepare_batch_context_detects_spec_complete` verifies `spec_complete: true` response

## [0.8.7] - 2026-01-03

### Added

- **Research Nodes**: New `research` task type for spec-integrated research workflows
  - Add research nodes via `task(action="add", task_type="research", research_type="...", blocking_mode="...")`
  - Supported `research_type`: chat, consensus, thinkdeep, ideate, deep
  - `blocking_mode` controls dependency behavior: `none`, `soft` (default), `hard`
  - Research nodes with soft/none blocking don't block dependent tasks

- **Spec-Integrated Research Actions**: New research tool actions for spec nodes
  - `node-execute`: Execute research workflow linked to spec node
  - `node-record`: Record research findings to spec node
  - `node-status`: Get research node status and linked session info
  - `node-findings`: Retrieve recorded findings from spec node

- **Git Commit Suggestions**: Task completion now suggests commits based on git cadence
  - Response includes `suggest_commit`, `commit_scope`, `commit_message_hint`
  - Respects `[git].commit_cadence` config (task, phase, spec)

- **Environment get-config Action**: Read configuration sections from foundry-mcp.toml
  - `environment(action="get-config", sections=["implement", "git"])`
  - Supports filtering by specific key within a section

- **Research Memory Universal Lookup**: Added `load_session_by_id()` for loading any session type by ID prefix

### Fixed

- **Spec Validation**: Added `failed` to allowed task statuses in `_validate_spec_structure()`
  - Batch operations set `status: "failed"` on task failure, but validation rejected it
  - Now allows: `pending`, `in_progress`, `completed`, `blocked`, `failed`

- **Test Fixtures**: Fixed inline spec fixtures missing required `status` field on `spec-root` nodes

### Removed

- **Obsolete Test**: Removed `test_flags_with_context` referencing removed feature flags system

## [0.8.6] - 2026-01-03

### Fixed

- **Spec Validation**: Added `failed` to allowed task statuses in `_validate_spec_structure()`
  - Batch operations set `status: "failed"` on task failure, but validation rejected it
  - Now allows: `pending`, `in_progress`, `completed`, `blocked`, `failed`

- **Test Fixtures**: Fixed inline spec fixtures missing required `status` field on `spec-root` nodes
  - Updated 8 test specs in `test_batch_operations.py` to include `status: "in_progress"`

### Removed

- **Obsolete Test**: Removed `test_flags_with_context` from `test_sdd_cli_runtime.py`
  - Test referenced `get_cli_flags()` and `flags_for_discovery()` from removed feature flags system

## [0.8.5] - 2026-01-03

### Changed

- **Default Disabled Tools**: Added `environment` to default `disabled_tools` list
  - Tools disabled by default: `error`, `metrics`, `health`, `environment`
  - These tools are only used during setup or for dashboard features

## [0.8.4] - 2026-01-03

### Changed

- **Tools Configuration Section**: Added dedicated `[tools]` config section for tool settings
  - `disabled_tools` now preferred under `[tools]` (backward compatible with `[server]`)
  - Comprehensive documentation of all available tools and their purposes
  - Updated sample config and setup template with `[tools]` section
  - Default recommendation: disable `error`, `metrics`, `health` to save context tokens

## [0.8.3] - 2026-01-03

### Removed

- **Feature Flags System**: Removed the entire feature flags subsystem to simplify codebase
  - Deleted `src/foundry_mcp/core/feature_flags.py` and `src/foundry_mcp/cli/flags.py`
  - Removed `dev_docs/mcp_best_practices/14-feature-flags.md` documentation
  - Server tools now always use unified manifest (previously feature-flag controlled)
  - Removed all feature flag imports and usage across tools

### Added

- **Task Add `phase_id` Alias**: Added `phase_id` as alias for `parent` parameter in task `add` action
  - Provides more intuitive parameter name when adding tasks to phases
  - Falls back to `parent` if `phase_id` not provided

## [0.8.2] - 2026-01-03

### Fixed

- **CLI Provider Error Extraction**: Improved error message extraction from CLI tool outputs
  - `ClaudeProvider`: Extract errors from JSON output with `is_error: true` field
  - `CodexProvider`: Extract errors from JSONL events (`type: error`, `type: turn.failed`)
  - `CursorAgentProvider`: Check stdout for plain text errors (not just stderr)
  - `GeminiProvider`: Parse text + JSON error format, skip unhelpful `[object Object]` messages
  - `OpenCodeProvider`: Extract errors from wrapper JSONL output

- **OpenCode NODE_PATH Configuration**: Added `_ensure_node_path()` to include global npm modules
  - Allows `@opencode-ai/sdk` to be installed globally rather than bundled
  - Detects global npm root via `npm root -g` and adds to NODE_PATH

## [0.8.1] - 2026-01-03

### Added

- **Batch Operations for Parallel Task Execution**: New actions for autonomous multi-task workflows
  - `prepare-batch`: Find independent tasks for parallel execution with file-path conflict detection
  - `start-batch`: Atomically start multiple tasks as in_progress (all-or-nothing validation)
  - `complete-batch`: Complete multiple tasks with partial failure support
  - `reset-batch`: Reset batch on failure, returning tasks to pending status
  - Token budget support for context-aware batch sizing
  - Stale task detection for tasks stuck in_progress beyond threshold
  - Dependency graph visualization in prepare-batch responses

- **Autonomous Session Context Tracking**: CLI context management for continuous task processing
  - `AutonomousSession` class for tracking batch state across operations
  - `ContextTracker` for managing session lifecycle and context limits
  - Integration with task router for session-aware operations

- **Configurable Tool Disabling**: Added `disabled_tools` configuration option to selectively disable MCP tools
  - Configure via environment variable: `FOUNDRY_MCP_DISABLED_TOOLS=error,health,metrics,test`
  - Configure via TOML: `[server] disabled_tools = ["error", "health", "metrics", "test"]`
  - Tools remain in codebase but are not registered with the MCP server when disabled
  - Useful for reducing context window usage by hiding unused tools

### Changed

- Enhanced development guide with batch operations documentation and usage examples

## [0.7.11] - 2025-12-30

### Fixed

- **Consensus Workflow Provider Spec Parsing**: Extended provider spec parsing fix to consensus workflow
  - Full specs like `[cli]codex:gpt-5.2` in `consensus_providers` config now work correctly
  - Parses specs in `execute()`, `_query_provider_sync()`, and `_query_single_provider()`
  - Filters providers by base ID availability while preserving model selection

## [0.7.10] - 2025-12-30

### Fixed

- **Research Workflow Provider Spec Parsing**: Fixed `_resolve_provider` to handle full provider specs
  - Full specs like `[cli]codex:gpt-5.2-codex` in `default_provider` config now work correctly
  - Extracts base provider ID for availability check (e.g., `codex` from full spec)
  - Passes model from spec to `resolve_provider()` for proper model selection
  - Caches providers by full spec string to differentiate model variants

## [0.7.9] - 2025-12-30

### Fixed

- **Provider Detector Cache Isolation**: Fixed test pollution where availability cache persisted across tests
  - `reset_detectors()` now clears `_AVAILABILITY_CACHE` to ensure fresh detection
  - Prevents false negatives when test order affects cached availability results

- **Research E2E Test Fixtures**: Added missing `max_messages_per_thread` to mock_config fixture
  - Fixed `TypeError: '>=' not supported between instances of 'MagicMock' and 'int'`
  - Research chat workflow now works correctly in test mode

- **OpenCode Model Validation Test**: Removed obsolete test for empty model validation
  - Model validation was delegated to CLI in v0.7.5 but test was not removed
  - OpenCode provider correctly passes any model to CLI for validation

## [0.7.8] - 2025-12-30

### Fixed

- **Consensus Event Loop Conflict**: Fixed `asyncio.run() cannot be called from a running event loop` error
  - Replaced `asyncio.run()` with `ThreadPoolExecutor` for parallel provider execution
  - Works correctly within MCP server's event loop context
  - New `_execute_parallel_sync()` and `_query_provider_sync()` methods for thread-based parallelism

- **Research Timeout Configuration**: Fixed thinkdeep and other workflows timing out after 30 seconds
  - Added `default_timeout` config option to `[research]` section (default: 60 seconds)
  - Workflows now use configurable timeout from config instead of hardcoded 30s
  - Longer-running investigation workflows like thinkdeep no longer timeout prematurely

## [0.7.7] - 2025-12-30

### Added

- **Research ProviderSpec Alignment**: Research config now supports full ProviderSpec notation like consultation
  - `default_provider` accepts both simple IDs (`"gemini"`) and ProviderSpec (`"[cli]gemini:gemini-2.5-flash"`)
  - `consensus_providers` accepts mixed notation for flexible model selection per provider
  - New `ResearchConfig.get_default_provider_spec()` helper parses default provider
  - New `ResearchConfig.get_consensus_provider_specs()` helper parses consensus providers
  - New `ProviderSpec.parse_flexible()` method for backward-compatible parsing
  - Workflows (`chat`, `consensus`, `thinkdeep`, `ideate`) now extract models from specs
  - Added `[research]` section to sample config with notation examples

## [0.7.6] - 2025-12-30

### Fixed

- **Research Tools Feature Flag**: Fixed bug where `research_tools = true` in `[features]` config section was ignored
  - Root cause 1: `research_tools` flag was never registered in the feature flag registry
  - Root cause 2: `[features]` section in TOML config was not being read
  - Added flag registration in `research.py` following `provider.py` pattern
  - Added global override support to `FeatureFlagRegistry` for config-based flag settings
  - Added `[features]` section handling in `ServerConfig._load_toml()`
  - Added `FOUNDRY_MCP_FEATURES` environment variable support (format: `flag1=true,flag2=false`)

### Added

- **Feature Flag Global Overrides**: New methods on `FeatureFlagRegistry`:
  - `set_global_override(flag_name, enabled)` - Set config-based override for all clients
  - `clear_global_override(flag_name)` - Clear a global override
  - `clear_all_global_overrides()` - Clear all global overrides
  - `apply_config_overrides(features)` - Apply multiple overrides from config dict

### Dependencies

- Added `filelock>=3.20.1` as a required dependency

## [Unreleased]

## [0.8.1] - 2026-01-03

### Added

- **Batch Operations for Parallel Task Execution**: New actions for autonomous multi-task workflows
  - `prepare-batch`: Find independent tasks for parallel execution with file-path conflict detection
  - `start-batch`: Atomically start multiple tasks as in_progress (all-or-nothing validation)
  - `complete-batch`: Complete multiple tasks with partial failure support
  - `reset-batch`: Reset batch on failure, returning tasks to pending status
  - Token budget support for context-aware batch sizing
  - Stale task detection for tasks stuck in_progress beyond threshold
  - Dependency graph visualization in prepare-batch responses

- **Autonomous Session Context Tracking**: CLI context management for continuous task processing
  - `AutonomousSession` class for tracking batch state across operations
  - `ContextTracker` for managing session lifecycle and context limits
  - Integration with task router for session-aware operations

### Changed

- Enhanced development guide with batch operations documentation and usage examples

## [0.8.0] - 2026-01-01

### Added

- **Deep Research Workflow**: Multi-phase iterative research with parallel source gathering
  - Background async execution with immediate `research_id` return for non-blocking operation
  - Five-phase pipeline: decomposition → search → analyze → synthesize → report
  - Query decomposition with strategic sub-query generation based on research intent
  - Gap detection and iterative refinement loops for comprehensive coverage
  - Final synthesis with confidence scoring (low/medium/high/confirmed) and source citations
  - Crash handler infrastructure for session recovery
  - Research tool actions:
    - `deep-research`: Start, continue, or resume research sessions
    - `deep-research-status`: Poll running research status with phase progress
    - `deep-research-report`: Get final markdown report with citations and audit trail
    - `deep-research-list`: List sessions with cursor-based pagination
    - `deep-research-delete`: Clean up research sessions

- **Search Provider System**: Extensible multi-provider search architecture
  - **Google Custom Search**: Web search via Google's Custom Search JSON API
  - **Perplexity AI**: Sonar model integration for AI-powered search
  - **Tavily AI**: Search and extract modes with content analysis
  - **Semantic Scholar**: Academic paper search with citation metadata
  - Domain quality tiers (authoritative → unreliable) for source credibility scoring
  - Rate limiting and error handling per provider
  - Pluggable provider interface for custom backends

- **Research Configuration**: New `[research.deep]` config section
  - Per-provider API key configuration (Google, Perplexity, Tavily, Semantic Scholar)
  - Timeout, iteration, and concurrency controls
  - Source quality and domain tier preferences
  - Storage path and session TTL settings

- **Deep Research Documentation**: Comprehensive workflow documentation
  - `docs/concepts/deep_research_workflow.md`: Architecture and usage guide
  - `docs/examples/deep-research/`: Example research sessions and reports

- **Provider Availability Caching**: Cache provider detection results to speed up MCP tool calls
  - New `[providers] availability_cache_ttl` config option (default: 3600 seconds)
  - Reduces repeated calls from ~5s to ~0s

### Changed

- **Provider Model Validation Removed**: Model allowlists removed from all CLI providers
  - Providers no longer pre-register or validate model IDs against hardcoded lists
  - Any model string is now passed through to the underlying CLI for validation
  - Eliminates sync issues when providers release new models
  - Affected providers: `claude`, `gemini`, `codex`, `cursor-agent`
  - Default models remain as fallbacks: opus, pro, gpt-5.2, composer-1

- **BREAKING: Simplified Spec Templates**: Removed pre-baked spec templates (simple, medium, complex, security)
  - Only `empty` template is now supported - creates a blank spec with no phases
  - Use phase templates (`planning`, `implementation`, `testing`, `security`, `documentation`) to add structure
  - Default template changed from `medium` to `empty`
  - Mission statement no longer required (was required for medium/complex)
  - `_requires_rich_task_fields()` now checks explicit `complexity` metadata instead of template
  - Passing deprecated templates (simple, medium, complex, security) returns validation error

### Fixed

- **AI Consultation Config Loading**: Fixed issue where AI consultation features returned `model_used: "none"` because config was loaded from CWD instead of workspace path
  - `review.py`: Now loads `foundry-mcp.toml` from workspace path for fidelity reviews
  - `plan.py`: Added `_find_config_file()` helper to walk up directories and find config

### Migration

**Spec Templates:**
```python
# Old approach (no longer works)
authoring(action="spec-create", name="my-feature", template="medium", mission="...")

# New approach
authoring(action="spec-create", name="my-feature")
authoring(action="phase-template", template_action="apply", template_name="planning", spec_id="...")
authoring(action="phase-template", template_action="apply", template_name="implementation", spec_id="...")
```

**Deep Research:**
```python
# Start a deep research session (returns immediately with research_id)
research(action="deep-research", query="What are the best practices for LLM evaluation?")

# Poll for status
research(action="deep-research-status", research_id="...")

# Get final report when complete
research(action="deep-research-report", research_id="...")
```

## [0.7.0] - 2025-12-30

### Added

- **Research Router**: New unified research tool providing multi-model orchestration capabilities
  - **chat**: Single-model conversation with thread persistence
    - Thread creation with title and system prompt
    - Conversation continuation via thread_id
    - Token budgeting for context management
    - Thread CRUD operations (list, get, delete)
  - **consensus**: Multi-model parallel consultation with synthesis
    - Parallel execution via asyncio.gather with semaphore limiting
    - Four synthesis strategies: all_responses, synthesize, majority, first_valid
    - Partial failure handling with min_responses and require_all options
    - Configurable timeout per provider
  - **thinkdeep**: Hypothesis-driven systematic investigation
    - Investigation step execution with state persistence
    - Hypothesis creation and tracking with evidence accumulation
    - Confidence level progression (speculation -> confirmed)
    - Convergence detection based on depth and confidence
  - **ideate**: Creative brainstorming with idea clustering
    - Four-phase workflow: divergent, convergent, selection, elaboration
    - Multi-perspective idea generation
    - Automatic clustering and scoring
    - Detailed plan elaboration for selected clusters
- **ResearchConfig**: New configuration section for research workflows
  - Configurable storage path, TTL, max messages per thread
  - Default provider and consensus provider list
  - ThinkDeep max depth and Ideate perspectives
- **Research Data Models**: Pydantic models for all workflow states
  - Enums: WorkflowType, ConfidenceLevel, ConsensusStrategy, ThreadStatus, IdeationPhase
  - Conversation, ThinkDeep, Ideate, and Consensus state models
- **File-Based Memory Storage**: Persistent state management for research sessions
  - FileStorageBackend with CRUD operations
  - File locking via filelock for thread safety
  - TTL-based cleanup for expired sessions
- **Research Test Suite**: 149 tests covering models, memory, and router
- **Feature Flag**: `research_tools` flag (experimental) gates research tool access

## [0.6.0] - 2025-12-29

### Added

- **Notes Intake System**: Fast-capture queue for rapid idea/task capture with automatic triage workflow
  - **intake-add**: Add items to the intake queue with title, description, priority (p0-p4), tags, source, and requester fields
    - Idempotency key support for deduplication (checks last 100 items)
    - Tag normalization to lowercase
    - Full dry-run support for validation without persistence
  - **intake-list**: List pending intake items in FIFO order with cursor-based pagination
    - Configurable page size (1-200, default 50)
    - Efficient line-hint seeking with fallback to full scan
    - Returns total_count for queue size visibility
  - **intake-dismiss**: Mark items as dismissed with optional reason
    - Atomic file rewrite pattern for data integrity
    - Supports dry-run mode
  - JSONL-based storage at `specs/.notes/intake.jsonl` with fcntl file locking
  - Automatic file rotation at 1000 items or 1MB
  - Thread-safe and cross-process safe with 5-second lock timeout
  - Security hardening: path traversal prevention, prompt injection sanitization, control character stripping
  - Feature flag gated: `intake_tools` (experimental, opt-in)
- **Intake Schema**: JSON Schema for intake-v1 format with comprehensive validation constraints
- **Intake Documentation**: User guide at `docs/guides/intake.md`
- **RESOURCE_BUSY Error Code**: New error code for lock contention scenarios

## [0.5.1] - 2025-12-27

### Added

- **Phase Metadata Updates**: New `authoring action=phase-update-metadata` for updating phase-level metadata
  - Supports updating `estimated_hours`, `description`, and `purpose` fields
  - Full dry-run support for previewing changes
  - Tracks previous values for audit purposes
  - Core function `update_phase_metadata()` in `spec.py` with comprehensive validation

### Fixed

- **Lifecycle Tool Router Compatibility**: Fixed `_handle_move()` and other lifecycle handlers receiving unexpected keyword arguments (`force`, `to_folder`) from the unified router dispatch
  - All lifecycle handlers now accept full parameter set for router compatibility
  - Resolves errors like `_handle_move() got an unexpected keyword argument 'force'`

## [0.5.0] - 2025-12-27

### Added

- **Spec Modification Capabilities**: Complete implementation of dynamic spec modification (7 phases, 54 tasks)
  - **Task Hierarchy Mutations**: `task action=move` for repositioning tasks within/across phases with circular reference prevention
  - **Dependency Management**: `task action=add-dependency`, `task action=remove-dependency` for blocks/blocked_by/depends relationships
  - **Task Requirements**: `task action=add-requirement` for adding structured requirements to tasks
  - **Bulk Operations**: `authoring action=phase-add-bulk` for batch phase creation, `authoring action=phase-template` for applying predefined structures
  - **Metadata Batch Updates**: `task action=metadata-batch` with AND-based filtering by node_type, phase_id, or pattern regex
  - **Find-Replace**: `authoring action=spec-find-replace` with regex support and scope filtering for bulk spec modifications
  - **Spec Rollback**: `authoring action=spec-rollback` for restoring specs from automatic backups
  - **Spec History & Diff**: `spec action=history` for backup timeline, `spec action=diff` for comparing specs
  - **Validation Enhancements**: `spec action=completeness-check` with weighted scoring (0-100), `spec action=duplicate-detection` with configurable similarity threshold
- **Standardized Error Codes**: New `ErrorCode` enum with semantic error codes per 07-error-semantics.md
- **Contract Tests**: Comprehensive test suite for response-v2 envelope compliance across all phases

### Changed

- Updated capabilities manifest with 15 new actions documented
- Spec modification spec moved from pending to active (100% complete)

## [0.4.2] - 2025-12-24

### Added

- **Preflight Validation**: `authoring action=spec-create dry_run=true` now generates and validates the full spec, returning `is_valid`, `error_count`, `warning_count`, and detailed diagnostics before actual creation
- **Schema Introspection**: New `spec action=schema` returns all valid enum values (templates, node_types, statuses, task_categories, verification_types, journal_entry_types, blocker_types, status_folders) for LLM/client discovery

### Changed

- **Spec Field Requirements**: Medium/complex specs now require `metadata.mission` and task metadata for `task_category`, `description`, and `acceptance_criteria`; implementation/refactoring tasks must include `file_path`
- **Task Metadata Updates**: `task update-metadata` now accepts `acceptance_criteria` and aligns task category validation with the canonical spec categories

## [0.4.1] - 2025-12-24

### Added

- **Batch Metadata Utilities**: New task actions for bulk operations
  - `task action=metadata-batch`: Apply metadata updates to multiple nodes with AND-based filtering by `node_type`, `phase_id`, or `pattern` regex
  - `task action=fix-verification-types`: Auto-fix invalid/missing verification types on verify nodes with legacy mapping support
  - Both actions support `dry_run` mode for previewing changes
- **Phase-First Authoring**: New `authoring action=phase-add-bulk` for creating multiple phases at once with metadata defaults
- **Spec Mission Field**: Added `mission` field to spec metadata schema for concise goal statements
- **Workflow Timeout Override**: AI consultation now supports workflow-specific timeout configuration

### Changed

- **JSON Output Optimization**: CLI and MCP server now emit minified JSON (no indentation) for smaller payloads
- **Fidelity Review Improvements**: Better path resolution with workspace_root support, graceful handling of non-JSON provider responses
- **Provider Configuration**: Updated OpenCode model IDs and default model; reordered provider priority
- **Claude Provider Tests**: Updated to use Haiku model for faster test execution

### Fixed

- Fixed parameter filtering in error_list handler to prevent unexpected argument errors
- Fixed duplicate file paths in fidelity review implementation artifacts
- Synced `__init__.py` version with `pyproject.toml`

## [0.4.0] - 2025-12-23

### Changed

- **Verification Types**: Aligned task API and spec validator to use canonical values (`run-tests`, `fidelity`, `manual`)
  - Task API now accepts `run-tests`, `fidelity`, `manual` (previously `auto`, `manual`, `none`)
  - Spec validator updated to match canonical schema values
  - Legacy values automatically mapped: `test` → `run-tests`, `auto` → `run-tests`

### Added

- **Auto-fix for `INVALID_VERIFICATION_TYPE`**: Specs with legacy verification types are now auto-fixable via `validate-fix`
- **Auto-fix for `INVALID_ROOT_PARENT`**: Specs where spec-root has non-null parent are now auto-fixable

### Removed

- Removed `foundry-mcp-ctl` package and mode-toggling feature - server now always runs with all tools registered

## [0.3.4] - 2025-12-21

_Note: Mode toggling features added in this version were subsequently removed._

## [0.3.3] - 2025-12-17

### Changed
- **Dashboard**: Refactored pages module to views with cleaner organization
- **Dashboard**: Improved data stores with better caching and filtering
- **Observability**: Added action label to tool metrics for router-level granularity
- **Providers**: Codex CLI now ignores unsupported parameters (warning instead of error)

### Added
- Dashboard PID file tracking for cross-CLI process management
- Tool usage dashboard view with action-level breakdown
- OpenCode Node.js wrapper for subprocess execution
- Integration tests for provider smoke testing, fidelity review flow, and plan review flow

### Fixed
- Codex provider environment handling (unsets OPENAI_API_KEY/OPENAI_BASE_URL that interfere with CLI)
- Minor fixes to Claude and Gemini providers

## [0.3.2] - 2025-12-16

### Added
- Launcher script (`bin/foundry-mcp`) for configurable Python interpreter selection
- `FOUNDRY_MCP_PYTHON` environment variable to override the default Python interpreter

### Fixed
- Removed duplicate `spec_id` and `node_id` fields from task progress response

## [0.3.1] - 2025-12-16

### Removed
- Removed `code` unified tool (find-class, find-function, callers, callees, trace, impact actions) from MCP surface. Unified manifest reduced from 17 to 16 tools.

## [0.3.0] - 2025-12-15

### Changed
- Consolidated the MCP tool surface into 17 unified routers (tool + `action`) and aligned CLI/MCP naming.
- Updated documentation and manifests to reflect the unified router contract.

### Added
- New completed specs documenting MCP tool consolidation and removal of docquery/rendering/docgen.
- Unified-manifest budget telemetry (Prometheus metrics, recording rules, alerting rules, and dashboard panels).

### Removed
- Legacy per-tool MCP modules and legacy CLI command surfaces in favor of unified routers.
- Docquery/rendering/docgen modules and generated docs previously under `docs/generated/`.

## [0.2.1] - 2025-12-08

### Changed
- **Dashboard**: Replaced aiohttp+vanilla JS dashboard with Streamlit for better visualizations and interactivity
- Dashboard dependencies changed from `aiohttp` to `streamlit`, `plotly`, `pandas`
- Default dashboard port changed from 8080 to 8501 (Streamlit default)

### Added
- New Streamlit dashboard with 5 pages: Overview, Errors, Metrics, Providers, SDD Workflow
- Interactive Plotly charts with zoom, pan, and hover tooltips
- Data export functionality (CSV/JSON download buttons)
- Cached data access via `@st.cache_data` for performance
- CLI commands: `dashboard start`, `dashboard stop`, `dashboard status`
- New SDD Workflow page for spec progress tracking, phase burndown, task status
- Plan review tool (`plan-review`) for AI-assisted specification review

### Removed
- Old aiohttp-based dashboard server and static JS/CSS files
