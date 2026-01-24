# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
