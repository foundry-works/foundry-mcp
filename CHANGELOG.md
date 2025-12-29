# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2025-12-29

### Added

- **Bikelane Intake System**: Fast-capture queue for rapid idea/task capture with automatic triage workflow
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
  - JSONL-based storage at `specs/.bikelane/intake.jsonl` with fcntl file locking
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
