# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4] - 2025-12-21

### Added

- **Mode toggling**: New `foundry-mcp-ctl` package enables runtime switching between full (16 tools) and minimal (1 wake tool) modes
  - `foundry-mcp-ctl wrap` - Wrapper that manages server lifecycle based on mode
  - `foundry-mcp-ctl helper` - MCP server providing `set_sdd_mode` and `get_sdd_mode` tools
- Wake tool (`sdd_wake`) for minimal mode - surfaces available actions and prompts mode switch
- `FOUNDRY_MODE` environment variable: `full` (default) or `minimal`
- New CLI entry point: `foundry-mcp-ctl`

### Changed

- `register_unified_tools()` now respects `FOUNDRY_MODE` to conditionally load tool routers
- README documents mode toggling configuration for context token savings

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
