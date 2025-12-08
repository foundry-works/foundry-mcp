# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Added `phase-add` MCP tool, core helper, and CLI command for creating new phases with verification scaffolding, plus refreshed docs, manifest metadata, and tests to cover the new workflow.

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
