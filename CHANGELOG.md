# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-25

### Added
- Initial project setup with pyproject.toml and hatchling build system
- Core spec operations module (`foundry_mcp.core.spec`):
  - `load_spec` - Load JSON spec files by ID or path
  - `save_spec` - Save specs with atomic writes and automatic backups
  - `find_spec_file` - Locate spec files across status folders
  - `find_specs_directory` - Auto-discover specs directory
  - `list_specs` - List specs with filtering by status
  - `get_node` / `update_node` - Hierarchy node operations
- Package structure following Python best practices
- FastMCP and MCP dependencies for server implementation

### Technical Decisions
- Extracted core spec operations from claude-sdd-toolkit as standalone module
- Removed external dependencies for portability
- Atomic file writes with `.tmp` extension for data safety
- Automatic backup creation in `.backups/` directory before saves
