# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Validation module (`foundry_mcp.core.validation`):
  - `validate_spec` - Validate spec structure and return structured Diagnostic objects
  - `get_fix_actions` - Generate fix actions from validation diagnostics
  - `apply_fixes` - Apply auto-fixes to spec files with backup support
  - `calculate_stats` - Calculate comprehensive spec statistics
  - Structured diagnostics with code, message, severity, category, location, suggested_fix
  - Auto-fix support for common issues (counts, hierarchy, metadata)
- Journal module (`foundry_mcp.core.journal`):
  - `add_journal_entry` - Add timestamped journal entries with entry types
  - `get_journal_entries` - Retrieve and filter journal entries
  - `mark_blocked` - Mark tasks as blocked with blocker metadata
  - `unblock` - Unblock tasks with resolution tracking
  - `list_blocked_tasks` - List all blocked tasks in a spec
  - `find_unjournaled_tasks` - Find completed tasks needing journal entries
  - Entry types: status_change, deviation, blocker, decision, note
  - Blocker types: dependency, technical, resource, decision
- MCP validation tools (`foundry_mcp.tools.validation`):
  - `foundry_validate_spec` - Validate spec and return structured diagnostics
  - `foundry_fix_spec` - Apply auto-fixes with dry-run support
  - `foundry_spec_stats` - Get comprehensive spec statistics
  - `foundry_validate_and_fix` - Combined validation and fix in one operation
- MCP journal tools (`foundry_mcp.tools.journal`):
  - `foundry_add_journal` - Add journal entries with optional task association
  - `foundry_get_journal` - Retrieve journal entries with filtering
  - `foundry_mark_blocked` - Mark tasks as blocked with metadata
  - `foundry_unblock` - Unblock tasks and track resolution
  - `foundry_list_blocked` - List all blocked tasks
  - `foundry_unjournaled_tasks` - Find tasks needing journal entries
- Rendering module (`foundry_mcp.core.rendering`):
  - `render_spec_to_markdown` - Generate formatted markdown with RenderOptions
  - `render_progress_bar` - Text-based progress visualization
  - `render_task_list` - Flat task listing with status filtering
  - `get_status_icon` - Status to icon mapping (⏳🔄✅🚫❌)
  - RenderOptions dataclass with mode, include_metadata, include_progress, etc.
- Lifecycle module (`foundry_mcp.core.lifecycle`):
  - `move_spec` - Move specs between folders with transition validation
  - `activate_spec` - Move pending to active
  - `complete_spec` - Move to completed with force option
  - `archive_spec` - Move to archived
  - `get_lifecycle_state` - Get folder, progress, can_complete/can_archive
  - `list_specs_by_folder` - Organized spec listing
  - MoveResult and LifecycleState dataclasses
- MCP rendering tools (`foundry_mcp.tools.rendering`):
  - `foundry_render_spec` - Render spec to markdown with mode options
  - `foundry_render_progress` - ASCII progress bars for spec and phases
  - `foundry_list_tasks` - Flat task list with filtering
- MCP lifecycle tools (`foundry_mcp.tools.lifecycle`):
  - `foundry_move_spec` - Move spec between folders
  - `foundry_activate_spec` - Activate pending spec
  - `foundry_complete_spec` - Mark spec completed
  - `foundry_archive_spec` - Archive spec
  - `foundry_lifecycle_state` - Get lifecycle state and transition eligibility
  - `foundry_list_specs_by_folder` - List specs by folder
- Task operations module (`foundry_mcp.core.task`):
  - `get_next_task` - Find next actionable task based on status and dependencies
  - `check_dependencies` - Analyze blocking and soft dependencies
  - `prepare_task` - Prepare complete context for task implementation
  - `get_previous_sibling` - Get context from previously executed sibling task
  - `get_parent_context` - Get parent task metadata and position
  - `get_phase_context` - Get current phase progress and blockers
  - `get_task_journal_summary` - Get journal entries for a task
  - `is_unblocked` / `is_in_current_phase` - Dependency and phase helpers
- Progress calculation module (`foundry_mcp.core.progress`):
  - `recalculate_progress` - Recursively update task counts
  - `update_parent_status` - Propagate status changes up hierarchy
  - `get_progress_summary` - Get completion percentages and stats
  - `list_phases` - List all phases with progress
  - `get_task_counts_by_status` - Count tasks by status
- MCP task tools (`foundry_mcp.tools.tasks`):
  - `foundry_prepare_task` - Prepare task with full context
  - `foundry_next_task` - Find next actionable task
  - `foundry_task_info` - Get detailed task information
  - `foundry_check_deps` - Check dependency status
  - `foundry_update_status` - Update task status
  - `foundry_complete_task` - Mark task complete with journal
  - `foundry_start_task` - Start working on a task
  - `foundry_progress` - Get spec/phase progress
- Unit tests for task operations (30 tests)

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
