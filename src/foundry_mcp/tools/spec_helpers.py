"""
Spec helper tools for foundry-mcp.

Provides MCP tools for spec discovery, validation, and analysis.
These tools provide file relationship discovery, pattern matching,
dependency cycle detection, and path validation using direct Python
API calls to core modules.
"""

import glob as glob_module
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.spec import find_specs_directory, find_spec_file, load_spec

logger = logging.getLogger(__name__)

# Metrics singleton for spec helper tools
_metrics = get_metrics()


def register_spec_helper_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register spec helper tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-find-related-files",
    )
    def spec_find_related_files(
        file_path: str,
        spec_id: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Locate files referenced by a spec node or related to a source file.

        Discovers relationships between source files and specification nodes
        by searching spec metadata for file path references.

        WHEN TO USE:
        - Finding files associated with a spec task
        - Discovering file relationships before making changes
        - Understanding spec-to-code mappings
        - Validating that referenced files exist

        Args:
            file_path: Source file path to find relationships for
            spec_id: Optional spec ID to narrow search scope
            include_metadata: Include additional metadata about relationships

        Returns:
            JSON object with related file information:
            - file_path: The queried file path
            - related_files: List of related file objects with paths and relationship types
            - spec_references: List of specs that reference this file
            - total_count: Number of related files found
        """
        tool_name = "find_related_files"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not file_path:
                return asdict(error_response(
                    "file_path is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a file_path parameter",
                ))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-related-files",
                action="find_related",
                file_path=file_path,
                spec_id=spec_id,
            )

            # Resolve workspace path
            ws_path = Path.cwd()

            # Find specs directory
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    data={"file_path": file_path, "workspace": str(ws_path)},
                ))

            # Normalize the file path for comparison
            normalized_file_path = str(Path(file_path).resolve())

            related_files: List[Dict[str, Any]] = []
            spec_references: List[Dict[str, Any]] = []

            # Determine which specs to search
            if spec_id:
                # Search only the specified spec
                spec_file = find_spec_file(spec_id, specs_dir)
                if spec_file:
                    spec_files = [spec_file]
                else:
                    spec_files = []
            else:
                # Search all specs
                spec_files = []
                for folder in ["active", "pending", "completed", "archived"]:
                    folder_path = specs_dir / folder
                    if folder_path.exists():
                        spec_files.extend(folder_path.glob("*.json"))

            # Search each spec for file references
            for spec_file in spec_files:
                spec_data = load_spec(spec_file)
                if not spec_data:
                    continue

                current_spec_id = spec_data.get("spec_id", spec_file.stem)
                hierarchy = spec_data.get("hierarchy", {})

                # Search hierarchy for file_path references
                for node_id, node in hierarchy.items():
                    metadata = node.get("metadata", {})
                    node_file_path = metadata.get("file_path", "")

                    # Check if this node references the target file
                    if node_file_path:
                        node_path_resolved = str(Path(node_file_path).resolve()) if not Path(node_file_path).is_absolute() else node_file_path

                        if normalized_file_path == node_path_resolved or file_path in node_file_path:
                            spec_references.append({
                                "spec_id": current_spec_id,
                                "node_id": node_id,
                                "title": node.get("title", ""),
                                "relationship": "references",
                            })

                        # Also track related files from the same spec
                        if node_file_path not in [f.get("path") for f in related_files]:
                            related_files.append({
                                "path": node_file_path,
                                "spec_id": current_spec_id,
                                "node_id": node_id,
                                "relationship": "sibling" if spec_references else "same_spec",
                            })

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"spec_helpers.{tool_name}.duration_ms", duration_ms)

            return asdict(success_response(
                file_path=file_path,
                related_files=related_files,
                spec_references=spec_references,
                total_count=len(related_files),
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception("Unexpected error in spec-find-related-files")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="spec helpers"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-find-patterns",
    )
    def spec_find_patterns(
        pattern: str,
        directory: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Search specs and codebase for structural or code patterns.

        Searches across spec contents and source files using glob patterns.
        Returns matching files and locations.

        WHEN TO USE:
        - Finding files matching a specific pattern (e.g., "*.spec.ts")
        - Searching for structural patterns in the codebase
        - Discovering test files or configuration files
        - Auditing file organization

        Args:
            pattern: Glob pattern to search for (e.g., "*.ts", "src/**/*.spec.ts")
            directory: Optional directory to scope the search
            include_metadata: Include additional metadata about the search

        Returns:
            JSON object with pattern match results:
            - pattern: The search pattern used
            - matches: List of matching file paths
            - total_count: Number of matches found
        """
        tool_name = "find_patterns"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not pattern:
                return asdict(error_response(
                    "pattern is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a glob pattern (e.g., '*.ts', 'src/**/*.py')",
                ))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-patterns",
                action="find_patterns",
                pattern=pattern,
                directory=directory,
            )

            # Resolve search directory
            search_dir = Path(directory) if directory else Path.cwd()
            if not search_dir.exists():
                return asdict(error_response(
                    f"Directory does not exist: {search_dir}",
                    error_code="NOT_FOUND",
                    error_type="validation",
                    data={"directory": str(search_dir)},
                    remediation="Provide a valid directory path",
                ))

            # Use glob to find matching files
            # Handle both relative and absolute patterns
            if pattern.startswith("/"):
                # Absolute pattern
                glob_pattern = pattern
            else:
                # Relative pattern - combine with directory
                glob_pattern = str(search_dir / pattern)

            # Use recursive glob
            matches: List[str] = []
            for match in glob_module.glob(glob_pattern, recursive=True):
                match_path = Path(match)
                if match_path.is_file():
                    # Return relative paths for cleaner output
                    try:
                        rel_path = match_path.relative_to(search_dir)
                        matches.append(str(rel_path))
                    except ValueError:
                        # Path is not relative to search_dir
                        matches.append(str(match_path))

            # Sort matches for consistent output
            matches.sort()

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"spec_helpers.{tool_name}.duration_ms", duration_ms)

            response_data: Dict[str, Any] = {
                "pattern": pattern,
                "matches": matches,
                "total_count": len(matches),
                "duration_ms": round(duration_ms, 2),
            }

            if directory:
                response_data["directory"] = str(search_dir)

            return asdict(success_response(**response_data))

        except Exception as e:
            logger.exception("Unexpected error in spec-find-patterns")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="spec helpers"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-detect-cycles",
    )
    def spec_detect_cycles(
        spec_id: str,
        include_metadata: bool = False,
    ) -> dict:
        """
        Detect cyclic dependencies in a specification's task dependency graph.

        Analyzes task dependencies using DFS to identify any circular references
        that would prevent task completion.

        WHEN TO USE:
        - Validating a specification before starting implementation
        - Debugging blocked tasks that can't be started
        - Auditing dependency structure after spec modifications
        - Ensuring task graph is acyclic before phase planning

        Args:
            spec_id: The specification ID to analyze
            include_metadata: Include additional metadata about the analysis

        Returns:
            JSON object with cycle detection results:
            - spec_id: The analyzed specification ID
            - has_cycles: Boolean indicating if cycles were detected
            - cycles: List of detected cycles (each cycle is a list of task IDs)
            - cycle_count: Number of cycles detected
            - affected_tasks: List of task IDs involved in cycles
        """
        tool_name = "detect_cycles"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter",
                ))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-detect-cycles",
                action="detect_cycles",
                spec_id=spec_id,
            )

            # Resolve workspace path
            ws_path = Path.cwd()

            # Find specs directory
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                ))

            # Find and load spec
            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Specification '{spec_id}' not found",
                    error_code="SPEC_NOT_FOUND",
                    error_type="not_found",
                    data={"spec_id": spec_id},
                    remediation="Verify the spec ID exists using spec-list",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            hierarchy = spec_data.get("hierarchy", {})

            # Build dependency graph from blocked_by relationships
            # node_id -> list of nodes it depends on (blocked_by)
            dep_graph: Dict[str, List[str]] = {}
            for node_id, node in hierarchy.items():
                deps = node.get("dependencies", {})
                blocked_by = deps.get("blocked_by", [])
                dep_graph[node_id] = blocked_by

            # Detect cycles using DFS with path tracking
            cycles: List[List[str]] = []
            visited: Set[str] = set()
            rec_stack: Set[str] = set()

            def find_cycles_from(node_id: str, path: List[str]) -> None:
                """DFS to find all cycles starting from node_id."""
                if node_id in rec_stack:
                    # Found a cycle - extract it from path
                    cycle_start = path.index(node_id)
                    cycle = path[cycle_start:] + [node_id]
                    cycles.append(cycle)
                    return

                if node_id in visited:
                    return

                visited.add(node_id)
                rec_stack.add(node_id)

                for dep_id in dep_graph.get(node_id, []):
                    find_cycles_from(dep_id, path + [node_id])

                rec_stack.remove(node_id)

            # Run DFS from all nodes
            for node_id in dep_graph:
                if node_id not in visited:
                    find_cycles_from(node_id, [])

            # Deduplicate cycles (same cycle can be found from different starting points)
            unique_cycles: List[List[str]] = []
            seen_cycle_sets: List[Set[str]] = []
            for cycle in cycles:
                cycle_set = set(cycle)
                if cycle_set not in seen_cycle_sets:
                    seen_cycle_sets.append(cycle_set)
                    unique_cycles.append(cycle)

            # Extract affected tasks
            affected_tasks: List[str] = []
            seen_affected: Set[str] = set()
            for cycle in unique_cycles:
                for task_id in cycle:
                    if task_id not in seen_affected:
                        seen_affected.add(task_id)
                        affected_tasks.append(task_id)

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"spec_helpers.{tool_name}.duration_ms", duration_ms)

            return asdict(success_response(
                spec_id=spec_id,
                has_cycles=len(unique_cycles) > 0,
                cycles=unique_cycles,
                cycle_count=len(unique_cycles),
                affected_tasks=affected_tasks,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception("Unexpected error in spec-detect-cycles")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="spec helpers"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="spec-validate-paths",
    )
    def spec_validate_paths(
        paths: List[str],
        base_directory: Optional[str] = None,
        include_metadata: bool = False,
    ) -> dict:
        """
        Validate that file paths exist on disk.

        Checks that file references in specifications or code actually
        exist in the filesystem using pathlib.

        WHEN TO USE:
        - Validating spec file references before implementation
        - Auditing broken file references after refactoring
        - Pre-flight checks before large-scale changes
        - Ensuring spec metadata file_path entries are current

        Args:
            paths: List of file paths to validate
            base_directory: Optional base directory for resolving relative paths
            include_metadata: Include additional metadata about the validation

        Returns:
            JSON object with path validation results:
            - paths_checked: Number of paths validated
            - valid_paths: List of paths that exist
            - invalid_paths: List of paths that do not exist
            - all_valid: Boolean indicating if all paths are valid
            - valid_count: Number of valid paths
            - invalid_count: Number of invalid paths
        """
        tool_name = "validate_paths"
        start_time = time.perf_counter()

        try:
            # Validate required parameters
            if not paths:
                return asdict(error_response(
                    "paths is required and must be a non-empty list",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a list of file paths to validate",
                ))

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-validate-paths",
                action="validate_paths",
                path_count=len(paths),
                base_directory=base_directory,
            )

            # Resolve base directory
            base_dir = Path(base_directory) if base_directory else Path.cwd()

            valid_paths: List[str] = []
            invalid_paths: List[str] = []

            # Check each path
            for path_str in paths:
                path = Path(path_str)

                # If path is relative, resolve against base directory
                if not path.is_absolute():
                    path = base_dir / path

                # Check if path exists
                if path.exists():
                    valid_paths.append(path_str)
                else:
                    invalid_paths.append(path_str)

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"spec_helpers.{tool_name}.duration_ms", duration_ms)

            response_data: Dict[str, Any] = {
                "paths_checked": len(paths),
                "valid_paths": valid_paths,
                "invalid_paths": invalid_paths,
                "all_valid": len(invalid_paths) == 0,
                "valid_count": len(valid_paths),
                "invalid_count": len(invalid_paths),
                "duration_ms": round(duration_ms, 2),
            }

            if base_directory:
                response_data["base_directory"] = str(base_dir)

            return asdict(success_response(**response_data))

        except Exception as e:
            logger.exception("Unexpected error in spec-validate-paths")
            _metrics.counter(f"spec_helpers.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="spec helpers"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
