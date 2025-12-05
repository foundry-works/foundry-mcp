"""
Analysis tools for foundry-mcp.

Provides MCP tools for analyzing SDD specifications and performing
structural/quality analysis on specs and their dependencies.
Uses direct Python API calls to core modules.
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.spec import find_specs_directory, find_spec_file, load_spec
from foundry_mcp.core.task import check_dependencies

logger = logging.getLogger(__name__)

# Metrics singleton for analysis tools
_metrics = get_metrics()


def register_analysis_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register analysis tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="spec-analyze",
    )
    def spec_analyze(
        directory: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Perform deep structural and quality analysis on SDD specifications.

        Analyzes the specification directory to provide heuristics about
        spec health, structure, and documentation availability.

        WHEN TO USE:
        - Evaluating spec health before starting work
        - Checking for documentation availability
        - Understanding project SDD structure
        - Auditing spec organization and quality

        Args:
            directory: Directory to analyze (default: current directory)
            path: Project root path (default: current directory)

        Returns:
            JSON object with analysis results:
            - directory: Analyzed directory path
            - has_specs: Whether specs directory exists
            - documentation_available: Whether generated docs exist
            - Additional heuristics from sdd analyze
        """
        tool_name = "spec_analyze"
        start_time = time.perf_counter()

        try:
            # Resolve workspace path
            ws_path = Path(directory or path or ".").resolve()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-analyze",
                action="analyze_specs",
                directory=str(ws_path),
            )

            # Find specs directory
            specs_dir = find_specs_directory(ws_path)
            has_specs = specs_dir is not None

            # Build analysis results
            analysis_data: Dict[str, Any] = {
                "directory": str(ws_path),
                "has_specs": has_specs,
                "specs_dir": str(specs_dir) if specs_dir else None,
            }

            if has_specs and specs_dir:
                # Count specs by folder
                folder_counts: Dict[str, int] = {}
                for folder in ["active", "pending", "completed", "archived"]:
                    folder_path = specs_dir / folder
                    if folder_path.exists():
                        spec_files = list(folder_path.glob("*.json"))
                        folder_counts[folder] = len(spec_files)
                    else:
                        folder_counts[folder] = 0

                analysis_data["spec_counts"] = folder_counts
                analysis_data["total_specs"] = sum(folder_counts.values())

                # Check for documentation
                docs_dir = specs_dir / ".human-readable"
                analysis_data["documentation_available"] = docs_dir.exists() and any(docs_dir.glob("*.md"))

                # Check for codebase.json
                codebase_json = ws_path / "docs" / "codebase.json"
                analysis_data["codebase_docs_available"] = codebase_json.exists()

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"analysis.{tool_name}.duration_ms", duration_ms)

            return asdict(success_response(
                duration_ms=round(duration_ms, 2),
                **analysis_data,
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="analysis"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))

    @canonical_tool(
        mcp,
        canonical_name="review-parse-feedback",
    )
    def review_parse_feedback(
        spec_id: str,
        review_path: str,
        output_path: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Transform review feedback into structured modification actions.

        Converts review markdown or JSON into a structured suggestions file
        that can be applied to modify the specification.

        WHEN TO USE:
        - Processing AI-generated review feedback
        - Converting review notes into actionable changes
        - Preparing review feedback for automated application
        - Transforming unstructured feedback into spec modifications

        Args:
            spec_id: Specification ID being reviewed
            review_path: Path to review report file (.md or .json)
            output_path: Output path for suggestions JSON (default: <review>.suggestions.json)
            path: Project root path (default: current directory)

        Returns:
            JSON object with parsing results:
            - suggestions_count: Number of suggestions extracted
            - output_file: Path to generated suggestions file
            - categories: Breakdown of suggestion types
        """
        # Review feedback parsing requires complex text/markdown parsing with
        # AI-powered understanding of review comments. This functionality is
        # not available as a direct core API.
        # Use the sdd-toolkit:sdd-modify skill for applying review feedback.
        return asdict(
            error_response(
                "Review feedback parsing requires complex text/markdown parsing. "
                "Use the sdd-toolkit:sdd-modify skill to apply review feedback.",
                error_code="NOT_IMPLEMENTED",
                error_type="unavailable",
                data={
                    "spec_id": spec_id,
                    "review_path": review_path,
                    "output_path": output_path,
                    "alternative": "sdd-toolkit:sdd-modify skill",
                    "feature_status": "requires_complex_parsing",
                },
                remediation="Use the sdd-toolkit:sdd-modify skill which provides "
                "review feedback application with proper parsing support.",
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="spec-analyze-deps",
    )
    def spec_analyze_deps(
        spec_id: str,
        bottleneck_threshold: Optional[int] = None,
        path: Optional[str] = None,
    ) -> dict:
        """
        Inspect dependency graph health for an SDD specification.

        Analyzes dependency relationships between tasks, identifies bottlenecks,
        and surfaces potential blocking issues.

        WHEN TO USE:
        - Identifying blocking tasks before starting work
        - Finding bottlenecks in the dependency graph
        - Understanding critical path for completion
        - Auditing spec structure for dependency issues

        Args:
            spec_id: Specification ID or path to analyze
            bottleneck_threshold: Minimum tasks blocked to flag as bottleneck (default: 3)
            path: Project root path (default: current directory)

        Returns:
            JSON object with dependency analysis:
            - dependency_count: Total number of dependencies
            - bottlenecks: Tasks blocking many others
            - circular_deps: Any circular dependency issues
            - critical_path: Tasks on the longest dependency chain
        """
        tool_name = "spec_analyze_deps"
        start_time = time.perf_counter()
        threshold = bottleneck_threshold if bottleneck_threshold is not None else 3

        try:
            # Validate required parameters
            if not spec_id:
                return asdict(error_response(
                    "spec_id is required",
                    error_code="MISSING_REQUIRED",
                    error_type="validation",
                    remediation="Provide a spec_id parameter (e.g., my-feature-spec)",
                ))

            # Resolve workspace path
            ws_path = Path(path) if path else Path.cwd()

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-analyze-deps",
                action="analyze_dependencies",
                spec_id=spec_id,
            )

            # Find and load spec
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                return asdict(error_response(
                    f"Specs directory not found in {ws_path}",
                    data={"spec_id": spec_id, "workspace": str(ws_path)},
                ))

            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
                return asdict(error_response(
                    f"Spec '{spec_id}' not found",
                    error_code="NOT_FOUND",
                    error_type="analysis",
                    data={"spec_id": spec_id, "specs_dir": str(specs_dir)},
                    remediation="Ensure the spec exists in specs/active or specs/pending",
                ))

            spec_data = load_spec(spec_file)
            if not spec_data:
                return asdict(error_response(
                    f"Failed to load spec '{spec_id}'",
                    data={"spec_id": spec_id, "spec_file": str(spec_file)},
                ))

            hierarchy = spec_data.get("hierarchy", {})

            # Analyze dependencies
            dependency_count = 0
            bottlenecks: List[Dict[str, Any]] = []
            blocks_count: Dict[str, int] = {}  # task_id -> number of tasks it blocks

            # Build dependency graph and count
            for node_id, node in hierarchy.items():
                deps = node.get("dependencies", {})
                blocked_by = deps.get("blocked_by", [])
                blocks = deps.get("blocks", [])

                dependency_count += len(blocked_by)

                # Track how many tasks each task blocks
                for blocker_id in blocked_by:
                    blocks_count[blocker_id] = blocks_count.get(blocker_id, 0) + 1

            # Find bottlenecks (tasks that block many others)
            for task_id, count in blocks_count.items():
                if count >= threshold:
                    task = hierarchy.get(task_id, {})
                    bottlenecks.append({
                        "task_id": task_id,
                        "title": task.get("title", ""),
                        "status": task.get("status", ""),
                        "blocks_count": count,
                    })

            # Sort bottlenecks by blocks_count descending
            bottlenecks.sort(key=lambda x: x["blocks_count"], reverse=True)

            # Check for cycles using DFS
            visited: set = set()
            rec_stack: set = set()
            circular_deps: List[str] = []

            def detect_cycle(node_id: str, path: List[str]) -> bool:
                visited.add(node_id)
                rec_stack.add(node_id)

                node = hierarchy.get(node_id, {})
                for child_id in node.get("children", []):
                    if child_id not in visited:
                        if detect_cycle(child_id, path + [child_id]):
                            return True
                    elif child_id in rec_stack:
                        circular_deps.append(" -> ".join(path + [child_id]))
                        return True

                rec_stack.remove(node_id)
                return False

            if "spec-root" in hierarchy:
                detect_cycle("spec-root", ["spec-root"])

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "success"})
            _metrics.timer(f"analysis.{tool_name}.duration_ms", duration_ms)

            return asdict(success_response(
                spec_id=spec_id,
                dependency_count=dependency_count,
                bottlenecks=bottlenecks,
                bottleneck_threshold=threshold,
                circular_deps=circular_deps,
                has_cycles=len(circular_deps) > 0,
                duration_ms=round(duration_ms, 2),
            ))

        except Exception as e:
            logger.exception(f"Unexpected error in {tool_name}")
            _metrics.counter(f"analysis.{tool_name}", labels={"status": "error"})
            return asdict(error_response(
                sanitize_error_message(e, context="analysis"),
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
