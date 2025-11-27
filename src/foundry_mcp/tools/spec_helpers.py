"""
Spec helper tools for foundry-mcp.

Provides MCP tools for spec discovery, validation, and analysis.
These tools wrap SDD CLI commands to provide file relationship discovery,
pattern matching, dependency cycle detection, and path validation.
"""

import json
import logging
import subprocess
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics

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

        Wraps the SDD CLI find-related-files command to discover relationships
        between source files and specification nodes. Returns file paths that
        are referenced in spec metadata or structurally related.

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
        try:
            # Build command
            cmd = ["sdd", "find-related-files", file_path, "--json"]

            if spec_id:
                cmd.extend(["--spec-id", spec_id])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-related-files",
                action="find_related",
                file_path=file_path,
                spec_id=spec_id,
            )

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response data
                related_files: List[Dict[str, Any]] = output_data.get("related_files", [])
                spec_references: List[Dict[str, Any]] = output_data.get("spec_references", [])

                data: Dict[str, Any] = {
                    "file_path": file_path,
                    "related_files": related_files,
                    "spec_references": spec_references,
                    "total_count": len(related_files),
                }

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter("spec_helpers.find_related_files", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter("spec_helpers.find_related_files", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to find related files: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the file path exists and SDD CLI is available",
                ))

        except subprocess.TimeoutExpired:
            _metrics.counter("spec_helpers.find_related_files", labels={"status": "timeout"})
            return asdict(error_response(
                "Command timed out after 30 seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with a smaller scope or check system resources",
            ))
        except FileNotFoundError:
            _metrics.counter("spec_helpers.find_related_files", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-find-related-files")
            _metrics.counter("spec_helpers.find_related_files", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
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

        Wraps the SDD CLI find-pattern command to search across spec contents
        and source files using glob patterns. Returns matching files and locations.

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
        try:
            # Build command
            cmd = ["sdd", "find-pattern", pattern, "--json"]

            if directory:
                cmd.extend(["--directory", directory])

            # Log the operation
            audit_log(
                "tool_invocation",
                tool="spec-find-patterns",
                action="find_patterns",
                pattern=pattern,
                directory=directory,
            )

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse the JSON output
            if result.returncode == 0:
                try:
                    output_data = json.loads(result.stdout) if result.stdout.strip() else {}
                except json.JSONDecodeError:
                    output_data = {}

                # Build response data
                matches: List[str] = output_data.get("matches", output_data.get("files", []))

                data: Dict[str, Any] = {
                    "pattern": pattern,
                    "matches": matches,
                    "total_count": len(matches),
                }

                if directory:
                    data["directory"] = directory

                if include_metadata:
                    data["metadata"] = {
                        "command": " ".join(cmd),
                        "exit_code": result.returncode,
                    }

                # Track metrics
                _metrics.counter("spec_helpers.find_patterns", labels={"status": "success"})

                return asdict(success_response(data))
            else:
                # Command failed
                error_msg = result.stderr.strip() if result.stderr else "Command failed"
                _metrics.counter("spec_helpers.find_patterns", labels={"status": "error"})

                return asdict(error_response(
                    f"Failed to find patterns: {error_msg}",
                    error_code="COMMAND_FAILED",
                    error_type="internal",
                    remediation="Check that the pattern is valid and SDD CLI is available",
                ))

        except subprocess.TimeoutExpired:
            _metrics.counter("spec_helpers.find_patterns", labels={"status": "timeout"})
            return asdict(error_response(
                "Command timed out after 30 seconds",
                error_code="TIMEOUT",
                error_type="unavailable",
                remediation="Try with a more specific pattern or smaller scope",
            ))
        except FileNotFoundError:
            _metrics.counter("spec_helpers.find_patterns", labels={"status": "cli_not_found"})
            return asdict(error_response(
                "SDD CLI not found in PATH",
                error_code="CLI_NOT_FOUND",
                error_type="internal",
                remediation="Ensure SDD CLI is installed and available in PATH",
            ))
        except Exception as e:
            logger.exception("Unexpected error in spec-find-patterns")
            _metrics.counter("spec_helpers.find_patterns", labels={"status": "error"})
            return asdict(error_response(
                f"Unexpected error: {str(e)}",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                remediation="Check logs for details",
            ))
