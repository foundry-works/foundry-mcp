"""
Documentation generation tools for foundry-mcp.

Provides MCP tools for generating human-facing documentation bundles
from SDD specifications. Wraps the `sdd doc` CLI to produce markdown
and HTML documentation artifacts.
"""

import logging
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    SLOW_TIMEOUT,
    BACKGROUND_TIMEOUT,
)
from foundry_mcp.core.observability import (
    get_metrics,
    get_audit_logger,
    mcp_tool,
)
from foundry_mcp.core.security import (
    is_prompt_injection,
    validate_size,
    MAX_STRING_LENGTH,
)

logger = logging.getLogger(__name__)

# Circuit breaker for documentation generation operations
_doc_breaker = CircuitBreaker(
    name="documentation",
    failure_threshold=3,
    recovery_timeout=60.0,
)

# Default timeout for doc generation (can take longer for large specs)
DOC_GENERATION_TIMEOUT = 120

# LLM doc generation can take much longer due to AI processing
LLM_DOC_GENERATION_TIMEOUT = 600

# Fidelity review timeout (AI consultation)
FIDELITY_REVIEW_TIMEOUT = 600


def _run_sdd_doc_command(
    args: List[str],
    timeout: int = DOC_GENERATION_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD doc CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd doc CLI
        timeout: Command timeout in seconds

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "doc"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            import json

            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw_output": result.stdout}}
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_sdd_render_command(
    args: List[str],
    timeout: int = DOC_GENERATION_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD render CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd render CLI
        timeout: Command timeout in seconds

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "render"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            import json

            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw_output": result.stdout}}
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_sdd_llm_doc_gen_command(
    args: List[str],
    timeout: int = LLM_DOC_GENERATION_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD llm-doc-gen CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd llm-doc-gen CLI
        timeout: Command timeout in seconds (default 600s for LLM processing)

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "llm-doc-gen"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            import json

            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw_output": result.stdout}}
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"LLM doc generation timed out after {timeout} seconds. "
            "Consider using --resume to continue from checkpoint.",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _run_sdd_fidelity_review_command(
    args: List[str],
    timeout: int = FIDELITY_REVIEW_TIMEOUT,
) -> Dict[str, Any]:
    """Run an SDD fidelity-review CLI command and return parsed JSON output.

    Args:
        args: Command arguments to pass to sdd fidelity-review CLI
        timeout: Command timeout in seconds (default 600s for AI consultation)

    Returns:
        Dict with parsed JSON output or error info
    """
    try:
        result = subprocess.run(
            ["sdd", "fidelity-review"] + args + ["--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            import json

            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": {"raw_output": result.stdout}}
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout or "Command failed",
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Fidelity review timed out after {timeout} seconds. "
            "Consider reviewing a smaller scope (single task or phase).",
        }
    except FileNotFoundError:
        return {
            "success": False,
            "error": "sdd CLI not found. Ensure sdd-toolkit is installed.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_documentation_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register documentation generation tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()

    @canonical_tool(
        mcp,
        canonical_name="spec-doc",
    )
    @mcp_tool(tool_name="spec-doc", emit_metrics=True, audit=True)
    def spec_doc(
        spec_id: str,
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        include_progress: bool = True,
        include_journal: bool = False,
        mode: str = "basic",
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate human-facing documentation bundle from a specification.

        Creates formatted documentation artifacts (markdown/HTML) from
        an SDD specification. Wraps `sdd render` to produce documentation
        suitable for stakeholders, project tracking, and archive.

        Args:
            spec_id: Specification ID to document
            output_format: Output format - 'markdown' or 'md' (HTML planned)
            output_path: Custom output path (default: specs/.human-readable/<spec_id>.md)
            include_progress: Include visual progress bars and stats
            include_journal: Include recent journal entries in output
            mode: Rendering mode - 'basic' (fast) or 'enhanced' (AI features)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - output_path: Path where documentation was written
            - format: Format of generated documentation
            - spec_id: Specification ID
            - title: Specification title
            - stats: Documentation statistics (sections, tasks, etc.)

        WHEN TO USE:
        - Generate stakeholder-friendly documentation from specs
        - Create progress reports for project tracking
        - Archive completed specifications in readable format
        - Export specs for external sharing or review
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _doc_breaker.can_execute():
                status = _doc_breaker.get_status()
                metrics.counter(
                    "documentation.circuit_breaker_open",
                    labels={"tool": "spec-doc"},
                )
                return asdict(
                    error_response(
                        "Documentation generation temporarily unavailable",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                        remediation="Wait and retry. The service is recovering from errors.",
                    )
                )

            # Validate output_format
            if output_format not in ("markdown", "md"):
                return asdict(
                    error_response(
                        f"Invalid output_format: {output_format}. "
                        "Supported formats: 'markdown', 'md'",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'markdown' or 'md' as output_format.",
                    )
                )

            # Validate mode
            if mode not in ("basic", "enhanced"):
                return asdict(
                    error_response(
                        f"Invalid mode: {mode}. Must be 'basic' or 'enhanced'.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use 'basic' for fast rendering or 'enhanced' for AI features.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("spec_id", spec_id),
                ("output_path", output_path),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-doc", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Build command arguments for sdd render
            cmd_args = [spec_id, "--format", output_format, "--mode", mode]

            if output_path:
                cmd_args.extend(["--output", output_path])

            if workspace:
                cmd_args.extend(["--path", workspace])

            # Execute render command
            result = _run_sdd_render_command(cmd_args)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "documentation.spec_doc_time",
                duration_ms,
                labels={"mode": mode, "format": output_format},
            )

            if not result["success"]:
                _doc_breaker.record_failure()

                # Check for specific error patterns
                error_msg = result["error"]
                if "not found" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-doc", "error_type": "not_found"},
                    )
                    return asdict(
                        error_response(
                            f"Specification not found: {spec_id}",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list.",
                        )
                    )

                if "timed out" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-doc", "error_type": "timeout"},
                    )

                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-doc", "error_type": "command_failure"},
                )
                return asdict(error_response(error_msg))

            _doc_breaker.record_success()
            metrics.counter(
                "documentation.success",
                labels={"tool": "spec-doc", "mode": mode, "format": output_format},
            )
            logger.info(
                f"spec-doc completed for {spec_id}",
                extra={"spec_id": spec_id, "duration_ms": round(duration_ms, 2)},
            )

            # Parse the successful response
            doc_data = result["data"]

            # Build response with documentation metadata
            response_data = {
                "spec_id": spec_id,
                "format": output_format,
                "mode": mode,
                "output_path": doc_data.get(
                    "output_path",
                    f"specs/.human-readable/{spec_id}.md",
                ),
            }

            # Add optional fields if present
            if "title" in doc_data:
                response_data["title"] = doc_data["title"]

            if "stats" in doc_data:
                response_data["stats"] = doc_data["stats"]
            elif "total_tasks" in doc_data:
                response_data["stats"] = {
                    "total_tasks": doc_data.get("total_tasks", 0),
                    "completed_tasks": doc_data.get("completed_tasks", 0),
                    "total_sections": doc_data.get("total_sections", 0),
                }

            if include_progress and "progress_percentage" in doc_data:
                response_data["progress_percentage"] = doc_data["progress_percentage"]

            return asdict(
                success_response(
                    **response_data,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for documentation: {e}")
            return asdict(
                error_response(
                    "Documentation generation temporarily unavailable",
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"retry_after_seconds": e.retry_after},
                    remediation="Wait and retry. The service is recovering from errors.",
                )
            )
        except Exception as e:
            _doc_breaker.record_failure()
            logger.error(f"Error generating documentation: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-doc-llm",
    )
    @mcp_tool(tool_name="spec-doc-llm", emit_metrics=True, audit=True)
    def spec_doc_llm(
        directory: str,
        output_dir: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        batch_size: int = 3,
        use_cache: bool = True,
        resume: bool = False,
        clear_cache: bool = False,
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Generate comprehensive LLM-powered documentation for a project.

        Uses Large Language Models to analyze code and generate rich,
        context-aware documentation with explanations, examples, and
        architectural insights.

        Args:
            directory: Project directory to document
            output_dir: Output directory for documentation (default: ./docs)
            name: Project name (default: directory name)
            description: Project description for documentation context
            batch_size: Number of shards to process per batch (default: 3)
            use_cache: Enable persistent caching of parse results
            resume: Resume from previous interrupted generation
            clear_cache: Clear the cache before generating documentation
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - output_dir: Directory where documentation was generated
            - files_generated: Number of documentation files created
            - total_shards: Number of code shards processed
            - duration_seconds: Time taken for generation
            - project_name: Name of the documented project

        WHEN TO USE:
        - Generate comprehensive AI-enhanced documentation for codebases
        - Create rich developer guides with context and explanations
        - Document complex projects with architectural insights
        - Resume interrupted documentation generation
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _doc_breaker.can_execute():
                status = _doc_breaker.get_status()
                metrics.counter(
                    "documentation.circuit_breaker_open",
                    labels={"tool": "spec-doc-llm"},
                )
                return asdict(
                    error_response(
                        "LLM documentation generation temporarily unavailable",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                        remediation="Wait and retry. The service is recovering from errors.",
                    )
                )

            # Validate directory
            if not directory:
                return asdict(
                    error_response(
                        "Directory path is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a valid directory path to document.",
                    )
                )

            # Validate batch_size
            if batch_size < 1 or batch_size > 10:
                return asdict(
                    error_response(
                        f"Invalid batch_size: {batch_size}. Must be between 1 and 10.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use a batch_size between 1 and 10.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("directory", directory),
                ("output_dir", output_dir),
                ("name", name),
                ("description", description),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-doc-llm", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Validate description size (can be longer than other fields)
            if description:
                size_result = validate_size(
                    description, "description", max_string_length=MAX_STRING_LENGTH
                )
                if not size_result.is_valid:
                    return asdict(
                        error_response(
                            f"Description too long: {size_result.violations[0][1]}",
                            error_code="VALIDATION_ERROR",
                            error_type="validation",
                            remediation=f"Limit description to {MAX_STRING_LENGTH} characters.",
                        )
                    )

            # Build command arguments for sdd llm-doc-gen generate
            cmd_args = ["generate", directory]

            if output_dir:
                cmd_args.extend(["--output-dir", output_dir])

            if name:
                cmd_args.extend(["--name", name])

            if description:
                cmd_args.extend(["--description", description])

            cmd_args.extend(["--batch-size", str(batch_size)])

            if use_cache:
                cmd_args.append("--cache")

            if resume:
                cmd_args.append("--resume")

            if clear_cache:
                cmd_args.append("--clear-cache")

            if workspace:
                cmd_args.extend(["--path", workspace])

            # Execute LLM doc generation command
            result = _run_sdd_llm_doc_gen_command(cmd_args)

            duration_ms = (time.perf_counter() - start_time) * 1000
            duration_seconds = duration_ms / 1000
            metrics.timer(
                "documentation.spec_doc_llm_time",
                duration_ms,
                labels={"batch_size": str(batch_size), "cached": str(use_cache)},
            )

            if not result["success"]:
                _doc_breaker.record_failure()

                # Check for specific error patterns
                error_msg = result["error"]
                if "not found" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-doc-llm", "error_type": "not_found"},
                    )
                    return asdict(
                        error_response(
                            f"Directory not found: {directory}",
                            error_code="NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the directory path exists.",
                        )
                    )

                if "timed out" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-doc-llm", "error_type": "timeout"},
                    )
                    return asdict(
                        error_response(
                            error_msg,
                            error_code="UNAVAILABLE",
                            error_type="unavailable",
                            data={"can_resume": True},
                            remediation="Use resume=True to continue from checkpoint.",
                        )
                    )

                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-doc-llm", "error_type": "command_failure"},
                )
                return asdict(error_response(error_msg))

            _doc_breaker.record_success()
            metrics.counter(
                "documentation.success",
                labels={
                    "tool": "spec-doc-llm",
                    "batch_size": str(batch_size),
                    "cached": str(use_cache),
                },
            )
            logger.info(
                f"spec-doc-llm completed for {directory}",
                extra={
                    "directory": directory,
                    "duration_ms": round(duration_ms, 2),
                    "batch_size": batch_size,
                },
            )

            # Parse the successful response
            doc_data = result["data"]

            # Build response with documentation metadata
            response_data = {
                "directory": directory,
                "output_dir": doc_data.get("output_dir", output_dir or "./docs"),
                "duration_seconds": round(duration_seconds, 2),
            }

            # Add optional fields if present
            if "project_name" in doc_data:
                response_data["project_name"] = doc_data["project_name"]
            elif name:
                response_data["project_name"] = name

            if "files_generated" in doc_data:
                response_data["files_generated"] = doc_data["files_generated"]

            if "total_shards" in doc_data:
                response_data["total_shards"] = doc_data["total_shards"]

            if "stats" in doc_data:
                response_data["stats"] = doc_data["stats"]

            return asdict(
                success_response(
                    **response_data,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for LLM documentation: {e}")
            return asdict(
                error_response(
                    "LLM documentation generation temporarily unavailable",
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"retry_after_seconds": e.retry_after},
                    remediation="Wait and retry. The service is recovering from errors.",
                )
            )
        except Exception as e:
            _doc_breaker.record_failure()
            logger.error(f"Error generating LLM documentation: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="spec-review-fidelity",
    )
    @mcp_tool(tool_name="spec-review-fidelity", emit_metrics=True, audit=True)
    def spec_review_fidelity(
        spec_id: str,
        task_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        use_ai: bool = True,
        ai_tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        consensus_threshold: int = 2,
        incremental: bool = False,
        include_tests: bool = True,
        base_branch: str = "main",
        workspace: Optional[str] = None,
    ) -> dict:
        """
        Compare implementation against specification and identify deviations.

        Performs a fidelity review to verify that code implementation matches
        the specification requirements. Uses AI consultation for comprehensive
        analysis.

        Args:
            spec_id: Specification ID to review against
            task_id: Review specific task implementation (mutually exclusive with phase_id)
            phase_id: Review entire phase implementation (mutually exclusive with task_id)
            files: Review specific file(s) only
            use_ai: Enable AI consultation for analysis (default: True)
            ai_tools: Specific AI tools to consult (default: all available)
            model: Specific model to use for AI consultation
            consensus_threshold: Minimum models that must agree (default: 2)
            incremental: Only review changed files since last run
            include_tests: Include test results in review (default: True)
            base_branch: Base branch for git diff (default: main)
            workspace: Optional workspace path (defaults to config)

        Returns:
            JSON object with:
            - spec_id: Specification reviewed
            - scope: Review scope (task/phase/files)
            - verdict: Overall fidelity verdict (pass/partial/fail)
            - deviations: List of identified deviations
            - recommendations: Suggested fixes
            - consensus: AI model consensus information

        WHEN TO USE:
        - Verify implementation matches specification requirements
        - Check for drift between code and documented behavior
        - Review completed tasks/phases for compliance
        - Generate fidelity reports for documentation
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _doc_breaker.can_execute():
                status = _doc_breaker.get_status()
                metrics.counter(
                    "documentation.circuit_breaker_open",
                    labels={"tool": "spec-review-fidelity"},
                )
                return asdict(
                    error_response(
                        "Fidelity review temporarily unavailable",
                        error_code="UNAVAILABLE",
                        error_type="unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                        remediation="Wait and retry. The service is recovering from errors.",
                    )
                )

            # Validate spec_id
            if not spec_id:
                return asdict(
                    error_response(
                        "Specification ID is required",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        remediation="Provide a valid spec_id to review.",
                    )
                )

            # Validate mutual exclusivity of task_id and phase_id
            if task_id and phase_id:
                return asdict(
                    error_response(
                        "Cannot specify both task_id and phase_id",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Provide either task_id OR phase_id, not both.",
                    )
                )

            # Validate consensus_threshold
            if consensus_threshold < 1 or consensus_threshold > 5:
                return asdict(
                    error_response(
                        f"Invalid consensus_threshold: {consensus_threshold}. Must be between 1 and 5.",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        remediation="Use a consensus_threshold between 1 and 5.",
                    )
                )

            # Input validation: check for prompt injection in user-controlled fields
            for field_name, field_value in [
                ("spec_id", spec_id),
                ("task_id", task_id),
                ("phase_id", phase_id),
                ("model", model),
                ("base_branch", base_branch),
                ("workspace", workspace),
            ]:
                if field_value and is_prompt_injection(field_value):
                    metrics.counter(
                        "documentation.security_blocked",
                        labels={"tool": "spec-review-fidelity", "reason": "prompt_injection"},
                    )
                    return asdict(
                        error_response(
                            f"Input validation failed for {field_name}",
                            error_code="VALIDATION_ERROR",
                            error_type="security",
                            remediation="Remove special characters or instruction-like patterns from input.",
                        )
                    )

            # Validate files array for injection
            if files:
                for idx, file_path in enumerate(files):
                    if is_prompt_injection(file_path):
                        metrics.counter(
                            "documentation.security_blocked",
                            labels={"tool": "spec-review-fidelity", "reason": "prompt_injection"},
                        )
                        return asdict(
                            error_response(
                                f"Input validation failed for files[{idx}]",
                                error_code="VALIDATION_ERROR",
                                error_type="security",
                                remediation="Remove special characters or instruction-like patterns from file paths.",
                            )
                        )

            # Build command arguments
            cmd_args = [spec_id]

            if task_id:
                cmd_args.extend(["--task", task_id])
            elif phase_id:
                cmd_args.extend(["--phase", phase_id])

            if files:
                cmd_args.extend(["--files"] + files)

            if not use_ai:
                cmd_args.append("--no-ai")
            else:
                if ai_tools:
                    cmd_args.extend(["--ai-tools"] + ai_tools)
                if model:
                    cmd_args.extend(["--model", model])

            cmd_args.extend(["--consensus-threshold", str(consensus_threshold)])

            if incremental:
                cmd_args.append("--incremental")

            if not include_tests:
                cmd_args.append("--no-tests")

            cmd_args.extend(["--base-branch", base_branch])

            if workspace:
                cmd_args.extend(["--path", workspace])

            # Execute fidelity review command
            result = _run_sdd_fidelity_review_command(cmd_args)

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.timer(
                "documentation.fidelity_review_time",
                duration_ms,
                labels={
                    "scope": "task" if task_id else ("phase" if phase_id else "spec"),
                    "ai_enabled": str(use_ai),
                },
            )

            if not result["success"]:
                _doc_breaker.record_failure()

                # Check for specific error patterns
                error_msg = result["error"]
                if "not found" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-review-fidelity", "error_type": "not_found"},
                    )
                    return asdict(
                        error_response(
                            f"Specification not found: {spec_id}",
                            error_code="SPEC_NOT_FOUND",
                            error_type="not_found",
                            remediation="Verify the spec ID exists using spec-list.",
                        )
                    )

                if "timed out" in error_msg.lower():
                    metrics.counter(
                        "documentation.errors",
                        labels={"tool": "spec-review-fidelity", "error_type": "timeout"},
                    )
                    return asdict(
                        error_response(
                            error_msg,
                            error_code="UNAVAILABLE",
                            error_type="unavailable",
                            remediation="Try reviewing a smaller scope (single task or phase).",
                        )
                    )

                metrics.counter(
                    "documentation.errors",
                    labels={"tool": "spec-review-fidelity", "error_type": "command_failure"},
                )
                return asdict(error_response(error_msg))

            _doc_breaker.record_success()

            # Parse the successful response
            review_data = result["data"]

            # Determine scope
            scope = "task" if task_id else ("phase" if phase_id else "spec")

            metrics.counter(
                "documentation.success",
                labels={
                    "tool": "spec-review-fidelity",
                    "scope": scope,
                    "ai_enabled": str(use_ai),
                },
            )
            logger.info(
                f"spec-review-fidelity completed for {spec_id}",
                extra={
                    "spec_id": spec_id,
                    "scope": scope,
                    "duration_ms": round(duration_ms, 2),
                    "ai_enabled": use_ai,
                },
            )

            # Build response with review results
            response_data = {
                "spec_id": spec_id,
                "scope": scope,
            }

            if task_id:
                response_data["task_id"] = task_id
            elif phase_id:
                response_data["phase_id"] = phase_id

            # Add review results
            if "verdict" in review_data:
                response_data["verdict"] = review_data["verdict"]
            elif "consensus" in review_data:
                response_data["verdict"] = review_data["consensus"].get("verdict", "unknown")

            if "deviations" in review_data:
                response_data["deviations"] = review_data["deviations"]

            if "recommendations" in review_data:
                response_data["recommendations"] = review_data["recommendations"]

            if "consensus" in review_data:
                response_data["consensus"] = review_data["consensus"]

            if "issue_counts" in review_data:
                response_data["issue_counts"] = review_data["issue_counts"]

            if "artifacts" in review_data:
                response_data["artifacts"] = review_data["artifacts"]

            return asdict(
                success_response(
                    **response_data,
                    telemetry={"duration_ms": round(duration_ms, 2)},
                )
            )

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for fidelity review: {e}")
            return asdict(
                error_response(
                    "Fidelity review temporarily unavailable",
                    error_code="UNAVAILABLE",
                    error_type="unavailable",
                    data={"retry_after_seconds": e.retry_after},
                    remediation="Wait and retry. The service is recovering from errors.",
                )
            )
        except Exception as e:
            _doc_breaker.record_failure()
            logger.error(f"Error performing fidelity review: {e}")
            return asdict(
                error_response(
                    str(e),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )

    logger.debug("Registered documentation tools: spec-doc, spec-doc-llm, spec-review-fidelity")
