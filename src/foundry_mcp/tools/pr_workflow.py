"""
PR workflow tools for foundry-mcp.

Provides MCP tools for LLM-powered GitHub PR creation with SDD context.
Wraps SDD CLI create-pr commands with AI-enhanced PR descriptions.

Resilience features:
- Circuit breaker for SDD CLI calls (opens after 5 consecutive failures)
- Timing metrics for all tool invocations
- Graceful degradation when LLM not configured
"""

import json
import logging
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.resilience import (
    CircuitBreaker,
    CircuitBreakerError,
    SLOW_TIMEOUT,
)

logger = logging.getLogger(__name__)

# Metrics singleton for PR workflow tools
_metrics = get_metrics()

# Circuit breaker for SDD CLI PR operations
_pr_breaker = CircuitBreaker(
    name="sdd_cli_pr",
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3,
)

# Default timeout for PR operations (120 seconds)
PR_TIMEOUT: float = SLOW_TIMEOUT


def _get_llm_status() -> Dict[str, Any]:
    """Get LLM configuration status for PR operations."""
    try:
        from foundry_mcp.core.llm_config import get_llm_config

        config = get_llm_config()
        return {
            "configured": config.get_api_key() is not None,
            "provider": config.provider.value,
            "model": config.get_model(),
        }
    except ImportError:
        return {"configured": False, "error": "LLM config not available"}
    except Exception as e:
        return {"configured": False, "error": str(e)}


def _run_pr_command(
    cmd: List[str],
    tool_name: str,
    timeout: float = PR_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Execute an SDD PR CLI command with circuit breaker protection."""
    if not _pr_breaker.can_execute():
        status = _pr_breaker.get_status()
        _metrics.counter(f"pr_workflow.{tool_name}", labels={"status": "circuit_open"})
        raise CircuitBreakerError(
            f"SDD PR circuit breaker is open (retry after {status.get('retry_after_seconds', 0):.1f}s)",
            breaker_name="sdd_cli_pr",
            state=_pr_breaker.state,
            retry_after=status.get("retry_after_seconds"),
        )

    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            _pr_breaker.record_success()
        else:
            _pr_breaker.record_failure()

        return result

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        _pr_breaker.record_failure()
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer(f"pr_workflow.{tool_name}.duration_ms", elapsed_ms)


def register_pr_workflow_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register PR workflow tools with the FastMCP server."""

    @canonical_tool(
        mcp,
        canonical_name="pr-create-with-spec",
    )
    @mcp_tool(tool_name="pr-create-with-spec", emit_metrics=True, audit=True)
    def pr_create_with_spec(
        spec_id: str,
        title: Optional[str] = None,
        base_branch: str = "main",
        include_journals: bool = True,
        include_diffs: bool = True,
        model: Optional[str] = None,
        path: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict:
        """
        Create a GitHub PR with AI-enhanced description from SDD spec context.

        Wraps the SDD create-pr command to scaffold PRs with rich context
        from the specification including task completions, journal entries,
        and AI-generated summaries.

        WHEN TO USE:
        - After completing a phase or set of tasks
        - When ready to submit work for review
        - To generate comprehensive PR descriptions automatically

        Args:
            spec_id: Specification ID to create PR for
            title: PR title (default: auto-generated from spec)
            base_branch: Base branch for PR (default: main)
            include_journals: Include journal entries in PR description
            include_diffs: Include git diffs in LLM context
            model: LLM model for description generation
            path: Project root path
            dry_run: Preview PR content without creating

        Returns:
            JSON object with:
            - spec_id: The specification used
            - pr_url: URL of created PR (if not dry_run)
            - title: PR title
            - description_preview: Preview of generated description
            - llm_status: LLM configuration status

        LIMITATIONS:
        - Requires GitHub CLI (gh) to be installed and authenticated
        - Requires LLM for enhanced descriptions (falls back to basic if unavailable)
        - Git working tree must be clean or changes staged
        """
        start_time = time.perf_counter()

        try:
            llm_status = _get_llm_status()

            # Build command
            cmd = ["foundry-cli", "create-pr", spec_id, "--json"]

            if title:
                cmd.extend(["--title", title])
            cmd.extend(["--base", base_branch])

            if include_journals:
                cmd.append("--include-journals")
            if include_diffs:
                cmd.append("--include-diffs")
            if model:
                cmd.extend(["--model", model])
            if path:
                cmd.extend(["--path", path])
            if dry_run:
                cmd.append("--dry-run")

            # Dry run mode
            if dry_run:
                result = _run_pr_command(cmd, "pr-create-with-spec")

                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else "PR preview failed"
                    return asdict(
                        error_response(
                            f"PR preview failed: {error_msg}",
                            data={"spec_id": spec_id, "exit_code": result.returncode},
                        )
                    )

                try:
                    preview_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    preview_data = {"raw_output": result.stdout}

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Remove spec_id from preview_data if present to avoid duplicate keyword arg
                preview_data.pop("spec_id", None)
                return asdict(
                    success_response(
                        spec_id=spec_id,
                        dry_run=True,
                        llm_status=llm_status,
                        duration_ms=round(duration_ms, 2),
                        **preview_data,
                    )
                )

            # Execute PR creation
            result = _run_pr_command(cmd, "pr-create-with-spec")

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "PR creation failed"
                return asdict(
                    error_response(
                        f"PR creation failed: {error_msg}",
                        data={"spec_id": spec_id, "exit_code": result.returncode},
                    )
                )

            try:
                pr_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                pr_data = {"raw_output": result.stdout}

            duration_ms = (time.perf_counter() - start_time) * 1000
            _metrics.timer(
                "pr_workflow.pr_create.duration_ms",
                duration_ms,
                labels={"include_journals": str(include_journals)},
            )

            # Remove spec_id from pr_data if present to avoid duplicate keyword arg
            pr_data.pop("spec_id", None)
            return asdict(
                success_response(
                    spec_id=spec_id,
                    llm_status=llm_status,
                    duration_ms=round(duration_ms, 2),
                    **pr_data,
                )
            )

        except CircuitBreakerError as e:
            return asdict(
                error_response(
                    str(e),
                    data={
                        "spec_id": spec_id,
                        "retry_after_seconds": e.retry_after,
                        "breaker_state": e.state,
                    },
                )
            )
        except subprocess.TimeoutExpired:
            return asdict(
                error_response(
                    f"PR creation timed out after {PR_TIMEOUT}s",
                    data={"spec_id": spec_id, "timeout_seconds": PR_TIMEOUT},
                )
            )
        except FileNotFoundError:
            return asdict(
                error_response(
                    "foundry-cli not found. Ensure foundry-mcp is installed and in PATH.",
                    data={"spec_id": spec_id},
                )
            )
        except Exception as e:
            logger.exception(f"Error creating PR for {spec_id}")
            return asdict(
                error_response(
                    f"PR creation error: {str(e)}",
                    data={"spec_id": spec_id, "error_type": type(e).__name__},
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="pr-get-spec-context",
    )
    @mcp_tool(tool_name="pr-get-spec-context", emit_metrics=True, audit=False)
    def pr_get_spec_context(
        spec_id: str,
        include_tasks: bool = True,
        include_journals: bool = True,
        include_progress: bool = True,
        path: Optional[str] = None,
    ) -> dict:
        """
        Get specification context for PR description generation.

        Retrieves comprehensive information about a spec that can be used
        to craft meaningful PR descriptions, including completed tasks,
        journal entries, and overall progress.

        WHEN TO USE:
        - Before creating a PR to understand what to include
        - To gather context for manual PR creation
        - For debugging PR description generation

        Args:
            spec_id: Specification ID
            include_tasks: Include completed task summaries
            include_journals: Include recent journal entries
            include_progress: Include phase/task progress stats
            path: Project root path

        Returns:
            JSON object with:
            - spec_id: The specification ID
            - title: Spec title
            - tasks: Completed tasks (if requested)
            - journals: Recent journal entries (if requested)
            - progress: Progress statistics (if requested)
        """
        start_time = time.perf_counter()

        try:
            # Build command to get spec info
            cmd = ["foundry-cli", "progress", spec_id, "--json"]
            if path:
                cmd.extend(["--path", path])

            result = _run_pr_command(cmd, "pr-get-spec-context")

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Failed to get spec context"
                return asdict(
                    error_response(
                        f"Failed to get spec context: {error_msg}",
                        data={"spec_id": spec_id},
                    )
                )

            try:
                spec_data = json.loads(result.stdout)
            except json.JSONDecodeError:
                spec_data = {}

            context = {
                "spec_id": spec_id,
            }

            if include_progress:
                context["progress"] = {
                    "total_tasks": spec_data.get("total_tasks", 0),
                    "completed_tasks": spec_data.get("completed_tasks", 0),
                    "percentage": spec_data.get("percentage", 0),
                    "current_phase": spec_data.get("current_phase"),
                }

            # Get journal entries if requested
            if include_journals:
                journal_cmd = ["foundry-cli", "get-journal", spec_id, "--json"]
                if path:
                    journal_cmd.extend(["--path", path])

                try:
                    journal_result = subprocess.run(
                        journal_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30.0,
                    )
                    if journal_result.returncode == 0:
                        try:
                            journal_data = json.loads(journal_result.stdout)
                            # Get last 5 entries
                            entries = journal_data.get("entries", [])[-5:]
                            context["journals"] = entries
                        except json.JSONDecodeError:
                            context["journals"] = []
                except Exception:
                    context["journals"] = []

            duration_ms = (time.perf_counter() - start_time) * 1000

            return asdict(
                success_response(
                    duration_ms=round(duration_ms, 2),
                    **context,
                )
            )

        except CircuitBreakerError as e:
            return asdict(
                error_response(str(e), data={"spec_id": spec_id})
            )
        except Exception as e:
            logger.exception(f"Error getting spec context for {spec_id}")
            return asdict(
                error_response(
                    f"Error: {str(e)}",
                    data={"spec_id": spec_id, "error_type": type(e).__name__},
                )
            )
