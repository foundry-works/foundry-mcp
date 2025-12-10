"""
Documentation generation tools for foundry-mcp.

Provides MCP tools for fidelity review operations against SDD specifications.
"""

import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    sanitize_error_message,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import (
    get_metrics,
    mcp_tool,
)
from foundry_mcp.core.security import (
    is_prompt_injection,
)
from foundry_mcp.core.spec import find_specs_directory, load_spec, find_spec_file

logger = logging.getLogger(__name__)


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
        canonical_name="spec-review-fidelity",
    )
    @mcp_tool(tool_name="spec-review-fidelity", emit_metrics=True, audit=True)
    def spec_review_fidelity(
        spec_id: str,
        task_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        files: Optional[List[str]] = None,
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

        LIMITATIONS:
        - Requires AI consultation (no non-AI mode available)
        - At least one AI provider must be configured (gemini, codex, cursor-agent, claude, opencode)
        """
        try:
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
                        labels={
                            "tool": "spec-review-fidelity",
                            "reason": "prompt_injection",
                        },
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
                            labels={
                                "tool": "spec-review-fidelity",
                                "reason": "prompt_injection",
                            },
                        )
                        return asdict(
                            error_response(
                                f"Input validation failed for files[{idx}]",
                                error_code="VALIDATION_ERROR",
                                error_type="security",
                                remediation="Remove special characters or instruction-like patterns from file paths.",
                            )
                        )

            # Resolve workspace and find spec
            ws_path = Path(workspace) if workspace else Path.cwd()
            specs_dir = find_specs_directory(ws_path)
            if not specs_dir:
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "specs_dir_not_found",
                    },
                )
                return asdict(
                    error_response(
                        "Could not find specs directory",
                        error_code="SPECS_DIR_NOT_FOUND",
                        error_type="not_found",
                        remediation="Ensure you're in a project with a specs/ directory",
                    )
                )

            # Find the spec file to validate it exists
            spec_file = find_spec_file(spec_id, specs_dir)
            if not spec_file:
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

            # Determine scope for review
            scope = "task" if task_id else ("phase" if phase_id else "spec")

            # Load spec data
            spec_data = load_spec(spec_file)
            spec_title = spec_data.get("title", spec_id)
            spec_description = spec_data.get("description", "")

            # Import consultation layer components
            try:
                from foundry_mcp.core.ai_consultation import (
                    ConsensusResult,
                    ConsultationOrchestrator,
                    ConsultationRequest,
                    ConsultationWorkflow,
                )
            except ImportError as exc:
                logger.debug(f"AI consultation import error: {exc}")
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "import_error",
                    },
                )
                return asdict(
                    error_response(
                        "AI consultation layer not available",
                        error_code="AI_NOT_AVAILABLE",
                        error_type="unavailable",
                        remediation="Ensure foundry_mcp.core.ai_consultation is properly installed",
                    )
                )

            # Determine review scope description
            if task_id:
                review_scope = f"Task {task_id}"
            elif phase_id:
                review_scope = f"Phase {phase_id}"
            elif files:
                review_scope = f"Files: {', '.join(files)}"
            else:
                review_scope = "Full specification"

            # Build context for fidelity review
            spec_requirements = _build_spec_requirements(spec_data, task_id, phase_id)
            implementation_artifacts = _build_implementation_artifacts(
                spec_data, task_id, phase_id, files, incremental, base_branch
            )
            test_results = _build_test_results(spec_data, task_id, phase_id)
            journal_entries = _build_journal_entries(spec_data, task_id, phase_id)

            # Initialize orchestrator with preferred provider if specified
            # ai_tools param maps to provider list
            preferred_providers = ai_tools if ai_tools else []
            if model:
                # If specific model requested, it may indicate a provider preference
                pass  # model handled in request

            orchestrator = ConsultationOrchestrator(
                preferred_providers=preferred_providers,
            )

            # Check if providers are available
            first_provider = ai_tools[0] if ai_tools else None
            if not orchestrator.is_available(provider_id=first_provider):
                provider_msg = (
                    f" (requested: {first_provider})" if first_provider else ""
                )
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "no_provider",
                    },
                )
                # Get detailed unavailability reasons for diagnostics
                from foundry_mcp.core.providers.detectors import (
                    get_provider_unavailability_reasons,
                )

                unavailability_reasons = get_provider_unavailability_reasons()
                return asdict(
                    error_response(
                        f"Fidelity review requested but no providers available{provider_msg}",
                        error_code="AI_NO_PROVIDER",
                        error_type="unavailable",
                        data={
                            "spec_id": spec_id,
                            "requested_provider": first_provider,
                            "provider_status": unavailability_reasons,
                        },
                        remediation=(
                            "Install and configure an AI provider. Run 'which claude' or "
                            "'which gemini' to verify binaries are in PATH. Set "
                            "FOUNDRY_<PROVIDER>_AVAILABLE_OVERRIDE=1 to force availability."
                        ),
                    )
                )

            # Create consultation request
            request = ConsultationRequest(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                prompt_id="FIDELITY_REVIEW_V1",
                context={
                    "spec_id": spec_id,
                    "spec_title": spec_title,
                    "spec_description": f"**Description:** {spec_description}"
                    if spec_description
                    else "",
                    "review_scope": review_scope,
                    "spec_requirements": spec_requirements,
                    "implementation_artifacts": implementation_artifacts,
                    "test_results": test_results,
                    "journal_entries": journal_entries,
                },
                provider_id=first_provider,
            )

            # Execute consultation
            try:
                result = orchestrator.consult(request, use_cache=True)
            except Exception as exc:
                logger.exception("AI consultation failed")
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "consultation_error",
                    },
                )
                return asdict(
                    error_response(
                        "AI consultation failed",
                        error_code="AI_CONSULTATION_ERROR",
                        error_type="error",
                        data={
                            "spec_id": spec_id,
                            "review_scope": review_scope,
                        },
                        remediation="Check provider configuration and try again",
                    )
                )

            # Determine if this is a multi-model consensus result
            is_consensus = isinstance(result, ConsensusResult)

            # Check for consultation failure
            # For ConsensusResult: check .success property
            # For ConsultationResult: check .error attribute
            if (
                result is None
                or (not is_consensus and result.error)
                or (is_consensus and not result.success)
            ):
                if is_consensus:
                    error_msg = (
                        "; ".join(result.warnings)
                        if result.warnings
                        else "All providers failed"
                    )
                else:
                    error_msg = (
                        result.error if result else "Consultation returned no result"
                    )
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "consultation_failed",
                    },
                )
                # Get detailed provider unavailability reasons for diagnostics
                from foundry_mcp.core.providers.detectors import (
                    get_provider_unavailability_reasons,
                )

                unavailability_reasons = get_provider_unavailability_reasons()
                return asdict(
                    error_response(
                        f"AI consultation failed: {error_msg}",
                        error_code="AI_CONSULTATION_FAILED",
                        error_type="unavailable",
                        data={
                            "spec_id": spec_id,
                            "review_scope": review_scope,
                            "error_details": error_msg,
                            "provider_id": result.responses[0].provider_id
                            if is_consensus and result.responses
                            else (result.provider_id if result else "none"),
                            "provider_status": unavailability_reasons,
                            "mode": "multi_model" if is_consensus else "single_model",
                        },
                        remediation=(
                            "Check provider availability. Ensure at least one AI provider "
                            "(claude, gemini, codex) is installed and accessible in PATH. "
                            "Run 'which claude' or 'which gemini' to verify."
                        ),
                    )
                )

            # Check for empty content (consultation completed but no response)
            # For ConsensusResult use primary_content, for ConsultationResult use content
            result_content = result.primary_content if is_consensus else result.content
            if not result_content:
                metrics.counter(
                    "documentation.errors",
                    labels={
                        "tool": "spec-review-fidelity",
                        "error_type": "empty_response",
                    },
                )
                from foundry_mcp.core.providers.detectors import (
                    get_provider_unavailability_reasons,
                )

                unavailability_reasons = get_provider_unavailability_reasons()
                if is_consensus:
                    provider_info = (
                        result.responses[0].provider_id if result.responses else "none"
                    )
                    model_info = (
                        result.responses[0].model_used if result.responses else "none"
                    )
                else:
                    provider_info = result.provider_id
                    model_info = result.model_used
                return asdict(
                    error_response(
                        "AI consultation returned empty response",
                        error_code="AI_EMPTY_RESPONSE",
                        error_type="error",
                        data={
                            "spec_id": spec_id,
                            "review_scope": review_scope,
                            "provider_id": provider_info,
                            "model_used": model_info,
                            "provider_status": unavailability_reasons,
                            "mode": "multi_model" if is_consensus else "single_model",
                        },
                        remediation=(
                            "The AI provider returned an empty response. This may indicate "
                            "a provider configuration issue or timeout. Try again or check "
                            "provider logs."
                        ),
                    )
                )

            # Parse JSON response (use result_content which handles both result types)
            parsed_response = None
            if result_content:
                try:
                    content = result_content
                    if "```json" in content:
                        start = content.find("```json") + 7
                        end = content.find("```", start)
                        if end > start:
                            content = content[start:end].strip()
                    elif "```" in content:
                        start = content.find("```") + 3
                        end = content.find("```", start)
                        if end > start:
                            content = content[start:end].strip()
                    parsed_response = json.loads(content)
                except (json.JSONDecodeError, ValueError) as exc:
                    # JSON parsing failed - return error instead of unknown verdict
                    logger.debug(f"JSON parse error in fidelity review: {exc}")
                    metrics.counter(
                        "documentation.errors",
                        labels={
                            "tool": "spec-review-fidelity",
                            "error_type": "invalid_response",
                        },
                    )
                    if is_consensus:
                        provider_info = (
                            result.responses[0].provider_id
                            if result.responses
                            else "none"
                        )
                        model_info = (
                            result.responses[0].model_used
                            if result.responses
                            else "none"
                        )
                    else:
                        provider_info = result.provider_id
                        model_info = result.model_used
                    return asdict(
                        error_response(
                            "AI response could not be parsed as valid JSON",
                            error_code="AI_INVALID_RESPONSE",
                            error_type="error",
                            data={
                                "spec_id": spec_id,
                                "review_scope": review_scope,
                                "provider_id": provider_info,
                                "model_used": model_info,
                                "mode": "multi_model"
                                if is_consensus
                                else "single_model",
                            },
                            remediation=(
                                "The AI provider returned a response that could not be parsed "
                                "as valid JSON. This may indicate the prompt needs adjustment or "
                                "the model response format is unexpected."
                            ),
                        )
                    )

            # Build successful response
            metrics.counter(
                "documentation.success",
                labels={"tool": "spec-review-fidelity"},
            )

            # Build consensus info based on result type
            if is_consensus:
                # Multi-model consensus result
                responses_data = [
                    {
                        "provider_id": r.provider_id,
                        "model_used": r.model_used,
                        "success": r.success,
                        "error": r.error,
                    }
                    for r in result.responses
                ]
                agreement_data = None
                if result.agreement:
                    agreement_data = {
                        "total_providers": result.agreement.total_providers,
                        "successful_providers": result.agreement.successful_providers,
                        "failed_providers": result.agreement.failed_providers,
                        "success_rate": result.agreement.success_rate,
                        "has_consensus": result.agreement.has_consensus,
                    }
                consensus_info = {
                    "mode": "multi_model",
                    "responses": responses_data,
                    "agreement": agreement_data,
                    "threshold": consensus_threshold,
                }
            else:
                # Single-model result
                consensus_info = {
                    "mode": "single_model",
                    "provider": result.provider_id if result else None,
                    "model": result.model_used if result else None,
                    "cached": result.cache_hit if result else False,
                    "threshold": consensus_threshold,
                }

            return asdict(
                success_response(
                    {
                        "spec_id": spec_id,
                        "title": spec_title,
                        "scope": scope,
                        "review_scope": review_scope,
                        "task_id": task_id,
                        "phase_id": phase_id,
                        "files": files,
                        "verdict": parsed_response.get("verdict", "unknown")
                        if parsed_response
                        else "unknown",
                        "deviations": parsed_response.get("deviations", [])
                        if parsed_response
                        else [],
                        "recommendations": parsed_response.get("recommendations", [])
                        if parsed_response
                        else [],
                        "consensus": consensus_info,
                        "response": parsed_response
                        if parsed_response
                        else result_content,
                        "raw_response": result_content if not parsed_response else None,
                        "incremental": incremental,
                        "base_branch": base_branch,
                    }
                )
            )

        except Exception as e:
            logger.exception("Error in spec-review-fidelity")
            metrics.counter(
                "documentation.errors",
                labels={"tool": "spec-review-fidelity", "error_type": "internal"},
            )
            return asdict(
                error_response(
                    sanitize_error_message(e, context="fidelity review"),
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                    remediation="Check logs for details",
                )
            )

    logger.debug("Registered documentation tools: spec-review-fidelity")


# =============================================================================
# Helper Functions for Fidelity Review
# =============================================================================


def _build_spec_requirements(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
) -> str:
    """Build spec requirements section for fidelity review context."""
    lines = []

    if task_id:
        # Find specific task
        task = _find_task(spec_data, task_id)
        if task:
            lines.append(f"### Task: {task.get('title', task_id)}")
            lines.append(f"- **Status:** {task.get('status', 'unknown')}")
            if task.get("metadata", {}).get("details"):
                lines.append("- **Details:**")
                for detail in task["metadata"]["details"]:
                    lines.append(f"  - {detail}")
            if task.get("metadata", {}).get("file_path"):
                lines.append(f"- **Expected file:** {task['metadata']['file_path']}")
    elif phase_id:
        # Find specific phase
        phase = _find_phase(spec_data, phase_id)
        if phase:
            lines.append(f"### Phase: {phase.get('title', phase_id)}")
            lines.append(f"- **Status:** {phase.get('status', 'unknown')}")
            child_nodes = _get_child_nodes(spec_data, phase)
            if child_nodes:
                lines.append("- **Tasks:**")
                for child in child_nodes:
                    lines.append(
                        f"  - {child.get('id', 'unknown')}: {child.get('title', 'Unknown task')}"
                    )
    else:
        # Full spec
        lines.append(f"### Specification: {spec_data.get('title', 'Unknown')}")
        if spec_data.get("description"):
            lines.append(f"- **Description:** {spec_data['description']}")
        if spec_data.get("assumptions"):
            lines.append("- **Assumptions:**")
            for assumption in spec_data["assumptions"][:5]:
                if isinstance(assumption, dict):
                    lines.append(f"  - {assumption.get('text', str(assumption))}")
                else:
                    lines.append(f"  - {assumption}")

    return "\n".join(lines) if lines else "*No requirements available*"


def _build_implementation_artifacts(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
    files: Optional[List[str]],
    incremental: bool,
    base_branch: str,
) -> str:
    """Build implementation artifacts section for fidelity review context."""
    lines = []

    # Collect file paths to review
    file_paths = []
    if files:
        file_paths = list(files)
    elif task_id:
        task = _find_task(spec_data, task_id)
        if task and task.get("metadata", {}).get("file_path"):
            file_paths = [task["metadata"]["file_path"]]
    elif phase_id:
        phase = _find_phase(spec_data, phase_id)
        if phase:
            for child in _get_child_nodes(spec_data, phase):
                if child.get("metadata", {}).get("file_path"):
                    file_paths.append(child["metadata"]["file_path"])

    # If incremental, get changed files from git diff
    if incremental:
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base_branch],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                changed_files = result.stdout.strip().split("\n")
                if file_paths:
                    # Intersect with specified files
                    file_paths = [f for f in file_paths if f in changed_files]
                else:
                    file_paths = changed_files
                lines.append(
                    f"*Incremental review: {len(file_paths)} changed files since {base_branch}*\n"
                )
        except Exception:
            lines.append(f"*Warning: Could not get git diff from {base_branch}*\n")

    # Read file contents (limited)
    for file_path in file_paths[:5]:  # Limit to 5 files
        path = Path(file_path)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                # Truncate large files
                if len(content) > 10000:
                    content = content[:10000] + "\n... [truncated] ..."
                file_type = path.suffix.lstrip(".") or "text"
                lines.append(f"### File: `{file_path}`")
                lines.append(f"```{file_type}")
                lines.append(content)
                lines.append("```\n")
            except Exception as e:
                lines.append(f"### File: `{file_path}`")
                lines.append(f"*Error reading file: {e}*\n")
        else:
            lines.append(f"### File: `{file_path}`")
            lines.append("*File not found*\n")

    if not lines:
        lines.append("*No implementation artifacts available*")

    return "\n".join(lines)


def _build_test_results(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
) -> str:
    """Build test results section for fidelity review context."""
    # Check journal for test-related entries
    journal = spec_data.get("journal", [])
    test_entries = [
        entry
        for entry in journal
        if "test" in entry.get("title", "").lower()
        or "verify" in entry.get("title", "").lower()
    ]

    if test_entries:
        lines = ["*Recent test-related journal entries:*"]
        for entry in test_entries[-3:]:  # Last 3 entries
            lines.append(
                f"- **{entry.get('title', 'Unknown')}** ({entry.get('timestamp', 'unknown')})"
            )
            if entry.get("content"):
                # Truncate long content
                content = entry["content"][:500]
                if len(entry["content"]) > 500:
                    content += "..."
                lines.append(f"  {content}")
        return "\n".join(lines)

    return "*No test results available*"


def _build_journal_entries(
    spec_data: Dict[str, Any],
    task_id: Optional[str],
    phase_id: Optional[str],
) -> str:
    """Build journal entries section for fidelity review context."""
    journal = spec_data.get("journal", [])

    if task_id:
        # Filter to task-related entries
        journal = [entry for entry in journal if entry.get("task_id") == task_id]

    if journal:
        lines = [f"*{len(journal)} journal entries found:*"]
        for entry in journal[-5:]:  # Last 5 entries
            entry_type = entry.get("entry_type", "note")
            lines.append(
                f"- **[{entry_type}]** {entry.get('title', 'Untitled')} ({entry.get('timestamp', 'unknown')[:10] if entry.get('timestamp') else 'unknown'})"
            )
        return "\n".join(lines)

    return "*No journal entries found*"


def _find_task(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Find a task by ID in the spec hierarchy (new or legacy format)."""
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    if task_id in hierarchy_nodes:
        return hierarchy_nodes[task_id]

    hierarchy = spec_data.get("hierarchy", {})
    children = hierarchy.get("children") if isinstance(hierarchy, dict) else None
    if children:
        return _search_hierarchy_children(children, task_id)
    return None


def _find_phase(spec_data: Dict[str, Any], phase_id: str) -> Optional[Dict[str, Any]]:
    """Find a phase by ID in the spec hierarchy (new or legacy format)."""
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    if phase_id in hierarchy_nodes:
        return hierarchy_nodes[phase_id]

    hierarchy = spec_data.get("hierarchy", {})
    children = hierarchy.get("children") if isinstance(hierarchy, dict) else None
    if children:
        return _search_hierarchy_children(children, phase_id)
    return None


def _get_hierarchy_nodes(spec_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return mapping of hierarchy node IDs to node data."""
    hierarchy = spec_data.get("hierarchy", {})
    nodes: Dict[str, Dict[str, Any]] = {}

    if isinstance(hierarchy, dict):
        # New format: dict keyed by node_id -> node metadata
        if (
            all(isinstance(value, dict) for value in hierarchy.values())
            and "children" not in hierarchy
        ):
            for node_id, node in hierarchy.items():
                node_copy = dict(node)
                node_copy.setdefault("id", node_id)
                nodes[node_id] = node_copy
            return nodes

        # Legacy format: nested children arrays
        if hierarchy.get("children"):
            _collect_hierarchy_nodes(hierarchy, nodes)

    return nodes


def _collect_hierarchy_nodes(
    node: Dict[str, Any], nodes: Dict[str, Dict[str, Any]]
) -> None:
    """Recursively collect nodes for legacy hierarchy structure."""
    node_id = node.get("id")
    if node_id:
        nodes[node_id] = node
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            _collect_hierarchy_nodes(child, nodes)


def _search_hierarchy_children(
    children: List[Dict[str, Any]], target_id: str
) -> Optional[Dict[str, Any]]:
    """Search nested children lists for a target ID."""
    for child in children:
        if child.get("id") == target_id:
            return child
        nested = child.get("children")
        if nested:
            result = _search_hierarchy_children(nested, target_id)
            if result:
                return result
    return None


def _get_child_nodes(
    spec_data: Dict[str, Any], node: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Resolve child references (IDs or embedded dicts) to node data."""
    children = node.get("children", []) or []
    if not children:
        return []

    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    resolved: List[Dict[str, Any]] = []
    for child in children:
        if isinstance(child, dict):
            resolved.append(child)
        elif isinstance(child, str):
            child_node = hierarchy_nodes.get(child)
            if child_node:
                resolved.append(child_node)
    return resolved
