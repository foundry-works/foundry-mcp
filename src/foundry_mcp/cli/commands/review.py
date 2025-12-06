"""Review commands for SDD CLI.

Provides commands for spec review including:
- Quick structural review (no LLM required)
- AI-powered full/security/feasibility reviews via ConsultationOrchestrator
- AI-powered fidelity reviews to compare implementation against spec

AI-enhanced reviews use:
- PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
- PLAN_REVIEW_QUICK_V1: Critical blockers and questions focus
- PLAN_REVIEW_SECURITY_V1: Security-focused review
- PLAN_REVIEW_FEASIBILITY_V1: Technical complexity assessment
- SYNTHESIS_PROMPT_V1: Multi-model response synthesis
- FIDELITY_REVIEW_V1: Implementation vs specification comparison
"""

from dataclasses import asdict
import json
import time
from typing import Any, Dict, List, Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.registry import get_context
from foundry_mcp.cli.resilience import (
    FAST_TIMEOUT,
    SLOW_TIMEOUT,
    MEDIUM_TIMEOUT,
    with_sync_timeout,
    handle_keyboard_interrupt,
)
from foundry_mcp.core.review import (
    quick_review,
    review_type_requires_llm,
    prepare_review_context,
)

logger = get_cli_logger()

# Default AI consultation timeout
DEFAULT_AI_TIMEOUT = 120.0

# Review types supported
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# Map review types to PLAN_REVIEW templates
REVIEW_TYPE_TO_TEMPLATE = {
    "full": "PLAN_REVIEW_FULL_V1",
    "security": "PLAN_REVIEW_SECURITY_V1",
    "feasibility": "PLAN_REVIEW_FEASIBILITY_V1",
}

REVIEW_TOOL_DEFINITIONS = [
    {
        "name": "quick-review",
        "description": "Structural validation with schema & progress checks (native).",
        "capabilities": ["structure", "progress", "quality"],
        "requires_llm": False,
    },
    {
        "name": "full-review",
        "description": "LLM-powered deep review via sdd-toolkit.",
        "capabilities": ["structure", "quality", "suggestions"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
    {
        "name": "security-review",
        "description": "Security-focused LLM analysis.",
        "capabilities": ["security", "trust_boundaries"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
    {
        "name": "feasibility-review",
        "description": "Implementation feasibility assessment (LLM).",
        "capabilities": ["complexity", "risk", "dependencies"],
        "requires_llm": True,
        "alternative": "sdd-toolkit:sdd-plan-review",
    },
]

# Fidelity review timeout (longer for AI consultation)
FIDELITY_TIMEOUT = 600


@click.group("review")
def review_group() -> None:
    """Spec review and fidelity checking commands."""
    pass


@review_group.command("spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.option(
    "--tools",
    help="Comma-separated list of review tools to use (LLM types only).",
)
@click.option(
    "--model",
    help="LLM model to use for review (LLM types only).",
)
@click.option(
    "--ai-provider",
    help="Explicit AI provider selection (e.g., gemini, cursor-agent).",
)
@click.option(
    "--ai-timeout",
    type=float,
    default=DEFAULT_AI_TIMEOUT,
    help=f"AI consultation timeout in seconds (default: {DEFAULT_AI_TIMEOUT}).",
)
@click.option(
    "--no-consultation-cache",
    is_flag=True,
    help="Bypass AI consultation cache (always query providers fresh).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be reviewed without executing.",
)
@click.pass_context
@cli_command("review-spec")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
    tools: Optional[str],
    model: Optional[str],
    ai_provider: Optional[str],
    ai_timeout: float,
    no_consultation_cache: bool,
    dry_run: bool,
) -> None:
    """Run a structural or AI-powered review on a specification."""
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    llm_status = _get_llm_status()

    # Quick review doesn't require LLM
    if not review_type_requires_llm(review_type):
        if dry_run:
            emit_success(
                {
                    "spec_id": spec_id,
                    "review_type": review_type,
                    "dry_run": True,
                    "llm_status": llm_status,
                    "message": "Dry run - quick review skipped",
                }
            )
            return

        quick_result = quick_review(spec_id=spec_id, specs_dir=specs_dir)
        duration_ms = (time.perf_counter() - start_time) * 1000

        payload = asdict(quick_result)
        payload["llm_status"] = llm_status

        emit_success(
            payload,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
        return

    # LLM-powered review types (full, security, feasibility)
    result = _run_ai_review(
        spec_id=spec_id,
        specs_dir=specs_dir,
        review_type=review_type,
        ai_provider=ai_provider,
        ai_timeout=ai_timeout,
        consultation_cache=not no_consultation_cache,
        dry_run=dry_run,
        llm_status=llm_status,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        result,
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("tools")
@click.pass_context
@cli_command("review-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Review tools lookup timed out")
def review_tools_cmd(ctx: click.Context) -> None:
    """List native and external review toolchains."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    tools_info = []
    for definition in REVIEW_TOOL_DEFINITIONS:
        requires_llm = definition.get("requires_llm", False)
        available = not requires_llm  # LLM reviews are handled by external workflows
        tool_info = {
            "name": definition["name"],
            "description": definition["description"],
            "capabilities": definition.get("capabilities", []),
            "requires_llm": requires_llm,
            "available": available,
            "status": "native" if available else "external",
        }
        if not available:
            tool_info["alternative"] = definition.get("alternative")
            tool_info["message"] = "Use the sdd-toolkit workflow for this review type"
        tools_info.append(tool_info)

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "tools": tools_info,
            "llm_status": llm_status,
            "review_types": REVIEW_TYPES,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("plan-tools")
@click.pass_context
@cli_command("review-plan-tools")
@handle_keyboard_interrupt()
@with_sync_timeout(FAST_TIMEOUT, "Plan tools lookup timed out")
def review_plan_tools_cmd(ctx: click.Context) -> None:
    """List available plan review toolchains."""
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    # Define plan review toolchains
    plan_tools = [
        {
            "name": "quick-review",
            "description": "Fast structural review for basic validation",
            "capabilities": ["structure", "syntax", "basic_quality"],
            "llm_required": False,
            "estimated_time": "< 10 seconds",
        },
        {
            "name": "full-review",
            "description": "Comprehensive review with LLM analysis",
            "capabilities": ["structure", "quality", "feasibility", "suggestions"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "security-review",
            "description": "Security-focused analysis of plan",
            "capabilities": ["security", "trust_boundaries", "data_flow"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
        {
            "name": "feasibility-review",
            "description": "Implementation feasibility assessment",
            "capabilities": ["complexity", "dependencies", "risk"],
            "llm_required": True,
            "estimated_time": "30-60 seconds",
        },
    ]

    # Add availability status (only quick review is native today)
    available_tools = []
    for tool in plan_tools:
        tool_info = tool.copy()
        if tool["llm_required"]:
            tool_info["status"] = "external"
            tool_info["available"] = False
            tool_info["reason"] = "Use the sdd-toolkit:sdd-plan-review workflow"
            tool_info["alternative"] = "sdd-toolkit:sdd-plan-review"
        else:
            tool_info["status"] = "native"
            tool_info["available"] = True
        available_tools.append(tool_info)

    recommendations = [
        "Use 'quick-review' for structural validation inside foundry-mcp",
        "Invoke sdd-toolkit:sdd-plan-review for AI-assisted plan analysis",
        "Configure LLM credentials when ready to adopt the toolkit workflow",
    ]

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "plan_tools": available_tools,
            "llm_status": llm_status,
            "recommendations": recommendations,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@review_group.command("fidelity")
@click.argument("spec_id")
@click.option(
    "--task",
    "task_id",
    help="Review specific task implementation.",
)
@click.option(
    "--phase",
    "phase_id",
    help="Review entire phase implementation.",
)
@click.option(
    "--files",
    multiple=True,
    help="Review specific file(s) only.",
)
@click.option(
    "--incremental",
    is_flag=True,
    help="Only review changed files since last run.",
)
@click.option(
    "--base-branch",
    default="main",
    help="Base branch for git diff.",
)
@click.option(
    "--ai-provider",
    help="Explicit AI provider selection (e.g., gemini, cursor-agent).",
)
@click.option(
    "--ai-timeout",
    type=float,
    default=DEFAULT_AI_TIMEOUT,
    help=f"AI consultation timeout in seconds (default: {DEFAULT_AI_TIMEOUT}).",
)
@click.option(
    "--no-consultation-cache",
    is_flag=True,
    help="Bypass AI consultation cache (always query providers fresh).",
)
@click.pass_context
@cli_command("review-fidelity")
@handle_keyboard_interrupt()
@with_sync_timeout(FIDELITY_TIMEOUT, "Fidelity review timed out")
def review_fidelity_cmd(
    ctx: click.Context,
    spec_id: str,
    task_id: Optional[str],
    phase_id: Optional[str],
    files: tuple,
    incremental: bool,
    base_branch: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    no_consultation_cache: bool,
) -> None:
    """Compare implementation against specification.

    SPEC_ID is the specification identifier.

    Performs a fidelity review to verify that code implementation
    matches the specification requirements using the AI consultation layer.
    """
    start_time = time.perf_counter()
    cli_ctx = get_context(ctx)
    specs_dir = cli_ctx.specs_dir
    consultation_cache = not no_consultation_cache

    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="VALIDATION_ERROR",
            error_type="validation",
            remediation="Use --specs-dir option or set SDD_SPECS_DIR environment variable",
            details={"hint": "Use --specs-dir or set SDD_SPECS_DIR"},
        )

    # Validate mutually exclusive options
    if task_id and phase_id:
        emit_error(
            "Cannot specify both --task and --phase",
            code="INVALID_OPTIONS",
            error_type="validation",
            remediation="Use either --task or --phase, not both",
            details={"hint": "Use either --task or --phase, not both"},
        )

    llm_status = _get_llm_status()

    # Determine scope
    scope = "spec"
    if task_id:
        scope = f"task:{task_id}"
    elif phase_id:
        scope = f"phase:{phase_id}"
    elif files:
        scope = f"files:{len(files)}"

    # Run the fidelity review
    result = _run_fidelity_review(
        spec_id=spec_id,
        task_id=task_id,
        phase_id=phase_id,
        files=list(files) if files else None,
        ai_provider=ai_provider,
        ai_timeout=ai_timeout,
        consultation_cache=consultation_cache,
        incremental=incremental,
        base_branch=base_branch,
        specs_dir=specs_dir,
        llm_status=llm_status,
        start_time=start_time,
    )

    duration_ms = (time.perf_counter() - start_time) * 1000
    emit_success(
        result,
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


def _run_ai_review(
    spec_id: str,
    specs_dir: Any,
    review_type: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    consultation_cache: bool,
    dry_run: bool,
    llm_status: dict,
) -> dict:
    """
    Run an AI-powered review using ConsultationOrchestrator.

    Args:
        spec_id: Specification ID to review
        specs_dir: Specs directory path
        review_type: Type of review (full, security, feasibility)
        ai_provider: Explicit provider selection
        ai_timeout: Consultation timeout in seconds
        consultation_cache: Whether to use consultation cache
        dry_run: Preview without executing
        llm_status: LLM configuration status

    Returns:
        Dict with review results or dry-run preview
    """
    from pathlib import Path

    # Get template for review type
    template_id = REVIEW_TYPE_TO_TEMPLATE.get(review_type)
    if template_id is None:
        emit_error(
            f"Unknown review type: {review_type}",
            code="INVALID_REVIEW_TYPE",
            error_type="validation",
            remediation=f"Use one of: {', '.join(REVIEW_TYPE_TO_TEMPLATE.keys())}",
            details={"review_type": review_type},
        )

    # Prepare review context
    context = prepare_review_context(
        spec_id=spec_id,
        specs_dir=Path(specs_dir) if specs_dir else None,
        include_tasks=True,
        include_journals=True,
    )

    if context is None:
        emit_error(
            f"Specification '{spec_id}' not found",
            code="SPEC_NOT_FOUND",
            error_type="not_found",
            remediation="Verify the spec ID and that the spec exists in the specs directory",
            details={"spec_id": spec_id},
        )

    # Dry run - preview what would be reviewed
    if dry_run:
        return {
            "spec_id": spec_id,
            "review_type": review_type,
            "template_id": template_id,
            "dry_run": True,
            "llm_status": llm_status,
            "ai_provider": ai_provider,
            "consultation_cache": consultation_cache,
            "message": f"Dry run - {review_type} review would use template {template_id}",
            "spec_title": context.title,
            "task_count": context.stats.total_tasks if context.stats else 0,
        }

    # Import consultation layer components
    try:
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
    except ImportError as exc:
        emit_error(
            "AI consultation layer not available",
            code="AI_NOT_AVAILABLE",
            error_type="unavailable",
            remediation="Ensure foundry_mcp.core.ai_consultation is properly installed",
        )

    # Initialize orchestrator with preferred provider if specified
    preferred_providers = [ai_provider] if ai_provider else []
    orchestrator = ConsultationOrchestrator(
        preferred_providers=preferred_providers,
        default_timeout=ai_timeout,
    )

    # Check if any providers are available
    if not orchestrator.is_available(provider_id=ai_provider):
        provider_msg = f" (requested: {ai_provider})" if ai_provider else ""
        emit_error(
            f"AI-enhanced review requested but no providers available{provider_msg}",
            code="AI_NO_PROVIDER",
            error_type="unavailable",
            remediation="Install and configure an AI provider (gemini, cursor-agent, codex) "
            "or use --type quick for non-AI review.",
            details={
                "spec_id": spec_id,
                "review_type": review_type,
                "requested_provider": ai_provider,
                "llm_status": llm_status,
            },
        )

    # Build context for prompt template
    spec_content = json.dumps(context.spec_data, indent=2)

    # Create consultation request - orchestrator handles prompt building
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.PLAN_REVIEW,
        prompt_id=template_id,
        context={
            "spec_content": spec_content,
            "spec_id": spec_id,
            "title": context.title,
            "review_type": review_type,
        },
        provider_id=ai_provider,
        timeout=ai_timeout,
    )

    # Execute consultation
    try:
        result = orchestrator.consult(request, use_cache=consultation_cache)
    except Exception as exc:
        logger.exception(f"AI consultation failed for {spec_id}")
        emit_error(
            "AI consultation failed",
            code="AI_CONSULTATION_ERROR",
            error_type="error",
            remediation="Check provider configuration and try again",
            details={
                "spec_id": spec_id,
                "review_type": review_type,
            },
        )

    # Build response
    return {
        "spec_id": spec_id,
        "title": context.title,
        "review_type": review_type,
        "template_id": template_id,
        "llm_status": llm_status,
        "ai_provider": result.provider_id if result else ai_provider,
        "consultation_cache": consultation_cache,
        "response": result.content if result else None,
        "model": result.model_used if result else None,
        "cached": result.cache_hit if result else False,
        "stats": {
            "total_tasks": context.stats.total_tasks if context.stats else 0,
            "completed_tasks": context.stats.completed_tasks if context.stats else 0,
            "progress_percentage": context.progress.get("percentage", 0)
            if context.progress
            else 0,
        },
    }


def _get_llm_status() -> dict:
    """Get LLM configuration status."""
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
        logger.debug(f"Failed to get LLM config: {e}")
        return {"configured": False, "error": "Failed to load LLM configuration"}


def _run_fidelity_review(
    spec_id: str,
    task_id: Optional[str],
    phase_id: Optional[str],
    files: Optional[List[str]],
    ai_provider: Optional[str],
    ai_timeout: float,
    consultation_cache: bool,
    incremental: bool,
    base_branch: str,
    specs_dir: Any,
    llm_status: Dict[str, Any],
    start_time: float,
) -> Dict[str, Any]:
    """
    Run a fidelity review using the AI consultation layer.

    Args:
        spec_id: Specification ID to review against
        task_id: Optional task ID for task-scoped review
        phase_id: Optional phase ID for phase-scoped review
        files: Optional list of files to review
        ai_provider: Explicit AI provider selection
        ai_timeout: Consultation timeout in seconds
        consultation_cache: Whether to use consultation cache
        incremental: Only review changed files
        base_branch: Base branch for git diff
        specs_dir: Path to specs directory
        llm_status: LLM configuration status
        start_time: Start time for duration tracking

    Returns:
        Dict with fidelity review results
    """
    from pathlib import Path

    # Import consultation layer components
    try:
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
    except ImportError as exc:
        emit_error(
            "AI consultation layer not available",
            code="AI_NOT_AVAILABLE",
            error_type="unavailable",
            remediation="Ensure foundry_mcp.core.ai_consultation is properly installed",
        )

    # Load spec
    try:
        from foundry_mcp.core.spec import load_spec, find_spec_file

        spec_file = find_spec_file(spec_id, specs_dir)
        if not spec_file:
            emit_error(
                f"Specification not found: {spec_id}",
                code="SPEC_NOT_FOUND",
                error_type="not_found",
                remediation="Verify the spec ID exists using 'sdd list'",
                details={"spec_id": spec_id},
            )
        spec_data = load_spec(spec_file)
    except Exception as exc:
        logger.exception(f"Failed to load spec {spec_id}")
        emit_error(
            "Failed to load spec",
            code="SPEC_LOAD_ERROR",
            error_type="error",
            remediation="Check that the spec file is valid JSON",
            details={"spec_id": spec_id},
        )

    # Determine review scope
    if task_id:
        review_scope = f"Task {task_id}"
    elif phase_id:
        review_scope = f"Phase {phase_id}"
    elif files:
        review_scope = f"Files: {', '.join(files)}"
    else:
        review_scope = "Full specification"

    # Build context for fidelity review
    spec_title = spec_data.get("title", spec_id)
    spec_description = spec_data.get("description", "")

    # Build spec requirements from task details
    spec_requirements = _build_spec_requirements(spec_data, task_id, phase_id)

    # Build implementation artifacts (file contents, git diff if incremental)
    implementation_artifacts = _build_implementation_artifacts(
        spec_data, task_id, phase_id, files, incremental, base_branch
    )

    # Build test results section
    test_results = _build_test_results(spec_data, task_id, phase_id)

    # Build journal entries section
    journal_entries = _build_journal_entries(spec_data, task_id, phase_id)

    # Initialize orchestrator
    preferred_providers = [ai_provider] if ai_provider else []
    orchestrator = ConsultationOrchestrator(
        preferred_providers=preferred_providers,
        default_timeout=ai_timeout,
    )

    # Check if providers are available
    if not orchestrator.is_available(provider_id=ai_provider):
        provider_msg = f" (requested: {ai_provider})" if ai_provider else ""
        emit_error(
            f"Fidelity review requested but no providers available{provider_msg}",
            code="AI_NO_PROVIDER",
            error_type="unavailable",
            remediation="Install and configure an AI provider (gemini, cursor-agent, codex)",
            details={
                "spec_id": spec_id,
                "requested_provider": ai_provider,
                "llm_status": llm_status,
            },
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
        provider_id=ai_provider,
        timeout=ai_timeout,
    )

    # Execute consultation
    try:
        result = orchestrator.consult(request, use_cache=consultation_cache)
    except Exception as exc:
        logger.exception(f"AI fidelity consultation failed for {spec_id}")
        emit_error(
            "AI consultation failed",
            code="AI_CONSULTATION_ERROR",
            error_type="error",
            remediation="Check provider configuration and try again",
            details={
                "spec_id": spec_id,
                "review_scope": review_scope,
            },
        )

    # Parse JSON response if possible
    parsed_response = None
    if result and result.content:
        try:
            # Try to extract JSON from markdown code blocks if present
            content = result.content
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
        except (json.JSONDecodeError, ValueError):
            # Fall back to raw content
            pass

    # Build response
    return {
        "spec_id": spec_id,
        "title": spec_title,
        "review_scope": review_scope,
        "task_id": task_id,
        "phase_id": phase_id,
        "files": files,
        "verdict": parsed_response.get("verdict", "unknown")
        if parsed_response
        else "unknown",
        "llm_status": llm_status,
        "ai_provider": result.provider_id if result else ai_provider,
        "consultation_cache": consultation_cache,
        "response": parsed_response
        if parsed_response
        else result.content
        if result
        else None,
        "raw_response": result.content if result and not parsed_response else None,
        "model": result.model_used if result else None,
        "cached": result.cache_hit if result else False,
        "incremental": incremental,
        "base_branch": base_branch,
    }


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
    from pathlib import Path
    import subprocess

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
                f"- **[{entry_type}]** {entry.get('title', 'Untitled')} ({entry.get('timestamp', 'unknown')[:10]})"
            )
        return "\n".join(lines)

    return "*No journal entries found*"


def _find_task(spec_data: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
    """Find a task by ID in the spec hierarchy (new or legacy format)."""
    hierarchy_nodes = _get_hierarchy_nodes(spec_data)
    if task_id in hierarchy_nodes:
        return hierarchy_nodes[task_id]

    # Legacy tree structure fallback
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

    # Legacy tree structure fallback
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
        # New format: dict keyed by node_id -> node
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


# Top-level aliases
@click.command("review-spec")
@click.argument("spec_id")
@click.option(
    "--type",
    "review_type",
    type=click.Choice(REVIEW_TYPES),
    default="quick",
    help="Type of review to perform.",
)
@click.option(
    "--ai-provider",
    help="Explicit AI provider selection.",
)
@click.option(
    "--ai-timeout",
    type=float,
    default=DEFAULT_AI_TIMEOUT,
    help="AI consultation timeout.",
)
@click.option(
    "--no-consultation-cache",
    is_flag=True,
    help="Bypass AI consultation cache.",
)
@click.pass_context
@cli_command("review-spec-alias")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Review timed out")
def review_spec_alias_cmd(
    ctx: click.Context,
    spec_id: str,
    review_type: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    no_consultation_cache: bool,
) -> None:
    """Run a structural or AI-powered review on a specification (alias)."""
    # Delegate to main command
    ctx.invoke(
        review_spec_cmd,
        spec_id=spec_id,
        review_type=review_type,
        ai_provider=ai_provider,
        ai_timeout=ai_timeout,
        no_consultation_cache=no_consultation_cache,
    )
