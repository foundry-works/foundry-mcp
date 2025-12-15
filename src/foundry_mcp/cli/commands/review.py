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
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.tools.unified.documentation_helpers import (
    _build_implementation_artifacts,
    _build_journal_entries,
    _build_spec_requirements,
    _build_test_results,
)
from foundry_mcp.tools.unified.review_helpers import (
    DEFAULT_AI_TIMEOUT,
    REVIEW_TYPES,
    _get_llm_status,
    _run_ai_review,
    _run_quick_review,
)

logger = get_cli_logger()


def _emit_review_envelope(envelope: Dict[str, Any], *, duration_ms: float) -> None:
    """Emit a response-v2 envelope returned by shared review helpers."""

    if envelope.get("success") is True:
        emit_success(
            envelope.get("data", {}),
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
        return

    payload = envelope.get("data") or {}

    error_code = payload.get("error_code", "INTERNAL_ERROR")
    if hasattr(error_code, "value"):
        error_code = error_code.value

    error_type = payload.get("error_type", "internal")
    if hasattr(error_type, "value"):
        error_type = error_type.value

    emit_error(
        envelope.get("error") or "Review failed",
        code=str(error_code),
        error_type=str(error_type),
        remediation=payload.get("remediation"),
        details=payload.get("details"),
    )


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
@cli_command("spec")
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

    if review_type == "quick":
        envelope = _run_quick_review(
            spec_id=spec_id,
            specs_dir=specs_dir,
            dry_run=dry_run,
            llm_status=llm_status,
            start_time=start_time,
        )
    else:
        envelope = _run_ai_review(
            spec_id=spec_id,
            specs_dir=specs_dir,
            review_type=review_type,
            ai_provider=ai_provider,
            model=model,
            ai_timeout=ai_timeout,
            consultation_cache=not no_consultation_cache,
            dry_run=dry_run,
            llm_status=llm_status,
            start_time=start_time,
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    _emit_review_envelope(envelope, duration_ms=duration_ms)


@review_group.command("tools")
@click.pass_context
@cli_command("tools")
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
@cli_command("plan-tools")
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
@cli_command("fidelity")
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
    if task_id:
        pass
    elif phase_id:
        pass
    elif files:
        f"files:{len(files)}"

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

    # Import consultation layer components
    try:
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationWorkflow,
        )
    except ImportError:
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
    except Exception:
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
    orchestrator = ConsultationOrchestrator(
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
    except Exception:
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
