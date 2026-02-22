"""Plan review commands for Foundry CLI.

Provides commands for reviewing markdown implementation plans
before converting them to formal JSON specifications.
"""

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from foundry_mcp.cli.logging import cli_command, get_cli_logger
from foundry_mcp.cli.output import emit_error, emit_success
from foundry_mcp.cli.resilience import (
    MEDIUM_TIMEOUT,
    SLOW_TIMEOUT,
    handle_keyboard_interrupt,
    with_sync_timeout,
)
from foundry_mcp.core.spec import find_specs_directory

logger = get_cli_logger()

# Default AI consultation timeout
DEFAULT_AI_TIMEOUT = 360.0

_PLAN_REVIEW_TEMPLATE_ID = "MARKDOWN_PLAN_REVIEW_FULL_V1"


def _extract_plan_name(plan_path: str) -> str:
    """Extract plan name from file path."""
    path = Path(plan_path)
    return path.stem


def _parse_review_summary(content: str) -> dict:
    """
    Parse review content to extract summary counts.

    Returns dict with counts for each section.
    """
    summary = {
        "critical_blockers": 0,
        "major_suggestions": 0,
        "minor_suggestions": 0,
        "questions": 0,
        "praise": 0,
    }

    # Count bullet points in each section
    sections = {
        "Critical Blockers": "critical_blockers",
        "Major Suggestions": "major_suggestions",
        "Minor Suggestions": "minor_suggestions",
        "Questions": "questions",
        "Praise": "praise",
    }

    for section_name, key in sections.items():
        # Find section and count top-level bullets
        pattern = rf"##\s*{section_name}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            section_content = match.group(1)
            # Count lines starting with "- **" (top-level items)
            items = re.findall(r"^\s*-\s+\*\*\[", section_content, re.MULTILINE)
            # If no items found with category tags, count plain bullets
            if not items:
                items = re.findall(r"^\s*-\s+\*\*", section_content, re.MULTILINE)
            # Don't count "None identified" as an item
            if "None identified" in section_content and len(items) <= 1:
                summary[key] = 0
            else:
                summary[key] = len(items)

    return summary


def _format_inline_summary(summary: dict) -> str:
    """Format summary dict into inline text."""
    parts = []
    if summary["critical_blockers"]:
        parts.append(f"{summary['critical_blockers']} critical blocker(s)")
    if summary["major_suggestions"]:
        parts.append(f"{summary['major_suggestions']} major suggestion(s)")
    if summary["minor_suggestions"]:
        parts.append(f"{summary['minor_suggestions']} minor suggestion(s)")
    if summary["questions"]:
        parts.append(f"{summary['questions']} question(s)")
    if summary["praise"]:
        parts.append(f"{summary['praise']} praise item(s)")

    if not parts:
        return "No issues identified"
    return ", ".join(parts)


def _get_llm_status() -> dict:
    """Get current LLM provider status."""
    try:
        from foundry_mcp.core.providers import available_providers

        providers = available_providers()
        return {
            "available": len(providers) > 0,
            "providers": providers,
        }
    except ImportError:
        return {
            "available": False,
            "providers": [],
        }


@click.group("plan")
def plan_group() -> None:
    """Markdown plan review commands."""
    pass


@plan_group.command("review")
@click.argument("plan_path")
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
@cli_command("review")
@handle_keyboard_interrupt()
@with_sync_timeout(SLOW_TIMEOUT, "Plan review timed out")
def plan_review_cmd(
    ctx: click.Context,
    plan_path: str,
    ai_provider: Optional[str],
    ai_timeout: float,
    no_consultation_cache: bool,
    dry_run: bool,
) -> None:
    """Review a markdown implementation plan with AI feedback.

    Analyzes markdown plans before they become formal JSON specifications.
    Writes review output to specs/.plan-reviews/<plan-name>-review.md.

    Examples:

        foundry plan review ./PLAN.md

        foundry plan review ./PLAN.md --ai-provider gemini
    """
    start_time = time.perf_counter()

    llm_status = _get_llm_status()

    # Resolve plan path
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = Path.cwd() / plan_file

    # Check if plan file exists
    if not plan_file.exists():
        emit_error(
            f"Plan file not found: {plan_path}",
            code="PLAN_NOT_FOUND",
            error_type="not_found",
            remediation="Ensure the markdown plan exists at the specified path",
        )
        return

    # Read plan content
    try:
        plan_content = plan_file.read_text(encoding="utf-8")
    except Exception as e:
        emit_error(
            f"Failed to read plan file: {e}",
            code="READ_ERROR",
            error_type="internal",
            remediation="Check file permissions and encoding",
        )
        return

    # Check for empty file
    if not plan_content.strip():
        emit_error(
            "Plan file is empty",
            code="EMPTY_PLAN",
            error_type="validation",
            remediation="Add content to the markdown plan before reviewing",
        )
        return

    plan_name = _extract_plan_name(plan_path)

    # Dry run - just show what would happen
    if dry_run:
        duration_ms = (time.perf_counter() - start_time) * 1000
        emit_success(
            {
                "plan_path": str(plan_file),
                "plan_name": plan_name,
                "dry_run": True,
                "llm_status": llm_status,
                "message": "Dry run - review skipped",
            },
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
        return

    # Check LLM availability
    if not llm_status["available"]:
        emit_error(
            "No AI provider available for plan review",
            code="AI_NO_PROVIDER",
            error_type="ai_provider",
            remediation="Configure an AI provider: set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY",
            details={"required_providers": ["gemini", "codex", "cursor-agent"]},
        )
        return

    # Build consultation request
    template_id = _PLAN_REVIEW_TEMPLATE_ID

    try:
        from foundry_mcp.core.ai_consultation import (
            ConsultationOrchestrator,
            ConsultationRequest,
            ConsultationResult,
            ConsultationWorkflow,
        )

        orchestrator = ConsultationOrchestrator()

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.MARKDOWN_PLAN_REVIEW,
            prompt_id=template_id,
            context={
                "plan_content": plan_content,
                "plan_name": plan_name,
                "plan_path": str(plan_file),
            },
            provider_id=ai_provider,
            timeout=ai_timeout,
        )

        result = orchestrator.consult(
            request,
            use_cache=not no_consultation_cache,
        )

        # Handle ConsultationResult
        if isinstance(result, ConsultationResult):
            if not result.success:
                emit_error(
                    f"AI consultation failed: {result.error}",
                    code="AI_PROVIDER_ERROR",
                    error_type="ai_provider",
                    remediation="Check AI provider configuration or try again later",
                )
                return

            review_content = result.content
            provider_used = result.provider_id
        else:
            # ConsensusResult
            if not result.success:
                emit_error(
                    "AI consultation failed - no successful responses",
                    code="AI_PROVIDER_ERROR",
                    error_type="ai_provider",
                    remediation="Check AI provider configuration or try again later",
                )
                return

            review_content = result.primary_content
            provider_used = result.responses[0].provider_id if result.responses else "unknown"

    except ImportError:
        emit_error(
            "AI consultation module not available",
            code="INTERNAL_ERROR",
            error_type="internal",
            remediation="Check installation of foundry-mcp",
        )
        return
    except Exception as e:
        emit_error(
            f"AI consultation failed: {e}",
            code="AI_PROVIDER_ERROR",
            error_type="ai_provider",
            remediation="Check AI provider configuration or try again later",
        )
        return

    # Parse review summary
    summary = _parse_review_summary(review_content)
    inline_summary = _format_inline_summary(summary)

    # Find specs directory and write review to specs/.plan-reviews/
    specs_dir = find_specs_directory()
    if specs_dir is None:
        emit_error(
            "No specs directory found for storing plan review",
            code="SPECS_NOT_FOUND",
            error_type="validation",
            remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
        )
        return

    plan_reviews_dir = specs_dir / ".plan-reviews"
    try:
        plan_reviews_dir.mkdir(parents=True, exist_ok=True)
        review_file = plan_reviews_dir / f"{plan_name}-review.md"
        review_file.write_text(review_content, encoding="utf-8")
    except Exception as e:
        emit_error(
            f"Failed to write review file: {e}",
            code="WRITE_ERROR",
            error_type="internal",
            remediation="Check write permissions for specs/.plan-reviews/ directory",
        )
        return

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "plan_path": str(plan_file),
            "plan_name": plan_name,
            "review_path": str(review_file),
            "summary": summary,
            "inline_summary": inline_summary,
            "llm_status": llm_status,
            "provider_used": provider_used,
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


# Plan template
PLAN_TEMPLATE = """# {name}

## Mission

[Single sentence describing the primary goal — becomes metadata.mission]

## Objectives

- [Objective 1 — each becomes an array item in metadata.objectives]
- [Objective 2]

## Success Criteria

- [ ] [Measurable criterion 1 — becomes metadata.success_criteria]
- [ ] [Measurable criterion 2]

## Assumptions

- [What we believe to be true — becomes metadata.assumptions]

## Constraints

- [Hard limits we must work within — becomes metadata.constraints]

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | [low/medium/high] | [low/medium/high] | [Mitigation strategy] |

## Open Questions

- [Unresolved question — becomes metadata.open_questions]

## Dependencies

- [External/internal dependencies]

## Phases

### Phase 1: [Phase Name]

**Goal:** [What this phase accomplishes — becomes phase purpose]

**Description:** [Detailed description — becomes phase description]

#### Tasks

- **[Task title]** `[task_category]` `[complexity]`
  - Description: [What to do]
  - File: [repo-relative path, or "N/A" for investigation/research/decision]
  - Acceptance criteria:
    - [How to verify this task is done]
    - [Another criterion]
  - Depends on: [other task titles, or "none"]

#### Verification

- **Run tests:** [test command]
- **Fidelity review:** Compare implementation to spec
- **Manual checks:** [any manual steps, or "none"]
"""


def _slugify(name: str) -> str:
    """Convert a name to a URL-friendly slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug


@plan_group.command("create")
@click.argument("name")
@click.pass_context
@cli_command("create")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Plan creation timed out")
def plan_create_cmd(
    ctx: click.Context,
    name: str,
) -> None:
    """Create a new markdown implementation plan.

    Creates a plan file in specs/.plans/ using the standard template.

    Examples:

        foundry plan create "Add user authentication"

        foundry plan create "Refactor database layer"
    """
    start_time = time.perf_counter()

    # Find specs directory
    specs_dir = find_specs_directory()
    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="SPECS_NOT_FOUND",
            error_type="validation",
            remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
        )
        return

    # Create .plans directory if needed
    plans_dir = specs_dir / ".plans"
    try:
        plans_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        emit_error(
            f"Failed to create plans directory: {e}",
            code="WRITE_ERROR",
            error_type="internal",
            remediation="Check write permissions for specs/.plans/ directory",
        )
        return

    # Generate plan filename with date suffix (mirrors spec_id date convention)
    plan_slug = _slugify(name)
    date_suffix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    plan_slug_dated = f"{plan_slug}-{date_suffix}"
    plan_file = plans_dir / f"{plan_slug_dated}.md"

    # Check if plan already exists
    if plan_file.exists():
        emit_error(
            f"Plan already exists: {plan_file}",
            code="DUPLICATE_ENTRY",
            error_type="conflict",
            remediation="Use a different name or delete the existing plan",
            details={"plan_path": str(plan_file)},
        )
        return

    # Generate plan content from template
    plan_content = PLAN_TEMPLATE.format(name=name)

    # Write plan file
    try:
        plan_file.write_text(plan_content, encoding="utf-8")
    except Exception as e:
        emit_error(
            f"Failed to write plan file: {e}",
            code="WRITE_ERROR",
            error_type="internal",
            remediation="Check write permissions for specs/.plans/ directory",
        )
        return

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "plan_name": name,
            "plan_slug": plan_slug_dated,
            "plan_path": str(plan_file),
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )


@plan_group.command("list")
@click.pass_context
@cli_command("list")
@handle_keyboard_interrupt()
@with_sync_timeout(MEDIUM_TIMEOUT, "Plan listing timed out")
def plan_list_cmd(ctx: click.Context) -> None:
    """List all markdown implementation plans.

    Lists plans from specs/.plans/ directory.

    Examples:

        foundry plan list
    """
    start_time = time.perf_counter()

    # Find specs directory
    specs_dir = find_specs_directory()
    if specs_dir is None:
        emit_error(
            "No specs directory found",
            code="SPECS_NOT_FOUND",
            error_type="validation",
            remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
        )
        return

    plans_dir = specs_dir / ".plans"

    # Check if plans directory exists
    if not plans_dir.exists():
        emit_success(
            {
                "plans": [],
                "count": 0,
                "plans_dir": str(plans_dir),
            },
            telemetry={"duration_ms": round((time.perf_counter() - start_time) * 1000, 2)},
        )
        return

    # List all markdown files in plans directory
    plans = []
    for plan_file in sorted(plans_dir.glob("*.md")):
        stat = plan_file.stat()
        plans.append(
            {
                "name": plan_file.stem,
                "path": str(plan_file),
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            }
        )

    # Check for reviews
    reviews_dir = specs_dir / ".plan-reviews"
    for plan in plans:
        plan_name = plan["name"]
        review_files = list(reviews_dir.glob(f"{plan_name}-*.md")) if reviews_dir.exists() else []
        plan["reviews"] = [rf.stem for rf in review_files]
        plan["has_review"] = len(review_files) > 0

    duration_ms = (time.perf_counter() - start_time) * 1000

    emit_success(
        {
            "plans": plans,
            "count": len(plans),
            "plans_dir": str(plans_dir),
        },
        telemetry={"duration_ms": round(duration_ms, 2)},
    )
