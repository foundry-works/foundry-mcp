"""
Plan review tools for foundry-mcp.

Provides MCP tools for reviewing markdown implementation plans
before converting them to formal JSON specifications. Enables
iterative review cycles with AI feedback.
"""

import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    success_response,
    error_response,
    ai_no_provider_error,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import (
    get_metrics,
    mcp_tool,
)
from foundry_mcp.core.security import (
    is_prompt_injection,
)
from foundry_mcp.core.ai_consultation import (
    ConsultationOrchestrator,
    ConsultationRequest,
    ConsultationWorkflow,
    ConsultationResult,
)
from foundry_mcp.core.providers import available_providers
from foundry_mcp.core.spec import find_specs_directory

logger = logging.getLogger(__name__)

# Review types supported
REVIEW_TYPES = ["quick", "full", "security", "feasibility"]

# Map review types to MARKDOWN_PLAN_REVIEW templates
REVIEW_TYPE_TO_TEMPLATE = {
    "full": "MARKDOWN_PLAN_REVIEW_FULL_V1",
    "quick": "MARKDOWN_PLAN_REVIEW_QUICK_V1",
    "security": "MARKDOWN_PLAN_REVIEW_SECURITY_V1",
    "feasibility": "MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1",
}


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
    providers = available_providers()
    return {
        "available": len(providers) > 0,
        "providers": providers,
    }


def register_plan_review_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register plan review tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()

    @canonical_tool(
        mcp,
        canonical_name="plan-review",
    )
    @mcp_tool(tool_name="plan-review", emit_metrics=True, audit=True)
    def plan_review(
        plan_path: str,
        review_type: str = "full",
        ai_provider: Optional[str] = None,
        ai_timeout: float = 120.0,
        consultation_cache: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Review a markdown implementation plan with AI feedback.

        Analyzes markdown plans before they become formal JSON specifications.
        Supports iterative review cycles to refine plans with AI feedback.

        Args:
            plan_path: Path to markdown plan file
            review_type: Type of review - "quick", "full", "security", or "feasibility"
            ai_provider: Explicit AI provider selection (e.g., gemini, cursor-agent)
            ai_timeout: AI consultation timeout in seconds (default: 120)
            consultation_cache: Whether to use AI consultation cache (default: True)
            dry_run: Preview what would be reviewed without executing

        Returns:
            JSON object with:
            - plan_path: Path to the reviewed plan
            - plan_name: Name extracted from plan file
            - review_type: Type of review performed
            - review_path: Path where review was written
            - summary: Counts of blockers, suggestions, questions, praise
            - inline_summary: Human-readable summary string
            - llm_status: LLM configuration status

        WHEN TO USE:
        - Review markdown plans before creating JSON specs
        - Get AI feedback on implementation approaches
        - Iterate on plan refinements before committing
        - Validate completeness, architecture, and risks

        LIMITATIONS:
        - Requires LLM configuration for all review types
        - Plan file must be valid markdown
        """
        start_time = time.perf_counter()

        # Validate review_type
        if review_type not in REVIEW_TYPES:
            return asdict(
                error_response(
                    f"Invalid review_type: {review_type}. Must be one of: {', '.join(REVIEW_TYPES)}",
                    error_code="INVALID_REVIEW_TYPE",
                    error_type="validation",
                    remediation=f"Use one of: {', '.join(REVIEW_TYPES)}",
                )
            )

        # Input validation: check for prompt injection
        for field_name, field_value in [
            ("plan_path", plan_path),
            ("ai_provider", ai_provider),
        ]:
            if field_value and is_prompt_injection(field_value):
                metrics.counter(
                    "plan_review.security_blocked",
                    labels={"tool": "plan-review", "reason": "prompt_injection"},
                )
                return asdict(
                    error_response(
                        f"Input validation failed for {field_name}",
                        error_code="VALIDATION_ERROR",
                        error_type="security",
                        remediation="Remove special characters or instruction-like patterns from input.",
                    )
                )

        llm_status = _get_llm_status()

        # Resolve plan path
        plan_file = Path(plan_path)
        if not plan_file.is_absolute():
            plan_file = Path.cwd() / plan_file

        # Check if plan file exists
        if not plan_file.exists():
            metrics.counter(
                "plan_review.errors",
                labels={"tool": "plan-review", "error_type": "not_found"},
            )
            return asdict(
                error_response(
                    f"Plan file not found: {plan_path}",
                    error_code="PLAN_NOT_FOUND",
                    error_type="not_found",
                    remediation="Ensure the markdown plan exists at the specified path",
                )
            )

        # Read plan content
        try:
            plan_content = plan_file.read_text(encoding="utf-8")
        except Exception as e:
            metrics.counter(
                "plan_review.errors",
                labels={"tool": "plan-review", "error_type": "read_error"},
            )
            return asdict(
                error_response(
                    f"Failed to read plan file: {e}",
                    error_code="READ_ERROR",
                    error_type="internal",
                    remediation="Check file permissions and encoding",
                )
            )

        # Check for empty file
        if not plan_content.strip():
            metrics.counter(
                "plan_review.errors",
                labels={"tool": "plan-review", "error_type": "empty_plan"},
            )
            return asdict(
                error_response(
                    "Plan file is empty",
                    error_code="EMPTY_PLAN",
                    error_type="validation",
                    remediation="Add content to the markdown plan before reviewing",
                )
            )

        plan_name = _extract_plan_name(plan_path)

        # Dry run - just show what would happen
        if dry_run:
            return asdict(
                success_response(
                    data={
                        "plan_path": str(plan_file),
                        "plan_name": plan_name,
                        "review_type": review_type,
                        "dry_run": True,
                        "llm_status": llm_status,
                        "message": "Dry run - review skipped",
                    },
                    telemetry={"duration_ms": round((time.perf_counter() - start_time) * 1000, 2)},
                )
            )

        # Check LLM availability
        if not llm_status["available"]:
            return asdict(
                ai_no_provider_error(
                    "No AI provider available for plan review",
                    required_providers=["gemini", "codex", "cursor-agent"],
                )
            )

        # Build consultation request
        template_id = REVIEW_TYPE_TO_TEMPLATE[review_type]

        try:
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
                use_cache=consultation_cache,
            )

            # Handle ConsultationResult
            consensus_info = None  # Only set for multi-model
            if isinstance(result, ConsultationResult):
                if not result.success:
                    return asdict(
                        error_response(
                            f"AI consultation failed: {result.error}",
                            error_code="AI_PROVIDER_ERROR",
                            error_type="ai_provider",
                            remediation="Check AI provider configuration or try again later",
                        )
                    )

                review_content = result.content
                provider_used = result.provider_id
            else:
                # ConsensusResult
                if not result.success:
                    return asdict(
                        error_response(
                            "AI consultation failed - no successful responses",
                            error_code="AI_PROVIDER_ERROR",
                            error_type="ai_provider",
                            remediation="Check AI provider configuration or try again later",
                        )
                    )

                review_content = result.primary_content
                # Report all providers consulted for consensus
                providers_consulted = [r.provider_id for r in result.responses]
                provider_used = providers_consulted[0] if providers_consulted else "unknown"
                # Store full consensus info for response
                consensus_info = {
                    "providers_consulted": providers_consulted,
                    "successful": result.agreement.successful_providers,
                    "failed": result.agreement.failed_providers,
                }

        except Exception as e:
            metrics.counter(
                "plan_review.errors",
                labels={"tool": "plan-review", "error_type": "consultation_error"},
            )
            return asdict(
                error_response(
                    f"AI consultation failed: {e}",
                    error_code="AI_PROVIDER_ERROR",
                    error_type="ai_provider",
                    remediation="Check AI provider configuration or try again later",
                )
            )

        # Parse review summary
        summary = _parse_review_summary(review_content)
        inline_summary = _format_inline_summary(summary)

        # Find specs directory and write review to specs/.plan-reviews/
        specs_dir = find_specs_directory()
        if specs_dir is None:
            return asdict(
                error_response(
                    "No specs directory found for storing plan review",
                    error_code="SPECS_NOT_FOUND",
                    error_type="validation",
                    remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
                )
            )

        plan_reviews_dir = specs_dir / ".plan-reviews"
        try:
            plan_reviews_dir.mkdir(parents=True, exist_ok=True)
            review_file = plan_reviews_dir / f"{plan_name}-{review_type}.md"
            review_file.write_text(review_content, encoding="utf-8")
        except Exception as e:
            metrics.counter(
                "plan_review.errors",
                labels={"tool": "plan-review", "error_type": "write_error"},
            )
            return asdict(
                error_response(
                    f"Failed to write review file: {e}",
                    error_code="WRITE_ERROR",
                    error_type="internal",
                    remediation="Check write permissions for specs/.plan-reviews/ directory",
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        metrics.counter(
            "plan_review.completed",
            labels={"tool": "plan-review", "review_type": review_type},
        )

        response_data = {
            "plan_path": str(plan_file),
            "plan_name": plan_name,
            "review_type": review_type,
            "review_path": str(review_file),
            "summary": summary,
            "inline_summary": inline_summary,
            "llm_status": llm_status,
            "provider_used": provider_used,
        }
        if consensus_info:
            response_data["consensus"] = consensus_info

        return asdict(
            success_response(
                data=response_data,
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    # Plan templates
    PLAN_TEMPLATES = {
        "simple": """# {name}

## Objective

[Describe the primary goal of this plan]

## Scope

[What is included/excluded from this plan]

## Tasks

1. [Task 1]
2. [Task 2]
3. [Task 3]

## Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
""",
        "detailed": """# {name}

## Objective

[Describe the primary goal of this plan]

## Scope

### In Scope
- [Item 1]
- [Item 2]

### Out of Scope
- [Item 1]

## Phases

### Phase 1: [Phase Name]

**Purpose**: [Why this phase exists]

**Tasks**:
1. [Task 1]
2. [Task 2]

**Verification**: [How to verify phase completion]

### Phase 2: [Phase Name]

**Purpose**: [Why this phase exists]

**Tasks**:
1. [Task 1]
2. [Task 2]

**Verification**: [How to verify phase completion]

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| [Risk 1] | [High/Medium/Low] | [Mitigation strategy] |

## Success Criteria

- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]
""",
    }

    def _slugify(name: str) -> str:
        """Convert a name to a URL-friendly slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "-", slug)
        return slug

    @canonical_tool(
        mcp,
        canonical_name="plan-create",
    )
    @mcp_tool(tool_name="plan-create", emit_metrics=True, audit=True)
    def plan_create(
        name: str,
        template: str = "detailed",
    ) -> dict:
        """
        Create a new markdown implementation plan.

        Creates a plan file in specs/.plans/ with the specified template.

        Args:
            name: Human-readable name for the plan
            template: Template to use - "simple" or "detailed" (default: "detailed")

        Returns:
            JSON object with:
            - plan_name: The plan name
            - plan_slug: URL-friendly slug
            - plan_path: Path to created plan file
            - template: Template used

        WHEN TO USE:
        - Create a new implementation plan before starting complex work
        - Initialize a plan that will be reviewed and refined
        - Set up a structured document for phased implementation

        LIMITATIONS:
        - Plan name must be unique (no duplicate slugs)
        - Template must be "simple" or "detailed"
        """
        start_time = time.perf_counter()

        # Validate template
        if template not in PLAN_TEMPLATES:
            return asdict(
                error_response(
                    f"Invalid template: {template}. Must be one of: simple, detailed",
                    error_code="INVALID_TEMPLATE",
                    error_type="validation",
                    remediation="Use 'simple' or 'detailed' template",
                )
            )

        # Input validation: check for prompt injection
        if is_prompt_injection(name):
            metrics.counter(
                "plan_create.security_blocked",
                labels={"tool": "plan-create", "reason": "prompt_injection"},
            )
            return asdict(
                error_response(
                    "Input validation failed for name",
                    error_code="VALIDATION_ERROR",
                    error_type="security",
                    remediation="Remove special characters or instruction-like patterns from input.",
                )
            )

        # Find specs directory
        specs_dir = find_specs_directory()
        if specs_dir is None:
            return asdict(
                error_response(
                    "No specs directory found",
                    error_code="SPECS_NOT_FOUND",
                    error_type="validation",
                    remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
                )
            )

        # Create .plans directory if needed
        plans_dir = specs_dir / ".plans"
        try:
            plans_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return asdict(
                error_response(
                    f"Failed to create plans directory: {e}",
                    error_code="WRITE_ERROR",
                    error_type="internal",
                    remediation="Check write permissions for specs/.plans/ directory",
                )
            )

        # Generate plan filename
        plan_slug = _slugify(name)
        plan_file = plans_dir / f"{plan_slug}.md"

        # Check if plan already exists
        if plan_file.exists():
            return asdict(
                error_response(
                    f"Plan already exists: {plan_file}",
                    error_code="DUPLICATE_ENTRY",
                    error_type="conflict",
                    remediation="Use a different name or delete the existing plan",
                    details={"plan_path": str(plan_file)},
                )
            )

        # Generate plan content from template
        plan_content = PLAN_TEMPLATES[template].format(name=name)

        # Write plan file
        try:
            plan_file.write_text(plan_content, encoding="utf-8")
        except Exception as e:
            return asdict(
                error_response(
                    f"Failed to write plan file: {e}",
                    error_code="WRITE_ERROR",
                    error_type="internal",
                    remediation="Check write permissions for specs/.plans/ directory",
                )
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        metrics.counter(
            "plan_create.completed",
            labels={"tool": "plan-create", "template": template},
        )

        return asdict(
            success_response(
                data={
                    "plan_name": name,
                    "plan_slug": plan_slug,
                    "plan_path": str(plan_file),
                    "template": template,
                },
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    @canonical_tool(
        mcp,
        canonical_name="plan-list",
    )
    @mcp_tool(tool_name="plan-list", emit_metrics=True, audit=True)
    def plan_list() -> dict:
        """
        List all markdown implementation plans.

        Lists plans from specs/.plans/ directory with review status.

        Returns:
            JSON object with:
            - plans: Array of plan objects with name, path, size, modified, reviews
            - count: Total number of plans
            - plans_dir: Path to plans directory

        WHEN TO USE:
        - View available plans before starting work
        - Check which plans have been reviewed
        - Find a plan to continue working on
        """
        start_time = time.perf_counter()

        # Find specs directory
        specs_dir = find_specs_directory()
        if specs_dir is None:
            return asdict(
                error_response(
                    "No specs directory found",
                    error_code="SPECS_NOT_FOUND",
                    error_type="validation",
                    remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
                )
            )

        plans_dir = specs_dir / ".plans"

        # Check if plans directory exists
        if not plans_dir.exists():
            return asdict(
                success_response(
                    data={
                        "plans": [],
                        "count": 0,
                        "plans_dir": str(plans_dir),
                    },
                    telemetry={"duration_ms": round((time.perf_counter() - start_time) * 1000, 2)},
                )
            )

        # List all markdown files in plans directory
        plans = []
        for plan_file in sorted(plans_dir.glob("*.md")):
            stat = plan_file.stat()
            plans.append({
                "name": plan_file.stem,
                "path": str(plan_file),
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })

        # Check for reviews
        reviews_dir = specs_dir / ".plan-reviews"
        for plan in plans:
            plan_name = plan["name"]
            review_files = list(reviews_dir.glob(f"{plan_name}-*.md")) if reviews_dir.exists() else []
            plan["reviews"] = [rf.stem for rf in review_files]
            plan["has_review"] = len(review_files) > 0

        duration_ms = (time.perf_counter() - start_time) * 1000

        metrics.counter(
            "plan_list.completed",
            labels={"tool": "plan-list"},
        )

        return asdict(
            success_response(
                data={
                    "plans": plans,
                    "count": len(plans),
                    "plans_dir": str(plans_dir),
                },
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )
