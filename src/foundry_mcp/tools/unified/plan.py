"""Unified plan tooling with action routing."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.ai_consultation import (
    ConsultationOrchestrator,
    ConsultationRequest,
    ConsultationResult,
    ConsultationWorkflow,
    ConsensusResult,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.providers import available_providers
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    ai_no_provider_error,
    error_response,
    success_response,
)
from foundry_mcp.core.security import is_prompt_injection
from foundry_mcp.core.spec import find_specs_directory
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

REVIEW_TYPES = ["quick", "full", "security", "feasibility"]
REVIEW_TYPE_TO_TEMPLATE = {
    "full": "MARKDOWN_PLAN_REVIEW_FULL_V1",
    "quick": "MARKDOWN_PLAN_REVIEW_QUICK_V1",
    "security": "MARKDOWN_PLAN_REVIEW_SECURITY_V1",
    "feasibility": "MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1",
}


def _extract_plan_name(plan_path: str) -> str:
    """Extract plan name from file path."""

    return Path(plan_path).stem


def _parse_review_summary(content: str) -> dict:
    """Parse review markdown to extract section counts."""

    summary = {
        "critical_blockers": 0,
        "major_suggestions": 0,
        "minor_suggestions": 0,
        "questions": 0,
        "praise": 0,
    }

    sections = {
        "Critical Blockers": "critical_blockers",
        "Major Suggestions": "major_suggestions",
        "Minor Suggestions": "minor_suggestions",
        "Questions": "questions",
        "Praise": "praise",
    }

    for section_name, key in sections.items():
        pattern = rf"##\s*{section_name}\s*\n(.*?)(?=\n##|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        section_content = match.group(1)
        items = re.findall(r"^\s*-\s+\*\*\[", section_content, re.MULTILINE)
        if not items:
            items = re.findall(r"^\s*-\s+\*\*", section_content, re.MULTILINE)
        if "None identified" in section_content and len(items) <= 1:
            summary[key] = 0
        else:
            summary[key] = len(items)

    return summary


def _format_inline_summary(summary: dict) -> str:
    """Format summary dict into inline human-readable string."""

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

    return ", ".join(parts) if parts else "No issues identified"


def _get_llm_status() -> dict:
    """Return current provider availability."""

    providers = available_providers()
    return {"available": bool(providers), "providers": providers}


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
    """Convert a name to a slug."""

    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"[-\s]+", "-", slug)


def perform_plan_review(
    *,
    plan_path: str,
    review_type: str = "full",
    ai_provider: Optional[str] = None,
    ai_timeout: float = 120.0,
    consultation_cache: bool = True,
    dry_run: bool = False,
) -> dict:
    """Execute the plan review workflow and return serialized response."""

    start_time = time.perf_counter()

    if review_type not in REVIEW_TYPES:
        return asdict(
            error_response(
                f"Invalid review_type: {review_type}. Must be one of: {', '.join(REVIEW_TYPES)}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {', '.join(REVIEW_TYPES)}",
                details={"review_type": review_type, "allowed": REVIEW_TYPES},
            )
        )

    for field_name, field_value in (
        ("plan_path", plan_path),
        ("ai_provider", ai_provider),
    ):
        if field_value and is_prompt_injection(field_value):
            _metrics.counter(
                "plan_review.security_blocked",
                labels={"tool": "plan-review", "reason": "prompt_injection"},
            )
            return asdict(
                error_response(
                    f"Input validation failed for {field_name}",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Remove special characters or instruction-like patterns from input.",
                )
            )

    llm_status = _get_llm_status()

    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = Path.cwd() / plan_file

    if not plan_file.exists():
        _metrics.counter(
            "plan_review.errors",
            labels={"tool": "plan-review", "error_type": "not_found"},
        )
        return asdict(
            error_response(
                f"Plan file not found: {plan_path}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure the markdown plan exists at the specified path",
                details={"plan_path": plan_path},
            )
        )

    try:
        plan_content = plan_file.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors
        _metrics.counter(
            "plan_review.errors",
            labels={"tool": "plan-review", "error_type": "read_error"},
        )
        return asdict(
            error_response(
                f"Failed to read plan file: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check file permissions and encoding",
                details={"plan_path": str(plan_file)},
            )
        )

    if not plan_content.strip():
        _metrics.counter(
            "plan_review.errors",
            labels={"tool": "plan-review", "error_type": "empty_plan"},
        )
        return asdict(
            error_response(
                "Plan file is empty",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Add content to the markdown plan before reviewing",
                details={"plan_path": str(plan_file)},
            )
        )

    plan_name = _extract_plan_name(plan_file.name)

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
                telemetry={
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
                },
            )
        )

    if not llm_status["available"]:
        return asdict(
            ai_no_provider_error(
                "No AI provider available for plan review",
                required_providers=["gemini", "codex", "cursor-agent"],
            )
        )

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
        result = orchestrator.consult(request, use_cache=consultation_cache)

        consensus_info: Optional[dict] = None
        provider_used: Optional[str] = None

        if isinstance(result, ConsultationResult):
            if not result.success:
                return asdict(
                    error_response(
                        f"AI consultation failed: {result.error}",
                        error_code=ErrorCode.AI_PROVIDER_ERROR,
                        error_type=ErrorType.AI_PROVIDER,
                        remediation="Check AI provider configuration or try again later",
                    )
                )
            review_content = result.content
            provider_used = result.provider_id
        elif isinstance(result, ConsensusResult):
            if not result.success:
                return asdict(
                    error_response(
                        "AI consultation failed - no successful responses",
                        error_code=ErrorCode.AI_PROVIDER_ERROR,
                        error_type=ErrorType.AI_PROVIDER,
                        remediation="Check AI provider configuration or try again later",
                    )
                )
            review_content = result.primary_content
            providers_consulted = [r.provider_id for r in result.responses]
            provider_used = providers_consulted[0] if providers_consulted else "unknown"
            consensus_info = {
                "providers_consulted": providers_consulted,
                "successful": result.agreement.successful_providers
                if result.agreement
                else 0,
                "failed": result.agreement.failed_providers if result.agreement else 0,
            }
        else:  # pragma: no cover - defensive branch
            logger.error("Unknown consultation result type: %s", type(result))
            return asdict(
                error_response(
                    "Unsupported consultation result",
                    error_code=ErrorCode.AI_PROVIDER_ERROR,
                    error_type=ErrorType.AI_PROVIDER,
                )
            )
    except Exception as exc:  # pragma: no cover - orchestration errors
        _metrics.counter(
            "plan_review.errors",
            labels={"tool": "plan-review", "error_type": "consultation_error"},
        )
        return asdict(
            error_response(
                f"AI consultation failed: {exc}",
                error_code=ErrorCode.AI_PROVIDER_ERROR,
                error_type=ErrorType.AI_PROVIDER,
                remediation="Check AI provider configuration or try again later",
            )
        )

    summary = _parse_review_summary(review_content)
    inline_summary = _format_inline_summary(summary)

    specs_dir = find_specs_directory()
    if specs_dir is None:
        return asdict(
            error_response(
                "No specs directory found for storing plan review",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
            )
        )

    plan_reviews_dir = specs_dir / ".plan-reviews"
    try:
        plan_reviews_dir.mkdir(parents=True, exist_ok=True)
        review_file = plan_reviews_dir / f"{plan_name}-{review_type}.md"
        review_file.write_text(review_content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors
        _metrics.counter(
            "plan_review.errors",
            labels={"tool": "plan-review", "error_type": "write_error"},
        )
        return asdict(
            error_response(
                f"Failed to write review file: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check write permissions for specs/.plan-reviews/ directory",
            )
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    _metrics.counter(
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


def perform_plan_create(name: str, template: str = "detailed") -> dict:
    """Create a markdown implementation plan using the requested template."""

    start_time = time.perf_counter()

    if template not in PLAN_TEMPLATES:
        return asdict(
            error_response(
                f"Invalid template: {template}. Must be one of: simple, detailed",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Use 'simple' or 'detailed' template",
                details={
                    "template": template,
                    "allowed": sorted(PLAN_TEMPLATES.keys()),
                },
            )
        )

    if is_prompt_injection(name):
        _metrics.counter(
            "plan_create.security_blocked",
            labels={"tool": "plan-create", "reason": "prompt_injection"},
        )
        return asdict(
            error_response(
                "Input validation failed for name",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Remove special characters or instruction-like patterns from input.",
            )
        )

    specs_dir = find_specs_directory()
    if specs_dir is None:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
            )
        )

    plans_dir = specs_dir / ".plans"
    try:
        plans_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return asdict(
            error_response(
                f"Failed to create plans directory: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check write permissions for specs/.plans/ directory",
            )
        )

    plan_slug = _slugify(name)
    plan_file = plans_dir / f"{plan_slug}.md"

    if plan_file.exists():
        return asdict(
            error_response(
                f"Plan already exists: {plan_file}",
                error_code=ErrorCode.DUPLICATE_ENTRY,
                error_type=ErrorType.CONFLICT,
                remediation="Use a different name or delete the existing plan",
                details={"plan_path": str(plan_file)},
            )
        )

    plan_content = PLAN_TEMPLATES[template].format(name=name)
    try:
        plan_file.write_text(plan_content, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - filesystem errors
        return asdict(
            error_response(
                f"Failed to write plan file: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check write permissions for specs/.plans/ directory",
            )
        )

    duration_ms = (time.perf_counter() - start_time) * 1000
    _metrics.counter(
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


def perform_plan_list() -> dict:
    """List plans stored in specs/.plans and any associated reviews."""

    start_time = time.perf_counter()

    specs_dir = find_specs_directory()
    if specs_dir is None:
        return asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Create a specs/ directory with pending/active/completed/archived subdirectories",
            )
        )

    plans_dir = specs_dir / ".plans"
    if not plans_dir.exists():
        return asdict(
            success_response(
                data={"plans": [], "count": 0, "plans_dir": str(plans_dir)},
                telemetry={
                    "duration_ms": round((time.perf_counter() - start_time) * 1000, 2)
                },
            )
        )

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

    reviews_dir = specs_dir / ".plan-reviews"
    for plan in plans:
        plan_name = plan["name"]
        if reviews_dir.exists():
            review_files = list(reviews_dir.glob(f"{plan_name}-*.md"))
        else:
            review_files = []
        plan["reviews"] = [rf.stem for rf in review_files]
        plan["has_review"] = bool(review_files)

    duration_ms = (time.perf_counter() - start_time) * 1000
    _metrics.counter("plan_list.completed", labels={"tool": "plan-list"})

    return asdict(
        success_response(
            data={"plans": plans, "count": len(plans), "plans_dir": str(plans_dir)},
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


_ACTION_SUMMARY = {
    "create": "Create markdown plan templates in specs/.plans",
    "list": "Enumerate existing markdown plans and review coverage",
    "review": "Run AI-assisted review workflows for markdown plans",
}


def _handle_plan_create(**payload: Any) -> dict:
    name = payload.get("name")
    template = payload.get("template", "detailed")
    if not name:
        return asdict(
            error_response(
                "Missing required parameter 'name' for plan.create",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a plan name when action=create",
            )
        )
    return perform_plan_create(name=name, template=template)


def _handle_plan_list(**_: Any) -> dict:
    return perform_plan_list()


def _handle_plan_review(**payload: Any) -> dict:
    plan_path = payload.get("plan_path")
    if not plan_path:
        return asdict(
            error_response(
                "Missing required parameter 'plan_path' for plan.review",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a markdown plan path when action=review",
            )
        )
    return perform_plan_review(
        plan_path=plan_path,
        review_type=payload.get("review_type", "full"),
        ai_provider=payload.get("ai_provider"),
        ai_timeout=payload.get("ai_timeout", 120.0),
        consultation_cache=payload.get("consultation_cache", True),
        dry_run=payload.get("dry_run", False),
    )


_PLAN_ROUTER = ActionRouter(
    tool_name="plan",
    actions=[
        ActionDefinition(
            name="create",
            handler=_handle_plan_create,
            summary=_ACTION_SUMMARY["create"],
        ),
        ActionDefinition(
            name="list", handler=_handle_plan_list, summary=_ACTION_SUMMARY["list"]
        ),
        ActionDefinition(
            name="review",
            handler=_handle_plan_review,
            summary=_ACTION_SUMMARY["review"],
            aliases=("plan-review",),
        ),
    ],
)


def _dispatch_plan_action(action: str, payload: Dict[str, Any]) -> dict:
    try:
        return _PLAN_ROUTER.dispatch(action=action, **payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported plan action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_plan_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated plan tool."""

    @canonical_tool(
        mcp,
        canonical_name="plan",
    )
    @mcp_tool(tool_name="plan", emit_metrics=True, audit=True)
    def plan(
        action: str,
        name: Optional[str] = None,
        template: str = "detailed",
        plan_path: Optional[str] = None,
        review_type: str = "full",
        ai_provider: Optional[str] = None,
        ai_timeout: float = 120.0,
        consultation_cache: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """Execute plan workflows via the action router."""

        payload = {
            "name": name,
            "template": template,
            "plan_path": plan_path,
            "review_type": review_type,
            "ai_provider": ai_provider,
            "ai_timeout": ai_timeout,
            "consultation_cache": consultation_cache,
            "dry_run": dry_run,
        }
        return _dispatch_plan_action(action=action, payload=payload)

    logger.debug("Registered unified plan tool")


__all__ = [
    "register_unified_plan_tool",
    "perform_plan_review",
    "perform_plan_create",
    "perform_plan_list",
]
