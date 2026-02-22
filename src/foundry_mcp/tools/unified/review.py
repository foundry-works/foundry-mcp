"""Unified review tooling with action routing.

Consolidates spec review, review tool discovery, and fidelity review
into a single `review(action=...)` entry point.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.ai_consultation import (
    ConsensusResult,
    ConsultationOrchestrator,
    ConsultationRequest,
    ConsultationResult,
    ConsultationWorkflow,
    ProviderResponse,
)
from foundry_mcp.core.llm_config.consultation import get_consultation_config, load_consultation_config
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.prompts.fidelity_review import (
    FIDELITY_SYNTHESIZED_RESPONSE_SCHEMA,
)
from foundry_mcp.core.providers import get_provider_statuses
from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.core.security import is_prompt_injection
from foundry_mcp.core.spec import find_spec_file, find_specs_directory, load_spec
from foundry_mcp.tools.unified.common import dispatch_with_standard_errors
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
)

from .documentation_helpers import (
    _build_implementation_artifacts,
    _build_journal_entries,
    _build_spec_overview,
    _build_spec_requirements,
    _build_subsequent_phases,
    _build_test_results,
)
from .review_helpers import (
    DEFAULT_AI_TIMEOUT,
    REVIEW_TYPES,
    _get_llm_status,
    _run_ai_review,
    _run_quick_review,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()


def _parse_json_content(content: str) -> Optional[dict]:
    if not content:
        return None

    candidate = content
    if "```json" in candidate:
        start = candidate.find("```json") + 7
        end = candidate.find("```", start)
        if end > start:
            candidate = candidate[start:end].strip()
    elif "```" in candidate:
        start = candidate.find("```") + 3
        end = candidate.find("```", start)
        if end > start:
            candidate = candidate[start:end].strip()

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _handle_spec_review(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")
    # Get default review_type from consultation config (used when not provided or None)
    consultation_config = get_consultation_config()
    workflow_config = consultation_config.get_workflow_config("plan_review")
    default_review_type = workflow_config.default_review_type
    review_type = payload.get("review_type") or default_review_type

    if not isinstance(spec_id, str) or not spec_id.strip():
        return asdict(
            error_response(
                "spec_id is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a valid spec_id",
            )
        )

    if review_type not in REVIEW_TYPES:
        return asdict(
            error_response(
                f"Invalid review_type: {review_type}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {', '.join(REVIEW_TYPES)}",
            )
        )

    start_time = time.perf_counter()
    llm_status = _get_llm_status()

    path = payload.get("path")
    ai_provider = payload.get("ai_provider")
    model = payload.get("model")

    for field_name, field_value in [
        ("spec_id", spec_id),
        ("path", path),
        ("ai_provider", ai_provider),
        ("model", model),
    ]:
        if field_value and isinstance(field_value, str) and is_prompt_injection(field_value):
            return asdict(
                error_response(
                    f"Input validation failed for {field_name}",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Remove instruction-like patterns from input.",
                )
            )

    specs_dir = None
    if isinstance(path, str) and path.strip():
        candidate = Path(path)
        if candidate.is_dir():
            specs_dir = candidate
        elif candidate.is_file():
            specs_dir = candidate.parent
        else:
            return asdict(
                error_response(
                    f"Invalid path: {path}",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Provide an existing directory or spec file path.",
                )
            )
    else:
        specs_dir = config.specs_dir

    dry_run_value = payload.get("dry_run", False)
    if dry_run_value is not None and not isinstance(dry_run_value, bool):
        return asdict(
            error_response(
                "dry_run must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide dry_run=true|false",
                details={"field": "dry_run"},
            )
        )
    dry_run = dry_run_value if isinstance(dry_run_value, bool) else False

    if review_type == "quick":
        return _run_quick_review(
            spec_id=spec_id,
            specs_dir=specs_dir,
            dry_run=dry_run,
            llm_status=llm_status,
            start_time=start_time,
        )

    try:
        ai_timeout = float(payload.get("ai_timeout", DEFAULT_AI_TIMEOUT))
    except (TypeError, ValueError):
        return asdict(
            error_response(
                "ai_timeout must be a number",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide ai_timeout as a float (seconds).",
            )
        )

    if ai_timeout <= 0:
        return asdict(
            error_response(
                "ai_timeout must be greater than 0",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide ai_timeout as a positive number of seconds.",
            )
        )

    consultation_cache_value = payload.get("consultation_cache", True)
    if consultation_cache_value is not None and not isinstance(consultation_cache_value, bool):
        return asdict(
            error_response(
                "consultation_cache must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide consultation_cache=true|false",
                details={"field": "consultation_cache"},
            )
        )
    consultation_cache = consultation_cache_value if isinstance(consultation_cache_value, bool) else True

    return _run_ai_review(
        spec_id=spec_id,
        specs_dir=specs_dir,
        review_type=review_type,
        ai_provider=ai_provider,
        model=model,
        ai_timeout=ai_timeout,
        consultation_cache=consultation_cache,
        dry_run=dry_run,
        llm_status=llm_status,
        start_time=start_time,
    )


def _handle_list_tools(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    start_time = time.perf_counter()

    try:
        llm_status = _get_llm_status()

        provider_statuses = get_provider_statuses()
        tools_info = [
            {
                "name": provider_id,
                "available": is_available,
                "status": "available" if is_available else "unavailable",
                "reason": None,
                "checked_at": None,
            }
            for provider_id, is_available in provider_statuses.items()
        ]

        duration_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer("review.review_list_tools.duration_ms", duration_ms)

        return asdict(
            success_response(
                tools=tools_info,
                llm_status=llm_status,
                review_types=REVIEW_TYPES,
                available_count=sum(1 for tool in tools_info if tool.get("available")),
                total_count=len(tools_info),
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    except Exception as exc:
        logger.exception("Error listing review tools")
        return asdict(
            error_response(
                f"Error listing review tools: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
            )
        )


def _handle_list_plan_tools(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    start_time = time.perf_counter()

    try:
        llm_status = _get_llm_status()

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
                "description": "Feasibility and complexity assessment",
                "capabilities": ["complexity", "estimation", "risks"],
                "llm_required": True,
                "estimated_time": "30-60 seconds",
            },
        ]

        recommendations = [
            "Use 'quick-review' for a fast sanity check.",
            "Use 'full-review' before implementation for comprehensive feedback.",
            "Use 'security-review' for specs touching auth/data boundaries.",
            "Use 'feasibility-review' to validate scope/estimates.",
        ]

        duration_ms = (time.perf_counter() - start_time) * 1000
        _metrics.timer("review.review_list_plan_tools.duration_ms", duration_ms)

        return asdict(
            success_response(
                plan_tools=plan_tools,
                llm_status=llm_status,
                recommendations=recommendations,
                telemetry={"duration_ms": round(duration_ms, 2)},
            )
        )

    except Exception as exc:
        logger.exception("Error listing plan review tools")
        return asdict(
            error_response(
                f"Error listing plan review tools: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details",
            )
        )


def _handle_parse_feedback(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    spec_id = payload.get("spec_id")
    review_path = payload.get("review_path")
    output_path = payload.get("output_path")

    return asdict(
        error_response(
            "Review feedback parsing requires complex text/markdown parsing. "
            "Use the foundry:foundry-spec skill to apply review feedback.",
            error_code=ErrorCode.UNAVAILABLE,
            error_type=ErrorType.UNAVAILABLE,
            data={
                "spec_id": spec_id,
                "review_path": review_path,
                "output_path": output_path,
                "alternative": "foundry:foundry-spec skill",
                "feature_status": "requires_complex_parsing",
            },
            remediation="Use the foundry:foundry-spec skill for parsing support.",
        )
    )


def _format_fidelity_markdown(
    parsed: Dict[str, Any],
    spec_id: str,
    spec_title: str,
    scope: str,
    task_id: Optional[str] = None,
    phase_id: Optional[str] = None,
    provider_id: Optional[str] = None,
) -> str:
    """Format fidelity review JSON as human-readable markdown."""
    # Build scope detail
    scope_detail = scope
    if task_id:
        scope_detail += f" (task: {task_id})"
    elif phase_id:
        scope_detail += f" (phase: {phase_id})"

    lines = [
        f"# Fidelity Review: {spec_title}",
        "",
        f"**Spec ID:** {spec_id}",
        f"**Scope:** {scope_detail}",
        f"**Verdict:** {parsed.get('verdict', 'unknown')}",
        f"**Date:** {datetime.now().isoformat()}",
    ]
    if provider_id:
        lines.append(f"**Provider:** {provider_id}")
    lines.append("")

    # Summary section
    if parsed.get("summary"):
        lines.extend(["## Summary", "", parsed["summary"], ""])

    # Requirement Alignment
    req_align = parsed.get("requirement_alignment", {})
    if req_align:
        lines.extend(
            [
                "## Requirement Alignment",
                f"**Status:** {req_align.get('answer', 'unknown')}",
                "",
                req_align.get("details", ""),
                "",
            ]
        )

    # Success Criteria
    success = parsed.get("success_criteria", {})
    if success:
        lines.extend(
            [
                "## Success Criteria",
                f"**Status:** {success.get('met', 'unknown')}",
                "",
                success.get("details", ""),
                "",
            ]
        )

    # Deviations
    deviations = parsed.get("deviations", [])
    if deviations:
        lines.extend(["## Deviations", ""])
        for dev in deviations:
            severity = dev.get("severity", "unknown")
            description = dev.get("description", "")
            justification = dev.get("justification", "")
            lines.append(f"- **[{severity.upper()}]** {description}")
            if justification:
                lines.append(f"  - Justification: {justification}")
        lines.append("")

    # Test Coverage
    test_cov = parsed.get("test_coverage", {})
    if test_cov:
        lines.extend(
            [
                "## Test Coverage",
                f"**Status:** {test_cov.get('status', 'unknown')}",
                "",
                test_cov.get("details", ""),
                "",
            ]
        )

    # Code Quality
    code_quality = parsed.get("code_quality", {})
    if code_quality:
        lines.extend(["## Code Quality", ""])
        if code_quality.get("details"):
            lines.append(code_quality["details"])
            lines.append("")
        for issue in code_quality.get("issues", []):
            lines.append(f"- {issue}")
        lines.append("")

    # Documentation
    doc = parsed.get("documentation", {})
    if doc:
        lines.extend(
            [
                "## Documentation",
                f"**Status:** {doc.get('status', 'unknown')}",
                "",
                doc.get("details", ""),
                "",
            ]
        )

    # Issues
    issues = parsed.get("issues", [])
    if issues:
        lines.extend(["## Issues", ""])
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    # Recommendations
    recommendations = parsed.get("recommendations", [])
    if recommendations:
        lines.extend(["## Recommendations", ""])
        for rec in recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    # Verdict consensus (if synthesized)
    verdict_consensus = parsed.get("verdict_consensus", {})
    if verdict_consensus:
        lines.extend(["## Verdict Consensus", ""])
        votes = verdict_consensus.get("votes", {})
        for verdict_type, models in votes.items():
            if models:
                lines.append(f"- **{verdict_type}:** {', '.join(models)}")
        agreement = verdict_consensus.get("agreement_level", "")
        if agreement:
            lines.append(f"\n**Agreement Level:** {agreement}")
        notes = verdict_consensus.get("notes", "")
        if notes:
            lines.extend(["", notes])
        lines.append("")

    # Synthesis metadata
    synth_meta = parsed.get("synthesis_metadata", {})
    if synth_meta:
        lines.extend(["## Synthesis Metadata", ""])
        if synth_meta.get("models_consulted"):
            lines.append(f"- Models consulted: {', '.join(synth_meta['models_consulted'])}")
        if synth_meta.get("models_succeeded"):
            lines.append(f"- Models succeeded: {', '.join(synth_meta['models_succeeded'])}")
        if synth_meta.get("synthesis_provider"):
            lines.append(f"- Synthesis provider: {synth_meta['synthesis_provider']}")
        lines.append("")

    lines.extend(
        [
            "---",
            "*Generated by Foundry MCP Fidelity Review*",
        ]
    )

    return "\n".join(lines)


def _try_tiebreaker_review(
    *,
    orchestrator: ConsultationOrchestrator,
    successful_responses: List[ProviderResponse],
    all_responses: List[ProviderResponse],
    request: ConsultationRequest,
    failed_providers: List[Dict[str, Any]],
) -> tuple:
    """Detect split verdicts and invoke an unused provider as tiebreaker.

    When two reviewers disagree on the verdict, this attempts to bring in
    a third reviewer to break the tie before synthesis.

    Args:
        orchestrator: The consultation orchestrator for provider calls.
        successful_responses: Responses with success=True and non-empty content.
        all_responses: All responses from the initial consensus round (including
            failures). Used to determine which providers were already consulted,
            so that failed providers are also excluded — if a provider failed the
            initial round it is unlikely to succeed as a tiebreaker.
        request: The original consultation request (context is reused).
        failed_providers: Mutable list; tiebreaker failures are appended here.

    Returns:
        (successful_responses, successful_provider_ids) — the responses list may
        have one additional entry if a tiebreaker succeeded.
    """
    if len(successful_responses) < 2:
        return successful_responses, [r.provider_id for r in successful_responses]

    individual_verdicts: List[tuple] = []
    unparseable_providers: List[str] = []
    for resp in successful_responses:
        resp_parsed = _parse_json_content(resp.content)
        if resp_parsed and "verdict" in resp_parsed:
            individual_verdicts.append((resp.provider_id, resp_parsed["verdict"]))
        else:
            unparseable_providers.append(resp.provider_id)

    if unparseable_providers:
        logger.warning(
            "Could not parse verdict from providers: %s — excluded from split detection",
            unparseable_providers,
        )

    verdict_values = [v for _, v in individual_verdicts]
    verdicts_split = len(individual_verdicts) >= 2 and len(set(verdict_values)) > 1

    if not verdicts_split:
        return successful_responses, [r.provider_id for r in successful_responses]

    # If one verdict already holds a strict majority, synthesis can resolve via
    # majority vote — no need to spend time and cost on an extra reviewer.
    verdict_counts = Counter(verdict_values)
    majority_threshold = len(individual_verdicts) / 2
    has_majority = any(count > majority_threshold for count in verdict_counts.values())
    if has_majority:
        logger.info(
            "Split verdict detected (%s) but majority exists — skipping tiebreaker, synthesis will use majority vote",
            verdict_values,
        )
        return successful_responses, [r.provider_id for r in successful_responses]

    # Exclude all providers from the initial round (both successful and failed)
    # so we get a genuinely fresh perspective for the tiebreaker.
    used_provider_ids = {r.provider_id for r in all_responses}
    all_available = orchestrator.get_available_providers()
    tiebreaker_candidates = [pid for pid in all_available if pid not in used_provider_ids]

    if not tiebreaker_candidates:
        logger.info(
            "Split verdict detected (%s) but no unused providers available "
            "for tiebreaker — falling back to severity-based synthesis resolution",
            verdict_values,
        )
        return successful_responses, [r.provider_id for r in successful_responses]

    logger.info(
        "Split verdict detected (%s), trying tiebreaker from %d candidates",
        verdict_values,
        len(tiebreaker_candidates),
    )

    tiebreaker_response: Optional[ProviderResponse] = None
    for tiebreaker_pid in tiebreaker_candidates:
        logger.info("Attempting tiebreaker with provider: %s", tiebreaker_pid)

        tiebreaker_request = ConsultationRequest(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            prompt_id="FIDELITY_REVIEW_V1",
            context=request.context,
            provider_id=tiebreaker_pid,
        )

        try:
            # Bypass cache to ensure a fresh, independent evaluation
            tiebreaker_outcome = orchestrator.consult(tiebreaker_request, use_cache=False)
        except Exception as e:
            logger.warning("Tiebreaker provider %s failed: %s", tiebreaker_pid, e)
            failed_providers.append({"provider_id": tiebreaker_pid, "error": str(e)})
            continue

        if isinstance(tiebreaker_outcome, ConsultationResult):
            if tiebreaker_outcome.success and tiebreaker_outcome.content.strip():
                tiebreaker_response = ProviderResponse.from_result(tiebreaker_outcome)
        elif isinstance(tiebreaker_outcome, ConsensusResult):
            for cr in tiebreaker_outcome.responses:
                if cr.success and cr.content.strip():
                    tiebreaker_response = cr
                    break

        if tiebreaker_response is not None:
            successful_responses.append(tiebreaker_response)
            logger.info(
                "Tiebreaker provider %s succeeded, synthesis will use %d reviews",
                tiebreaker_response.provider_id,
                len(successful_responses),
            )
            break

        failed_providers.append({"provider_id": tiebreaker_pid, "error": "no usable content"})
        logger.warning(
            "Tiebreaker provider %s returned no usable content, trying next candidate",
            tiebreaker_pid,
        )

    if tiebreaker_response is None:
        logger.warning(
            "All %d tiebreaker candidates exhausted, proceeding with %d-reviewer synthesis",
            len(tiebreaker_candidates),
            len(successful_responses),
        )

    return successful_responses, [r.provider_id for r in successful_responses]


def _handle_fidelity(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Best-effort fidelity review.

    Note: the canonical `spec-review-fidelity` tool remains the source of truth
    for fidelity review behavior; this action is primarily to support the
    consolidated manifest.
    """

    start_time = time.perf_counter()
    spec_id = payload.get("spec_id")
    task_id = payload.get("task_id")
    phase_id = payload.get("phase_id")
    files = payload.get("files")
    ai_tools = payload.get("ai_tools")
    model = payload.get("model")
    consensus_threshold = payload.get("consensus_threshold", 2)
    incremental_value = payload.get("incremental", False)
    if incremental_value is not None and not isinstance(incremental_value, bool):
        return asdict(
            error_response(
                "incremental must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide incremental=true|false",
                details={"field": "incremental"},
            )
        )
    incremental = incremental_value if isinstance(incremental_value, bool) else False

    include_tests_value = payload.get("include_tests", True)
    if include_tests_value is not None and not isinstance(include_tests_value, bool):
        return asdict(
            error_response(
                "include_tests must be a boolean",
                error_code=ErrorCode.INVALID_FORMAT,
                error_type=ErrorType.VALIDATION,
                remediation="Provide include_tests=true|false",
                details={"field": "include_tests"},
            )
        )
    include_tests = include_tests_value if isinstance(include_tests_value, bool) else True
    base_branch = payload.get("base_branch", "main")
    workspace = payload.get("workspace")

    if not isinstance(spec_id, str) or not spec_id:
        return asdict(
            error_response(
                "Specification ID is required",
                error_code=ErrorCode.MISSING_REQUIRED,
                error_type=ErrorType.VALIDATION,
                remediation="Provide a valid spec_id to review.",
            )
        )

    if task_id and phase_id:
        return asdict(
            error_response(
                "Cannot specify both task_id and phase_id",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Provide either task_id OR phase_id, not both.",
            )
        )

    if not isinstance(consensus_threshold, int) or consensus_threshold < 1 or consensus_threshold > 5:
        return asdict(
            error_response(
                f"Invalid consensus_threshold: {consensus_threshold}. Must be between 1 and 5.",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Use a consensus_threshold between 1 and 5.",
            )
        )

    for field_name, field_value in [
        ("spec_id", spec_id),
        ("task_id", task_id),
        ("phase_id", phase_id),
        ("model", model),
        ("base_branch", base_branch),
        ("workspace", workspace),
    ]:
        if field_value and isinstance(field_value, str) and is_prompt_injection(field_value):
            return asdict(
                error_response(
                    f"Input validation failed for {field_name}",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Remove instruction-like patterns from input.",
                )
            )

    if files:
        for idx, file_path in enumerate(files):
            if isinstance(file_path, str) and is_prompt_injection(file_path):
                return asdict(
                    error_response(
                        f"Input validation failed for files[{idx}]",
                        error_code=ErrorCode.VALIDATION_ERROR,
                        error_type=ErrorType.VALIDATION,
                        remediation="Remove instruction-like patterns from file paths.",
                    )
                )

    ws_path = Path(workspace) if isinstance(workspace, str) and workspace else Path.cwd()
    specs_dir = find_specs_directory(str(ws_path))
    if not specs_dir:
        return asdict(
            error_response(
                "Could not find specs directory",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Ensure you're in a project with a specs/ directory",
            )
        )

    spec_file = find_spec_file(spec_id, specs_dir)
    if not spec_file:
        return asdict(
            error_response(
                f"Specification not found: {spec_id}",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Verify the spec ID exists using spec(action="list").',
            )
        )

    spec_data = load_spec(spec_id, specs_dir)
    if not spec_data:
        return asdict(
            error_response(
                f"Failed to load specification: {spec_id}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check spec JSON validity and retry.",
            )
        )

    scope = "task" if task_id else ("phase" if phase_id else "spec")

    # Setup fidelity reviews directory and file naming
    fidelity_reviews_dir = Path(specs_dir) / ".fidelity-reviews"
    base_name = f"{spec_id}-{scope}"
    if task_id:
        base_name += f"-{task_id}"
    elif phase_id:
        base_name += f"-{phase_id}"
    provider_review_paths: List[Dict[str, Any]] = []
    review_path: Optional[str] = None

    spec_requirements = _build_spec_requirements(spec_data, task_id, phase_id, exclude_fidelity_verify=True)
    spec_overview = _build_spec_overview(spec_data)
    implementation_artifacts = _build_implementation_artifacts(
        spec_data,
        task_id,
        phase_id,
        files,
        incremental,
        base_branch,
        workspace_root=ws_path,
        exclude_fidelity_verify=True,
    )
    test_results = (
        _build_test_results(spec_data, task_id, phase_id, exclude_fidelity_verify=True) if include_tests else ""
    )
    journal_entries = _build_journal_entries(spec_data, task_id, phase_id, exclude_fidelity_verify=True)
    subsequent_phases = _build_subsequent_phases(spec_data, phase_id)

    preferred_providers = ai_tools if isinstance(ai_tools, list) else []
    first_provider = preferred_providers[0] if preferred_providers else None

    # Load consultation config from workspace path to get provider priority list
    config_file = ws_path / "foundry-mcp.toml"
    consultation_config = load_consultation_config(config_file=config_file)
    orchestrator = ConsultationOrchestrator(config=consultation_config)
    if not orchestrator.is_available(provider_id=first_provider):
        return asdict(
            error_response(
                "Fidelity review requested but no providers available",
                error_code=ErrorCode.AI_NO_PROVIDER,
                error_type=ErrorType.UNAVAILABLE,
                data={"spec_id": spec_id, "requested_provider": first_provider},
                remediation="Install/configure an AI provider (claude/gemini/codex)",
            )
        )

    request = ConsultationRequest(
        workflow=ConsultationWorkflow.FIDELITY_REVIEW,
        prompt_id="FIDELITY_REVIEW_V1",
        context={
            "spec_id": spec_id,
            "spec_title": spec_data.get("title", spec_id),
            "spec_description": spec_data.get("description", ""),
            "review_scope": scope,
            "spec_overview": spec_overview,
            "spec_requirements": spec_requirements,
            "implementation_artifacts": implementation_artifacts,
            "test_results": test_results,
            "journal_entries": journal_entries,
            "subsequent_phases": subsequent_phases,
        },
        provider_id=first_provider,
        model=model,
    )

    result = orchestrator.consult(request, use_cache=True)
    is_consensus = isinstance(result, ConsensusResult)
    synthesis_performed = False
    synthesis_error = None
    successful_providers: List[str] = []
    failed_providers: List[Dict[str, Any]] = []

    if is_consensus:
        # Extract provider details for visibility
        failed_providers = [{"provider_id": r.provider_id, "error": r.error} for r in result.responses if not r.success]
        # Filter for truly successful responses (success=True AND non-empty content)
        successful_responses = [r for r in result.responses if r.success and r.content.strip()]
        successful_providers = [r.provider_id for r in successful_responses]

        # Try to resolve split verdicts with a tiebreaker reviewer
        successful_responses, successful_providers = _try_tiebreaker_review(
            orchestrator=orchestrator,
            successful_responses=successful_responses,
            all_responses=result.responses,
            request=request,
            failed_providers=failed_providers,
        )

        if len(successful_responses) >= 2:
            # Multi-model mode: run synthesis to consolidate reviews
            model_reviews_json = ""
            for response in successful_responses:
                model_reviews_json += (
                    f"\n---\n## Review by {response.provider_id}\n\n```json\n{response.content}\n```\n"
                )

            # Write individual provider review files
            try:
                fidelity_reviews_dir.mkdir(parents=True, exist_ok=True)
                for response in successful_responses:
                    provider_parsed = _parse_json_content(response.content)
                    provider_file = fidelity_reviews_dir / f"{base_name}-{response.provider_id}.md"
                    if provider_parsed:
                        provider_md = _format_fidelity_markdown(
                            provider_parsed,
                            spec_id,
                            spec_data.get("title", spec_id),
                            scope,
                            task_id=task_id,
                            phase_id=phase_id,
                            provider_id=response.provider_id,
                        )
                        provider_file.write_text(provider_md, encoding="utf-8")
                        provider_review_paths.append(
                            {
                                "provider_id": response.provider_id,
                                "path": str(provider_file),
                            }
                        )
                    else:
                        # JSON parsing failed - write raw content as fallback
                        logger.warning(
                            "Provider %s returned non-JSON content, writing raw response",
                            response.provider_id,
                        )
                        raw_md = (
                            f"# Fidelity Review (Raw): {spec_id}\n\n"
                            f"**Provider:** {response.provider_id}\n"
                            f"**Note:** Response could not be parsed as JSON\n\n"
                            f"## Raw Response\n\n```\n{response.content}\n```\n"
                        )
                        provider_file.write_text(raw_md, encoding="utf-8")
                        provider_review_paths.append(
                            {
                                "provider_id": response.provider_id,
                                "path": str(provider_file),
                                "parse_error": True,
                            }
                        )
            except Exception as e:
                logger.warning("Failed to write provider review files: %s", e)

            logger.info(
                "Running fidelity synthesis for %d provider reviews: %s",
                len(successful_responses),
                successful_providers,
            )

            synthesis_request = ConsultationRequest(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                prompt_id="FIDELITY_SYNTHESIS_PROMPT_V1",
                context={
                    "spec_id": spec_id,
                    "spec_title": spec_data.get("title", spec_id),
                    "review_scope": scope,
                    "num_models": len(successful_responses),
                    "model_reviews": model_reviews_json,
                    "response_schema": FIDELITY_SYNTHESIZED_RESPONSE_SCHEMA,
                },
                provider_id=successful_providers[0],
                model=model,
            )

            try:
                synthesis_result = orchestrator.consult(synthesis_request, use_cache=True)
            except Exception as e:
                logger.error("Fidelity synthesis call crashed: %s", e, exc_info=True)
                synthesis_result = None

            # Handle both ConsultationResult and ConsensusResult from synthesis
            synthesis_success = False
            synthesis_content = None
            if synthesis_result:
                if isinstance(synthesis_result, ConsultationResult) and synthesis_result.success:
                    synthesis_content = synthesis_result.content
                    synthesis_success = bool(synthesis_content and synthesis_content.strip())
                elif isinstance(synthesis_result, ConsensusResult) and synthesis_result.success:
                    synthesis_content = synthesis_result.primary_content
                    synthesis_success = bool(synthesis_content and synthesis_content.strip())

            if synthesis_success and synthesis_content:
                content = synthesis_content
                synthesis_performed = True
            else:
                # Synthesis failed - fall back to first provider's content
                error_detail = "unknown"
                if synthesis_result is None:
                    error_detail = "synthesis crashed (see logs)"
                elif isinstance(synthesis_result, ConsultationResult):
                    error_detail = synthesis_result.error or "empty response"
                elif isinstance(synthesis_result, ConsensusResult):
                    error_detail = "empty synthesis content"
                logger.warning(
                    "Fidelity synthesis call failed (%s), falling back to first provider's content",
                    error_detail,
                )
                content = result.primary_content
                synthesis_error = error_detail
        else:
            # Single successful provider - use its content directly (no synthesis needed)
            content = result.primary_content
    else:
        content = result.content

    parsed = _parse_json_content(content)
    verdict = parsed.get("verdict") if parsed else "unknown"

    # Write main fidelity review file
    if parsed:
        try:
            fidelity_reviews_dir.mkdir(parents=True, exist_ok=True)
            main_md = _format_fidelity_markdown(
                parsed,
                spec_id,
                spec_data.get("title", spec_id),
                scope,
                task_id=task_id,
                phase_id=phase_id,
            )
            review_file = fidelity_reviews_dir / f"{base_name}.md"
            review_file.write_text(main_md, encoding="utf-8")
            review_path = str(review_file)
        except Exception as e:
            logger.warning("Failed to write main fidelity review file: %s", e)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Build consensus info with synthesis details
    consensus_info: Dict[str, Any] = {
        "mode": "multi_model" if is_consensus else "single_model",
        "threshold": consensus_threshold,
        "provider_id": getattr(result, "provider_id", None),
        "model_used": getattr(result, "model_used", None),
        "synthesis_performed": synthesis_performed,
    }

    if is_consensus:
        consensus_info["successful_providers"] = successful_providers
        consensus_info["failed_providers"] = failed_providers
        if synthesis_error:
            consensus_info["synthesis_error"] = synthesis_error

    # Include additional synthesized fields if available
    response_data: Dict[str, Any] = {
        "spec_id": spec_id,
        "title": spec_data.get("title", spec_id),
        "scope": scope,
        "verdict": verdict,
        "deviations": parsed.get("deviations") if parsed else [],
        "recommendations": parsed.get("recommendations") if parsed else [],
        "consensus": consensus_info,
    }

    # Add file paths if reviews were written
    if review_path:
        response_data["review_path"] = review_path
    if provider_review_paths:
        response_data["provider_reviews"] = provider_review_paths

    # Add synthesis-specific fields if synthesis was performed
    if synthesis_performed and parsed:
        if "verdict_consensus" in parsed:
            response_data["verdict_consensus"] = parsed["verdict_consensus"]
        if "synthesis_metadata" in parsed:
            response_data["synthesis_metadata"] = parsed["synthesis_metadata"]

    return asdict(
        success_response(
            data=response_data,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


def _handle_fidelity_gate(*, config: ServerConfig, payload: Dict[str, Any]) -> dict:
    """Run a phase fidelity review for autonomous gate enforcement.

    This action is designed for the autonomous execution system. It runs a
    fidelity review for a phase and writes gate evidence to the session state.
    The orchestrator (via session-step command="next") validates and applies
    gate policy based on this evidence.

    Request fields:
    - spec_id (required): Spec ID
    - session_id (required): Session ID
    - phase_id (required): Phase ID to review
    - step_id (required): Step ID from the run_fidelity_gate next_step
    - Existing fidelity inputs (ai_tools, model, consensus_threshold, etc.)

    Response data:
    - spec_id, session_id, phase_id, step_id
    - gate_attempt_id: Unique ID for this review attempt
    - verdict: pass | fail | warn
    - gate_policy: Echo of active session policy
    - gate_passed_preview: bool preview (command="next" recomputes authoritatively)
    - review_path: Path to review file (if written)
    - findings: Summary of issues (if any)
    """
    from datetime import datetime, timezone

    from ulid import ULID

    from foundry_mcp.core.autonomy.memory import AutonomyStorage
    from foundry_mcp.core.autonomy.models.enums import GatePolicy, GateVerdict
    from foundry_mcp.core.autonomy.models.gates import PendingGateEvidence
    from foundry_mcp.core.autonomy.server_secret import compute_integrity_checksum

    start_time = time.perf_counter()

    # Required parameters
    spec_id = payload.get("spec_id")
    session_id = payload.get("session_id")
    phase_id = payload.get("phase_id")
    step_id = payload.get("step_id")

    # Validate required parameters
    _fidelity_gate_field_hints = {
        "spec_id": "Provide the spec_id from the session.",
        "session_id": "Provide the session_id.",
        "phase_id": "Use next_step.phase_id from the orchestrator step.",
        "step_id": "Use next_step.step_id from the orchestrator step.",
    }
    for field_name, field_value in [
        ("spec_id", spec_id),
        ("session_id", session_id),
        ("phase_id", phase_id),
        ("step_id", step_id),
    ]:
        if not isinstance(field_value, str) or not field_value.strip():
            return asdict(
                error_response(
                    f"{field_name} is required for fidelity-gate action",
                    error_code=ErrorCode.MISSING_REQUIRED,
                    error_type=ErrorType.VALIDATION,
                    remediation=_fidelity_gate_field_hints[field_name],
                )
            )

    # Security validation for string inputs
    for field_name, field_value in [
        ("spec_id", spec_id),
        ("session_id", session_id),
        ("phase_id", phase_id),
        ("step_id", step_id),
    ]:
        if isinstance(field_value, str) and is_prompt_injection(field_value):
            return asdict(
                error_response(
                    f"Input validation failed for {field_name}",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    remediation="Remove instruction-like patterns from input.",
                )
            )

    workspace = payload.get("workspace")
    ws_path = Path(workspace) if isinstance(workspace, str) and workspace else Path.cwd()

    # Load session to get gate_policy
    storage = AutonomyStorage(workspace_path=ws_path)
    session = storage.load(str(session_id))

    if session is None:
        return asdict(
            error_response(
                f"Session not found: {session_id}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the session ID exists.",
            )
        )

    if session.spec_id != spec_id:
        return asdict(
            error_response(
                f"Session {session_id} is for spec {session.spec_id}, not {spec_id}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation="Ensure session_id matches the spec_id.",
            )
        )

    # Build fidelity review payload (reuse existing handler)
    fidelity_payload = {
        "spec_id": spec_id,
        "phase_id": phase_id,
        "ai_tools": payload.get("ai_tools"),
        "model": payload.get("model"),
        "consensus_threshold": payload.get("consensus_threshold", 2),
        "include_tests": payload.get("include_tests", True),
        "workspace": workspace,
        "files": payload.get("files"),
        "incremental": payload.get("incremental", False),
        "base_branch": payload.get("base_branch", "main"),
    }

    # Run the fidelity review
    fidelity_result = _handle_fidelity(config=config, payload=fidelity_payload)

    # Extract verdict from fidelity result
    if fidelity_result.get("success"):
        verdict_str = fidelity_result.get("data", {}).get("verdict", "unknown")
        review_path = fidelity_result.get("data", {}).get("review_path")
        deviations = fidelity_result.get("data", {}).get("deviations", [])
    else:
        # Fidelity review failed - treat as fail verdict
        verdict_str = "fail"
        review_path = None
        deviations = [{"error": fidelity_result.get("error", "Unknown error")}]

    # Normalize verdict to enum value
    verdict_map = {
        "pass": GateVerdict.PASS,
        "warn": GateVerdict.WARN,
        "fail": GateVerdict.FAIL,
    }
    verdict = verdict_map.get(verdict_str.lower(), GateVerdict.FAIL)

    # Generate gate_attempt_id
    gate_attempt_id = f"gate_{ULID()}"

    # Compute gate_passed_preview based on policy and verdict
    gate_policy = session.gate_policy
    if gate_policy == GatePolicy.STRICT:
        gate_passed_preview = verdict == GateVerdict.PASS
    elif gate_policy == GatePolicy.LENIENT:
        gate_passed_preview = verdict in (GateVerdict.PASS, GateVerdict.WARN)
    else:  # MANUAL
        gate_passed_preview = False  # Manual always requires human review

    # Write pending gate evidence to session
    now = datetime.now(timezone.utc)

    # Compute integrity checksum for tamper detection (P1.3)
    integrity_checksum = compute_integrity_checksum(
        gate_attempt_id=gate_attempt_id,
        step_id=str(step_id),
        phase_id=str(phase_id),
        verdict=verdict.value,
    )

    session.pending_gate_evidence = PendingGateEvidence(
        gate_attempt_id=gate_attempt_id,
        step_id=str(step_id),  # Validated non-None above
        phase_id=str(phase_id),  # Validated non-None above
        verdict=verdict,
        issued_at=now,
        integrity_checksum=integrity_checksum,
    )
    session.updated_at = now
    session.state_version += 1

    # Save session
    storage.save(session)

    duration_ms = (time.perf_counter() - start_time) * 1000

    # Build findings from deviations
    findings = []
    if deviations:
        for dev in deviations[:10]:  # Cap at 10 for response size
            if isinstance(dev, dict):
                findings.append(
                    {
                        "type": dev.get("type", "unknown"),
                        "description": dev.get("description", str(dev)),
                    }
                )
            else:
                findings.append({"type": "issue", "description": str(dev)})

    response_data = {
        "spec_id": spec_id,
        "session_id": session_id,
        "phase_id": phase_id,
        "step_id": step_id,
        "gate_attempt_id": gate_attempt_id,
        "verdict": verdict.value,
        "gate_policy": gate_policy.value,
        "gate_passed_preview": gate_passed_preview,
        "findings": findings,
    }

    if review_path:
        response_data["review_path"] = review_path

    return asdict(
        success_response(
            data=response_data,
            telemetry={"duration_ms": round(duration_ms, 2)},
        )
    )


_ACTIONS = [
    ActionDefinition(name="spec", handler=_handle_spec_review, summary="Review a spec", aliases=("spec-review",)),
    ActionDefinition(
        name="fidelity",
        handler=_handle_fidelity,
        summary="Run a fidelity review",
    ),
    ActionDefinition(
        name="fidelity-gate",
        handler=_handle_fidelity_gate,
        summary="Run a fidelity gate review for autonomous execution",
    ),
    ActionDefinition(
        name="parse-feedback",
        handler=_handle_parse_feedback,
        summary="Parse reviewer feedback into structured issues",
    ),
    ActionDefinition(
        name="list-tools",
        handler=_handle_list_tools,
        summary="List available review tools",
    ),
    ActionDefinition(
        name="list-plan-tools",
        handler=_handle_list_plan_tools,
        summary="List available plan review toolchains",
    ),
]

_REVIEW_ROUTER = ActionRouter(tool_name="review", actions=_ACTIONS)


def _dispatch_review_action(*, action: str, payload: Dict[str, Any], config: ServerConfig) -> dict:
    return dispatch_with_standard_errors(_REVIEW_ROUTER, "review", action, payload=payload, config=config)


def register_unified_review_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated review tool."""

    @canonical_tool(mcp, canonical_name="review")
    @mcp_tool(tool_name="review", emit_metrics=True, audit=True)
    def review(
        action: str,
        spec_id: Optional[str] = None,
        review_type: Optional[str] = None,
        tools: Optional[str] = None,
        model: Optional[str] = None,
        ai_provider: Optional[str] = None,
        ai_timeout: float = DEFAULT_AI_TIMEOUT,
        consultation_cache: bool = True,
        path: Optional[str] = None,
        dry_run: bool = False,
        task_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        ai_tools: Optional[List[str]] = None,
        consensus_threshold: int = 2,
        incremental: bool = False,
        include_tests: bool = True,
        base_branch: str = "main",
        workspace: Optional[str] = None,
        review_path: Optional[str] = None,
        output_path: Optional[str] = None,
        session_id: Optional[str] = None,
        step_id: Optional[str] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "review_type": review_type,
            "tools": tools,
            "model": model,
            "ai_provider": ai_provider,
            "ai_timeout": ai_timeout,
            "consultation_cache": consultation_cache,
            "path": path,
            "dry_run": dry_run,
            "task_id": task_id,
            "phase_id": phase_id,
            "files": files,
            "ai_tools": ai_tools,
            "consensus_threshold": consensus_threshold,
            "incremental": incremental,
            "include_tests": include_tests,
            "base_branch": base_branch,
            "workspace": workspace,
            "review_path": review_path,
            "output_path": output_path,
            "session_id": session_id,
            "step_id": step_id,
        }
        return _dispatch_review_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified review tool")


__all__ = [
    "register_unified_review_tool",
]
