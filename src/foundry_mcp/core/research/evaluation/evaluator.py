"""Core evaluation logic — LLM-as-judge for research quality.

Builds a structured evaluation prompt with per-dimension rubrics, calls
an LLM to score the report, parses the JSON response, and produces an
:class:`EvaluationResult`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.evaluation.dimensions import DIMENSIONS, Dimension
from foundry_mcp.core.research.evaluation.scoring import (
    EvaluationResult,
    build_dimension_score,
    compute_composite,
)
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    extract_json,
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    LLMCallResult,
    execute_llm_call,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a research quality evaluator. Your task is to objectively assess the
quality of a research report across multiple dimensions using the rubrics
provided. Be critical and precise — differentiate between good and great
work. Score each dimension independently.

You MUST respond with ONLY a valid JSON object, no extra text."""

_MAX_REPORT_CHARS = 80_000  # Truncate very long reports to fit context
_MAX_SOURCES_IN_PROMPT = 30  # Limit source list size
_MAX_RAW_NOTES_CHARS = 30_000  # Cap raw notes context for groundedness


def _build_evaluation_prompt(
    query: str,
    report: str,
    sources: list[dict[str, Any]],
    dimensions: tuple[Dimension, ...] = DIMENSIONS,
    *,
    raw_notes: Optional[list[str]] = None,
) -> str:
    """Build the user prompt for LLM evaluation.

    Args:
        query: Original research query.
        report: Final research report text.
        sources: List of source dicts with title, url, quality fields.
        dimensions: Evaluation dimensions to score.
        raw_notes: Optional raw notes from topic researchers, used as
            ground-truth context for the groundedness dimension.
            Matches ODR's ``eval_groundedness`` pattern.

    Returns:
        Formatted evaluation prompt.
    """
    # Truncate report if needed
    display_report = report
    if len(report) > _MAX_REPORT_CHARS:
        display_report = report[:_MAX_REPORT_CHARS] + "\n\n[... report truncated for evaluation ...]"

    # Format source list (sanitize web-derived titles)
    source_lines = []
    for i, src in enumerate(sources[:_MAX_SOURCES_IN_PROMPT], 1):
        title = sanitize_external_content(src.get("title", "Untitled"))
        url = src.get("url", "")
        quality = src.get("quality", "")
        line = f"  {i}. {title}"
        if url:
            line += f" ({url})"
        if quality:
            line += f" [quality: {quality}]"
        source_lines.append(line)
    sources_text = "\n".join(source_lines) if source_lines else "  (no sources listed)"

    # Phase 3c (ODR alignment): Include raw notes as ground-truth context
    # for the groundedness dimension.  This matches ODR's eval_groundedness
    # which uses raw researcher output as the evidence baseline.
    raw_notes_section = ""
    if raw_notes:
        raw_notes_text = "\n---\n".join(
            sanitize_external_content(note) for note in raw_notes
        )
        if len(raw_notes_text) > _MAX_RAW_NOTES_CHARS:
            raw_notes_text = raw_notes_text[:_MAX_RAW_NOTES_CHARS] + "\n\n[... notes truncated ...]"
        raw_notes_section = f"""

## Raw Research Evidence (ground truth for groundedness evaluation)
The following are uncompressed research notes gathered by topic researchers.
Use these as the ground-truth evidence base when scoring the **Groundedness**
dimension — assess whether the report's claims are supported by this evidence.

{raw_notes_text}
"""

    # Format rubrics
    rubric_blocks = []
    for dim in dimensions:
        rubric_blocks.append(
            f"### {dim.display_name} ({dim.name})\n"
            f"{dim.description}\n\n"
            f"Scoring rubric:\n{dim.rubric}"
        )
    rubrics_text = "\n\n".join(rubric_blocks)

    # Build dimension names list for JSON schema
    dim_names = ", ".join(f'"{d.name}"' for d in dimensions)

    return f"""\
Evaluate the following research report.

## Research Query
{query}

## Sources Used
{sources_text}
{raw_notes_section}
## Research Report
{display_report}

---

## Evaluation Dimensions

{rubrics_text}

---

## Instructions

Score each dimension on a 1-5 scale using the rubrics above. For each
dimension, provide:
- A score (integer 1-5)
- A brief rationale (1-2 sentences explaining the score)

Respond with a JSON object in this exact format:
{{
  "scores": {{
    "<dimension_name>": {{
      "score": <1-5>,
      "rationale": "<explanation>"
    }}
  }}
}}

The dimension names are: {dim_names}.

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_evaluation_response(
    content: str,
    dimensions: tuple[Dimension, ...] = DIMENSIONS,
) -> EvaluationResult:
    """Parse LLM evaluation response into an EvaluationResult.

    Args:
        content: Raw LLM response text.
        dimensions: Expected dimensions to extract scores for.

    Returns:
        EvaluationResult with parsed scores and composite.

    Raises:
        ValueError: If response cannot be parsed or is missing required fields.
    """
    json_str = extract_json(content)
    if json_str is None:
        raise ValueError("No JSON object found in evaluation response")

    data = json.loads(json_str)

    scores_data = data.get("scores")
    if not isinstance(scores_data, dict):
        raise ValueError("Expected 'scores' object in evaluation response")

    dimension_scores = []
    for dim in dimensions:
        dim_data = scores_data.get(dim.name)
        if dim_data is None:
            # Dimension missing — assign neutral imputed score
            dimension_scores.append(
                build_dimension_score(dim.name, 3, "Not evaluated", imputed=True)
            )
            continue

        raw_score = dim_data.get("score", 3)
        if not isinstance(raw_score, (int, float)):
            raw_score = 3
        rationale = str(dim_data.get("rationale", ""))

        dimension_scores.append(build_dimension_score(dim.name, int(raw_score), rationale))

    # Compute composite
    composite, variance, weights = compute_composite(dimension_scores)

    # Track imputation in metadata
    imputed_count = sum(1 for ds in dimension_scores if ds.imputed)
    metadata: dict[str, object] = {}
    if imputed_count > 0:
        imputed_names = [ds.name for ds in dimension_scores if ds.imputed]
        metadata["imputed_count"] = imputed_count
        metadata["warnings"] = [
            f"{imputed_count} dimension(s) imputed with neutral score (3/5): {', '.join(imputed_names)}"
        ]

    return EvaluationResult(
        dimension_scores=dimension_scores,
        composite_score=composite,
        score_variance=variance,
        weights=weights,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def evaluate_report(
    workflow: Any,
    state: DeepResearchState,
    provider_id: Optional[str],
    model: Optional[str],
    timeout: float,
) -> EvaluationResult | WorkflowResult:
    """Evaluate a completed research report using LLM-as-judge.

    Builds an evaluation prompt from the report and sources, calls the
    LLM, parses dimension scores, computes the composite, and stores
    results in ``state.metadata["evaluation"]``.

    Args:
        workflow: DeepResearchWorkflow instance (provides config, memory,
            audit, provider execution).
        state: Research state with a completed report.
        provider_id: LLM provider for evaluation (resolved from config
            if ``None``).
        model: Model override (resolved from config if ``None``).
        timeout: Timeout for the evaluation LLM call.

    Returns:
        EvaluationResult on success, or WorkflowResult on LLM error.
    """
    if not state.report:
        return WorkflowResult(
            success=False,
            content="",
            error="Cannot evaluate: report not yet generated",
        )

    # Build source dicts for the prompt
    source_dicts = [
        {
            "title": s.title,
            "url": s.url or "",
            "quality": getattr(s, "quality", ""),
        }
        for s in state.sources
    ]

    user_prompt = _build_evaluation_prompt(
        query=state.original_query,
        report=state.report,
        sources=source_dicts,
        raw_notes=state.raw_notes if state.raw_notes else None,
    )

    # Audit: evaluation started
    workflow._write_audit_event(
        state,
        "evaluation.started",
        data={"task_id": state.id},
    )

    # Execute LLM call via shared lifecycle helper
    call_result = await execute_llm_call(
        workflow=workflow,
        state=state,
        phase_name="evaluation",
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        provider_id=provider_id,
        model=model,
        temperature=0.1,  # Low temperature for consistent scoring
        timeout=timeout,
        role="evaluation",
    )

    # LLM-level error
    if isinstance(call_result, WorkflowResult):
        workflow._write_audit_event(
            state,
            "evaluation.failed",
            data={"task_id": state.id, "error": call_result.error},
        )
        return call_result

    assert isinstance(call_result, LLMCallResult)
    content = call_result.result.content or ""

    # Parse the evaluation response
    try:
        eval_result = _parse_evaluation_response(content)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Evaluation response parsing failed: %s", exc)
        workflow._write_audit_event(
            state,
            "evaluation.failed",
            data={"task_id": state.id, "error": str(exc)},
        )
        return WorkflowResult(
            success=False,
            content="",
            error=f"Failed to parse evaluation response: {exc}",
            metadata={"raw_response": content[:2000]},
        )

    # Attach metadata
    eval_result.metadata = {
        "provider_id": call_result.result.provider_id,
        "model_used": call_result.result.model_used,
        "duration_ms": call_result.llm_call_duration_ms,
        "research_id": state.id,
    }

    # Store in state metadata
    state.metadata["evaluation"] = eval_result.to_dict()
    workflow.memory.save_deep_research(state)

    # Audit: evaluation completed
    workflow._write_audit_event(
        state,
        "evaluation.completed",
        data={
            "task_id": state.id,
            "composite_score": eval_result.composite_score,
            "dimension_count": len(eval_result.dimension_scores),
            "provider_id": call_result.result.provider_id,
            "duration_ms": call_result.llm_call_duration_ms,
        },
    )

    return eval_result
