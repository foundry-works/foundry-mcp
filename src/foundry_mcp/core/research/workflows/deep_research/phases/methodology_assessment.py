"""Methodology quality assessment for deep research.

Extracts study design, sample size, effect size, limitations, and biases
from research source content via LLM extraction.

**Experimental** — produces approximate heuristics, not validated instruments.
No numeric rigor score; provides structured metadata to the synthesis LLM
for qualitative judgment.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable, Optional

from foundry_mcp.core.research.models.sources import (
    MethodologyAssessment,
    ResearchSource,
    SourceType,
    StudyDesign,
)
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    sanitize_external_content,
)

logger = logging.getLogger(__name__)

# Type for standalone LLM call function: (system_prompt, user_prompt) -> content or None
LLMCallFn = Callable[[str, str], Awaitable[Optional[str]]]

# Minimum content length to attempt assessment
MIN_CONTENT_LENGTH = 200

# Valid study design values for LLM output validation
_VALID_STUDY_DESIGNS = {d.value for d in StudyDesign}

# Valid confidence levels
_VALID_CONFIDENCE = {"high", "medium", "low"}

# ---------------------------------------------------------------------------
# LLM Extraction Prompt
# ---------------------------------------------------------------------------

METHODOLOGY_EXTRACTION_SYSTEM_PROMPT = """\
You are a research methodology analyst. Extract structured methodology \
metadata from the provided research source content.

Return ONLY a JSON object with these fields:
- study_design: one of {study_designs}
- sample_size: integer or null if not reported
- sample_description: brief description of participants/sample or null
- effect_size: reported effect size as string (e.g. "d=0.45", "OR=2.3") or null
- statistical_significance: reported significance as string (e.g. "p<0.001") or null
- limitations_noted: list of limitations mentioned (empty list if none)
- potential_biases: list of potential biases identified (empty list if none)
- confidence: your confidence in the assessment — "high" (full text with clear methods section), "medium" (substantial content but methods unclear), or "low" (abstract only or limited content)

Rules:
- Use null for any field where information is not clearly stated
- Do NOT guess or infer values — only extract what is explicitly stated
- For study_design, use "unknown" if the design cannot be determined
- Keep sample_description under 50 words
- Keep each limitation and bias under 30 words
- Return valid JSON only, no markdown or explanation
""".format(study_designs=", ".join(sorted(_VALID_STUDY_DESIGNS)))


def _build_extraction_user_prompt(
    source_title: str,
    content: str,
    content_basis: str,
) -> str:
    """Build the user prompt for methodology extraction."""
    basis_note = (
        "Note: This content is from the paper abstract only. "
        "Information will be limited."
        if content_basis == "abstract"
        else "This content is from the full paper text."
    )

    return (
        f"Source: {sanitize_external_content(source_title)}\n"
        f"{basis_note}\n\n"
        f"Content:\n{sanitize_external_content(content[:8000])}"  # Cap content to avoid token overflow
    )


def _parse_llm_response(
    raw_response: str,
    source_id: str,
    content_basis: str,
) -> MethodologyAssessment:
    """Parse LLM JSON response into a MethodologyAssessment.

    Handles malformed JSON gracefully by returning an UNKNOWN assessment.
    Forces confidence to "low" for abstract-only content.
    """
    try:
        # Strip markdown fences if present
        text = raw_response.strip()
        if text.startswith("```"):
            # Remove opening ```json or ``` and closing ```
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse methodology extraction response for %s",
            source_id,
        )
        return MethodologyAssessment(
            source_id=source_id,
            content_basis=content_basis,
            confidence="low",
        )

    # Validate and coerce study_design
    raw_design = data.get("study_design", "unknown")
    study_design = (
        StudyDesign(raw_design)
        if raw_design in _VALID_STUDY_DESIGNS
        else StudyDesign.UNKNOWN
    )

    # Validate sample_size
    sample_size = data.get("sample_size")
    if sample_size is not None:
        try:
            sample_size = int(sample_size)
        except (ValueError, TypeError):
            sample_size = None

    # Determine confidence — forced to "low" for abstract-only
    if content_basis == "abstract":
        confidence = "low"
    else:
        raw_confidence = data.get("confidence", "medium")
        confidence = raw_confidence if raw_confidence in _VALID_CONFIDENCE else "medium"

    return MethodologyAssessment(
        source_id=source_id,
        study_design=study_design,
        sample_size=sample_size,
        sample_description=data.get("sample_description"),
        effect_size=data.get("effect_size"),
        statistical_significance=data.get("statistical_significance"),
        limitations_noted=data.get("limitations_noted") or [],
        potential_biases=data.get("potential_biases") or [],
        confidence=confidence,
        content_basis=content_basis,
    )


def _get_source_content(
    source: ResearchSource,
    min_content_length: int = MIN_CONTENT_LENGTH,
) -> tuple[str, str]:
    """Extract usable content and content_basis from a source.

    Returns (content_text, content_basis) where content_basis is
    "full_text" or "abstract".
    """
    # Prefer full content
    if source.content and len(source.content) >= min_content_length:
        return source.content, "full_text"
    # Fall back to snippet (typically abstract for academic sources)
    if source.snippet and len(source.snippet) >= min_content_length:
        return source.snippet, "abstract"
    return "", ""


class MethodologyAssessor:
    """Assesses methodological quality of research sources via LLM extraction.

    Only assesses ACADEMIC sources with sufficient content (>200 chars).
    Confidence is forced to "low" for abstract-only content.

    Args:
        provider_id: LLM provider to use (None = workflow default).
        model: Model override (None = provider default).
        timeout: Timeout per LLM extraction call in seconds.
        min_content_length: Minimum chars to trigger assessment.
    """

    def __init__(
        self,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
        min_content_length: int = MIN_CONTENT_LENGTH,
    ) -> None:
        self._provider_id = provider_id
        self._model = model
        self._timeout = timeout
        self._min_content_length = min_content_length

    def filter_assessable_sources(
        self,
        sources: list[ResearchSource],
    ) -> list[tuple[ResearchSource, str, str]]:
        """Filter sources to those eligible for methodology assessment.

        Returns list of (source, content_text, content_basis) tuples for
        ACADEMIC sources with sufficient content.
        """
        min_len = self._min_content_length
        eligible: list[tuple[ResearchSource, str, str]] = []
        for source in sources:
            if source.source_type != SourceType.ACADEMIC:
                continue
            content_text, content_basis = _get_source_content(
                source, min_content_length=min_len,
            )
            if not content_text or len(content_text) < min_len:
                continue
            eligible.append((source, content_text, content_basis))
        return eligible

    async def assess_sources(
        self,
        sources: list[ResearchSource],
        workflow: Any = None,
        state: Any = None,
        llm_call_fn: Optional[LLMCallFn] = None,
    ) -> list[MethodologyAssessment]:
        """Extract methodology metadata from research sources.

        Filters to ACADEMIC sources with >min_content_length chars, then
        makes individual LLM calls for metadata extraction.

        Args:
            sources: All research sources (will be filtered to academic).
            workflow: DeepResearchWorkflow instance for LLM calls (optional).
            state: DeepResearchState for execute_llm_call (optional).
            llm_call_fn: Standalone async callable ``(system_prompt, user_prompt) -> content``
                for making LLM calls without a full workflow object.  Takes
                precedence over *workflow*/*state* when provided.

        Returns:
            List of MethodologyAssessment, one per assessed source.
        """
        eligible = self.filter_assessable_sources(sources)
        if not eligible:
            logger.info("No eligible sources for methodology assessment")
            return []

        logger.info(
            "Assessing methodology for %d/%d sources",
            len(eligible),
            len(sources),
        )

        assessments: list[MethodologyAssessment] = []

        for source, content_text, content_basis in eligible:
            assessment = await self._assess_single(
                source=source,
                content_text=content_text,
                content_basis=content_basis,
                workflow=workflow,
                state=state,
                llm_call_fn=llm_call_fn,
            )
            assessments.append(assessment)

        logger.info(
            "Completed methodology assessment: %d assessments, "
            "%d with known study design",
            len(assessments),
            sum(1 for a in assessments if a.study_design != StudyDesign.UNKNOWN),
        )

        return assessments

    async def _assess_single(
        self,
        source: ResearchSource,
        content_text: str,
        content_basis: str,
        workflow: Any = None,
        state: Any = None,
        llm_call_fn: Optional[LLMCallFn] = None,
    ) -> MethodologyAssessment:
        """Assess a single source via LLM extraction.

        Falls back to an UNKNOWN assessment on any failure.
        """
        user_prompt = _build_extraction_user_prompt(
            source_title=source.title,
            content=content_text,
            content_basis=content_basis,
        )

        try:
            # Path 1: Standalone LLM call function (from handler context)
            if llm_call_fn is not None:
                content = await llm_call_fn(
                    METHODOLOGY_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt,
                )
                if content is not None:
                    return _parse_llm_response(content, source.id, content_basis)
                else:
                    logger.warning(
                        "Methodology assessment LLM call returned None for %s",
                        source.id,
                    )

            # Path 2: Full workflow pipeline (from deep research context)
            elif workflow is not None and state is not None:
                from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
                    LLMCallResult,
                    execute_llm_call,
                )

                result = await execute_llm_call(
                    workflow=workflow,
                    state=state,
                    phase_name="methodology_assessment",
                    system_prompt=METHODOLOGY_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    provider_id=self._provider_id,
                    model=self._model,
                    temperature=0.1,
                    timeout=self._timeout,
                    role="methodology_assessor",
                )

                if isinstance(result, LLMCallResult):
                    return _parse_llm_response(
                        result.result.content,
                        source.id,
                        content_basis,
                    )
                else:
                    # WorkflowResult (error) — fall through to default
                    logger.warning(
                        "Methodology assessment LLM call failed for %s: %s",
                        source.id,
                        getattr(result, "error", "unknown"),
                    )
            else:
                logger.debug(
                    "No llm_call_fn or workflow/state provided; skipping LLM call for %s",
                    source.id,
                )
        except Exception:
            logger.warning(
                "Methodology assessment failed for source %s",
                source.id,
                exc_info=True,
            )

        # Fallback: unknown assessment
        return MethodologyAssessment(
            source_id=source.id,
            content_basis=content_basis or "abstract",
            confidence="low",
        )


def format_methodology_context(
    assessments: list[MethodologyAssessment],
    id_to_citation: dict[str, int],
    sources: list[ResearchSource],
) -> str:
    """Format methodology assessments as a synthesis prompt section.

    Produces a ``## Methodology Context`` section suitable for injection
    into the synthesis prompt. Only includes assessments with meaningful
    metadata (study_design != UNKNOWN or has sample_size/effect_size).

    Args:
        assessments: List of methodology assessments.
        id_to_citation: source-id → citation-number mapping.
        sources: All research sources (for title lookup).

    Returns:
        Formatted methodology context section, or empty string if no
        meaningful assessments.
    """
    if not assessments:
        return ""

    # Build source lookup
    source_map: dict[str, ResearchSource] = {s.id: s for s in sources}

    lines: list[str] = []

    for assessment in assessments:
        # Skip assessments with no meaningful metadata
        has_design = assessment.study_design != StudyDesign.UNKNOWN
        has_sample = assessment.sample_size is not None
        has_effect = assessment.effect_size is not None
        if not (has_design or has_sample or has_effect):
            continue

        source = source_map.get(assessment.source_id)
        if not source:
            continue

        cn = id_to_citation.get(assessment.source_id)
        label = f"[{cn}]" if cn is not None else f"[{assessment.source_id}]"
        title = source.title

        # Header line: [1] Smith et al. (2021) — Randomized controlled trial, N=450
        design_label = assessment.study_design.value.replace("_", " ").title()
        parts = [f"{label} {title} — {design_label}"]
        if assessment.sample_size is not None:
            parts[0] += f", N={assessment.sample_size}"

        lines.append(parts[0])

        # Effect size and significance
        if assessment.effect_size or assessment.statistical_significance:
            effect_parts = []
            if assessment.effect_size:
                effect_parts.append(sanitize_external_content(assessment.effect_size))
            if assessment.statistical_significance:
                effect_parts.append(f"({assessment.statistical_significance})")
            lines.append(f"    Effect: {' '.join(effect_parts)}")

        # Sample description
        if assessment.sample_description:
            lines.append(f"    Sample: {sanitize_external_content(assessment.sample_description)}")

        # Limitations
        if assessment.limitations_noted:
            safe_limitations = [sanitize_external_content(l) for l in assessment.limitations_noted]
            lines.append(f"    Limitations: {'; '.join(safe_limitations)}")

        # Biases
        if assessment.potential_biases:
            safe_biases = [sanitize_external_content(b) for b in assessment.potential_biases]
            lines.append(f"    Potential biases: {'; '.join(safe_biases)}")

        # Content basis caveat
        if assessment.content_basis == "abstract":
            lines.append("    (extracted from: abstract — treat with lower confidence)")
        else:
            lines.append(f"    (extracted from: {assessment.content_basis})")

        lines.append("")  # blank line between entries

    if not lines:
        return ""

    header = (
        "## Methodology Context\n"
        "The following methodology metadata was extracted from academic sources. "
        "Use this as context for qualitative weighting of evidence — "
        "not as ground truth.\n"
    )
    return header + "\n".join(lines)
