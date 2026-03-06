"""LLM-interpreted Research Confidence section for deep research reports.

Appends a ``## Research Confidence`` section that contextualizes claim
verification data for the reader.  An LLM interprets the raw verification
stats in relation to the query type and content, so the reader understands
what the numbers mean for *their specific question* rather than seeing
opaque metrics.

Two-step approach:
  Step 1 — ``build_confidence_context``: deterministic data assembly from
           claim verification results (no LLM, no I/O).
  Step 2 — ``generate_confidence_section``: short LLM call that interprets
           the structured context and produces a reader-friendly section.
           Falls back to a deterministic bullet-point summary on LLM failure.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.deep_research import (
        ClaimVerificationResult,
        DeepResearchState,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for confidence interpretation
# ---------------------------------------------------------------------------

_CONFIDENCE_SYSTEM_PROMPT = """\
You are writing a brief Research Confidence section for a research report.
You will receive the original research question, the query classification,
and structured verification data showing how claims in the report were
checked against gathered sources.

Your job is to help the reader calibrate their trust in the report by
explaining what was verified, what wasn't, and why — in terms that make
sense for THIS specific type of question.

Rules:
- Write 150-300 words as a ## Research Confidence section in markdown
- Do NOT use the word "fidelity" or expose raw decimal scores
- Do NOT use hedging language ("it should be noted", "it is worth mentioning")
- Distinguish between claims that are naturally inferential for this query \
type (synthesis, recommendations, comparisons) vs. claims where source \
evidence is genuinely absent (factual assertions without backing)
- Name specific gaps if sub-queries failed (e.g., "Data on X could not be \
retrieved from any source")
- Mention corrections if any were made ("One claim about X was found to \
contradict source data and was corrected")
- Note iteration count if >1 ("This report was refined over N research cycles")
- Be direct and concise — this is an appendix, not a new analysis"""


# ---------------------------------------------------------------------------
# Step 1: Deterministic context assembly
# ---------------------------------------------------------------------------


def build_confidence_context(state: "DeepResearchState") -> dict[str, Any] | None:
    """Assemble structured verification data for the confidence LLM call.

    Pure data transformation — no LLM, no I/O.

    Returns:
        A dict suitable for JSON serialization into the LLM prompt,
        or ``None`` if no claim verification data is available.
    """
    cv: ClaimVerificationResult | None = state.claim_verification
    if cv is None:
        return None

    # Verdict distribution
    verdict_distribution = {
        "supported": cv.claims_supported,
        "partially_supported": cv.claims_partially_supported,
        "unsupported": cv.claims_unsupported,
        "contradicted": cv.claims_contradicted,
        "total_verified": cv.claims_verified,
        "total_extracted": cv.claims_extracted,
    }

    # Per-section breakdown of unsupported claims
    section_unsupported: Counter[str] = Counter()
    # Claim-type breakdown of unsupported claims
    type_unsupported: Counter[str] = Counter()
    unsupported_claims_by_type: dict[str, list[str]] = {}

    for detail in cv.details:
        if detail.verdict == "UNSUPPORTED":
            section = detail.report_section or "Unknown"
            section_unsupported[section] += 1
            claim_type = detail.claim_type or "unknown"
            type_unsupported[claim_type] += 1
            unsupported_claims_by_type.setdefault(claim_type, []).append(detail.claim)

    # Failed sub-queries
    failed_queries = [
        {"query": sq.query, "rationale": sq.rationale}
        for sq in state.failed_sub_queries()
    ]

    # Corrections applied
    corrections = []
    for detail in cv.details:
        if detail.correction_applied:
            corrections.append({
                "claim": detail.claim,
                "corrected_text": detail.corrected_text,
            })

    context: dict[str, Any] = {
        "original_query": state.original_query,
        "query_type": state.metadata.get("_query_type"),
        "verdict_distribution": verdict_distribution,
        "sections_with_unsupported_claims": dict(section_unsupported.most_common()),
        "unsupported_by_claim_type": dict(type_unsupported),
        "unsupported_claims_detail": {
            k: v[:5] for k, v in unsupported_claims_by_type.items()  # Cap at 5 per type
        },
        "failed_sub_queries": failed_queries,
        "corrections_applied": len(corrections),
        "correction_summaries": corrections[:5],  # Cap at 5
        "fidelity_trajectory": state.fidelity_scores,
        "iteration_count": state.iteration,
        "max_iterations": state.max_iterations,
        "source_count": len(state.sources),
    }

    return context


# ---------------------------------------------------------------------------
# Step 2: LLM interpretation (with deterministic fallback)
# ---------------------------------------------------------------------------


def _build_deterministic_fallback(context: dict[str, Any]) -> str:
    """Produce a bullet-point summary from raw data when the LLM call fails."""
    vd = context.get("verdict_distribution", {})
    total = vd.get("total_verified", 0)
    supported = vd.get("supported", 0)
    partial = vd.get("partially_supported", 0)
    unsupported = vd.get("unsupported", 0)
    contradicted = vd.get("contradicted", 0)

    lines = ["## Research Confidence\n"]

    if total > 0:
        lines.append(
            f"- **{total} claims** were verified against source material: "
            f"{supported} fully supported, {partial} partially supported, "
            f"{unsupported} unsupported, {contradicted} contradicted."
        )
    else:
        lines.append("- No claims were verified against source material.")

    corrections = context.get("corrections_applied", 0)
    if corrections:
        lines.append(f"- **{corrections} correction(s)** were applied where source data contradicted report claims.")

    failed = context.get("failed_sub_queries", [])
    if failed:
        query_texts = [fq["query"] for fq in failed[:3]]
        lines.append(f"- **{len(failed)} sub-query(ies) returned no results:** {'; '.join(query_texts)}")

    iteration_count = context.get("iteration_count", 1)
    if iteration_count > 1:
        lines.append(f"- This report was refined over **{iteration_count} research cycles**.")

    source_count = context.get("source_count", 0)
    if source_count:
        lines.append(f"- **{source_count} sources** were gathered and consulted.")

    return "\n".join(lines)


async def generate_confidence_section(
    state: "DeepResearchState",
    llm_call_fn: Callable,
    *,
    query_type: str | None = None,
) -> str | None:
    """Generate an LLM-interpreted Research Confidence section.

    Uses a fast model (haiku-class) for the interpretation call.
    Returns markdown string starting with ``## Research Confidence``,
    or ``None`` if no claim verification data exists.

    Falls back to a deterministic summary on LLM failure.

    Args:
        state: The deep research state with claim verification results.
        llm_call_fn: Async callable with signature
            ``(system_prompt, user_prompt) -> str``.
            The caller is responsible for binding provider/model/timeout.
        query_type: Optional query classification override.
    """
    context = build_confidence_context(state)
    if context is None:
        return None

    if query_type:
        context["query_type"] = query_type

    user_prompt = json.dumps(context, indent=2, default=str)

    try:
        result = await llm_call_fn(_CONFIDENCE_SYSTEM_PROMPT, user_prompt)

        if not isinstance(result, str) or not result.strip():
            logger.warning("Confidence LLM returned empty result, using fallback")
            return _build_deterministic_fallback(context)

        # Ensure section starts with the expected heading
        text = result.strip()
        if not text.startswith("## Research Confidence"):
            text = "## Research Confidence\n\n" + text

        return text

    except Exception as exc:
        logger.warning(
            "Confidence section LLM call failed for research %s: %s, using fallback",
            state.id,
            exc,
        )
        return _build_deterministic_fallback(context)
