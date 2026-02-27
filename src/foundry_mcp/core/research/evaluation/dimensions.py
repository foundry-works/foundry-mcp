"""Evaluation dimension definitions and rubrics.

Each dimension defines a 1-5 scoring rubric that an LLM judge uses to
evaluate a completed research report.  The rubrics are intentionally
concise — they fit in a single evaluation prompt without overwhelming
the judge model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """A single evaluation dimension with scoring rubric.

    Attributes:
        name: Machine-readable dimension identifier (e.g. ``"depth"``).
        display_name: Human-readable label for reports.
        description: One-line description of what the dimension measures.
        rubric: Multi-line rubric mapping scores 1-5 to quality criteria.
            Used verbatim in the LLM evaluation prompt.
    """

    name: str
    display_name: str
    description: str
    rubric: str


DEPTH = Dimension(
    name="depth",
    display_name="Research Depth",
    description="Thoroughness of investigation across the research topic.",
    rubric=(
        "1: Superficial — only surface-level facts, no exploration of nuances or sub-topics.\n"
        "2: Shallow — covers obvious aspects but misses important sub-topics or context.\n"
        "3: Moderate — addresses main aspects with some depth, but lacks detail in key areas.\n"
        "4: Thorough — explores most important sub-topics with meaningful detail and context.\n"
        "5: Comprehensive — exhaustive investigation covering nuances, edge cases, and "
        "multiple perspectives with rich detail."
    ),
)

SOURCE_QUALITY = Dimension(
    name="source_quality",
    display_name="Source Quality",
    description="Credibility, diversity, and recency of cited sources.",
    rubric=(
        "1: Poor — few or no credible sources; relies on unreliable or outdated material.\n"
        "2: Weak — limited sources with questionable credibility or all from one domain.\n"
        "3: Adequate — uses mostly credible sources but limited diversity or recency.\n"
        "4: Strong — diverse credible sources from multiple domains, mostly current.\n"
        "5: Excellent — authoritative, diverse, recent sources from academic, industry, "
        "and primary sources with clear provenance."
    ),
)

ANALYTICAL_RIGOR = Dimension(
    name="analytical_rigor",
    display_name="Analytical Rigor",
    description="Quality of reasoning, evidence use, and critical analysis.",
    rubric=(
        "1: Absent — no analysis; merely restates source material without reasoning.\n"
        "2: Weak — minimal analysis; claims made without supporting evidence.\n"
        "3: Moderate — some analytical reasoning but inconsistent evidence linkage.\n"
        "4: Strong — clear reasoning with evidence-backed claims and identified limitations.\n"
        "5: Rigorous — systematic analysis with strong evidence chains, addresses "
        "counter-arguments, acknowledges uncertainty, and draws well-supported conclusions."
    ),
)

COMPLETENESS = Dimension(
    name="completeness",
    display_name="Completeness",
    description="Coverage of all dimensions and perspectives of the research query.",
    rubric=(
        "1: Fragmentary — addresses only a small fraction of the query's scope.\n"
        "2: Partial — covers some aspects but leaves major dimensions unaddressed.\n"
        "3: Moderate — addresses most explicit query dimensions but misses implicit ones.\n"
        "4: Thorough — covers all explicit dimensions and most implicit perspectives.\n"
        "5: Complete — fully addresses all query dimensions including implicit needs, "
        "edge cases, and alternative perspectives."
    ),
)

GROUNDEDNESS = Dimension(
    name="groundedness",
    display_name="Groundedness",
    description="Whether claims are supported by cited evidence.",
    rubric=(
        "1: Ungrounded — most claims lack citations; appears speculative.\n"
        "2: Weakly grounded — some citations but many claims unsupported or mis-cited.\n"
        "3: Partially grounded — key claims cited but secondary claims lack support.\n"
        "4: Well-grounded — most claims supported by appropriate citations.\n"
        "5: Fully grounded — virtually all factual claims backed by cited evidence; "
        "clear distinction between sourced facts and analytical commentary."
    ),
)

STRUCTURE = Dimension(
    name="structure",
    display_name="Structure & Readability",
    description="Organization, readability, and citation consistency.",
    rubric=(
        "1: Disorganized — no clear structure; difficult to follow; inconsistent citations.\n"
        "2: Weak — some structure but poor flow; citation format varies.\n"
        "3: Adequate — recognizable sections and consistent citations, but could be clearer.\n"
        "4: Well-structured — logical flow with clear sections, consistent citations, "
        "and good readability.\n"
        "5: Excellent — professional structure with clear narrative arc, consistent "
        "citation format, effective use of sections/subsections, and executive summary."
    ),
)

PRACTICAL_VALUE = Dimension(
    name="practical_value",
    display_name="Practical Value",
    description="Actionability, specificity of insights, and usefulness to the reader.",
    rubric=(
        "1: Abstract/generic — no actionable takeaways.\n"
        "2: Mostly generic with occasional specific points.\n"
        "3: Some actionable insights but lacks specificity in key areas.\n"
        "4: Mostly actionable with specific, useful recommendations.\n"
        "5: Highly actionable — specific, concrete insights the reader can immediately use."
    ),
)

BALANCE = Dimension(
    name="balance",
    display_name="Balance & Objectivity",
    description="Presentation of multiple perspectives, acknowledgment of limitations and counterarguments.",
    rubric=(
        "1: One-sided — no acknowledgment of alternatives or limitations.\n"
        "2: Mostly one-sided with token mention of other views.\n"
        "3: Acknowledges alternatives but doesn't explore them meaningfully.\n"
        "4: Fair treatment of multiple perspectives with clear limitation acknowledgment.\n"
        "5: Exemplary balance — thorough multi-perspective analysis with nuanced "
        "limitation discussion."
    ),
)

#: All evaluation dimensions in canonical order.
DIMENSIONS: tuple[Dimension, ...] = (
    DEPTH,
    SOURCE_QUALITY,
    ANALYTICAL_RIGOR,
    COMPLETENESS,
    GROUNDEDNESS,
    STRUCTURE,
    PRACTICAL_VALUE,
    BALANCE,
)

#: Lookup by name for quick access.
DIMENSION_BY_NAME: dict[str, Dimension] = {d.name: d for d in DIMENSIONS}
