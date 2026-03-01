"""Citation post-processing for deep research reports.

Scans synthesized reports for inline [N] citations, verifies consistency
against the source registry, removes dangling references, and appends
a deterministic Sources/References section built from state rather than
LLM output.  Supports APA 7th-edition formatting when the research
profile's ``citation_style`` is ``"apa"`` or the query type is
``"literature_review"``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.deep_research import DeepResearchState
    from foundry_mcp.core.research.models.sources import ResearchSource

logger = logging.getLogger(__name__)

# Matches [N] where N is one or more digits, but NOT inside markdown link
# syntax like [text](url). The negative lookahead (?!\() ensures we skip
# patterns followed by a parenthesised URL.
_CITATION_RE = re.compile(r"\[(\d+)\](?!\()")


def extract_cited_numbers(report: str) -> set[int]:
    """Extract all citation numbers referenced in the report.

    Finds all ``[N]`` patterns in the report text. Ignores markdown
    link syntax by only matching bare numeric brackets.

    Args:
        report: The markdown report text.

    Returns:
        Set of cited integer citation numbers.
    """
    return {int(m.group(1)) for m in _CITATION_RE.finditer(report)}


# ---------------------------------------------------------------------------
# APA 7th-edition formatting
# ---------------------------------------------------------------------------


def _apa_authors(authors_str: str) -> str:
    """Format a comma-separated author string per APA 7th-edition rules.

    APA rules:
    - 1-20 authors: list all, with ``&`` before last.
    - 21+ authors: first 19, ``...``, last author (rare — we use >5 as
      the practical "et al." threshold consistent with in-text norms).
    - For our data the Semantic Scholar provider already joins names with
      commas, so we split on ``, `` and re-join with APA punctuation.

    Returns the formatted author string, or empty string if input is empty.
    """
    if not authors_str:
        return ""

    authors = [a.strip() for a in authors_str.split(",") if a.strip()]
    if not authors:
        return ""

    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} & {authors[1]}"
    if len(authors) <= 5:
        return ", ".join(authors[:-1]) + f", & {authors[-1]}"

    # >5 authors → first author et al.
    return f"{authors[0]} et al."


def format_source_apa(source: "ResearchSource") -> str:
    """Format a single source as an APA 7th-edition reference entry.

    Gracefully degrades based on available metadata:
    - **Full academic**: ``Authors (Year). Title. *Venue*. https://doi.org/DOI``
    - **Partial academic**: omits venue or DOI when unavailable.
    - **Web source**: ``Author/Organization (Year). Title. *Site Name*. URL``
    - **Minimal fallback**: ``Title. URL``

    Args:
        source: A ``ResearchSource`` with optional ``metadata`` dict
            containing ``authors``, ``year``, ``venue``, ``doi``, etc.

    Returns:
        A single APA-formatted reference string (no trailing newline).
    """
    meta = source.metadata or {}
    authors = _apa_authors(meta.get("authors", ""))
    year = meta.get("year")
    venue = meta.get("venue")
    doi = meta.get("doi")
    title = source.title or "Untitled"

    # Year component
    year_str = f"({year})" if year else "(n.d.)"

    # Build URL — prefer DOI link, fall back to source URL
    url = f"https://doi.org/{doi}" if doi else (source.url or "")

    # Assemble reference
    parts: list[str] = []

    if authors:
        parts.append(f"{authors} {year_str}. {title}.")
    else:
        # No author: title moves to author position per APA
        parts.append(f"{title} {year_str}.")

    if venue:
        parts.append(f"*{venue}*.")

    if url:
        parts.append(url)

    return " ".join(parts)


def build_sources_section(
    state: "DeepResearchState",
    *,
    cited_only: bool = False,
    cited_numbers: set[int] | None = None,
    format_style: str = "default",
) -> str:
    """Build a deterministic Sources / References section from state.

    Args:
        state: Research state containing all sources with citation numbers.
        cited_only: If True, only include sources that were actually cited.
        cited_numbers: Pre-computed set of cited numbers (avoids re-scanning).
        format_style: ``"default"`` produces the existing
            ``[N] [Title](URL)`` format under a ``## Sources`` heading.
            ``"apa"`` produces APA 7th-edition entries under a
            ``## References`` heading.

    Returns:
        Markdown string for the Sources/References section (including heading).
    """
    citation_map = state.get_citation_map()
    if not citation_map:
        return ""

    use_apa = format_style == "apa"
    heading = "## References" if use_apa else "## Sources"
    lines = ["", heading, ""]

    for cn in sorted(citation_map):
        if cited_only and (cited_numbers is None or cn not in cited_numbers):
            continue
        source = citation_map[cn]

        if use_apa:
            lines.append(f"[{cn}] {format_source_apa(source)}")
        else:
            title = source.title or "Untitled"
            if source.url:
                lines.append(f"[{cn}] [{title}]({source.url})")
            else:
                lines.append(f"[{cn}] {title}")

    lines.append("")
    return "\n".join(lines)


def remove_dangling_citations(report: str, valid_numbers: set[int]) -> str:
    """Remove citation markers that reference non-existent sources.

    Replaces ``[N]`` with empty string when N is not in *valid_numbers*.

    Args:
        report: The markdown report text.
        valid_numbers: Set of citation numbers that exist in state.

    Returns:
        Report with dangling citations removed.
    """

    def _replace(match: re.Match) -> str:
        num = int(match.group(1))
        if num in valid_numbers:
            return match.group(0)
        return ""

    return _CITATION_RE.sub(_replace, report)


def strip_llm_sources_section(report: str) -> str:
    """Remove any Sources/References section generated by the LLM.

    Looks for common heading patterns (``## Sources``, ``## References``,
    ``## Works Cited``) and removes everything from that heading to the
    next heading of equal or higher level, or the end of the report.

    Args:
        report: The markdown report text.

    Returns:
        Report with LLM-generated sources section removed.
    """
    # Match ## Sources, ## References, ## Works Cited (case-insensitive)
    pattern = re.compile(
        r"^(#{1,2}\s+(?:Sources|References|Works\s+Cited|Bibliography))\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(report)
    if not match:
        return report

    start = match.start()
    heading_level = match.group(1).count("#")

    # Find the next heading of equal or higher level
    rest = report[match.end() :]
    next_heading = re.search(
        rf"^#{{{1},{heading_level}}}\s+\S",
        rest,
        re.MULTILINE,
    )
    if next_heading:
        end = match.end() + next_heading.start()
    else:
        end = len(report)

    # Strip and clean up trailing whitespace
    return report[:start].rstrip() + report[end:]


def _resolve_format_style(
    state: "DeepResearchState",
    query_type: str | None = None,
) -> str:
    """Determine the citation format style from profile and query type.

    Resolution order:
    1. ``query_type == "literature_review"`` → always ``"apa"``
    2. ``state.research_profile.citation_style`` (e.g. ``"apa"``)
    3. Fallback → ``"default"``
    """
    if query_type == "literature_review":
        return "apa"

    style = state.research_profile.citation_style
    if style and style != "default":
        return style

    return "default"


def postprocess_citations(
    report: str,
    state: "DeepResearchState",
    *,
    query_type: str | None = None,
) -> tuple[str, dict]:
    """Run full citation post-processing pipeline on a report.

    Steps:
    1. Extract all cited ``[N]`` numbers from the report.
    2. Remove any LLM-generated Sources section.
    3. Remove dangling citations (referencing non-existent sources).
    4. Append a deterministic Sources/References section from state.

    The ``format_style`` for the appended section is resolved from the
    research profile's ``citation_style`` and the detected ``query_type``.
    Literature-review queries always produce APA references.

    Args:
        report: The synthesized markdown report.
        state: Research state with all sources and citation numbers.
        query_type: The classified query type (e.g. ``"literature_review"``).
            When ``"literature_review"``, forces APA formatting regardless
            of profile setting.

    Returns:
        Tuple of (processed_report, metadata_dict) where metadata contains
        citation statistics for audit logging.
    """
    citation_map = state.get_citation_map()
    valid_numbers = set(citation_map.keys())

    # 1. Extract cited numbers
    cited_numbers = extract_cited_numbers(report)

    # 2. Strip any LLM-generated sources section
    report = strip_llm_sources_section(report)

    # 3. Remove dangling citations
    dangling = cited_numbers - valid_numbers
    if dangling:
        logger.warning(
            "Removing %d dangling citation(s): %s",
            len(dangling),
            sorted(dangling),
        )
        report = remove_dangling_citations(report, valid_numbers)
        # Recompute after removal
        cited_numbers = extract_cited_numbers(report)

    # 4. Resolve format style and append deterministic section
    format_style = _resolve_format_style(state, query_type)
    sources_section = build_sources_section(state, format_style=format_style)
    if sources_section:
        report = report.rstrip() + "\n" + sources_section

    # Compute unreferenced sources (have citation number but never cited)
    unreferenced = valid_numbers - cited_numbers
    if unreferenced:
        logger.info(
            "%d source(s) have citation numbers but were not referenced in the report: %s",
            len(unreferenced),
            sorted(unreferenced),
        )

    metadata = {
        "total_citations_in_report": len(cited_numbers),
        "total_sources_with_numbers": len(valid_numbers),
        "dangling_citations_removed": len(dangling),
        "unreferenced_sources": len(unreferenced),
        "format_style": format_style,
    }

    return report, metadata
