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


def extract_cited_numbers(report: str, *, max_citation: int | None = None) -> set[int]:
    """Extract all citation numbers referenced in the report.

    Finds all ``[N]`` patterns in the report text. Ignores markdown
    link syntax by only matching bare numeric brackets.

    Args:
        report: The markdown report text.
        max_citation: If provided, exclude numbers above this value.
            Pass ``len(state.sources)`` or ``max(valid_numbers)`` to
            filter out year references like ``[2025]``.

    Returns:
        Set of cited integer citation numbers.
    """
    numbers = {int(m.group(1)) for m in _CITATION_RE.finditer(report)}
    if max_citation is not None:
        numbers = {n for n in numbers if n <= max_citation}
    return numbers


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
    entries: list[str] = []

    for cn in sorted(citation_map):
        if cited_only and (cited_numbers is None or cn not in cited_numbers):
            continue
        source = citation_map[cn]

        if use_apa:
            entries.append(f"[{cn}] {format_source_apa(source)}")
        else:
            title = source.title or "Untitled"
            if source.url:
                entries.append(f"[{cn}] [{title}]({source.url})")
            else:
                entries.append(f"[{cn}] {title}")

    if not entries:
        return ""

    return "\n".join(["", heading, ""] + entries + [""])


def remove_dangling_citations(
    report: str,
    valid_numbers: set[int],
    *,
    max_citation: int | None = None,
) -> str:
    """Remove citation markers that reference non-existent sources.

    Replaces ``[N]`` with empty string when N is not in *valid_numbers*
    and N is within the plausible citation range.

    Args:
        report: The markdown report text.
        valid_numbers: Set of citation numbers that exist in state.
        max_citation: If provided, numbers above this are preserved
            (assumed to be year references like ``[2025]``, not citations).

    Returns:
        Report with dangling citations removed.
    """

    def _replace(match: re.Match) -> str:
        num = int(match.group(1))
        # Preserve numbers above the plausible citation range
        # (likely year references like [2025], not citations)
        if max_citation is not None and num > max_citation:
            return match.group(0)
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


def renumber_citations(
    report: str,
    state: "DeepResearchState",
    *,
    max_citation: int | None = None,
) -> tuple[str, dict[int, int]]:
    """Renumber inline citations to reading order (1, 2, 3, ...).

    Scans the report left-to-right for ``[N]`` citations in order of
    first appearance and builds a mapping from old citation numbers to
    new sequential numbers.  Updates the report text, ``source.citation_number``
    on all state sources, and ``state.next_citation_number``.

    Args:
        report: The markdown report text (after dangling removal).
        state: Research state — sources are mutated in place.
        max_citation: Numbers above this are assumed to be year references
            (e.g. ``[2025]``) and are left untouched.

    Returns:
        Tuple of (renumbered_report, renumber_map) where renumber_map
        is ``{old_number: new_number}``.  Empty dict when no renumbering
        was needed.
    """
    # 1. Collect citation numbers in first-appearance order
    seen: dict[int, int] = {}  # old -> new
    next_new = 1
    for m in _CITATION_RE.finditer(report):
        num = int(m.group(1))
        if max_citation is not None and num > max_citation:
            continue
        if num not in seen:
            seen[num] = next_new
            next_new += 1

    # Nothing to renumber — either no citations or already in order
    if not seen or all(k == v for k, v in seen.items()):
        return report, {}

    # 2. Replace all [old] → [new] in the report
    def _replace(match: re.Match) -> str:
        num = int(match.group(1))
        if num in seen:
            return f"[{seen[num]}]"
        return match.group(0)

    report = _CITATION_RE.sub(_replace, report)

    # 3. Update source citation numbers on state.
    # Cited sources get their new numbers from the map.
    # Uncited sources are reassigned sequential numbers after the cited
    # ones to avoid collisions (e.g., uncited source with cn=2 clashing
    # with a cited source renumbered to cn=2).
    next_uncited = next_new  # continues from where cited numbering left off
    for source in state.sources:
        if source.citation_number is not None and source.citation_number in seen:
            source.citation_number = seen[source.citation_number]
        elif source.citation_number is not None:
            source.citation_number = next_uncited
            next_uncited += 1

    # 4. Update next_citation_number
    state.next_citation_number = next_uncited

    return report, seen


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

    style = state.research_profile.citation_style if state.research_profile else None
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
    4. Renumber citations to reading order (1, 2, 3, ...).
    5. Append a deterministic Sources/References section from state.

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

    # Upper bound for plausible citation numbers.  Numbers above this
    # threshold are assumed to be year references (e.g. [2025]) rather
    # than citations.  We use max(actual_max, 999) so that hallucinated
    # citation numbers like [99] are still caught as dangling while year
    # references are left intact.  When there are no sources at all,
    # max_cn is None (no filtering — every [N] is dangling).
    max_cn: int | None = None
    if valid_numbers:
        max_cn = max(max(valid_numbers), 999)

    # 1. Extract cited numbers
    cited_numbers = extract_cited_numbers(report, max_citation=max_cn)

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
        report = remove_dangling_citations(report, valid_numbers, max_citation=max_cn)
        # Recompute after removal
        cited_numbers = extract_cited_numbers(report, max_citation=max_cn)

    # 4. Renumber citations to reading order (1, 2, 3, ...)
    report, renumber_map = renumber_citations(report, state, max_citation=max_cn)
    if renumber_map:
        logger.info(
            "Renumbered %d citation(s) to reading order: %s",
            len(renumber_map),
            renumber_map,
        )
        # Recompute cited numbers after renumbering
        cited_numbers = extract_cited_numbers(report, max_citation=max_cn)

    # 5. Resolve format style and append deterministic section
    format_style = _resolve_format_style(state, query_type)
    sources_section = build_sources_section(
        state,
        cited_only=True,
        cited_numbers=cited_numbers,
        format_style=format_style,
    )
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
        "renumbered_count": len(renumber_map),
    }

    return report, metadata
