"""RIS export for deep research sources.

Converts ResearchSource objects to RIS (Research Information Systems) format
compatible with Zotero, Mendeley, EndNote, and other reference managers.

RIS format specification:
- Each entry starts with TY (type) and ends with ER (end record)
- Fields are two-letter tags followed by ``  - `` and the value
- Multiple authors use multiple AU lines
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.sources import ResearchSource


def _determine_ris_type(source: "ResearchSource") -> str:
    """Determine the RIS entry type for a source.

    - Academic with venue -> JOUR (journal article)
    - Conference -> CONF (conference proceeding)
    - Web/other -> ELEC (electronic source)
    """
    meta = source.metadata or {}
    venue = (meta.get("venue") or meta.get("journal") or "").lower()

    if any(kw in venue for kw in ("conference", "proceedings", "workshop", "symposium")):
        return "CONF"

    from foundry_mcp.core.research.models.sources import SourceType

    if source.source_type == SourceType.ACADEMIC:
        return "JOUR"

    return "ELEC"


def source_to_ris_entry(source: "ResearchSource") -> str:
    """Convert a single ResearchSource to an RIS entry string.

    Args:
        source: Research source with metadata.

    Returns:
        RIS entry block (TY through ER).
    """
    meta = source.metadata or {}
    lines: list[str] = []

    # Entry type
    ris_type = _determine_ris_type(source)
    lines.append(f"TY  - {ris_type}")

    # Title
    lines.append(f"TI  - {source.title}")

    # Authors (one AU line per author)
    authors = meta.get("authors") or []
    for a in authors:
        name = a.get("name", a) if isinstance(a, dict) else str(a)
        if name:
            lines.append(f"AU  - {name}")

    # Year
    year = meta.get("year")
    if year:
        lines.append(f"PY  - {year}")

    # Journal/venue
    venue = meta.get("venue") or meta.get("journal")
    if venue:
        lines.append(f"JO  - {venue}")

    # DOI
    doi = meta.get("doi")
    if doi:
        lines.append(f"DO  - {doi}")

    # URL
    if source.url:
        lines.append(f"UR  - {source.url}")

    # Abstract
    if source.snippet:
        lines.append(f"AB  - {source.snippet[:1000]}")

    # Volume, issue, pages
    volume = meta.get("volume")
    if volume:
        lines.append(f"VL  - {volume}")
    issue = meta.get("issue")
    if issue:
        lines.append(f"IS  - {issue}")
    pages = meta.get("pages")
    if pages and "-" in str(pages):
        parts = str(pages).split("-", 1)
        lines.append(f"SP  - {parts[0].strip()}")
        lines.append(f"EP  - {parts[1].strip()}")
    elif pages:
        lines.append(f"SP  - {pages}")

    # Keywords / fields of study
    fields = meta.get("fields_of_study") or meta.get("fields") or []
    if isinstance(fields, list):
        for f in fields:
            if isinstance(f, str) and f.strip():
                lines.append(f"KW  - {f.strip()}")

    # End record
    lines.append("ER  - ")

    return "\n".join(lines)


def sources_to_ris(sources: list["ResearchSource"]) -> str:
    """Convert a list of research sources to RIS format.

    Generates one TY-ER block per source.

    Args:
        sources: List of research sources to export.

    Returns:
        Complete RIS file content as a string.
    """
    if not sources:
        return ""

    entries = [source_to_ris_entry(source) for source in sources]
    return "\n\n".join(entries) + "\n"
