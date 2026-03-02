"""BibTeX export for deep research sources (PLAN-3 item 5a).

Converts ResearchSource objects to BibTeX format entries compatible with
LaTeX, BibLaTeX, and reference managers like Zotero, Mendeley, and EndNote.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.core.research.models.sources import ResearchSource

# Characters that need escaping in BibTeX field values.
_BIBTEX_SPECIAL = str.maketrans({
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "~": r"\textasciitilde{}",
})


def _escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters in a string."""
    return text.translate(_BIBTEX_SPECIAL)


def _generate_citation_key(source: "ResearchSource") -> str:
    """Generate a stable citation key from source metadata.

    Format: ``firstauthor_year_firsttwowords`` (lowercase, ASCII-only).
    Falls back to source ID if insufficient metadata.
    """
    meta = source.metadata or {}
    parts: list[str] = []

    # First author surname
    authors = meta.get("authors") or []
    if authors:
        first = authors[0]
        name = first.get("name", first) if isinstance(first, dict) else str(first)
        # Extract surname (last word of name)
        surname = name.strip().split()[-1] if name.strip() else ""
        # Normalize to ASCII
        surname = unicodedata.normalize("NFKD", surname).encode("ascii", "ignore").decode()
        surname = re.sub(r"[^a-zA-Z]", "", surname).lower()
        if surname:
            parts.append(surname)

    # Year
    year = meta.get("year")
    if year:
        parts.append(str(year))

    # First two significant words from title
    title_words = re.findall(r"[a-zA-Z]+", source.title.lower())
    stop_words = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "is", "with"}
    significant = [w for w in title_words if w not in stop_words][:2]
    parts.extend(significant)

    if parts:
        return "_".join(parts)
    # Fallback to source ID
    return source.id.replace("-", "_")


def _determine_entry_type(source: "ResearchSource") -> str:
    """Determine the BibTeX entry type for a source.

    - venue contains "conference" or "proceedings" -> @inproceedings
    - venue is present -> @article
    - Otherwise -> @misc
    """
    meta = source.metadata or {}
    venue = (meta.get("venue") or meta.get("journal") or "").lower()

    if any(kw in venue for kw in ("conference", "proceedings", "workshop", "symposium")):
        return "inproceedings"
    if venue:
        return "article"
    return "misc"


def source_to_bibtex_entry(source: "ResearchSource", citation_key: str = "") -> str:
    """Convert a single ResearchSource to a BibTeX entry string.

    Args:
        source: Research source with metadata.
        citation_key: Optional override for the citation key.

    Returns:
        BibTeX entry string (e.g., ``@article{key, ...}``).
    """
    if not citation_key:
        citation_key = _generate_citation_key(source)

    entry_type = _determine_entry_type(source)
    meta = source.metadata or {}

    fields: list[str] = []
    fields.append(f"  title = {{{_escape_bibtex(source.title)}}}")

    # Authors
    authors = meta.get("authors") or []
    if authors:
        author_names: list[str] = []
        for a in authors:
            name = a.get("name", a) if isinstance(a, dict) else str(a)
            if name:
                author_names.append(_escape_bibtex(name))
        if author_names:
            fields.append(f"  author = {{{' and '.join(author_names)}}}")

    # Year
    year = meta.get("year")
    if year:
        fields.append(f"  year = {{{year}}}")

    # Venue/journal
    venue = meta.get("venue") or meta.get("journal")
    if venue:
        if entry_type == "inproceedings":
            fields.append(f"  booktitle = {{{_escape_bibtex(venue)}}}")
        else:
            fields.append(f"  journal = {{{_escape_bibtex(venue)}}}")

    # DOI
    doi = meta.get("doi")
    if doi:
        fields.append(f"  doi = {{{doi}}}")

    # URL
    if source.url:
        fields.append(f"  url = {{{source.url}}}")

    # Abstract (snippet)
    if source.snippet:
        fields.append(f"  abstract = {{{_escape_bibtex(source.snippet[:500])}}}")

    fields_str = ",\n".join(fields)
    return f"@{entry_type}{{{citation_key},\n{fields_str}\n}}"


def sources_to_bibtex(sources: list["ResearchSource"]) -> str:
    """Convert a list of research sources to BibTeX format.

    Generates one entry per source with unique citation keys.

    Args:
        sources: List of research sources to export.

    Returns:
        Complete BibTeX file content as a string.
    """
    if not sources:
        return ""

    used_keys: set[str] = set()
    entries: list[str] = []

    for source in sources:
        key = _generate_citation_key(source)
        # Ensure uniqueness
        original_key = key
        counter = 2
        while key in used_keys:
            key = f"{original_key}_{counter}"
            counter += 1
        used_keys.add(key)

        entries.append(source_to_bibtex_entry(source, key))

    return "\n\n".join(entries) + "\n"
