"""Reference export utilities for deep research.

Generates BibTeX and RIS formatted bibliographies from ResearchSource
metadata collected during deep research sessions.
"""

from foundry_mcp.core.research.export.bibtex import source_to_bibtex_entry, sources_to_bibtex
from foundry_mcp.core.research.export.ris import source_to_ris_entry, sources_to_ris

__all__ = [
    "source_to_bibtex_entry",
    "sources_to_bibtex",
    "source_to_ris_entry",
    "sources_to_ris",
]
