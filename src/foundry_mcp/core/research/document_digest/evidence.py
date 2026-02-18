"""Evidence extraction mixin for document digest.

Provides evidence extraction, relevance scoring, snippet building,
and locator generation used by DocumentDigestor.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Optional

from foundry_mcp.core.research.models.digest import EvidenceSnippet

logger = logging.getLogger(__name__)


class EvidenceExtractionMixin:
    """Mixin providing evidence extraction and locator generation for DocumentDigestor."""

    def _extract_evidence(
        self,
        text: str,
        query: str,
        *,
        max_snippets: Optional[int] = None,
    ) -> list[tuple[str, int, float]]:
        """Extract evidence snippets from text based on query relevance.

        Chunks the text and scores each chunk based on query term matching.
        Returns the top-scoring chunks as evidence snippets with their
        original position and relevance score.

        Scoring formula:
        - For each query term found in chunk (case-insensitive):
          score += 1 / (1 + log(term_frequency_in_corpus))
        - This gives higher weight to rarer terms

        Tie-breakers (applied in order):
        1. Higher score wins
        2. Earlier position wins (lower index)
        3. Longer chunk wins (more context)

        Empty/short query fallback:
        - If query is empty or < 3 chars, uses positional scoring
        - Early chunks get higher scores (1.0 - position/total)

        Args:
            text: Source text to extract evidence from.
            query: Research query to match against.
            max_snippets: Maximum number of snippets to return.
                Defaults to config.max_evidence_snippets.

        Returns:
            List of tuples (snippet_text, position_index, score).
            Sorted by score descending, then position ascending.

        Examples:
            >>> evidence = digestor._extract_evidence(
            ...     "Climate change affects coastal cities. Rising seas threaten infrastructure.",
            ...     "climate coastal impact"
            ... )
            >>> len(evidence) <= digestor.config.max_evidence_snippets
            True
        """
        if max_snippets is None:
            max_snippets = self.config.max_evidence_snippets

        # Chunk the text using configured sizing constraints
        target_size = min(self.config.chunk_size, self.config.max_snippet_length)
        chunks = self._chunk_text(
            text,
            target_size=target_size,
            max_size=self.config.max_snippet_length,
            min_size=min(50, self.config.max_snippet_length),
        )
        if not chunks:
            return []

        # Handle empty/short query with positional fallback
        if not query or len(query.strip()) < 3:
            return self._score_by_position(chunks, max_snippets)

        # Extract and normalize query terms
        query_terms = self._extract_terms(query)
        if not query_terms:
            return self._score_by_position(chunks, max_snippets)

        # Calculate corpus term frequencies for IDF-like weighting
        corpus_text = text.lower()
        term_frequencies = {}
        for term in query_terms:
            term_frequencies[term] = corpus_text.count(term.lower())

        # Score each chunk
        scored_chunks: list[tuple[str, int, float, int]] = []
        for idx, chunk in enumerate(chunks):
            score = self._score_chunk(chunk, query_terms, term_frequencies)
            # Store: (chunk, position, score, length) for tie-breaking
            scored_chunks.append((chunk, idx, score, len(chunk)))

        # Sort by: score DESC, position ASC, length DESC
        scored_chunks.sort(key=lambda x: (-x[2], x[1], -x[3]))

        # Return top N as (text, position, score)
        return [(chunk, pos, score) for chunk, pos, score, _ in scored_chunks[:max_snippets]]

    def _extract_terms(self, query: str) -> list[str]:
        """Extract normalized terms from query for matching.

        Splits query on whitespace and punctuation, lowercases,
        and filters out stopwords and very short terms.

        Args:
            query: Query string to extract terms from.

        Returns:
            List of normalized query terms.
        """
        # Common English stopwords to filter out
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "must",
            "that", "which", "who", "whom", "this", "these", "those",
            "it", "its", "as", "if", "when", "where", "how", "what", "why",
        }

        # Split on non-alphanumeric characters
        raw_terms = re.split(r"[^a-zA-Z0-9]+", query.lower())

        # Filter: remove stopwords and terms < 2 chars
        terms = [
            term for term in raw_terms
            if term and len(term) >= 2 and term not in stopwords
        ]

        return terms

    def _score_chunk(
        self,
        chunk: str,
        query_terms: list[str],
        term_frequencies: dict[str, int],
    ) -> float:
        """Score a chunk based on query term matches.

        Uses a term matching formula where each matched term contributes
        to the score with IDF-inspired weighting: rarer terms in the
        corpus contribute more to relevance.

        Formula: score += 1 / (1 + log(corpus_frequency + 1))

        Args:
            chunk: Text chunk to score.
            query_terms: Normalized query terms.
            term_frequencies: Term -> corpus count mapping.

        Returns:
            Relevance score (higher = more relevant).
        """
        chunk_lower = chunk.lower()
        score = 0.0

        for term in query_terms:
            if term in chunk_lower:
                corpus_freq = term_frequencies.get(term, 0)

                # IDF-inspired weighting: rarer terms score higher
                term_weight = 1.0 / (1.0 + math.log(corpus_freq + 1))
                score += term_weight

        return score

    def _score_by_position(
        self,
        chunks: list[str],
        max_snippets: int,
    ) -> list[tuple[str, int, float]]:
        """Score chunks by position (fallback for empty/short queries).

        Earlier chunks get higher scores, assuming important content
        tends to appear early in documents.

        Args:
            chunks: List of text chunks.
            max_snippets: Maximum snippets to return.

        Returns:
            List of (text, position, score) sorted by position.
        """
        total = len(chunks)
        results: list[tuple[str, int, float]] = []

        for idx, chunk in enumerate(chunks):
            # Score decreases linearly with position
            # First chunk = 1.0, last chunk = 1/total
            score = 1.0 - (idx / total) if total > 1 else 1.0
            results.append((chunk, idx, score))

        # Already sorted by position (ascending), take top N
        return results[:max_snippets]

    def _build_evidence_snippets(
        self,
        canonical_text: str,
        query: str,
        *,
        page_boundaries: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[EvidenceSnippet]:
        """Build evidence snippets with scoring and locators.

        Orchestrates the evidence extraction pipeline:
        1. Extract and score evidence chunks from canonical text
        2. Generate locators for each chunk
        3. Construct EvidenceSnippet objects with all metadata

        Args:
            canonical_text: Normalized source text.
            query: Research query for relevance scoring.
            page_boundaries: Optional PDF page boundaries for locators.

        Returns:
            List of EvidenceSnippet objects, limited by config.max_evidence_snippets.
        """
        if not self.config.include_evidence:
            return []

        # Extract evidence with relevance scoring
        evidence_tuples = self._extract_evidence(
            canonical_text,
            query,
            max_snippets=self.config.max_evidence_snippets,
        )

        if not evidence_tuples:
            return []

        # Generate locators in original text order to keep search positions valid,
        # then map back to relevance order.
        indexed_tuples = list(enumerate(evidence_tuples))
        indexed_tuples.sort(key=lambda item: item[1][1])  # sort by position index

        ordered_texts = [text for _, (text, _, _) in indexed_tuples]
        ordered_locators = self._generate_locators_batch(
            canonical_text,
            ordered_texts,
            page_boundaries=page_boundaries,
        )

        locators_by_index: list[tuple[str, int, int]] = [("char:0-0", 0, 0)] * len(
            evidence_tuples
        )
        for ordered_idx, (original_idx, _) in enumerate(indexed_tuples):
            locators_by_index[original_idx] = ordered_locators[ordered_idx]

        # Build EvidenceSnippet objects
        # Note: No truncation applied here - chunks already respect max_size (500)
        # from _chunk_text(). Display truncation is applied at render time per spec.
        snippets: list[EvidenceSnippet] = []
        for i, (text, _, score) in enumerate(evidence_tuples):
            locator_str, _, _ = locators_by_index[i]

            # Normalize score to 0.0-1.0 range
            # (scores from _extract_evidence may exceed 1.0)
            normalized_score = min(1.0, max(0.0, score))

            snippets.append(
                EvidenceSnippet(
                    text=text,
                    locator=locator_str,
                    relevance_score=normalized_score,
                )
            )

        return snippets

    def _generate_locator(
        self,
        canonical_text: str,
        snippet_text: str,
        search_start: int = 0,
        *,
        page_number: Optional[int] = None,
    ) -> tuple[str, int, int]:
        """Generate a locator string for a text snippet.

        Creates a locator that uniquely identifies the snippet's position
        within the canonical text. The locator format allows direct
        retrieval: canonical_text[start:end] == snippet_text.

        Locator formats:
        - Text: "char:{start}-{end}" (e.g., "char:100-250")
        - PDF: "page:{n}:char:{start}-{end}" (e.g., "page:3:char:100-250")

        Offset conventions:
        - start: 0-based index of first character
        - end: exclusive (Python slice convention)
        - Page numbers are 1-based (human-readable)

        Args:
            canonical_text: The normalized source text to search.
            snippet_text: The exact snippet text to locate.
            search_start: Position to start searching from (for efficiency
                when locating multiple snippets in order).
            page_number: Optional 1-based page number for PDF sources.
                If provided, generates page-prefixed locator.

        Returns:
            Tuple of (locator_string, start_offset, end_offset).
            If snippet not found, returns ("char:0-0", 0, 0).

        Examples:
            >>> text = "The quick brown fox jumps over the lazy dog."
            >>> locator, start, end = digestor._generate_locator(text, "brown fox")
            >>> locator
            'char:10-19'
            >>> text[start:end]
            'brown fox'

            >>> locator, _, _ = digestor._generate_locator(text, "fox", page_number=2)
            >>> locator
            'page:2:char:16-19'
        """
        # Find the snippet in the canonical text
        start = canonical_text.find(snippet_text, search_start)

        if start == -1:
            # Snippet not found - return null locator
            snippet_hash = hashlib.sha256(snippet_text.encode("utf-8")).hexdigest()[:8]
            logger.warning(
                "Snippet not found in canonical text (len=%d, hash=%s)",
                len(snippet_text),
                snippet_hash,
            )
            return ("char:0-0", 0, 0)

        end = start + len(snippet_text)

        # Build locator string
        if page_number is not None:
            locator = f"page:{page_number}:char:{start}-{end}"
        else:
            locator = f"char:{start}-{end}"

        return (locator, start, end)

    def _generate_locators_batch(
        self,
        canonical_text: str,
        snippets: list[str],
        *,
        page_boundaries: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[tuple[str, int, int]]:
        """Generate locators for multiple snippets efficiently.

        Processes snippets in order, using the previous end position as
        the search start for better performance on large texts.

        For PDF sources with page boundaries, automatically determines
        which page each snippet belongs to and includes it in the locator.

        Args:
            canonical_text: The normalized source text.
            snippets: List of snippet texts to locate.
            page_boundaries: Optional list of (page_num, start_char, end_char)
                tuples defining page boundaries in the canonical text.
                Page numbers should be 1-based.

        Returns:
            List of (locator, start, end) tuples, one per snippet.
            Order matches input snippets list.

        Examples:
            >>> locators = digestor._generate_locators_batch(
            ...     "First chunk. Second chunk. Third chunk.",
            ...     ["First chunk", "Second chunk", "Third chunk"]
            ... )
            >>> len(locators) == 3
            True
        """
        results: list[tuple[str, int, int]] = []
        search_pos = 0

        for snippet in snippets:
            # Determine page number if boundaries provided
            page_num = None
            if page_boundaries:
                # Find which page contains the expected position
                for pnum, pstart, pend in page_boundaries:
                    # First try to find snippet starting from search_pos
                    test_start = canonical_text.find(snippet, search_pos)
                    if test_start >= pstart and test_start < pend:
                        page_num = pnum
                        break

            locator, start, end = self._generate_locator(
                canonical_text,
                snippet,
                search_start=search_pos,
                page_number=page_num,
            )

            results.append((locator, start, end))

            # Update search position for next snippet (if found)
            if end > 0:
                search_pos = end

        return results
