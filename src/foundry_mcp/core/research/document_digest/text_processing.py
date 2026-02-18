"""Text processing mixin for document digest.

Provides text normalization, canonicalization, and chunking logic
used by DocumentDigestor.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextProcessingMixin:
    """Mixin providing text normalization and chunking for DocumentDigestor."""

    def _normalize_text(self, text: str) -> str:
        """Normalize text to canonical form.

        Applies a deterministic normalization pipeline to ensure consistent
        hashing and text processing. The pipeline is designed to be
        idempotent - applying it multiple times produces the same result.

        Normalization steps (in order):
        1. HTML entity decoding (&amp; -> &, &lt; -> <, etc.)
        2. HTML tag stripping (removes <tag> and </tag>)
        3. Unicode normalization to NFC form
        4. Whitespace collapse (multiple spaces/newlines -> single space)

        Args:
            text: Raw text to normalize.

        Returns:
            Normalized canonical text suitable for hashing and evidence extraction.

        Examples:
            >>> digestor._normalize_text("Hello&nbsp;World")
            'Hello World'
            >>> digestor._normalize_text("<p>Hello</p> <b>World</b>")
            'Hello World'
            >>> digestor._normalize_text("Hello\\n\\n\\nWorld")
            'Hello World'
        """
        return self._canonicalize_text(text)

    def _canonicalize_pages(
        self,
        text: str,
        page_boundaries: list[tuple[int, int, int]],
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Canonicalize text while preserving PDF page boundary mapping.

        Args:
            text: Raw source text.
            page_boundaries: List of (page_num, start, end) offsets into raw text.

        Returns:
            Tuple of (canonical_text, canonical_page_boundaries).
        """
        canonical_pages: list[str] = []
        canonical_bounds: list[tuple[int, int, int]] = []
        cursor = 0

        for page_num, start, end in page_boundaries:
            page_text = text[start:end]
            page_canonical = self._canonicalize_text(page_text)

            if canonical_pages:
                cursor += 2  # Account for "\n\n" separator between pages

            page_start = cursor
            page_end = page_start + len(page_canonical)
            canonical_bounds.append((page_num, page_start, page_end))
            canonical_pages.append(page_canonical)
            cursor = page_end

        canonical_text = "\n\n".join(canonical_pages)
        return canonical_text, canonical_bounds

    def _canonicalize_text(self, text: str) -> str:
        """Apply canonical text normalization pipeline.

        This is the core normalization implementation. The method is separate
        from _normalize_text to allow direct access for testing while
        maintaining the existing public interface.

        Normalization pipeline:
        1. Decode HTML entities (&amp; -> &, &lt; -> <, &nbsp; -> space, etc.)
        2. Strip HTML tags (both opening and closing)
        3. Normalize Unicode to NFC form (composed characters)
        4. Collapse whitespace (multiple spaces/newlines/tabs -> single space)
        5. Strip leading/trailing whitespace

        Args:
            text: Raw text to normalize.

        Returns:
            Canonical text form.
        """
        if not text:
            return ""

        # Step 1: Decode HTML entities
        # Handles &amp; &lt; &gt; &quot; &nbsp; and numeric entities like &#39;
        result = html.unescape(text)

        # Step 2: Strip HTML tags
        # Simple regex that handles <tag>, </tag>, <tag attr="value">, etc.
        result = re.sub(r"<[^>]+>", " ", result)

        # Step 3: Unicode normalization to NFC
        # NFC is the canonical form for text comparison
        # Composes characters (e.g., 'e' as single codepoint vs e + combining accent)
        result = unicodedata.normalize("NFC", result)

        # Step 4: Collapse whitespace
        # Replace all whitespace sequences (spaces, tabs, newlines) with single space
        result = re.sub(r"\s+", " ", result)

        # Step 5: Strip leading/trailing whitespace
        result = result.strip()

        return result

    def _chunk_text(
        self,
        text: str,
        *,
        target_size: int = 400,
        max_size: int = 500,
        min_size: int = 50,
    ) -> list[str]:
        """Chunk text into segments for evidence extraction.

        Splits text into chunks using boundary-aware logic that respects
        natural text boundaries when possible. Chunks target a specific
        size but will extend to reach a clean boundary up to max_size.
        Small trailing chunks below min_size are merged with the previous.

        Boundary detection priority (highest to lowest):
        1. Paragraph boundaries (double newline or blank line)
        2. Sentence boundaries (. ! ? followed by space or end)
        3. Clause boundaries (, ; : followed by space)
        4. Word boundaries (space)
        5. Hard cut (last resort at max_size)

        Args:
            text: Text to chunk.
            target_size: Target chunk size in characters. Default 400.
            max_size: Maximum chunk size before hard cut. Default 500.
            min_size: Minimum chunk size; smaller chunks merge. Default 50.

        Returns:
            List of text chunks. May be empty if input is empty/whitespace.

        Examples:
            >>> digestor._chunk_text("Short text")
            ['Short text']
            >>> chunks = digestor._chunk_text("First paragraph.\\n\\nSecond paragraph.")
            >>> len(chunks) >= 1
            True
        """
        if not text or not text.strip():
            return []

        # Ensure text is normalized (no leading/trailing whitespace)
        text = text.strip()

        # If text fits within target, return as single chunk
        if len(text) <= target_size:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            # If remaining text fits in target, add it and stop
            if len(remaining) <= target_size:
                chunks.append(remaining)
                break

            # Find the best boundary within max_size
            chunk_end = self._find_chunk_boundary(
                remaining,
                target_size=target_size,
                max_size=max_size,
            )

            # Extract chunk and strip
            chunk = remaining[:chunk_end].strip()
            remaining = remaining[chunk_end:].strip()

            if chunk:
                chunks.append(chunk)

        # Merge small final chunk with previous if below min_size
        if len(chunks) >= 2 and len(chunks[-1]) < min_size:
            merged = chunks[-2] + " " + chunks[-1]
            # Only merge if result doesn't exceed max_size
            if len(merged) <= max_size:
                chunks[-2] = merged
                chunks.pop()

        return chunks

    def _find_chunk_boundary(
        self,
        text: str,
        *,
        target_size: int,
        max_size: int,
    ) -> int:
        """Find the best boundary position for chunking.

        Searches for natural text boundaries starting from target_size
        up to max_size. Returns the position immediately after the
        boundary marker (so the marker is included in the chunk).

        Boundary priority:
        1. Paragraph (\\n\\n) - look backward from target first
        2. Sentence (. ! ?) - followed by space or at end
        3. Clause (, ; :) - followed by space
        4. Word (space)
        5. Hard cut at max_size

        Args:
            text: Text to find boundary in.
            target_size: Start searching from this position.
            max_size: Maximum position (hard cut fallback).

        Returns:
            Position to cut at (exclusive).
        """
        # Clamp max_size to actual text length
        effective_max = min(max_size, len(text))
        effective_target = min(target_size, len(text))

        # Priority 1: Paragraph boundary (double newline)
        # Look backward from target first, then forward to max
        para_pos = self._find_boundary_bidirectional(
            text,
            patterns=["\n\n", "\r\n\r\n"],
            target=effective_target,
            max_pos=effective_max,
        )
        if para_pos > 0:
            return para_pos

        # Priority 2: Sentence boundary (. ! ? followed by space or at end)
        sent_pos = self._find_sentence_boundary(
            text,
            target=effective_target,
            max_pos=effective_max,
        )
        if sent_pos > 0:
            return sent_pos

        # Priority 3: Clause boundary (; : , followed by space)
        clause_pos = self._find_boundary_bidirectional(
            text,
            patterns=["; ", ": ", ", "],
            target=effective_target,
            max_pos=effective_max,
            include_pattern=True,
        )
        if clause_pos > 0:
            return clause_pos

        # Priority 4: Word boundary (space)
        word_pos = self._find_boundary_bidirectional(
            text,
            patterns=[" "],
            target=effective_target,
            max_pos=effective_max,
            include_pattern=False,
        )
        if word_pos > 0:
            return word_pos

        # Priority 5: Hard cut at max_size
        return effective_max

    def _find_boundary_bidirectional(
        self,
        text: str,
        patterns: list[str],
        target: int,
        max_pos: int,
        include_pattern: bool = True,
    ) -> int:
        """Find boundary pattern, searching backward from target then forward.

        Args:
            text: Text to search.
            patterns: Pattern strings to look for.
            target: Start position for search.
            max_pos: Maximum position to search forward.
            include_pattern: If True, include pattern length in result.

        Returns:
            Position after boundary, or 0 if not found.
        """
        best_backward = 0
        best_forward = 0

        for pattern in patterns:
            # Search backward from target
            backward = text.rfind(pattern, 0, target)
            if backward > best_backward:
                if include_pattern:
                    best_backward = backward + len(pattern)
                else:
                    best_backward = backward

            # Search forward from target to max_pos
            forward = text.find(pattern, target, max_pos)
            if forward > 0 and (best_forward == 0 or forward < best_forward):
                if include_pattern:
                    best_forward = forward + len(pattern)
                else:
                    best_forward = forward

        # Prefer backward result if found and reasonably close to target
        # (within 100 chars), otherwise take forward if available
        if best_backward > 0 and target - best_backward <= 100:
            return best_backward
        if best_forward > 0:
            return best_forward
        if best_backward > 0:
            return best_backward

        return 0

    def _find_sentence_boundary(
        self,
        text: str,
        target: int,
        max_pos: int,
    ) -> int:
        """Find sentence boundary (. ! ? followed by space or at end).

        Handles edge cases like abbreviations by requiring space after
        punctuation (except at text end).

        Args:
            text: Text to search.
            target: Start position for search.
            max_pos: Maximum position.

        Returns:
            Position after sentence end, or 0 if not found.
        """
        sentence_markers = ".!?"

        # Search backward from target
        best_backward = 0
        for i in range(target - 1, -1, -1):
            if text[i] in sentence_markers:
                # Check if followed by space or at end
                if i + 1 >= len(text) or text[i + 1] in " \n\t":
                    best_backward = i + 1
                    break

        # Search forward from target to max_pos
        best_forward = 0
        for i in range(target, min(max_pos, len(text))):
            if text[i] in sentence_markers:
                # Check if followed by space or at end
                if i + 1 >= len(text) or text[i + 1] in " \n\t":
                    best_forward = i + 1
                    break

        # Prefer backward if reasonably close (within 100 chars)
        if best_backward > 0 and target - best_backward <= 100:
            return best_backward
        if best_forward > 0:
            return best_forward
        if best_backward > 0:
            return best_backward

        return 0
