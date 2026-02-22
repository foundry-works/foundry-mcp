"""Fragment ID utilities and digest models for deep research.

Provides stable fragment ID generation/parsing for chunked content tracking,
and Pydantic models for compressed document digests.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Fragment ID Utilities
# =============================================================================


def make_fragment_id(base_id: str, fragment_index: int) -> str:
    """Generate a stable fragment ID for chunked content.

    Creates a predictable ID for content fragments by appending a
    fragment index to the base item ID. This enables tracking fidelity
    at the chunk level while maintaining parent item relationships.

    Args:
        base_id: Base item ID (e.g., "src-abc123")
        fragment_index: Zero-based index of the fragment/chunk

    Returns:
        Fragment ID in format "{base_id}#fragment-{N}"

    Examples:
        >>> make_fragment_id("src-abc123", 0)
        'src-abc123#fragment-0'
        >>> make_fragment_id("src-abc123", 3)
        'src-abc123#fragment-3'
    """
    return f"{base_id}#fragment-{fragment_index}"


def parse_fragment_id(fragment_id: str) -> tuple[str, Optional[int]]:
    """Parse a fragment ID into base ID and fragment index.

    Extracts the base item ID and optional fragment index from a
    fragment ID. If the ID doesn't contain a fragment suffix, returns
    the original ID with None for the fragment index.

    Args:
        fragment_id: ID that may contain fragment suffix

    Returns:
        Tuple of (base_id, fragment_index) where fragment_index is
        None if no fragment suffix was present

    Examples:
        >>> parse_fragment_id("src-abc123#fragment-0")
        ('src-abc123', 0)
        >>> parse_fragment_id("src-abc123")
        ('src-abc123', None)
    """
    if "#fragment-" not in fragment_id:
        return fragment_id, None

    base_id, suffix = fragment_id.rsplit("#fragment-", 1)
    try:
        fragment_index = int(suffix)
        return base_id, fragment_index
    except ValueError:
        # Invalid fragment suffix, return original as-is
        return fragment_id, None


def is_fragment_id(item_id: str) -> bool:
    """Check if an ID is a fragment ID.

    Args:
        item_id: ID to check

    Returns:
        True if the ID contains a fragment suffix

    Examples:
        >>> is_fragment_id("src-abc123#fragment-0")
        True
        >>> is_fragment_id("src-abc123")
        False
    """
    _, fragment_index = parse_fragment_id(item_id)
    return fragment_index is not None


def get_base_id(item_id: str) -> str:
    """Get the base ID from a potentially fragment ID.

    Strips the fragment suffix if present, returning the original
    item ID.

    Args:
        item_id: ID that may contain fragment suffix

    Returns:
        Base item ID without fragment suffix

    Examples:
        >>> get_base_id("src-abc123#fragment-0")
        'src-abc123'
        >>> get_base_id("src-abc123")
        'src-abc123'
    """
    base_id, _ = parse_fragment_id(item_id)
    return base_id


# =============================================================================
# Digest Models (Document compression for deep research)
# =============================================================================


class EvidenceSnippet(BaseModel):
    """A text snippet extracted from source content for citation support.

    Evidence snippets preserve exact substrings from the canonical text
    along with locators that enable verification and citation generation.
    The locator format varies by content type (HTML/text vs PDF).

    Locator Formats:
        - HTML/Text: "char:{start}-{end}" (e.g., "char:1500-1800")
        - PDF: "page:{n}:char:{start}-{end}" (e.g., "page:3:char:200-450")
        - PDF (no page): "char:{start}-{end}" (fallback if page detection fails)

    Indexing Semantics:
        - Start/end are 0-based character positions
        - End boundary is exclusive (Python slice semantics)
        - Page numbers are 1-based
        - Offsets reference canonical (normalized) text

    Attributes:
        text: Exact substring from canonical text (max 500 chars).
              No truncation markers - display formatting applied at render time.
        locator: Position reference in format appropriate to content type.
        relevance_score: Query relevance score from 0.0 (irrelevant) to 1.0 (highly relevant).
    """

    text: str = Field(
        ...,
        max_length=500,
        description="Exact substring from canonical text for citation",
    )
    locator: str = Field(
        ...,
        description="Position reference (e.g., 'char:1500-1800' or 'page:3:char:200-450')",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Query relevance score from 0.0 to 1.0",
    )


class DigestPayload(BaseModel):
    """Structured digest of document content for deep research.

    DigestPayload v1.0 is the on-wire format for compressed document content.
    It replaces raw source text with a structured summary, key points, and
    evidence snippets while preserving citation traceability.

    The payload is self-describing via `content_type` and `query_hash` fields,
    allowing consumers to validate and process it without surrounding metadata.

    Query Conditioning:
        Digests are query-conditioned - the summary focus and evidence selection
        depend on the research query. The `query_hash` field (8-char hex) enables
        cache invalidation when the query changes.

    Storage:
        - Serialized as JSON string in `source.content`
        - `source.content_type` set to "digest/v1"
        - When archival enabled, `source_text_hash` matches archived canonical text

    Archival Contract (when deep_research_archive_content=true):
        - Path: `{archive_dir}/{source_id}/{source_text_hash}.txt`
        - Archive dir default: `~/.foundry-mcp/research_archives/`
        - Format: UTF-8 encoded canonical text (post-normalization)
        - Retention: 30 days default (configurable via deep_research_archive_retention_days)
        - `source_text_hash` is computed BEFORE archival from canonical text
        - Evidence snippet locators reference offsets in the archived canonical text
        - Traceability: `archived_text[start:end] == snippet.text` when archive exists
        - `source.metadata["_digest_archive_hash"]` tracks linkage to archive

    Consumer Rules:
        1. Detect via `source.content_type == "digest/v1"`
        2. Parse `source.content` as JSON, validate against schema
        3. SKIP further summarization (already compressed)
        4. Use `evidence_snippets` for citations
        5. Use `digest_chars` for token budget estimation

    Attributes:
        version: Schema version, always "1.0" for this version.
        content_type: Self-describing type identifier, always "digest/v1".
        query_hash: 8-character hex hash of the research query for cache keying.
        summary: Condensed summary of source content (max 2000 chars).
        key_points: Extracted key points as bullet items (max 10, each max 500 chars).
        evidence_snippets: Relevant text excerpts with locators (max 10).
        original_chars: Character count of original source before digest.
        digest_chars: Character count of digest output (for budget estimation).
        compression_ratio: Ratio of digest_chars to original_chars (0.0 to 1.0).
        source_text_hash: SHA256 hash of canonical text, prefixed with "sha256:".
    """

    version: str = Field(
        default="1.0",
        description="Schema version",
    )
    content_type: str = Field(
        default="digest/v1",
        description="Self-describing content type identifier",
    )
    query_hash: str = Field(
        ...,
        min_length=8,
        max_length=8,
        pattern=r"^[a-f0-9]{8}$",
        description="8-character hex hash of the research query",
    )
    summary: str = Field(
        ...,
        max_length=2000,
        description="Condensed summary of source content",
    )
    key_points: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Extracted key points (max 10 items, each max 500 chars)",
    )
    evidence_snippets: list[EvidenceSnippet] = Field(
        default_factory=list,
        max_length=10,
        description="Relevant text excerpts with locators for citation (max 10)",
    )
    original_chars: int = Field(
        ...,
        ge=0,
        description="Character count of original source before digest",
    )
    digest_chars: int = Field(
        ...,
        ge=0,
        description="Character count of digest output",
    )
    compression_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio of digest_chars to original_chars",
    )
    source_text_hash: str = Field(
        ...,
        pattern=r"^sha256:[a-f0-9]{64}$",
        description="SHA256 hash of canonical text, prefixed with 'sha256:'",
    )

    @field_validator("key_points")
    @classmethod
    def validate_key_points_length(cls, v: list[str]) -> list[str]:
        """Validate each key point does not exceed 500 characters."""
        for i, point in enumerate(v):
            if len(point) > 500:
                raise ValueError(f"key_points[{i}] exceeds maximum length of 500 characters (got {len(point)})")
        return v

    @property
    def is_valid_digest(self) -> bool:
        """Check if this is a valid v1.0 digest payload."""
        return self.version == "1.0" and self.content_type == "digest/v1"

    def to_json(self) -> str:
        """Serialize to JSON string for storage in source.content."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "DigestPayload":
        """Deserialize from JSON string stored in source.content."""
        return cls.model_validate_json(json_str)
