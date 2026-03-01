"""Source and finding models for deep research workflows."""

import hashlib
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, model_validator

from foundry_mcp.core.research.models.enums import ConfidenceLevel


class SourceType(str, Enum):
    """Types of research sources that can be discovered.

    V1 Implementation:
    - WEB: General web search results (via Tavily/Google)
    - ACADEMIC: Academic papers and journals (via Semantic Scholar)

    Future Extensions (placeholders):
    - EXPERT: Expert profiles and interviews (reserved)
    - CODE: Code repositories and examples (reserved for GitHub search)
    - NEWS: News articles and press releases
    - DOCUMENTATION: Technical documentation
    """

    WEB = "web"
    ACADEMIC = "academic"
    EXPERT = "expert"  # Future: expert profiles, interviews
    CODE = "code"  # Future: GitHub, code search


class SourceQuality(str, Enum):
    """Quality assessment for research sources.

    Quality levels are assigned during the ANALYSIS phase based on:
    - Source authority and credibility
    - Content recency and relevance
    - Citation count and peer review status (for academic)
    - Domain reputation (for web sources)
    """

    UNKNOWN = "unknown"  # Not yet assessed
    LOW = "low"  # Questionable reliability
    MEDIUM = "medium"  # Generally reliable
    HIGH = "high"  # Authoritative source


class ResearchMode(str, Enum):
    """Research modes that control source prioritization.

    Each mode applies different domain-based quality heuristics:
    - GENERAL: No domain preferences, balanced approach (default)
    - ACADEMIC: Prioritizes journals, publishers, preprints
    - TECHNICAL: Prioritizes official docs, arxiv, code repositories
    """

    GENERAL = "general"
    ACADEMIC = "academic"
    TECHNICAL = "technical"


# Domain tier lists for source quality assessment by research mode
# Patterns support wildcards: "*.edu" matches any .edu domain
DOMAIN_TIERS: dict[str, dict[str, list[str]]] = {
    "academic": {
        "high": [
            # Aggregators & indexes
            "scholar.google.com",
            "semanticscholar.org",
            "pubmed.gov",
            "ncbi.nlm.nih.gov",
            "jstor.org",
            # Major publishers
            "springer.com",
            "link.springer.com",
            "sciencedirect.com",
            "elsevier.com",
            "wiley.com",
            "onlinelibrary.wiley.com",
            "tandfonline.com",  # Taylor & Francis
            "sagepub.com",
            "nature.com",
            "science.org",  # AAAS/Science
            "frontiersin.org",
            "plos.org",
            "journals.plos.org",
            "mdpi.com",
            "oup.com",
            "academic.oup.com",  # Oxford
            "cambridge.org",
            # Preprints & open access
            "arxiv.org",
            "biorxiv.org",
            "medrxiv.org",
            "psyarxiv.com",
            "ssrn.com",
            # Field-specific
            "apa.org",
            "psycnet.apa.org",  # Psychology
            "aclanthology.org",  # Computational linguistics
            # CS/Tech academic
            "acm.org",
            "dl.acm.org",
            "ieee.org",
            "ieeexplore.ieee.org",
            # Institutional patterns
            "*.edu",
            "*.ac.uk",
            "*.edu.au",
        ],
        "low": [
            "reddit.com",
            "quora.com",
            "medium.com",
            "linkedin.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "pinterest.com",
            "instagram.com",
            "tiktok.com",
            "youtube.com",  # Can have good content but inconsistent
        ],
    },
    "technical": {
        "high": [
            # Preprints (technical papers)
            "arxiv.org",
            # Official documentation patterns
            "docs.*",
            "developer.*",
            "*.dev",
            "devdocs.io",
            # Code & technical resources
            "github.com",
            "stackoverflow.com",
            "stackexchange.com",
            # Language/framework official sites
            "python.org",
            "docs.python.org",
            "nodejs.org",
            "rust-lang.org",
            "doc.rust-lang.org",
            "go.dev",
            "typescriptlang.org",
            "react.dev",
            "vuejs.org",
            "angular.io",
            # Cloud providers
            "aws.amazon.com",
            "cloud.google.com",
            "docs.microsoft.com",
            "learn.microsoft.com",
            "azure.microsoft.com",
            # Tech company engineering blogs
            "engineering.fb.com",
            "netflixtechblog.com",
            "uber.com/blog/engineering",
            "blog.google",
            # Academic (relevant for technical research)
            "acm.org",
            "dl.acm.org",
            "ieee.org",
            "ieeexplore.ieee.org",
        ],
        "low": [
            "reddit.com",
            "quora.com",
            "linkedin.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "pinterest.com",
        ],
    },
    "general": {
        "high": [],  # No domain preferences
        "low": [
            # Still deprioritize social media
            "pinterest.com",
            "facebook.com",
            "instagram.com",
            "tiktok.com",
        ],
    },
}


class SubQuery(BaseModel):
    """A decomposed sub-query for focused research.

    During the PLANNING phase, the original research query is decomposed
    into multiple focused sub-queries. Each sub-query targets a specific
    aspect of the research question and can be executed independently
    during the GATHERING phase.

    Status transitions:
    - pending -> executing -> completed (success path)
    - pending -> executing -> failed (error path)
    """

    id: str = Field(default_factory=lambda: f"subq-{uuid4().hex[:8]}")
    query: str = Field(..., description="The focused sub-query text")
    rationale: Optional[str] = Field(
        default=None,
        description="Why this sub-query was generated and what aspect it covers",
    )
    priority: int = Field(
        default=1,
        description="Execution priority (1=highest, larger=lower priority)",
    )
    status: str = Field(
        default="pending",
        description="Current status: pending, executing, completed, failed",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of ResearchSource objects found for this query",
    )
    findings_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of what was found from this sub-query",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None)
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_completed(self, findings: Optional[str] = None) -> None:
        """Mark this sub-query as successfully completed.

        Args:
            findings: Optional summary of findings from this sub-query
        """
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)
        if findings:
            self.findings_summary = findings

    def mark_failed(self, error: str) -> None:
        """Mark this sub-query as failed with an error message.

        Args:
            error: Description of why the sub-query failed
        """
        self.status = "failed"
        self.completed_at = datetime.now(timezone.utc)
        self.error = error


class ResearchSource(BaseModel):
    """A source discovered during research.

    Sources are collected during the GATHERING phase when sub-queries
    are executed against search providers. Each source represents a
    piece of external content (web page, paper, etc.) that may contain
    relevant information for the research query.

    Quality is assessed during the ANALYSIS phase based on source
    authority, content relevance, and other factors.
    """

    id: str = Field(default_factory=lambda: f"src-{uuid4().hex[:8]}")
    url: Optional[str] = Field(
        default=None,
        description="URL of the source (may be None for non-web sources)",
    )
    title: str = Field(..., description="Title or headline of the source")
    source_type: SourceType = Field(
        default=SourceType.WEB,
        description="Type of source (web, academic, etc.)",
    )
    quality: SourceQuality = Field(
        default=SourceQuality.UNKNOWN,
        description="Assessed quality level of this source",
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Brief excerpt or description from the source",
    )
    content: Optional[str] = Field(
        default=None,
        description="Full extracted content (if follow_links enabled)",
    )
    raw_content: Optional[str] = Field(
        default=None,
        description="Original unprocessed content before fetch-time summarization. "
        "Preserved for fallback and audit when summarization is active.",
    )
    content_type: str = Field(
        default="text/plain",
        description="Content type identifier (e.g., 'text/plain', 'digest/v1')",
    )
    sub_query_id: Optional[str] = Field(
        default=None,
        description="ID of the SubQuery that discovered this source",
    )
    citation_number: Optional[int] = Field(
        default=None,
        description="Stable 1-indexed citation number assigned when the source enters state",
    )
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_digest(self) -> bool:
        """Check if this source contains a DigestPayload.

        Returns True if content_type is 'digest/v1', indicating the content
        field contains a serialized DigestPayload JSON string rather than
        raw text.

        Consumers should check this property before processing content to
        determine whether to parse as DigestPayload or treat as raw text.
        """
        return self.content_type == "digest/v1"

    def _content_hash(self) -> str:
        """Generate a hash of the source content for cache keying.

        Returns the first 32 characters of the SHA-256 hex digest,
        providing 128 bits of collision resistance. This hash is
        deterministic for the same content and can be used as a
        cache key for token count caching across sessions.

        Returns:
            32-character hex string. Returns hash of empty string
            if content is None or empty.
        """
        content = self.content or ""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def _token_cache_key(self, provider: str, model: str) -> str:
        """Generate a cache key for token count lookup.

        The key format includes content hash, content length, provider,
        and model to ensure uniqueness. Content length provides additional
        collision protection beyond the 32-char hash.

        Args:
            provider: Provider ID (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4", "claude-3")

        Returns:
            Cache key in format "{hash_32}:{length}:{provider}:{model}"
        """
        content_len = len(self.content) if self.content else 0
        return f"{self._content_hash()}:{content_len}:{provider}:{model}"

    def _get_cached_token_count(self, provider: str, model: str) -> Optional[int]:
        """Retrieve cached token count for this source.

        Looks up the token count in the internal _token_cache metadata
        field. Returns None if no cache exists or if the key is not found.

        Args:
            provider: Provider ID
            model: Model name

        Returns:
            Cached token count, or None if not cached
        """
        cache = self.metadata.get("_token_cache")
        if not cache or cache.get("v") != 1:
            return None
        key = self._token_cache_key(provider, model)
        return cache.get("counts", {}).get(key)

    def _set_cached_token_count(self, provider: str, model: str, count: int) -> None:
        """Store token count in the internal cache.

        Initializes the cache structure if needed and stores the count
        under the appropriate key. The cache uses version 1 schema with
        underscore prefix to mark it as internal.

        Schema: metadata['_token_cache'] = {
            'v': 1,
            'counts': {'{hash_32}:{len}:{provider}:{model}': count, ...}
        }

        Args:
            provider: Provider ID
            model: Model name
            count: Token count to cache
        """
        if "_token_cache" not in self.metadata:
            self.metadata["_token_cache"] = {"v": 1, "counts": {}}
        cache = self.metadata["_token_cache"]
        if "counts" not in cache:
            cache["counts"] = {}
        key = self._token_cache_key(provider, model)
        cache["counts"][key] = count

    def public_metadata(self) -> dict[str, Any]:
        """Return metadata with internal fields excluded.

        Filters out metadata keys starting with underscore (e.g., _token_cache)
        for API responses. Internal fields are still persisted to state files
        via model_dump().

        Returns:
            Dict with internal fields (underscore-prefixed keys) removed.
        """
        return {k: v for k, v in self.metadata.items() if not k.startswith("_")}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict with internal fields filtered out.

        Returns a dict suitable for API responses and external consumption.
        Filters out:
        - Internal metadata keys (underscore-prefixed, e.g., _raw_content,
          _token_cache, _digest_archive_hash)
        - raw_content (large, internal — use model_dump() for full serialization)

        For full serialization including internal fields, use model_dump().

        Returns:
            Dict with internal metadata fields removed.
        """
        data = self.model_dump()
        # Replace metadata with filtered version
        data["metadata"] = self.public_metadata()
        # Exclude raw_content from API responses (large, internal)
        data.pop("raw_content", None)
        return data


class ResearchFinding(BaseModel):
    """A key finding extracted from research sources.

    Findings are extracted during the ANALYSIS phase by examining
    source content and identifying key insights. Each finding has
    an associated confidence level and links back to supporting sources.

    Findings are organized by category/theme during synthesis to
    create a structured report.
    """

    id: str = Field(default_factory=lambda: f"find-{uuid4().hex[:8]}")
    content: str = Field(..., description="The key finding or insight")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.MEDIUM,
        description="Confidence level in this finding",
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of ResearchSource objects supporting this finding",
    )
    sub_query_id: Optional[str] = Field(
        default=None,
        description="ID of SubQuery that produced this finding",
    )
    category: Optional[str] = Field(
        default=None,
        description="Theme or category for organizing findings",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchGap(BaseModel):
    """An identified gap in the research requiring follow-up.

    Gaps are identified during the ANALYSIS and SYNTHESIS phases when
    the research reveals missing information or unanswered questions.
    Each gap includes suggested follow-up queries that can be used
    in subsequent refinement iterations.

    Gaps drive the REFINEMENT phase: if unresolved gaps exist and
    max_iterations hasn't been reached, the workflow loops back
    to GATHERING with new sub-queries derived from gap suggestions.
    """

    id: str = Field(default_factory=lambda: f"gap-{uuid4().hex[:8]}")
    description: str = Field(
        ...,
        description="Description of the knowledge gap or missing information",
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Follow-up queries that could fill this gap",
    )
    priority: int = Field(
        default=1,
        description="Priority for follow-up (1=highest, larger=lower priority)",
    )
    resolved: bool = Field(
        default=False,
        description="Whether this gap has been addressed in a refinement iteration",
    )
    resolution_notes: Optional[str] = Field(
        default=None,
        description="Notes on how the gap was resolved",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Methodology Quality Assessment (PLAN-4 Item 3)
# ---------------------------------------------------------------------------


class StudyDesign(str, Enum):
    """Study design classification for methodology assessment.

    Covers the standard hierarchy of evidence from meta-analyses down to
    expert opinion.  Used by ``MethodologyAssessor`` to label each source
    with a study-design tag that the synthesis LLM can use for qualitative
    weighting.
    """

    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    RCT = "randomized_controlled_trial"
    QUASI_EXPERIMENTAL = "quasi_experimental"
    COHORT = "cohort_study"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    QUALITATIVE = "qualitative"
    CASE_STUDY = "case_study"
    THEORETICAL = "theoretical"
    OPINION = "expert_opinion"
    UNKNOWN = "unknown"


class MethodologyAssessment(BaseModel):
    """Structured methodology metadata extracted from a research source.

    Produces approximate heuristics — **no numeric rigor score**.  Provides
    structured metadata to the synthesis LLM for qualitative judgment.

    Confidence is forced to ``"low"`` when the assessment is based only on
    the abstract (``content_basis == "abstract"``).
    """

    source_id: str = Field(..., description="ID of the assessed ResearchSource")
    study_design: StudyDesign = Field(
        default=StudyDesign.UNKNOWN,
        description="Classified study design (e.g. RCT, cohort, qualitative)",
    )
    sample_size: Optional[int] = Field(
        default=None,
        description="Reported sample size (N), if extractable",
    )
    sample_description: Optional[str] = Field(
        default=None,
        description="Brief description of the sample/participants",
    )
    effect_size: Optional[str] = Field(
        default=None,
        description="Reported effect size (e.g. 'd=0.45', 'OR=2.3')",
    )
    statistical_significance: Optional[str] = Field(
        default=None,
        description="Reported statistical significance (e.g. 'p<0.001')",
    )
    limitations_noted: list[str] = Field(
        default_factory=list,
        description="Limitations acknowledged or detected",
    )
    potential_biases: list[str] = Field(
        default_factory=list,
        description="Potential biases identified",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="low",
        description="Extraction confidence: 'high', 'medium', or 'low'",
    )
    content_basis: str = Field(
        default="abstract",
        description="Content used for assessment: 'abstract' or 'full_text'",
    )

    @model_validator(mode="after")
    def _enforce_abstract_confidence(self) -> "MethodologyAssessment":
        """Force confidence to 'low' when assessment is based on abstract only."""
        if self.content_basis == "abstract" and self.confidence != "low":
            logger.warning(
                "Downgrading confidence from '%s' to 'low' for abstract-based assessment (source %s)",
                self.confidence,
                self.source_id,
            )
            self.confidence = "low"
        return self
