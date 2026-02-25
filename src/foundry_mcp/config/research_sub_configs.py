"""Nested sub-config dataclasses for ResearchConfig.

These provide typed, organized views over the flat ResearchConfig fields.
They are used as return types for ResearchConfig's grouped property
accessors (e.g. ``config.tavily_config``, ``config.perplexity_config``).

New code should prefer accessing configuration through these sub-configs
for clarity, while the flat fields on ResearchConfig remain for backward
compatibility.

Introduced in Phase 3 (PA.1) to address ResearchConfig's growing field
count (~80+ fields across search providers, deep research phases, model
routing, and digest settings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TavilyConfig:
    """Tavily search and extract configuration.

    Groups all ``tavily_*`` fields from ResearchConfig into a single
    typed config object.
    """

    api_key: Optional[str] = None
    search_depth: str = "basic"
    topic: str = "general"
    news_days: Optional[int] = None
    include_images: bool = False
    country: Optional[str] = None
    chunks_per_source: int = 3
    auto_parameters: bool = False
    extract_depth: str = "basic"
    extract_include_images: bool = False


@dataclass(frozen=True)
class PerplexityConfig:
    """Perplexity search configuration.

    Groups all ``perplexity_*`` fields from ResearchConfig.
    """

    api_key: Optional[str] = None
    search_context_size: str = "medium"
    max_tokens: int = 50000
    max_tokens_per_page: int = 2048
    recency_filter: Optional[str] = None
    country: Optional[str] = None


@dataclass(frozen=True)
class SemanticScholarConfig:
    """Semantic Scholar search configuration.

    Groups all ``semantic_scholar_*`` fields from ResearchConfig.
    """

    api_key: Optional[str] = None
    publication_types: Optional[List[str]] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    use_extended_fields: bool = True


@dataclass(frozen=True)
class ModelRoleConfig:
    """Role-based model routing configuration (Phase 6).

    Groups all ``deep_research_*_provider`` / ``deep_research_*_model``
    fields that control cost-optimised routing of LLM calls to different
    models based on task role (research, report, reflection, etc.).

    **Cost-tier defaults:** High-volume, low-complexity roles (summarization,
    compression) automatically use a cheap model (``2.0-flash``) when no
    explicit model is configured.  This mirrors ODR's pattern of routing
    summarization to ~10x cheaper models.  See
    ``ResearchConfig._COST_TIER_MODEL_DEFAULTS`` for the mapping.
    """

    research_provider: Optional[str] = None
    research_model: Optional[str] = None
    report_provider: Optional[str] = None
    report_model: Optional[str] = None
    reflection_provider: Optional[str] = None
    reflection_model: Optional[str] = None
    supervision_provider: Optional[str] = None
    supervision_model: Optional[str] = None
    topic_reflection_provider: Optional[str] = None
    topic_reflection_model: Optional[str] = None
    clarification_provider: Optional[str] = None
    clarification_model: Optional[str] = None
    compression_provider: Optional[str] = None
    compression_model: Optional[str] = None
    summarization_provider: Optional[str] = None
    summarization_model: Optional[str] = None
    digest_provider: Optional[str] = None
    digest_providers: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DeepResearchConfig:
    """Deep research workflow configuration.

    Groups workflow settings and tuning knobs from the ``deep_research_*``
    namespace on ResearchConfig.
    """

    # Clarification / supervision toggles
    allow_clarification: bool = True
    enable_supervision: bool = True

    # Supervision settings
    max_supervision_rounds: int = 6
    supervision_min_sources_per_query: int = 2

    # Core workflow settings
    max_iterations: int = 3
    max_sub_queries: int = 5
    max_sources: int = 5
    follow_links: bool = True
    timeout: float = 600.0
    max_concurrent: int = 3
    mode: str = "general"
    audit_artifacts: bool = True

    # Topic agent settings
    topic_max_searches: int = 10  # backward-compat alias for topic_max_tool_calls
    enable_content_dedup: bool = True
    content_dedup_threshold: float = 0.8

    # Per-phase timeouts
    planning_timeout: float = 360.0
    synthesis_timeout: float = 600.0

    # Per-phase provider overrides
    planning_provider: Optional[str] = None
    synthesis_provider: Optional[str] = None

    # Per-phase fallback provider lists
    planning_providers: List[str] = field(default_factory=list)
    synthesis_providers: List[str] = field(default_factory=list)

    # Retry settings
    max_retries: int = 2
    retry_delay: float = 5.0

    # Search providers
    providers: List[str] = field(default_factory=lambda: ["tavily", "google", "semantic_scholar"])

    # Stale task detection
    stale_task_seconds: float = 300.0

    # Document digest configuration
    digest_min_chars: int = 10000
    digest_max_sources: int = 8
    digest_timeout: float = 120.0
    digest_max_concurrent: int = 3
    digest_include_evidence: bool = True
    digest_evidence_max_chars: int = 400
    digest_max_evidence_snippets: int = 5
    digest_fetch_pdfs: bool = False
    archive_content: bool = False
    archive_retention_days: int = 30

    # Evaluation configuration
    evaluation_provider: Optional[str] = None
    evaluation_model: Optional[str] = None
    evaluation_timeout: float = 360.0
