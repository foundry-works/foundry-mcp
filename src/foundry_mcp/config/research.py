"""Research workflow configuration.

Contains ResearchConfig — the configuration dataclass for all research
workflows (CHAT, CONSENSUS, THINKDEEP, IDEATE, DEEP_RESEARCH).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from foundry_mcp.config.parsing import _parse_bool, _parse_provider_spec

if TYPE_CHECKING:
    from foundry_mcp.core.llm_config.provider_spec import ProviderSpec

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Configuration for research workflows (CHAT, CONSENSUS, THINKDEEP, IDEATE, DEEP_RESEARCH).

    Attributes:
        enabled: Master switch for research tools
        ttl_hours: Time-to-live for stored states in hours
        max_messages_per_thread: Maximum messages retained in a conversation thread
        default_provider: Default LLM provider for single-model workflows
        consensus_providers: List of provider IDs for CONSENSUS workflow
        thinkdeep_max_depth: Maximum investigation depth for THINKDEEP workflow
        ideate_perspectives: List of perspectives for IDEATE brainstorming
        default_timeout: Default timeout in seconds for provider calls (thinkdeep uses 2x)
        deep_research_max_iterations: Maximum refinement iterations for DEEP_RESEARCH
        deep_research_max_sub_queries: Maximum sub-queries for query decomposition
        deep_research_max_sources: Maximum sources per sub-query
        deep_research_follow_links: Whether to follow and extract content from links
        deep_research_timeout: Default timeout per operation in seconds
        deep_research_max_concurrent: Maximum concurrent operations
        deep_research_providers: Ordered list of search providers for deep research
        deep_research_audit_artifacts: Whether to write per-run audit artifacts
        search_rate_limit: Global rate limit for search APIs (requests per minute)
        max_concurrent_searches: Maximum concurrent search requests (for asyncio.Semaphore)
        per_provider_rate_limits: Per-provider rate limits in requests per minute
        tavily_api_key: API key for Tavily search provider (optional, reads from TAVILY_API_KEY env var)
        perplexity_api_key: API key for Perplexity Search (optional, reads from PERPLEXITY_API_KEY env var)
        google_api_key: API key for Google Custom Search (optional, reads from GOOGLE_API_KEY env var)
        google_cse_id: Google Custom Search Engine ID (optional, reads from GOOGLE_CSE_ID env var)
        semantic_scholar_api_key: API key for Semantic Scholar (optional, reads from SEMANTIC_SCHOLAR_API_KEY env var)
        tavily_search_depth: Tavily search depth ("basic", "advanced", "fast", "ultra_fast")
        tavily_topic: Tavily search topic ("general", "news")
        tavily_news_days: Days limit for news search (1-365, only for topic="news")
        tavily_include_images: Include image results in Tavily search
        tavily_country: ISO country code to boost results from (e.g., "US")
        tavily_chunks_per_source: Chunks per source for advanced search (1-5)
        tavily_auto_parameters: Let Tavily auto-configure parameters based on query
        tavily_extract_depth: Tavily extract depth ("basic", "advanced")
        tavily_extract_include_images: Include images in Tavily extract results
        perplexity_search_context_size: Perplexity context size ("low", "medium", "high")
        perplexity_max_tokens: Perplexity maximum tokens for response (default: 50000)
        perplexity_max_tokens_per_page: Perplexity maximum tokens per page (default: 2048)
        perplexity_recency_filter: Perplexity time filter ("day", "week", "month", "year")
        perplexity_country: Perplexity geographic filter (ISO 3166-1 alpha-2 code, e.g., "US")
        token_management_enabled: Master switch for token management features
        token_safety_margin: Fraction of budget to reserve as safety buffer (0.0-1.0)
        runtime_overhead: Tokens reserved for runtime overhead (e.g., Claude Code context)
        model_context_overrides: Per-model context/output limit overrides
        summarization_provider: Primary LLM provider for content summarization
        summarization_providers: Fallback providers for summarization (tried in order)
        summarization_timeout: Timeout per summarization request in seconds
        summarization_cache_enabled: Whether to cache summarization results
        allow_content_dropping: Allow dropping low-priority content when budget exhausted
        content_archive_enabled: Archive dropped/compressed content to disk
        content_archive_ttl_hours: TTL for archived content in hours (default: 168 = 7 days)
        research_archive_dir: Directory for content archive storage (default: research_dir/.archive)
        status_persistence_throttle_seconds: Minimum seconds between status saves (default: 5, 0 = always persist)
    """

    enabled: bool = True
    ttl_hours: int = 24
    max_messages_per_thread: int = 100
    default_provider: str = "gemini"
    consensus_providers: List[str] = field(default_factory=lambda: ["gemini", "claude"])
    thinkdeep_max_depth: int = 5
    ideate_perspectives: List[str] = field(default_factory=lambda: ["technical", "creative", "practical", "visionary"])
    default_timeout: float = 360.0  # 360 seconds default for AI CLI providers
    # Deep research clarification phase configuration
    deep_research_allow_clarification: bool = True
    deep_research_clarification_provider: Optional[str] = None  # Uses default_provider if not set

    # Deep research LLM-driven supervisor reflection
    deep_research_enable_reflection: bool = True  # Master switch for LLM reflection at phase boundaries
    deep_research_reflection_provider: Optional[str] = None  # Uses default_provider if not set
    deep_research_reflection_timeout: float = 60.0  # Timeout per reflection call (seconds)

    # Deep research contradiction detection in analysis phase
    deep_research_enable_contradiction_detection: bool = True  # LLM-based contradiction detection between findings

    # Deep research parallel topic researcher agents
    deep_research_enable_topic_agents: bool = True  # Master switch for per-topic ReAct loops in gathering
    deep_research_topic_max_searches: int = 3  # Max search iterations per topic (ReAct loop limit)
    deep_research_topic_reflection_provider: Optional[str] = None  # Uses default_provider if not set

    # Deep research configuration
    deep_research_max_iterations: int = 3
    deep_research_max_sub_queries: int = 5
    deep_research_max_sources: int = 5
    deep_research_follow_links: bool = True
    deep_research_timeout: float = 600.0  # Whole workflow timeout
    deep_research_max_concurrent: int = 3
    # Per-phase timeout overrides (seconds) - uses deep_research_timeout if not set
    deep_research_planning_timeout: float = 360.0
    deep_research_analysis_timeout: float = 360.0
    deep_research_synthesis_timeout: float = 600.0  # Synthesis may take longer
    deep_research_refinement_timeout: float = 360.0
    # Per-phase provider overrides - uses default_provider if not set
    deep_research_planning_provider: Optional[str] = None
    deep_research_analysis_provider: Optional[str] = None
    deep_research_synthesis_provider: Optional[str] = None
    deep_research_refinement_provider: Optional[str] = None
    # Per-phase fallback provider lists (for retry/fallback on failure)
    # On failure, tries next provider in the list until success or exhaustion
    deep_research_planning_providers: List[str] = field(default_factory=list)
    deep_research_analysis_providers: List[str] = field(default_factory=list)
    deep_research_synthesis_providers: List[str] = field(default_factory=list)
    deep_research_refinement_providers: List[str] = field(default_factory=list)
    # Retry settings for deep research phases
    deep_research_max_retries: int = 2  # Retry attempts per provider
    deep_research_retry_delay: float = 5.0  # Seconds between retries
    deep_research_providers: List[str] = field(default_factory=lambda: ["tavily", "google", "semantic_scholar"])
    deep_research_audit_artifacts: bool = True
    # Research mode: "general" | "academic" | "technical"
    deep_research_mode: str = "general"
    # Search rate limiting configuration
    search_rate_limit: int = 60  # requests per minute (global default)
    max_concurrent_searches: int = 3  # for asyncio.Semaphore in gathering phase
    per_provider_rate_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "tavily": 60,  # Tavily free tier: ~1 req/sec
            "perplexity": 60,  # Perplexity: ~1 req/sec (pricing: $5/1k requests)
            "google": 100,  # Google CSE: 100 queries/day free, ~100/min paid
            "semantic_scholar": 100,  # Semantic Scholar: 100 req/5min unauthenticated
        }
    )
    # Search provider API keys (all optional, read from env vars if not set)
    tavily_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    # Token management configuration
    token_management_enabled: bool = True  # Master switch for token management
    token_safety_margin: float = 0.15  # Fraction of budget to reserve as buffer
    runtime_overhead: int = 60000  # Tokens for Claude Code runtime overhead
    model_context_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Summarization configuration
    summarization_provider: Optional[str] = None  # Primary provider for summarization
    summarization_providers: List[str] = field(default_factory=list)  # Fallback providers
    summarization_timeout: float = 60.0  # Timeout per summarization request (seconds)
    summarization_cache_enabled: bool = True  # Cache summarization results
    # Content dropping and archival configuration
    allow_content_dropping: bool = False  # Allow dropping low-priority content
    content_archive_enabled: bool = False  # Archive dropped content to disk
    content_archive_ttl_hours: int = 168  # TTL for archived content (7 days)
    research_archive_dir: Optional[str] = None  # Directory for archive storage

    # Tavily search configuration
    tavily_search_depth: str = "basic"  # "basic", "advanced", "fast", "ultra_fast"
    tavily_topic: str = "general"  # "general", "news"
    tavily_news_days: Optional[int] = None  # 1-365, only for topic="news"
    tavily_include_images: bool = False
    tavily_country: Optional[str] = None  # ISO 3166-1 alpha-2 code (e.g., "US")
    tavily_chunks_per_source: int = 3  # 1-5, only for advanced search
    tavily_auto_parameters: bool = False  # Let Tavily auto-configure based on query
    # Internal flags to track explicit config overrides
    tavily_search_depth_configured: bool = field(default=False, init=False, repr=False)
    tavily_chunks_per_source_configured: bool = field(default=False, init=False, repr=False)

    # Tavily extract configuration
    tavily_extract_depth: str = "basic"  # "basic", "advanced"
    tavily_extract_include_images: bool = False
    # Tavily extract integration with deep research
    tavily_extract_in_deep_research: bool = False  # Enable extract as follow-up step
    tavily_extract_max_urls: int = 5  # Max URLs to extract per deep research run

    # Perplexity search configuration
    perplexity_search_context_size: str = "medium"  # "low", "medium", "high"
    perplexity_max_tokens: int = 50000  # Maximum tokens for response
    perplexity_max_tokens_per_page: int = 2048  # Maximum tokens per page
    perplexity_recency_filter: Optional[str] = None  # "day", "week", "month", "year"
    perplexity_country: Optional[str] = None  # ISO 3166-1 alpha-2 code (e.g., "US")

    # Semantic Scholar search configuration
    semantic_scholar_publication_types: Optional[List[str]] = None  # Filter by publication types
    semantic_scholar_sort_by: Optional[str] = None  # Sort field: citationCount, publicationDate, paperId
    semantic_scholar_sort_order: str = "desc"  # Sort direction: asc or desc
    semantic_scholar_use_extended_fields: bool = True  # Include TLDR and extended metadata

    # Stale task detection threshold for deep research background tasks
    deep_research_stale_task_seconds: float = 300.0  # Seconds of inactivity before a task is considered stale

    # Status persistence throttling (reduces disk I/O during deep research)
    status_persistence_throttle_seconds: int = 5  # Minimum seconds between status saves (0 = always persist)

    # Audit verbosity level for deep research artifact writes
    audit_verbosity: str = "full"  # "full" or "minimal" - controls JSONL audit payload size

    # Document digest configuration (for large content compression in deep research)
    deep_research_digest_policy: str = "auto"  # "off", "auto", "always", "proactive"
    deep_research_digest_min_chars: int = 10000  # Minimum chars before digest is applied
    deep_research_digest_max_sources: int = 8  # Max sources to digest per batch
    deep_research_digest_timeout: float = 120.0  # Timeout per digest operation (seconds)
    deep_research_digest_max_concurrent: int = 3  # Max concurrent digest operations
    deep_research_digest_include_evidence: bool = True  # Include evidence snippets
    deep_research_digest_evidence_max_chars: int = 400  # Max chars per evidence snippet
    deep_research_digest_max_evidence_snippets: int = 5  # Max evidence snippets per digest
    deep_research_digest_fetch_pdfs: bool = False  # Whether to fetch and extract PDF content
    deep_research_archive_content: bool = False  # Archive canonical text for digested sources
    deep_research_archive_retention_days: int = 30  # Days to retain archived digest content (0 = keep indefinitely)
    # Digest LLM provider configuration (uses analysis provider if not set)
    deep_research_digest_provider: Optional[str] = None  # Primary provider for digest
    deep_research_digest_providers: List[str] = field(default_factory=list)  # Fallback providers

    @classmethod
    def from_toml_dict(cls, data: Dict[str, Any]) -> "ResearchConfig":
        """Create config from TOML dict (typically [research] section).

        Args:
            data: Dict from TOML parsing

        Returns:
            ResearchConfig instance
        """
        # Parse consensus_providers - handle both string and list
        consensus_providers = data.get("consensus_providers", ["gemini", "claude"])
        if isinstance(consensus_providers, str):
            consensus_providers = [p.strip() for p in consensus_providers.split(",")]

        # Parse ideate_perspectives - handle both string and list
        ideate_perspectives = data.get("ideate_perspectives", ["technical", "creative", "practical", "visionary"])
        if isinstance(ideate_perspectives, str):
            ideate_perspectives = [p.strip() for p in ideate_perspectives.split(",")]

        # Parse deep_research_providers - handle both string and list
        deep_research_providers = data.get("deep_research_providers", ["tavily", "google", "semantic_scholar"])
        if isinstance(deep_research_providers, str):
            deep_research_providers = [p.strip() for p in deep_research_providers.split(",") if p.strip()]

        # Parse per-phase fallback provider lists
        def _parse_provider_list(key: str) -> List[str]:
            val = data.get(key, [])
            if isinstance(val, str):
                return [p.strip() for p in val.split(",") if p.strip()]
            return list(val) if val else []

        deep_research_planning_providers = _parse_provider_list("deep_research_planning_providers")
        deep_research_analysis_providers = _parse_provider_list("deep_research_analysis_providers")
        deep_research_synthesis_providers = _parse_provider_list("deep_research_synthesis_providers")
        deep_research_refinement_providers = _parse_provider_list("deep_research_refinement_providers")

        # Parse per_provider_rate_limits - handle dict from TOML
        per_provider_rate_limits = data.get(
            "per_provider_rate_limits",
            {
                "tavily": 60,
                "perplexity": 60,
                "google": 100,
                "semantic_scholar": 100,
            },
        )
        if isinstance(per_provider_rate_limits, dict):
            # Convert values to int
            per_provider_rate_limits = {k: int(v) for k, v in per_provider_rate_limits.items()}

        config = cls(
            enabled=_parse_bool(data.get("enabled", True)),
            ttl_hours=int(data.get("ttl_hours", 24)),
            max_messages_per_thread=int(data.get("max_messages_per_thread", 100)),
            default_provider=str(data.get("default_provider", "gemini")),
            consensus_providers=consensus_providers,
            thinkdeep_max_depth=int(data.get("thinkdeep_max_depth", 5)),
            ideate_perspectives=ideate_perspectives,
            default_timeout=float(data.get("default_timeout", 360.0)),
            # Deep research clarification phase
            deep_research_allow_clarification=_parse_bool(data.get("deep_research_allow_clarification", True)),
            deep_research_clarification_provider=data.get("deep_research_clarification_provider"),
            # Deep research LLM-driven reflection
            deep_research_enable_reflection=_parse_bool(data.get("deep_research_enable_reflection", True)),
            deep_research_reflection_provider=data.get("deep_research_reflection_provider"),
            deep_research_reflection_timeout=float(data.get("deep_research_reflection_timeout", 60.0)),
            # Deep research contradiction detection
            deep_research_enable_contradiction_detection=_parse_bool(data.get("deep_research_enable_contradiction_detection", True)),
            # Deep research parallel topic researcher agents
            deep_research_enable_topic_agents=_parse_bool(data.get("deep_research_enable_topic_agents", True)),
            deep_research_topic_max_searches=int(data.get("deep_research_topic_max_searches", 3)),
            deep_research_topic_reflection_provider=data.get("deep_research_topic_reflection_provider"),
            # Deep research configuration
            deep_research_max_iterations=int(data.get("deep_research_max_iterations", 3)),
            deep_research_max_sub_queries=int(data.get("deep_research_max_sub_queries", 5)),
            deep_research_max_sources=int(data.get("deep_research_max_sources", 5)),
            deep_research_follow_links=_parse_bool(data.get("deep_research_follow_links", True)),
            deep_research_timeout=float(data.get("deep_research_timeout", 600.0)),
            deep_research_max_concurrent=int(data.get("deep_research_max_concurrent", 3)),
            # Per-phase timeout overrides (match class defaults)
            deep_research_planning_timeout=float(data.get("deep_research_planning_timeout", 360.0)),
            deep_research_analysis_timeout=float(data.get("deep_research_analysis_timeout", 360.0)),
            deep_research_synthesis_timeout=float(data.get("deep_research_synthesis_timeout", 600.0)),
            deep_research_refinement_timeout=float(data.get("deep_research_refinement_timeout", 360.0)),
            # Per-phase provider overrides
            deep_research_planning_provider=data.get("deep_research_planning_provider"),
            deep_research_analysis_provider=data.get("deep_research_analysis_provider"),
            deep_research_synthesis_provider=data.get("deep_research_synthesis_provider"),
            deep_research_refinement_provider=data.get("deep_research_refinement_provider"),
            # Per-phase fallback provider lists
            deep_research_planning_providers=deep_research_planning_providers,
            deep_research_analysis_providers=deep_research_analysis_providers,
            deep_research_synthesis_providers=deep_research_synthesis_providers,
            deep_research_refinement_providers=deep_research_refinement_providers,
            # Retry settings
            deep_research_max_retries=int(data.get("deep_research_max_retries", 2)),
            deep_research_retry_delay=float(data.get("deep_research_retry_delay", 5.0)),
            deep_research_providers=deep_research_providers,
            deep_research_audit_artifacts=_parse_bool(data.get("deep_research_audit_artifacts", True)),
            # Research mode
            deep_research_mode=str(data.get("deep_research_mode", "general")),
            # Search rate limiting configuration
            search_rate_limit=int(data.get("search_rate_limit", 60)),
            max_concurrent_searches=int(data.get("max_concurrent_searches", 3)),
            per_provider_rate_limits=per_provider_rate_limits,
            # Search provider API keys (None means not set in TOML, will check env vars)
            tavily_api_key=data.get("tavily_api_key"),
            perplexity_api_key=data.get("perplexity_api_key"),
            google_api_key=data.get("google_api_key"),
            google_cse_id=data.get("google_cse_id"),
            semantic_scholar_api_key=data.get("semantic_scholar_api_key"),
            # Tavily search configuration
            tavily_search_depth=str(data.get("tavily_search_depth", "basic")),
            tavily_topic=str(data.get("tavily_topic", "general")),
            tavily_news_days=int(data["tavily_news_days"]) if data.get("tavily_news_days") is not None else None,
            tavily_include_images=_parse_bool(data.get("tavily_include_images", False)),
            tavily_country=data.get("tavily_country"),  # None or str
            tavily_chunks_per_source=int(data.get("tavily_chunks_per_source", 3)),
            tavily_auto_parameters=_parse_bool(data.get("tavily_auto_parameters", False)),
            # Tavily extract configuration
            tavily_extract_depth=str(data.get("tavily_extract_depth", "basic")),
            tavily_extract_include_images=_parse_bool(data.get("tavily_extract_include_images", False)),
            # Tavily extract in deep research
            tavily_extract_in_deep_research=_parse_bool(data.get("tavily_extract_in_deep_research", False)),
            tavily_extract_max_urls=int(data.get("tavily_extract_max_urls", 5)),
            # Perplexity search configuration
            perplexity_search_context_size=str(data.get("perplexity_search_context_size", "medium")),
            perplexity_max_tokens=int(data.get("perplexity_max_tokens", 50000)),
            perplexity_max_tokens_per_page=int(data.get("perplexity_max_tokens_per_page", 2048)),
            perplexity_recency_filter=data.get("perplexity_recency_filter"),  # None or str
            perplexity_country=data.get("perplexity_country"),  # None or str
            # Semantic Scholar search configuration
            semantic_scholar_publication_types=data.get("semantic_scholar_publication_types"),  # None or list
            semantic_scholar_sort_by=data.get("semantic_scholar_sort_by"),  # None or str
            semantic_scholar_sort_order=str(data.get("semantic_scholar_sort_order", "desc")),
            semantic_scholar_use_extended_fields=_parse_bool(data.get("semantic_scholar_use_extended_fields", True)),
            # Token management configuration
            token_management_enabled=_parse_bool(data.get("token_management_enabled", True)),
            token_safety_margin=float(data.get("token_safety_margin", 0.15)),
            runtime_overhead=int(data.get("runtime_overhead", 60000)),
            model_context_overrides=data.get("model_context_overrides", {}),
            # Summarization configuration
            summarization_provider=data.get("summarization_provider"),
            summarization_providers=_parse_provider_list("summarization_providers"),
            summarization_timeout=float(data.get("summarization_timeout", 60.0)),
            summarization_cache_enabled=_parse_bool(data.get("summarization_cache_enabled", True)),
            # Content dropping and archival configuration
            allow_content_dropping=_parse_bool(data.get("allow_content_dropping", False)),
            content_archive_enabled=_parse_bool(data.get("content_archive_enabled", False)),
            content_archive_ttl_hours=int(data.get("content_archive_ttl_hours", 168)),
            research_archive_dir=data.get("research_archive_dir"),
            # Stale task detection
            deep_research_stale_task_seconds=float(data.get("deep_research_stale_task_seconds", 300.0)),
            # Status persistence throttling
            status_persistence_throttle_seconds=int(data.get("status_persistence_throttle_seconds", 5)),
            # Audit verbosity
            audit_verbosity=str(data.get("audit_verbosity", "full")),
            # Document digest configuration
            deep_research_digest_policy=str(data.get("deep_research_digest_policy", "auto")),
            deep_research_digest_min_chars=int(data.get("deep_research_digest_min_chars", 10000)),
            deep_research_digest_max_sources=int(data.get("deep_research_digest_max_sources", 8)),
            deep_research_digest_timeout=float(data.get("deep_research_digest_timeout", 120.0)),
            deep_research_digest_max_concurrent=int(data.get("deep_research_digest_max_concurrent", 3)),
            deep_research_digest_include_evidence=_parse_bool(data.get("deep_research_digest_include_evidence", True)),
            deep_research_digest_evidence_max_chars=int(data.get("deep_research_digest_evidence_max_chars", 400)),
            deep_research_digest_max_evidence_snippets=int(data.get("deep_research_digest_max_evidence_snippets", 5)),
            deep_research_digest_fetch_pdfs=_parse_bool(data.get("deep_research_digest_fetch_pdfs", False)),
            deep_research_archive_content=_parse_bool(data.get("deep_research_archive_content", False)),
            deep_research_archive_retention_days=int(data.get("deep_research_archive_retention_days", 30)),
            deep_research_digest_provider=data.get("deep_research_digest_provider"),
            deep_research_digest_providers=_parse_provider_list("deep_research_digest_providers"),
        )
        config.tavily_search_depth_configured = "tavily_search_depth" in data
        config.tavily_chunks_per_source_configured = "tavily_chunks_per_source" in data
        return config

    def __post_init__(self) -> None:
        """Validate configuration fields after initialization."""
        self._validate_tavily_config()
        self._validate_perplexity_config()
        self._validate_semantic_scholar_config()
        self._validate_status_persistence_config()
        self._validate_audit_verbosity_config()
        self._validate_digest_config()

    def _validate_tavily_config(self) -> None:
        """Validate all Tavily configuration fields.

        Raises:
            ValueError: If any Tavily config field has an invalid value.
        """
        import re

        # Validate search_depth
        valid_search_depths = {"basic", "advanced", "fast", "ultra_fast"}
        if self.tavily_search_depth not in valid_search_depths:
            raise ValueError(
                f"Invalid tavily_search_depth: {self.tavily_search_depth!r}. "
                f"Must be one of: {sorted(valid_search_depths)}"
            )

        # Validate topic
        valid_topics = {"general", "news"}
        if self.tavily_topic not in valid_topics:
            raise ValueError(f"Invalid tavily_topic: {self.tavily_topic!r}. Must be one of: {sorted(valid_topics)}")

        # Validate news_days (1-365 or None)
        if self.tavily_news_days is not None:
            if not isinstance(self.tavily_news_days, int) or self.tavily_news_days < 1 or self.tavily_news_days > 365:
                raise ValueError(
                    f"Invalid tavily_news_days: {self.tavily_news_days!r}. Must be an integer between 1 and 365."
                )

        # Validate country (ISO 3166-1 alpha-2 or None)
        if self.tavily_country is not None:
            if not isinstance(self.tavily_country, str) or not re.match(r"^[A-Z]{2}$", self.tavily_country):
                raise ValueError(
                    f"Invalid tavily_country: {self.tavily_country!r}. "
                    "Must be a 2-letter uppercase ISO 3166-1 alpha-2 code (e.g., 'US', 'GB')."
                )

        # Validate chunks_per_source (1-5)
        if (
            not isinstance(self.tavily_chunks_per_source, int)
            or self.tavily_chunks_per_source < 1
            or self.tavily_chunks_per_source > 5
        ):
            raise ValueError(
                f"Invalid tavily_chunks_per_source: {self.tavily_chunks_per_source!r}. "
                "Must be an integer between 1 and 5."
            )

        # Validate extract_depth
        valid_extract_depths = {"basic", "advanced"}
        if self.tavily_extract_depth not in valid_extract_depths:
            raise ValueError(
                f"Invalid tavily_extract_depth: {self.tavily_extract_depth!r}. "
                f"Must be one of: {sorted(valid_extract_depths)}"
            )

    def _validate_perplexity_config(self) -> None:
        """Validate all Perplexity configuration fields.

        Raises:
            ValueError: If any Perplexity config field has an invalid value.
        """
        import re

        # Validate search_context_size
        valid_context_sizes = {"low", "medium", "high"}
        if self.perplexity_search_context_size not in valid_context_sizes:
            raise ValueError(
                f"Invalid perplexity_search_context_size: {self.perplexity_search_context_size!r}. "
                f"Must be one of: {sorted(valid_context_sizes)}"
            )

        # Validate max_tokens (positive integer)
        if not isinstance(self.perplexity_max_tokens, int) or self.perplexity_max_tokens < 1:
            raise ValueError(
                f"Invalid perplexity_max_tokens: {self.perplexity_max_tokens!r}. Must be a positive integer."
            )

        # Validate max_tokens_per_page (positive integer)
        if not isinstance(self.perplexity_max_tokens_per_page, int) or self.perplexity_max_tokens_per_page < 1:
            raise ValueError(
                f"Invalid perplexity_max_tokens_per_page: {self.perplexity_max_tokens_per_page!r}. "
                "Must be a positive integer."
            )

        # Validate recency_filter (day/week/month/year or None)
        if self.perplexity_recency_filter is not None:
            valid_recency_filters = {"day", "week", "month", "year"}
            if self.perplexity_recency_filter not in valid_recency_filters:
                raise ValueError(
                    f"Invalid perplexity_recency_filter: {self.perplexity_recency_filter!r}. "
                    f"Must be one of: {sorted(valid_recency_filters)} or None."
                )

        # Validate country (ISO 3166-1 alpha-2 or None)
        if self.perplexity_country is not None:
            if not isinstance(self.perplexity_country, str) or not re.match(r"^[A-Z]{2}$", self.perplexity_country):
                raise ValueError(
                    f"Invalid perplexity_country: {self.perplexity_country!r}. "
                    "Must be a 2-letter uppercase ISO 3166-1 alpha-2 code (e.g., 'US', 'GB')."
                )

    def _validate_semantic_scholar_config(self) -> None:
        """Validate all Semantic Scholar configuration fields.

        Raises:
            ValueError: If any Semantic Scholar config field has an invalid value.
        """
        # Valid publication types from Semantic Scholar API
        valid_publication_types = {
            "Review",
            "JournalArticle",
            "Conference",
            "CaseReport",
            "ClinicalTrial",
            "Dataset",
            "Editorial",
            "LettersAndComments",
            "MetaAnalysis",
            "News",
            "Study",
            "Book",
            "BookSection",
        }

        # Validate publication_types (list of valid types or None)
        if self.semantic_scholar_publication_types is not None:
            if not isinstance(self.semantic_scholar_publication_types, list):
                raise ValueError(
                    f"Invalid semantic_scholar_publication_types: {self.semantic_scholar_publication_types!r}. "
                    "Must be a list of publication types or None."
                )
            invalid_types = set(self.semantic_scholar_publication_types) - valid_publication_types
            if invalid_types:
                raise ValueError(
                    f"Invalid semantic_scholar_publication_types: {sorted(invalid_types)}. "
                    f"Must be from: {sorted(valid_publication_types)}"
                )

        # Valid sort fields
        valid_sort_fields = {"paperId", "publicationDate", "citationCount"}

        # Validate sort_by (valid field or None)
        if self.semantic_scholar_sort_by is not None:
            if self.semantic_scholar_sort_by not in valid_sort_fields:
                raise ValueError(
                    f"Invalid semantic_scholar_sort_by: {self.semantic_scholar_sort_by!r}. "
                    f"Must be one of: {sorted(valid_sort_fields)} or None."
                )

        # Validate sort_order (asc or desc)
        valid_sort_orders = {"asc", "desc"}
        if self.semantic_scholar_sort_order not in valid_sort_orders:
            raise ValueError(
                f"Invalid semantic_scholar_sort_order: {self.semantic_scholar_sort_order!r}. "
                f"Must be one of: {sorted(valid_sort_orders)}"
            )

    def _validate_status_persistence_config(self) -> None:
        """Validate status persistence configuration fields.

        Raises:
            ValueError: If status_persistence_throttle_seconds is negative.
        """
        if self.status_persistence_throttle_seconds < 0:
            raise ValueError(
                f"Invalid status_persistence_throttle_seconds: "
                f"{self.status_persistence_throttle_seconds!r}. "
                "Must be >= 0 (0 means always persist, positive values set "
                "minimum seconds between status saves)."
            )

    def _validate_audit_verbosity_config(self) -> None:
        """Validate audit verbosity configuration field.

        Raises:
            ValueError: If audit_verbosity has an invalid value.
        """
        valid_verbosity_levels = {"full", "minimal"}
        if self.audit_verbosity not in valid_verbosity_levels:
            raise ValueError(
                f"Invalid audit_verbosity: {self.audit_verbosity!r}. Must be one of: {sorted(valid_verbosity_levels)}"
            )

    def _validate_digest_config(self) -> None:
        """Validate document digest configuration fields.

        Raises:
            ValueError: If any digest config field has an invalid value.
        """
        # Validate digest_policy
        valid_policies = {"off", "auto", "always", "proactive"}
        if self.deep_research_digest_policy not in valid_policies:
            raise ValueError(
                f"Invalid deep_research_digest_policy: {self.deep_research_digest_policy!r}. "
                f"Must be one of: {sorted(valid_policies)}"
            )

        # Validate min_chars (must be positive)
        if self.deep_research_digest_min_chars < 0:
            raise ValueError(
                f"Invalid deep_research_digest_min_chars: {self.deep_research_digest_min_chars!r}. Must be >= 0."
            )

        # Validate max_sources (must be positive)
        if self.deep_research_digest_max_sources < 1:
            raise ValueError(
                f"Invalid deep_research_digest_max_sources: {self.deep_research_digest_max_sources!r}. Must be >= 1."
            )

        # Validate timeout (must be positive)
        if self.deep_research_digest_timeout <= 0:
            raise ValueError(
                f"Invalid deep_research_digest_timeout: {self.deep_research_digest_timeout!r}. Must be > 0."
            )

        # Validate max_concurrent (must be positive)
        if self.deep_research_digest_max_concurrent < 1:
            raise ValueError(
                f"Invalid deep_research_digest_max_concurrent: {self.deep_research_digest_max_concurrent!r}. "
                "Must be >= 1."
            )

        # Validate evidence_max_chars (must be positive)
        if self.deep_research_digest_evidence_max_chars < 1:
            raise ValueError(
                f"Invalid deep_research_digest_evidence_max_chars: {self.deep_research_digest_evidence_max_chars!r}. "
                "Must be >= 1."
            )

        # Validate max_evidence_snippets (must be positive)
        if self.deep_research_digest_max_evidence_snippets < 1:
            raise ValueError(
                f"Invalid deep_research_digest_max_evidence_snippets: {self.deep_research_digest_max_evidence_snippets!r}. "
                "Must be >= 1."
            )

        # Validate retention days (0 means keep indefinitely)
        if self.deep_research_archive_retention_days < 0:
            raise ValueError(
                f"Invalid deep_research_archive_retention_days: {self.deep_research_archive_retention_days!r}. "
                "Must be >= 0."
            )

    def get_provider_rate_limit(self, provider: str) -> int:
        """Get rate limit for a specific provider.

        Returns the provider-specific rate limit if configured,
        otherwise falls back to the global search_rate_limit.

        Args:
            provider: Provider name (e.g., "tavily", "google", "semantic_scholar")

        Returns:
            Rate limit in requests per minute
        """
        return self.per_provider_rate_limits.get(provider, self.search_rate_limit)

    def get_phase_timeout(self, phase: str) -> float:
        """Get timeout for a specific deep research phase.

        Returns the phase-specific timeout if configured, otherwise
        falls back to deep_research_timeout.

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement", "gathering")

        Returns:
            Timeout in seconds for the phase
        """
        phase_timeouts = {
            "clarification": self.deep_research_planning_timeout,  # Reuse planning timeout
            "planning": self.deep_research_planning_timeout,
            "analysis": self.deep_research_analysis_timeout,
            "synthesis": self.deep_research_synthesis_timeout,
            "refinement": self.deep_research_refinement_timeout,
            "gathering": self.deep_research_timeout,  # Gathering uses default
        }
        return phase_timeouts.get(phase.lower(), self.deep_research_timeout)

    def get_phase_provider(self, phase: str) -> str:
        """Get LLM provider ID for a specific deep research phase.

        Returns the phase-specific provider if configured, otherwise
        falls back to default_provider. Supports both simple names ("gemini")
        and ProviderSpec format ("[cli]gemini:pro").

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            Provider ID for the phase (e.g., "gemini", "opencode")
        """
        provider_id, _ = self.resolve_phase_provider(phase)
        return provider_id

    def resolve_phase_provider(self, phase: str) -> Tuple[str, Optional[str]]:
        """Resolve provider ID and model for a deep research phase.

        Parses ProviderSpec format ("[cli]gemini:pro") or simple names ("gemini").
        Returns (provider_id, model) tuple for use with the provider registry.

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            Tuple of (provider_id, model) where model may be None
        """
        phase_providers = {
            "planning": self.deep_research_planning_provider,
            "analysis": self.deep_research_analysis_provider,
            "synthesis": self.deep_research_synthesis_provider,
            "refinement": self.deep_research_refinement_provider,
        }
        configured = phase_providers.get(phase.lower())
        spec_str = configured or self.default_provider
        return _parse_provider_spec(spec_str)

    def get_phase_fallback_providers(self, phase: str) -> List[str]:
        """Get fallback provider list for a specific deep research phase.

        Returns the phase-specific fallback provider list if configured,
        otherwise returns an empty list (no fallback).

        Args:
            phase: Phase name ("planning", "analysis", "synthesis", "refinement")

        Returns:
            List of fallback provider IDs to try on failure
        """
        phase_fallbacks = {
            "planning": self.deep_research_planning_providers,
            "analysis": self.deep_research_analysis_providers,
            "synthesis": self.deep_research_synthesis_providers,
            "refinement": self.deep_research_refinement_providers,
        }
        return phase_fallbacks.get(phase.lower(), [])

    def get_reflection_provider(self) -> str:
        """Get LLM provider ID for supervisor reflection calls.

        Returns the reflection-specific provider if configured, otherwise
        falls back to default_provider.

        Returns:
            Provider ID for reflection calls
        """
        if self.deep_research_reflection_provider:
            provider_id, _ = _parse_provider_spec(self.deep_research_reflection_provider)
            return provider_id
        provider_id, _ = _parse_provider_spec(self.default_provider)
        return provider_id

    def get_digest_provider(self, analysis_provider: Optional[str] = None) -> str:
        """Get LLM provider ID for document digest operations.

        Returns the digest-specific provider if configured, otherwise
        falls back to analysis_provider (if provided) or default_provider.

        Args:
            analysis_provider: Optional analysis provider to use as fallback

        Returns:
            Provider ID for digest operations (e.g., "gemini", "opencode")
        """
        if self.deep_research_digest_provider:
            provider_id, _ = _parse_provider_spec(self.deep_research_digest_provider)
            return provider_id
        if analysis_provider:
            return analysis_provider
        provider_id, _ = _parse_provider_spec(self.default_provider)
        return provider_id

    def get_digest_fallback_providers(self) -> List[str]:
        """Get fallback provider list for document digest operations.

        Returns the digest-specific fallback provider list if configured,
        otherwise returns an empty list (no fallback).

        Returns:
            List of fallback provider IDs to try on failure
        """
        return self.deep_research_digest_providers

    def get_search_provider_api_key(
        self,
        provider: str,
        required: bool = True,
    ) -> Optional[str]:
        """Get API key for a search provider with fallback to environment variables.

        Checks config value first, then falls back to environment variable.
        Raises ValueError with clear error message if required and not found.

        Args:
            provider: Provider name ("tavily", "google", "semantic_scholar")
            required: If True, raises ValueError when key is missing (default: True)

        Returns:
            API key string, or None if not required and not found

        Raises:
            ValueError: If required=True and no API key is found

        Example:
            # Get Tavily API key (will raise if missing)
            api_key = config.research.get_search_provider_api_key("tavily")

            # Get Semantic Scholar API key (optional, returns None if missing)
            api_key = config.research.get_search_provider_api_key(
                "semantic_scholar", required=False
            )
        """
        # Map provider names to config attributes and env vars
        provider_config = {
            "tavily": {
                "config_key": "tavily_api_key",
                "env_var": "TAVILY_API_KEY",
                "setup_url": "https://tavily.com/",
            },
            "perplexity": {
                "config_key": "perplexity_api_key",
                "env_var": "PERPLEXITY_API_KEY",
                "setup_url": "https://docs.perplexity.ai/",
            },
            "google": {
                "config_key": "google_api_key",
                "env_var": "GOOGLE_API_KEY",
                "setup_url": "https://console.cloud.google.com/apis/credentials",
            },
            "google_cse": {
                "config_key": "google_cse_id",
                "env_var": "GOOGLE_CSE_ID",
                "setup_url": "https://cse.google.com/",
            },
            "semantic_scholar": {
                "config_key": "semantic_scholar_api_key",
                "env_var": "SEMANTIC_SCHOLAR_API_KEY",
                "setup_url": "https://www.semanticscholar.org/product/api",
            },
        }

        provider_lower = provider.lower()
        if provider_lower not in provider_config:
            raise ValueError(
                f"Unknown search provider: '{provider}'. Valid providers: {', '.join(provider_config.keys())}"
            )

        config_info = provider_config[provider_lower]
        config_key = config_info["config_key"]
        env_var = config_info["env_var"]

        # Check config value first
        api_key = getattr(self, config_key, None)

        # Fall back to environment variable
        if not api_key:
            api_key = os.environ.get(env_var)

        # Handle missing key
        if not api_key:
            if required:
                raise ValueError(
                    f"{provider.title()} API key not configured. "
                    f"Set via {env_var} environment variable or "
                    f"'research.{config_key}' in foundry-mcp.toml. "
                    f"Get an API key at: {config_info['setup_url']}"
                )
            return None

        return api_key

    def get_google_credentials(self, required: bool = True) -> tuple[Optional[str], Optional[str]]:
        """Get both Google API key and CSE ID for Google Custom Search.

        Convenience method that retrieves both required credentials for
        Google Custom Search API.

        Args:
            required: If True, raises ValueError when either credential is missing

        Returns:
            Tuple of (api_key, cse_id)

        Raises:
            ValueError: If required=True and either credential is missing
        """
        api_key = self.get_search_provider_api_key("google", required=required)
        cse_id = self.get_search_provider_api_key("google_cse", required=required)
        return api_key, cse_id

    def get_default_provider_spec(self) -> "ProviderSpec":
        """Parse default_provider into a ProviderSpec."""
        from foundry_mcp.core.llm_config.provider_spec import ProviderSpec

        return ProviderSpec.parse_flexible(self.default_provider)

    def get_consensus_provider_specs(self) -> List["ProviderSpec"]:
        """Parse consensus_providers into ProviderSpec list."""
        from foundry_mcp.core.llm_config.provider_spec import ProviderSpec

        return [ProviderSpec.parse_flexible(p) for p in self.consensus_providers]

    def get_model_context_override(
        self,
        provider: str,
        model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get context/output limit overrides for a specific model.

        Looks up overrides in model_context_overrides using the format:
        - "{provider}" for provider-wide overrides
        - "{provider}:{model}" for model-specific overrides (takes precedence)

        Args:
            provider: Provider identifier (e.g., "claude", "gemini")
            model: Optional model identifier (e.g., "opus", "flash")

        Returns:
            Dict with override values (context_window, max_output_tokens, etc.)
            or None if no overrides configured

        Example:
            # Config in TOML:
            # [research.model_context_overrides."claude:opus"]
            # context_window = 150000
            # max_output_tokens = 16000

            overrides = config.research.get_model_context_override("claude", "opus")
            # Returns {"context_window": 150000, "max_output_tokens": 16000}
        """
        if not self.model_context_overrides:
            return None

        # Try model-specific key first (e.g., "claude:opus")
        if model:
            model_key = f"{provider}:{model}"
            if model_key in self.model_context_overrides:
                return self.model_context_overrides[model_key]

        # Fall back to provider-wide key (e.g., "claude")
        if provider in self.model_context_overrides:
            return self.model_context_overrides[provider]

        return None

    def get_summarization_provider_chain(self) -> List[str]:
        """Get the ordered list of summarization providers to try.

        Returns providers in order: primary provider first (if set),
        then fallback providers, with duplicates removed.

        Returns:
            List of provider IDs to try for summarization
        """
        chain: List[str] = []
        seen: set[str] = set()

        if self.summarization_provider:
            chain.append(self.summarization_provider)
            seen.add(self.summarization_provider)

        for provider in self.summarization_providers:
            if provider not in seen:
                chain.append(provider)
                seen.add(provider)

        return chain

    def get_archive_dir(self, research_dir: Optional[Path] = None) -> Path:
        """Get the resolved content archive directory path.

        Priority:
        1. Explicitly configured research_archive_dir
        2. Default: research_dir/.archive

        Args:
            research_dir: Optional research directory to use for default path.
                         If not provided, uses specs/.research/.archive

        Returns:
            Path to content archive directory
        """
        if self.research_archive_dir:
            return Path(self.research_archive_dir).expanduser()

        # Fall back to default: research_dir/.archive
        base_research = research_dir or Path("specs/.research")
        return base_research / ".archive"
