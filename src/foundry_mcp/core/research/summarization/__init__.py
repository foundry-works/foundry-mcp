"""Content summarization utilities for deep research workflows.

Provides LLM-based content compression with configurable summarization levels,
provider chain with fallback, retry logic, and caching support.

Key Components:
    - SummarizationLevel: Enum defining compression levels (RAW to HEADLINE)
    - ContentSummarizer: Main class for summarizing content with provider chain

Usage:
    from foundry_mcp.core.research.summarization import (
        ContentSummarizer,
        SummarizationLevel,
    )

    # Create summarizer with provider configuration
    summarizer = ContentSummarizer(
        summarization_provider="claude",
        summarization_providers=["gemini", "codex"],
    )

    # Summarize content
    result = await summarizer.summarize(
        content="Long article text...",
        level=SummarizationLevel.KEY_POINTS,
    )
"""

from foundry_mcp.core.errors.research import (
    ProviderExhaustedError,
    SummarizationError,
    SummarizationValidationError,
)

from .cache import SummaryCache
from .constants import (
    _SUMMARY_CACHE_MAX_SIZE,
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MAX_RETRIES,
    RETRY_DELAY,
)
from .models import (
    SummarizationConfig,
    SummarizationFunc,
    SummarizationLevel,
    SummarizationResult,
)
from .summarizer import ContentSummarizer

__all__ = [
    # Constants
    "CHARS_PER_TOKEN",
    "CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "MAX_RETRIES",
    "RETRY_DELAY",
    "_SUMMARY_CACHE_MAX_SIZE",
    # Models
    "SummarizationConfig",
    "SummarizationFunc",
    "SummarizationLevel",
    "SummarizationResult",
    # Cache
    "SummaryCache",
    # Summarizer
    "ContentSummarizer",
    # Errors (re-exported for convenience)
    "ProviderExhaustedError",
    "SummarizationError",
    "SummarizationValidationError",
]
