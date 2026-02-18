"""Constants for content summarization configuration."""

from __future__ import annotations

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAY = 3.0  # seconds

# Chunking configuration
DEFAULT_CHUNK_SIZE = 8000  # tokens (conservative for most models)
CHUNK_OVERLAP = 200  # tokens overlap between chunks
CHARS_PER_TOKEN = 4  # approximate for heuristic estimation

# Cache configuration
_SUMMARY_CACHE_MAX_SIZE = 1000  # Maximum cached summaries
