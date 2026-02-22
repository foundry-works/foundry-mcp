"""Token estimation with fallback chain and caching.

Provides:
    - estimate_tokens(): Token estimation with provider/tiktoken/heuristic fallback
    - clear_token_cache(): Clear the estimation cache
    - get_cache_stats(): Get cache statistics
    - register_provider_tokenizer(): Register provider-specific tokenizers
    - TokenCountEstimateWarning: Warning for heuristic fallback
"""

import hashlib
import logging
import threading
import warnings
from functools import lru_cache
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Optional tiktoken import for accurate token counting
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None  # type: ignore
    _TIKTOKEN_AVAILABLE = False


# Cache for token estimates: maps (content_hash, provider) -> token_count
_TOKEN_ESTIMATE_CACHE: dict[tuple[str, str], int] = {}
_TOKEN_CACHE_LOCK = threading.Lock()

# Maximum cache size to prevent unbounded memory growth
_MAX_CACHE_SIZE = 10_000


class TokenCountEstimateWarning(UserWarning):
    """Warning emitted when using character-based heuristic for token estimation."""

    pass


# Provider-specific tokenizer factories (for future extension)
_PROVIDER_TOKENIZERS: dict[str, Callable[[str], int]] = {}


def register_provider_tokenizer(provider: str, tokenizer: Callable[[str], int]) -> None:
    """Register a provider-specific tokenizer function.

    Args:
        provider: Provider identifier (e.g., "claude", "gemini")
        tokenizer: Function that takes content string and returns token count

    Example:
        def my_tokenizer(content: str) -> int:
            return len(my_api.count_tokens(content))
        register_provider_tokenizer("my_provider", my_tokenizer)
    """
    _PROVIDER_TOKENIZERS[provider.lower()] = tokenizer


def _content_hash(content: str) -> str:
    """Generate a hash of content for cache keying.

    Uses SHA-256 truncated to 16 characters for reasonable uniqueness
    while keeping cache keys compact.

    Args:
        content: Text content to hash

    Returns:
        Hex string hash of the content
    """
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]


@lru_cache(maxsize=32)
def _get_cached_encoding(model_name: str) -> Any:
    """Get a cached tiktoken encoding for the given model name.

    Uses lru_cache to avoid repeated encoding lookups, which can be
    expensive as tiktoken loads encoding data from disk.

    Args:
        model_name: Model name to get encoding for, or "" for default cl100k_base

    Returns:
        tiktoken Encoding object

    Raises:
        RuntimeError: If tiktoken is not available
    """
    if not _TIKTOKEN_AVAILABLE or tiktoken is None:
        raise RuntimeError("tiktoken is not available")

    if model_name:
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Model not found, fall back to cl100k_base (GPT-4/Claude-like)
            return tiktoken.get_encoding("cl100k_base")
    else:
        # Default to cl100k_base for modern models
        return tiktoken.get_encoding("cl100k_base")


def _estimate_with_tiktoken(content: str, model: Optional[str] = None) -> Optional[int]:
    """Attempt to estimate tokens using tiktoken.

    Args:
        content: Text to estimate
        model: Optional model name for encoding selection

    Returns:
        Token count if tiktoken available and successful, None otherwise
    """
    if not _TIKTOKEN_AVAILABLE or tiktoken is None:
        return None

    try:
        encoding = _get_cached_encoding(model or "")
        return len(encoding.encode(content))
    except Exception as e:
        logger.debug(f"tiktoken estimation failed: {e}")
        return None


def _estimate_heuristic(content: str) -> int:
    """Estimate tokens using character-based heuristic.

    Uses the common approximation of ~4 characters per token for
    English text. This is a rough estimate and may be inaccurate
    for non-English text, code, or special characters.

    Args:
        content: Text to estimate

    Returns:
        Estimated token count (minimum 1)
    """
    # ~4 characters per token is a common approximation
    # Add 1 to handle empty strings and ensure minimum of 1
    return max(1, len(content) // 4)


def estimate_tokens(
    content: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    use_cache: bool = True,
    warn_on_heuristic: bool = True,
) -> int:
    """Estimate the token count for content.

    Uses a fallback chain for estimation:
    1. Provider-native tokenizer (if registered)
    2. tiktoken (if available)
    3. Character/4 heuristic (always available)

    Results are cached by content hash and provider for efficiency.

    Args:
        content: Text content to estimate tokens for
        provider: Optional provider for provider-specific estimation
        model: Optional model for model-specific estimation
        use_cache: Whether to use/update the cache (default True)
        warn_on_heuristic: Emit warning when falling back to heuristic (default True)

    Returns:
        Estimated token count (minimum 1)

    Warns:
        TokenCountEstimateWarning: When using character-based heuristic fallback

    Example:
        # Basic usage
        tokens = estimate_tokens("Hello, world!")

        # With provider context
        tokens = estimate_tokens(long_content, provider="claude", model="opus")

        # Disable caching for one-off estimates
        tokens = estimate_tokens(content, use_cache=False)
    """
    if not content:
        return 0

    provider_key = (provider or "").lower()
    cache_key = (_content_hash(content), provider_key)

    # Check cache first
    if use_cache:
        with _TOKEN_CACHE_LOCK:
            if cache_key in _TOKEN_ESTIMATE_CACHE:
                return _TOKEN_ESTIMATE_CACHE[cache_key]

    estimate: Optional[int] = None

    # Try provider-native tokenizer first
    if provider_key and provider_key in _PROVIDER_TOKENIZERS:
        try:
            estimate = _PROVIDER_TOKENIZERS[provider_key](content)
            logger.debug(f"Used provider-native tokenizer for {provider_key}")
        except Exception as e:
            logger.debug(f"Provider tokenizer failed for {provider_key}: {e}")

    # Try tiktoken if provider-native didn't work
    if estimate is None:
        estimate = _estimate_with_tiktoken(content, model)
        if estimate is not None:
            logger.debug("Used tiktoken for token estimation")

    # Fall back to heuristic
    if estimate is None:
        estimate = _estimate_heuristic(content)
        logger.debug("Used character heuristic for token estimation")

        if warn_on_heuristic:
            warnings.warn(
                "TOKEN_COUNT_ESTIMATE_USED: Using character-based heuristic for token "
                f"estimation (provider={provider or 'unknown'}). Install tiktoken for "
                "more accurate counts.",
                TokenCountEstimateWarning,
                stacklevel=2,
            )

    # Update cache (with size limit)
    if use_cache:
        with _TOKEN_CACHE_LOCK:
            if len(_TOKEN_ESTIMATE_CACHE) >= _MAX_CACHE_SIZE:
                # Simple eviction: clear half the cache
                keys_to_remove = list(_TOKEN_ESTIMATE_CACHE.keys())[: _MAX_CACHE_SIZE // 2]
                for key in keys_to_remove:
                    del _TOKEN_ESTIMATE_CACHE[key]
            _TOKEN_ESTIMATE_CACHE[cache_key] = estimate

    return estimate


def clear_token_cache() -> int:
    """Clear the token estimation cache.

    Returns:
        Number of entries cleared

    Example:
        cleared = clear_token_cache()
        print(f"Cleared {cleared} cached estimates")
    """
    with _TOKEN_CACHE_LOCK:
        count = len(_TOKEN_ESTIMATE_CACHE)
        _TOKEN_ESTIMATE_CACHE.clear()
    return count


def get_cache_stats() -> dict[str, int]:
    """Get statistics about the token estimation cache.

    Returns:
        Dict with 'size' and 'max_size' keys

    Example:
        stats = get_cache_stats()
        print(f"Cache: {stats['size']}/{stats['max_size']} entries")
    """
    with _TOKEN_CACHE_LOCK:
        size = len(_TOKEN_ESTIMATE_CACHE)
    return {
        "size": size,
        "max_size": _MAX_CACHE_SIZE,
    }
