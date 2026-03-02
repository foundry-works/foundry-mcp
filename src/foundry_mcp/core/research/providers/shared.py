"""Shared provider utilities for HTTP-backed research providers.

Extracts common boilerplate from the 5 HTTP-backed research providers
(Tavily, Perplexity, Google, Semantic Scholar, Tavily Extract) into
reusable, tested utilities.

Architecture constraints:
    - Imports only from stdlib and httpx types (no httpx.AsyncClient creation)
    - SECURITY: All error parsing and settings resolution redact API keys
      and sensitive headers — never expose secrets in logs, error messages,
      or return values.

Utilities are organized by cohesion:
    Pure parsing helpers:
        - parse_retry_after(response) -> Optional[float]
        - extract_error_message(response) -> str
        - parse_iso_date(date_str) -> Optional[datetime]
        - extract_domain(url) -> Optional[str]

    Parameterized patterns:
        - classify_http_error(error, provider_name, custom_classifier) -> ErrorClassification
        - create_resilience_executor(provider_name, config, classify_error) -> executor
        - check_provider_health(provider_name, api_key, base_url) -> bool
        - resolve_provider_settings(provider_name, env_key, api_key, base_url, ...) -> dict
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
)
from urllib.parse import urlparse

if TYPE_CHECKING:
    import httpx

    from foundry_mcp.core.research.models.sources import ResearchSource
    from foundry_mcp.core.research.providers.resilience.models import ErrorClassification, ErrorType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex to detect potential API keys / bearer tokens in strings
_SECRET_PATTERN = re.compile(
    r"(?i)"
    r"(?:"
    r"(?:api[_-]?key|token|bearer|authorization|secret|password|credential)"
    r"[\s:=]+"
    r")"
    r"['\"]?([^\s'\"]{8,})['\"]?",
)

# Common date formats tried after ISO 8601
_COMMON_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
)

# Headers that should never appear in logs/errors
_SENSITIVE_HEADERS = frozenset(
    {
        "authorization",
        "x-api-key",
        "api-key",
        "apikey",
        "cookie",
        "set-cookie",
        "proxy-authorization",
    }
)


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------


def _redact_value(value: str) -> str:
    """Fully redact a secret value.

    Args:
        value: The secret string to redact.

    Returns:
        Redacted string ``"****"``.
    """
    return "****"


def redact_secrets(text: str) -> str:
    """Remove API keys and sensitive tokens from a text string.

    Scans for patterns like ``api_key=...``, ``Bearer ...``, ``token: ...``
    and replaces the secret portion with a redacted placeholder.

    Args:
        text: Input text that may contain secrets.

    Returns:
        Text with secrets replaced by redacted placeholders.
    """
    if not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        full = match.group(0)
        secret = match.group(1)
        return full.replace(secret, _redact_value(secret))

    return _SECRET_PATTERN.sub(_replace, text)


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with sensitive values redacted.

    Args:
        headers: HTTP header mapping (case-insensitive keys).

    Returns:
        New dict with sensitive header values replaced by ``"****"``.
    """
    result: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _SENSITIVE_HEADERS:
            result[key] = _redact_value(value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Pure parsing helpers
# ---------------------------------------------------------------------------


def parse_retry_after(response: "httpx.Response") -> Optional[float]:
    """Parse the ``Retry-After`` header from an HTTP response.

    Handles numeric (integer or float) values only.  RFC 7231 date-based
    values are not supported and will return ``None``.

    Args:
        response: An httpx Response object.

    Returns:
        Seconds to wait before retrying, or ``None`` if the header is
        missing or unparseable.
    """
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


def extract_error_message(
    response: "httpx.Response",
    *,
    provider_format: Optional[Callable[[dict[str, Any]], str]] = None,
) -> str:
    """Extract and redact an error message from an HTTP error response.

    Tries to parse JSON from the response body.  If *provider_format* is
    given it is called first with the parsed JSON dict; if it returns a
    non-empty string that value is used.  Otherwise the standard
    ``{"error": ...}`` / ``{"message": ...}`` patterns are tried.

    The returned message is always run through :func:`redact_secrets`.

    Args:
        response: An httpx Response object.
        provider_format: Optional callable ``(json_data) -> str`` for
            provider-specific JSON shapes (e.g. Google's nested ``error``
            dict).

    Returns:
        A human-readable, secret-redacted error message.
    """
    try:
        data = response.json()

        # Provider-specific extraction first
        if provider_format is not None:
            result = provider_format(data)
            if result:
                return redact_secrets(result)

        # Standard patterns
        error_field = data.get("error")
        if isinstance(error_field, dict):
            msg = error_field.get("message", str(error_field))
        elif isinstance(error_field, str):
            msg = error_field
        else:
            msg = data.get("message", response.text[:200])

        return redact_secrets(str(msg))
    except Exception:
        text = response.text[:200] if response.text else "Unknown error"
        return redact_secrets(text)


def parse_iso_date(
    date_str: Optional[str],
    *,
    extra_formats: Optional[tuple[str, ...]] = None,
) -> Optional[datetime]:
    """Parse a date string, trying ISO 8601 first then common formats.

    Args:
        date_str: The date string to parse.  ``None`` / empty returns ``None``.
        extra_formats: Additional ``strptime`` format strings to try after
            the built-in common formats.

    Returns:
        Parsed :class:`datetime`, or ``None`` if parsing fails.
    """
    if not date_str:
        return None

    # ISO 8601 (handles "Z" suffix)
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        pass

    # Common date formats
    formats = _COMMON_DATE_FORMATS
    if extra_formats:
        formats = formats + extra_formats

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def extract_domain(url: str) -> Optional[str]:
    """Extract the network location (domain) from a URL.

    Args:
        url: A full URL string.

    Returns:
        The ``netloc`` component (e.g. ``"example.com"``), or ``None``
        if the URL is empty or unparseable.
    """
    if not url:
        return None
    try:
        parsed = urlparse(url)
        return parsed.netloc or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def truncate_abstract(
    abstract: Optional[str],
    max_length: int = 500,
) -> Optional[str]:
    """Truncate an abstract for use as a snippet field.

    Shared utility for academic providers (Semantic Scholar, OpenAlex, CrossRef)
    that need to produce short snippets from full abstracts.

    Args:
        abstract: Full abstract text.
        max_length: Maximum snippet length.

    Returns:
        Truncated abstract with ``"..."`` suffix, or ``None`` if input is empty.
    """
    if not abstract:
        return None
    if len(abstract) <= max_length:
        return abstract
    truncated = abstract[:max_length]
    last_space = truncated.rfind(" ")
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."


# ---------------------------------------------------------------------------
# Parameterized patterns
# ---------------------------------------------------------------------------


def extract_status_code(error_message: str) -> Optional[int]:
    """Extract an HTTP status code from an error message string.

    Looks for patterns like ``"HTTP 503"``, ``"API error 429:"``, ``"status 404"``,
    or a bare status code at the start of the message.

    Uses an anchored regex to avoid false positives from incidental 3-digit
    numbers in error messages (e.g. "Found 200 results").

    Args:
        error_message: Error message that may contain an HTTP status code.

    Returns:
        The extracted status code as an ``int``, or ``None`` if none found.
    """
    if not error_message:
        return None
    # Match status codes preceded by HTTP/status/error keywords, or at start of string.
    # Anchored to known patterns to avoid false positives from incidental numbers
    # (e.g. "Found 200 results" should NOT extract 200).
    match = re.search(
        r"(?:HTTP|status|error)\s*(?:code\s*)?:?\s*(\d{3})\b|^(\d{3})\s",
        error_message,
        re.IGNORECASE,
    )
    if match:
        code = int(match.group(1) or match.group(2))
        if 100 <= code <= 599:
            return code
    return None


# Default resilience behaviour for each ErrorType.
# Mapping: ErrorType -> (retryable, trips_breaker)
_ERROR_TYPE_DEFAULTS: dict[str, tuple[bool, bool]] = {
    "rate_limit": (True, False),
    "quota_exceeded": (True, False),
    "server_error": (True, True),
    "timeout": (True, True),
    "network": (True, True),
    "authentication": (False, False),
    "invalid_request": (False, False),
    "unknown": (False, True),
}


def classify_with_registry(
    error: Exception,
    error_classifiers: dict[int, "ErrorType"],
    provider_name: str,
) -> Optional["ErrorClassification"]:
    """Check ERROR_CLASSIFIERS registry for a matching status code.

    Shared implementation used by both ``SearchProvider.classify_error``
    and ``CrossrefProvider.classify_error`` to avoid code duplication.

    Args:
        error: The exception to classify.
        error_classifiers: Mapping of HTTP status codes to ``ErrorType``.
        provider_name: Provider name (unused, reserved for future logging).

    Returns:
        An ``ErrorClassification`` if the error matches a registry entry,
        or ``None`` to fall through to generic classification.
    """
    from foundry_mcp.core.research.providers.base import SearchProviderError
    from foundry_mcp.core.research.providers.resilience import (
        ErrorClassification,
    )

    if error_classifiers and isinstance(error, SearchProviderError):
        code = getattr(error, "status_code", None) or extract_status_code(str(error))
        if code is not None and code in error_classifiers:
            error_type = error_classifiers[code]
            retryable, trips_breaker = _ERROR_TYPE_DEFAULTS.get(
                error_type.value, (False, True)
            )
            return ErrorClassification(
                retryable=retryable,
                trips_breaker=trips_breaker,
                error_type=error_type,
            )
    return None


def classify_http_error(
    error: Exception,
    provider_name: str,
    custom_classifier: Optional[Callable[[Exception], Optional["ErrorClassification"]]] = None,
) -> "ErrorClassification":
    """Classify an exception for resilience decisions.

    Implements the shared classification logic common to all HTTP providers.
    If *custom_classifier* is provided it is called first; if it returns a
    non-``None`` :class:`ErrorClassification` that value is used immediately.

    Classification rules (applied in order):
        1. ``custom_classifier(error)`` if provided
        2. ``AuthenticationError`` → not retryable, no breaker trip
        3. ``RateLimitError`` → retryable, no breaker trip, backoff from retry_after
        4. ``SearchProviderError`` with 5xx → retryable, trips breaker
        5. ``SearchProviderError`` with 400 → not retryable, no breaker trip
        6. ``SearchProviderError`` other → uses its ``retryable`` flag
        7. ``httpx.TimeoutException`` → retryable, trips breaker
        8. ``httpx.RequestError`` → retryable, trips breaker
        9. Default → not retryable, trips breaker

    Args:
        error: The exception to classify.
        provider_name: Provider identifier (for logging/metrics).
        custom_classifier: Optional callable that gets first crack at
            classification.  Return ``None`` to fall through to shared logic.

    Returns:
        An :class:`ErrorClassification` instance.
    """
    # Lazy imports to avoid circular references and keep this module
    # importable from stdlib-only contexts.
    from foundry_mcp.core.research.providers.base import (
        AuthenticationError,
        RateLimitError,
        SearchProviderError,
    )
    from foundry_mcp.core.research.providers.resilience import (
        ErrorClassification,
        ErrorType,
    )

    # 1. Custom classifier gets first shot
    if custom_classifier is not None:
        result = custom_classifier(error)
        if result is not None:
            return result

    # 2. AuthenticationError
    if isinstance(error, AuthenticationError):
        return ErrorClassification(
            retryable=False,
            trips_breaker=False,
            error_type=ErrorType.AUTHENTICATION,
        )

    # 3. RateLimitError
    if isinstance(error, RateLimitError):
        return ErrorClassification(
            retryable=True,
            trips_breaker=False,
            backoff_seconds=error.retry_after,
            error_type=ErrorType.RATE_LIMIT,
        )

    # 4-6. SearchProviderError
    if isinstance(error, SearchProviderError):
        code = getattr(error, "status_code", None) or extract_status_code(str(error))
        if code is not None and code >= 500:
            return ErrorClassification(
                retryable=True,
                trips_breaker=True,
                error_type=ErrorType.SERVER_ERROR,
            )
        if code == 400:
            return ErrorClassification(
                retryable=False,
                trips_breaker=False,
                error_type=ErrorType.INVALID_REQUEST,
            )
        return ErrorClassification(
            retryable=error.retryable,
            trips_breaker=error.retryable,
            error_type=ErrorType.UNKNOWN,
        )

    # 7-8. httpx transport-level errors (checked by class name to avoid
    #       hard-importing httpx at module level)
    error_type_name = type(error).__name__.lower()
    if "timeout" in error_type_name:
        return ErrorClassification(
            retryable=True,
            trips_breaker=True,
            error_type=ErrorType.TIMEOUT,
        )
    if "request" in error_type_name or "connect" in error_type_name:
        return ErrorClassification(
            retryable=True,
            trips_breaker=True,
            error_type=ErrorType.NETWORK,
        )

    # 9. Default
    return ErrorClassification(
        retryable=False,
        trips_breaker=True,
        error_type=ErrorType.UNKNOWN,
    )


def create_resilience_executor(
    provider_name: str,
    config: Any,
    classify_error: Callable[[Exception], Any],
) -> Callable[..., Any]:
    """Create a resilience-wrapped executor for a provider.

    Returns an async function that:
    1. Calls ``execute_with_resilience`` with the given config
    2. Translates resilience exceptions into provider exceptions

    The returned coroutine has signature::

        async def executor(func: Callable[[], Awaitable[T]]) -> T

    This does NOT create or cache ``httpx.AsyncClient`` instances — that
    remains the caller's responsibility.

    Args:
        provider_name: Provider identifier.
        config: A :class:`ProviderResilienceConfig` instance.
        classify_error: Error classifier callable.

    Returns:
        An async executor function.
    """

    async def executor(func: Callable[..., Any], *, timeout: float = 30.0) -> Any:
        """Execute *func* with the full resilience stack.

        Args:
            func: Zero-argument async callable to execute.
            timeout: Per-request timeout in seconds.

        Returns:
            The return value of *func*.

        Raises:
            SearchProviderError: On non-recoverable failures.
            RateLimitError: When rate limit exceeded.
        """
        # Lazy imports inside the coroutine so that patches are visible
        from foundry_mcp.core.errors.resilience import CircuitBreakerError
        from foundry_mcp.core.research.providers.base import (
            RateLimitError,
            SearchProviderError,
        )
        from foundry_mcp.core.research.providers.resilience import (
            RateLimitWaitError,
            TimeBudgetExceededError,
            execute_with_resilience,
            get_resilience_manager,
        )

        time_budget = timeout * (config.max_retries + 1)
        try:
            return await execute_with_resilience(
                func,
                provider_name=provider_name,
                time_budget=time_budget,
                classify_error=classify_error,
                manager=get_resilience_manager(),
                resilience_config=config,
            )
        except CircuitBreakerError as e:
            raise SearchProviderError(
                provider=provider_name,
                message=f"Circuit breaker open: {e}",
                retryable=False,
            ) from e
        except RateLimitWaitError as e:
            raise RateLimitError(
                provider=provider_name,
                retry_after=e.wait_needed,
            ) from e
        except TimeBudgetExceededError as e:
            raise SearchProviderError(
                provider=provider_name,
                message=f"Request timed out: {e}",
                retryable=True,
            ) from e
        except SearchProviderError:
            raise
        except Exception as e:
            classification = classify_error(e)
            raise SearchProviderError(
                provider=provider_name,
                message=redact_secrets(f"Request failed after retries: {e}"),
                retryable=classification.retryable,
                original_error=e,
            ) from e

    return executor


async def check_provider_health(
    provider_name: str,
    api_key: Optional[str],
    base_url: str,
    *,
    test_func: Optional[Callable[..., Any]] = None,
) -> bool:
    """Check if a provider API is accessible.

    If *test_func* is provided it is awaited as the health probe.
    Otherwise returns ``True`` (no-op health check).

    API keys are **never** included in log messages.

    Args:
        provider_name: Provider identifier for logging.
        api_key: The API key (used only to verify it is set, not logged).
        base_url: The base URL for the provider API.
        test_func: Optional async callable to use as health probe.

    Returns:
        ``True`` if the provider is healthy, ``False`` otherwise.
    """
    from foundry_mcp.core.errors.search import AuthenticationError

    if not api_key:
        logger.error(
            "%s health check failed: API key not configured",
            provider_name,
        )
        return False

    if test_func is None:
        return True

    try:
        await test_func()
        return True
    except AuthenticationError:
        logger.error(
            "%s health check failed: invalid API key",
            provider_name,
        )
        return False
    except Exception as e:
        logger.warning(
            "%s health check failed: %s",
            provider_name,
            redact_secrets(str(e)),
        )
        return False


def resolve_provider_settings(
    provider_name: str,
    env_key: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_base_url: str = "",
    timeout: float = 30.0,
    max_retries: int = 3,
    rate_limit: float = 1.0,
    required: bool = True,
    extra_env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Resolve provider settings from explicit params and environment.

    Resolution order (highest priority first):
    1. Explicit keyword arguments
    2. Environment variables
    3. Defaults

    The returned dict **never** contains the raw API key value.
    Instead ``api_key`` is present as the resolved (but redacted in logs)
    value and ``api_key_source`` indicates where it came from.

    Args:
        provider_name: Human-readable provider name.
        env_key: Environment variable name for the API key
            (e.g. ``"TAVILY_API_KEY"``).
        api_key: Explicit API key (takes priority over env var).
        base_url: Explicit base URL (takes priority over default).
        default_base_url: Default base URL if none provided.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts.
        rate_limit: Requests per second.
        required: If ``True`` (default), raise ``ValueError`` when no
            API key is found.
        extra_env: Additional env vars to resolve, mapping
            ``{setting_name: env_var_name}``.  Values are included
            in the returned dict.

    Returns:
        Dict with resolved settings::

            {
                "api_key": "<resolved key>",
                "api_key_source": "explicit" | "environment" | None,
                "base_url": "<resolved url>",
                "timeout": float,
                "max_retries": int,
                "rate_limit": float,
                ...extra_env results...
            }

    Raises:
        ValueError: If *required* is True and no API key is found.
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get(env_key)
    key_source: Optional[str] = None

    if api_key:
        key_source = "explicit"
    elif resolved_key:
        key_source = "environment"

    if required and not resolved_key:
        raise ValueError(
            f"{provider_name} API key required. Provide via api_key parameter or {env_key} environment variable."
        )

    # Resolve base URL
    resolved_url = (base_url or default_base_url).rstrip("/")

    result: dict[str, Any] = {
        "api_key": resolved_key,
        "api_key_source": key_source,
        "base_url": resolved_url,
        "timeout": timeout,
        "max_retries": max_retries,
        "rate_limit": rate_limit,
    }

    # Resolve extra env vars
    if extra_env:
        for setting_name, env_var in extra_env.items():
            result[setting_name] = os.environ.get(env_var)

    return result


# ---------------------------------------------------------------------------
# Fetch-time source summarization (Phase 1)
# ---------------------------------------------------------------------------

_SOURCE_SUMMARIZATION_PROMPT = string.Template("""\
You are a research assistant. Summarize the following web page content for \
a researcher who needs to quickly assess relevance and extract key facts.

Produce a JSON object with two fields:

{
  "summary": "A concise narrative summary (roughly 25-30% of original length). \
Capture main points, key arguments, conclusions, and important context.",
  "key_excerpts": ["up to 5 verbatim quotes from the original text that are \
most important for citation"]
}

Content-type-specific guidance:
- News articles: preserve who, what, when, where, why, how (5W1H)
- Scientific/research: preserve methodology, key results, conclusions, and \
sample sizes or confidence intervals
- Opinion/analysis: preserve main arguments and supporting evidence
- Product/technical: preserve key features, specifications, and unique \
differentiators

Rules:
- Target ~25-30% of original length for the summary (unless content is \
already concise)
- Preserve: key facts, statistics, data points, important quotes, dates, \
names, locations
- Maintain chronological order for time-sensitive content
- key_excerpts must be exact verbatim quotes from the original (max 5)
- Return ONLY valid JSON, no additional text

Content to summarize:
$content""")

_SUMMARIZATION_TIMEOUT: float = 60.0  # seconds per source
_DEFAULT_MAX_CONTENT_LENGTH: int = 50_000  # chars; matches open_deep_research


@dataclass
class SourceSummarizationResult:
    """Result of summarizing a single research source at fetch time.

    Attributes:
        executive_summary: Narrative summary of the source content.
        key_excerpts: Up to 5 verbatim quotes from the original.
        input_tokens: Tokens consumed by the summarization prompt.
        output_tokens: Tokens generated in the summary response.
    """

    executive_summary: str
    key_excerpts: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class SourceSummarizer:
    """Summarizes raw search result content at fetch time using a cheap LLM.

    Produces an executive summary and key verbatim excerpts for each source,
    reducing token pressure on downstream analysis phases while preserving
    citation-quality quotes.

    The summarizer uses the foundry provider system for LLM calls and supports
    timeout-based fallback (returns original content on failure).

    Args:
        provider_id: LLM provider to use for summarization.
        model: Optional model override.
        timeout: Per-source summarization timeout in seconds (default: 60).
        max_concurrent: Maximum parallel summarization calls (default: 3).
        max_content_length: Truncate source content to this many chars before
            summarization (default: 50,000, matching open_deep_research).
    """

    def __init__(
        self,
        provider_id: str,
        model: Optional[str] = None,
        timeout: float = _SUMMARIZATION_TIMEOUT,
        max_concurrent: int = 3,
        max_content_length: int = _DEFAULT_MAX_CONTENT_LENGTH,
    ):
        self._provider_id = provider_id
        self._model = model
        self._timeout = timeout
        self._max_concurrent = max_concurrent
        self._max_content_length = max_content_length

    async def summarize_source(
        self,
        content: str,
    ) -> SourceSummarizationResult:
        """Summarize a single source's content.

        Args:
            content: Raw content string to summarize.

        Returns:
            SourceSummarizationResult with executive summary and key excerpts.

        Raises:
            Exception: Propagated from provider on failure (caller should catch).
        """
        # Binary content guard — prevent sending garbled bytes to LLM
        from foundry_mcp.core.research.content_classifier import is_binary_content

        if is_binary_content(content):
            logger.warning("Skipping summarization: binary content detected")
            return SourceSummarizationResult(
                executive_summary="[Content skipped: binary/non-text document detected]",
                key_excerpts=[],
                input_tokens=0,
                output_tokens=0,
            )

        from foundry_mcp.core.providers import (
            ProviderHooks,
            ProviderRequest,
            ProviderStatus,
        )
        from foundry_mcp.core.providers.registry import resolve_provider

        hooks = ProviderHooks()
        provider = resolve_provider(self._provider_id, hooks=hooks)
        if provider is None:
            raise RuntimeError(f"Summarization provider not available: {self._provider_id}")

        # Cap input length to avoid blowing the context window on
        # full-page extracted content (matches open_deep_research).
        if self._max_content_length and len(content) > self._max_content_length:
            content = content[: self._max_content_length]

        # Use safe_substitute to prevent KeyError from Python format
        # string patterns in web content (e.g., "{system}", "{__class__}").
        # Also sanitize web content to strip prompt-injection vectors.
        from foundry_mcp.core.research.workflows.deep_research._helpers import (
            sanitize_external_content,
        )

        content = sanitize_external_content(content)
        prompt = _SOURCE_SUMMARIZATION_PROMPT.safe_substitute(content=content)

        # Note: no timeout here — the caller (summarize_sources) wraps this
        # in asyncio.wait_for() for cancellation.  Duplicating the timeout
        # would cause confusing race conditions between the two layers.
        request = ProviderRequest(
            prompt=prompt,
            max_tokens=2000,
            model=self._model,
        )

        result = await asyncio.to_thread(provider.generate, request)
        if result.status != ProviderStatus.SUCCESS:
            error_msg = result.stderr or "Unknown error"
            raise RuntimeError(f"Summarization failed ({self._provider_id}): {error_msg}")

        # Parse the response into executive summary + excerpts
        executive_summary, key_excerpts = self._parse_summary_response(result.content)

        # Extract token counts from ProviderResult.tokens (TokenUsage)
        input_tokens = result.tokens.input_tokens if result.tokens else 0
        output_tokens = result.tokens.output_tokens if result.tokens else 0

        return SourceSummarizationResult(
            executive_summary=executive_summary,
            key_excerpts=key_excerpts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    async def summarize_sources(
        self,
        sources: list["ResearchSource"],
    ) -> dict[str, SourceSummarizationResult]:
        """Summarize multiple sources in parallel with concurrency control.

        For each source with non-empty content, runs summarization with a
        timeout. On failure or timeout, the source retains its original
        content (no summary produced).

        Args:
            sources: List of ResearchSource objects to summarize.

        Returns:
            Dict mapping source.id -> SourceSummarizationResult for sources
            that were successfully summarized. Sources that failed or had no
            content are omitted from the result.
        """
        semaphore = asyncio.Semaphore(self._max_concurrent)
        results: dict[str, SourceSummarizationResult] = {}

        async def _summarize_one(source: "ResearchSource") -> None:
            if not source.content:
                return

            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        self.summarize_source(source.content),
                        timeout=self._timeout,
                    )
                    results[source.id] = result
                except asyncio.TimeoutError:
                    logger.warning(
                        "Source summarization timed out after %.0fs: %s",
                        self._timeout,
                        source.id,
                    )
                except Exception as e:
                    logger.warning(
                        "Source summarization failed for %s: %s",
                        source.id,
                        redact_secrets(str(e)),
                    )

        tasks = [_summarize_one(source) for source in sources]
        await asyncio.gather(*tasks)

        return results

    @staticmethod
    def _parse_summary_response(response: str) -> tuple[str, list[str]]:
        """Parse LLM summary response into executive summary and excerpts.

        Tries JSON parsing first (new format with ``summary`` +
        ``key_excerpts`` fields).  Falls back to markdown-section parsing
        for backward compatibility with the old ``## Executive Summary`` /
        ``## Key Excerpts`` format.

        Args:
            response: Raw LLM response text.

        Returns:
            Tuple of (executive_summary, key_excerpts).
        """
        import json as _json

        # --- Primary path: JSON ---
        # Strip markdown code fences if present
        stripped = response.strip()
        if stripped.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()

        try:
            data = _json.loads(stripped)
            if isinstance(data, dict):
                summary = str(data.get("summary", "")).strip()
                raw_excerpts = data.get("key_excerpts", [])
                if isinstance(raw_excerpts, list):
                    excerpts = [str(e).strip().strip('"').strip("'") for e in raw_excerpts[:5] if e]
                else:
                    excerpts = []
                if summary:
                    return summary, excerpts
        except (_json.JSONDecodeError, TypeError, ValueError):
            pass

        # --- Fallback: markdown section parsing (backward compat) ---
        executive_summary = response.strip()
        key_excerpts: list[str] = []

        excerpts_markers = ["## Key Excerpts", "## Key excerpts", "**Key Excerpts**"]
        summary_markers = ["## Executive Summary", "## Executive summary", "**Executive Summary**"]

        excerpts_header_start = -1
        excerpts_start = -1
        for marker in excerpts_markers:
            idx = response.find(marker)
            if idx != -1:
                excerpts_header_start = idx
                excerpts_start = idx + len(marker)
                break

        summary_start = -1
        for marker in summary_markers:
            idx = response.find(marker)
            if idx != -1:
                summary_start = idx + len(marker)
                break

        if summary_start != -1:
            summary_end = excerpts_header_start if excerpts_header_start != -1 else len(response)
            executive_summary = response[summary_start:summary_end].strip()
        elif excerpts_header_start != -1:
            executive_summary = response[:excerpts_header_start].strip()

        if excerpts_start != -1:
            excerpts_text = response[excerpts_start:]
            for line in excerpts_text.strip().split("\n"):
                line = line.strip()
                if line.startswith(("- ", "* ", "\u2022 ")):
                    excerpt = line.lstrip("-*\u2022 ").strip().strip('"').strip("'")
                    if excerpt:
                        key_excerpts.append(excerpt)
                        if len(key_excerpts) >= 5:
                            break

        return executive_summary, key_excerpts

    @staticmethod
    def format_summarized_content(
        executive_summary: str,
        key_excerpts: list[str],
    ) -> str:
        """Format summary + excerpts into a single content string.

        Produces structured ``<summary>`` and ``<key_excerpts>`` tags that
        replace the original source content, matching the format used by
        open_deep_research for researcher-facing source presentation.

        Args:
            executive_summary: The narrative summary.
            key_excerpts: List of verbatim quotes.

        Returns:
            Formatted content string with XML-style tags.
        """
        parts = [f"<summary>{executive_summary}</summary>"]
        if key_excerpts:
            excerpts_text = ", ".join(f'"{e}"' for e in key_excerpts)
            parts.append(f"<key_excerpts>{excerpts_text}</key_excerpts>")
        return "\n\n".join(parts)
