"""Content extraction handler: extract."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.responses.builders import (
    error_response,
    success_response,
)
from foundry_mcp.core.responses.types import (
    ErrorCode,
    ErrorType,
)
from foundry_mcp.tools.unified.param_schema import List_, validate_payload

from ._helpers import _get_config

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_EXTRACT_SCHEMA = {
    "urls": List_(required=True),
}

logger = logging.getLogger(__name__)


def _handle_extract(
    *,
    urls: Optional[list[str]] = None,
    extract_depth: str = "basic",
    include_images: bool = False,
    format: str = "markdown",
    query: Optional[str] = None,
    chunks_per_source: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Extract content from URLs using Tavily Extract API.

    Response envelope patterns (per MCP best practices):
    - Full success: success=True, data contains sources and stats, error=None
    - Partial success: success=True, data.failed_urls populated, meta.warnings contains summary
    - Total failure: success=False, data contains error_code/error_type/remediation/details

    Error codes:
    - VALIDATION_ERROR: Invalid parameters or URL format
    - INVALID_URL: URL parsing or scheme validation failed
    - BLOCKED_HOST: SSRF protection blocked the URL
    - RATE_LIMIT_EXCEEDED: API rate limit hit
    - TIMEOUT: Request timeout
    - EXTRACT_FAILED: General extraction failure

    Args:
        urls: List of URLs to extract content from (required, max 10).
        extract_depth: "basic" or "advanced" (default: "basic").
        include_images: Include images in results (default: False).
        format: Output format, "markdown" or "text" (default: "markdown").
        query: Optional query for relevance-based chunk reranking.
        chunks_per_source: Chunks per URL, 1-5 (default: 3).

    Returns:
        MCP response envelope with extracted content as ResearchSource objects.
    """
    import asyncio
    import os
    from concurrent.futures import ThreadPoolExecutor

    from foundry_mcp.core.errors.search import (
        AuthenticationError,
        RateLimitError,
        SearchProviderError,
    )
    from foundry_mcp.core.research.providers.tavily_extract import (
        TavilyExtractProvider,
        UrlValidationError,
        validate_extract_url_async,
    )

    payload = {"urls": urls}
    err = validate_payload(payload, _EXTRACT_SCHEMA, tool_name="research", action="extract")
    if err:
        return err

    # Get API key from config or environment
    config = _get_config()
    api_key = config.research.tavily_api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return asdict(
            error_response(
                "Tavily API key not configured",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Set TAVILY_API_KEY environment variable or tavily_api_key in config",
            )
        )

    def _run_async(coro: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Avoid blocking a running loop by executing in a worker thread.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()

    # Pre-validate URLs and track validation failures (async DNS checks)
    async def _validate_urls_async(url_list: list[str]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        valid: list[str] = []
        failed: list[str] = []
        details: list[dict[str, Any]] = []
        for url in url_list:
            try:
                await validate_extract_url_async(url)
                valid.append(url)
            except UrlValidationError as e:
                failed.append(url)
                details.append(
                    {
                        "url": url,
                        "error": e.reason,
                        "error_code": e.error_code,
                    }
                )
        return valid, failed, details

    assert isinstance(urls, list)
    valid_urls, failed_urls, error_details = _run_async(_validate_urls_async(urls))

    # If all URLs failed validation, return total failure
    if not valid_urls:
        return asdict(
            error_response(
                f"All {len(urls)} URLs failed validation",
                error_code="INVALID_URL",
                error_type=ErrorType.VALIDATION,
                remediation="Check URL formats and ensure they are publicly accessible HTTP/HTTPS URLs",
                details={
                    "failed_urls": failed_urls,
                    "error_details": error_details,
                },
            )
        )

    try:
        provider = TavilyExtractProvider(api_key=api_key)

        # Build extract kwargs
        extract_kwargs: dict[str, Any] = {
            "extract_depth": extract_depth,
            "include_images": include_images,
            "format": format,
        }
        if query is not None:
            extract_kwargs["query"] = query
        if chunks_per_source is not None:
            extract_kwargs["chunks_per_source"] = chunks_per_source

        # Execute extraction for valid URLs only
        extract_kwargs["validate_urls"] = False
        sources = _run_async(provider.extract(valid_urls, **extract_kwargs))

        # Convert ResearchSource objects to dicts
        source_dicts = []
        succeeded_urls = set()
        for src in sources:
            metadata = src.public_metadata() if hasattr(src, "public_metadata") else src.metadata
            src_dict = {
                "url": src.url,
                "title": src.title,
                "source_type": src.source_type.value if src.source_type else "web",
                "snippet": src.snippet,
                "content": src.content,
                "metadata": metadata,
            }
            source_dicts.append(src_dict)
            if src.url:
                succeeded_urls.add(src.url)

        # Check for URLs that were valid but failed extraction
        for url in valid_urls:
            if url not in succeeded_urls:
                failed_urls.append(url)
                error_details.append(
                    {
                        "url": url,
                        "error": "Extraction returned no content",
                        "error_code": "EXTRACT_FAILED",
                    }
                )

        # Build response based on success/failure pattern
        stats = {
            "requested": len(urls),
            "succeeded": len(sources),
            "failed": len(failed_urls),
        }

        # Determine response type
        if len(sources) == 0:
            # Total failure: no sources extracted
            return asdict(
                error_response(
                    f"Extract failed: no content extracted from {len(urls)} URLs",
                    error_code="EXTRACT_FAILED",
                    error_type=ErrorType.INTERNAL,
                    remediation="Check that URLs are publicly accessible and contain extractable content",
                    details={
                        "failed_urls": failed_urls,
                        "error_details": error_details,
                    },
                )
            )
        elif failed_urls:
            # Partial success: some URLs succeeded, some failed
            warnings = [f"{len(failed_urls)} of {len(urls)} URLs failed extraction"]
            return asdict(
                success_response(
                    data={
                        "action": "extract",
                        "sources": source_dicts,
                        "stats": stats,
                        "failed_urls": failed_urls,
                        "error_details": error_details,
                    },
                    warnings=warnings,
                )
            )
        else:
            # Full success: all URLs extracted
            return asdict(
                success_response(
                    data={
                        "action": "extract",
                        "sources": source_dicts,
                        "stats": stats,
                    }
                )
            )

    except AuthenticationError as e:
        return asdict(
            error_response(
                f"Authentication failed: {e}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check that TAVILY_API_KEY is valid",
                details={
                    "failed_urls": urls,
                    "error_details": [
                        {
                            "url": None,
                            "error": str(e),
                            "error_code": "AUTHENTICATION_ERROR",
                        }
                    ],
                },
            )
        )
    except RateLimitError as e:
        return asdict(
            error_response(
                f"Rate limit exceeded: {e}",
                error_code="RATE_LIMIT_EXCEEDED",
                error_type=ErrorType.RATE_LIMIT,
                remediation=f"Wait {e.retry_after or 60} seconds before retrying"
                if hasattr(e, "retry_after")
                else "Wait before retrying",
                details={
                    "failed_urls": urls,
                    "error_details": [
                        {
                            "url": None,
                            "error": str(e),
                            "error_code": "RATE_LIMIT_EXCEEDED",
                        }
                    ],
                },
            )
        )
    except SearchProviderError as e:
        message = str(e)
        original = getattr(e, "original_error", None)
        timeout_detected = "timeout" in e.message.lower() or "timed out" in e.message.lower()
        if original is not None:
            if isinstance(original, asyncio.TimeoutError):
                timeout_detected = True
            elif "timeout" in type(original).__name__.lower():
                timeout_detected = True

        if timeout_detected:
            return asdict(
                error_response(
                    f"Extract request timed out: {message}",
                    error_code="TIMEOUT",
                    error_type=ErrorType.UNAVAILABLE,
                    remediation="Try with fewer URLs or increase timeout",
                    details={
                        "failed_urls": urls,
                        "error_details": [
                            {
                                "url": None,
                                "error": message,
                                "error_code": "TIMEOUT",
                            }
                        ],
                    },
                )
            )

        return asdict(
            error_response(
                f"Extract failed: {message}",
                error_code="EXTRACT_FAILED",
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details or try with different URLs",
                details={
                    "failed_urls": urls if urls else [],
                    "error_details": [
                        {
                            "url": None,
                            "error": message,
                            "error_code": "EXTRACT_FAILED",
                        }
                    ],
                },
            )
        )
    except UrlValidationError as e:
        return asdict(
            error_response(
                f"URL validation failed: {e.reason}",
                error_code=e.error_code,
                error_type=ErrorType.VALIDATION,
                details={
                    "failed_urls": [e.url],
                    "error_details": [
                        {
                            "url": e.url,
                            "error": e.reason,
                            "error_code": e.error_code,
                        }
                    ],
                },
            )
        )
    except ValueError as e:
        return asdict(
            error_response(
                str(e),
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                details={
                    "failed_urls": urls if urls else [],
                    "error_details": [
                        {
                            "url": None,
                            "error": str(e),
                            "error_code": ErrorCode.VALIDATION_ERROR,
                        }
                    ],
                },
            )
        )
    except asyncio.TimeoutError:
        return asdict(
            error_response(
                "Extract request timed out",
                error_code="TIMEOUT",
                error_type=ErrorType.UNAVAILABLE,
                remediation="Try with fewer URLs or increase timeout",
                details={
                    "failed_urls": urls,
                    "error_details": [
                        {
                            "url": None,
                            "error": "Request timed out",
                            "error_code": "TIMEOUT",
                        }
                    ],
                },
            )
        )
    except Exception as e:
        logger.exception("Extract failed: %s", e)
        return asdict(
            error_response(
                f"Extract failed: {e}",
                error_code="EXTRACT_FAILED",
                error_type=ErrorType.INTERNAL,
                remediation="Check logs for details or try with different URLs",
                details={
                    "failed_urls": urls if urls else [],
                    "error_details": [
                        {
                            "url": None,
                            "error": str(e),
                            "error_code": "EXTRACT_FAILED",
                        }
                    ],
                },
            )
        )
