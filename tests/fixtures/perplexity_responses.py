"""
Perplexity API response fixtures for testing.

This module provides reusable mock responses for Perplexity Search API,
matching the actual API response format with all new fields included.

Fixtures are designed for:
- Unit tests (mocking httpx responses)
- Integration tests (mocking provider methods)
- Contract compatibility tests (validating response parsing)

Fixture Freshness: 2026-01-27
API Reference: https://docs.perplexity.ai/
"""

from datetime import datetime
from typing import Any

# =============================================================================
# Perplexity Search API Response Fixtures
# =============================================================================


def perplexity_search_response_basic() -> dict[str, Any]:
    """Basic Perplexity search response with minimal fields.

    Returns:
        Mock API response for default search
    """
    return {
        "results": [
            {
                "title": "Machine Learning Fundamentals",
                "url": "https://example.com/ml-basics",
                "snippet": "Machine learning is a subset of AI that enables systems to learn...",
                "date": "2024-06-15T10:30:00Z",
            },
            {
                "title": "Deep Learning Guide",
                "url": "https://example.org/deep-learning",
                "snippet": "Deep learning uses neural networks with multiple layers...",
                "date": "2024-07-01T14:00:00Z",
            },
        ],
    }


def perplexity_search_response_with_context_size_high() -> dict[str, Any]:
    """Perplexity search response with search_context_size='high'.

    Returns:
        Mock API response for high context search (more comprehensive results)
    """
    return {
        "results": [
            {
                "title": "Comprehensive Transformer Architecture Guide",
                "url": "https://arxiv.org/abs/transformer-guide",
                "snippet": "The transformer architecture revolutionized NLP with its "
                "self-attention mechanism. This comprehensive guide covers "
                "all aspects including positional encoding, multi-head attention, "
                "and feed-forward networks used in modern LLMs.",
                "date": "2024-01-15T09:00:00Z",
            },
            {
                "title": "State of AI Report 2024",
                "url": "https://example.com/ai-report-2024",
                "snippet": "An in-depth analysis of AI trends, breakthroughs, and "
                "challenges. Covers foundation models, multimodal AI, AI safety, "
                "and industry adoption patterns across sectors.",
                "date": "2024-12-01T00:00:00Z",
            },
        ],
    }


def perplexity_search_response_with_recency_filter() -> dict[str, Any]:
    """Perplexity search response with recency_filter applied.

    Returns:
        Mock API response for recent results only
    """
    return {
        "results": [
            {
                "title": "Latest AI Developments This Week",
                "url": "https://technews.com/ai-weekly",
                "snippet": "Breaking news on recent AI advancements...",
                "date": "2026-01-25T08:00:00Z",
            },
            {
                "title": "New GPT Model Released",
                "url": "https://openai.com/blog/new-release",
                "snippet": "OpenAI announces their latest language model...",
                "date": "2026-01-24T16:30:00Z",
            },
        ],
    }


def perplexity_search_response_with_date_filters() -> dict[str, Any]:
    """Perplexity search response with date range filters.

    Returns:
        Mock API response for date-filtered search
    """
    return {
        "results": [
            {
                "title": "Q3 2024 AI Industry Review",
                "url": "https://example.com/q3-review",
                "snippet": "Analysis of AI industry trends from July to September 2024...",
                "date": "2024-09-30T00:00:00Z",
            },
            {
                "title": "Summer 2024 ML Benchmarks",
                "url": "https://mlbench.org/summer-2024",
                "snippet": "Benchmark results from the summer testing period...",
                "date": "2024-08-15T00:00:00Z",
            },
        ],
    }


def perplexity_search_response_with_last_updated() -> dict[str, Any]:
    """Perplexity search response with last_updated fields.

    Returns:
        Mock API response with last_updated instead of date
    """
    return {
        "results": [
            {
                "title": "Continuously Updated ML Guide",
                "url": "https://mlguide.com/living-document",
                "snippet": "A regularly updated guide to machine learning...",
                "last_updated": "2026-01-20",
            },
            {
                "title": "Wiki: Neural Networks",
                "url": "https://wiki.ai/neural-networks",
                "snippet": "Community-maintained documentation on neural networks...",
                "last_updated": "2026-01-18",
            },
        ],
    }


def perplexity_search_response_with_country() -> dict[str, Any]:
    """Perplexity search response with country filter applied.

    Returns:
        Mock API response for geo-filtered search (US results)
    """
    return {
        "results": [
            {
                "title": "US AI Research Initiatives",
                "url": "https://nsf.gov/ai-research",
                "snippet": "National Science Foundation AI research programs...",
                "date": "2024-11-01T00:00:00Z",
            },
            {
                "title": "Silicon Valley AI Startups 2024",
                "url": "https://techcrunch.com/sv-ai-startups",
                "snippet": "Overview of AI startups in the Bay Area...",
                "date": "2024-10-15T00:00:00Z",
            },
        ],
    }


def perplexity_search_response_empty() -> dict[str, Any]:
    """Perplexity search response with no results.

    Returns:
        Mock API response with empty results
    """
    return {
        "results": [],
    }


def perplexity_search_response_with_raw_content() -> dict[str, Any]:
    """Perplexity search response with include_raw_content=True.

    Returns:
        Mock API response with full page content
    """
    return {
        "results": [
            {
                "title": "Understanding Attention Mechanisms",
                "url": "https://example.com/attention",
                "snippet": "A deep dive into attention mechanisms in neural networks...",
                "raw_content": "# Understanding Attention Mechanisms\n\n"
                "Attention mechanisms have revolutionized deep learning by allowing "
                "models to focus on relevant parts of the input when producing output.\n\n"
                "## Self-Attention\n\n"
                "Self-attention, or intra-attention, relates different positions of a "
                "single sequence to compute a representation of the sequence.\n\n"
                "## Multi-Head Attention\n\n"
                "Multi-head attention allows the model to jointly attend to information "
                "from different representation subspaces at different positions.",
                "date": "2024-05-01T00:00:00Z",
            },
        ],
    }


# =============================================================================
# Error Response Fixtures
# =============================================================================


def perplexity_error_response_401() -> dict[str, Any]:
    """Perplexity API 401 unauthorized response.

    Returns:
        Mock error response for invalid API key
    """
    return {
        "error": "Unauthorized",
        "message": "Invalid API key provided",
    }


def perplexity_error_response_429() -> dict[str, Any]:
    """Perplexity API 429 rate limit response.

    Returns:
        Mock error response for rate limiting
    """
    return {
        "error": "Too Many Requests",
        "message": "Rate limit exceeded. Please wait before retrying.",
    }


def perplexity_error_response_400_invalid_context_size() -> dict[str, Any]:
    """Perplexity API 400 response for invalid search_context_size.

    Returns:
        Mock error response for validation error
    """
    return {
        "error": "Bad Request",
        "message": "Invalid search_context_size. Must be one of: low, medium, high",
    }


def perplexity_error_response_400_invalid_date() -> dict[str, Any]:
    """Perplexity API 400 response for invalid date format.

    Returns:
        Mock error response for date validation error
    """
    return {
        "error": "Bad Request",
        "message": "Invalid date format. Expected MM/DD/YYYY",
    }


def perplexity_error_response_500() -> dict[str, Any]:
    """Perplexity API 500 internal server error response.

    Returns:
        Mock error response for server error
    """
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
    }


# =============================================================================
# Fixture Metadata
# =============================================================================


FIXTURE_METADATA = {
    "version": "1.0.0",
    "last_updated": "2026-01-27",
    "api_version": "v1",
    "api_docs": "https://docs.perplexity.ai/",
    "fixtures": {
        "search": [
            "perplexity_search_response_basic",
            "perplexity_search_response_with_context_size_high",
            "perplexity_search_response_with_recency_filter",
            "perplexity_search_response_with_date_filters",
            "perplexity_search_response_with_last_updated",
            "perplexity_search_response_with_country",
            "perplexity_search_response_empty",
            "perplexity_search_response_with_raw_content",
        ],
        "errors": [
            "perplexity_error_response_401",
            "perplexity_error_response_429",
            "perplexity_error_response_400_invalid_context_size",
            "perplexity_error_response_400_invalid_date",
            "perplexity_error_response_500",
        ],
    },
}


def get_fixture_freshness_date() -> str:
    """Get the date when fixtures were last updated.

    Returns:
        ISO date string of last update
    """
    return FIXTURE_METADATA["last_updated"]


def check_fixture_freshness(max_age_days: int = 90) -> bool:
    """Check if fixtures are still fresh.

    Args:
        max_age_days: Maximum age in days before fixtures are considered stale

    Returns:
        True if fixtures are fresh, False if stale
    """
    last_updated = datetime.fromisoformat(FIXTURE_METADATA["last_updated"])
    age = (datetime.now() - last_updated).days
    return age <= max_age_days
