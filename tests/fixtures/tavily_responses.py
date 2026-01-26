"""
Tavily API response fixtures for testing.

This module provides reusable mock responses for Tavily Search and Extract APIs,
matching the actual API response format with all new fields included.

Fixtures are designed for:
- Unit tests (mocking httpx responses)
- Integration tests (mocking provider methods)
- Contract compatibility tests (validating response parsing)

Fixture Freshness: 2026-01-26
API Reference: https://docs.tavily.com/
"""

from datetime import datetime
from typing import Any

# =============================================================================
# Tavily Search API Response Fixtures
# =============================================================================


def tavily_search_response_basic() -> dict[str, Any]:
    """Basic Tavily search response with minimal fields.

    Returns:
        Mock API response for search_depth="basic"
    """
    return {
        "query": "machine learning trends",
        "answer": None,
        "images": [],
        "results": [
            {
                "title": "Machine Learning Trends 2024",
                "url": "https://example.com/ml-trends",
                "content": "Top machine learning trends include...",
                "score": 0.95,
                "published_date": "2024-06-15",
            },
            {
                "title": "AI and ML Industry Report",
                "url": "https://example.org/ai-report",
                "content": "The AI industry continues to grow...",
                "score": 0.89,
                "published_date": "2024-07-01",
            },
        ],
        "response_time": 1.234,
    }


def tavily_search_response_advanced() -> dict[str, Any]:
    """Advanced Tavily search response with raw_content and chunks.

    Returns:
        Mock API response for search_depth="advanced"
    """
    return {
        "query": "deep learning architectures",
        "answer": "Deep learning architectures have evolved significantly...",
        "images": [
            "https://example.com/images/transformer.png",
            "https://example.com/images/cnn.png",
        ],
        "results": [
            {
                "title": "Transformer Architecture Guide",
                "url": "https://arxiv.org/abs/transformer",
                "content": "The transformer architecture revolutionized NLP...",
                "raw_content": "# Transformer Architecture\n\nThe transformer architecture, "
                "introduced in 'Attention is All You Need' (2017), has become "
                "the foundation for modern NLP models...\n\n## Key Components\n\n"
                "1. Self-attention mechanism\n2. Positional encoding\n"
                "3. Feed-forward networks\n\n## Applications\n\n"
                "Transformers power GPT, BERT, and other large language models.",
                "score": 0.98,
                "published_date": "2024-01-15",
            },
            {
                "title": "CNN vs Transformer Comparison",
                "url": "https://example.com/cnn-vs-transformer",
                "content": "Comparing CNNs and Transformers for vision tasks...",
                "raw_content": "# CNN vs Transformer\n\nConvolutional Neural Networks "
                "have long been the standard for computer vision, but Vision "
                "Transformers (ViT) are gaining ground...\n\n## Performance\n\n"
                "ViT excels on large datasets while CNNs are more data-efficient.",
                "score": 0.91,
                "published_date": "2024-03-20",
            },
        ],
        "response_time": 2.456,
    }


def tavily_search_response_with_images() -> dict[str, Any]:
    """Tavily search response with include_images=True.

    Returns:
        Mock API response with image results
    """
    return {
        "query": "neural network diagrams",
        "answer": None,
        "images": [
            "https://example.com/images/nn-diagram-1.png",
            "https://example.com/images/nn-diagram-2.png",
            "https://example.com/images/backprop.gif",
        ],
        "results": [
            {
                "title": "Neural Network Visualization",
                "url": "https://example.com/nn-viz",
                "content": "Visual guide to neural network architectures...",
                "score": 0.93,
            },
        ],
        "response_time": 1.567,
    }


def tavily_search_response_news() -> dict[str, Any]:
    """Tavily search response for topic="news" with days limit.

    Returns:
        Mock API response for news search
    """
    return {
        "query": "AI regulations",
        "answer": None,
        "images": [],
        "results": [
            {
                "title": "EU AI Act Implementation Timeline",
                "url": "https://reuters.com/eu-ai-act",
                "content": "The European Union's AI Act enters enforcement phase...",
                "score": 0.97,
                "published_date": "2024-12-10",
            },
            {
                "title": "US Proposes AI Safety Guidelines",
                "url": "https://nytimes.com/us-ai-guidelines",
                "content": "New federal guidelines aim to ensure AI safety...",
                "score": 0.94,
                "published_date": "2024-12-08",
            },
        ],
        "response_time": 0.987,
    }


def tavily_search_response_empty() -> dict[str, Any]:
    """Tavily search response with no results.

    Returns:
        Mock API response with empty results
    """
    return {
        "query": "very obscure query that matches nothing",
        "answer": None,
        "images": [],
        "results": [],
        "response_time": 0.234,
    }


def tavily_search_response_with_answer() -> dict[str, Any]:
    """Tavily search response with include_answer=True.

    Returns:
        Mock API response with AI-generated answer
    """
    return {
        "query": "What is the capital of France?",
        "answer": "The capital of France is Paris. Paris is the largest city in France "
        "and serves as the country's political, economic, and cultural center. "
        "It is located in the north-central part of the country on the Seine River.",
        "images": [],
        "results": [
            {
                "title": "Paris - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "content": "Paris is the capital and largest city of France...",
                "score": 0.99,
            },
        ],
        "response_time": 1.123,
    }


# =============================================================================
# Tavily Extract API Response Fixtures
# =============================================================================


def tavily_extract_response_basic() -> dict[str, Any]:
    """Basic Tavily extract response for single URL.

    Returns:
        Mock API response for extract_depth="basic"
    """
    return {
        "results": [
            {
                "url": "https://example.com/article",
                "title": "Understanding Machine Learning",
                "raw_content": "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn and improve from experience without "
                "being explicitly programmed. This article explores the fundamentals "
                "of ML including supervised learning, unsupervised learning, and "
                "reinforcement learning approaches.",
                "images": [],
                "favicon": "https://example.com/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 2.345,
    }


def tavily_extract_response_advanced() -> dict[str, Any]:
    """Advanced Tavily extract response with chunks.

    Returns:
        Mock API response for extract_depth="advanced"
    """
    return {
        "results": [
            {
                "url": "https://arxiv.org/abs/attention",
                "title": "Attention Is All You Need",
                "raw_content": "We propose a new simple network architecture, "
                "the Transformer, based solely on attention mechanisms...",
                "chunks": [
                    "Abstract: The dominant sequence transduction models are based "
                    "on complex recurrent or convolutional neural networks that "
                    "include an encoder and a decoder.",
                    "We propose a new simple network architecture, the Transformer, "
                    "based solely on attention mechanisms, dispensing with recurrence "
                    "and convolutions entirely.",
                    "Experiments on two machine translation tasks show these models "
                    "to be superior in quality while being more parallelizable and "
                    "requiring significantly less time to train.",
                ],
                "images": [
                    "https://arxiv.org/images/transformer-arch.png",
                ],
                "favicon": "https://arxiv.org/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 3.456,
    }


def tavily_extract_response_multiple_urls() -> dict[str, Any]:
    """Tavily extract response for multiple URLs.

    Returns:
        Mock API response for batch extraction
    """
    return {
        "results": [
            {
                "url": "https://example.com/page1",
                "title": "Page One Title",
                "raw_content": "Content of the first page with important information...",
                "images": [],
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "url": "https://example.com/page2",
                "title": "Page Two Title",
                "raw_content": "Content of the second page with different information...",
                "images": ["https://example.com/page2/image.jpg"],
                "favicon": "https://example.com/favicon.ico",
            },
            {
                "url": "https://example.org/article",
                "title": "External Article",
                "raw_content": "This is content from an external source...",
                "images": [],
                "favicon": "https://example.org/favicon.ico",
            },
        ],
        "failed_results": [],
        "response_time": 4.567,
    }


def tavily_extract_response_partial_failure() -> dict[str, Any]:
    """Tavily extract response with some URLs failing.

    Returns:
        Mock API response with partial success
    """
    return {
        "results": [
            {
                "url": "https://example.com/success",
                "title": "Successfully Extracted",
                "raw_content": "This page was extracted successfully...",
                "images": [],
                "favicon": "https://example.com/favicon.ico",
            },
        ],
        "failed_results": [
            {
                "url": "https://blocked-site.com/page",
                "error": "URL blocked by robots.txt",
            },
            {
                "url": "https://timeout-site.com/slow",
                "error": "Request timeout after 30s",
            },
        ],
        "response_time": 5.678,
    }


def tavily_extract_response_with_images() -> dict[str, Any]:
    """Tavily extract response with include_images=True.

    Returns:
        Mock API response with image URLs
    """
    return {
        "results": [
            {
                "url": "https://blog.example.com/illustrated-guide",
                "title": "Illustrated Guide to Neural Networks",
                "raw_content": "This comprehensive guide includes diagrams and "
                "illustrations explaining neural network concepts...",
                "images": [
                    "https://blog.example.com/images/nn-intro.png",
                    "https://blog.example.com/images/perceptron.png",
                    "https://blog.example.com/images/backprop.gif",
                    "https://blog.example.com/images/cnn-layers.png",
                    "https://blog.example.com/images/rnn-unrolled.png",
                ],
                "favicon": "https://blog.example.com/favicon.ico",
            }
        ],
        "failed_results": [],
        "response_time": 3.234,
    }


def tavily_extract_response_empty() -> dict[str, Any]:
    """Tavily extract response with no successful extractions.

    Returns:
        Mock API response with all failures
    """
    return {
        "results": [],
        "failed_results": [
            {
                "url": "https://paywalled-site.com/article",
                "error": "Content behind paywall",
            },
        ],
        "response_time": 1.234,
    }


# =============================================================================
# Error Response Fixtures
# =============================================================================


def tavily_error_response_401() -> dict[str, Any]:
    """Tavily API 401 unauthorized response.

    Returns:
        Mock error response for invalid API key
    """
    return {
        "error": "Unauthorized",
        "message": "Invalid API key provided",
        "status_code": 401,
    }


def tavily_error_response_429() -> dict[str, Any]:
    """Tavily API 429 rate limit response.

    Returns:
        Mock error response for rate limiting
    """
    return {
        "error": "Too Many Requests",
        "message": "Rate limit exceeded. Please wait before retrying.",
        "status_code": 429,
        "retry_after": 60,
    }


def tavily_error_response_500() -> dict[str, Any]:
    """Tavily API 500 internal server error response.

    Returns:
        Mock error response for server error
    """
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred. Please try again later.",
        "status_code": 500,
    }


# =============================================================================
# Fixture Metadata
# =============================================================================


FIXTURE_METADATA = {
    "version": "1.0.0",
    "last_updated": "2026-01-26",
    "api_version": "v1",
    "api_docs": "https://docs.tavily.com/",
    "fixtures": {
        "search": [
            "tavily_search_response_basic",
            "tavily_search_response_advanced",
            "tavily_search_response_with_images",
            "tavily_search_response_news",
            "tavily_search_response_empty",
            "tavily_search_response_with_answer",
        ],
        "extract": [
            "tavily_extract_response_basic",
            "tavily_extract_response_advanced",
            "tavily_extract_response_multiple_urls",
            "tavily_extract_response_partial_failure",
            "tavily_extract_response_with_images",
            "tavily_extract_response_empty",
        ],
        "errors": [
            "tavily_error_response_401",
            "tavily_error_response_429",
            "tavily_error_response_500",
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
