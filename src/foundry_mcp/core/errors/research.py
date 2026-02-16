"""Research workflow error classes.

Moved from foundry_mcp.core.research.pdf_extractor,
foundry_mcp.core.research.summarization, foundry_mcp.core.research.context_budget,
and foundry_mcp.core.research.providers.tavily_extract for centralized error management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.core.research.summarization import SummarizationLevel


# =============================================================================
# PDF Extraction Errors
# =============================================================================


class PDFSecurityError(Exception):
    """Base exception for PDF security violations."""
    pass


class SSRFError(PDFSecurityError):
    """Raised when SSRF protection blocks a request."""
    pass


class InvalidPDFError(PDFSecurityError):
    """Raised when PDF validation fails (magic bytes, content-type)."""
    pass


class PDFSizeError(PDFSecurityError):
    """Raised when PDF exceeds size limits."""
    pass


# =============================================================================
# Summarization Errors
# =============================================================================


class SummarizationError(Exception):
    """Base exception for summarization errors."""

    pass


class ProviderExhaustedError(SummarizationError):
    """Raised when all providers in the chain have failed."""

    def __init__(self, errors: list[tuple[str, Exception]]):
        self.errors = errors
        provider_msgs = [f"{p}: {e}" for p, e in errors]
        super().__init__(
            f"All summarization providers failed: {', '.join(provider_msgs)}"
        )


class SummarizationValidationError(SummarizationError):
    """Raised when summarization output fails validation."""

    def __init__(self, message: str, level: SummarizationLevel, missing_fields: list[str]):
        self.level = level
        self.missing_fields = missing_fields
        super().__init__(f"{message}: missing {missing_fields} for {level.value} level")


# =============================================================================
# Context Budget Errors
# =============================================================================


class ProtectedContentOverflowError(Exception):
    """Raised when protected content exceeds budget even after headline compression.

    This error indicates that the protected content is too large to fit within
    the available token budget, even after applying the most aggressive
    compression (headline level, ~10% of original).

    Attributes:
        protected_tokens: Total tokens required by protected content at headline level
        budget: Available token budget
        item_ids: List of protected item IDs that couldn't fit
        remediation: Suggested remediation steps
    """

    def __init__(
        self,
        protected_tokens: int,
        budget: int,
        item_ids: list[str],
        remediation: Optional[str] = None,
    ):
        self.protected_tokens = protected_tokens
        self.budget = budget
        self.item_ids = item_ids
        self.remediation = remediation or (
            f"Protected content requires {protected_tokens} tokens at headline level, "
            f"but only {budget} tokens available. "
            "Options: (1) Increase context budget, (2) Reduce number of protected items, "
            "(3) Mark fewer items as protected, (4) Use a model with larger context window."
        )
        super().__init__(self.remediation)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_type": "protected_content_overflow",
            "protected_tokens": self.protected_tokens,
            "budget": self.budget,
            "item_ids": self.item_ids,
            "remediation": self.remediation,
        }


# =============================================================================
# URL Validation Errors
# =============================================================================


class UrlValidationError(ValueError):
    """Raised when URL validation fails (SSRF protection).

    Attributes:
        url: The URL that failed validation.
        reason: Human-readable explanation of the failure.
        error_code: Machine-readable error code (INVALID_URL or BLOCKED_HOST).
    """

    def __init__(self, url: str, reason: str, error_code: str = "INVALID_URL"):
        self.url = url
        self.reason = reason
        self.error_code = error_code
        super().__init__(f"URL validation failed for {url!r}: {reason}")
