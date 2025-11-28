"""
LLM Provider abstraction for foundry-mcp.

Provides a unified interface for interacting with different LLM providers
(OpenAI, Anthropic, local models) with consistent error handling,
rate limiting, and observability.

Example:
    from foundry_mcp.core.llm_provider import (
        LLMProvider, ChatMessage, ChatRole, CompletionRequest
    )

    class MyProvider(LLMProvider):
        async def complete(self, request: CompletionRequest) -> CompletionResponse:
            # Implementation
            pass

        async def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
            # Implementation
            pass

        async def embed(self, texts: List[str], **kwargs) -> EmbeddingResponse:
            # Implementation
            pass
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ChatRole(str, Enum):
    """Role of a message in a chat conversation.

    SYSTEM: System instructions/context
    USER: User input
    ASSISTANT: Model response
    TOOL: Tool/function call result
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    """Reason why the model stopped generating.

    STOP: Natural completion (hit stop sequence or end)
    LENGTH: Hit max_tokens limit
    TOOL_CALL: Model wants to call a tool/function
    CONTENT_FILTER: Filtered due to content policy
    ERROR: Generation error occurred
    """

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


# =============================================================================
# Data Classes - Messages
# =============================================================================


@dataclass
class ToolCall:
    """A tool/function call requested by the model.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool/function to call
        arguments: JSON-encoded arguments for the call
    """

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class ChatMessage:
    """A message in a chat conversation.

    Attributes:
        role: The role of the message sender
        content: The text content of the message
        name: Optional name for the sender (for multi-user chats)
        tool_calls: List of tool calls if role is ASSISTANT
        tool_call_id: ID of the tool call this responds to (if role is TOOL)
    """

    role: ChatRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        result: Dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            result["content"] = self.content
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


# =============================================================================
# Data Classes - Requests
# =============================================================================


@dataclass
class CompletionRequest:
    """Request for text completion (non-chat).

    Attributes:
        prompt: The prompt to complete
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        stop: Stop sequences
        model: Model identifier (optional, uses provider default)
    """

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    model: Optional[str] = None


@dataclass
class ChatRequest:
    """Request for chat completion.

    Attributes:
        messages: The conversation messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling parameter
        stop: Stop sequences
        model: Model identifier (optional, uses provider default)
        tools: Tool/function definitions for function calling
        tool_choice: How to handle tool selection ('auto', 'none', or specific)
    """

    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop: Optional[List[str]] = None
    model: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class EmbeddingRequest:
    """Request for text embeddings.

    Attributes:
        texts: List of texts to embed
        model: Model identifier (optional, uses provider default)
        dimensions: Output dimension size (if supported by model)
    """

    texts: List[str]
    model: Optional[str] = None
    dimensions: Optional[int] = None


# =============================================================================
# Data Classes - Responses
# =============================================================================


@dataclass
class TokenUsage:
    """Token usage statistics.

    Attributes:
        prompt_tokens: Tokens in the input
        completion_tokens: Tokens in the output
        total_tokens: Total tokens used
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResponse:
    """Response from text completion.

    Attributes:
        text: The generated text
        finish_reason: Why generation stopped
        usage: Token usage statistics
        model: Model that generated the response
        raw_response: Original API response (for debugging)
    """

    text: str
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class ChatResponse:
    """Response from chat completion.

    Attributes:
        message: The assistant's response message
        finish_reason: Why generation stopped
        usage: Token usage statistics
        model: Model that generated the response
        raw_response: Original API response (for debugging)
    """

    message: ChatMessage
    finish_reason: FinishReason = FinishReason.STOP
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding request.

    Attributes:
        embeddings: List of embedding vectors
        usage: Token usage statistics
        model: Model that generated the embeddings
        dimensions: Dimension size of embeddings
    """

    embeddings: List[List[float]]
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: Optional[str] = None
    dimensions: Optional[int] = None


# =============================================================================
# Exceptions
# =============================================================================


class LLMError(Exception):
    """Base exception for LLM operations.

    Attributes:
        message: Human-readable error description
        provider: Name of the provider that raised the error
        retryable: Whether the operation can be retried
        status_code: HTTP status code if applicable
    """

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


class RateLimitError(LLMError):
    """Rate limit exceeded error.

    Attributes:
        retry_after: Seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        provider: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, provider=provider, retryable=True, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(LLMError):
    """Authentication failed error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=401)


class InvalidRequestError(LLMError):
    """Invalid request error (bad parameters, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        param: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=400)
        self.param = param


class ModelNotFoundError(LLMError):
    """Requested model not found or not accessible."""

    def __init__(
        self,
        message: str,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=404)
        self.model = model


class ContentFilterError(LLMError):
    """Content was filtered due to policy violation."""

    def __init__(
        self,
        message: str = "Content filtered",
        *,
        provider: Optional[str] = None,
    ):
        super().__init__(message, provider=provider, retryable=False, status_code=400)


# =============================================================================
# Abstract Base Class
# =============================================================================


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Defines the interface that all LLM provider implementations must follow.
    Provides consistent methods for completion, chat, and embedding operations.

    Attributes:
        name: Provider name (e.g., 'openai', 'anthropic', 'local')
        default_model: Default model to use if not specified in requests

    Example:
        class OpenAIProvider(LLMProvider):
            name = "openai"
            default_model = "gpt-4"

            async def complete(self, request: CompletionRequest) -> CompletionResponse:
                # Call OpenAI API
                pass
    """

    name: str = "base"
    default_model: str = ""

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion.

        Args:
            request: Completion request with prompt and parameters

        Returns:
            CompletionResponse with generated text

        Raises:
            LLMError: On API or generation errors
            RateLimitError: If rate limited
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion.

        Args:
            request: Chat request with messages and parameters

        Returns:
            ChatResponse with assistant message

        Raises:
            LLMError: On API or generation errors
            RateLimitError: If rate limited
            AuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for texts.

        Args:
            request: Embedding request with texts

        Returns:
            EmbeddingResponse with embedding vectors

        Raises:
            LLMError: On API or generation errors
            RateLimitError: If rate limited
            AuthenticationError: If authentication fails
        """
        pass

    async def stream_chat(
        self, request: ChatRequest
    ) -> AsyncIterator[ChatResponse]:
        """Stream chat completion tokens.

        Default implementation yields a single complete response.
        Providers can override for true streaming support.

        Args:
            request: Chat request with messages and parameters

        Yields:
            ChatResponse chunks as they are generated

        Raises:
            LLMError: On API or generation errors
        """
        response = await self.chat(request)
        yield response

    async def stream_complete(
        self, request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion tokens.

        Default implementation yields a single complete response.
        Providers can override for true streaming support.

        Args:
            request: Completion request with prompt and parameters

        Yields:
            CompletionResponse chunks as they are generated

        Raises:
            LLMError: On API or generation errors
        """
        response = await self.complete(request)
        yield response

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text.

        Default implementation provides a rough estimate.
        Providers should override with accurate tokenization.

        Args:
            text: Text to count tokens for
            model: Model to use for tokenization (optional)

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    def validate_request(self, request: Union[CompletionRequest, ChatRequest, EmbeddingRequest]) -> None:
        """Validate a request before sending.

        Override to add provider-specific validation.

        Args:
            request: Request to validate

        Raises:
            InvalidRequestError: If request is invalid
        """
        if isinstance(request, CompletionRequest):
            if not request.prompt:
                raise InvalidRequestError("Prompt cannot be empty", provider=self.name)
            if request.max_tokens < 1:
                raise InvalidRequestError("max_tokens must be positive", provider=self.name, param="max_tokens")

        elif isinstance(request, ChatRequest):
            if not request.messages:
                raise InvalidRequestError("Messages cannot be empty", provider=self.name)
            if request.max_tokens < 1:
                raise InvalidRequestError("max_tokens must be positive", provider=self.name, param="max_tokens")

        elif isinstance(request, EmbeddingRequest):
            if not request.texts:
                raise InvalidRequestError("Texts cannot be empty", provider=self.name)

    def get_model(self, requested: Optional[str] = None) -> str:
        """Get the model to use for a request.

        Args:
            requested: Explicitly requested model (optional)

        Returns:
            Model identifier to use
        """
        return requested or self.default_model

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.

        Default implementation tries a minimal chat request.
        Providers can override with more efficient checks.

        Returns:
            True if provider is healthy, False otherwise
        """
        try:
            request = ChatRequest(
                messages=[ChatMessage(role=ChatRole.USER, content="ping")],
                max_tokens=1,
            )
            await self.chat(request)
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.name}: {e}")
            return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ChatRole",
    "FinishReason",
    # Data Classes
    "ToolCall",
    "ChatMessage",
    "CompletionRequest",
    "ChatRequest",
    "EmbeddingRequest",
    "TokenUsage",
    "CompletionResponse",
    "ChatResponse",
    "EmbeddingResponse",
    # Exceptions
    "LLMError",
    "RateLimitError",
    "AuthenticationError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ContentFilterError",
    # Provider ABC
    "LLMProvider",
]
