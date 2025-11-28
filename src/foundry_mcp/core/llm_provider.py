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
# OpenAI Provider Implementation
# =============================================================================


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation.

    Supports GPT-4, GPT-3.5-turbo, and embedding models via the OpenAI API.

    Attributes:
        name: Provider identifier ('openai')
        default_model: Default chat model ('gpt-4')
        default_embedding_model: Default embedding model
        api_key: OpenAI API key
        organization: Optional organization ID
        base_url: API base URL (for proxies/Azure)

    Example:
        provider = OpenAIProvider(api_key="sk-...")
        response = await provider.chat(ChatRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello!")]
        ))
    """

    name: str = "openai"
    default_model: str = "gpt-4"
    default_embedding_model: str = "text-embedding-3-small"

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
    ):
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: API base URL (defaults to OpenAI's API)
            default_model: Override default chat model
            default_embedding_model: Override default embedding model
        """
        import os

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        self.base_url = base_url or "https://api.openai.com/v1"

        if default_model:
            self.default_model = default_model
        if default_embedding_model:
            self.default_embedding_model = default_embedding_model

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise LLMError(
                    "openai package not installed. Install with: pip install openai",
                    provider=self.name,
                )

            if not self.api_key:
                raise AuthenticationError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key.",
                    provider=self.name,
                )

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )

        return self._client

    def _handle_api_error(self, error: Exception) -> None:
        """Convert OpenAI errors to LLMError types."""
        error_str = str(error)
        error_type = type(error).__name__

        if "rate_limit" in error_str.lower() or error_type == "RateLimitError":
            # Try to extract retry-after
            retry_after = None
            if hasattr(error, "response"):
                retry_after = getattr(error.response.headers, "get", lambda x: None)(
                    "retry-after"
                )
                if retry_after:
                    try:
                        retry_after = float(retry_after)
                    except ValueError:
                        retry_after = None
            raise RateLimitError(error_str, provider=self.name, retry_after=retry_after)

        if "authentication" in error_str.lower() or error_type == "AuthenticationError":
            raise AuthenticationError(error_str, provider=self.name)

        if "invalid" in error_str.lower() or error_type == "BadRequestError":
            raise InvalidRequestError(error_str, provider=self.name)

        if "not found" in error_str.lower() or error_type == "NotFoundError":
            raise ModelNotFoundError(error_str, provider=self.name)

        if "content_filter" in error_str.lower():
            raise ContentFilterError(error_str, provider=self.name)

        # Generic error
        raise LLMError(error_str, provider=self.name, retryable=True)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion using OpenAI's API.

        Note: Uses chat completions API internally as legacy completions
        are deprecated for most models.
        """
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            # Use chat completions API (legacy completions deprecated)
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            choice = response.choices[0]
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return CompletionResponse(
                text=choice.message.content or "",
                finish_reason=self._map_finish_reason(choice.finish_reason),
                usage=usage,
                model=response.model,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise  # Unreachable but keeps type checker happy

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion using OpenAI's API."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            # Convert messages to OpenAI format
            messages = [msg.to_dict() for msg in request.messages]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if request.stop:
                kwargs["stop"] = request.stop
            if request.tools:
                kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice

            response = await client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            message = choice.message

            # Parse tool calls if present
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in message.tool_calls
                ]

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return ChatResponse(
                message=ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=message.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=self._map_finish_reason(choice.finish_reason),
                usage=usage,
                model=response.model,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using OpenAI's API."""
        self.validate_request(request)
        client = self._get_client()
        model = request.model or self.default_embedding_model

        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "input": request.texts,
            }

            if request.dimensions:
                kwargs["dimensions"] = request.dimensions

            response = await client.embeddings.create(**kwargs)

            embeddings = [item.embedding for item in response.data]

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                usage=usage,
                model=response.model,
                dimensions=len(embeddings[0]) if embeddings else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream chat completion tokens from OpenAI."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            messages = [msg.to_dict() for msg in request.messages]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True,
            }

            if request.stop:
                kwargs["stop"] = request.stop

            stream = await client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield ChatResponse(
                        message=ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content=delta.content,
                        ),
                        finish_reason=self._map_finish_reason(choice.finish_reason) if choice.finish_reason else FinishReason.STOP,
                        model=chunk.model,
                    )

        except Exception as e:
            self._handle_api_error(e)
            raise

    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Map OpenAI finish reason to FinishReason enum."""
        if reason is None:
            return FinishReason.STOP

        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL,
            "function_call": FinishReason.TOOL_CALL,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason, FinishReason.STOP)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens using tiktoken (if available)."""
        try:
            import tiktoken

            model_name = model or self.default_model
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            return len(encoding.encode(text))
        except ImportError:
            # Fall back to rough estimate
            return super().count_tokens(text, model)


# =============================================================================
# Anthropic Provider Implementation
# =============================================================================


class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation.

    Supports Claude 3 models (opus, sonnet, haiku) via the Anthropic API.

    Attributes:
        name: Provider identifier ('anthropic')
        default_model: Default chat model ('claude-3-sonnet-20240229')
        api_key: Anthropic API key
        base_url: API base URL (for proxies)
        max_tokens_default: Default max tokens for responses

    Example:
        provider = AnthropicProvider(api_key="sk-ant-...")
        response = await provider.chat(ChatRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello!")]
        ))

    Note:
        Anthropic does not support embeddings. The embed() method will raise
        an error if called.
    """

    name: str = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"
    max_tokens_default: int = 4096

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        max_tokens_default: Optional[int] = None,
    ):
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            base_url: API base URL (defaults to Anthropic's API)
            default_model: Override default chat model
            max_tokens_default: Override default max tokens
        """
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url

        if default_model:
            self.default_model = default_model
        if max_tokens_default:
            self.max_tokens_default = max_tokens_default

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the Anthropic client (lazy initialization)."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise LLMError(
                    "anthropic package not installed. Install with: pip install anthropic",
                    provider=self.name,
                )

            if not self.api_key:
                raise AuthenticationError(
                    "Anthropic API key not provided. Set ANTHROPIC_API_KEY or pass api_key.",
                    provider=self.name,
                )

            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = AsyncAnthropic(**kwargs)

        return self._client

    def _handle_api_error(self, error: Exception) -> None:
        """Convert Anthropic errors to LLMError types."""
        error_str = str(error)
        error_type = type(error).__name__

        if "rate_limit" in error_str.lower() or error_type == "RateLimitError":
            retry_after = None
            if hasattr(error, "response") and error.response:
                retry_after_str = error.response.headers.get("retry-after")
                if retry_after_str:
                    try:
                        retry_after = float(retry_after_str)
                    except ValueError:
                        pass
            raise RateLimitError(error_str, provider=self.name, retry_after=retry_after)

        if "authentication" in error_str.lower() or error_type == "AuthenticationError":
            raise AuthenticationError(error_str, provider=self.name)

        if "invalid" in error_str.lower() or error_type == "BadRequestError":
            raise InvalidRequestError(error_str, provider=self.name)

        if "not found" in error_str.lower() or error_type == "NotFoundError":
            raise ModelNotFoundError(error_str, provider=self.name)

        # Generic error
        raise LLMError(error_str, provider=self.name, retryable=True)

    def _convert_messages(self, messages: List[ChatMessage]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert ChatMessages to Anthropic format, extracting system message.

        Anthropic requires system message as a separate parameter.

        Returns:
            Tuple of (system_message, messages_list)
        """
        system_message = None
        converted = []

        for msg in messages:
            if msg.role == ChatRole.SYSTEM:
                # Anthropic takes system as separate param
                system_message = msg.content
            elif msg.role == ChatRole.TOOL:
                # Tool results in Anthropic format
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
            elif msg.role == ChatRole.ASSISTANT and msg.tool_calls:
                # Assistant message with tool calls
                content: List[Dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    import json
                    content.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": json.loads(tc.arguments) if tc.arguments else {},
                    })
                converted.append({"role": "assistant", "content": content})
            else:
                # Regular user/assistant message
                converted.append({
                    "role": msg.role.value,
                    "content": msg.content or "",
                })

        return system_message, converted

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion using Anthropic's API.

        Uses the messages API internally.
        """
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            response = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens or self.max_tokens_default,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_sequences=request.stop or [],
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return CompletionResponse(
                text=text,
                finish_reason=self._map_stop_reason(response.stop_reason),
                usage=usage,
                model=response.model,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion using Anthropic's API."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            system_message, messages = self._convert_messages(request.messages)

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens_default,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if system_message:
                kwargs["system"] = system_message
            if request.stop:
                kwargs["stop_sequences"] = request.stop
            if request.tools:
                # Convert OpenAI-style tools to Anthropic format
                kwargs["tools"] = self._convert_tools(request.tools)

            response = await client.messages.create(**kwargs)

            # Parse response content
            content_text = None
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content_text = (content_text or "") + block.text
                elif block.type == "tool_use":
                    import json
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                    ))

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )

            return ChatResponse(
                message=ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=content_text,
                    tool_calls=tool_calls if tool_calls else None,
                ),
                finish_reason=self._map_stop_reason(response.stop_reason),
                usage=usage,
                model=response.model,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Anthropic does not support embeddings.

        Raises:
            LLMError: Always, as embeddings are not supported
        """
        raise LLMError(
            "Anthropic does not support embeddings. Use OpenAI or a local embedding model.",
            provider=self.name,
            retryable=False,
        )

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream chat completion tokens from Anthropic."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            system_message, messages = self._convert_messages(request.messages)

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens or self.max_tokens_default,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if system_message:
                kwargs["system"] = system_message
            if request.stop:
                kwargs["stop_sequences"] = request.stop

            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield ChatResponse(
                        message=ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content=text,
                        ),
                        finish_reason=FinishReason.STOP,
                        model=model,
                    )

        except Exception as e:
            self._handle_api_error(e)
            raise

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tool definitions to Anthropic format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                converted.append({
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        return converted

    def _map_stop_reason(self, reason: Optional[str]) -> FinishReason:
        """Map Anthropic stop reason to FinishReason enum."""
        if reason is None:
            return FinishReason.STOP

        mapping = {
            "end_turn": FinishReason.STOP,
            "stop_sequence": FinishReason.STOP,
            "max_tokens": FinishReason.LENGTH,
            "tool_use": FinishReason.TOOL_CALL,
        }
        return mapping.get(reason, FinishReason.STOP)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Estimate tokens for Anthropic models.

        Uses anthropic's token counting if available, otherwise rough estimate.
        """
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key or "dummy")
            return client.count_tokens(text)
        except (ImportError, Exception):
            # Rough estimate: ~4 characters per token
            return len(text) // 4


# =============================================================================
# Local Provider Implementation (Ollama/llama.cpp)
# =============================================================================


class LocalProvider(LLMProvider):
    """Local LLM provider using Ollama or llama.cpp compatible API.

    Supports local models via Ollama's OpenAI-compatible API endpoint.

    Attributes:
        name: Provider identifier ('local')
        default_model: Default model ('llama3.2')
        base_url: Local API endpoint (default: http://localhost:11434/v1)

    Example:
        # Using Ollama (default)
        provider = LocalProvider()
        response = await provider.chat(ChatRequest(
            messages=[ChatMessage(role=ChatRole.USER, content="Hello!")]
        ))

        # Custom endpoint
        provider = LocalProvider(base_url="http://localhost:8080/v1")
    """

    name: str = "local"
    default_model: str = "llama3.2"
    default_embedding_model: str = "nomic-embed-text"

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the local provider.

        Args:
            base_url: API base URL (defaults to Ollama's OpenAI-compatible endpoint)
            default_model: Override default chat model
            default_embedding_model: Override default embedding model
            api_key: Optional API key (some local servers may require it)
        """
        self.base_url = base_url or "http://localhost:11434/v1"
        self.api_key = api_key or "ollama"  # Ollama accepts any key

        if default_model:
            self.default_model = default_model
        if default_embedding_model:
            self.default_embedding_model = default_embedding_model

        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Get or create the OpenAI-compatible client for local server."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise LLMError(
                    "openai package not installed. Install with: pip install openai",
                    provider=self.name,
                )

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

        return self._client

    def _handle_api_error(self, error: Exception) -> None:
        """Convert API errors to LLMError types."""
        error_str = str(error)

        if "connection" in error_str.lower() or "refused" in error_str.lower():
            raise LLMError(
                f"Cannot connect to local server at {self.base_url}. "
                "Ensure Ollama is running: ollama serve",
                provider=self.name,
                retryable=True,
            )

        if "model" in error_str.lower() and "not found" in error_str.lower():
            raise ModelNotFoundError(
                f"Model not found. Pull it first: ollama pull {self.default_model}",
                provider=self.name,
            )

        raise LLMError(error_str, provider=self.name, retryable=True)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a text completion using local model."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            choice = response.choices[0]
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return CompletionResponse(
                text=choice.message.content or "",
                finish_reason=self._map_finish_reason(choice.finish_reason),
                usage=usage,
                model=response.model,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Generate a chat completion using local model."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            messages = [msg.to_dict() for msg in request.messages]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            }

            if request.stop:
                kwargs["stop"] = request.stop
            if request.tools:
                kwargs["tools"] = request.tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice

            response = await client.chat.completions.create(**kwargs)

            choice = response.choices[0]
            message = choice.message

            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    )
                    for tc in message.tool_calls
                ]

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return ChatResponse(
                message=ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=message.content,
                    tool_calls=tool_calls,
                ),
                finish_reason=self._map_finish_reason(choice.finish_reason),
                usage=usage,
                model=response.model,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings using local model."""
        self.validate_request(request)
        client = self._get_client()
        model = request.model or self.default_embedding_model

        try:
            response = await client.embeddings.create(
                model=model,
                input=request.texts,
            )

            embeddings = [item.embedding for item in response.data]

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                usage=usage,
                model=response.model,
                dimensions=len(embeddings[0]) if embeddings else None,
            )

        except Exception as e:
            self._handle_api_error(e)
            raise

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[ChatResponse]:
        """Stream chat completion tokens from local model."""
        self.validate_request(request)
        client = self._get_client()
        model = self.get_model(request.model)

        try:
            messages = [msg.to_dict() for msg in request.messages]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": True,
            }

            if request.stop:
                kwargs["stop"] = request.stop

            stream = await client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    yield ChatResponse(
                        message=ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content=delta.content,
                        ),
                        finish_reason=self._map_finish_reason(choice.finish_reason) if choice.finish_reason else FinishReason.STOP,
                        model=chunk.model,
                    )

        except Exception as e:
            self._handle_api_error(e)
            raise

    def _map_finish_reason(self, reason: Optional[str]) -> FinishReason:
        """Map finish reason to FinishReason enum."""
        if reason is None:
            return FinishReason.STOP

        mapping = {
            "stop": FinishReason.STOP,
            "length": FinishReason.LENGTH,
            "tool_calls": FinishReason.TOOL_CALL,
        }
        return mapping.get(reason, FinishReason.STOP)

    async def health_check(self) -> bool:
        """Check if local server is accessible."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Check Ollama-style health endpoint
                response = await client.get(
                    self.base_url.replace("/v1", "") + "/api/tags",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
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
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "LocalProvider",
]
