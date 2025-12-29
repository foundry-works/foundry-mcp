"""
Unit tests for LLM provider abstraction and config.

Tests cover:
- LLMProvider base class and concrete implementations (OpenAI, Anthropic, Local)
- LLMConfig parsing from TOML and environment variables
- WorkflowConfig parsing and validation
- Error handling and exception hierarchy
- Data classes for requests and responses
"""

import os
import pytest
from unittest.mock import patch

# LLM Provider imports
from foundry_mcp.core.llm_provider import (
    # Enums
    ChatRole,
    FinishReason,
    # Data Classes
    ToolCall,
    ChatMessage,
    CompletionRequest,
    ChatRequest,
    EmbeddingRequest,
    TokenUsage,
    CompletionResponse,
    ChatResponse,
    EmbeddingResponse,
    # Exceptions
    LLMError,
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ContentFilterError,
    # Providers
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    LocalProvider,
)

# LLM Config imports
from foundry_mcp.core.llm_config import (
    LLMProviderType,
    LLMConfig,
    get_llm_config,
    set_llm_config,
    reset_llm_config,
    DEFAULT_MODELS,
    API_KEY_ENV_VARS,
    WorkflowMode,
    WorkflowConfig,
    get_workflow_config,
    set_workflow_config,
    reset_workflow_config,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestChatRole:
    """Tests for ChatRole enum."""

    def test_values(self):
        """Test ChatRole enum values."""
        assert ChatRole.SYSTEM.value == "system"
        assert ChatRole.USER.value == "user"
        assert ChatRole.ASSISTANT.value == "assistant"
        assert ChatRole.TOOL.value == "tool"

    def test_str_enum(self):
        """Test ChatRole is a string enum."""
        assert isinstance(ChatRole.USER, str)
        assert ChatRole.USER == "user"


class TestFinishReason:
    """Tests for FinishReason enum."""

    def test_values(self):
        """Test FinishReason enum values."""
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_CALL.value == "tool_call"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.ERROR.value == "error"


# =============================================================================
# Data Class Tests
# =============================================================================


class TestToolCall:
    """Tests for ToolCall data class."""

    def test_creation(self):
        """Test ToolCall creation."""
        tc = ToolCall(id="call_123", name="get_weather", arguments='{"city": "NYC"}')
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == '{"city": "NYC"}'


class TestChatMessage:
    """Tests for ChatMessage data class."""

    def test_simple_message(self):
        """Test simple message creation."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello!")
        assert msg.role == ChatRole.USER
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_calls is None

    def test_to_dict_simple(self):
        """Test to_dict for simple message."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello!")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello!"}

    def test_to_dict_with_name(self):
        """Test to_dict with name."""
        msg = ChatMessage(role=ChatRole.USER, content="Hello!", name="Alice")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello!", "name": "Alice"}

    def test_to_dict_with_tool_calls(self):
        """Test to_dict with tool calls."""
        tc = ToolCall(id="call_1", name="func", arguments='{}')
        msg = ChatMessage(role=ChatRole.ASSISTANT, content=None, tool_calls=[tc])
        result = msg.to_dict()
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_1"
        assert result["tool_calls"][0]["type"] == "function"
        assert result["tool_calls"][0]["function"]["name"] == "func"

    def test_to_dict_with_tool_call_id(self):
        """Test to_dict with tool_call_id."""
        msg = ChatMessage(
            role=ChatRole.TOOL, content="Result", tool_call_id="call_1"
        )
        result = msg.to_dict()
        assert result == {"role": "tool", "content": "Result", "tool_call_id": "call_1"}


class TestCompletionRequest:
    """Tests for CompletionRequest data class."""

    def test_defaults(self):
        """Test default values."""
        req = CompletionRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.top_p == 1.0
        assert req.stop is None
        assert req.model is None

    def test_custom_values(self):
        """Test custom values."""
        req = CompletionRequest(
            prompt="Test",
            max_tokens=100,
            temperature=0.5,
            stop=["END"],
            model="gpt-4",
        )
        assert req.max_tokens == 100
        assert req.temperature == 0.5
        assert req.stop == ["END"]
        assert req.model == "gpt-4"


class TestChatRequest:
    """Tests for ChatRequest data class."""

    def test_defaults(self):
        """Test default values."""
        msgs = [ChatMessage(role=ChatRole.USER, content="Hi")]
        req = ChatRequest(messages=msgs)
        assert len(req.messages) == 1
        assert req.max_tokens == 1024
        assert req.temperature == 0.7

    def test_with_tools(self):
        """Test with tools."""
        msgs = [ChatMessage(role=ChatRole.USER, content="Hi")]
        tools = [{"type": "function", "function": {"name": "test"}}]
        req = ChatRequest(messages=msgs, tools=tools, tool_choice="auto")
        assert req.tools == tools
        assert req.tool_choice == "auto"


class TestEmbeddingRequest:
    """Tests for EmbeddingRequest data class."""

    def test_creation(self):
        """Test creation."""
        req = EmbeddingRequest(texts=["Hello", "World"])
        assert req.texts == ["Hello", "World"]
        assert req.model is None
        assert req.dimensions is None


class TestTokenUsage:
    """Tests for TokenUsage data class."""

    def test_defaults(self):
        """Test default values are zero."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestCompletionResponse:
    """Tests for CompletionResponse data class."""

    def test_creation(self):
        """Test creation."""
        resp = CompletionResponse(text="Hello!")
        assert resp.text == "Hello!"
        assert resp.finish_reason == FinishReason.STOP
        assert resp.usage.total_tokens == 0


class TestChatResponse:
    """Tests for ChatResponse data class."""

    def test_creation(self):
        """Test creation."""
        msg = ChatMessage(role=ChatRole.ASSISTANT, content="Hi!")
        resp = ChatResponse(message=msg)
        assert resp.message.content == "Hi!"
        assert resp.finish_reason == FinishReason.STOP


class TestEmbeddingResponse:
    """Tests for EmbeddingResponse data class."""

    def test_creation(self):
        """Test creation."""
        resp = EmbeddingResponse(embeddings=[[0.1, 0.2], [0.3, 0.4]])
        assert len(resp.embeddings) == 2
        assert resp.dimensions is None


# =============================================================================
# Exception Tests
# =============================================================================


class TestLLMError:
    """Tests for LLMError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        err = LLMError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.provider is None
        assert err.retryable is False
        assert err.status_code is None

    def test_with_metadata(self):
        """Test error with metadata."""
        err = LLMError(
            "Error",
            provider="openai",
            retryable=True,
            status_code=500,
        )
        assert err.provider == "openai"
        assert err.retryable is True
        assert err.status_code == 500


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_creation(self):
        """Test creation."""
        err = RateLimitError(provider="openai", retry_after=30.0)
        assert err.retryable is True
        assert err.status_code == 429
        assert err.retry_after == 30.0


class TestAuthenticationError:
    """Tests for AuthenticationError exception."""

    def test_creation(self):
        """Test creation."""
        err = AuthenticationError(provider="openai")
        assert err.retryable is False
        assert err.status_code == 401


class TestInvalidRequestError:
    """Tests for InvalidRequestError exception."""

    def test_creation(self):
        """Test creation."""
        err = InvalidRequestError("Bad param", param="temperature")
        assert err.retryable is False
        assert err.status_code == 400
        assert err.param == "temperature"


class TestModelNotFoundError:
    """Tests for ModelNotFoundError exception."""

    def test_creation(self):
        """Test creation."""
        err = ModelNotFoundError("Not found", model="gpt-5")
        assert err.status_code == 404
        assert err.model == "gpt-5"


class TestContentFilterError:
    """Tests for ContentFilterError exception."""

    def test_creation(self):
        """Test creation."""
        err = ContentFilterError(provider="openai")
        assert err.retryable is False
        assert err.status_code == 400


# =============================================================================
# LLMProvider Base Class Tests
# =============================================================================


class TestLLMProviderBase:
    """Tests for LLMProvider base class methods."""

    class MockProvider(LLMProvider):
        """Mock provider for testing base class."""

        name = "mock"
        default_model = "mock-model"

        async def complete(self, request):
            return CompletionResponse(text="completed")

        async def chat(self, request):
            return ChatResponse(
                message=ChatMessage(role=ChatRole.ASSISTANT, content="chatted")
            )

        async def embed(self, request):
            return EmbeddingResponse(embeddings=[[0.1]])

    def test_count_tokens_default(self):
        """Test default token counting (rough estimate)."""
        provider = self.MockProvider()
        # ~4 chars per token
        count = provider.count_tokens("Hello world!")
        assert count == 3  # 12 chars / 4

    def test_get_model_with_default(self):
        """Test get_model returns default."""
        provider = self.MockProvider()
        assert provider.get_model() == "mock-model"

    def test_get_model_with_requested(self):
        """Test get_model returns requested."""
        provider = self.MockProvider()
        assert provider.get_model("custom-model") == "custom-model"

    def test_validate_request_empty_prompt(self):
        """Test validation rejects empty prompt."""
        provider = self.MockProvider()
        req = CompletionRequest(prompt="")
        with pytest.raises(InvalidRequestError, match="empty"):
            provider.validate_request(req)

    def test_validate_request_invalid_max_tokens(self):
        """Test validation rejects invalid max_tokens."""
        provider = self.MockProvider()
        req = CompletionRequest(prompt="Hello", max_tokens=0)
        with pytest.raises(InvalidRequestError, match="max_tokens"):
            provider.validate_request(req)

    def test_validate_request_empty_messages(self):
        """Test validation rejects empty messages."""
        provider = self.MockProvider()
        req = ChatRequest(messages=[])
        with pytest.raises(InvalidRequestError, match="empty"):
            provider.validate_request(req)

    def test_validate_request_empty_texts(self):
        """Test validation rejects empty texts."""
        provider = self.MockProvider()
        req = EmbeddingRequest(texts=[])
        with pytest.raises(InvalidRequestError, match="empty"):
            provider.validate_request(req)

    @pytest.mark.asyncio
    async def test_stream_chat_default(self):
        """Test default stream_chat yields single response."""
        provider = self.MockProvider()
        msgs = [ChatMessage(role=ChatRole.USER, content="Hi")]
        req = ChatRequest(messages=msgs)

        chunks = []
        async for chunk in provider.stream_chat(req):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].message.content == "chatted"

    @pytest.mark.asyncio
    async def test_stream_complete_default(self):
        """Test default stream_complete yields single response."""
        provider = self.MockProvider()
        req = CompletionRequest(prompt="Hello")

        chunks = []
        async for chunk in provider.stream_complete(req):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].text == "completed"


# =============================================================================
# OpenAIProvider Tests
# =============================================================================

# Check if openai package is available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai package not installed")
class TestOpenAIProvider:
    """Tests for OpenAIProvider."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider()
            assert provider.name == "openai"
            assert provider.default_model == "gpt-4"
            assert provider.api_key == "test-key"

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            provider = OpenAIProvider(default_model="gpt-3.5-turbo")
            assert provider.default_model == "gpt-3.5-turbo"

    def test_get_client_no_key(self):
        """Test _get_client raises without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing key
            provider = OpenAIProvider(api_key=None)
            provider.api_key = None
            with pytest.raises(AuthenticationError):
                provider._get_client()

    def test_map_finish_reason(self):
        """Test finish reason mapping."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            provider = OpenAIProvider()
            assert provider._map_finish_reason("stop") == FinishReason.STOP
            assert provider._map_finish_reason("length") == FinishReason.LENGTH
            assert provider._map_finish_reason("tool_calls") == FinishReason.TOOL_CALL
            assert provider._map_finish_reason(None) == FinishReason.STOP
            assert provider._map_finish_reason("unknown") == FinishReason.STOP

    def test_handle_api_error_rate_limit(self):
        """Test API error handling for rate limit."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            provider = OpenAIProvider()
            with pytest.raises(RateLimitError):
                provider._handle_api_error(Exception("rate_limit exceeded"))

    def test_handle_api_error_auth(self):
        """Test API error handling for auth."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=False):
            provider = OpenAIProvider()
            with pytest.raises(AuthenticationError):
                provider._handle_api_error(Exception("authentication failed"))


# =============================================================================
# AnthropicProvider Tests
# =============================================================================


class TestAnthropicProvider:
    """Tests for AnthropicProvider."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            provider = AnthropicProvider()
            assert provider.name == "anthropic"
            assert provider.default_model == "claude-sonnet-4-20250514"
            assert provider.api_key == "test-key"

    def test_embed_not_supported(self):
        """Test embed raises for Anthropic."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            provider = AnthropicProvider()

            @pytest.mark.asyncio
            async def test():
                with pytest.raises(LLMError, match="not support"):
                    await provider.embed(EmbeddingRequest(texts=["test"]))

            import asyncio
            asyncio.run(test())

    def test_convert_messages_system(self):
        """Test system message extraction."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            provider = AnthropicProvider()
            messages = [
                ChatMessage(role=ChatRole.SYSTEM, content="You are helpful"),
                ChatMessage(role=ChatRole.USER, content="Hi"),
            ]
            system, converted = provider._convert_messages(messages)
            assert system == "You are helpful"
            assert len(converted) == 1
            assert converted[0]["role"] == "user"

    def test_convert_tools(self):
        """Test tool conversion to Anthropic format."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            provider = AnthropicProvider()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            converted = provider._convert_tools(tools)
            assert len(converted) == 1
            assert converted[0]["name"] == "get_weather"
            assert converted[0]["description"] == "Get weather"
            assert "input_schema" in converted[0]

    def test_map_stop_reason(self):
        """Test stop reason mapping."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}, clear=False):
            provider = AnthropicProvider()
            assert provider._map_stop_reason("end_turn") == FinishReason.STOP
            assert provider._map_stop_reason("max_tokens") == FinishReason.LENGTH
            assert provider._map_stop_reason("tool_use") == FinishReason.TOOL_CALL
            assert provider._map_stop_reason(None) == FinishReason.STOP


# =============================================================================
# LocalProvider Tests
# =============================================================================


class TestLocalProvider:
    """Tests for LocalProvider."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        provider = LocalProvider()
        assert provider.name == "local"
        assert provider.default_model == "llama3.2"
        assert provider.base_url == "http://localhost:11434/v1"

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        provider = LocalProvider(base_url="http://localhost:8080/v1")
        assert provider.base_url == "http://localhost:8080/v1"

    def test_handle_connection_error(self):
        """Test connection error handling."""
        provider = LocalProvider()
        with pytest.raises(LLMError, match="Cannot connect"):
            provider._handle_api_error(Exception("connection refused"))

    def test_handle_model_not_found(self):
        """Test model not found error handling."""
        provider = LocalProvider()
        with pytest.raises(ModelNotFoundError, match="not found"):
            provider._handle_api_error(Exception("model not found"))


# =============================================================================
# LLMConfig Tests
# =============================================================================


class TestLLMProviderType:
    """Tests for LLMProviderType enum."""

    def test_values(self):
        """Test enum values."""
        assert LLMProviderType.OPENAI.value == "openai"
        assert LLMProviderType.ANTHROPIC.value == "anthropic"
        assert LLMProviderType.LOCAL.value == "local"


class TestLLMConfig:
    """Tests for LLMConfig data class."""

    def test_defaults(self):
        """Test default values."""
        config = LLMConfig()
        assert config.provider == LLMProviderType.OPENAI
        assert config.api_key is None
        assert config.model is None
        assert config.timeout == 30
        assert config.temperature == 0.7

    def test_get_api_key_explicit(self):
        """Test get_api_key with explicit key."""
        config = LLMConfig(api_key="my-key")
        assert config.get_api_key() == "my-key"

    def test_get_api_key_env_unified(self):
        """Test get_api_key from unified env var."""
        config = LLMConfig()
        with patch.dict(os.environ, {"FOUNDRY_MCP_LLM_API_KEY": "unified-key"}):
            assert config.get_api_key() == "unified-key"

    def test_get_api_key_env_provider_specific(self):
        """Test get_api_key from provider-specific env var."""
        config = LLMConfig(provider=LLMProviderType.OPENAI)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=False):
            # Clear unified key
            os.environ.pop("FOUNDRY_MCP_LLM_API_KEY", None)
            assert config.get_api_key() == "openai-key"

    def test_get_model_explicit(self):
        """Test get_model with explicit model."""
        config = LLMConfig(model="gpt-4-turbo")
        assert config.get_model() == "gpt-4-turbo"

    def test_get_model_default(self):
        """Test get_model returns provider default."""
        config = LLMConfig(provider=LLMProviderType.ANTHROPIC)
        assert config.get_model() == DEFAULT_MODELS[LLMProviderType.ANTHROPIC]

    def test_validate_timeout(self):
        """Test validation rejects invalid timeout."""
        config = LLMConfig(timeout=0)
        with pytest.raises(ValueError, match="timeout"):
            config.validate()

    def test_validate_max_tokens(self):
        """Test validation rejects invalid max_tokens."""
        config = LLMConfig(max_tokens=-1)
        with pytest.raises(ValueError, match="max_tokens"):
            config.validate()

    def test_validate_temperature(self):
        """Test validation rejects invalid temperature."""
        config = LLMConfig(temperature=3.0)
        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_validate_api_key_required(self):
        """Test validation requires API key for non-local."""
        config = LLMConfig(provider=LLMProviderType.OPENAI)
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                config.validate()

    def test_validate_local_no_key(self):
        """Test validation allows local without key."""
        config = LLMConfig(provider=LLMProviderType.LOCAL)
        # Should not raise
        config.validate()

    def test_from_dict(self):
        """Test from_dict parsing."""
        data = {
            "provider": "anthropic",
            "api_key": "my-key",
            "model": "claude-3-opus",
            "timeout": 60,
            "temperature": 0.5,
        }
        config = LLMConfig.from_dict(data)
        assert config.provider == LLMProviderType.ANTHROPIC
        assert config.api_key == "my-key"
        assert config.model == "claude-3-opus"
        assert config.timeout == 60
        assert config.temperature == 0.5

    def test_from_dict_invalid_provider(self):
        """Test from_dict rejects invalid provider."""
        with pytest.raises(ValueError, match="Invalid provider"):
            LLMConfig.from_dict({"provider": "invalid"})

    def test_from_env(self):
        """Test from_env parsing."""
        env = {
            "FOUNDRY_MCP_LLM_PROVIDER": "anthropic",
            "FOUNDRY_MCP_LLM_API_KEY": "env-key",
            "FOUNDRY_MCP_LLM_MODEL": "claude-3-haiku",
            "FOUNDRY_MCP_LLM_TIMEOUT": "45",
            "FOUNDRY_MCP_LLM_TEMPERATURE": "0.3",
        }
        with patch.dict(os.environ, env, clear=False):
            config = LLMConfig.from_env()
            assert config.provider == LLMProviderType.ANTHROPIC
            assert config.api_key == "env-key"
            assert config.model == "claude-3-haiku"
            assert config.timeout == 45
            assert config.temperature == 0.3


class TestLLMConfigGlobalFunctions:
    """Tests for global LLM config functions."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_llm_config()

    def test_set_and_get(self):
        """Test set_llm_config and get_llm_config."""
        config = LLMConfig(provider=LLMProviderType.LOCAL)
        set_llm_config(config)
        result = get_llm_config()
        assert result.provider == LLMProviderType.LOCAL

    def test_reset(self):
        """Test reset_llm_config."""
        config = LLMConfig(model="test")
        set_llm_config(config)
        reset_llm_config()
        # Next get should reload
        result = get_llm_config()
        assert result.model != "test"


# =============================================================================
# WorkflowConfig Tests
# =============================================================================


class TestWorkflowMode:
    """Tests for WorkflowMode enum."""

    def test_values(self):
        """Test enum values."""
        assert WorkflowMode.SINGLE.value == "single"
        assert WorkflowMode.AUTONOMOUS.value == "autonomous"
        assert WorkflowMode.BATCH.value == "batch"


class TestWorkflowConfig:
    """Tests for WorkflowConfig data class."""

    def test_defaults(self):
        """Test default values."""
        config = WorkflowConfig()
        assert config.mode == WorkflowMode.SINGLE
        assert config.auto_validate is True
        assert config.journal_enabled is True
        assert config.batch_size == 5
        assert config.context_threshold == 85

    def test_validate_batch_size(self):
        """Test validation rejects invalid batch_size."""
        config = WorkflowConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            config.validate()

    def test_validate_context_threshold_low(self):
        """Test validation rejects low context_threshold."""
        config = WorkflowConfig(context_threshold=40)
        with pytest.raises(ValueError, match="context_threshold"):
            config.validate()

    def test_validate_context_threshold_high(self):
        """Test validation rejects high context_threshold."""
        config = WorkflowConfig(context_threshold=110)
        with pytest.raises(ValueError, match="context_threshold"):
            config.validate()

    def test_from_dict(self):
        """Test from_dict parsing."""
        data = {
            "mode": "autonomous",
            "auto_validate": False,
            "journal_enabled": False,
            "batch_size": 10,
            "context_threshold": 90,
        }
        config = WorkflowConfig.from_dict(data)
        assert config.mode == WorkflowMode.AUTONOMOUS
        assert config.auto_validate is False
        assert config.journal_enabled is False
        assert config.batch_size == 10
        assert config.context_threshold == 90

    def test_from_dict_invalid_mode(self):
        """Test from_dict rejects invalid mode."""
        with pytest.raises(ValueError, match="Invalid workflow mode"):
            WorkflowConfig.from_dict({"mode": "invalid"})

    def test_from_env(self):
        """Test from_env parsing."""
        env = {
            "FOUNDRY_MCP_WORKFLOW_MODE": "batch",
            "FOUNDRY_MCP_WORKFLOW_AUTO_VALIDATE": "false",
            "FOUNDRY_MCP_WORKFLOW_BATCH_SIZE": "20",
            "FOUNDRY_MCP_WORKFLOW_CONTEXT_THRESHOLD": "75",
        }
        with patch.dict(os.environ, env, clear=False):
            config = WorkflowConfig.from_env()
            assert config.mode == WorkflowMode.BATCH
            assert config.auto_validate is False
            assert config.batch_size == 20
            assert config.context_threshold == 75


class TestWorkflowConfigGlobalFunctions:
    """Tests for global workflow config functions."""

    def teardown_method(self):
        """Reset global config after each test."""
        reset_workflow_config()

    def test_set_and_get(self):
        """Test set_workflow_config and get_workflow_config."""
        config = WorkflowConfig(mode=WorkflowMode.AUTONOMOUS)
        set_workflow_config(config)
        result = get_workflow_config()
        assert result.mode == WorkflowMode.AUTONOMOUS

    def test_reset(self):
        """Test reset_workflow_config."""
        config = WorkflowConfig(batch_size=99)
        set_workflow_config(config)
        reset_workflow_config()
        result = get_workflow_config()
        assert result.batch_size != 99


# =============================================================================
# Integration Tests
# =============================================================================


class TestDefaultModelsAndEnvVars:
    """Tests for module-level constants."""

    def test_default_models_complete(self):
        """Test all providers have default models."""
        for provider_type in LLMProviderType:
            assert provider_type in DEFAULT_MODELS

    def test_api_key_env_vars_complete(self):
        """Test all providers have env var mappings."""
        for provider_type in LLMProviderType:
            assert provider_type in API_KEY_ENV_VARS
