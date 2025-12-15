"""
Unit tests for foundry_mcp.core.providers implementation modules.

Tests cover each provider (Gemini, Codex, Cursor Agent, Claude, OpenCode) with:
- Provider instantiation and configuration
- Model validation and selection
- Request validation
- Command building and subprocess invocation (mocked)
- Output parsing and token usage extraction
- Error handling (timeout, unavailable, execution errors)
- Streaming support
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence
from unittest.mock import patch

import pytest

from foundry_mcp.core.providers.base import (
    ProviderExecutionError,
    ProviderHooks,
    ProviderRequest,
    ProviderStatus,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def make_mock_runner(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    raises: Optional[Exception] = None,
):
    """Create a mock runner that returns specified subprocess result."""

    def runner(
        command: Sequence[str],
        *,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[str] = None,
    ):
        if raises:
            raise raises
        return subprocess.CompletedProcess(
            args=list(command),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    return runner


@pytest.fixture
def hooks():
    """Provide default empty hooks."""
    return ProviderHooks()


@pytest.fixture
def stream_chunks():
    """Track streaming chunks for tests."""
    chunks = []

    def on_chunk(chunk, metadata):
        chunks.append(chunk)

    return chunks, on_chunk


# =============================================================================
# GeminiProvider Tests
# =============================================================================


class TestGeminiProvider:
    """Tests for GeminiProvider implementation."""

    def test_instantiation_default_model(self, hooks):
        """GeminiProvider should use default model when none specified."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        provider = GeminiProvider(metadata=GEMINI_METADATA, hooks=hooks)
        assert provider._model == "gemini-2.5-flash"

    def test_instantiation_custom_model(self, hooks):
        """GeminiProvider should accept valid custom model."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        provider = GeminiProvider(metadata=GEMINI_METADATA, hooks=hooks, model="pro")
        assert provider._model == "pro"

    def test_instantiation_invalid_model_raises(self, hooks):
        """GeminiProvider should reject unknown models."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        with pytest.raises(ProviderExecutionError, match="Unsupported Gemini model"):
            GeminiProvider(metadata=GEMINI_METADATA, hooks=hooks, model="invalid-model")

    def test_custom_binary(self, hooks):
        """GeminiProvider should accept custom binary path."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, binary="/custom/gemini"
        )
        assert provider._binary == "/custom/gemini"

    def test_validate_request_rejects_unsupported_params(self, hooks):
        """GeminiProvider should reject unsupported request parameters."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        provider = GeminiProvider(metadata=GEMINI_METADATA, hooks=hooks)
        request = ProviderRequest(prompt="test", temperature=0.5)

        with pytest.raises(ProviderExecutionError, match="does not support"):
            provider._validate_request(request)

    def test_build_command_includes_allowed_tools(self, hooks):
        """GeminiProvider command should include allowed tools."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        provider = GeminiProvider(metadata=GEMINI_METADATA, hooks=hooks)
        command = provider._build_command("pro", "test prompt")

        assert "gemini" in command[0]
        assert "--output-format" in command
        assert "json" in command
        assert "--allowed-tools" in command
        assert "-p" in command

    def test_successful_execution(self, hooks):
        """GeminiProvider should return valid result on success."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        mock_output = json.dumps(
            {
                "response": "Hello from Gemini!",
                "model": "gemini-2.5-flash",
                "stats": {
                    "models": {
                        "gemini-2.5-flash": {
                            "tokens": {"prompt": 10, "candidates": 5, "total": 15}
                        }
                    }
                },
            }
        )
        runner = make_mock_runner(stdout=mock_output)
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        result = provider.generate(ProviderRequest(prompt="Hello"))

        assert result.content == "Hello from Gemini!"
        assert result.status == ProviderStatus.SUCCESS
        assert result.provider_id == "gemini"
        assert result.tokens.input_tokens == 10
        assert result.tokens.output_tokens == 5

    def test_nonzero_exit_raises_execution_error(self, hooks):
        """GeminiProvider should raise on non-zero exit code."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        runner = make_mock_runner(returncode=1, stderr="Command failed")
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderExecutionError, match="exited with code 1"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_empty_output_raises_execution_error(self, hooks):
        """GeminiProvider should raise on empty output."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        runner = make_mock_runner(stdout="")
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderExecutionError, match="empty output"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_invalid_json_raises_execution_error(self, hooks):
        """GeminiProvider should raise on invalid JSON output."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        runner = make_mock_runner(stdout="not valid json")
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderExecutionError, match="invalid JSON"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_timeout_raises_timeout_error(self, hooks):
        """GeminiProvider should raise timeout error when subprocess times out."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        runner = make_mock_runner(raises=subprocess.TimeoutExpired("gemini", 30))
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderTimeoutError):
            provider.generate(ProviderRequest(prompt="test"))

    def test_file_not_found_raises_unavailable_error(self, hooks):
        """GeminiProvider should raise unavailable error when binary not found."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        runner = make_mock_runner(raises=FileNotFoundError("gemini not found"))
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderUnavailableError, match="not available"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_streaming_emits_chunk(self, hooks, stream_chunks):
        """GeminiProvider should emit stream chunk when streaming enabled."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        chunks, on_chunk = stream_chunks
        hooks = ProviderHooks(on_stream_chunk=on_chunk)

        mock_output = json.dumps({"response": "Streamed content"})
        runner = make_mock_runner(stdout=mock_output)
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        result = provider.generate(ProviderRequest(prompt="test", stream=True))

        assert result.content == "Streamed content"
        assert len(chunks) == 1
        assert chunks[0].content == "Streamed content"

    def test_model_override_in_request(self, hooks):
        """GeminiProvider should use model from request metadata."""
        from foundry_mcp.core.providers.gemini import GEMINI_METADATA, GeminiProvider

        mock_output = json.dumps({"response": "OK", "model": "pro"})
        runner = make_mock_runner(stdout=mock_output)
        provider = GeminiProvider(
            metadata=GEMINI_METADATA, hooks=hooks, runner=runner
        )

        request = ProviderRequest(prompt="test", metadata={"model": "pro"})
        result = provider.generate(request)

        assert "pro" in result.model_used


# =============================================================================
# CodexProvider Tests
# =============================================================================


class TestCodexProvider:
    """Tests for CodexProvider implementation."""

    def test_instantiation_default_model(self, hooks):
        """CodexProvider should use default model when none specified."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks)
        assert provider._model == "gpt-5.1-codex"

    def test_instantiation_invalid_model_raises(self, hooks):
        """CodexProvider should reject unknown models."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        with pytest.raises(ProviderExecutionError, match="Unsupported Codex model"):
            CodexProvider(metadata=CODEX_METADATA, hooks=hooks, model="invalid")

    def test_build_command_includes_sandbox(self, hooks):
        """CodexProvider command should include sandbox flags."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks)
        command = provider._build_command("gpt-5.1-codex", "test", [])

        assert "codex" in command[0]
        assert "exec" in command
        assert "--sandbox" in command
        assert "read-only" in command
        assert "--json" in command

    def test_successful_execution_jsonl(self, hooks):
        """CodexProvider should parse JSONL output correctly."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        # Simulate Codex JSONL output
        events = [
            {"type": "thread.started", "thread_id": "t123"},
            {"type": "item.delta", "content": {"text": "Hello"}},
            {"type": "item.completed", "content": {"text": "Hello from Codex!"}},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            },
        ]
        mock_output = "\n".join(json.dumps(e) for e in events)

        runner = make_mock_runner(stdout=mock_output)
        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks, runner=runner)

        result = provider.generate(ProviderRequest(prompt="Hello"))

        assert result.content == "Hello from Codex!"
        assert result.status == ProviderStatus.SUCCESS
        assert result.tokens.input_tokens == 10
        assert result.tokens.output_tokens == 20

    def test_nonzero_exit_raises_execution_error(self, hooks):
        """CodexProvider should raise on non-zero exit code."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        runner = make_mock_runner(returncode=1, stderr="Sandbox error")
        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks, runner=runner)

        with pytest.raises(ProviderExecutionError, match="exited with code 1"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_empty_output_raises_execution_error(self, hooks):
        """CodexProvider should raise on empty output."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        runner = make_mock_runner(stdout="")
        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks, runner=runner)

        with pytest.raises(ProviderExecutionError, match="empty output"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_invalid_jsonl_raises_execution_error(self, hooks):
        """CodexProvider should raise on invalid JSONL."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        runner = make_mock_runner(stdout="not valid json\nmore bad data")
        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks, runner=runner)

        with pytest.raises(ProviderExecutionError, match="invalid JSON"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_attachment_handling(self, hooks):
        """CodexProvider should add attachments as --image flags."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks)
        command = provider._build_command(
            "gpt-5.1-codex", "test", ["/path/to/image.png"]
        )

        assert "--image" in command
        assert "/path/to/image.png" in command

    def test_streaming_emits_deltas(self, hooks, stream_chunks):
        """CodexProvider should emit stream chunks for item.delta events."""
        from foundry_mcp.core.providers.codex import CODEX_METADATA, CodexProvider

        chunks, on_chunk = stream_chunks
        hooks = ProviderHooks(on_stream_chunk=on_chunk)

        events = [
            {"type": "item.delta", "content": {"text": "chunk1"}},
            {"type": "item.delta", "content": {"text": "chunk2"}},
            {"type": "item.completed", "content": {"text": "chunk1chunk2"}},
        ]
        mock_output = "\n".join(json.dumps(e) for e in events)

        runner = make_mock_runner(stdout=mock_output)
        provider = CodexProvider(metadata=CODEX_METADATA, hooks=hooks, runner=runner)

        provider.generate(ProviderRequest(prompt="test", stream=True))

        assert len(chunks) == 2
        assert chunks[0].content == "chunk1"
        assert chunks[1].content == "chunk2"


# =============================================================================
# CursorAgentProvider Tests
# =============================================================================


class TestCursorAgentProvider:
    """Tests for CursorAgentProvider implementation."""

    def test_instantiation_default_model(self, hooks):
        """CursorAgentProvider should use default model when none specified."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        provider = CursorAgentProvider(metadata=CURSOR_METADATA, hooks=hooks)
        assert provider._model == "composer-1"

    def test_instantiation_invalid_model_raises(self, hooks):
        """CursorAgentProvider should reject unknown models."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        with pytest.raises(ProviderExecutionError, match="Unsupported Cursor Agent"):
            CursorAgentProvider(metadata=CURSOR_METADATA, hooks=hooks, model="invalid")

    def test_build_command_includes_print_and_json(self, hooks):
        """CursorAgentProvider command should include --print and output format."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        provider = CursorAgentProvider(metadata=CURSOR_METADATA, hooks=hooks)
        request = ProviderRequest(prompt="test")
        command = provider._build_command(request, "composer-1")

        assert "--print" in command
        assert "--output-format" in command
        assert "json" in command

    def test_successful_execution_json_mode(self, hooks, tmp_path):
        """CursorAgentProvider should parse JSON output correctly."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        mock_output = json.dumps(
            {
                "result": "Hello from Cursor!",
                "model": "composer-1",
                "usage": {"input_tokens": 15, "output_tokens": 8, "total_tokens": 23},
            }
        )
        runner = make_mock_runner(stdout=mock_output)

        # Create mock cursor config dir
        with patch.object(Path, "home", return_value=tmp_path):
            (tmp_path / ".cursor").mkdir(exist_ok=True)
            provider = CursorAgentProvider(
                metadata=CURSOR_METADATA, hooks=hooks, runner=runner
            )

            result = provider.generate(ProviderRequest(prompt="Hello"))

            assert result.content == "Hello from Cursor!"
            assert result.status == ProviderStatus.SUCCESS
            assert result.tokens.input_tokens == 15

    def test_attachments_not_supported(self, hooks, tmp_path):
        """CursorAgentProvider should reject attachments."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        with patch.object(Path, "home", return_value=tmp_path):
            (tmp_path / ".cursor").mkdir(exist_ok=True)
            provider = CursorAgentProvider(metadata=CURSOR_METADATA, hooks=hooks)

            request = ProviderRequest(prompt="test", attachments=["image.png"])

            with pytest.raises(ProviderExecutionError, match="does not support"):
                provider.generate(request)

    def test_fallback_to_text_mode(self, hooks, tmp_path):
        """CursorAgentProvider should fall back when --output-format not supported."""
        from foundry_mcp.core.providers.cursor_agent import (
            CURSOR_METADATA,
            CursorAgentProvider,
        )

        # First call fails with unknown option, retry succeeds
        call_count = [0]

        def runner(command, *, timeout=None, env=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return subprocess.CompletedProcess(
                    args=list(command),
                    returncode=1,
                    stdout="",
                    stderr="unknown option: --output-format",
                )
            return subprocess.CompletedProcess(
                args=list(command),
                returncode=0,
                stdout="Plain text response",
                stderr="",
            )

        with patch.object(Path, "home", return_value=tmp_path):
            (tmp_path / ".cursor").mkdir(exist_ok=True)
            provider = CursorAgentProvider(
                metadata=CURSOR_METADATA, hooks=hooks, runner=runner
            )

            result = provider.generate(ProviderRequest(prompt="test"))

            assert result.content == "Plain text response"
            assert call_count[0] == 2  # Retry happened


# =============================================================================
# ClaudeProvider Tests
# =============================================================================


class TestClaudeProvider:
    """Tests for ClaudeProvider implementation."""

    def test_instantiation_default_model(self, hooks):
        """ClaudeProvider should use default model when none specified."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        provider = ClaudeProvider(metadata=CLAUDE_METADATA, hooks=hooks)
        assert provider._model == "sonnet"

    def test_instantiation_invalid_model_raises(self, hooks):
        """ClaudeProvider should reject unknown models."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        with pytest.raises(ProviderExecutionError, match="Unsupported Claude model"):
            ClaudeProvider(metadata=CLAUDE_METADATA, hooks=hooks, model="invalid")

    def test_build_command_includes_allowed_and_disallowed(self, hooks):
        """ClaudeProvider command should include tool restrictions."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        provider = ClaudeProvider(metadata=CLAUDE_METADATA, hooks=hooks)
        command = provider._build_command("sonnet", "test prompt")

        assert "--print" in command
        assert "--output-format" in command
        assert "json" in command
        assert "--allowed-tools" in command
        assert "--disallowed-tools" in command
        assert "--system-prompt" in command

    def test_successful_execution(self, hooks):
        """ClaudeProvider should return valid result on success."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        mock_output = json.dumps(
            {
                "result": "Hello from Claude!",
                "modelUsage": {"claude-sonnet-4-5-20250929": {}},
                "usage": {"input_tokens": 12, "output_tokens": 6},
            }
        )
        runner = make_mock_runner(stdout=mock_output)
        provider = ClaudeProvider(
            metadata=CLAUDE_METADATA, hooks=hooks, runner=runner
        )

        result = provider.generate(ProviderRequest(prompt="Hello"))

        assert result.content == "Hello from Claude!"
        assert result.status == ProviderStatus.SUCCESS
        assert result.tokens.input_tokens == 12
        assert result.tokens.output_tokens == 6

    def test_validate_request_rejects_unsupported_params(self, hooks):
        """ClaudeProvider should reject unsupported request parameters."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        provider = ClaudeProvider(metadata=CLAUDE_METADATA, hooks=hooks)
        request = ProviderRequest(prompt="test", temperature=0.5)

        with pytest.raises(ProviderExecutionError, match="does not support"):
            provider._validate_request(request)

    def test_nonzero_exit_raises_execution_error(self, hooks):
        """ClaudeProvider should raise on non-zero exit code."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        runner = make_mock_runner(returncode=1, stderr="Auth failed")
        provider = ClaudeProvider(
            metadata=CLAUDE_METADATA, hooks=hooks, runner=runner
        )

        with pytest.raises(ProviderExecutionError, match="exited with code 1"):
            provider.generate(ProviderRequest(prompt="test"))

    def test_streaming_emits_chunk(self, hooks, stream_chunks):
        """ClaudeProvider should emit stream chunk when streaming enabled."""
        from foundry_mcp.core.providers.claude import CLAUDE_METADATA, ClaudeProvider

        chunks, on_chunk = stream_chunks
        hooks = ProviderHooks(on_stream_chunk=on_chunk)

        mock_output = json.dumps({"result": "Streamed"})
        runner = make_mock_runner(stdout=mock_output)
        provider = ClaudeProvider(
            metadata=CLAUDE_METADATA, hooks=hooks, runner=runner
        )

        provider.generate(ProviderRequest(prompt="test", stream=True))

        assert len(chunks) == 1
        assert chunks[0].content == "Streamed"


# =============================================================================
# OpenCodeProvider Tests
# =============================================================================


class TestOpenCodeProvider:
    """Tests for OpenCodeProvider implementation."""

    def test_instantiation_default_model(self, hooks):
        """OpenCodeProvider should use default model when none specified."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        provider = OpenCodeProvider(metadata=OPENCODE_METADATA, hooks=hooks)
        assert provider._model == "default"

    def test_instantiation_empty_model_raises(self, hooks):
        """OpenCodeProvider should reject empty model identifiers."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        # OpenCode validates empty models - either "" or whitespace-only
        with pytest.raises(ProviderExecutionError, match="cannot be empty"):
            OpenCodeProvider(metadata=OPENCODE_METADATA, hooks=hooks, model="   ")

    def test_validate_request_rejects_attachments(self, hooks):
        """OpenCodeProvider should reject attachments."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        provider = OpenCodeProvider(metadata=OPENCODE_METADATA, hooks=hooks)
        request = ProviderRequest(prompt="test", attachments=["image.png"])

        with pytest.raises(ProviderExecutionError, match="does not support"):
            provider._validate_request(request)

    @patch("foundry_mcp.core.providers.opencode.OpenCodeProvider._is_port_open")
    def test_successful_execution(self, mock_port, hooks, tmp_path):
        """OpenCodeProvider should parse line-delimited JSON output."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        mock_port.return_value = True  # Server already running

        # Note: done response text takes precedence when content_parts is empty
        output_lines = [
            {
                "type": "done",
                "response": {
                    "text": "Hello from OpenCode!",
                    "model": "gpt-4",
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 4,
                        "total_tokens": 12,
                    },
                },
            },
        ]
        mock_output = "\n".join(json.dumps(line) for line in output_lines)

        runner = make_mock_runner(stdout=mock_output)
        provider = OpenCodeProvider(
            metadata=OPENCODE_METADATA,
            hooks=hooks,
            runner=runner,
            wrapper_path=tmp_path / "wrapper.js",
        )

        result = provider.generate(ProviderRequest(prompt="Hello"))

        assert result.content == "Hello from OpenCode!"
        assert result.status == ProviderStatus.SUCCESS
        assert result.tokens.input_tokens == 8
        assert result.tokens.output_tokens == 4

    @patch("foundry_mcp.core.providers.opencode.OpenCodeProvider._is_port_open")
    def test_wrapper_error_raises_execution_error(self, mock_port, hooks, tmp_path):
        """OpenCodeProvider should raise when wrapper returns error event."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        mock_port.return_value = True

        output_lines = [{"type": "error", "message": "API key invalid"}]
        mock_output = "\n".join(json.dumps(line) for line in output_lines)

        runner = make_mock_runner(stdout=mock_output)
        provider = OpenCodeProvider(
            metadata=OPENCODE_METADATA,
            hooks=hooks,
            runner=runner,
            wrapper_path=tmp_path / "wrapper.js",
        )

        with pytest.raises(ProviderExecutionError, match="API key invalid"):
            provider.generate(ProviderRequest(prompt="test"))

    @patch("foundry_mcp.core.providers.opencode.OpenCodeProvider._is_port_open")
    def test_nonzero_exit_raises_execution_error(self, mock_port, hooks, tmp_path):
        """OpenCodeProvider should raise on non-zero exit code."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        mock_port.return_value = True

        runner = make_mock_runner(returncode=1, stderr="Node error")
        provider = OpenCodeProvider(
            metadata=OPENCODE_METADATA,
            hooks=hooks,
            runner=runner,
            wrapper_path=tmp_path / "wrapper.js",
        )

        with pytest.raises(ProviderExecutionError, match="exited with code 1"):
            provider.generate(ProviderRequest(prompt="test"))

    @patch("foundry_mcp.core.providers.opencode.OpenCodeProvider._is_port_open")
    def test_streaming_emits_chunks(self, mock_port, hooks, stream_chunks, tmp_path):
        """OpenCodeProvider should emit stream chunks."""
        from foundry_mcp.core.providers.opencode import (
            OPENCODE_METADATA,
            OpenCodeProvider,
        )

        mock_port.return_value = True
        chunks, on_chunk = stream_chunks
        hooks = ProviderHooks(on_stream_chunk=on_chunk)

        output_lines = [
            {"type": "chunk", "content": "part1"},
            {"type": "chunk", "content": "part2"},
            {"type": "done", "response": {"text": "part1part2"}},
        ]
        mock_output = "\n".join(json.dumps(line) for line in output_lines)

        runner = make_mock_runner(stdout=mock_output)
        provider = OpenCodeProvider(
            metadata=OPENCODE_METADATA,
            hooks=hooks,
            runner=runner,
            wrapper_path=tmp_path / "wrapper.js",
        )

        provider.generate(ProviderRequest(prompt="test", stream=True))

        assert len(chunks) >= 2


# =============================================================================
# is_*_available Function Tests
# =============================================================================


class TestAvailabilityFunctions:
    """Tests for provider availability check functions."""

    @patch("foundry_mcp.core.providers.gemini.detect_provider_availability")
    def test_is_gemini_available(self, mock_detect):
        """is_gemini_available should delegate to detector."""
        from foundry_mcp.core.providers.gemini import is_gemini_available

        mock_detect.return_value = True
        assert is_gemini_available() is True
        mock_detect.assert_called_with("gemini")

    @patch("foundry_mcp.core.providers.codex.detect_provider_availability")
    def test_is_codex_available(self, mock_detect):
        """is_codex_available should delegate to detector."""
        from foundry_mcp.core.providers.codex import is_codex_available

        mock_detect.return_value = False
        assert is_codex_available() is False
        mock_detect.assert_called_with("codex")

    @patch("foundry_mcp.core.providers.cursor_agent.detect_provider_availability")
    def test_is_cursor_agent_available(self, mock_detect):
        """is_cursor_agent_available should delegate to detector."""
        from foundry_mcp.core.providers.cursor_agent import is_cursor_agent_available

        mock_detect.return_value = True
        assert is_cursor_agent_available() is True
        mock_detect.assert_called_with("cursor-agent")

    @patch("foundry_mcp.core.providers.claude.detect_provider_availability")
    def test_is_claude_available(self, mock_detect):
        """is_claude_available should delegate to detector."""
        from foundry_mcp.core.providers.claude import is_claude_available

        mock_detect.return_value = True
        assert is_claude_available() is True
        mock_detect.assert_called_with("claude")

    @patch("foundry_mcp.core.providers.opencode.detect_provider_availability")
    def test_is_opencode_available(self, mock_detect):
        """is_opencode_available should delegate to detector."""
        from foundry_mcp.core.providers.opencode import is_opencode_available

        mock_detect.return_value = False
        assert is_opencode_available() is False
        mock_detect.assert_called_with("opencode")


# =============================================================================
# create_provider Factory Tests
# =============================================================================


class TestCreateProviderFactories:
    """Tests for create_provider factory functions."""

    def test_gemini_create_provider(self, hooks):
        """Gemini create_provider should return configured provider."""
        from foundry_mcp.core.providers.gemini import create_provider

        provider = create_provider(hooks=hooks, model="pro")
        assert provider._model == "pro"

    def test_codex_create_provider(self, hooks):
        """Codex create_provider should return configured provider."""
        from foundry_mcp.core.providers.codex import create_provider

        provider = create_provider(hooks=hooks)
        assert provider._model == "gpt-5.1-codex"

    def test_cursor_agent_create_provider(self, hooks):
        """Cursor Agent create_provider should return configured provider."""
        from foundry_mcp.core.providers.cursor_agent import create_provider

        provider = create_provider(hooks=hooks)
        assert provider._model == "composer-1"

    def test_claude_create_provider(self, hooks):
        """Claude create_provider should return configured provider."""
        from foundry_mcp.core.providers.claude import create_provider

        provider = create_provider(hooks=hooks, model="haiku")
        assert provider._model == "haiku"

    def test_opencode_create_provider(self, hooks):
        """OpenCode create_provider should return configured provider."""
        from foundry_mcp.core.providers.opencode import create_provider

        provider = create_provider(hooks=hooks)
        assert provider._model == "default"

    def test_factory_accepts_dependencies_and_overrides(self, hooks):
        """Factory should inject dependencies and apply overrides."""
        from foundry_mcp.core.providers.gemini import create_provider

        custom_runner = make_mock_runner(stdout='{"response": "test"}')
        custom_env = {"GEMINI_API_KEY": "test-key"}

        provider = create_provider(
            hooks=hooks,
            dependencies={"runner": custom_runner, "env": custom_env},
            overrides={"binary": "/custom/gemini", "model": "pro"},
        )

        assert provider._binary == "/custom/gemini"
        assert provider._model == "pro"
        assert provider._env == custom_env


# =============================================================================
# Provider Registration Tests
# =============================================================================


class TestProviderRegistration:
    """Tests for automatic provider registration."""

    def test_gemini_registered(self):
        """Gemini provider should be registered."""
        from foundry_mcp.core.providers.registry import get_registration

        reg = get_registration("gemini")
        assert reg is not None
        assert reg.metadata.provider_id == "gemini"
        assert "cli" in reg.tags

    def test_codex_registered(self):
        """Codex provider should be registered."""
        from foundry_mcp.core.providers.registry import get_registration

        reg = get_registration("codex")
        assert reg is not None
        assert reg.metadata.provider_id == "codex"
        assert "sandboxed" in reg.tags

    def test_cursor_agent_registered(self):
        """Cursor Agent provider should be registered."""
        from foundry_mcp.core.providers.registry import get_registration

        reg = get_registration("cursor-agent")
        assert reg is not None
        assert reg.metadata.provider_id == "cursor-agent"

    def test_claude_registered(self):
        """Claude provider should be registered."""
        from foundry_mcp.core.providers.registry import get_registration

        reg = get_registration("claude")
        assert reg is not None
        assert reg.metadata.provider_id == "claude"
        assert "thinking" in reg.tags

    def test_opencode_registered(self):
        """OpenCode provider should be registered."""
        from foundry_mcp.core.providers.registry import get_registration

        reg = get_registration("opencode")
        assert reg is not None
        assert reg.metadata.provider_id == "opencode"
        assert "sdk" in reg.tags
