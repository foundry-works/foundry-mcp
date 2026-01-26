"""
OpenCode AI provider implementation.

Bridges the OpenCode AI Node.js SDK wrapper to the ProviderContext contract by
handling availability checks, server management, wrapper script execution,
response parsing, and token usage normalization.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .base import (
    ProviderCapability,
    ProviderContext,
    ProviderExecutionError,
    ProviderHooks,
    ProviderMetadata,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StreamChunk,
    TokenUsage,
)
from .detectors import detect_provider_availability
from .registry import register_provider

logger = logging.getLogger(__name__)

DEFAULT_BINARY = "node"
DEFAULT_WRAPPER_SCRIPT = Path(__file__).parent / "opencode_wrapper.js"
DEFAULT_TIMEOUT_SECONDS = 360
DEFAULT_SERVER_URL = "http://localhost:4096"
SERVER_STARTUP_TIMEOUT = 30
AVAILABILITY_OVERRIDE_ENV = "OPENCODE_AVAILABLE_OVERRIDE"
CUSTOM_BINARY_ENV = "OPENCODE_BINARY"
CUSTOM_WRAPPER_ENV = "OPENCODE_WRAPPER_SCRIPT"

# Read-only tools configuration for OpenCode server
# Uses dual-layer protection: tool disabling + permission denial
READONLY_TOOLS_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "tools": {
        # Disable write operations
        "write": False,
        "edit": False,
        "patch": False,
        "todowrite": False,
        # Disable shell execution
        "bash": False,
        # Enable read operations
        "read": True,
        "grep": True,
        "glob": True,
        "list": True,
        "todoread": True,
        "task": True,
        # Disable web operations (data exfiltration risk)
        "webfetch": False,
    },
    "permission": {
        # Double-guard with permission denials
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "external_directory": "deny",
    },
}

# System prompt warning about tool limitations
SHELL_COMMAND_WARNING = """
IMPORTANT SECURITY NOTE: This session is running in read-only mode with the following restrictions:
1. File write operations (write, edit, patch) are disabled
2. Shell command execution (bash) is disabled
3. Web operations (webfetch) are disabled to prevent data exfiltration
4. Only read operations are available (read, grep, glob, list)
5. Attempts to modify files, execute commands, or access the web will be blocked by the server
"""


class RunnerProtocol(Protocol):
    """Callable signature used for executing Node.js wrapper commands."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError


def _default_runner(
    command: Sequence[str],
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    input_data: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke the OpenCode wrapper via subprocess."""
    return subprocess.run(  # noqa: S603,S607 - intentional wrapper invocation
        list(command),
        capture_output=True,
        text=True,
        input=input_data,
        timeout=timeout,
        env=env,
        check=False,
    )


OPENCODE_METADATA = ProviderMetadata(
    provider_id="opencode",
    display_name="OpenCode AI SDK",
    models=[],  # Model validation delegated to CLI
    default_model="openai/gpt-5.1-codex-mini",
    capabilities={ProviderCapability.TEXT, ProviderCapability.STREAMING},
    security_flags={"writes_allowed": False, "read_only": True},
    extra={
        "wrapper": "opencode_wrapper.js",
        "server_url": DEFAULT_SERVER_URL,
        "configurable": True,
        "readonly_config": READONLY_TOOLS_CONFIG,
    },
)


class OpenCodeProvider(ProviderContext):
    """ProviderContext implementation backed by the OpenCode AI wrapper."""

    def __init__(
        self,
        metadata: ProviderMetadata,
        hooks: ProviderHooks,
        *,
        model: Optional[str] = None,
        binary: Optional[str] = None,
        wrapper_path: Optional[Path] = None,
        runner: Optional[RunnerProtocol] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ):
        super().__init__(metadata, hooks)
        self._runner = runner or _default_runner
        self._binary = binary or os.environ.get(CUSTOM_BINARY_ENV, DEFAULT_BINARY)
        self._wrapper_path = wrapper_path or Path(
            os.environ.get(CUSTOM_WRAPPER_ENV, str(DEFAULT_WRAPPER_SCRIPT))
        )

        # Prepare environment for subprocess with secure API key handling
        self._env = self._prepare_subprocess_env(env)

        self._timeout = timeout or DEFAULT_TIMEOUT_SECONDS

        # Validate model - reject empty or whitespace-only strings
        effective_model = model or metadata.default_model or "openai/gpt-5.1-codex-mini"
        if not effective_model or not effective_model.strip():
            raise ProviderExecutionError(
                "Model identifier cannot be empty",
                provider="opencode",
            )
        self._model = effective_model

        self._server_process: Optional[subprocess.Popen[bytes]] = None
        self._config_file_path: Optional[Path] = None

    def __del__(self) -> None:
        """Clean up server process and config file on provider destruction."""
        # Clean up server process
        if hasattr(self, "_server_process") and self._server_process is not None:
            try:
                self._server_process.terminate()
                # Give it a moment to terminate gracefully
                try:
                    self._server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't terminate
                    self._server_process.kill()
            except (OSError, ProcessLookupError):
                # Process already terminated, ignore
                pass
            finally:
                self._server_process = None

        # Clean up config file
        self._cleanup_config_file()

    def _prepare_subprocess_env(
        self, custom_env: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Prepare environment variables for subprocess execution.

        Merges current process environment with custom overrides and ensures
        required OpenCode variables are present.
        """
        # Start with a copy of the current environment
        subprocess_env = os.environ.copy()

        # Merge custom environment if provided
        if custom_env:
            subprocess_env.update(custom_env)

        # Ensure OPENCODE_SERVER_URL is set (use default if not provided)
        if "OPENCODE_SERVER_URL" not in subprocess_env:
            subprocess_env["OPENCODE_SERVER_URL"] = DEFAULT_SERVER_URL

        # Note: OPENCODE_API_KEY should be provided via environment or custom_env
        # We don't set a default value for security reasons

        # Add global npm modules to NODE_PATH so wrapper can find @opencode-ai/sdk
        # This allows the SDK to be installed globally rather than bundled
        self._ensure_node_path(subprocess_env)

        return subprocess_env

    def _ensure_node_path(self, env: Dict[str, str]) -> None:
        """
        Ensure NODE_PATH includes global npm modules and local node_modules.

        This allows the wrapper script to import @opencode-ai/sdk whether it's
        installed globally (npm install -g @opencode-ai/sdk) or locally in the
        providers directory.
        """
        node_paths: List[str] = []

        # Add existing NODE_PATH entries
        if env.get("NODE_PATH"):
            node_paths.extend(env["NODE_PATH"].split(os.pathsep))

        # Add local node_modules (alongside wrapper script)
        local_node_modules = self._wrapper_path.parent / "node_modules"
        if local_node_modules.exists():
            node_paths.append(str(local_node_modules))

        # Detect and add global npm root
        try:
            result = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                global_root = result.stdout.strip()
                if global_root not in node_paths:
                    node_paths.append(global_root)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            # npm not available or timed out - skip global path
            pass

        if node_paths:
            env["NODE_PATH"] = os.pathsep.join(node_paths)

    def _create_readonly_config(self) -> Path:
        """
        Create temporary opencode.json with read-only tool restrictions.

        Returns:
            Path to the temporary config file

        Note:
            - Tool blocking may not work for MCP tools (OpenCode issue #3756)
            - Config is server-wide, affecting all sessions on this server instance
        """
        # Create temp directory for config
        temp_dir = Path(tempfile.mkdtemp(prefix="opencode_readonly_"))

        # Create config file
        config_path = temp_dir / "opencode.json"
        with open(config_path, "w") as f:
            json.dump(READONLY_TOOLS_CONFIG, f, indent=2)

        return config_path

    def _cleanup_config_file(self) -> None:
        """Remove temporary config file and directory."""
        if hasattr(self, "_config_file_path") and self._config_file_path is not None:
            try:
                # Remove config file
                if self._config_file_path.exists():
                    self._config_file_path.unlink()

                # Remove temp directory
                temp_dir = self._config_file_path.parent
                if temp_dir.exists():
                    temp_dir.rmdir()
            except (OSError, FileNotFoundError):
                # File already removed or doesn't exist, ignore
                pass
            finally:
                self._config_file_path = None

    def _is_port_open(self, port: int, host: str = "localhost") -> bool:
        """Check if a TCP port is open and accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result == 0
        except (socket.error, OSError):
            return False

    def _is_opencode_server_healthy(self, server_url: str) -> bool:
        """Verify the opencode server is actually responding (not just port open)."""
        import urllib.request
        import urllib.error

        try:
            # Try to hit the opencode server - it should respond to HTTP
            req = urllib.request.Request(
                f"{server_url}/session",
                method="POST",
                headers={"Content-Type": "application/json"},
                data=b"{}",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                # Any response (even error) means server is alive
                return resp.status < 500
        except urllib.error.HTTPError as e:
            # 4xx errors mean server is alive but rejected request - that's OK
            return e.code < 500
        except Exception:
            # Connection refused, timeout, etc. - server not healthy
            return False

    def _ensure_server_running(self) -> None:
        """Ensure OpenCode server is running, start if necessary."""
        # Extract port from server URL (default: 4096)
        server_url = (
            self._env.get("OPENCODE_SERVER_URL", DEFAULT_SERVER_URL)
            if self._env
            else DEFAULT_SERVER_URL
        )
        try:
            # Parse port from URL (e.g., "http://localhost:4096" -> 4096)
            port = int(server_url.split(":")[-1].rstrip("/"))
        except (ValueError, IndexError):
            port = 4096

        # Check if server is already running and healthy
        if self._is_port_open(port):
            if self._is_opencode_server_healthy(server_url):
                logger.debug(f"OpenCode server already running on port {port}")
                return
            else:
                # Port is open but server not responding properly
                logger.warning(
                    f"Port {port} is open but OpenCode server not responding. "
                    "Another process may be using this port."
                )
                raise ProviderExecutionError(
                    f"Port {port} is in use but OpenCode server is not responding. "
                    "Possible causes: another process using this port, "
                    "provider quota exceeded, or authentication issue. "
                    f"Try: lsof -i :{port} | opencode config",
                    provider=self.metadata.provider_id,
                )

        # Server not running - need to start it
        logger.info(f"OpenCode server not running on port {port}, attempting to start...")
        # Look for opencode binary in node_modules/.bin first
        opencode_binary = None
        node_modules_bin = Path("node_modules/.bin/opencode")

        if node_modules_bin.exists():
            opencode_binary = str(node_modules_bin)
        else:
            # Fall back to global opencode if available
            try:
                result = subprocess.run(
                    ["which", "opencode"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    opencode_binary = result.stdout.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if not opencode_binary:
            raise ProviderUnavailableError(
                "OpenCode server not running and 'opencode' binary not found in node_modules/.bin or PATH",
                provider=self.metadata.provider_id,
            )

        # Create read-only configuration file
        self._config_file_path = self._create_readonly_config()

        # Start server in background
        # Prepare environment with API keys and configuration
        server_env = self._prepare_subprocess_env(self._env)
        # Set OPENCODE_CONFIG to point to our readonly config
        server_env["OPENCODE_CONFIG"] = str(self._config_file_path)

        try:
            self._server_process = subprocess.Popen(
                [opencode_binary, "serve", "--hostname=127.0.0.1", f"--port={port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=server_env,  # Pass environment variables to server
                start_new_session=True,  # Detach from parent
            )
        except (OSError, subprocess.SubprocessError) as e:
            raise ProviderExecutionError(
                f"Failed to start OpenCode server: {e}",
                provider=self.metadata.provider_id,
            ) from e

        # Wait for server to become available and healthy
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            if self._is_port_open(port) and self._is_opencode_server_healthy(server_url):
                logger.info(f"OpenCode server started successfully on port {port}")
                return
            time.sleep(0.5)

        # Timeout - server didn't start
        if self._server_process:
            self._server_process.terminate()
            self._server_process = None

        raise ProviderTimeoutError(
            f"OpenCode server failed to start within {SERVER_STARTUP_TIMEOUT} seconds",
            provider=self.metadata.provider_id,
            elapsed=float(SERVER_STARTUP_TIMEOUT),
            timeout=float(SERVER_STARTUP_TIMEOUT),
        )

    def _validate_request(self, request: ProviderRequest) -> None:
        """Validate request parameters supported by OpenCode."""
        unsupported: List[str] = []
        if request.attachments:
            unsupported.append("attachments")
        if unsupported:
            raise ProviderExecutionError(
                f"OpenCode does not support: {', '.join(unsupported)}",
                provider=self.metadata.provider_id,
            )

    def _build_prompt(self, request: ProviderRequest) -> str:
        """Build the prompt with system prompt and security warning."""
        system_parts = []
        if request.system_prompt:
            system_parts.append(request.system_prompt.strip())
        system_parts.append(SHELL_COMMAND_WARNING.strip())

        if system_parts:
            return f"{chr(10).join(system_parts)}\n\n{request.prompt}"
        return request.prompt

    def _resolve_model(self, request: ProviderRequest) -> str:
        """Resolve model from request or use default."""
        # 1. Check request.model first (from ProviderRequest constructor)
        if request.model:
            return str(request.model)
        # 2. Fallback to metadata override (alternate path)
        model_override = request.metadata.get("model") if request.metadata else None
        if model_override:
            return str(model_override)
        # 3. Fallback to instance default
        return self._model

    def _emit_stream_if_requested(self, content: str, *, stream: bool) -> None:
        """Emit streaming chunk if streaming is enabled."""
        if not stream or not content:
            return
        self._emit_stream_chunk(StreamChunk(content=content, index=0))

    def _extract_error_from_jsonl(self, stdout: str) -> Optional[str]:
        """
        Extract error message from OpenCode wrapper JSONL output.

        The wrapper outputs errors as {"type":"error","code":"...","message":"..."}.
        """
        if not stdout:
            return None

        for line in stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "error":
                msg = event.get("message", "")
                if msg:
                    return msg

        return None

    def _execute(self, request: ProviderRequest) -> ProviderResult:
        """Execute generation request via OpenCode wrapper."""
        self._validate_request(request)

        # Ensure server is running before making request
        self._ensure_server_running()

        model = self._resolve_model(request)

        # Build JSON payload for wrapper stdin
        payload = {
            "prompt": self._build_prompt(request),
            "system_prompt": request.system_prompt,
            "config": {
                "model": model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        }

        # Build command to invoke wrapper
        command = [self._binary, str(self._wrapper_path)]
        if request.stream:
            command.append("--stream")

        # Execute wrapper with JSON payload via stdin
        timeout = request.timeout or self._timeout
        try:
            completed = self._runner(
                command,
                timeout=int(timeout) if timeout else None,
                env=self._env,
                input_data=json.dumps(payload),
            )
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(
                f"Node.js binary '{self._binary}' not found",
                provider=self.metadata.provider_id,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderTimeoutError(
                f"OpenCode wrapper timed out after {timeout}s",
                provider=self.metadata.provider_id,
                elapsed=float(timeout) if timeout else None,
                timeout=float(timeout) if timeout else None,
            ) from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            logger.debug(f"OpenCode wrapper stderr: {stderr or 'no stderr'}")

            # Extract error from JSONL stdout (wrapper outputs {"type":"error","message":"..."})
            jsonl_error = self._extract_error_from_jsonl(completed.stdout)

            error_msg = f"OpenCode wrapper exited with code {completed.returncode}"
            if jsonl_error:
                error_msg += f": {jsonl_error[:500]}"
            elif stderr:
                error_msg += f": {stderr[:500]}"
            raise ProviderExecutionError(
                error_msg,
                provider=self.metadata.provider_id,
            )

        # Parse line-delimited JSON output
        content_parts: List[str] = []
        final_usage: Optional[TokenUsage] = None
        raw_payload: Dict[str, Any] = {}
        reported_model = model

        for line in completed.stdout.strip().split("\n"):
            if not line.strip():
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.debug(f"OpenCode wrapper JSON parse error: {exc}")
                raise ProviderExecutionError(
                    "OpenCode wrapper returned invalid JSON response",
                    provider=self.metadata.provider_id,
                ) from exc

            msg_type = msg.get("type")

            if msg_type == "chunk":
                # Streaming chunk
                chunk_content = msg.get("content", "")
                content_parts.append(chunk_content)
                if request.stream:
                    self._emit_stream_chunk(
                        StreamChunk(content=chunk_content, index=len(content_parts) - 1)
                    )

            elif msg_type == "done":
                # Final response with metadata
                response_data = msg.get("response", {})
                final_text = response_data.get("text", "")
                if final_text and not content_parts:
                    content_parts.append(final_text)

                # Extract model from response
                reported_model = response_data.get("model", model)

                # Extract token usage
                usage_data = response_data.get("usage", {})
                final_usage = TokenUsage(
                    input_tokens=usage_data.get("prompt_tokens", 0),
                    output_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )
                raw_payload = response_data

            elif msg_type == "error":
                # Error from wrapper
                error_msg = msg.get("message", "Unknown error")
                raise ProviderExecutionError(
                    f"OpenCode wrapper error: {error_msg}",
                    provider=self.metadata.provider_id,
                )

        # Combine all content parts
        final_content = "".join(content_parts)

        # Validate we got actual content (defense in depth)
        if not final_content.strip():
            stderr_info = (completed.stderr or "").strip() or "none"
            raise ProviderExecutionError(
                f"OpenCode wrapper returned empty response. "
                f"Server may not be running on port 4096. Stderr: {stderr_info}",
                provider=self.metadata.provider_id,
            )

        # Emit final content if streaming was requested
        self._emit_stream_if_requested(final_content, stream=request.stream)

        # Use default usage if not provided
        if final_usage is None:
            final_usage = TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0)

        return ProviderResult(
            content=final_content,
            provider_id=self.metadata.provider_id,
            model_used=f"{self.metadata.provider_id}:{reported_model}",
            status=ProviderStatus.SUCCESS,
            tokens=final_usage,
            stderr=(completed.stderr or "").strip() or None,
            raw_payload=raw_payload,
        )


def is_opencode_available() -> bool:
    """OpenCode provider availability check."""
    return detect_provider_availability("opencode")


def create_provider(
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    dependencies: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> OpenCodeProvider:
    """
    Factory function for creating OpenCodeProvider instances.

    Args:
        hooks: Provider hooks for callbacks
        model: Optional model ID override
        dependencies: Optional dependencies (runner, env, binary)
        overrides: Optional parameter overrides

    Returns:
        Configured OpenCodeProvider instance
    """
    dependencies = dependencies or {}
    overrides = overrides or {}

    runner = dependencies.get("runner")
    env = dependencies.get("env")
    binary = overrides.get("binary") or dependencies.get("binary")
    wrapper_path = overrides.get("wrapper_path") or dependencies.get("wrapper_path")
    timeout = overrides.get("timeout") or dependencies.get("timeout")
    selected_model = overrides.get("model") if overrides.get("model") else model

    return OpenCodeProvider(
        metadata=OPENCODE_METADATA,
        hooks=hooks,
        model=selected_model,  # type: ignore[arg-type]
        binary=binary,  # type: ignore[arg-type]
        wrapper_path=wrapper_path,  # type: ignore[arg-type]
        runner=runner if runner is not None else None,  # type: ignore[arg-type]
        env=env if env is not None else None,  # type: ignore[arg-type]
        timeout=timeout if timeout is not None else None,  # type: ignore[arg-type]
    )


# Register the provider immediately so consumers can resolve it by id.
register_provider(
    "opencode",
    factory=create_provider,
    metadata=OPENCODE_METADATA,
    availability_check=is_opencode_available,
    description="OpenCode AI SDK adapter with Node.js wrapper",
    tags=("sdk", "text", "streaming", "read-only"),
    replace=True,
)


__all__ = [
    "OpenCodeProvider",
    "create_provider",
    "is_opencode_available",
    "OPENCODE_METADATA",
]
