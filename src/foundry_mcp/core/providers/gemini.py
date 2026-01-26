"""
Gemini CLI provider implementation.

Bridges the `gemini` command-line interface to the ProviderContext contract by
handling availability checks, safe command construction, response parsing, and
token usage normalization.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
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

DEFAULT_BINARY = "gemini"
DEFAULT_TIMEOUT_SECONDS = 360
AVAILABILITY_OVERRIDE_ENV = "GEMINI_CLI_AVAILABLE_OVERRIDE"
CUSTOM_BINARY_ENV = "GEMINI_CLI_BINARY"

# Read-only tools allowed for safe codebase exploration
# Based on Gemini CLI tool names (both class names and function names supported)
ALLOWED_TOOLS = [
    # Core file operations (read-only)
    "ReadFileTool",
    "read_file",
    "ReadManyFilesTool",
    "read_many_files",
    "LSTool",
    "list_directory",
    "GlobTool",
    "glob",
    "GrepTool",
    "search_file_content",
    # Shell commands - file viewing
    "ShellTool(cat)",
    "ShellTool(head)",
    "ShellTool(tail)",
    "ShellTool(bat)",
    # Shell commands - directory listing/navigation
    "ShellTool(ls)",
    "ShellTool(tree)",
    "ShellTool(pwd)",
    "ShellTool(which)",
    "ShellTool(whereis)",
    # Shell commands - search/find
    "ShellTool(grep)",
    "ShellTool(rg)",
    "ShellTool(ag)",
    "ShellTool(find)",
    "ShellTool(fd)",
    # Shell commands - git operations (read-only)
    "ShellTool(git log)",
    "ShellTool(git show)",
    "ShellTool(git diff)",
    "ShellTool(git status)",
    "ShellTool(git grep)",
    "ShellTool(git blame)",
    # Shell commands - text processing
    "ShellTool(wc)",
    "ShellTool(cut)",
    "ShellTool(paste)",
    "ShellTool(column)",
    "ShellTool(sort)",
    "ShellTool(uniq)",
    # Shell commands - data formats
    "ShellTool(jq)",
    "ShellTool(yq)",
    # Shell commands - file analysis
    "ShellTool(file)",
    "ShellTool(stat)",
    "ShellTool(du)",
    "ShellTool(df)",
    # Shell commands - checksums/hashing
    "ShellTool(md5sum)",
    "ShellTool(shasum)",
    "ShellTool(sha256sum)",
    "ShellTool(sha512sum)",
]

# System prompt addition warning about piped command vulnerability
PIPED_COMMAND_WARNING = """
IMPORTANT SECURITY NOTE: When using shell commands, avoid piped commands (e.g., cat file.txt | wc -l).
Piped commands bypass the tool allowlist checks in Gemini CLI - only the first command in a pipe is validated.
Instead, use sequential commands or alternative approaches to achieve the same result safely.
"""


class RunnerProtocol(Protocol):
    """Callable signature used for executing Gemini CLI commands."""

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
    """Invoke the Gemini CLI via subprocess."""
    return subprocess.run(  # noqa: S603,S607 - intentional CLI invocation
        list(command),
        capture_output=True,
        text=True,
        input=input_data,
        timeout=timeout,
        env=env,
        check=False,
    )


GEMINI_METADATA = ProviderMetadata(
    provider_id="gemini",
    display_name="Google Gemini CLI",
    models=[],  # Model validation delegated to CLI
    default_model="pro",
    capabilities={ProviderCapability.TEXT, ProviderCapability.STREAMING, ProviderCapability.VISION},
    security_flags={"writes_allowed": False},
    extra={"cli": "gemini", "output_format": "json"},
)


class GeminiProvider(ProviderContext):
    """ProviderContext implementation backed by the Gemini CLI."""

    def __init__(
        self,
        metadata: ProviderMetadata,
        hooks: ProviderHooks,
        *,
        model: Optional[str] = None,
        binary: Optional[str] = None,
        runner: Optional[RunnerProtocol] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ):
        super().__init__(metadata, hooks)
        self._runner = runner or _default_runner
        self._binary = binary or os.environ.get(CUSTOM_BINARY_ENV, DEFAULT_BINARY)
        self._env = env
        self._timeout = timeout or DEFAULT_TIMEOUT_SECONDS
        self._model = model or metadata.default_model or "pro"

    def _validate_request(self, request: ProviderRequest) -> None:
        """Validate and normalize request, ignoring unsupported parameters."""
        unsupported: List[str] = []
        if request.temperature is not None:
            unsupported.append("temperature")
        if request.max_tokens is not None:
            unsupported.append("max_tokens")
        if request.attachments:
            unsupported.append("attachments")
        if unsupported:
            # Log warning but continue - ignore unsupported parameters
            logger.warning(
                f"Gemini CLI ignoring unsupported parameters: {', '.join(unsupported)}"
            )

    def _build_prompt(self, request: ProviderRequest) -> str:
        # Build the system prompt with security warning
        system_parts = []
        if request.system_prompt:
            system_parts.append(request.system_prompt.strip())
        system_parts.append(PIPED_COMMAND_WARNING.strip())

        if system_parts:
            return f"{chr(10).join(system_parts)}\n\n{request.prompt}"
        return request.prompt

    def _build_command(self, model: str) -> List[str]:
        """
        Build Gemini CLI command with read-only tool restrictions.

        Prompt is passed via stdin to avoid CLI argument length limits.
        """
        command = [self._binary, "--output-format", "json"]

        # Add allowed tools for read-only enforcement
        for tool in ALLOWED_TOOLS:
            command.extend(["--allowed-tools", tool])

        # Insert model if specified
        if model:
            command[1:1] = ["-m", model]

        return command

    def _run(
        self, command: Sequence[str], timeout: Optional[float], input_data: Optional[str] = None
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                command, timeout=int(timeout) if timeout else None, env=self._env, input_data=input_data
            )
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(
                f"Gemini CLI '{self._binary}' is not available on PATH.",
                provider=self.metadata.provider_id,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderTimeoutError(
                f"Command timed out after {exc.timeout} seconds",
                provider=self.metadata.provider_id,
                elapsed=float(exc.timeout) if exc.timeout else None,
                timeout=float(exc.timeout) if exc.timeout else None,
            ) from exc

    def _parse_output(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ProviderExecutionError(
                "Gemini CLI returned empty output.",
                provider=self.metadata.provider_id,
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug(f"Gemini CLI JSON parse error: {exc}")
            raise ProviderExecutionError(
                "Gemini CLI returned invalid JSON response",
                provider=self.metadata.provider_id,
            ) from exc

    def _extract_usage(self, payload: Dict[str, Any]) -> TokenUsage:
        stats = payload.get("stats") or {}
        models_section = stats.get("models") or {}
        first_model = next(iter(models_section.values()), {})
        tokens = first_model.get("tokens") or {}
        return TokenUsage(
            input_tokens=int(tokens.get("prompt") or tokens.get("input") or 0),
            output_tokens=int(tokens.get("candidates") or tokens.get("output") or 0),
            total_tokens=int(tokens.get("total") or 0),
        )

    def _resolve_model(self, request: ProviderRequest) -> str:
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
        if not stream or not content:
            return
        self._emit_stream_chunk(StreamChunk(content=content, index=0))

    def _extract_error_from_output(self, stdout: str) -> Optional[str]:
        """
        Extract error message from Gemini CLI output.

        Gemini CLI outputs errors as text lines followed by JSON. Example:
        'Error when talking to Gemini API Full report available at: /tmp/...
        {"error": {"type": "Error", "message": "[object Object]", "code": 1}}'

        The JSON message field is often unhelpful ("[object Object]"), so we
        prefer the text prefix which contains the actual error description.
        """
        if not stdout:
            return None

        lines = stdout.strip().split("\n")
        error_parts: List[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip "Loaded cached credentials" info line
            if line.startswith("Loaded cached"):
                continue

            # Try to parse as JSON
            if line.startswith("{"):
                try:
                    payload = json.loads(line)
                    error = payload.get("error", {})
                    if isinstance(error, dict):
                        msg = error.get("message", "")
                        # Skip unhelpful "[object Object]" message
                        if msg and msg != "[object Object]":
                            error_parts.append(msg)
                except json.JSONDecodeError:
                    pass
            else:
                # Text line - likely contains the actual error message
                # Extract the part before "Full report available at:"
                if "Full report available at:" in line:
                    line = line.split("Full report available at:")[0].strip()
                if line:
                    error_parts.append(line)

        return "; ".join(error_parts) if error_parts else None

    def _execute(self, request: ProviderRequest) -> ProviderResult:
        self._validate_request(request)
        model = self._resolve_model(request)
        prompt = self._build_prompt(request)
        command = self._build_command(model)
        timeout = request.timeout or self._timeout
        # Pass prompt via stdin to avoid CLI argument length limits
        completed = self._run(command, timeout=timeout, input_data=prompt)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            logger.debug(f"Gemini CLI stderr: {stderr or 'no stderr'}")

            # Extract error from stdout (Gemini outputs errors as text + JSON)
            stdout_error = self._extract_error_from_output(completed.stdout)

            error_msg = f"Gemini CLI exited with code {completed.returncode}"
            if stdout_error:
                error_msg += f": {stdout_error[:500]}"
            elif stderr:
                error_msg += f": {stderr[:500]}"
            raise ProviderExecutionError(
                error_msg,
                provider=self.metadata.provider_id,
            )

        payload = self._parse_output(completed.stdout)
        content = str(payload.get("response") or payload.get("content") or "").strip()
        reported_model = payload.get("model") or next(
            iter((payload.get("stats") or {}).get("models") or {}), model
        )
        usage = self._extract_usage(payload)

        self._emit_stream_if_requested(content, stream=request.stream)

        return ProviderResult(
            content=content,
            provider_id=self.metadata.provider_id,
            model_used=f"{self.metadata.provider_id}:{reported_model}",
            status=ProviderStatus.SUCCESS,
            tokens=usage,
            stderr=(completed.stderr or "").strip() or None,
            raw_payload=payload,
        )


def is_gemini_available() -> bool:
    """Gemini CLI availability check."""
    return detect_provider_availability("gemini")


def create_provider(
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    dependencies: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> GeminiProvider:
    """
    Factory used by the provider registry.

    dependencies/overrides allow callers (or tests) to inject runner/env/binary.
    """
    dependencies = dependencies or {}
    overrides = overrides or {}
    runner = dependencies.get("runner")
    env = dependencies.get("env")
    binary = overrides.get("binary") or dependencies.get("binary")
    timeout = overrides.get("timeout")
    selected_model = overrides.get("model") if overrides.get("model") else model

    return GeminiProvider(
        metadata=GEMINI_METADATA,
        hooks=hooks,
        model=selected_model,  # type: ignore[arg-type]
        binary=binary,  # type: ignore[arg-type]
        runner=runner if runner is not None else None,  # type: ignore[arg-type]
        env=env if env is not None else None,  # type: ignore[arg-type]
        timeout=timeout if timeout is not None else None,  # type: ignore[arg-type]
    )


# Register the provider immediately so consumers can resolve it by id.
register_provider(
    "gemini",
    factory=create_provider,
    metadata=GEMINI_METADATA,
    availability_check=is_gemini_available,
    description="Google Gemini CLI adapter",
    tags=("cli", "text", "vision"),
    replace=True,
)


__all__ = [
    "GeminiProvider",
    "create_provider",
    "is_gemini_available",
    "GEMINI_METADATA",
]
