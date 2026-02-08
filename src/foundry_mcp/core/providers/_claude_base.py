"""Shared base for Claude CLI-based providers (claude, claude-zai).

Extracts the common logic for availability checks, command construction,
response parsing, tool restrictions, and token usage normalization so that
concrete providers only need to specify their identity (binary name,
environment variables, metadata, display label).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .base import (
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

logger = logging.getLogger(__name__)

# ── Read-only tool restrictions shared by all Claude CLI providers ──────────

ALLOWED_TOOLS = [
    # File operations (read-only)
    "Read",
    "Grep",
    "Glob",
    # Task delegation
    "Task",
    # Bash commands - file viewing
    "Bash(cat)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(bat:*)",
    # Bash commands - directory listing/navigation
    "Bash(ls:*)",
    "Bash(tree:*)",
    "Bash(pwd)",
    "Bash(which:*)",
    "Bash(whereis:*)",
    # Bash commands - search/find
    "Bash(grep:*)",
    "Bash(rg:*)",
    "Bash(ag:*)",
    "Bash(find:*)",
    "Bash(fd:*)",
    # Bash commands - git operations (read-only)
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(git diff:*)",
    "Bash(git status:*)",
    "Bash(git grep:*)",
    "Bash(git blame:*)",
    "Bash(git branch:*)",
    "Bash(git rev-parse:*)",
    "Bash(git describe:*)",
    "Bash(git ls-tree:*)",
    # Bash commands - text processing
    "Bash(wc:*)",
    "Bash(cut:*)",
    "Bash(paste:*)",
    "Bash(column:*)",
    "Bash(sort:*)",
    "Bash(uniq:*)",
    # Bash commands - data formats
    "Bash(jq:*)",
    "Bash(yq:*)",
    # Bash commands - file analysis
    "Bash(file:*)",
    "Bash(stat:*)",
    "Bash(du:*)",
    "Bash(df:*)",
    # Bash commands - checksums/hashing
    "Bash(md5sum:*)",
    "Bash(shasum:*)",
    "Bash(sha256sum:*)",
    "Bash(sha512sum:*)",
]

DISALLOWED_TOOLS = [
    "Write",
    "Edit",
    # Web operations (data exfiltration risk)
    "WebSearch",
    "WebFetch",
    # Dangerous file operations
    "Bash(rm:*)",
    "Bash(rmdir:*)",
    "Bash(dd:*)",
    "Bash(mkfs:*)",
    "Bash(fdisk:*)",
    # File modifications
    "Bash(touch:*)",
    "Bash(mkdir:*)",
    "Bash(mv:*)",
    "Bash(cp:*)",
    "Bash(chmod:*)",
    "Bash(chown:*)",
    "Bash(sed:*)",
    "Bash(awk:*)",
    # Git write operations
    "Bash(git add:*)",
    "Bash(git commit:*)",
    "Bash(git push:*)",
    "Bash(git pull:*)",
    "Bash(git merge:*)",
    "Bash(git rebase:*)",
    "Bash(git reset:*)",
    "Bash(git checkout:*)",
    # Package installations
    "Bash(npm install:*)",
    "Bash(pip install:*)",
    "Bash(apt install:*)",
    "Bash(brew install:*)",
    # System operations
    "Bash(sudo:*)",
    "Bash(halt:*)",
    "Bash(reboot:*)",
    "Bash(shutdown:*)",
]

SHELL_COMMAND_WARNING = """
IMPORTANT SECURITY NOTE: When using shell commands, be aware of the following restrictions:
1. Only specific read-only commands are allowed (cat, grep, git log, etc.)
2. Write operations, file modifications, and destructive commands are blocked
3. Avoid using piped commands as they may bypass some security checks
4. Use sequential commands or alternative approaches when possible
"""


# ── Runner protocol ────────────────────────────────────────────────────────

class RunnerProtocol(Protocol):
    """Callable signature used for executing Claude CLI commands."""

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout: Optional[int] = None,
        env: Optional[Dict[str, str]] = None,
        input_data: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        raise NotImplementedError


def default_runner(
    command: Sequence[str],
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    input_data: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke a Claude CLI binary via subprocess."""
    return subprocess.run(  # noqa: S603,S607 - intentional CLI invocation
        list(command),
        capture_output=True,
        text=True,
        input=input_data,
        timeout=timeout,
        env=env,
        check=False,
    )


# ── Base provider ──────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_SECONDS = 360


class ClaudeCLIProviderBase(ProviderContext):
    """Shared implementation for providers backed by a Claude CLI binary.

    Subclasses only need to supply their own metadata, constants, and
    module-level registration.  All command construction, parsing, tool
    restriction, and error handling is shared here.
    """

    # Subclasses should set this for human-readable error messages.
    _cli_label: str = "Claude CLI"

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
        default_binary: str = "claude",
        custom_binary_env: str = "CLAUDE_CLI_BINARY",
    ):
        super().__init__(metadata, hooks)
        self._runner = runner or default_runner
        self._binary = binary or os.environ.get(custom_binary_env, default_binary)
        self._env = env
        self._timeout = timeout or DEFAULT_TIMEOUT_SECONDS
        self._model = model or metadata.default_model or "opus"

    # ── Request validation ─────────────────────────────────────────────

    def _validate_request(self, request: ProviderRequest) -> None:
        unsupported: List[str] = []
        if request.temperature is not None:
            unsupported.append("temperature")
        if request.max_tokens is not None:
            unsupported.append("max_tokens")
        if request.attachments:
            unsupported.append("attachments")
        if unsupported:
            logger.warning(
                f"{self._cli_label} ignoring unsupported parameters: {', '.join(unsupported)}"
            )

    # ── Command construction ───────────────────────────────────────────

    def _build_command(
        self, model: str, system_prompt: Optional[str] = None
    ) -> List[str]:
        command = [self._binary, "--print", "--output-format", "json"]
        command.extend(["--allowed-tools"] + ALLOWED_TOOLS)
        command.extend(["--disallowed-tools"] + DISALLOWED_TOOLS)

        full_system_prompt = system_prompt or ""
        if full_system_prompt:
            full_system_prompt = f"{full_system_prompt.strip()}\n\n{SHELL_COMMAND_WARNING.strip()}"
        else:
            full_system_prompt = SHELL_COMMAND_WARNING.strip()
        command.extend(["--system-prompt", full_system_prompt])

        if model and model != self.metadata.default_model:
            command.extend(["--model", model])
        return command

    # ── Subprocess execution ───────────────────────────────────────────

    def _run(
        self,
        command: Sequence[str],
        timeout: Optional[float],
        input_data: Optional[str] = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                command,
                timeout=int(timeout) if timeout else None,
                env=self._env,
                input_data=input_data,
            )
        except FileNotFoundError as exc:
            raise ProviderUnavailableError(
                f"{self._cli_label} '{self._binary}' is not available on PATH.",
                provider=self.metadata.provider_id,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ProviderTimeoutError(
                f"Command timed out after {exc.timeout} seconds",
                provider=self.metadata.provider_id,
                elapsed=float(exc.timeout) if exc.timeout else None,
                timeout=float(exc.timeout) if exc.timeout else None,
            ) from exc

    # ── Output parsing ─────────────────────────────────────────────────

    def _parse_output(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()
        if not text:
            raise ProviderExecutionError(
                f"{self._cli_label} returned empty output.",
                provider=self.metadata.provider_id,
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug(f"{self._cli_label} JSON parse error: {exc}")
            raise ProviderExecutionError(
                f"{self._cli_label} returned invalid JSON response",
                provider=self.metadata.provider_id,
            ) from exc

    def _extract_usage(self, payload: Dict[str, Any]) -> TokenUsage:
        usage = payload.get("usage") or {}
        return TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
            total_tokens=int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
        )

    def _resolve_model(self, request: ProviderRequest) -> str:
        if request.model:
            return str(request.model)
        model_override = request.metadata.get("model") if request.metadata else None
        if model_override:
            return str(model_override)
        return self._model

    def _emit_stream_if_requested(self, content: str, *, stream: bool) -> None:
        if not stream or not content:
            return
        self._emit_stream_chunk(StreamChunk(content=content, index=0))

    def _extract_error_from_json(self, stdout: str) -> Optional[str]:
        if not stdout:
            return None
        try:
            payload = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return None
        if payload.get("is_error"):
            result = payload.get("result", "")
            if result:
                return str(result)
        error = payload.get("error")
        if error:
            if isinstance(error, dict):
                return error.get("message") or str(error)
            return str(error)
        return None

    # ── Main execution ─────────────────────────────────────────────────

    def _execute(self, request: ProviderRequest) -> ProviderResult:
        self._validate_request(request)
        model = self._resolve_model(request)
        command = self._build_command(model, system_prompt=request.system_prompt)
        timeout = request.timeout or self._timeout
        completed = self._run(command, timeout=timeout, input_data=request.prompt)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            logger.debug(f"{self._cli_label} stderr: {stderr or 'no stderr'}")
            json_error = self._extract_error_from_json(completed.stdout)
            error_msg = f"{self._cli_label} exited with code {completed.returncode}"
            if json_error:
                error_msg += f": {json_error[:500]}"
            elif stderr:
                error_msg += f": {stderr[:500]}"
            raise ProviderExecutionError(
                error_msg,
                provider=self.metadata.provider_id,
            )

        payload = self._parse_output(completed.stdout)
        content = str(
            payload.get("result") or payload.get("content") or ""
        ).strip()
        model_usage = payload.get("modelUsage") or {}
        reported_model = list(model_usage.keys())[0] if model_usage else model
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
