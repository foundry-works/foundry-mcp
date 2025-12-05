"""
Server context tools for foundry-mcp.

Provides MCP tools for querying server capabilities, workspace info,
and configuration (with sensitive data masked).
"""

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response, sanitize_error_message
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.resilience import CircuitBreaker
from foundry_mcp.core.observability import get_metrics, mcp_tool

logger = logging.getLogger(__name__)

# Circuit breaker for context operations
_context_breaker = CircuitBreaker(
    name="context",
    failure_threshold=5,
    recovery_timeout=30.0,
)


def _mask_sensitive_value(value: Optional[str], visible_chars: int = 4) -> Optional[str]:
    """Mask a sensitive string value, showing only first few characters.

    Args:
        value: The value to mask
        visible_chars: Number of characters to show at start

    Returns:
        Masked string like "sk-a...****" or None if value is None
    """
    if not value:
        return None
    if len(value) <= visible_chars:
        return "*" * len(value)
    return value[:visible_chars] + "..." + "*" * 4


def _get_llm_config_safe() -> Dict[str, Any]:
    """Get LLM configuration with sensitive data masked.

    Returns:
        Dict with LLM config, API keys masked
    """
    try:
        from foundry_mcp.core.llm_config import get_llm_config, LLMProviderType

        config = get_llm_config()
        return {
            "provider": config.provider.value,
            "model": config.get_model(),
            "timeout": config.timeout,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "base_url": config.base_url,
            "organization": _mask_sensitive_value(config.organization),
            "api_key_configured": config.get_api_key() is not None,
            "api_key_source": _get_api_key_source(config),
        }
    except ImportError:
        return {"error": "LLM config module not available"}
    except Exception as e:
        logger.warning(f"Failed to get LLM config: {e}")
        return {"error": "Failed to load LLM configuration"}


def _get_api_key_source(config) -> Optional[str]:
    """Determine where the API key is sourced from.

    Returns:
        String indicating source: "config", "env_unified", "env_provider", or None
    """
    import os
    from foundry_mcp.core.llm_config import API_KEY_ENV_VARS

    if config.api_key:
        return "config"
    if os.environ.get("FOUNDRY_MCP_LLM_API_KEY"):
        return "env_unified"
    env_var = API_KEY_ENV_VARS.get(config.provider, "")
    if env_var and os.environ.get(env_var):
        return "env_provider"
    return None


def _get_workflow_config_safe() -> Dict[str, Any]:
    """Get workflow configuration.

    Returns:
        Dict with workflow config
    """
    try:
        from foundry_mcp.core.llm_config import get_workflow_config

        config = get_workflow_config()
        return {
            "mode": config.mode.value,
            "auto_validate": config.auto_validate,
            "journal_enabled": config.journal_enabled,
            "batch_size": config.batch_size,
            "context_threshold": config.context_threshold,
        }
    except ImportError:
        return {"error": "Workflow config module not available"}
    except Exception as e:
        logger.warning(f"Failed to get workflow config: {e}")
        return {"error": "Failed to load workflow configuration"}


def register_context_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register context tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """
    metrics = get_metrics()

    @canonical_tool(
        mcp,
        canonical_name="get-server-context",
    )
    @mcp_tool(tool_name="get-server-context", emit_metrics=True, audit=False)
    def get_server_context(
        include_llm: bool = True,
        include_workflow: bool = True,
        include_workspace: bool = True,
        include_capabilities: bool = True,
    ) -> dict:
        """
        Get server context including capabilities, workspace info, and configuration.

        Returns workspace paths, LLM provider settings (with masked secrets),
        workflow configuration, and negotiated capabilities.

        Args:
            include_llm: Include LLM provider configuration
            include_workflow: Include workflow mode configuration
            include_workspace: Include workspace paths and spec directory info
            include_capabilities: Include server capabilities and feature flags

        Returns:
            JSON object with:
            - server: Server name and version
            - workspace: Paths and spec directory info (if requested)
            - llm: LLM provider config with masked API keys (if requested)
            - workflow: Workflow mode settings (if requested)
            - capabilities: Enabled features and tool categories (if requested)

        WHEN TO USE:
        - Discover server capabilities before using features
        - Check which LLM provider is configured
        - Verify workspace paths are correct
        - Debug configuration issues
        - Check workflow mode before starting tasks
        """
        start_time = time.perf_counter()

        try:
            # Circuit breaker check
            if not _context_breaker.can_execute():
                status = _context_breaker.get_status()
                metrics.counter(
                    "context.circuit_breaker_open",
                    labels={"tool": "get-server-context"},
                )
                return asdict(
                    error_response(
                        "Context operations temporarily unavailable",
                        data={
                            "retry_after_seconds": status.get("retry_after_seconds"),
                            "breaker_state": status.get("state"),
                        },
                    )
                )

            result: Dict[str, Any] = {
                "server": {
                    "name": config.server_name,
                    "version": config.server_version,
                },
            }

            # Workspace info
            if include_workspace:
                result["workspace"] = {
                    "specs_dir": str(config.specs_dir) if config.specs_dir else None,
                    "workspace_roots": [str(p) for p in config.workspace_roots],
                    "journals_path": str(config.journals_path) if config.journals_path else None,
                }

            # LLM configuration (masked)
            if include_llm:
                result["llm"] = _get_llm_config_safe()

            # Workflow configuration
            if include_workflow:
                result["workflow"] = _get_workflow_config_safe()

            # Server capabilities
            if include_capabilities:
                result["capabilities"] = {
                    "tools": {
                        "spec_management": True,
                        "task_operations": True,
                        "validation": True,
                        "journaling": True,
                        "lifecycle": True,
                        "documentation": True,
                        "testing": True,
                        "llm_integration": True,
                    },
                    "resources": {
                        "specs": True,
                        "templates": True,
                        "journals": True,
                    },
                    "prompts": {
                        "start_feature": True,
                        "debug_test": True,
                        "complete_phase": True,
                        "review_spec": True,
                    },
                    "features": {
                        "auth_enabled": config.require_auth,
                        "structured_logging": config.structured_logging,
                    },
                }

            _context_breaker.record_success()
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.histogram(
                "context.duration_ms",
                duration_ms,
                labels={"tool": "get-server-context"},
            )

            return asdict(success_response(result))

        except Exception as e:
            _context_breaker.record_failure()
            logger.exception("Error getting server context")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="server context"),
                )
            )

    @canonical_tool(
        mcp,
        canonical_name="get-llm-status",
    )
    @mcp_tool(tool_name="get-llm-status", emit_metrics=True, audit=False)
    def get_llm_status() -> dict:
        """
        Get LLM provider status and health check.

        Returns current LLM configuration and performs a lightweight
        connectivity check if possible.

        Returns:
            JSON object with:
            - provider: Current provider type
            - model: Configured model
            - configured: Whether API key is available
            - source: Where API key is sourced from
            - health: Provider health status (if checkable)

        WHEN TO USE:
        - Verify LLM provider is correctly configured
        - Check API key availability before LLM operations
        - Debug LLM connection issues
        """
        start_time = time.perf_counter()

        try:
            llm_config = _get_llm_config_safe()

            result = {
                "provider": llm_config.get("provider"),
                "model": llm_config.get("model"),
                "configured": llm_config.get("api_key_configured", False),
                "source": llm_config.get("api_key_source"),
                "timeout": llm_config.get("timeout"),
                "max_tokens": llm_config.get("max_tokens"),
            }

            # Add health check hint
            if not result["configured"] and result["provider"] != "local":
                result["health"] = "unconfigured"
                result["hint"] = (
                    f"Set FOUNDRY_MCP_LLM_API_KEY or provider-specific env var "
                    f"to enable {result['provider']} provider"
                )
            else:
                result["health"] = "ready"

            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics.histogram(
                "context.duration_ms",
                duration_ms,
                labels={"tool": "get-llm-status"},
            )

            return asdict(success_response(result))

        except Exception as e:
            logger.exception("Error getting LLM status")
            return asdict(
                error_response(
                    sanitize_error_message(e, context="LLM status"),
                )
            )
