"""FastMCP server for foundry-mcp.

This server exposes the unified 17-router tool surface described in
`mcp/capabilities_manifest.json`.

Note: Legacy per-tool-name MCP endpoints are intentionally not registered.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig, get_config
from foundry_mcp.core.observability import audit_log, get_observability_manager
from foundry_mcp.core.feature_flags import get_flag_service
from foundry_mcp.resources.specs import register_spec_resources
from foundry_mcp.prompts.workflows import register_workflow_prompts
from foundry_mcp.tools.unified import register_unified_tools

logger = logging.getLogger(__name__)


def _init_observability(config: ServerConfig) -> None:
    """Initialize the observability stack from server configuration."""

    obs_config = config.observability
    if not obs_config.enabled:
        logger.debug("Observability disabled in configuration")
        return

    manager = get_observability_manager()
    manager.initialize(obs_config)

    tracing_status = "enabled" if manager.is_tracing_enabled() else "disabled"
    metrics_status = "enabled" if manager.is_metrics_enabled() else "disabled"
    logger.info(
        "Observability initialized: tracing=%s, metrics=%s",
        tracing_status,
        metrics_status,
    )


def _init_error_collection(config: ServerConfig) -> None:
    """Initialize the error collection infrastructure."""

    err_config = config.error_collection
    if not err_config.enabled:
        logger.debug("Error collection disabled in configuration")
        return

    try:
        from foundry_mcp.core.error_collection import get_error_collector
        from foundry_mcp.core.error_store import get_error_store

        storage_path = err_config.get_storage_path()
        store = get_error_store(storage_path)
        collector = get_error_collector()
        collector.initialize(store, err_config)
        logger.info("Error collection initialized: storage_path=%s", storage_path)
    except Exception as exc:
        # Don't fail server startup due to optional error collection
        logger.warning("Failed to initialize error collection: %s", exc)


def _init_metrics_persistence(config: ServerConfig) -> None:
    """Initialize the metrics persistence infrastructure."""

    metrics_config = config.metrics_persistence
    if not metrics_config.enabled:
        logger.debug("Metrics persistence disabled in configuration")
        return

    try:
        from foundry_mcp.core.metrics_persistence import initialize_metrics_persistence
        from foundry_mcp.core.metrics_store import get_metrics_store

        collector = initialize_metrics_persistence(metrics_config)
        if collector is None:
            return

        storage_path = metrics_config.get_storage_path()
        store = get_metrics_store(storage_path)
        deleted_count = store.cleanup(
            retention_days=metrics_config.retention_days,
            max_records=metrics_config.max_records,
        )

        if deleted_count > 0:
            logger.info("Metrics cleanup: removed %s old records", deleted_count)

        logger.info("Metrics persistence initialized: storage_path=%s", storage_path)
    except Exception as exc:
        # Don't fail server startup due to optional persistence
        logger.warning("Failed to initialize metrics persistence: %s", exc)


def _apply_feature_flag_overrides_from_env() -> None:
    """Apply comma-separated feature flag overrides from `FEATURE_FLAGS`."""

    raw = os.environ.get("FEATURE_FLAGS")
    if not raw:
        return

    flag_service = get_flag_service()
    for name in [part.strip() for part in raw.split(",") if part.strip()]:
        flag_service.set_override("anonymous", name, True)


def create_server(config: Optional[ServerConfig] = None) -> FastMCP:
    """Create and configure the FastMCP server instance."""

    if config is None:
        config = get_config()

    config.setup_logging()

    _apply_feature_flag_overrides_from_env()
    _init_observability(config)
    _init_error_collection(config)
    _init_metrics_persistence(config)

    mcp = FastMCP(name=config.server_name)

    # Unified-only tool surface
    register_unified_tools(mcp, config)

    # Resources and prompts
    register_spec_resources(mcp, config)
    register_workflow_prompts(mcp, config)

    logger.info("Server created: %s v%s", config.server_name, config.server_version)
    return mcp


def main() -> None:
    """Main entry point for the foundry-mcp server."""

    try:
        config = get_config()
        server = create_server(config)

        logger.info("Starting %s v%s", config.server_name, config.server_version)
        audit_log("tool_invocation", tool="server_start", version=config.server_version)

        server.run()

    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        get_observability_manager().shutdown()
        sys.exit(0)
    except BaseException as exc:
        logger.error("Server error: %s: %s", type(exc).__name__, exc)
        audit_log("tool_invocation", tool="server_error", error=str(exc), success=False)
        get_observability_manager().shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
