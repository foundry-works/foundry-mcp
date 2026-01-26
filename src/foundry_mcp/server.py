"""FastMCP server for foundry-mcp.

This server exposes the unified 16-router tool surface described in
`mcp/capabilities_manifest.json`.

Note: Legacy per-tool-name MCP endpoints are intentionally not registered.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig, get_config
from foundry_mcp.core.observability import audit_log, get_observability_manager
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


def _init_timeout_watchdog() -> None:
    """Initialize the timeout watchdog for background task monitoring."""
    try:
        from foundry_mcp.core.timeout_watchdog import start_watchdog

        async def _start_watchdog_async() -> None:
            await start_watchdog(
                poll_interval=10.0,  # Check every 10 seconds
                stale_threshold=300.0,  # 5 minutes without activity = stale
            )
            logger.info("Timeout watchdog started")

        import asyncio

        asyncio.create_task(_start_watchdog_async())
    except Exception as exc:
        # Don't fail server startup due to optional watchdog
        logger.warning("Failed to initialize timeout watchdog: %s", exc)


def _build_lifespan(config: ServerConfig):
    """Create server lifespan handler to manage background services."""

    @asynccontextmanager
    async def _lifespan(_app: FastMCP):
        _init_timeout_watchdog()
        _init_provider_executor(config)
        try:
            yield
        finally:
            await shutdown_timeout_watchdog()
            await shutdown_provider_executor()

    return _lifespan


async def shutdown_timeout_watchdog() -> None:
    """Shutdown the timeout watchdog gracefully.

    Should be called during server shutdown to stop the watchdog
    background task cleanly.
    """
    try:
        from foundry_mcp.core.timeout_watchdog import stop_watchdog

        await stop_watchdog(timeout=5.0)
        logger.info("Timeout watchdog stopped")
    except Exception as exc:
        logger.warning("Error stopping timeout watchdog: %s", exc)


def _init_provider_executor(config: ServerConfig) -> None:
    """Initialize the provider executor for blocking operation isolation.

    Creates a dedicated thread pool executor for CLI provider subprocess calls
    and other blocking operations to prevent event loop starvation.
    """
    try:
        from foundry_mcp.core.executor import configure_executor

        # Use config values or defaults
        pool_size = getattr(config, "executor_pool_size", 4)
        queue_limit = getattr(config, "executor_queue_limit", 100)
        enabled = getattr(config, "executor_isolation_enabled", True)

        executor = configure_executor(
            pool_size=pool_size,
            queue_limit=queue_limit,
            enabled=enabled,
        )
        executor.start()
        logger.info(
            "Provider executor initialized: pool_size=%d, queue_limit=%d, enabled=%s",
            pool_size,
            queue_limit,
            enabled,
        )
    except Exception as exc:
        # Don't fail server startup due to optional executor
        logger.warning("Failed to initialize provider executor: %s", exc)


async def shutdown_provider_executor() -> None:
    """Shutdown the provider executor gracefully.

    Should be called during server shutdown to stop the executor
    and wait for pending tasks to complete.
    """
    try:
        from foundry_mcp.core.executor import get_provider_executor

        executor = get_provider_executor()
        await executor.shutdown(wait=True)
        logger.info("Provider executor stopped")
    except Exception as exc:
        logger.warning("Error stopping provider executor: %s", exc)


def create_server(config: Optional[ServerConfig] = None) -> FastMCP:
    """Create and configure the FastMCP server instance."""

    if config is None:
        config = get_config()

    config.setup_logging()

    _init_observability(config)
    _init_error_collection(config)
    _init_metrics_persistence(config)
    mcp = FastMCP(name=config.server_name, lifespan=_build_lifespan(config))

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
