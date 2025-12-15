"""Context helpers shared by unified tool routers.

These helpers keep `server(action=...)` focused on routing/validation while
ensuring context/LLM status responses remain consistent and response-v2
compliant.

This module intentionally lives under `tools.unified` to avoid reintroducing
non-unified public tool surfaces.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)


def build_llm_status_response(*, request_id: Optional[str] = None) -> dict:
    """Return a standardized envelope describing LLM configuration."""

    try:
        from foundry_mcp.core.review import get_llm_status

        llm_status = get_llm_status()
        return asdict(success_response(llm_status=llm_status, request_id=request_id))
    except Exception as exc:
        logger.exception("Failed to build llm_status response")
        return asdict(
            error_response(
                f"Failed to build llm_status response: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check server logs.",
                request_id=request_id,
            )
        )


def build_server_context_response(
    config: ServerConfig,
    *,
    include_llm: bool = True,
    include_workflow: bool = True,
    include_workspace: bool = True,
    include_capabilities: bool = True,
    request_id: Optional[str] = None,
) -> dict:
    """Build a standardized server context payload."""

    payload: Dict[str, Any] = {
        "server": {
            "name": config.server_name,
            "version": config.server_version,
            "log_level": config.log_level,
        },
        "paths": {
            "specs_dir": str(config.specs_dir) if config.specs_dir else None,
            "journals_path": str(config.journals_path)
            if config.journals_path
            else None,
        },
    }

    if include_workspace:
        payload["workspace"] = {"roots": [str(p) for p in config.workspace_roots]}

    if include_workflow:
        payload["workflow"] = {"git": asdict(config.git)}

    if include_llm:
        try:
            from foundry_mcp.core.review import get_llm_status

            payload["llm_status"] = get_llm_status()
        except Exception as exc:
            logger.debug("Failed to compute llm_status: %s", exc)
            payload["llm_status"] = {"configured": False, "error": "unavailable"}

    if include_capabilities:
        try:
            from foundry_mcp.core.discovery import get_capabilities

            payload["capabilities"] = get_capabilities()
        except Exception as exc:
            logger.debug("Failed to compute capabilities: %s", exc)
            payload["capabilities"] = {}

    return asdict(success_response(data=payload, request_id=request_id))
