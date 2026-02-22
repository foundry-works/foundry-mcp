"""Unified authoring router — split into domain-focused handler modules."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import mcp_tool
from foundry_mcp.tools.unified.authoring_handlers._helpers import _ACTION_SUMMARY
from foundry_mcp.tools.unified.authoring_handlers.handlers_intake import (
    _handle_intake_add,
    _handle_intake_dismiss,
    _handle_intake_list,
)
from foundry_mcp.tools.unified.authoring_handlers.handlers_metadata import (
    _handle_assumption_add,
    _handle_assumption_list,
    _handle_revision_add,
)
from foundry_mcp.tools.unified.authoring_handlers.handlers_phase import (
    _handle_phase_add,
    _handle_phase_add_bulk,
    _handle_phase_move,
    _handle_phase_remove,
    _handle_phase_template,
    _handle_phase_update_metadata,
)
from foundry_mcp.tools.unified.authoring_handlers.handlers_spec import (
    _handle_spec_create,
    _handle_spec_find_replace,
    _handle_spec_rollback,
    _handle_spec_template,
    _handle_spec_update_frontmatter,
)
from foundry_mcp.tools.unified.common import dispatch_with_standard_errors
from foundry_mcp.tools.unified.router import ActionDefinition, ActionRouter

logger = logging.getLogger(__name__)

_ACTION_DEFINITIONS = [
    ActionDefinition(
        name="spec-create",
        handler=_handle_spec_create,
        summary=_ACTION_SUMMARY["spec-create"],
        aliases=("spec_create",),
    ),
    ActionDefinition(
        name="spec-template",
        handler=_handle_spec_template,
        summary=_ACTION_SUMMARY["spec-template"],
        aliases=("spec_template",),
    ),
    ActionDefinition(
        name="spec-update-frontmatter",
        handler=_handle_spec_update_frontmatter,
        summary=_ACTION_SUMMARY["spec-update-frontmatter"],
        aliases=("spec_update_frontmatter",),
    ),
    ActionDefinition(
        name="spec-find-replace",
        handler=_handle_spec_find_replace,
        summary=_ACTION_SUMMARY["spec-find-replace"],
        aliases=("spec_find_replace",),
    ),
    ActionDefinition(
        name="spec-rollback",
        handler=_handle_spec_rollback,
        summary=_ACTION_SUMMARY["spec-rollback"],
        aliases=("spec_rollback",),
    ),
    ActionDefinition(
        name="phase-add", handler=_handle_phase_add, summary=_ACTION_SUMMARY["phase-add"], aliases=("phase_add",)
    ),
    ActionDefinition(
        name="phase-add-bulk",
        handler=_handle_phase_add_bulk,
        summary=_ACTION_SUMMARY["phase-add-bulk"],
        aliases=("phase_add_bulk",),
    ),
    ActionDefinition(
        name="phase-template",
        handler=_handle_phase_template,
        summary=_ACTION_SUMMARY["phase-template"],
        aliases=("phase_template",),
    ),
    ActionDefinition(
        name="phase-move", handler=_handle_phase_move, summary=_ACTION_SUMMARY["phase-move"], aliases=("phase_move",)
    ),
    ActionDefinition(
        name="phase-update-metadata",
        handler=_handle_phase_update_metadata,
        summary=_ACTION_SUMMARY["phase-update-metadata"],
        aliases=("phase_update_metadata",),
    ),
    ActionDefinition(
        name="phase-remove",
        handler=_handle_phase_remove,
        summary=_ACTION_SUMMARY["phase-remove"],
        aliases=("phase_remove",),
    ),
    ActionDefinition(
        name="assumption-add",
        handler=_handle_assumption_add,
        summary=_ACTION_SUMMARY["assumption-add"],
        aliases=("assumption_add",),
    ),
    ActionDefinition(
        name="assumption-list",
        handler=_handle_assumption_list,
        summary=_ACTION_SUMMARY["assumption-list"],
        aliases=("assumption_list",),
    ),
    ActionDefinition(
        name="revision-add",
        handler=_handle_revision_add,
        summary=_ACTION_SUMMARY["revision-add"],
        aliases=("revision_add",),
    ),
    ActionDefinition(
        name="intake-add", handler=_handle_intake_add, summary=_ACTION_SUMMARY["intake-add"], aliases=("intake_add",)
    ),
    ActionDefinition(
        name="intake-list",
        handler=_handle_intake_list,
        summary=_ACTION_SUMMARY["intake-list"],
        aliases=("intake_list",),
    ),
    ActionDefinition(
        name="intake-dismiss",
        handler=_handle_intake_dismiss,
        summary=_ACTION_SUMMARY["intake-dismiss"],
        aliases=("intake_dismiss",),
    ),
]

_AUTHORING_ROUTER = ActionRouter(tool_name="authoring", actions=_ACTION_DEFINITIONS)


def _dispatch_authoring_action(*, action: str, payload: Dict[str, Any], config: ServerConfig) -> dict:
    return dispatch_with_standard_errors(_AUTHORING_ROUTER, "authoring", action, config=config, **payload)


def register_unified_authoring_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated authoring tool."""

    @canonical_tool(
        mcp,
        canonical_name="authoring",
    )
    @mcp_tool(tool_name="authoring", emit_metrics=True, audit=True)
    def authoring(
        action: str,
        spec_id: Optional[str] = None,
        name: Optional[str] = None,
        template: Optional[str] = None,
        category: Optional[str] = None,
        mission: Optional[str] = None,
        template_action: Optional[str] = None,
        template_name: Optional[str] = None,
        key: Optional[str] = None,
        value: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        purpose: Optional[str] = None,
        position: Optional[int] = None,
        link_previous: bool = True,
        phase_id: Optional[str] = None,
        force: bool = False,
        text: Optional[str] = None,
        assumption_type: Optional[str] = None,
        author: Optional[str] = None,
        version: Optional[str] = None,
        changes: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        phase: Optional[Dict[str, Any]] = None,
        metadata_defaults: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        path: Optional[str] = None,
        # spec-find-replace parameters
        find: Optional[str] = None,
        replace: Optional[str] = None,
        scope: Optional[str] = None,
        use_regex: bool = False,
        case_sensitive: bool = True,
        # intake parameters
        priority: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: Optional[str] = None,
        requester: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Execute authoring workflows via the action router."""

        payload = {k: v for k, v in locals().items() if k not in ("action", "config")}
        return _dispatch_authoring_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified authoring tool")


__all__ = [
    "register_unified_authoring_tool",
]
