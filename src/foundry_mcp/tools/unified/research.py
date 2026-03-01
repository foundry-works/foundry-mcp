"""Unified research router — delegates to research_handlers/ package.

This module is a backward-compatible shim. All handler logic now lives in
``foundry_mcp.tools.unified.research_handlers``.
"""

from __future__ import annotations

from foundry_mcp.core.research.workflows import (  # noqa: F401
    ChatWorkflow,
    ConsensusWorkflow,
    DeepResearchWorkflow,
    IdeateWorkflow,
    ThinkDeepWorkflow,
)
from foundry_mcp.tools.unified.research_handlers import (  # noqa: F401
    _RESEARCH_ROUTER,
    _dispatch_research_action,
    register_unified_research_tool,
)
from foundry_mcp.tools.unified.research_handlers._helpers import (  # noqa: F401
    _get_config,
    _get_memory,
    _validation_error,
)
from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (  # noqa: F401
    _handle_deep_research,
    _handle_deep_research_delete,
    _handle_deep_research_export,
    _handle_deep_research_list,
    _handle_deep_research_report,
    _handle_deep_research_status,
)
from foundry_mcp.tools.unified.research_handlers.handlers_extract import (  # noqa: F401
    _handle_extract,
)
from foundry_mcp.tools.unified.research_handlers.handlers_spec_nodes import (  # noqa: F401
    _handle_node_execute,
    _handle_node_findings,
    _handle_node_record,
    _handle_node_status,
    _load_research_node,
)
from foundry_mcp.tools.unified.research_handlers.handlers_threads import (  # noqa: F401
    _handle_thread_delete,
    _handle_thread_get,
    _handle_thread_list,
)
from foundry_mcp.tools.unified.research_handlers.handlers_workflows import (  # noqa: F401
    _handle_chat,
    _handle_consensus,
    _handle_ideate,
    _handle_thinkdeep,
)

__all__ = [
    "register_unified_research_tool",
]
