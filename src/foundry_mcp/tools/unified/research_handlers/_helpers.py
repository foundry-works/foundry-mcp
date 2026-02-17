"""Shared helpers for research handler modules."""

from __future__ import annotations

import logging
from typing import Optional

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.responses import ErrorCode
from foundry_mcp.tools.unified.common import (
    build_request_id,
    make_metric_name,
    make_validation_error_fn,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Action Summaries
# =============================================================================

_ACTION_SUMMARY = {
    "chat": "Single-model conversation with thread persistence",
    "consensus": "Multi-model parallel consultation with synthesis",
    "thinkdeep": "Hypothesis-driven systematic investigation",
    "ideate": "Creative brainstorming with idea clustering",
    "deep-research": "Multi-phase iterative deep research with query decomposition",
    "deep-research-status": "Get status of deep research session",
    "deep-research-report": "Get final report from deep research",
    "deep-research-list": "List deep research sessions",
    "deep-research-delete": "Delete a deep research session",
    "thread-list": "List conversation threads",
    "thread-get": "Get full thread details including messages",
    "thread-delete": "Delete a conversation thread",
    # Spec-integrated research actions
    "node-execute": "Execute research workflow linked to spec node",
    "node-record": "Record research findings to spec node",
    "node-status": "Get research node status and linked session info",
    "node-findings": "Retrieve recorded findings from spec node",
    # Tavily extract action
    "extract": "Extract content from URLs using Tavily Extract API",
}


# =============================================================================
# Module State
# =============================================================================

_config: Optional[ServerConfig] = None
_memory: Optional[ResearchMemory] = None


def _get_memory() -> ResearchMemory:
    """Get or create the research memory instance."""
    global _memory, _config
    if _memory is None:
        if _config is not None:
            _memory = ResearchMemory(
                base_path=_config.get_research_dir(),
                ttl_hours=_config.research.ttl_hours,
            )
        else:
            _memory = ResearchMemory()
    return _memory


def _get_config() -> ServerConfig:
    """Get the server config, raising if not initialized."""
    global _config
    if _config is None:
        # Create default config if not set
        _config = ServerConfig()
    return _config


# =============================================================================
# Helpers
# =============================================================================


def _request_id() -> str:
    return build_request_id("research")


def _metric(action: str) -> str:
    return make_metric_name("unified_tools.research", action)


_validation_error = make_validation_error_fn("research")
