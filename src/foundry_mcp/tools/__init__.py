"""MCP tools for foundry-mcp."""

from foundry_mcp.tools.queries import register_query_tools
from foundry_mcp.tools.tasks import register_task_tools
from foundry_mcp.tools.validation import register_validation_tools
from foundry_mcp.tools.journal import register_journal_tools
from foundry_mcp.tools.environment import register_environment_tools
from foundry_mcp.tools.spec_helpers import register_spec_helper_tools
from foundry_mcp.tools.authoring import register_authoring_tools
from foundry_mcp.tools.mutations import register_mutation_tools
from foundry_mcp.tools.documentation import register_documentation_tools
from foundry_mcp.tools.providers import register_provider_tools
from foundry_mcp.tools.plan_review import register_plan_review_tools

__all__ = [
    "register_query_tools",
    "register_task_tools",
    "register_validation_tools",
    "register_journal_tools",
    "register_environment_tools",
    "register_spec_helper_tools",
    "register_authoring_tools",
    "register_mutation_tools",
    "register_documentation_tools",
    "register_provider_tools",
    "register_plan_review_tools",
]
