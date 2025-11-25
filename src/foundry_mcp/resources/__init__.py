"""
MCP resources for foundry-mcp.

Provides resource handlers for spec and template access.
"""

from foundry_mcp.resources.specs import register_spec_resources

__all__ = ["register_spec_resources"]
