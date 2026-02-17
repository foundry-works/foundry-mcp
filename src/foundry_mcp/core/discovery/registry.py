"""
Tool registry for MCP tool metadata management.

Provides the ToolRegistry class and global registry accessor for
tool registration, discovery, and filtering capabilities.
"""

from typing import Any, Dict, List, Optional

from .types import ToolMetadata


class ToolRegistry:
    """
    Central registry for MCP tool metadata.

    Provides tool registration, discovery, and filtering capabilities.
    Used by MCP servers to expose tool information to clients.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(ToolMetadata(
        ...     name="get_user",
        ...     description="Get user by ID",
        ...     category="users",
        ... ))
        >>> tools = registry.list_tools(category="users")
    """

    def __init__(self) -> None:
        """Initialize empty tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: ToolMetadata) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool metadata to register

        Raises:
            ValueError: If tool with same name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool

        # Update category index
        if tool.category not in self._categories:
            self._categories[tool.category] = []
        self._categories[tool.category].append(tool.name)

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Tool name to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False

        tool = self._tools.pop(name)
        if tool.category in self._categories:
            self._categories[tool.category].remove(name)
            if not self._categories[tool.category]:
                del self._categories[tool.category]

        return True

    def get(self, name: str) -> Optional[ToolMetadata]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            ToolMetadata if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(
        self,
        *,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List available tools with filtering.

        Args:
            category: Filter by category
            tag: Filter by tag
            include_deprecated: Include deprecated tools (default: False)

        Returns:
            List of tool summaries
        """
        tools = list(self._tools.values())

        # Apply filters
        if category:
            tools = [t for t in tools if t.category == category]

        if tag:
            tools = [t for t in tools if tag in t.tags]

        if not include_deprecated:
            tools = [t for t in tools if not t.deprecated]

        return [t.to_summary() for t in tools]

    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed schema for a specific tool.

        Args:
            name: Tool name

        Returns:
            Detailed tool info including schema, or None if not found
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return tool.to_detailed()

    def list_categories(self) -> List[Dict[str, Any]]:
        """
        List tool categories with descriptions.

        Returns:
            List of categories with tool counts
        """
        result = []
        for category, tool_names in sorted(self._categories.items()):
            # Filter out deprecated tools from count
            active_count = sum(
                1
                for name in tool_names
                if name in self._tools and not self._tools[name].deprecated
            )

            result.append(
                {
                    "name": category,
                    "tool_count": active_count,
                    "tools": tool_names,
                }
            )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with total counts and breakdown
        """
        total = len(self._tools)
        deprecated = sum(1 for t in self._tools.values() if t.deprecated)

        return {
            "total_tools": total,
            "active_tools": total - deprecated,
            "deprecated_tools": deprecated,
            "categories": len(self._categories),
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
