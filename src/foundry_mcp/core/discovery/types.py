"""
Type definitions for tool metadata and discovery.

Contains the core data classes used throughout the discovery module:
ParameterType, ParameterMetadata, and ToolMetadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Schema version for discovery responses
SCHEMA_VERSION = "1.0.0"


class ParameterType(str, Enum):
    """Types for tool parameters in JSON Schema."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ParameterMetadata:
    """
    Metadata for a tool parameter.

    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, boolean, etc.)
        description: Human-readable description
        required: Whether parameter is required
        default: Default value if not provided
        constraints: JSON Schema constraints (pattern, min, max, enum, etc.)
        examples: Example values for documentation
    """

    name: str
    type: ParameterType
    description: str
    required: bool = False
    default: Any = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)


@dataclass
class ToolMetadata:
    """
    Comprehensive metadata for an MCP tool.

    Used for tool registration, discovery, and documentation generation.
    Supports categorization, versioning, deprecation, and rate limiting.

    Attributes:
        name: Unique tool identifier
        description: Human-readable description (supports markdown)
        parameters: List of parameter metadata
        category: Tool category for grouping (e.g., "files", "users", "code")
        version: Semantic version of the tool interface
        deprecated: Whether tool is deprecated
        deprecation_message: Message explaining deprecation and migration path
        rate_limit: Rate limit description (e.g., "100/minute")
        examples: List of usage examples with input/output
        related_tools: Names of related tools
        tags: Semantic tags for filtering
    """

    name: str
    description: str
    parameters: List[ParameterMetadata] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    deprecated: bool = False
    deprecation_message: Optional[str] = None
    rate_limit: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Generate JSON Schema from tool metadata.

        Returns:
            JSON Schema dict for tool input validation
        """
        type_mapping = {
            ParameterType.STRING: "string",
            ParameterType.INTEGER: "integer",
            ParameterType.NUMBER: "number",
            ParameterType.BOOLEAN: "boolean",
            ParameterType.ARRAY: "array",
            ParameterType.OBJECT: "object",
        }

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {
                "type": type_mapping[param.type],
                "description": param.description,
            }

            if param.default is not None:
                prop["default"] = param.default

            if param.examples:
                prop["examples"] = param.examples

            # Add constraints
            prop.update(param.constraints)

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "description": self.description,
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def to_summary(self) -> Dict[str, Any]:
        """
        Generate summary for tool listing.

        Returns:
            Compact dict for list_tools responses
        """
        desc = self.description
        if len(desc) > 200:
            desc = desc[:197] + "..."

        return {
            "name": self.name,
            "description": desc,
            "category": self.category,
            "version": self.version,
            "deprecated": self.deprecated,
            "tags": self.tags,
        }

    def to_detailed(self) -> Dict[str, Any]:
        """
        Generate detailed info for get_tool_schema responses.

        Returns:
            Full dict including schema, examples, and related tools
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "schema": self.to_json_schema(),
            "examples": self.examples,
            "related_tools": self.related_tools,
            "tags": self.tags,
            "deprecated": self.deprecated,
        }

        if self.rate_limit:
            result["rate_limit"] = self.rate_limit

        if self.deprecation_message:
            result["deprecation_message"] = self.deprecation_message

        return result
