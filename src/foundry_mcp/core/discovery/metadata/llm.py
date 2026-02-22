"""
LLM-powered tool discovery metadata.

Defines LLM_TOOL_METADATA, LLM_FEATURE_FLAGS,
and helper functions for LLM tool registration and capability negotiation.
"""

from typing import Any, Dict, Optional

from ..flags import FeatureFlagDescriptor
from ..registry import ToolRegistry, get_tool_registry
from ..types import ParameterMetadata, ParameterType, ToolMetadata

# Pre-defined metadata for LLM-powered tools
LLM_TOOL_METADATA: Dict[str, ToolMetadata] = {
    "spec-review": ToolMetadata(
        name="spec-review",
        description="Run an LLM-powered review session on a specification. "
        "Performs intelligent spec analysis and generates improvement suggestions. "
        "Supports multiple review types and external AI tool integration.",
        parameters=[
            ParameterMetadata(
                name="spec_id",
                type=ParameterType.STRING,
                description="Specification ID to review",
                required=True,
                examples=["feature-auth-2025-01-15-001", "bugfix-cache-2025-02-01-001"],
            ),
            ParameterMetadata(
                name="review_type",
                type=ParameterType.STRING,
                description="Type of review to perform (defaults to config value, typically 'full')",
                required=False,
                default="full",
                constraints={"enum": ["quick", "full", "security", "feasibility"]},
                examples=["full", "quick", "security"],
            ),
            ParameterMetadata(
                name="tools",
                type=ParameterType.STRING,
                description="Comma-separated list of review tools to use",
                required=False,
                examples=["cursor-agent", "gemini,codex", "cursor-agent,gemini,codex"],
            ),
            ParameterMetadata(
                name="model",
                type=ParameterType.STRING,
                description="LLM model to use for review (default: from config)",
                required=False,
                examples=["gpt-4o", "claude-3-sonnet"],
            ),
            ParameterMetadata(
                name="path",
                type=ParameterType.STRING,
                description="Project root path (default: current directory)",
                required=False,
            ),
            ParameterMetadata(
                name="dry_run",
                type=ParameterType.BOOLEAN,
                description="Show what would be reviewed without executing",
                required=False,
                default=False,
            ),
        ],
        category="llm",
        version="1.0.0",
        tags=["review", "llm", "ai", "analysis", "quality"],
        related_tools=["review-list-tools", "review-list-plan-tools", "spec-review-fidelity"],
        examples=[
            {
                "description": "Full review of a specification",
                "input": {"spec_id": "feature-auth-001", "review_type": "full"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "feature-auth-001",
                        "review_type": "full",
                        "findings": [],
                        "suggestions": ["Consider adding error handling"],
                    },
                },
            },
            {
                "description": "Security review with multiple tools",
                "input": {
                    "spec_id": "payment-flow-001",
                    "review_type": "security",
                    "tools": "cursor-agent,gemini",
                },
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "payment-flow-001",
                        "review_type": "security",
                        "findings": [{"severity": "medium", "issue": "Missing rate limiting"}],
                    },
                },
            },
        ],
    ),
    "review-list-tools": ToolMetadata(
        name="review-list-tools",
        description="List available review tools and pipelines. "
        "Returns the set of external AI tools that can be used for spec reviews "
        "along with their availability status.",
        parameters=[],
        category="llm",
        version="1.0.0",
        tags=["review", "discovery", "tools", "configuration"],
        related_tools=["spec-review", "review-list-plan-tools"],
        examples=[
            {
                "description": "List all available review tools",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "tools": [
                            {"name": "cursor-agent", "available": True, "version": "1.0.0"},
                            {"name": "gemini", "available": True, "version": "2.0"},
                            {"name": "codex", "available": False, "version": None},
                        ],
                        "llm_status": {"configured": True, "provider": "openai"},
                    },
                },
            }
        ],
    ),
    "review-list-plan-tools": ToolMetadata(
        name="review-list-plan-tools",
        description="Enumerate review toolchains available for plan analysis. "
        "Returns tools specifically designed for reviewing SDD plans "
        "including their capabilities and recommended usage.",
        parameters=[],
        category="llm",
        version="1.0.0",
        tags=["review", "planning", "tools", "recommendations"],
        related_tools=["spec-review", "review-list-tools"],
        examples=[
            {
                "description": "List plan review toolchains",
                "input": {},
                "output": {
                    "success": True,
                    "data": {
                        "plan_tools": [
                            {
                                "name": "quick-review",
                                "llm_required": False,
                                "status": "available",
                            },
                            {
                                "name": "full-review",
                                "llm_required": True,
                                "status": "available",
                            },
                        ],
                        "recommendations": ["Use 'full-review' for comprehensive analysis"],
                    },
                },
            }
        ],
    ),
    "spec-review-fidelity": ToolMetadata(
        name="spec-review-fidelity",
        description="Compare implementation against specification and identify deviations. "
        "Performs a fidelity review to verify that code implementation matches "
        "the specification requirements. Uses AI consultation for comprehensive analysis.",
        parameters=[
            ParameterMetadata(
                name="spec_id",
                type=ParameterType.STRING,
                description="Specification ID to review against",
                required=True,
                examples=["feature-auth-001", "api-v2-migration-001"],
            ),
            ParameterMetadata(
                name="task_id",
                type=ParameterType.STRING,
                description="Review specific task implementation (mutually exclusive with phase_id)",
                required=False,
            ),
            ParameterMetadata(
                name="phase_id",
                type=ParameterType.STRING,
                description="Review entire phase implementation (mutually exclusive with task_id)",
                required=False,
            ),
            ParameterMetadata(
                name="files",
                type=ParameterType.ARRAY,
                description="Review specific file(s) only",
                required=False,
                examples=[["src/auth.py"], ["src/api/users.py", "src/api/auth.py"]],
            ),
            ParameterMetadata(
                name="use_ai",
                type=ParameterType.BOOLEAN,
                description="Enable AI consultation for analysis",
                required=False,
                default=True,
            ),
            ParameterMetadata(
                name="ai_tools",
                type=ParameterType.ARRAY,
                description="Specific AI tools to consult",
                required=False,
                examples=[["cursor-agent", "gemini"]],
            ),
            ParameterMetadata(
                name="consensus_threshold",
                type=ParameterType.INTEGER,
                description="Minimum models that must agree",
                required=False,
                default=2,
                constraints={"minimum": 1, "maximum": 5},
            ),
            ParameterMetadata(
                name="incremental",
                type=ParameterType.BOOLEAN,
                description="Only review changed files since last run",
                required=False,
                default=False,
            ),
        ],
        category="llm",
        version="1.0.0",
        tags=["fidelity", "review", "verification", "compliance", "llm"],
        related_tools=["spec-review"],
        rate_limit="20/hour",
        examples=[
            {
                "description": "Fidelity review for a phase",
                "input": {"spec_id": "feature-auth-001", "phase_id": "phase-1"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "feature-auth-001",
                        "scope": "phase",
                        "verdict": "pass",
                        "deviations": [],
                        "consensus": {"models_consulted": 3, "agreement": "unanimous"},
                    },
                },
            },
            {
                "description": "Fidelity review with deviations found",
                "input": {"spec_id": "api-v2-001", "task_id": "task-2-1"},
                "output": {
                    "success": True,
                    "data": {
                        "spec_id": "api-v2-001",
                        "scope": "task",
                        "verdict": "partial",
                        "deviations": [
                            {
                                "task_id": "task-2-1",
                                "type": "missing_implementation",
                                "severity": "high",
                            }
                        ],
                    },
                },
            },
        ],
    ),
}


# LLM feature flags for capability negotiation
LLM_FEATURE_FLAGS: Dict[str, FeatureFlagDescriptor] = {
    "llm_tools": FeatureFlagDescriptor(
        name="llm_tools",
        description="LLM-powered review and documentation tools",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=[],
    ),
    "llm_multi_provider": FeatureFlagDescriptor(
        name="llm_multi_provider",
        description="Multi-provider AI tool support (cursor-agent, gemini, codex)",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools"],
    ),
    "llm_fidelity_review": FeatureFlagDescriptor(
        name="llm_fidelity_review",
        description="AI-powered fidelity review with consensus mechanism",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools", "llm_multi_provider"],
    ),
    "llm_data_only_fallback": FeatureFlagDescriptor(
        name="llm_data_only_fallback",
        description="Graceful fallback to data-only responses when LLM unavailable",
        state="stable",
        default_enabled=True,
        percentage_rollout=100,
        dependencies=["llm_tools"],
    ),
}


def register_llm_tools(registry: Optional[ToolRegistry] = None) -> ToolRegistry:
    """
    Register all LLM-powered tools in the registry.

    Args:
        registry: Optional registry to use. If None, uses global registry.

    Returns:
        The registry with LLM tools registered.
    """
    if registry is None:
        registry = get_tool_registry()

    for tool_metadata in LLM_TOOL_METADATA.values():
        try:
            registry.register(tool_metadata)
        except ValueError:
            # Tool already registered, skip
            pass

    return registry


def get_llm_capabilities() -> Dict[str, Any]:
    """
    Get LLM-related capabilities for capability negotiation.

    Returns:
        Dict with LLM tool availability, providers, and feature flags.
    """
    return {
        "llm_tools": {
            "supported": True,
            "tools": list(LLM_TOOL_METADATA.keys()),
            "description": "LLM-powered review, documentation, and fidelity tools",
        },
        "multi_provider": {
            "supported": True,
            "providers": ["cursor-agent", "gemini", "codex"],
            "description": "Multi-provider AI tool integration",
        },
        "data_only_fallback": {
            "supported": True,
            "description": "Graceful degradation when LLM unavailable",
        },
        "feature_flags": {name: flag.to_dict() for name, flag in LLM_FEATURE_FLAGS.items()},
    }


def is_llm_tool(tool_name: str) -> bool:
    """
    Check if a tool name is an LLM-powered tool.

    Args:
        tool_name: Name of the tool to check

    Returns:
        True if tool is an LLM-powered tool
    """
    return tool_name in LLM_TOOL_METADATA


def get_llm_tool_metadata(tool_name: str) -> Optional[ToolMetadata]:
    """
    Get metadata for a specific LLM tool.

    Args:
        tool_name: Name of the LLM tool

    Returns:
        ToolMetadata if found, None otherwise
    """
    return LLM_TOOL_METADATA.get(tool_name)
