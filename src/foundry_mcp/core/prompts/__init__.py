"""
Prompt templates and builders for AI consultation workflows.

This package provides:
1. PromptTemplate dataclass for defining structured prompts
2. PromptRegistry for registering and retrieving prompts by ID
3. Workflow-specific prompt builders for different consultation use cases

Workflow Coverage:
    - plan_review: Review and critique SDD specifications
    - fidelity_review: Compare implementation against specifications

Example Usage:
    from foundry_mcp.core.prompts import (
        PromptTemplate,
        PromptRegistry,
        get_prompt_builder,
    )
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow

    # Register a custom prompt template
    template = PromptTemplate(
        id="custom_analysis",
        version="1.0",
        system_prompt="You are an expert code reviewer.",
        user_template="Analyze the following code: {code}",
        required_context=["code"],
    )
    registry = PromptRegistry()
    registry.register(template)

    # Or use workflow-specific builders
    builder = get_prompt_builder(ConsultationWorkflow.PLAN_REVIEW)
    prompt = builder.build("spec_review", {"spec_content": "..."})
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow

logger = logging.getLogger(__name__)


# =============================================================================
# PromptTemplate Dataclass
# =============================================================================


@dataclass(frozen=True)
class PromptTemplate:
    """
    Structured prompt template for AI consultations.

    Defines a reusable prompt with system and user components, required
    context variables, and metadata for tracking and versioning.

    Attributes:
        id: Unique identifier for the template (e.g., "analyze_module")
        version: Template version for tracking changes (e.g., "1.0", "2.1")
        system_prompt: System/instruction prompt (sent as system message)
        user_template: User message template with {variable} placeholders
        required_context: List of required context keys for rendering
        optional_context: List of optional context keys (have defaults)
        metadata: Additional template metadata (author, tags, etc.)

    Example:
        template = PromptTemplate(
            id="analyze_module",
            version="1.0",
            system_prompt="You are an expert Python developer.",
            user_template="Analyze this module:\\n\\n{code}\\n\\nFile: {file_path}",
            required_context=["code", "file_path"],
            metadata={"author": "system", "category": "documentation"},
        )
    """

    id: str
    version: str
    system_prompt: str
    user_template: str
    required_context: List[str] = field(default_factory=list)
    optional_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate template after initialization."""
        if not self.id:
            raise ValueError("Template id cannot be empty")
        if not self.version:
            raise ValueError("Template version cannot be empty")
        if not self.user_template:
            raise ValueError("Template user_template cannot be empty")

    def get_variables(self) -> Set[str]:
        """
        Extract all variable names from the user template.

        Returns:
            Set of variable names found in {variable} placeholders
        """
        # Find all {variable} patterns, excluding {{ and }}
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        return set(re.findall(pattern, self.user_template))

    def validate_context(self, context: Dict[str, Any]) -> List[str]:
        """
        Validate that context contains all required keys.

        Args:
            context: Context dict to validate

        Returns:
            List of missing required keys (empty if valid)
        """
        missing = []
        for key in self.required_context:
            if key not in context:
                missing.append(key)
        return missing

    def render(
        self,
        context: Dict[str, Any],
        *,
        strict: bool = True,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Render the user template with context substitution.

        Args:
            context: Context dict with variable values
            strict: If True, raise ValueError for missing required keys
            defaults: Default values for optional context keys

        Returns:
            Rendered template string

        Raises:
            ValueError: If strict=True and required keys are missing
        """
        # Validate required context
        missing = self.validate_context(context)
        if missing and strict:
            raise ValueError(
                f"Missing required context keys for template '{self.id}': {missing}"
            )

        # Build render context with defaults
        render_context = dict(defaults or {})
        render_context.update(context)

        # Provide empty string for missing optional keys
        for key in self.optional_context:
            if key not in render_context:
                render_context[key] = ""

        try:
            return self.user_template.format(**render_context)
        except KeyError as exc:
            raise ValueError(
                f"Missing context key for template '{self.id}': {exc}"
            ) from exc


# =============================================================================
# PromptRegistry
# =============================================================================


class PromptRegistry:
    """
    Registry for managing prompt templates.

    Provides registration, retrieval, and listing of prompt templates
    by ID, with optional namespace support for organizing templates.

    Attributes:
        templates: Dict mapping template IDs to PromptTemplate instances

    Example:
        registry = PromptRegistry()

        # Register a template
        template = PromptTemplate(
            id="analyze_code",
            version="1.0",
            system_prompt="You are a code analyst.",
            user_template="Analyze: {code}",
            required_context=["code"],
        )
        registry.register(template)

        # Retrieve and use
        t = registry.get("analyze_code")
        prompt = t.render({"code": "def foo(): pass"})
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._templates: Dict[str, PromptTemplate] = {}

    def register(
        self,
        template: PromptTemplate,
        *,
        replace: bool = False,
    ) -> None:
        """
        Register a prompt template.

        Args:
            template: The PromptTemplate to register
            replace: If True, replace existing template with same ID

        Raises:
            ValueError: If template ID already registered and replace=False
        """
        if template.id in self._templates and not replace:
            raise ValueError(
                f"Template '{template.id}' is already registered. "
                "Use replace=True to overwrite."
            )

        self._templates[template.id] = template
        logger.debug(
            "Registered prompt template '%s' (version %s)",
            template.id,
            template.version,
        )

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Retrieve a template by ID.

        Args:
            template_id: The template identifier

        Returns:
            PromptTemplate if found, None otherwise
        """
        return self._templates.get(template_id)

    def get_required(self, template_id: str) -> PromptTemplate:
        """
        Retrieve a template by ID, raising if not found.

        Args:
            template_id: The template identifier

        Returns:
            The PromptTemplate

        Raises:
            KeyError: If template not found
        """
        template = self._templates.get(template_id)
        if template is None:
            available = ", ".join(sorted(self._templates.keys())) or "(none)"
            raise KeyError(
                f"Template '{template_id}' not found. Available: {available}"
            )
        return template

    def list_templates(self) -> List[str]:
        """
        Return list of registered template IDs.

        Returns:
            Sorted list of template IDs
        """
        return sorted(self._templates.keys())

    def unregister(self, template_id: str) -> bool:
        """
        Remove a template from the registry.

        Args:
            template_id: The template identifier

        Returns:
            True if template was removed, False if not found
        """
        if template_id in self._templates:
            del self._templates[template_id]
            logger.debug("Unregistered prompt template '%s'", template_id)
            return True
        return False

    def clear(self) -> None:
        """Remove all templates from the registry."""
        self._templates.clear()
        logger.debug("Cleared all prompt templates from registry")

    def render(
        self,
        template_id: str,
        context: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> str:
        """
        Render a template by ID with context.

        Convenience method combining get_required() and render().

        Args:
            template_id: The template identifier
            context: Context dict for rendering
            strict: If True, raise for missing required keys

        Returns:
            Rendered template string

        Raises:
            KeyError: If template not found
            ValueError: If strict=True and required keys missing
        """
        template = self.get_required(template_id)
        return template.render(context, strict=strict)

    def __len__(self) -> int:
        """Return number of registered templates."""
        return len(self._templates)

    def __contains__(self, template_id: str) -> bool:
        """Check if template ID is registered."""
        return template_id in self._templates


# =============================================================================
# Global Registry Instance
# =============================================================================


# Default global registry for application-wide templates
_global_registry: Optional[PromptRegistry] = None


def get_global_registry() -> PromptRegistry:
    """
    Get the global prompt registry singleton.

    Returns:
        The global PromptRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (primarily for testing)."""
    global _global_registry
    _global_registry = None


# =============================================================================
# PromptBuilder ABC (Workflow-specific)
# =============================================================================


class PromptBuilder(ABC):
    """
    Abstract base class for workflow-specific prompt builders.

    Subclasses implement prompt templates for specific consultation workflows,
    providing a consistent interface for prompt generation. Each builder
    manages its own set of templates internally.

    Methods:
        build: Generate a prompt from a template ID and context
        list_prompts: Return available prompt IDs for this workflow
    """

    @abstractmethod
    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a prompt from template ID and context.

        Args:
            prompt_id: Identifier for the prompt template
            context: Structured context data to inject

        Returns:
            The rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        raise NotImplementedError

    @abstractmethod
    def list_prompts(self) -> List[str]:
        """
        Return list of available prompt IDs.

        Returns:
            List of prompt template IDs
        """
        raise NotImplementedError


# =============================================================================
# Workflow Builder Factory
# =============================================================================


def get_prompt_builder(workflow: "ConsultationWorkflow") -> PromptBuilder:
    """
    Get the prompt builder for a consultation workflow.

    Args:
        workflow: The consultation workflow type

    Returns:
        PromptBuilder instance for the workflow

    Raises:
        ValueError: If workflow is not supported
    """
    # Import here to avoid circular imports
    from foundry_mcp.core.ai_consultation import ConsultationWorkflow
    from foundry_mcp.core.prompts.plan_review import PlanReviewPromptBuilder
    from foundry_mcp.core.prompts.fidelity_review import FidelityReviewPromptBuilder
    from foundry_mcp.core.prompts.markdown_plan_review import MarkdownPlanReviewPromptBuilder

    builders: Dict[ConsultationWorkflow, type[PromptBuilder]] = {
        ConsultationWorkflow.PLAN_REVIEW: PlanReviewPromptBuilder,
        ConsultationWorkflow.FIDELITY_REVIEW: FidelityReviewPromptBuilder,
        ConsultationWorkflow.MARKDOWN_PLAN_REVIEW: MarkdownPlanReviewPromptBuilder,
    }

    builder_class = builders.get(workflow)
    if builder_class is None:
        raise ValueError(f"Unsupported workflow: {workflow}")

    return builder_class()


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Template dataclass
    "PromptTemplate",
    # Registry
    "PromptRegistry",
    "get_global_registry",
    "reset_global_registry",
    # Builder ABC
    "PromptBuilder",
    "get_prompt_builder",
]
