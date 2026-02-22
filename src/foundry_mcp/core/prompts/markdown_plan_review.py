"""
Markdown plan review prompts for AI consultation workflows.

This module provides the prompt template for reviewing markdown plans
before converting them to formal JSON specifications. Supports iterative
review cycles to refine plans with AI feedback.

Templates:
    - MARKDOWN_PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review

Each template expects plan_content, plan_name, and optionally plan_path context.
"""

from __future__ import annotations

from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptRegistry, PromptTemplate

# =============================================================================
# Response Schema for MARKDOWN_PLAN_REVIEW Templates
# =============================================================================

_RESPONSE_SCHEMA = """
# Review Summary

## Critical Blockers
Issues that MUST be fixed before this becomes a spec.

- **[Category]** <Issue title>
  - **Description:** <What's wrong>
  - **Impact:** <Consequences if not fixed>
  - **Fix:** <Specific actionable recommendation>

## Major Suggestions
Significant improvements to strengthen the plan.

- **[Category]** <Issue title>
  - **Description:** <What's wrong>
  - **Impact:** <Consequences if not addressed>
  - **Fix:** <Specific actionable recommendation>

## Minor Suggestions
Smaller refinements.

- **[Category]** <Issue title>
  - **Description:** <What could be better>
  - **Fix:** <Specific actionable recommendation>

## Questions
Clarifications needed before proceeding.

- **[Category]** <Question>
  - **Context:** <Why this matters>
  - **Needed:** <What information would help>

## Praise
What the plan does well.

- **[Category]** <What works well>
  - **Why:** <What makes this effective>

---

**Important**:
- Use category tags: [Completeness], [Architecture], [Sequencing], [Over-engineering], [Risk], [Clarity]
- Include all sections even if empty (write "None identified" for empty sections)
- Be specific and actionable in all feedback
- For clarity issues, use Questions section rather than creating a separate category
- Do NOT generate feedback about ownership, responsibility, or team assignments (e.g., "who verifies", "who owns", "who is responsible")
"""


# =============================================================================
# System Prompts
# =============================================================================

_MARKDOWN_PLAN_REVIEW_SYSTEM_PROMPT = """You are an expert software architect conducting a technical review.
Your task is to provide constructive, actionable feedback on implementation plans
written in markdown format, BEFORE they become formal specifications.

Guidelines:
- Be thorough and specific - examine all aspects of the proposed approach
- Identify both strengths and opportunities for improvement
- Ask clarifying questions for ambiguities
- Propose alternatives when better approaches exist
- Focus on impact and prioritize feedback by potential consequences
- Be collaborative, not adversarial
- Remember: this is an early-stage plan, not a final spec
- Do NOT focus on ownership, responsibility, or team assignment concerns
- Avoid feedback like "who owns", "who verifies", "who is responsible for"
- Focus on technical requirements and verification steps themselves, not who performs them"""


# =============================================================================
# MARKDOWN_PLAN_REVIEW_FULL_V1
# =============================================================================

MARKDOWN_PLAN_REVIEW_FULL_V1 = PromptTemplate(
    id="MARKDOWN_PLAN_REVIEW_FULL_V1",
    version="1.0",
    system_prompt=_MARKDOWN_PLAN_REVIEW_SYSTEM_PROMPT,
    user_template="""You are conducting a comprehensive review of a markdown implementation plan.

**Plan Name**: {plan_name}
**Review Type**: Full (comprehensive analysis)

**Your role**: You are a collaborative senior peer helping refine the plan before it becomes a formal specification.

**Critical: Provide Constructive Feedback**

Effective reviews combine critical analysis with actionable guidance.

**Your evaluation guidelines**:
1. **Be thorough and specific** - Examine all aspects of the proposed approach
2. **Identify both strengths and opportunities** - Note what works well and what could improve
3. **Ask clarifying questions** - Highlight ambiguities that need resolution
4. **Propose alternatives** - Show better approaches when they exist
5. **Be actionable** - Provide specific, implementable recommendations
6. **Focus on impact** - Prioritize feedback by potential consequences

**Effective feedback patterns**:
- "Consider whether this approach handles X, Y, Z edge cases"
- "These estimates may be optimistic because..."
- "Strong design choice here because..."
- "Clarification needed: how does this handle scenario X?"

**Evaluate across 6 dimensions:**

1. **Completeness** - Are all required sections present?
   - Mission statement (single sentence)?
   - Objectives listed as discrete items?
   - Success criteria with measurable checkboxes?
   - Assumptions listed?
   - Constraints listed?
   - Risks with likelihood/impact/mitigation columns?
   - Open questions (if any unresolved items exist)?
   - Per-task: category tag, complexity tag, file path (or "N/A"), acceptance criteria, dependencies?
2. **Architecture** - Sound approach? Coupling concerns? Missing abstractions? Is the design testable?
3. **Sequencing** - Phases ordered correctly? Dependencies identified?
4. **Over-engineering** - Unnecessary abstractions? Premature generalization? Features beyond what was asked?
5. **Risk** - What could go wrong? Mitigation strategies?
6. **Clarity** - Unambiguous? Would another developer understand?

**MARKDOWN PLAN TO REVIEW:**

{plan_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Remember**: Your goal is to **help create a robust implementation plan**. Be specific, actionable, and balanced in your feedback. Identify both critical blockers and positive aspects of the plan.""",
    required_context=["plan_content", "plan_name"],
    optional_context=["response_schema", "plan_path"],
    metadata={
        "author": "foundry-mcp",
        "category": "markdown_plan_review",
        "workflow": "MARKDOWN_PLAN_REVIEW",
        "review_type": "full",
        "dimensions": [
            "Completeness",
            "Architecture",
            "Sequencing",
            "Over-engineering",
            "Risk",
            "Clarity",
        ],
        "description": "Comprehensive 6-dimension markdown plan review",
    },
)


# =============================================================================
# Template Registry
# =============================================================================


MARKDOWN_PLAN_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "MARKDOWN_PLAN_REVIEW_FULL_V1": MARKDOWN_PLAN_REVIEW_FULL_V1,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class MarkdownPlanReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for markdown plan review workflows.

    Provides access to MARKDOWN_PLAN_REVIEW_* templates for reviewing
    markdown plans before they become formal JSON specifications.

    Templates:
        - MARKDOWN_PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review

    Example:
        builder = MarkdownPlanReviewPromptBuilder()

        prompt = builder.build("MARKDOWN_PLAN_REVIEW_FULL_V1", {
            "plan_content": "...",
            "plan_name": "my-feature",
        })
    """

    def __init__(self) -> None:
        """Initialize the builder with all templates."""
        self._registry = PromptRegistry()
        for template in MARKDOWN_PLAN_REVIEW_TEMPLATES.values():
            self._registry.register(template)

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a markdown plan review prompt.

        Args:
            prompt_id: Template ID (MARKDOWN_PLAN_REVIEW_*)
            context: Context dict with required variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id not found or required context missing
        """
        if prompt_id not in MARKDOWN_PLAN_REVIEW_TEMPLATES:
            available = sorted(MARKDOWN_PLAN_REVIEW_TEMPLATES.keys())
            raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {available}")

        render_context = dict(context)
        # Add default response schema if not provided
        if "response_schema" not in render_context:
            render_context["response_schema"] = _RESPONSE_SCHEMA

        return self._registry.render(prompt_id, render_context)

    def list_prompts(self) -> List[str]:
        """
        Return all available prompt IDs.

        Returns:
            Sorted list of all prompt IDs
        """
        return sorted(MARKDOWN_PLAN_REVIEW_TEMPLATES.keys())

    def get_template(self, prompt_id: str) -> PromptTemplate:
        """
        Get a template by ID for inspection.

        Args:
            prompt_id: Template identifier

        Returns:
            The PromptTemplate

        Raises:
            KeyError: If not found
        """
        if prompt_id not in MARKDOWN_PLAN_REVIEW_TEMPLATES:
            available = sorted(MARKDOWN_PLAN_REVIEW_TEMPLATES.keys())
            raise KeyError(f"Template '{prompt_id}' not found. Available: {available}")
        return self._registry.get_required(prompt_id)


# =============================================================================
# Helper Functions
# =============================================================================


def get_response_schema() -> str:
    """
    Get the standard response schema for markdown plan reviews.

    Returns:
        Response schema markdown string
    """
    return _RESPONSE_SCHEMA


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Templates
    "MARKDOWN_PLAN_REVIEW_FULL_V1",
    "MARKDOWN_PLAN_REVIEW_TEMPLATES",
    # Builder
    "MarkdownPlanReviewPromptBuilder",
    # Helpers
    "get_response_schema",
]
