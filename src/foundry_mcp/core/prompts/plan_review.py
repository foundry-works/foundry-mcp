"""
Plan review prompts for AI consultation workflows.

This module provides prompt templates for plan review:

1. **PLAN_REVIEW_*_V1 Templates** (PromptTemplate instances):
   - PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
   - SYNTHESIS_PROMPT_V1: Multi-model response synthesis

Each PLAN_REVIEW_* template expects spec_content, spec_id, and title context.
"""

from __future__ import annotations

from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptRegistry, PromptTemplate

# =============================================================================
# Response Schema for PLAN_REVIEW Templates
# =============================================================================

_RESPONSE_SCHEMA = """
# Review Summary

## Critical Blockers
Issues that must be fixed before implementation can begin.

- **[Category]** <Issue title>
  - **Description:** <What's wrong>
  - **Impact:** <Consequences if not fixed>
  - **Fix:** <Specific actionable recommendation>

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design.

- **[Category]** <Issue title>
  - **Description:** <What's wrong>
  - **Impact:** <Consequences if not addressed>
  - **Fix:** <Specific actionable recommendation>

## Minor Suggestions
Smaller improvements and optimizations.

- **[Category]** <Issue title>
  - **Description:** <What could be better>
  - **Fix:** <Specific actionable recommendation>

## Questions
Clarifications needed or ambiguities to resolve.

- **[Category]** <Question>
  - **Context:** <Why this matters>
  - **Needed:** <What information would help>

## Praise
What the spec does well.

- **[Category]** <What works well>
  - **Why:** <What makes this effective>

---

**Important**:
- Use category tags: [Completeness], [Architecture], [Data Model], [Interface Design], [Security], [Verification]
- Include all sections even if empty (write "None identified" for empty sections)
- Be specific and actionable in all feedback
- For clarity issues, use Questions section rather than creating a separate category
- Attribution: In multi-model reviews, prefix items with "Flagged by [model-name]:" when applicable
- Do NOT generate feedback about ownership, responsibility, or team assignments (e.g., "who verifies", "who owns", "who is responsible")
"""


# =============================================================================
# System Prompts
# =============================================================================

_PLAN_REVIEW_SYSTEM_PROMPT = """You are an expert software architect conducting a technical review.
Your task is to provide constructive, actionable feedback on software specifications.

Guidelines:
- Be thorough and specific - examine all aspects of the design
- Identify both strengths and opportunities for improvement
- Ask clarifying questions for ambiguities
- Propose alternatives when better approaches exist
- Focus on impact and prioritize feedback by potential consequences
- Be collaborative, not adversarial
- Do NOT focus on ownership, responsibility, or team assignment concerns
- Avoid feedback like "who owns", "who verifies", "who is responsible for"
- Focus on technical requirements and verification steps themselves, not who performs them"""


# =============================================================================
# PLAN_REVIEW_FULL_V1
# =============================================================================

PLAN_REVIEW_FULL_V1 = PromptTemplate(
    id="PLAN_REVIEW_FULL_V1",
    version="1.0",
    system_prompt=_PLAN_REVIEW_SYSTEM_PROMPT,
    user_template="""You are conducting a comprehensive technical review of a software specification.

**Spec**: {spec_id}
**Title**: {title}
**Review Type**: Full (comprehensive analysis)

**Your role**: You are a collaborative senior peer helping refine the design and identify opportunities for improvement.

**Critical: Provide Constructive Feedback**

Effective reviews combine critical analysis with actionable guidance.

**Your evaluation guidelines**:
1. **Be thorough and specific** - Examine all aspects of the design
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

**Evaluate across 6 technical dimensions:**

1. **Completeness** - Identify missing sections, undefined requirements, ambiguous tasks
2. **Architecture** - Find design issues, coupling concerns, missing abstractions, scalability considerations
3. **Data Model** - Evaluate data structures, relationships, consistency, migration strategies
4. **Interface Design** - Review API contracts, component boundaries, integration patterns
5. **Security** - Identify authentication, authorization, data protection, and vulnerability concerns
6. **Verification** - Find testing gaps, missing verification steps, coverage opportunities

**SPECIFICATION TO REVIEW:**

{spec_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Remember**: Your goal is to **help create robust, well-designed software**. Be specific, actionable, and balanced in your feedback. Identify both critical blockers and positive aspects of the design.""",
    required_context=["spec_content", "spec_id", "title"],
    optional_context=["response_schema"],
    metadata={
        "author": "foundry-mcp",
        "category": "plan_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "full",
        "dimensions": [
            "Completeness",
            "Architecture",
            "Data Model",
            "Interface Design",
            "Security",
            "Verification",
        ],
        "description": "Comprehensive 6-dimension specification review",
    },
)


# =============================================================================
# SYNTHESIS_PROMPT_V1
# =============================================================================

SYNTHESIS_PROMPT_V1 = PromptTemplate(
    id="SYNTHESIS_PROMPT_V1",
    version="1.1",
    system_prompt="""You are an expert at synthesizing multiple technical reviews.
Your task is to consolidate diverse perspectives into actionable consensus.

Guidelines:
- Attribute findings to specific models with accurate severity
- Identify areas of agreement and disagreement
- Prioritize by consensus strength
- Preserve unique insights from each model
- Create actionable, consolidated recommendations
- NEVER escalate severity beyond what individual reviewers assigned""",
    user_template="""You are synthesizing {num_models} independent AI reviews of a specification.

**Specification**: {title} (`{spec_id}`)

**Your Task**: Read all reviews below and create a comprehensive synthesis.

**Severity Fidelity Rules** (you MUST follow these):
1. **No severity escalation**: Place each issue in the section matching the HIGHEST severity any individual reviewer assigned it. Never place an issue in a section ABOVE the maximum severity any reviewer gave it. If all reviewers called something "Minor", it MUST stay in Minor Suggestions.
2. **Accurate attribution**: When writing "flagged by:", include the severity each reviewer actually used if it differs from the section (e.g., "flagged by: codex (as minor), gemini (as major)").
3. **Cross-cutting observations**: If you notice that multiple lower-severity items from different reviewers collectively suggest a more significant concern, place this observation in the **Escalation Candidates** section with your reasoning. Do NOT move the individual items to a higher severity section.
4. **Empty sections**: If no reviewer flagged any issues at a given severity level, that section MUST say "None identified."

**Required Output** (Markdown format):

```markdown
# Synthesis

## Overall Assessment
- **Consensus Level**: Strong/Moderate/Weak/Conflicted (based on agreement across models)

## Critical Blockers
Issues that must be fixed before implementation (only if at least one reviewer flagged as critical):
- **[Category]** Issue title - flagged by: [model names]
  - Impact: ...
  - Recommended fix: ...

## Major Suggestions
Significant improvements (only if at least one reviewer flagged as major):
- **[Category]** Issue title - flagged by: [model names]
  - Description: ...
  - Recommended fix: ...

## Minor Suggestions
Smaller improvements and optimizations:
- **[Category]** Issue title - flagged by: [model names]
  - Description: ...
  - Recommended fix: ...

## Escalation Candidates
Cross-cutting concerns the synthesis believes may warrant higher priority than any single reviewer assigned. These are synthesis-level observations, not attributed to individual reviewers:
- **[Category]** Concern summary
  - Related findings: [which reviewers raised related items and at what severity]
  - Reasoning: [why these collectively may be more significant]
  - Suggested severity: [what the synthesis recommends the author consider]

## Questions for Author
Clarifications needed (common questions across models):
- **[Category]** Question - flagged by: [model names]
  - Context: Why this matters

## Design Strengths
What the spec does well (areas of agreement):
- **[Category]** Strength - noted by: [model names]
  - Why this is effective

## Points of Agreement
- What all/most models agree on

## Points of Disagreement
- Where models conflict
- Your assessment of the disagreement

## Synthesis Notes
- Overall themes across reviews
- Actionable next steps
```

**Important**:
- Respect the Severity Fidelity Rules above — do not invent Critical Blockers that no reviewer identified
- Attribute issues to specific models with their original severity
- Use "None identified." for empty severity sections
- Note where models agree vs. disagree
- Focus on synthesizing actionable feedback across all reviews

---

{model_reviews}""",
    required_context=["spec_id", "title", "num_models", "model_reviews"],
    optional_context=[],
    metadata={
        "author": "foundry-mcp",
        "category": "plan_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "synthesis",
        "description": "AI-powered synthesis of multiple model responses",
    },
)


# =============================================================================
# Template Registries
# =============================================================================


# PLAN_REVIEW_* templates registry (PromptTemplate instances)
PLAN_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "PLAN_REVIEW_FULL_V1": PLAN_REVIEW_FULL_V1,
    "SYNTHESIS_PROMPT_V1": SYNTHESIS_PROMPT_V1,
}


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class PlanReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for SDD plan review workflows.

    Provides access to PLAN_REVIEW_* templates (PromptTemplate instances).

    PLAN_REVIEW Templates:
        - PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
        - SYNTHESIS_PROMPT_V1: Multi-model response synthesis
    """

    def __init__(self) -> None:
        """Initialize the builder with all templates."""
        self._registry = PromptRegistry()
        # Register PLAN_REVIEW_* templates
        for template in PLAN_REVIEW_TEMPLATES.values():
            self._registry.register(template)

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a plan review prompt.

        Args:
            prompt_id: Template ID (PLAN_REVIEW_*, SYNTHESIS_PROMPT_V1,
                or SPEC_REVIEW_VS_PLAN_V1 for plan-enhanced reviews)
            context: Context dict with required variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id not found or required context missing
        """
        # Check if it's a PLAN_REVIEW_* template
        if prompt_id in PLAN_REVIEW_TEMPLATES:
            render_context = dict(context)
            # Add default response schema if not provided
            if "response_schema" not in render_context:
                render_context["response_schema"] = _RESPONSE_SCHEMA
            return self._registry.render(prompt_id, render_context)

        # Delegate to spec review builder for spec-vs-plan templates
        from foundry_mcp.core.prompts.spec_review import SPEC_REVIEW_TEMPLATES

        if prompt_id in SPEC_REVIEW_TEMPLATES:
            from foundry_mcp.core.prompts.spec_review import SpecReviewPromptBuilder

            return SpecReviewPromptBuilder().build(prompt_id, context)

        # Unknown prompt_id
        all_ids = sorted(list(PLAN_REVIEW_TEMPLATES.keys()) + list(SPEC_REVIEW_TEMPLATES.keys()))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {all_ids}")

    def list_prompts(self) -> List[str]:
        """
        Return all available prompt IDs.

        Returns:
            Sorted list of all prompt IDs (PLAN_REVIEW_* and SPEC_REVIEW_*)
        """
        from foundry_mcp.core.prompts.spec_review import SPEC_REVIEW_TEMPLATES

        return sorted(list(PLAN_REVIEW_TEMPLATES.keys()) + list(SPEC_REVIEW_TEMPLATES.keys()))

    def get_template(self, prompt_id: str) -> PromptTemplate:
        """
        Get a PLAN_REVIEW_* template by ID for inspection.

        Args:
            prompt_id: Template identifier (must be a PLAN_REVIEW_* template)

        Returns:
            The PromptTemplate

        Raises:
            KeyError: If not found or not a PLAN_REVIEW_* template
        """
        if prompt_id not in PLAN_REVIEW_TEMPLATES:
            available = sorted(PLAN_REVIEW_TEMPLATES.keys())
            raise KeyError(
                f"Template '{prompt_id}' not found. "
                f"Only PLAN_REVIEW_* templates can be inspected. Available: {available}"
            )
        return self._registry.get_required(prompt_id)


# =============================================================================
# Helper Functions
# =============================================================================


def get_response_schema() -> str:
    """
    Get the standard response schema for plan reviews.

    Returns:
        Response schema markdown string
    """
    return _RESPONSE_SCHEMA


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # PLAN_REVIEW_* templates
    "PLAN_REVIEW_FULL_V1",
    "SYNTHESIS_PROMPT_V1",
    "PLAN_REVIEW_TEMPLATES",
    # Builder
    "PlanReviewPromptBuilder",
    # Helpers
    "get_response_schema",
]
