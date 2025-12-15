"""
Plan review prompts for AI consultation workflows.

This module provides prompt templates for two categories of plan review:

1. **PLAN_REVIEW_*_V1 Templates** (PromptTemplate instances):
   - PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
   - PLAN_REVIEW_QUICK_V1: Critical blockers and questions focus
   - PLAN_REVIEW_SECURITY_V1: Security-focused review
   - PLAN_REVIEW_FEASIBILITY_V1: Technical complexity and risk assessment
   - SYNTHESIS_PROMPT_V1: Multi-model response synthesis

2. **Legacy Templates** (string templates):
   - review_spec: Comprehensive spec review
   - review_phase: Phase-specific review
   - security_review: Security-focused review
   - feasibility_review: Feasibility assessment
   - quick_review: Fast high-level review

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
- Be collaborative, not adversarial"""


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
# PLAN_REVIEW_QUICK_V1
# =============================================================================

PLAN_REVIEW_QUICK_V1 = PromptTemplate(
    id="PLAN_REVIEW_QUICK_V1",
    version="1.0",
    system_prompt=_PLAN_REVIEW_SYSTEM_PROMPT,
    user_template="""You are conducting a quick technical review of a software specification.

**Spec**: {spec_id}
**Title**: {title}
**Review Type**: Quick (focus on blockers and questions)

**Your role**: Identify critical blockers and key questions that need resolution before implementation.

**Focus on finding:**

1. **Critical Blockers**: What would prevent implementation from starting?
   - Missing required sections or requirements
   - Undefined dependencies or integrations
   - Unresolved technical decisions
   - Incomplete acceptance criteria

2. **Key Questions**: What needs clarification?
   - Ambiguous requirements or acceptance criteria
   - Unclear technical approaches
   - Missing context or rationale
   - Edge cases not addressed

**Evaluation areas**:
- **Completeness**: Are all necessary sections and requirements present?
- **Questions**: What clarifications are needed?

**SPECIFICATION TO REVIEW:**

{spec_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus primarily on Critical Blockers and Questions sections. Brief notes for other sections are sufficient.""",
    required_context=["spec_content", "spec_id", "title"],
    optional_context=["response_schema"],
    metadata={
        "author": "foundry-mcp",
        "category": "plan_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "quick",
        "focus": ["Critical Blockers", "Questions"],
        "description": "Quick review focusing on blockers and clarifications",
    },
)


# =============================================================================
# PLAN_REVIEW_SECURITY_V1
# =============================================================================

PLAN_REVIEW_SECURITY_V1 = PromptTemplate(
    id="PLAN_REVIEW_SECURITY_V1",
    version="1.0",
    system_prompt="""You are a security specialist reviewing software specifications.
Your task is to identify security vulnerabilities, risks, and recommend mitigations.

Guidelines:
- Focus on authentication, authorization, and data protection
- Identify injection risks and common vulnerabilities
- Consider OWASP Top 10 and industry security standards
- Provide specific, actionable remediation recommendations
- Prioritize findings by risk severity""",
    user_template="""You are conducting a security review of a software specification.

**Spec**: {spec_id}
**Title**: {title}
**Review Type**: Security (focus on vulnerabilities and risks)

**Your role**: Security specialist helping identify and mitigate potential vulnerabilities.

**Focus on security considerations:**

1. **Authentication & Authorization**:
   - Are authentication mechanisms properly designed?
   - Is authorization enforced at appropriate boundaries?
   - Does access control follow principle of least privilege?

2. **Data Protection**:
   - Is input validation comprehensive?
   - Are secrets managed securely?
   - Is data encrypted at rest and in transit?
   - Do error messages avoid leaking sensitive information?

3. **Common Vulnerabilities**:
   - Are injection attacks (SQL, command, XSS, CSRF) prevented?
   - Are security headers and protections in place?
   - Is rate limiting and DoS protection addressed?
   - Are insecure defaults avoided?

4. **Audit & Compliance**:
   - Is audit logging sufficient for security events?
   - Are privacy concerns addressed?
   - Are relevant compliance requirements considered?

**Evaluation areas**:
- **Security**: Authentication, authorization, data protection, vulnerability prevention
- **Architecture**: Security-relevant design decisions
- **Data Model**: Data sensitivity, encryption, access patterns

**SPECIFICATION TO REVIEW:**

{spec_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus primarily on Security category feedback. Include Critical Blockers for any security issues that must be addressed before implementation.""",
    required_context=["spec_content", "spec_id", "title"],
    optional_context=["response_schema"],
    metadata={
        "author": "foundry-mcp",
        "category": "plan_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "security",
        "focus": [
            "Authentication",
            "Authorization",
            "Data Protection",
            "Vulnerabilities",
        ],
        "description": "Security-focused review for vulnerabilities and risks",
    },
)


# =============================================================================
# PLAN_REVIEW_FEASIBILITY_V1
# =============================================================================

PLAN_REVIEW_FEASIBILITY_V1 = PromptTemplate(
    id="PLAN_REVIEW_FEASIBILITY_V1",
    version="1.0",
    system_prompt="""You are a pragmatic senior engineer assessing technical feasibility.
Your task is to identify implementation challenges, risks, and hidden complexity.

Guidelines:
- Identify non-obvious technical challenges
- Evaluate dependency availability and stability
- Assess resource and timeline realism
- Highlight areas of concentrated complexity
- Suggest risk mitigations and alternatives""",
    user_template="""You are conducting a technical complexity review of a software specification.

**Spec**: {spec_id}
**Title**: {title}
**Review Type**: Technical Complexity (focus on implementation challenges)

**Your role**: Pragmatic engineer helping identify technical challenges and implementation risks.

**Focus on technical considerations:**

1. **Hidden Complexity**:
   - What technical challenges might not be obvious?
   - Where does complexity concentrate?
   - What edge cases increase difficulty?

2. **Dependencies & Integration**:
   - Are all required dependencies identified?
   - Are external services/APIs available and documented?
   - Are integration points well-defined?
   - What dependency risks exist?

3. **Technical Constraints**:
   - What technical limitations could impact the design?
   - Are performance requirements achievable?
   - Are there scalability considerations?
   - What infrastructure requirements exist?

4. **Implementation Risks**:
   - What could go wrong during implementation?
   - Where are the highest-risk technical areas?
   - What mitigation strategies are needed?

**Evaluation areas**:
- **Completeness**: Are technical requirements fully specified?
- **Architecture**: Is the technical approach sound?
- **Data Model**: Are data complexity factors addressed?
- **Interface Design**: Are integration points well-defined?

**SPECIFICATION TO REVIEW:**

{spec_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus on technical challenges and risks. Identify Major Suggestions for areas of hidden complexity and Critical Blockers for missing technical requirements.""",
    required_context=["spec_content", "spec_id", "title"],
    optional_context=["response_schema"],
    metadata={
        "author": "foundry-mcp",
        "category": "plan_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "feasibility",
        "focus": ["Complexity", "Dependencies", "Risks"],
        "description": "Technical feasibility and risk assessment",
    },
)


# =============================================================================
# SYNTHESIS_PROMPT_V1
# =============================================================================

SYNTHESIS_PROMPT_V1 = PromptTemplate(
    id="SYNTHESIS_PROMPT_V1",
    version="1.0",
    system_prompt="""You are an expert at synthesizing multiple technical reviews.
Your task is to consolidate diverse perspectives into actionable consensus.

Guidelines:
- Attribute findings to specific models
- Identify areas of agreement and disagreement
- Prioritize by consensus strength
- Preserve unique insights from each model
- Create actionable, consolidated recommendations""",
    user_template="""You are synthesizing {num_models} independent AI reviews of a specification.

**Specification**: {title} (`{spec_id}`)

**Your Task**: Read all reviews below and create a comprehensive synthesis.

**Required Output** (Markdown format):

```markdown
# Synthesis

## Overall Assessment
- **Consensus Level**: Strong/Moderate/Weak/Conflicted (based on agreement across models)

## Critical Blockers
Issues that must be fixed before implementation (identified by multiple models):
- **[Category]** Issue title - flagged by: [model names]
  - Impact: ...
  - Recommended fix: ...

## Major Suggestions
Significant improvements that enhance quality, maintainability, or design:
- **[Category]** Issue title - flagged by: [model names]
  - Description: ...
  - Recommended fix: ...

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
- Attribute issues to specific models (e.g., "flagged by: gemini, codex")
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
    "PLAN_REVIEW_QUICK_V1": PLAN_REVIEW_QUICK_V1,
    "PLAN_REVIEW_SECURITY_V1": PLAN_REVIEW_SECURITY_V1,
    "PLAN_REVIEW_FEASIBILITY_V1": PLAN_REVIEW_FEASIBILITY_V1,
    "SYNTHESIS_PROMPT_V1": SYNTHESIS_PROMPT_V1,
}


# =============================================================================
# Legacy String Templates (for backwards compatibility)
# =============================================================================

# Legacy templates have been removed. Use PLAN_REVIEW_* templates.


# =============================================================================
# Template Registries
# =============================================================================


# PLAN_REVIEW_* templates registry (PromptTemplate instances)
PLAN_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "PLAN_REVIEW_FULL_V1": PLAN_REVIEW_FULL_V1,
    "PLAN_REVIEW_QUICK_V1": PLAN_REVIEW_QUICK_V1,
    "PLAN_REVIEW_SECURITY_V1": PLAN_REVIEW_SECURITY_V1,
    "PLAN_REVIEW_FEASIBILITY_V1": PLAN_REVIEW_FEASIBILITY_V1,
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
        - PLAN_REVIEW_QUICK_V1: Critical blockers and questions focus
        - PLAN_REVIEW_SECURITY_V1: Security-focused review
        - PLAN_REVIEW_FEASIBILITY_V1: Technical complexity assessment
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
            prompt_id: Template ID (PLAN_REVIEW_* or SYNTHESIS_PROMPT_V1)
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

        # Unknown prompt_id
        all_ids = sorted(list(PLAN_REVIEW_TEMPLATES.keys()))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {all_ids}")

    def list_prompts(self) -> List[str]:
        """
        Return all available prompt IDs.

        Returns:
            Sorted list of all prompt IDs (PLAN_REVIEW_*)
        """
        return sorted(list(PLAN_REVIEW_TEMPLATES.keys()))

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
    "PLAN_REVIEW_QUICK_V1",
    "PLAN_REVIEW_SECURITY_V1",
    "PLAN_REVIEW_FEASIBILITY_V1",
    "SYNTHESIS_PROMPT_V1",
    "PLAN_REVIEW_TEMPLATES",
    # Builder
    "PlanReviewPromptBuilder",
    # Helpers
    "get_response_schema",
]
