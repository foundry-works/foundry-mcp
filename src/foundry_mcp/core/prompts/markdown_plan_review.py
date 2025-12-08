"""
Markdown plan review prompts for AI consultation workflows.

This module provides prompt templates for reviewing markdown plans
before converting them to formal JSON specifications. Supports iterative
review cycles to refine plans with AI feedback.

Templates:
    - MARKDOWN_PLAN_REVIEW_FULL_V1: Comprehensive 6-dimension review
    - MARKDOWN_PLAN_REVIEW_QUICK_V1: Critical blockers only
    - MARKDOWN_PLAN_REVIEW_SECURITY_V1: Security-focused review
    - MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1: Complexity and risk assessment

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
- Use category tags: [Completeness], [Architecture], [Sequencing], [Feasibility], [Risk], [Clarity]
- Include all sections even if empty (write "None identified" for empty sections)
- Be specific and actionable in all feedback
- For clarity issues, use Questions section rather than creating a separate category
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
- Remember: this is an early-stage plan, not a final spec"""


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

1. **Completeness** - Are all phases/deliverables identified? Missing sections?
2. **Architecture** - Sound approach? Coupling concerns? Missing abstractions?
3. **Sequencing** - Phases ordered correctly? Dependencies identified?
4. **Feasibility** - Realistic estimates? Hidden complexity?
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
            "Feasibility",
            "Risk",
            "Clarity",
        ],
        "description": "Comprehensive 6-dimension markdown plan review",
    },
)


# =============================================================================
# MARKDOWN_PLAN_REVIEW_QUICK_V1
# =============================================================================

MARKDOWN_PLAN_REVIEW_QUICK_V1 = PromptTemplate(
    id="MARKDOWN_PLAN_REVIEW_QUICK_V1",
    version="1.0",
    system_prompt=_MARKDOWN_PLAN_REVIEW_SYSTEM_PROMPT,
    user_template="""You are conducting a quick review of a markdown implementation plan.

**Plan Name**: {plan_name}
**Review Type**: Quick (focus on blockers and questions)

**Your role**: Identify critical blockers and key questions that need resolution before this becomes a spec.

**Focus on finding:**

1. **Critical Blockers**: What would prevent this plan from becoming a valid spec?
   - Missing phases or deliverables
   - Undefined dependencies
   - Unresolved technical decisions
   - Incomplete objectives

2. **Key Questions**: What needs clarification?
   - Ambiguous requirements
   - Unclear technical approaches
   - Missing context or rationale
   - Edge cases not addressed

**Evaluation areas**:
- **Completeness**: Are all necessary sections present?
- **Questions**: What clarifications are needed?

**MARKDOWN PLAN TO REVIEW:**

{plan_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus primarily on Critical Blockers and Questions sections. Brief notes for other sections are sufficient.""",
    required_context=["plan_content", "plan_name"],
    optional_context=["response_schema", "plan_path"],
    metadata={
        "author": "foundry-mcp",
        "category": "markdown_plan_review",
        "workflow": "MARKDOWN_PLAN_REVIEW",
        "review_type": "quick",
        "focus": ["Critical Blockers", "Questions"],
        "description": "Quick review focusing on blockers and clarifications",
    },
)


# =============================================================================
# MARKDOWN_PLAN_REVIEW_SECURITY_V1
# =============================================================================

MARKDOWN_PLAN_REVIEW_SECURITY_V1 = PromptTemplate(
    id="MARKDOWN_PLAN_REVIEW_SECURITY_V1",
    version="1.0",
    system_prompt="""You are a security specialist reviewing implementation plans.
Your task is to identify security vulnerabilities, risks, and recommend mitigations
in the proposed implementation approach.

Guidelines:
- Focus on authentication, authorization, and data protection
- Identify injection risks and common vulnerabilities
- Consider OWASP Top 10 and industry security standards
- Provide specific, actionable remediation recommendations
- Prioritize findings by risk severity""",
    user_template="""You are conducting a security review of a markdown implementation plan.

**Plan Name**: {plan_name}
**Review Type**: Security (focus on vulnerabilities and risks)

**Your role**: Security specialist helping identify and mitigate potential vulnerabilities in the proposed approach.

**Focus on security considerations:**

1. **Authentication & Authorization**:
   - Are authentication mechanisms properly planned?
   - Is authorization enforced at appropriate boundaries?
   - Does the plan follow principle of least privilege?

2. **Data Protection**:
   - Is input validation planned?
   - Are secrets managed securely?
   - Is encryption considered for data at rest and in transit?
   - Do error handling plans avoid leaking sensitive information?

3. **Common Vulnerabilities**:
   - Are injection attacks (SQL, command, XSS, CSRF) considered?
   - Are security headers and protections planned?
   - Is rate limiting and DoS protection addressed?
   - Are insecure defaults avoided?

4. **Audit & Compliance**:
   - Is audit logging planned for security events?
   - Are privacy concerns addressed?
   - Are relevant compliance requirements considered?

**Evaluation areas**:
- **Security**: Authentication, authorization, data protection, vulnerability prevention
- **Architecture**: Security-relevant design decisions
- **Risk**: Security risks and mitigations

**MARKDOWN PLAN TO REVIEW:**

{plan_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus primarily on Security category feedback. Include Critical Blockers for any security issues that must be addressed before this becomes a spec.""",
    required_context=["plan_content", "plan_name"],
    optional_context=["response_schema", "plan_path"],
    metadata={
        "author": "foundry-mcp",
        "category": "markdown_plan_review",
        "workflow": "MARKDOWN_PLAN_REVIEW",
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
# MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1
# =============================================================================

MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1 = PromptTemplate(
    id="MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1",
    version="1.0",
    system_prompt="""You are a pragmatic senior engineer assessing technical feasibility.
Your task is to identify implementation challenges, risks, and hidden complexity
in the proposed approach.

Guidelines:
- Identify non-obvious technical challenges
- Evaluate dependency availability and stability
- Assess resource and timeline realism
- Highlight areas of concentrated complexity
- Suggest risk mitigations and alternatives""",
    user_template="""You are conducting a technical feasibility review of a markdown implementation plan.

**Plan Name**: {plan_name}
**Review Type**: Feasibility (focus on implementation challenges)

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
   - What technical limitations could impact the approach?
   - Are performance requirements achievable?
   - Are there scalability considerations?
   - What infrastructure requirements exist?

4. **Implementation Risks**:
   - What could go wrong during implementation?
   - Where are the highest-risk technical areas?
   - What mitigation strategies are needed?

**Evaluation areas**:
- **Completeness**: Are technical requirements fully specified?
- **Feasibility**: Is the technical approach sound?
- **Risk**: What are the technical risks?

**MARKDOWN PLAN TO REVIEW:**

{plan_content}

---

**Required Output Format** (Markdown):
{response_schema}

**Note**: Focus on technical challenges and risks. Identify Major Suggestions for areas of hidden complexity and Critical Blockers for missing technical requirements.""",
    required_context=["plan_content", "plan_name"],
    optional_context=["response_schema", "plan_path"],
    metadata={
        "author": "foundry-mcp",
        "category": "markdown_plan_review",
        "workflow": "MARKDOWN_PLAN_REVIEW",
        "review_type": "feasibility",
        "focus": ["Complexity", "Dependencies", "Risks"],
        "description": "Technical feasibility and risk assessment",
    },
)


# =============================================================================
# Template Registry
# =============================================================================


MARKDOWN_PLAN_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "MARKDOWN_PLAN_REVIEW_FULL_V1": MARKDOWN_PLAN_REVIEW_FULL_V1,
    "MARKDOWN_PLAN_REVIEW_QUICK_V1": MARKDOWN_PLAN_REVIEW_QUICK_V1,
    "MARKDOWN_PLAN_REVIEW_SECURITY_V1": MARKDOWN_PLAN_REVIEW_SECURITY_V1,
    "MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1": MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1,
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
        - MARKDOWN_PLAN_REVIEW_QUICK_V1: Critical blockers and questions focus
        - MARKDOWN_PLAN_REVIEW_SECURITY_V1: Security-focused review
        - MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1: Technical complexity assessment

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
            raise KeyError(
                f"Template '{prompt_id}' not found. Available: {available}"
            )
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
    "MARKDOWN_PLAN_REVIEW_QUICK_V1",
    "MARKDOWN_PLAN_REVIEW_SECURITY_V1",
    "MARKDOWN_PLAN_REVIEW_FEASIBILITY_V1",
    "MARKDOWN_PLAN_REVIEW_TEMPLATES",
    # Builder
    "MarkdownPlanReviewPromptBuilder",
    # Helpers
    "get_response_schema",
]
