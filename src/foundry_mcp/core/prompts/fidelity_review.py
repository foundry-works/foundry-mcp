"""
Prompt templates for fidelity review workflow.

This module provides prompts for comparing implementation against
specifications, identifying deviations, and assessing compliance
with documented requirements.

Prompt IDs (PromptTemplate-based):
    - FIDELITY_REVIEW_V1: Main 6-section fidelity review prompt
    - FIDELITY_DEVIATION_ANALYSIS_V1: Analyze identified deviations
    - FIDELITY_COMPLIANCE_SUMMARY_V1: Generate compliance summary

Legacy Prompt IDs (string templates for backward compatibility):
    - review_task: Compare task implementation against spec requirements
    - review_phase: Review entire phase for fidelity to spec
    - compare_files: Compare specific files against spec expectations
    - deviation_analysis: Analyze identified deviations for impact
    - compliance_summary: Generate compliance summary report
"""

from __future__ import annotations

from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptTemplate


# =============================================================================
# Response Schema
# =============================================================================


# JSON response schema for fidelity reviews - structured format for AI response
FIDELITY_RESPONSE_SCHEMA = """{
  "verdict": "pass|fail|partial|unknown",
  "summary": "Overall findings (any length).",
  "requirement_alignment": {
    "answer": "yes|no|partial",
    "details": "Explain how implementation aligns or diverges."
  },
  "success_criteria": {
    "met": "yes|no|partial",
    "details": "Call out verification steps passed or missing."
  },
  "deviations": [
    {
      "description": "Describe deviation from the spec.",
      "justification": "Optional rationale or evidence.",
      "severity": "critical|high|medium|low"
    }
  ],
  "test_coverage": {
    "status": "sufficient|insufficient|not_applicable",
    "details": "Summarise test evidence or gaps."
  },
  "code_quality": {
    "issues": ["Describe each notable quality concern."],
    "details": "Optional supporting commentary."
  },
  "documentation": {
    "status": "adequate|inadequate|not_applicable",
    "details": "Note doc updates or omissions."
  },
  "issues": ["Concise list of primary issues for consensus logic."],
  "recommendations": ["Actionable next steps to resolve findings."]
}"""


# =============================================================================
# Severity Categorization Keywords
# =============================================================================


# CRITICAL: Security vulnerabilities, data loss, crashes
CRITICAL_KEYWORDS = [
    "security",
    "vulnerability",
    "injection",
    "xss",
    "csrf",
    "authentication bypass",
    "unauthorized access",
    "data loss",
    "crash",
    "segfault",
    "memory leak",
    "remote code execution",
    "privilege escalation",
    "buffer overflow",
]

# HIGH: Incorrect behavior, spec violations, broken functionality
HIGH_KEYWORDS = [
    "incorrect",
    "wrong",
    "broken",
    "fails",
    "failure",
    "spec violation",
    "requirement not met",
    "does not match",
    "missing required",
    "critical bug",
    "data corruption",
    "logic error",
    "incorrect behavior",
]

# MEDIUM: Performance issues, missing tests, code quality
MEDIUM_KEYWORDS = [
    "performance",
    "slow",
    "inefficient",
    "optimization",
    "missing test",
    "no tests",
    "untested",
    "test coverage",
    "code quality",
    "maintainability",
    "complexity",
    "duplication",
    "refactor",
    "improvement needed",
]

# LOW: Style issues, documentation, minor improvements
LOW_KEYWORDS = [
    "style",
    "formatting",
    "naming",
    "documentation",
    "comment",
    "typo",
    "whitespace",
    "minor",
    "suggestion",
    "consider",
    "could be better",
]

# All severity keywords organized by level
SEVERITY_KEYWORDS = {
    "critical": CRITICAL_KEYWORDS,
    "high": HIGH_KEYWORDS,
    "medium": MEDIUM_KEYWORDS,
    "low": LOW_KEYWORDS,
}


# =============================================================================
# PromptTemplate-based Prompts (New Format)
# =============================================================================


# Main fidelity review prompt - 6-section structure
FIDELITY_REVIEW_V1 = PromptTemplate(
    id="FIDELITY_REVIEW_V1",
    version="1.0",
    system_prompt="""You are an expert code reviewer performing implementation fidelity analysis.

Your role is to compare actual code implementation against specification requirements
and identify any deviations, issues, or concerns.

CRITICAL CONSTRAINTS:
- This is a READ-ONLY review - you MUST NOT write, create, or modify ANY files
- Execute code or commands - ANALYSIS ONLY
- Provide findings as structured JSON in your response

Focus on:
1. Requirement alignment - Does implementation match spec?
2. Success criteria - Are verification steps satisfied?
3. Deviations - Any divergences from specification?
4. Test coverage - Are tests comprehensive?
5. Code quality - Any maintainability concerns?
6. Documentation - Is implementation properly documented?""",
    user_template="""# Implementation Fidelity Review

## 1. Context
**Spec ID:** {spec_id}
**Spec Title:** {spec_title}
{spec_description}
**Review Scope:** {review_scope}

## 2. Specification Requirements
{spec_requirements}

## 3. Implementation Artifacts
{implementation_artifacts}

## 4. Test Results
{test_results}

## 5. Journal Entries
{journal_entries}

## 6. Review Questions

Please evaluate the implementation against the specification:

1. **Requirement Alignment:** Does the implementation match the spec requirements?
2. **Success Criteria:** Are all verification steps satisfied?
3. **Deviations:** Are there any deviations from the spec? If so, are they justified?
4. **Test Coverage:** Are tests comprehensive and passing?
5. **Code Quality:** Are there any quality, maintainability, or security concerns?
6. **Documentation:** Is the implementation properly documented?

### Required Response Format

Respond **only** with valid JSON matching the schema below. Do not include Markdown, prose, or additional commentary outside the JSON object.

```json
{response_schema}
```

Rules:
- Use lowercase values shown for enumerated fields (e.g., `verdict`, status flags)
- Keep arrays as arrays (use `[]` when a section has nothing to report)
- Populate `issues` and `recommendations` with key takeaways
- Feel free to include additional keys if needed, but never omit the ones above
- Severity levels for deviations: critical, high, medium, low""",
    required_context=[
        "spec_id",
        "spec_title",
        "review_scope",
        "spec_requirements",
        "implementation_artifacts",
    ],
    optional_context=[
        "spec_description",
        "test_results",
        "journal_entries",
        "response_schema",
    ],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "implementation",
        "sections": [
            "Context",
            "Specification Requirements",
            "Implementation Artifacts",
            "Test Results",
            "Journal Entries",
            "Review Questions",
        ],
        "output_format": "json",
        "severity_levels": ["critical", "high", "medium", "low"],
    },
)


# Deviation analysis prompt - for deep-diving into identified deviations
FIDELITY_DEVIATION_ANALYSIS_V1 = PromptTemplate(
    id="FIDELITY_DEVIATION_ANALYSIS_V1",
    version="1.0",
    system_prompt="""You are an expert software architect analyzing specification deviations.

Your role is to assess the impact of identified deviations between implementation
and specification, determine risks, and recommend remediation strategies.

Focus on:
- Impact assessment (functional, security, performance, maintenance)
- Risk analysis and downstream effects
- Remediation options with effort/risk tradeoffs
- Prioritized recommendations""",
    user_template="""# Deviation Analysis

## Context
**Spec ID:** {spec_id}
**Analysis Scope:** {scope}

## Identified Deviations
{deviations_json}

## Original Requirements
{original_requirements}

## Analysis Requirements

For each deviation, provide:
1. **Impact Assessment**: Functional, security, performance, maintenance impact
2. **Risk Analysis**: Risks introduced by this deviation
3. **Downstream Effects**: How does this affect dependent components?
4. **Remediation Options**: Ways to address with effort/risk tradeoffs
5. **Recommendation**: accept|fix_now|fix_later|needs_discussion

### Required Response Format

Respond with valid JSON:

```json
{{
  "analysis_scope": "{scope}",
  "total_deviations": 0,
  "critical_count": 0,
  "deviation_analysis": [
    {{
      "deviation_id": "index or identifier",
      "original_deviation": "brief description",
      "impact_assessment": {{
        "functional_impact": "none|minor|moderate|major",
        "security_impact": "none|minor|moderate|major",
        "performance_impact": "none|minor|moderate|major",
        "maintenance_impact": "none|minor|moderate|major"
      }},
      "affected_components": ["list of affected components"],
      "downstream_effects": ["list of downstream effects"],
      "remediation_options": [
        {{
          "option": "description",
          "effort": "low|medium|high",
          "risk": "low|medium|high"
        }}
      ],
      "recommendation": "accept|fix_now|fix_later|needs_discussion",
      "rationale": "explanation for recommendation"
    }}
  ],
  "overall_risk_level": "low|medium|high|critical",
  "recommended_actions": ["prioritized list of recommended actions"],
  "summary": "overall deviation analysis summary"
}}
```""",
    required_context=["spec_id", "scope", "deviations_json", "original_requirements"],
    optional_context=[],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "analysis",
        "output_format": "json",
    },
)


# Compliance summary prompt - for generating overall compliance reports
FIDELITY_COMPLIANCE_SUMMARY_V1 = PromptTemplate(
    id="FIDELITY_COMPLIANCE_SUMMARY_V1",
    version="1.0",
    system_prompt="""You are an expert technical lead generating compliance reports.

Your role is to synthesize fidelity review findings into an executive summary
with clear compliance status, prioritized issues, and sign-off recommendations.

Focus on:
- Overall compliance score and status
- Phase-by-phase breakdown
- Critical blocking issues
- Sign-off readiness assessment""",
    user_template="""# Compliance Summary Report

## Specification
**Spec ID:** {spec_id}
**Title:** {spec_title}
**Total Phases:** {total_phases}
**Total Tasks:** {total_tasks}

## Fidelity Review Data
{review_data}

## Summary Requirements

Generate a compliance summary addressing:
1. **Overall Compliance**: What is the overall compliance level?
2. **Phase Breakdown**: Compliance by phase
3. **Critical Issues**: List critical compliance issues
4. **Recommendations**: Prioritized recommendations
5. **Sign-off Status**: Is the implementation ready for approval?

### Required Response Format

Respond with valid JSON:

```json
{{
  "spec_id": "{spec_id}",
  "spec_title": "{spec_title}",
  "overall_compliance": {{
    "score": 0-100,
    "status": "compliant|mostly_compliant|needs_work|non_compliant",
    "tasks_compliant": 0,
    "tasks_partial": 0,
    "tasks_non_compliant": 0
  }},
  "phase_breakdown": [
    {{
      "phase_id": "phase-id",
      "phase_title": "title",
      "compliance_score": 0-100,
      "status": "compliant|partial|non_compliant"
    }}
  ],
  "critical_issues": [
    {{
      "issue": "description",
      "location": "task or phase id",
      "priority": "p0|p1|p2",
      "remediation": "suggested fix"
    }}
  ],
  "recommendations": [
    {{
      "recommendation": "description",
      "priority": "critical|high|medium|low",
      "effort": "low|medium|high"
    }}
  ],
  "sign_off": {{
    "ready": true|false,
    "blocking_issues": ["list of issues that must be resolved"],
    "conditions": ["conditions for approval if any"]
  }},
  "summary": "executive summary of compliance status"
}}
```""",
    required_context=[
        "spec_id",
        "spec_title",
        "total_phases",
        "total_tasks",
        "review_data",
    ],
    optional_context=[],
    metadata={
        "workflow": "fidelity_review",
        "author": "system",
        "category": "reporting",
        "output_format": "json",
    },
)


# =============================================================================
# Template Registry (PromptTemplate-based)
# =============================================================================


FIDELITY_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "FIDELITY_REVIEW_V1": FIDELITY_REVIEW_V1,
    "FIDELITY_DEVIATION_ANALYSIS_V1": FIDELITY_DEVIATION_ANALYSIS_V1,
    "FIDELITY_COMPLIANCE_SUMMARY_V1": FIDELITY_COMPLIANCE_SUMMARY_V1,
}


# =============================================================================
# Legacy Prompt Templates (String-based for backward compatibility)
# =============================================================================

# Legacy templates have been removed. Use FIDELITY_REVIEW_V1 and related prompts.


# =============================================================================
# Prompt Builder Implementation
# =============================================================================


class FidelityReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for fidelity review workflow.

    Provides templates for comparing implementations against specifications
    and generating compliance reports.

    Supports PromptTemplate-based prompts (FIDELITY_*_V1).
    """

    def __init__(self) -> None:
        """Initialize the builder with template registries."""
        self._prompt_templates = FIDELITY_REVIEW_TEMPLATES

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a fidelity review prompt.

        Args:
            prompt_id: Template identifier. Supports:
                - PromptTemplate IDs: FIDELITY_REVIEW_V1, FIDELITY_DEVIATION_ANALYSIS_V1,
                  FIDELITY_COMPLIANCE_SUMMARY_V1
            context: Template context variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id is not recognized
        """
        # Check PromptTemplate registry
        if prompt_id in self._prompt_templates:
            template = self._prompt_templates[prompt_id]

            # Provide defaults for optional context
            render_context = dict(context)

            # Add response schema default
            if "response_schema" not in render_context:
                render_context["response_schema"] = FIDELITY_RESPONSE_SCHEMA

            # Add empty defaults for optional fields
            if "spec_description" not in render_context:
                render_context["spec_description"] = ""
            if "test_results" not in render_context:
                render_context["test_results"] = "*No test results available*"
            if "journal_entries" not in render_context:
                render_context["journal_entries"] = "*No journal entries found*"

            return template.render(render_context)

        # Unknown prompt_id
        available = ", ".join(sorted(self._prompt_templates.keys()))
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {available}")

    def list_prompts(self) -> List[str]:
        """Return available prompt IDs for fidelity review."""
        return sorted(list(self._prompt_templates.keys()))

    def get_severity_keywords(self, level: str) -> List[str]:
        """
        Get severity categorization keywords for a given level.

        Args:
            level: Severity level (critical, high, medium, low)

        Returns:
            List of keywords for that severity level
        """
        return SEVERITY_KEYWORDS.get(level.lower(), [])


__all__ = [
    # PromptTemplate instances
    "FIDELITY_REVIEW_V1",
    "FIDELITY_DEVIATION_ANALYSIS_V1",
    "FIDELITY_COMPLIANCE_SUMMARY_V1",
    # Template registries
    "FIDELITY_REVIEW_TEMPLATES",
    # Response schema
    "FIDELITY_RESPONSE_SCHEMA",
    # Severity keywords
    "SEVERITY_KEYWORDS",
    "CRITICAL_KEYWORDS",
    "HIGH_KEYWORDS",
    "MEDIUM_KEYWORDS",
    "LOW_KEYWORDS",
    # Builder
    "FidelityReviewPromptBuilder",
]
