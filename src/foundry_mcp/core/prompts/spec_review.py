"""
Spec-vs-plan review prompts for AI consultation workflows.

This module provides prompt templates for comparing a spec (JSON) against
its source plan (markdown) to identify translation gaps, missing coverage,
and undocumented additions.

Prompt IDs:
    - SPEC_REVIEW_VS_PLAN_V1: Compare spec against its plan for alignment

This review is auto-activated when ``review(action="spec", review_type="full")``
is called on a spec that has a ``metadata.plan_path`` pointing to a readable
plan file.  It does NOT replace standalone spec reviews — those continue to
use ``PLAN_REVIEW_FULL_V1`` when no plan is available.
"""

from __future__ import annotations

from typing import Any, Dict, List

from foundry_mcp.core.prompts import PromptBuilder, PromptRegistry, PromptTemplate

# =============================================================================
# Response Schema
# =============================================================================

SPEC_VS_PLAN_RESPONSE_SCHEMA = """{
  "verdict": "aligned|deviation|incomplete",
  "summary": "Overall alignment assessment between plan and spec.",
  "coverage": {
    "plan_phases_covered": 0,
    "plan_phases_total": 0,
    "plan_tasks_covered": 0,
    "plan_tasks_total": 0,
    "missing_items": ["task or phase title from plan that has no corresponding spec node"]
  },
  "fidelity": {
    "status": "aligned|diluted|diverged",
    "issues": [
      {
        "plan_item": "item from the plan",
        "spec_item": "corresponding item in the spec (or 'missing')",
        "concern": "description of the semantic gap"
      }
    ]
  },
  "metadata_alignment": {
    "success_criteria": "aligned|missing|partial",
    "constraints": "aligned|missing|partial",
    "risks": "aligned|missing|partial",
    "open_questions": "aligned|missing|partial"
  },
  "undocumented_additions": [
    {
      "spec_item": "item in spec not traceable to plan",
      "justification_needed": true
    }
  ],
  "issues": ["concise list of primary alignment issues"],
  "recommendations": ["actionable next steps to improve alignment"]
}"""


# =============================================================================
# System Prompt
# =============================================================================

_SPEC_REVIEW_SYSTEM_PROMPT = """You are an expert specification reviewer comparing a JSON spec \
against its source plan (markdown).

Your role is to identify translation gaps — places where the plan's intent was \
lost, diluted, or silently modified when translated into the spec. You also flag \
items in the spec that have no traceable origin in the plan.

CRITICAL CONSTRAINTS:
- Compare SEMANTICS, not syntax. Markdown prose and JSON will never match word-for-word.
- Do NOT re-evaluate whether the plan itself was good — that was the plan review's job.
- Do NOT check implementation code — that is the fidelity gate's job.
- "Undocumented additions" are NOT errors. They are items in the spec with no \
  plan origin. Flag them for human review, do not auto-reject.
- Do NOT focus on ownership, responsibility, or team assignment concerns.

SEMANTIC MATCHING EXAMPLES — use these to calibrate your matching threshold:

These ARE matches (acceptable semantic gaps):
- Plan: "Implement OAuth2 with PKCE" → Spec: "Add PKCE-based OAuth2 authentication flow" (rewording)
- Plan: "Add input validation" → Spec: "Validate and sanitize user input" (elaboration)
- Plan: "Write unit tests for the parser" → Spec: "Create parser test suite with edge cases" (refinement)
- Plan: "Set up CI pipeline" → Spec: "Configure GitHub Actions workflow for CI" (concretization)

These are NOT matches (real coverage gaps):
- Plan: "Implement OAuth2 with PKCE" → Spec: "Add basic password authentication" (different approach)
- Plan: "Add rate limiting to all API endpoints" → Spec: "Add rate limiting to /login endpoint" (reduced scope)
- Plan: "Migrate database to PostgreSQL" → Spec has no database migration task (dropped entirely)"""


# =============================================================================
# SPEC_REVIEW_VS_PLAN_V1
# =============================================================================

SPEC_REVIEW_VS_PLAN_V1 = PromptTemplate(
    id="SPEC_REVIEW_VS_PLAN_V1",
    version="1.0",
    system_prompt=_SPEC_REVIEW_SYSTEM_PROMPT,
    user_template="""You are comparing a software specification (JSON) against its source plan (markdown) \
to identify translation gaps.

**Spec ID**: {spec_id}
**Title**: {title}
**Review Type**: Spec-vs-Plan alignment

**Your role**: Verify that the spec faithfully captures the plan's intent, coverage, \
and metadata. The plan represents what was agreed upon with the human. The spec is the \
JSON translation. Any gap is a translation error or undocumented deviation.

**Evaluate across 7 comparison dimensions:**

1. **Coverage** — Every phase, task, and verification step in the plan has a \
corresponding spec node. Count phases and tasks covered vs total.

2. **Fidelity** — Spec tasks match the plan's intent (semantic alignment). \
Look for diluted requirements, shifted scope, or lost nuance.

3. **Success Criteria Mapping** — Plan's success criteria are reflected in \
`metadata.success_criteria` and/or task `acceptance_criteria`.

4. **Constraints Preserved** — Plan's constraints appear in `metadata.constraints`.

5. **Risks Preserved** — Plan's risk table is reflected in `metadata.risks`.

6. **Open Questions Preserved** — Plan's open questions appear in `metadata.open_questions`.

7. **Undocumented Additions** — Items in the spec that are not traceable to the plan. \
Flag them for review but do NOT treat them as errors. The spec may reasonably add \
error handling tasks, verification steps, or other implementation details not in the plan.

**PLAN (MARKDOWN — the agreed-upon design):**

{plan_content}

**SPECIFICATION (JSON — the translation to review):**

{spec_content}

---

**Required Output Format** (JSON):

Respond **only** with valid JSON matching the schema below. Do not include Markdown, \
prose, or additional commentary outside the JSON object.

```json
{response_schema}
```

**Verdict criteria:**
- "aligned": All plan items are covered in the spec with faithful semantic alignment. \
  Minor metadata gaps (e.g., one missing open question) are acceptable.
- "deviation": Spec covers most plan items but has semantic drift, diluted requirements, \
  or notable metadata gaps that should be addressed.
- "incomplete": Significant plan items are missing from the spec, or the spec \
  fundamentally misrepresents the plan's intent.""",
    required_context=["spec_content", "spec_id", "title", "plan_content"],
    optional_context=["response_schema"],
    metadata={
        "author": "foundry-mcp",
        "category": "spec_review",
        "workflow": "PLAN_REVIEW",
        "review_type": "spec-vs-plan",
        "dimensions": [
            "Coverage",
            "Fidelity",
            "Success Criteria Mapping",
            "Constraints Preserved",
            "Risks Preserved",
            "Open Questions Preserved",
            "Undocumented Additions",
        ],
        "description": "Compare spec against plan for translation fidelity",
        "output_format": "json",
    },
)


# =============================================================================
# Template Registry
# =============================================================================

SPEC_REVIEW_TEMPLATES: Dict[str, PromptTemplate] = {
    "SPEC_REVIEW_VS_PLAN_V1": SPEC_REVIEW_VS_PLAN_V1,
}


# =============================================================================
# Prompt Builder
# =============================================================================


class SpecReviewPromptBuilder(PromptBuilder):
    """
    Prompt builder for spec-vs-plan review workflows.

    Provides access to SPEC_REVIEW_* templates (PromptTemplate instances).

    Templates:
        - SPEC_REVIEW_VS_PLAN_V1: Compare spec against plan for alignment
    """

    def __init__(self) -> None:
        self._registry = PromptRegistry()
        for template in SPEC_REVIEW_TEMPLATES.values():
            self._registry.register(template)

    def build(self, prompt_id: str, context: Dict[str, Any]) -> str:
        """
        Build a spec review prompt.

        Args:
            prompt_id: Template ID (SPEC_REVIEW_VS_PLAN_V1)
            context: Context dict with required variables

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt_id not found or required context missing
        """
        if prompt_id in SPEC_REVIEW_TEMPLATES:
            render_context = dict(context)
            if "response_schema" not in render_context:
                render_context["response_schema"] = SPEC_VS_PLAN_RESPONSE_SCHEMA
            return self._registry.render(prompt_id, render_context)

        all_ids = sorted(SPEC_REVIEW_TEMPLATES.keys())
        raise ValueError(f"Unknown prompt_id '{prompt_id}'. Available: {all_ids}")

    def list_prompts(self) -> List[str]:
        return sorted(SPEC_REVIEW_TEMPLATES.keys())

    def get_template(self, prompt_id: str) -> PromptTemplate:
        """Get a template by ID for inspection."""
        if prompt_id not in SPEC_REVIEW_TEMPLATES:
            available = sorted(SPEC_REVIEW_TEMPLATES.keys())
            raise KeyError(f"Template '{prompt_id}' not found. Available: {available}")
        return self._registry.get_required(prompt_id)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "SPEC_REVIEW_VS_PLAN_V1",
    "SPEC_REVIEW_TEMPLATES",
    "SPEC_VS_PLAN_RESPONSE_SCHEMA",
    "SpecReviewPromptBuilder",
]
