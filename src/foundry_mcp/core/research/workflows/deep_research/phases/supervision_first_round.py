"""First-round decompose → critique → revise pipeline.

Extracted from ``supervision.py`` to isolate the three-call first-round
decomposition pipeline from the main delegation loop.  Every function here
takes explicit parameters (no ``self``); the *workflow* argument provides
access to cross-cutting concerns (audit events, cancellation, config).

LLM call functions (``execute_llm_call``, ``execute_structured_llm_call``)
are accessed through the ``supervision`` module at call time rather than
imported directly, so that test patches applied to the ``supervision``
module's namespace are respected.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    DelegationResponse,
    ResearchDirective,
    parse_delegation_response,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research.phases.supervision_coverage import (
    critique_has_issues,
)
from foundry_mcp.core.research.workflows.deep_research.phases.supervision_prompts import (
    build_critique_system_prompt,
    build_critique_user_prompt,
    build_first_round_delegation_system_prompt,
    build_first_round_delegation_user_prompt,
    build_revision_system_prompt,
    build_revision_user_prompt,
)

logger = logging.getLogger(__name__)


def _get_lifecycle_fns() -> tuple:
    """Import LLM call functions from the supervision module.

    Uses function-level import to avoid circular imports and to ensure
    that any test patches applied to the supervision module's namespace
    are respected.
    """
    from foundry_mcp.core.research.workflows.deep_research.phases import (
        supervision as _sv,
    )

    return _sv.execute_llm_call, _sv.execute_structured_llm_call


async def first_round_decompose_critique_revise(
    workflow: Any,
    state: DeepResearchState,
    think_output: Optional[str],
    provider_id: Optional[str],
    timeout: float,
) -> tuple[list[ResearchDirective], bool, Optional[str]]:
    """Three-call pipeline for first-round query decomposition.

    Replaces the single-call first-round delegation with a pipeline that
    separates generation, critique, and revision into distinct LLM calls
    for higher-quality directives:

    1. **Generate** — decompose the query into initial directives (existing
       first-round prompts).
    2. **Critique** — evaluate the initial directives for redundancy,
       coverage gaps, proportionality, and specificity issues.
    3. **Revise** — apply the critique to produce the final directive set.
       Skipped if the critique finds no issues.

    Args:
        workflow: DeepResearchWorkflow instance (provides audit, cancellation,
            directive caps, and legacy parser).
        state: Current research state.
        think_output: Decomposition strategy from think step.
        provider_id: LLM provider to use.
        timeout: Request timeout.

    Returns:
        Tuple of (directives list, research_complete flag, raw content).
    """
    effective_provider = provider_id or state.supervision_provider

    # --- Call 1: Generate initial directives ---
    initial_directives, research_complete, gen_content, should_skip = (
        await run_first_round_generate(
            workflow,
            state,
            think_output,
            effective_provider,
            timeout,
        )
    )
    if should_skip:
        return initial_directives, research_complete, gen_content

    # --- Call 2: Critique the initial directives ---
    initial_count = len(initial_directives)
    directives_json = json.dumps(
        [
            {
                "research_topic": d.research_topic,
                "perspective": d.perspective,
                "evidence_needed": d.evidence_needed,
                "priority": d.priority,
            }
            for d in initial_directives
        ],
        indent=2,
    )

    critique_text, needs_revision, should_return_initial = (
        await run_first_round_critique(
            workflow,
            state,
            initial_count,
            directives_json,
            effective_provider,
            timeout,
        )
    )
    if should_return_initial:
        return initial_directives, research_complete, gen_content

    # --- Call 3: Revise (skip if critique found no issues) ---
    if not needs_revision:
        logger.info(
            "First-round critique found no issues, using initial directives",
        )
        workflow._write_audit_event(
            state,
            "first_round_decomposition",
            data={
                "initial_directive_count": initial_count,
                "final_directive_count": initial_count,
                "critique_triggered_revision": False,
            },
        )
        return initial_directives, research_complete, gen_content

    return await run_first_round_revise(
        workflow,
        state,
        initial_directives,
        initial_count,
        directives_json,
        critique_text,
        research_complete,
        effective_provider,
        timeout,
        gen_content=gen_content,
    )


async def run_first_round_generate(
    workflow: Any,
    state: DeepResearchState,
    think_output: Optional[str],
    effective_provider: Optional[str],
    timeout: float,
) -> tuple[list[ResearchDirective], bool, Optional[str], bool]:
    """Run the generation LLM call for first-round decomposition.

    Args:
        workflow: DeepResearchWorkflow instance.
        state: Current research state.
        think_output: Decomposition strategy from think step.
        effective_provider: Resolved LLM provider ID.
        timeout: Request timeout.

    Returns:
        Tuple of ``(initial_directives, research_complete, raw_content,
        should_skip)`` where *should_skip* is ``True`` when the caller
        should return early (research_complete or no directives).
    """
    _, execute_structured_llm_call = _get_lifecycle_fns()

    workflow._check_cancellation(state)
    gen_result = await execute_structured_llm_call(
        workflow=workflow,
        state=state,
        phase_name="supervision_delegate_generate",
        system_prompt=build_first_round_delegation_system_prompt(),
        user_prompt=build_first_round_delegation_user_prompt(
            state,
            think_output,
        ),
        provider_id=effective_provider,
        model=state.supervision_model,
        temperature=0.3,
        timeout=timeout,
        parse_fn=parse_delegation_response,
        role="delegation",
    )

    if isinstance(gen_result, WorkflowResult):
        logger.warning(
            "First-round generate call failed: %s",
            gen_result.error,
        )
        return [], False, None, True

    # Extract initial directives
    if gen_result.parsed is not None:
        gen_delegation: DelegationResponse = gen_result.parsed
        initial_directives = workflow._apply_directive_caps(
            gen_delegation.directives,
            state,
        )
        research_complete = gen_delegation.research_complete
    else:
        logger.warning(
            "First-round generate parse failed, falling back to legacy parser",
        )
        initial_directives, research_complete = workflow._parse_delegation_response(
            gen_result.result.content,
            state,
        )

    initial_count = len(initial_directives)

    workflow._write_audit_event(
        state,
        "first_round_generate",
        data={
            "provider_id": gen_result.result.provider_id,
            "model_used": gen_result.result.model_used,
            "tokens_used": gen_result.result.tokens_used,
            "directive_count": initial_count,
            "research_complete": research_complete,
            "directive_topics": [d.research_topic[:100] for d in initial_directives],
        },
    )

    # If generation signalled research_complete or produced no directives,
    # skip critique/revision — there's nothing to refine.
    should_skip = research_complete or not initial_directives
    if should_skip:
        workflow._write_audit_event(
            state,
            "first_round_decomposition",
            data={
                "initial_directive_count": initial_count,
                "final_directive_count": initial_count,
                "critique_triggered_revision": False,
                "skip_reason": ("research_complete" if research_complete else "no_directives"),
            },
        )

    return initial_directives, research_complete, gen_result.result.content, should_skip


async def run_first_round_critique(
    workflow: Any,
    state: DeepResearchState,
    initial_count: int,
    directives_json: str,
    effective_provider: Optional[str],
    timeout: float,
) -> tuple[str, bool, bool]:
    """Run the critique LLM call for first-round decomposition.

    Args:
        workflow: DeepResearchWorkflow instance.
        state: Current research state.
        initial_count: Number of directives from the generate step.
        directives_json: JSON-serialized directives for the critique prompt.
        effective_provider: Resolved LLM provider ID.
        timeout: Request timeout.

    Returns:
        Tuple of ``(critique_text, needs_revision, should_return_initial)``
        where *should_return_initial* is ``True`` when the critique call
        failed and the caller should fall back to the initial directives.
    """
    execute_llm_call, _ = _get_lifecycle_fns()

    workflow._check_cancellation(state)

    critique_result = await execute_llm_call(
        workflow=workflow,
        state=state,
        phase_name="supervision_delegate_critique",
        system_prompt=build_critique_system_prompt(),
        user_prompt=build_critique_user_prompt(
            state,
            directives_json,
        ),
        provider_id=effective_provider,
        model=state.supervision_model,
        temperature=0.2,
        timeout=getattr(
            workflow.config,
            "deep_research_reflection_timeout",
            60.0,
        ),
        role="reflection",
    )

    if isinstance(critique_result, WorkflowResult):
        logger.warning(
            "First-round critique call failed: %s. Using initial directives.",
            critique_result.error,
        )
        workflow._write_audit_event(
            state,
            "first_round_decomposition",
            data={
                "initial_directive_count": initial_count,
                "final_directive_count": initial_count,
                "critique_triggered_revision": False,
                "skip_reason": "critique_failed",
            },
        )
        return "", False, True

    critique_text = critique_result.result.content or ""

    # Detect whether the critique flagged any issues worth revising.
    needs_revision = critique_has_issues(critique_text)

    workflow._write_audit_event(
        state,
        "first_round_critique",
        data={
            "provider_id": critique_result.result.provider_id,
            "model_used": critique_result.result.model_used,
            "tokens_used": critique_result.result.tokens_used,
            "needs_revision": needs_revision,
            "critique_length": len(critique_text),
        },
    )

    return critique_text, needs_revision, False


async def run_first_round_revise(
    workflow: Any,
    state: DeepResearchState,
    initial_directives: list[ResearchDirective],
    initial_count: int,
    directives_json: str,
    critique_text: str,
    research_complete: bool,
    effective_provider: Optional[str],
    timeout: float,
    *,
    gen_content: Optional[str] = None,
) -> tuple[list[ResearchDirective], bool, Optional[str]]:
    """Run the revision LLM call for first-round decomposition.

    Args:
        workflow: DeepResearchWorkflow instance.
        state: Current research state.
        initial_directives: Directives from the generate step (fallback).
        initial_count: Number of initial directives.
        directives_json: JSON-serialized initial directives.
        critique_text: Critique output to inform revision.
        research_complete: Research-complete flag from generate step.
        effective_provider: Resolved LLM provider ID.
        timeout: Request timeout.
        gen_content: Raw content from the generate step (used as fallback
            when the revision call fails).

    Returns:
        Tuple of ``(final_directives, research_complete, raw_content)``.
    """
    _, execute_structured_llm_call = _get_lifecycle_fns()

    workflow._check_cancellation(state)
    revise_result = await execute_structured_llm_call(
        workflow=workflow,
        state=state,
        phase_name="supervision_delegate_revise",
        system_prompt=build_revision_system_prompt(),
        user_prompt=build_revision_user_prompt(
            state,
            directives_json,
            critique_text,
        ),
        provider_id=effective_provider,
        model=state.supervision_model,
        temperature=0.3,
        timeout=timeout,
        parse_fn=parse_delegation_response,
        role="delegation",
    )

    if isinstance(revise_result, WorkflowResult):
        logger.warning(
            "First-round revision call failed: %s. Using initial directives.",
            revise_result.error,
        )
        workflow._write_audit_event(
            state,
            "first_round_decomposition",
            data={
                "initial_directive_count": initial_count,
                "final_directive_count": initial_count,
                "critique_triggered_revision": True,
                "revision_failed": True,
            },
        )
        return initial_directives, research_complete, gen_content

    # Extract revised directives
    if revise_result.parsed is not None:
        rev_delegation: DelegationResponse = revise_result.parsed
        final_directives = workflow._apply_directive_caps(
            rev_delegation.directives,
            state,
        )
        research_complete = rev_delegation.research_complete
    else:
        logger.warning(
            "Revision parse failed, falling back to legacy parser",
        )
        final_directives, research_complete = workflow._parse_delegation_response(
            revise_result.result.content,
            state,
        )

    final_count = len(final_directives)

    workflow._write_audit_event(
        state,
        "first_round_revise",
        data={
            "provider_id": revise_result.result.provider_id,
            "model_used": revise_result.result.model_used,
            "tokens_used": revise_result.result.tokens_used,
            "directive_count": final_count,
            "directive_topics": [d.research_topic[:100] for d in final_directives],
        },
    )

    workflow._write_audit_event(
        state,
        "first_round_decomposition",
        data={
            "initial_directive_count": initial_count,
            "final_directive_count": final_count,
            "critique_triggered_revision": True,
            "directives_delta": final_count - initial_count,
        },
    )

    logger.info(
        "First-round decompose→critique→revise: %d → %d directives",
        initial_count,
        final_count,
    )

    # Return the revision's raw content for message accumulation
    raw_content = revise_result.result.content
    return final_directives, research_complete, raw_content
