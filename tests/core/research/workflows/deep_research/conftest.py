"""Shared test fixtures and helpers for deep research tests.

Provides a canonical ``make_test_state()`` factory so that all test files
build ``DeepResearchState`` objects consistently.  File-specific helpers
can wrap this function to add phase-specific defaults.
"""

from __future__ import annotations

from typing import Any

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.sources import (
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)


def make_test_state(
    *,
    id: str = "deepres-test",
    query: str = "What are the benefits of renewable energy?",
    research_brief: str | None = "Detailed investigation of the research topic",
    phase: DeepResearchPhase = DeepResearchPhase.BRIEF,
    iteration: int = 1,
    max_iterations: int = 3,
    max_sources_per_query: int = 5,
    num_sub_queries: int = 0,
    num_pending_sub_queries: int = 0,
    sources_per_query: int = 0,
    supervision_round: int = 0,
    max_supervision_rounds: int = 6,
    system_prompt: str | None = None,
    clarification_constraints: dict[str, str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> DeepResearchState:
    """Create a ``DeepResearchState`` pre-populated for testing.

    Parameters
    ----------
    id:
        Stable test identifier (avoids UUID randomness in assertions).
    query:
        ``original_query`` value.
    research_brief:
        Optional research brief to set on the state.
    phase:
        Starting phase.
    iteration, max_iterations:
        Iteration control fields.
    max_sources_per_query:
        Cap on sources per sub-query.
    num_sub_queries:
        Number of *completed* sub-queries to pre-populate (with sources).
    num_pending_sub_queries:
        Number of *pending* sub-queries to append.
    sources_per_query:
        How many ``ResearchSource`` objects to create per completed sub-query.
    supervision_round, max_supervision_rounds:
        Supervision loop control.
    system_prompt:
        Optional system prompt.
    clarification_constraints:
        Optional clarification constraints dict.
    metadata:
        Extra metadata keys to merge into state.metadata.
    """
    state = DeepResearchState(
        id=id,
        original_query=query,
        research_brief=research_brief,
        phase=phase,
        iteration=iteration,
        max_iterations=max_iterations,
        max_sources_per_query=max_sources_per_query,
        supervision_round=supervision_round,
        max_supervision_rounds=max_supervision_rounds,
    )
    if system_prompt is not None:
        state.system_prompt = system_prompt
    if clarification_constraints is not None:
        state.clarification_constraints = clarification_constraints
    if metadata:
        state.metadata.update(metadata)

    # Completed sub-queries with sources
    for i in range(num_sub_queries):
        sq = SubQuery(
            id=f"sq-{i}",
            query=f"Sub-query {i}: aspect {i} of the topic",
            rationale=f"Rationale {i}",
            priority=i + 1,
            status="completed",
        )
        state.sub_queries.append(sq)
        for j in range(sources_per_query):
            src = ResearchSource(
                id=f"src-{i}-{j}",
                title=f"Source {j} for sq-{i}",
                url=f"https://example{j}.com/sq-{i}/{j}",
                content=f"Content about topic sq-{i}, finding {j}.",
                quality=SourceQuality.HIGH if j == 0 else SourceQuality.MEDIUM,
                source_type=SourceType.WEB,
                sub_query_id=sq.id,
            )
            state.append_source(src)

    # Pending sub-queries (no sources)
    for i in range(num_pending_sub_queries):
        state.sub_queries.append(
            SubQuery(
                id=f"sq-pending-{i}",
                query=f"Pending query {i}",
                rationale=f"Pending rationale {i}",
                priority=num_sub_queries + i + 1,
                status="pending",
            )
        )

    return state
