"""Brief phase mixin for DeepResearchWorkflow.

Transforms the raw user query into a structured research brief that
maximises specificity, fills unstated dimensions as open-ended, specifies
source preferences, and defines scope boundaries.  The brief drives all
downstream phases (planning, supervision, synthesis).

Adapted from open_deep_research's ``write_research_brief`` pattern.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.research.models.deep_research import (
    BUILTIN_PROFILES,
    DeepResearchState,
    ResearchBriefOutput,
    ResearchProfile,
    parse_brief_output,
)
from foundry_mcp.core.research.models.sources import ResearchMode
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (
    build_sanitized_context,
    sanitize_external_content,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_structured_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)


class BriefPhaseMixin:
    """Brief phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)

    See ``DeepResearchWorkflowProtocol`` in ``_protocols.py`` for the
    full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory

    # Stubs for Pyright — canonical signatures live in _protocols.py
    if TYPE_CHECKING:
        from foundry_mcp.core.research.models.deep_research import DeepResearchState as _S

        def _write_audit_event(
            self, state: _S | None, event_name: str, *, data: dict[str, Any] | None = ..., level: str = ...
        ) -> None: ...
        def _check_cancellation(self, state: _S) -> None: ...

    async def _execute_brief_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute brief phase: enrich the raw query into a research brief.

        Transforms the user's original query (plus any clarification
        constraints) into a detailed, structured research brief paragraph
        that maximises specificity and drives all downstream phases.

        Non-fatal: if brief generation fails, logs a warning and proceeds
        with the original query as-is.

        Args:
            state: Current research state
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with brief generation outcome
        """
        logger.info("Starting brief phase for query: %s", state.original_query[:100])

        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "brief",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        system_prompt = self._build_brief_system_prompt(state.research_profile)
        user_prompt = self._build_brief_user_prompt(state)

        self._check_cancellation(state)

        # Use structured output parsing with automatic retry on parse failure.
        # parse_brief_output() accepts both JSON and plain-text responses
        # for backward compatibility.
        call_result = await execute_structured_llm_call(
            workflow=self,
            state=state,
            phase_name="brief",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id,
            model=None,  # Resolved by role
            temperature=0.4,  # Moderate creativity — enrichment, not invention
            timeout=timeout,
            parse_fn=parse_brief_output,
            role="brief",  # Research-tier via _ROLE_RESOLUTION_CHAIN
        )

        if isinstance(call_result, WorkflowResult):
            # Brief generation failed — non-fatal, proceed with original query
            logger.warning(
                "Brief generation LLM call failed for research %s, proceeding with original query: %s",
                state.id,
                call_result.error,
            )
            self._write_audit_event(
                state,
                "brief_generation_failed",
                data={
                    "error": call_result.error,
                    "fallback": "original_query",
                },
                level="warning",
            )
            # Do NOT set state.research_brief — leave it as None so
            # planning's inline refinement sub-step can still run as fallback
            finalize_phase(self, state, "brief", phase_start_time)
            return WorkflowResult(
                success=True,  # Non-fatal failure
                content="Brief generation failed, using original query",
                metadata={
                    "research_id": state.id,
                    "brief_generated": False,
                    "fallback": "original_query",
                },
            )

        result = call_result.result

        # Extract brief text from structured parse or raw content
        if call_result.parsed is not None:
            brief_output: ResearchBriefOutput = call_result.parsed
            brief_text = brief_output.research_brief.strip()
        else:
            brief_text = (result.content or "").strip()

        if brief_text:
            state.research_brief = brief_text
            logger.info(
                "Brief generation complete (%d chars) for research %s",
                len(brief_text),
                state.id,
            )
        else:
            logger.warning(
                "Brief generation returned empty content for research %s, proceeding with original query",
                state.id,
            )

        # PLAN-2 Item 5: Adaptive provider selection from brief signals
        effective_brief = brief_text or state.original_query
        provider_hints: list[str] = []
        active_providers: list[str] = []
        if state.research_profile is not None:
            provider_hints = self._extract_provider_hints(effective_brief, state.research_profile)
            active_providers = self._apply_provider_hints(state, provider_hints)
        if provider_hints:
            logger.info(
                "Adaptive provider selection: hints=%s, active=%s (research %s)",
                provider_hints,
                active_providers,
                state.id,
            )

        # Persist state with the generated brief
        self.memory.save_deep_research(state)

        self._write_audit_event(
            state,
            "brief_result",
            data={
                "provider_id": result.provider_id,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
                "brief_length": len(brief_text),
                "brief_generated": bool(brief_text),
                "research_brief": state.research_brief,
                "provider_hints": provider_hints,
                "active_providers": active_providers,
            },
        )

        # PLAN-2 Item 5: Log adaptive provider selection provenance event
        if state.provenance is not None and provider_hints:
            state.provenance.append(
                phase="brief",
                event_type="provider_hints_extracted",
                summary=f"Adaptive provider selection: {len(provider_hints)} hint(s) extracted from brief",
                provider_hints=provider_hints,
                active_providers=active_providers,
                profile_name=state.research_profile.name if state.research_profile else "general",
            )

        # PLAN-1 Item 2: Log brief_generated provenance event
        if state.provenance is not None and brief_text:
            state.provenance.append(
                phase="brief",
                event_type="brief_generated",
                summary=f"Research brief generated ({len(brief_text)} chars)",
                brief_length=len(brief_text),
                provider_id=result.provider_id,
                model_used=result.model_used,
                tokens_used=result.tokens_used,
            )

        finalize_phase(self, state, "brief", phase_start_time)

        return WorkflowResult(
            success=True,
            content=state.research_brief or "Brief phase complete",
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "brief_generated": bool(brief_text),
                "brief_length": len(brief_text),
            },
        )

    def _build_brief_system_prompt(
        self,
        profile: Optional[ResearchProfile] = None,
    ) -> str:
        """Build system prompt for research brief generation.

        Adapts open_deep_research's ``transform_messages_into_research_topic_prompt``
        approach: maximise specificity, prefer primary sources, fill unstated
        dimensions as open-ended, avoid unwarranted assumptions.

        When an academic profile is active, appends instructions to probe for
        discipline, education level, time period, and methodology preferences —
        dimensions that fundamentally shape a literature review.

        Args:
            profile: Active research profile (used to inject academic dimensions)

        Returns:
            System prompt string
        """
        base = (
            "You are a research brief writer. Your task is to transform a "
            "user's research request into a detailed, structured research "
            "brief that will drive a multi-phase deep research workflow.\n\n"
            "Your brief MUST:\n"
            "1. **Maximise specificity**: Extract every concrete detail the "
            "user provided (names, dates, versions, quantities, geographic "
            "scope) and foreground them prominently.\n"
            "2. **Fill unstated dimensions as open-ended**: When the user "
            "leaves a dimension unspecified (time period, geography, "
            "methodology), explicitly note it as an open question the "
            "research should address — do NOT assume a value.\n"
            "3. **Specify source preferences**: Bias toward primary and "
            "official sources (specifications, documentation, peer-reviewed "
            "work, government datasets, original research papers) over "
            "aggregators or secondary commentary. Domain-specific guidance:\n"
            "   - Product/travel research → prefer official or primary "
            "websites, manufacturer pages, and reputable e-commerce sites "
            "for user reviews — not aggregator sites or SEO-heavy blogs.\n"
            "   - Academic/scientific queries → prefer the original paper or "
            "official journal publication — not survey papers or secondary "
            "summaries.\n"
            "   - People research → prefer LinkedIn profiles or personal "
            "websites.\n"
            "   - Language-specific queries → prioritize sources published "
            "in that language.\n"
            "   If the user's query is written in a non-English language, "
            "prioritize sources published in that language.\n"
            "4. **Define scope boundaries**: State what the research should "
            "include AND what it should exclude to prevent scope creep.\n"
            "5. **Use the first person**: Phrase the brief from the "
            "perspective of the user (e.g. 'I am looking for…' rather "
            "than 'Research the following topic…'). This preserves the "
            "user's voice and helps downstream researchers understand "
            "intent.\n\n"
        )

        # PLAN-1 Item 5a: Academic brief enrichment
        if profile is not None and profile.source_quality_mode == ResearchMode.ACADEMIC:
            base += self._build_academic_brief_instructions(profile)

        base += (
            "Output your response as a JSON object with this schema:\n"
            "{\n"
            '  "research_brief": "Complete brief as one or two well-structured paragraphs",\n'
            '  "scope_boundaries": "What the research should include and exclude" or null,\n'
            '  "source_preferences": "Preferred source types" or null\n'
            "}\n\n"
            "IMPORTANT: Return ONLY valid JSON, no markdown formatting or extra text."
        )
        return base

    @staticmethod
    def _build_academic_brief_instructions(profile: ResearchProfile) -> str:
        """Build academic enrichment instructions for the brief system prompt.

        Appends dimensions that fundamentally shape a literature review:
        disciplinary scope, time period, methodology preferences, education
        level/population, and source type hierarchy.

        When the profile has pre-specified constraints (e.g. disciplinary_scope,
        time_period), they are injected as pre-filled values so the brief
        writer incorporates them rather than leaving them open-ended.

        Args:
            profile: Active academic research profile

        Returns:
            Instruction block to append to the base system prompt
        """
        parts: list[str] = [
            "**ACADEMIC RESEARCH MODE**: This is an academic research request. "
            "In addition to the general requirements above, your brief MUST "
            "address the following dimensions:\n\n"
            "6. **Disciplinary scope**: Identify the primary discipline(s) and "
            "any relevant interdisciplinary connections. If the topic spans "
            "multiple fields, note which disciplinary perspective should be "
            "primary and which are secondary.\n"
            "7. **Time period**: Specify the temporal scope — both foundational "
            "works (seminal papers that established the field) and the recency "
            "window for current literature (e.g. last 5-10 years). If the user "
            "does not specify a time period, default to covering both foundational "
            "and recent work.\n"
            "8. **Methodology preferences**: Note preferred research methodologies "
            "(quantitative, qualitative, mixed methods, meta-analysis, theoretical/"
            "conceptual, case study, experimental). If not specified by the user, "
            "leave this as an open dimension.\n"
            "9. **Education level / population**: If the research concerns a specific "
            "population (e.g. K-12 students, clinical patients, organizational "
            "employees), specify it. If not applicable, omit.\n"
            "10. **Source type hierarchy**: Prioritize sources in this order: "
            "peer-reviewed journal articles > systematic reviews and meta-analyses > "
            "academic books and monographs > preprints and working papers > "
            "institutional reports. Deprioritize blogs, news articles, and "
            "Wikipedia.\n\n"
        ]

        # Inject profile-specified constraints as pre-filled values
        constraints: list[str] = []
        if profile.disciplinary_scope:
            constraints.append(
                f"- **Disciplinary scope (pre-specified)**: {', '.join(profile.disciplinary_scope)}"
            )
        if profile.time_period:
            constraints.append(
                f"- **Time period (pre-specified)**: {profile.time_period}"
            )
        if profile.methodology_preferences:
            constraints.append(
                f"- **Methodology preferences (pre-specified)**: {', '.join(profile.methodology_preferences)}"
            )
        if profile.source_type_hierarchy:
            constraints.append(
                f"- **Source type hierarchy (pre-specified)**: {' > '.join(profile.source_type_hierarchy)}"
            )

        if constraints:
            parts.append(
                "The following constraints have been pre-specified by the research "
                "profile. Incorporate them into the brief as fixed parameters rather "
                "than open questions:\n"
            )
            parts.extend(constraints)
            parts.append("\n")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # PLAN-2 Item 5: Adaptive Provider Selection
    # ------------------------------------------------------------------

    # Discipline keyword → provider hint mapping.  Keywords are matched
    # case-insensitively against the brief text.  Each tuple maps a set
    # of discipline signals to a recommended search provider.
    _DISCIPLINE_PROVIDER_MAP: list[tuple[list[str], str]] = [
        (
            # No PubMed MCP integration — route biomedical queries to
            # Semantic Scholar which covers biomedical literature via PubMed IDs.
            ["biomedical", "clinical", "health", "medical", "medicine", "epidemiology", "pharmaceutical"],
            "semantic_scholar",
        ),
        (
            ["computer science", "machine learning", "artificial intelligence", "deep learning", "natural language processing"],
            "semantic_scholar",
        ),
        (
            ["education", "pedagogy", "curriculum", "teaching", "learning outcomes"],
            "openalex",
        ),
        (
            ["social science", "sociology", "economics", "political science", "anthropology"],
            "openalex",
        ),
    ]

    @staticmethod
    def _extract_provider_hints(
        brief_text: str,
        profile: ResearchProfile,
    ) -> list[str]:
        """Extract provider hints from discipline signals in the brief.

        Scans the research brief for discipline keywords and maps them to
        recommended search providers.  This enables the pipeline to
        dynamically add discipline-specific providers when the brief
        reveals an academic domain.

        Args:
            brief_text: The generated research brief text.
            profile: Active research profile (used for disciplinary_scope).

        Returns:
            Deduplicated list of provider name hints (may include providers
            not currently available — caller must filter).
        """
        hints: list[str] = []
        text_lower = brief_text.lower()

        # Check brief text for discipline signals using word boundaries
        # to avoid false positives (e.g. "health" matching "healthy")
        for keywords, provider in BriefPhaseMixin._DISCIPLINE_PROVIDER_MAP:
            for keyword in keywords:
                if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                    if provider not in hints:
                        hints.append(provider)
                    break  # One keyword match per discipline group is sufficient

        # Also check profile.disciplinary_scope if specified
        if profile.disciplinary_scope:
            scope_text = " ".join(profile.disciplinary_scope).lower()
            for keywords, provider in BriefPhaseMixin._DISCIPLINE_PROVIDER_MAP:
                for keyword in keywords:
                    if re.search(rf"\b{re.escape(keyword)}\b", scope_text):
                        if provider not in hints:
                            hints.append(provider)
                        break

        return hints

    @staticmethod
    def _apply_provider_hints(
        state: DeepResearchState,
        hints: list[str],
        known_providers: frozenset[str] | None = None,
    ) -> list[str]:
        """Apply extracted provider hints to the session state.

        Merges hints additively with the profile's existing provider list.
        Hints are only applied when the profile uses built-in defaults
        (i.e., ``profile.name`` is in ``BUILTIN_PROFILES``).  Custom
        profiles with explicitly chosen providers are not modified.

        Unknown or unavailable provider hints are silently dropped.

        Args:
            state: Current research state (reads profile, writes metadata).
            hints: Provider name hints from ``_extract_provider_hints``.
            known_providers: Set of valid provider names.  When ``None``,
                defaults to the standard set of registered providers.

        Returns:
            The resolved active provider list stored in state.metadata.
        """
        if known_providers is None:
            known_providers = frozenset(
                {"tavily", "semantic_scholar", "google", "openalex", "crossref", "perplexity"}
            )

        profile = state.research_profile

        # Store raw hints in metadata regardless of application
        state.metadata["provider_hints"] = hints

        # Guard: legacy sessions (pre-PLAN-1) may have research_profile=None
        if profile is None:
            state.metadata["active_providers"] = []
            return []

        # Start with the profile's current provider list
        active: list[str] = list(profile.providers)

        # Only augment built-in profiles — custom profiles have explicit
        # provider choices that should not be overridden.
        is_builtin = profile.name in BUILTIN_PROFILES
        if is_builtin and hints:
            for hint in hints:
                # Silently drop unknown/unavailable providers
                if hint in known_providers and hint not in active:
                    active.append(hint)

        state.metadata["active_providers"] = active
        return active

    def _build_brief_user_prompt(self, state: DeepResearchState) -> str:
        """Build user prompt for research brief generation.

        Includes the original query, any clarification constraints from the
        CLARIFICATION phase, the current date for temporal context, and
        optional system prompt context.

        Args:
            state: Current research state

        Returns:
            User prompt string
        """
        ctx = build_sanitized_context(state)
        parts: list[str] = [
            f"Research request:\n{ctx['original_query']}",
        ]

        if state.system_prompt:
            parts.append(f"\nAdditional context provided by the user:\n{ctx['system_prompt']}")

        if state.clarification_constraints:
            parts.append("\nClarification constraints (already confirmed with the user):")
            for key, value in state.clarification_constraints.items():
                parts.append(f"- {sanitize_external_content(key)}: {sanitize_external_content(value)}")

        # Temporal context helps the brief scope time-sensitive research
        parts.append(f"\nCurrent date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")

        return "\n".join(parts)
