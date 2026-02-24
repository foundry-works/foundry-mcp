"""Synthesis phase mixin for DeepResearchWorkflow.

Generates a comprehensive markdown report from analyzed findings and sources.
"""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any, Optional

from foundry_mcp.core.research.context_budget import AllocationResult
from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._budgeting import (
    allocate_synthesis_budget,
    final_fit_validate,
)
from foundry_mcp.core.research.workflows.deep_research._constants import (
    SYNTHESIS_OUTPUT_RESERVED,
)
from foundry_mcp.core.research.workflows.deep_research._helpers import (
    fidelity_level_from_score,
)
from foundry_mcp.core.research.workflows.deep_research.phases._citation_postprocess import (
    postprocess_citations,
)
from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
    execute_llm_call,
    finalize_phase,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query-type classification and structure guidance
# ---------------------------------------------------------------------------

# Keywords/patterns for classifying the intent of a research query.
_COMPARISON_PATTERNS = re.compile(
    r"\b(compar\w*|vs\.?|versus|differ\w*|contrast|between .+ and )\b",
    re.IGNORECASE,
)
_ENUMERATION_PATTERNS = re.compile(
    r"\b(list\b|top \d|best \d|options|alternatives|examples of|types of)",
    re.IGNORECASE,
)
_HOWTO_PATTERNS = re.compile(
    r"\b(how to|how do|steps to|guide to|tutorial|setup|install\w*|implement\w*|build\w*)\b",
    re.IGNORECASE,
)


def _classify_query_type(query: str) -> str:
    """Classify a research query into a structural type.

    Returns one of: ``"comparison"``, ``"enumeration"``, ``"howto"``,
    or ``"explanation"`` (the default).
    """
    if _COMPARISON_PATTERNS.search(query):
        return "comparison"
    if _ENUMERATION_PATTERNS.search(query):
        return "enumeration"
    if _HOWTO_PATTERNS.search(query):
        return "howto"
    return "explanation"


_STRUCTURE_GUIDANCE: dict[str, str] = {
    "comparison": """\
For **comparison** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Overview of [Subject A]
## Overview of [Subject B]
## Comparative Analysis
## Conclusions""",
    "enumeration": """\
For **list/enumeration** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## [Item 1]
## [Item 2]
## [Item N]
Each item should be its own section when depth is needed. For short lists a single section with a table or bullet list is acceptable.""",
    "howto": """\
For **how-to** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Prerequisites
## Step 1: [Action]
## Step 2: [Action]
## Step N: [Action]
## Conclusions""",
    "explanation": """\
For **explanation/overview** queries, use this structure:
# Research Report: [Topic]
## Executive Summary
## Key Findings
### [Theme/Category 1]
### [Theme/Category 2]
## Conclusions""",
}


class SynthesisPhaseMixin:
    """Synthesis phase methods. Mixed into DeepResearchWorkflow.

    At runtime, ``self`` is a DeepResearchWorkflow instance providing:
    - config, memory, hooks, orchestrator (instance attributes)
    - _write_audit_event(), _check_cancellation() (cross-cutting methods)
    - _execute_provider_async() (inherited from ResearchWorkflowBase)
    """

    config: Any
    memory: Any

    if TYPE_CHECKING:

        def _write_audit_event(self, *args: Any, **kwargs: Any) -> None: ...
        def _check_cancellation(self, *args: Any, **kwargs: Any) -> None: ...

    async def _execute_synthesis_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout: float,
    ) -> WorkflowResult:
        """Execute synthesis phase: generate comprehensive report from findings.

        This phase:
        1. Builds a synthesis prompt with all findings grouped by theme
        2. Includes source references for citation
        3. Generates a structured markdown report with:
           - Executive summary
           - Key findings organized by theme
           - Source citations
           - Knowledge gaps and limitations
           - Conclusions with actionable insights
        4. Stores the report in state.report

        Args:
            state: Current research state with findings from analysis
            provider_id: LLM provider to use
            timeout: Request timeout in seconds

        Returns:
            WorkflowResult with synthesis outcome
        """
        # Check if we have material to synthesize: either analysis findings
        # or per-topic compressed findings (Phase 3 PLAN — collapsed pipeline).
        has_compressed = any(
            tr.compressed_findings
            for tr in state.topic_research_results
        )
        if not state.findings and not state.compressed_digest and not has_compressed:
            logger.warning("No findings to synthesize")
            # Generate a minimal report even without findings
            state.report = self._generate_empty_report(state)
            self._write_audit_event(
                state,
                "synthesis_result",
                data={
                    "provider_id": None,
                    "model_used": None,
                    "tokens_used": None,
                    "duration_ms": None,
                    "system_prompt": None,
                    "user_prompt": None,
                    "raw_response": None,
                    "report": state.report,
                    "empty_report": True,
                },
                level="warning",
            )
            return WorkflowResult(
                success=True,
                content=state.report,
                metadata={
                    "research_id": state.id,
                    "finding_count": 0,
                    "empty_report": True,
                },
            )

        logger.info(
            "Starting synthesis phase: %d findings, %d sources, %d topic results with compressed findings",
            len(state.findings),
            len(state.sources),
            sum(1 for tr in state.topic_research_results if tr.compressed_findings),
        )

        # Emit phase.started audit event
        phase_start_time = time.perf_counter()
        self._write_audit_event(
            state,
            "phase.started",
            data={
                "phase_name": "synthesis",
                "iteration": state.iteration,
                "task_id": state.id,
            },
        )

        # Allocate token budget for findings and sources
        allocation_result = allocate_synthesis_budget(
            state=state,
            provider_id=provider_id,
        )

        # Update state with allocation metadata
        # Store overall fidelity in metadata (content_fidelity is now per-item dict)
        state.dropped_content_ids = allocation_result.dropped_ids
        allocation_dict = allocation_result.to_dict()
        allocation_dict["overall_fidelity_level"] = fidelity_level_from_score(allocation_result.fidelity)
        state.content_allocation_metadata = allocation_dict

        logger.info(
            "Synthesis budget allocation: %d items allocated, %d dropped, fidelity=%.1f%%",
            len(allocation_result.items),
            len(allocation_result.dropped_ids),
            allocation_result.fidelity * 100,
        )

        # Build the synthesis prompt with allocated content
        system_prompt = self._build_synthesis_system_prompt(state)
        user_prompt = self._build_synthesis_user_prompt(state, allocation_result)

        # Final-fit validation before provider dispatch
        valid, _preflight, system_prompt, user_prompt = final_fit_validate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider_id=provider_id or state.synthesis_provider,
            model=state.synthesis_model,
            output_reserved=SYNTHESIS_OUTPUT_RESERVED,
            phase="synthesis",
        )

        if not valid:
            logger.warning("Synthesis phase final-fit validation failed, proceeding with truncated prompts")

        # Check for cancellation before making provider call
        self._check_cancellation(state)

        # ---------------------------------------------------------
        # LLM call with phase-specific outer retry on token-limit
        # errors.  Each outer retry pre-truncates the user prompt
        # (oldest content first) before handing it back to
        # execute_llm_call (which has its own inner structural
        # retries).
        # ---------------------------------------------------------
        from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
            MAX_PHASE_TOKEN_RETRIES,
            _is_context_window_exceeded,
            truncate_prompt_for_retry,
        )

        current_user_prompt = user_prompt
        outer_retries = 0
        result = None

        for outer_attempt in range(MAX_PHASE_TOKEN_RETRIES + 1):  # 0 = initial
            call_result = await execute_llm_call(
                workflow=self,
                state=state,
                phase_name="synthesis",
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                provider_id=provider_id or state.synthesis_provider,
                model=state.synthesis_model,
                temperature=0.5,
                timeout=timeout,
                error_metadata={
                    "finding_count": len(state.findings),
                    "guidance": "Try reducing the number of findings or source content included",
                    "outer_retries": outer_retries,
                },
                role="report",
            )

            if isinstance(call_result, WorkflowResult):
                if (
                    _is_context_window_exceeded(call_result)
                    and outer_attempt < MAX_PHASE_TOKEN_RETRIES
                ):
                    outer_retries += 1
                    current_user_prompt = truncate_prompt_for_retry(
                        user_prompt, outer_retries, MAX_PHASE_TOKEN_RETRIES,
                    )
                    logger.warning(
                        "Synthesis outer retry %d/%d: pre-truncating user "
                        "prompt by %d%%",
                        outer_retries,
                        MAX_PHASE_TOKEN_RETRIES,
                        int((0.1 + outer_retries * 0.1) * 100),
                    )
                    continue

                # Non-retryable error or retries exhausted
                if outer_retries > 0:
                    self._write_audit_event(
                        state,
                        "synthesis_retry_exhausted",
                        data={
                            "outer_retries": outer_retries,
                            "error": call_result.error,
                        },
                        level="warning",
                    )
                return call_result

            # Success — record retry metadata if retries were needed
            if outer_retries > 0:
                self._write_audit_event(
                    state,
                    "synthesis_retry_succeeded",
                    data={"outer_retries": outer_retries},
                )
            result = call_result.result
            break

        if result is None:
            # All outer retries exhausted — should be caught above, safety net
            return WorkflowResult(
                success=False,
                content="",
                error="Synthesis failed: all token-limit retries exhausted",
                metadata={
                    "research_id": state.id,
                    "outer_retries": outer_retries,
                },
            )

        # Extract the markdown report from the response
        report = self._extract_markdown_report(result.content)

        if not report:
            logger.warning("Failed to extract report from synthesis response")
            # Use raw content as fallback
            report = result.content

        # Post-process citations: remove dangling refs, append Sources section
        report, citation_metadata = postprocess_citations(report, state)

        # Store report in state
        state.report = report

        # Save state
        self.memory.save_deep_research(state)
        synthesis_audit_data: dict[str, Any] = {
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "duration_ms": result.duration_ms,
            "report_length": len(state.report),
            "citation_postprocess": citation_metadata,
        }
        if self.config.audit_verbosity == "full":
            synthesis_audit_data["system_prompt"] = system_prompt
            synthesis_audit_data["user_prompt"] = user_prompt
            synthesis_audit_data["raw_response"] = result.content
            synthesis_audit_data["report"] = state.report
        else:
            synthesis_audit_data["system_prompt_length"] = len(system_prompt)
            synthesis_audit_data["user_prompt_length"] = len(user_prompt)
            synthesis_audit_data["raw_response_length"] = len(result.content)
        self._write_audit_event(
            state,
            "synthesis_result",
            data=synthesis_audit_data,
        )

        logger.info(
            "Synthesis phase complete: report length %d chars",
            len(state.report),
        )

        finalize_phase(self, state, "synthesis", phase_start_time)

        return WorkflowResult(
            success=True,
            content=state.report,
            provider_id=result.provider_id,
            model_used=result.model_used,
            tokens_used=result.tokens_used,
            duration_ms=result.duration_ms,
            metadata={
                "research_id": state.id,
                "finding_count": len(state.findings),
                "source_count": len(state.sources),
                "report_length": len(state.report),
                "iteration": state.iteration,
            },
        )

    def _build_synthesis_system_prompt(self, state: DeepResearchState) -> str:
        """Build system prompt for report synthesis.

        The prompt is state-aware: it detects query language and adapts
        structural guidance based on query type.

        Args:
            state: Current research state used for language and structure hints

        Returns:
            System prompt string
        """
        query_type = _classify_query_type(state.original_query)
        structure_guidance = _STRUCTURE_GUIDANCE.get(query_type, _STRUCTURE_GUIDANCE["explanation"])

        return f"""You are a research synthesizer. Your task is to create a comprehensive, well-structured research report from analyzed findings.

## Language

Detect the language of the user's research query. Write the entire report in that same language. If the query is in English, write in English. If the query is in Chinese, write entirely in Chinese. Match the user's language exactly.

## Report Structure

Select a report structure suited to the query type. A structural hint is provided in the user prompt.

{structure_guidance}

These are suggestions. Section is a fluid concept — you can structure your report however you think is best, including in ways not listed above. Make sure sections are cohesive and make sense for the reader.

Include analysis of Conflicting Information and Limitations where they exist, but integrate them naturally into the relevant sections rather than forcing separate subsections. A Conclusions section with actionable insights is always valuable.

## Section Writing Rules

- Use ## for each section title (Markdown format).
- Write in paragraph form by default; use bullet points only when listing discrete items.
- Each section should be as long as necessary to deeply answer the question with the information gathered. Sections are expected to be thorough and detailed. You are writing a deep research report and users expect comprehensive answers.
- Do not refer to yourself or comment on the report itself — just write the report.

## Writing Quality

- Write directly and authoritatively. Do not hedge with openers like "it appears that", "it seems", or "based on available information".
- Never use meta-commentary about the report itself ("based on the research", "the findings show", "this report examines").
- Never refer to yourself ("as an AI", "I found that", "in my analysis").
- Use clear, professional language.
- Include all relevant findings. Do not omit information for brevity.

## Citations

- For the **first reference** to each source, use an inline markdown link followed by its citation number: `[Title](URL) [N]`. This makes sources immediately navigable in rendered markdown.
- For **subsequent references** to the same source, use just the citation number: `[N]`.
- The citation numbers correspond to the numbered sources provided in the input.
- Do NOT generate a Sources or References section — it will be appended automatically.
- Distinguish between high-confidence findings (well-supported) and lower-confidence insights.
- Citations are extremely important. Pay careful attention to getting these right. Users will often use citations to find more information on specific points.

IMPORTANT: Return ONLY the markdown report, no preamble or meta-commentary.

REMEMBER: The research and brief may be in English, but the final report MUST be written in the same language as the user's original query. This is critical — the user will only understand the answer if it matches their input language."""

    def _build_synthesis_user_prompt(
        self,
        state: DeepResearchState,
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Build user prompt with findings and sources for synthesis.

        Args:
            state: Current research state
            allocation_result: Optional budget allocation result for token-aware prompts

        Returns:
            User prompt string
        """
        # Build source_id → citation_number mapping for inline references
        id_to_citation = state.source_id_to_citation()

        prompt_parts = [
            f"# Research Query\n{state.original_query}",
            "",
            f"## Research Brief\n{state.research_brief or 'Direct research on the query'}",
            "",
        ]

        # When a global compressed digest is available, use it as the
        # primary findings source.  The digest already contains deduplicated
        # cross-topic findings with consistent citations, contradictions,
        # and gaps — so we can skip the per-finding enumeration below.
        if state.compressed_digest:
            prompt_parts.extend([
                "## Unified Research Digest",
                "",
                state.compressed_digest,
                "",
            ])
            # Still include the source reference and instructions below
            # (skip to source reference section)
            return self._build_synthesis_tail(
                state, prompt_parts, id_to_citation, allocation_result,
            )

        # ---------------------------------------------------------------
        # Phase 3 PLAN: Direct-from-compressed-findings path.
        #
        # When the ANALYSIS phase is skipped (collapsed pipeline), there
        # are no analysis findings but per-topic compressed_findings exist.
        # Build the synthesis prompt from those directly.
        # ---------------------------------------------------------------
        compressed_topics = [
            tr for tr in state.topic_research_results
            if tr.compressed_findings
        ]
        if not state.findings and compressed_topics:
            prompt_parts.extend([
                "## Research Findings by Topic",
                "",
            ])
            for tr in compressed_topics:
                # Resolve sub-query text for section header
                sq = state.get_sub_query(tr.sub_query_id)
                topic_label = sq.query if sq else tr.sub_query_id
                prompt_parts.append(f"### {topic_label}")
                prompt_parts.append("")
                prompt_parts.append(tr.compressed_findings or "")
                prompt_parts.append("")

            # Add contradictions and gaps (may still be populated from supervision)
            self._append_contradictions_and_gaps(state, prompt_parts, id_to_citation)

            return self._build_synthesis_tail(
                state, prompt_parts, id_to_citation, allocation_result,
            )

        # ---------------------------------------------------------------
        # Standard path: build from analysis findings.
        # ---------------------------------------------------------------
        prompt_parts.extend([
            "## Findings to Synthesize",
            "",
        ])

        # Group findings by category if available
        categorized: dict[str, list] = {}

        for finding in state.findings:
            category = finding.category or "General"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(finding)

        # Add findings by category - findings are protected, always included at full fidelity
        for category, findings in categorized.items():
            prompt_parts.append(f"### {category}")
            for f in findings:
                confidence_label = f.confidence.value if hasattr(f.confidence, "value") else str(f.confidence)
                # Map source IDs to citation numbers
                citation_refs = [f"[{id_to_citation[sid]}]" for sid in f.source_ids if sid in id_to_citation]
                source_refs = ", ".join(citation_refs) if citation_refs else "no sources"
                prompt_parts.append(f"- [{confidence_label.upper()}] {f.content}")
                prompt_parts.append(f"  Sources: {source_refs}")
            prompt_parts.append("")

        # Add detected contradictions and knowledge gaps
        self._append_contradictions_and_gaps(state, prompt_parts, id_to_citation)

        return self._build_synthesis_tail(
            state, prompt_parts, id_to_citation, allocation_result,
        )

    def _append_contradictions_and_gaps(
        self,
        state: DeepResearchState,
        prompt_parts: list[str],
        id_to_citation: dict[str, int],
    ) -> None:
        """Append contradiction and gap sections to the synthesis prompt.

        Shared by both the compressed-findings path and the standard
        findings path.

        Args:
            state: Current research state
            prompt_parts: Accumulated prompt sections (mutated in-place)
            id_to_citation: source-id → citation-number mapping
        """
        if state.contradictions:
            prompt_parts.append("## Contradictions Detected")
            prompt_parts.append(
                "The following contradictions were identified between findings. "
                "Address these explicitly in the report's 'Conflicting Information' section."
            )
            for contradiction in state.contradictions:
                severity_label = contradiction.severity.upper()
                prompt_parts.append(f"- [{severity_label}] {contradiction.description}")
                prompt_parts.append(f"  Conflicting findings: {', '.join(contradiction.finding_ids)}")
                if contradiction.resolution:
                    prompt_parts.append(f"  Suggested resolution: {contradiction.resolution}")
                if contradiction.preferred_source_id:
                    cn = id_to_citation.get(contradiction.preferred_source_id)
                    if cn is not None:
                        prompt_parts.append(f"  Preferred source: [{cn}]")
            prompt_parts.append("")

        if state.gaps:
            prompt_parts.append("## Knowledge Gaps Identified")
            for gap in state.gaps:
                status = "addressed" if gap.resolved else "unresolved"
                prompt_parts.append(f"- [{status}] {gap.description}")
            prompt_parts.append("")

    def _build_synthesis_tail(
        self,
        state: DeepResearchState,
        prompt_parts: list[str],
        id_to_citation: dict[str, int],
        allocation_result: Optional[AllocationResult] = None,
    ) -> str:
        """Append source reference and instructions to the synthesis prompt.

        Shared by both the standard findings path and the compressed-digest
        path in ``_build_synthesis_user_prompt``.

        Args:
            state: Current research state
            prompt_parts: Accumulated prompt sections (mutated in-place)
            id_to_citation: source-id → citation-number mapping
            allocation_result: Optional budget allocation result

        Returns:
            Complete user prompt string
        """
        # Add source reference list with citation numbers - use allocation-aware content
        prompt_parts.append("## Source Reference (use these citation numbers in your report)")

        if allocation_result:
            # Use allocated sources in priority order, applying token limits
            for item in allocation_result.items:
                # Skip findings (they're in the findings section)
                if not item.id.startswith("src-"):
                    continue

                source = next((s for s in state.sources if s.id == item.id), None)
                if not source:
                    continue

                cn = source.citation_number
                label = f"[{cn}]" if cn is not None else f"[{source.id}]"
                quality = source.quality.value if hasattr(source.quality, "value") else str(source.quality)
                prompt_parts.append(f"- **{label}**: {source.title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {source.url}")

                # Apply token-aware content limit for snippets
                if item.needs_summarization:
                    # Compressed: use allocated tokens to estimate character limit (~4 chars/token)
                    char_limit = max(50, item.allocated_tokens * 4)
                    if source.snippet:
                        snippet = source.snippet[:char_limit]
                        if len(source.snippet) > char_limit:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")
                else:
                    # Full fidelity: include snippet up to 200 chars
                    if source.snippet:
                        snippet = source.snippet[:200]
                        if len(source.snippet) > 200:
                            snippet += "..."
                        prompt_parts.append(f"  Snippet: {snippet}")

            # Note dropped sources if any
            if allocation_result.dropped_ids:
                dropped_sources = [sid for sid in allocation_result.dropped_ids if sid.startswith("src-")]
                if dropped_sources:
                    prompt_parts.append(
                        f"\n*Note: {len(dropped_sources)} additional source(s) omitted for context limits*"
                    )
        else:
            # Fallback: use first 30 sources (legacy behavior)
            for source in state.sources[:30]:
                cn = source.citation_number
                label = f"[{cn}]" if cn is not None else f"[{source.id}]"
                quality = source.quality.value if hasattr(source.quality, "value") else str(source.quality)
                prompt_parts.append(f"- {label}: {source.title} [{quality}]")
                if source.url:
                    prompt_parts.append(f"  URL: {source.url}")

        prompt_parts.append("")

        # Add synthesis instructions with query-type structural hint
        query_type = _classify_query_type(state.original_query)
        query_type_labels = {
            "comparison": "comparison (side-by-side analysis of alternatives)",
            "enumeration": "list/enumeration (discrete items or options)",
            "howto": "how-to (step-by-step procedural guide)",
            "explanation": "explanation/overview (topical deep-dive)",
        }
        type_label = query_type_labels.get(query_type, query_type)

        prompt_parts.extend(
            [
                "## Instructions",
                f"Generate a comprehensive research report addressing the query: '{state.original_query}'",
                "",
                f"**Query type hint:** {type_label} — adapt the report structure accordingly.",
                "",
                f"This is iteration {state.iteration} of {state.max_iterations}.",
                f"Total findings: {len(state.findings)}",
                f"Total sources: {len(state.sources)}",
                f"Unresolved gaps: {len(state.unresolved_gaps())}",
                "",
                "Create a well-structured markdown report following the format specified.",
            ]
        )

        return "\n".join(prompt_parts)

    def _extract_markdown_report(self, content: str) -> Optional[str]:
        """Extract markdown report from LLM response.

        The response should be pure markdown, but this handles cases where
        the LLM wraps it in code blocks or adds preamble.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted markdown report or None if extraction fails
        """
        if not content:
            return None

        # If content starts with markdown heading, it's likely clean
        if content.strip().startswith("#"):
            return content.strip()

        # Check for markdown code block wrapper
        if "```markdown" in content or "```md" in content:
            # Extract content between code blocks
            pattern = r"```(?:markdown|md)?\s*([\s\S]*?)```"
            matches = re.findall(pattern, content)
            if matches:
                return matches[0].strip()

        # Check for generic code block
        if "```" in content:
            pattern = r"```\s*([\s\S]*?)```"
            matches = re.findall(pattern, content)
            for match in matches:
                # Check if it looks like markdown (has headings)
                if match.strip().startswith("#") or "##" in match:
                    return match.strip()

        # Look for first heading and take everything from there
        heading_match = re.search(r"^(#[^\n]+)", content, re.MULTILINE)
        if heading_match:
            start_pos = heading_match.start()
            return content[start_pos:].strip()

        # If nothing else works, return the trimmed content
        return content.strip() if len(content.strip()) > 50 else None

    def _generate_empty_report(self, state: DeepResearchState) -> str:
        """Generate a minimal report when no findings are available.

        Args:
            state: Current research state

        Returns:
            Minimal markdown report
        """
        return f"""# Research Report

## Executive Summary

Research was conducted on the query: "{state.original_query}"

Unfortunately, the analysis phase did not yield extractable findings from the gathered sources. This may indicate:
- The sources lacked relevant information
- The query may need refinement
- Additional research iterations may be needed

## Research Query

{state.original_query}

## Research Brief

{state.research_brief or "No research brief generated."}

## Sources Examined

{len(state.sources)} source(s) were examined during this research session.

## Recommendations

1. Consider refining the research query for more specific results
2. Try additional research iterations if available
3. Review the gathered sources manually for relevant information

---

*Report generated with no extractable findings. Iteration {state.iteration}/{state.max_iterations}.*
"""
