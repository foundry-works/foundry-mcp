"""Async workflow execution engine for deep research.

Orchestrates the multi-phase workflow (clarification, brief, supervision,
synthesis) with cancellation support, error handling, and resource cleanup.

GATHERING is a legacy-resume-only phase — new workflows jump directly
from BRIEF to SUPERVISION (supervisor-owned decomposition).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.workflows.base import WorkflowResult
from foundry_mcp.core.research.workflows.deep_research._model_resolution import (
    resolve_phase_provider,
    safe_resolve_model_for_role,
)
from foundry_mcp.core.research.workflows.deep_research.source_quality import (
    _extract_hostname,
)

logger = logging.getLogger(__name__)


def _snapshot_iteration_counts(state: DeepResearchState) -> dict[str, int]:
    """Capture current source/finding/directive counts at iteration start.

    Returns a dict suitable for storing in ``state.metadata`` so that
    cancellation rollback can trim items added during a partial iteration.
    """
    return {
        "source_count": len(state.sources),
        "finding_count": len(state.findings),
        "topic_result_count": len(state.topic_research_results),
    }


def _rollback_partial_iteration(state: DeepResearchState) -> dict[str, int]:
    """Remove sources, findings, and topic results added during a partial iteration.

    Uses the snapshot stored in ``state.metadata["iteration_snapshot"]`` to
    trim collections back to their pre-iteration sizes.

    Returns a dict with counts of removed items for audit logging.
    """
    snapshot = state.metadata.get("iteration_snapshot")
    if not snapshot:
        return {"sources_removed": 0, "findings_removed": 0, "topic_results_removed": 0}

    removed = {}
    with state._state_lock:
        src_count = snapshot.get("source_count", 0)
        removed["sources_removed"] = max(0, len(state.sources) - src_count)
        state.sources = state.sources[:src_count]

        find_count = snapshot.get("finding_count", 0)
        removed["findings_removed"] = max(0, len(state.findings) - find_count)
        state.findings = state.findings[:find_count]

        tr_count = snapshot.get("topic_result_count", 0)
        removed["topic_results_removed"] = max(0, len(state.topic_research_results) - tr_count)
        state.topic_research_results = state.topic_research_results[:tr_count]

    return removed


def _validate_report_output_path(path: str) -> Path:
    """Validate a report output path, blocking directory traversal.

    Args:
        path: The report output path to validate.

    Returns:
        The resolved ``Path`` if it passes all checks.

    Raises:
        ValueError: If the path contains ``..`` segments or resolves
            outside its own parent directory.
    """
    raw = Path(path)

    # Belt-and-suspenders: reject ".." segments before resolution
    if ".." in raw.parts:
        raise ValueError(f"Path traversal detected in report output path: {path}")

    resolved = raw.resolve()

    # Ensure the resolved path still lives under the same parent directory
    # that the un-resolved path claimed.  This catches symlink-based escapes.
    if not resolved.parent.is_dir():
        raise ValueError(
            f"Parent directory does not exist for report output path: {resolved}"
        )

    return resolved


class WorkflowExecutionMixin:
    """Mixin providing async workflow execution for deep research.

    Requires the composing class to provide:
    - self.config: ResearchConfig
    - self.memory: ResearchMemory
    - self.hooks: SupervisorHooks
    - self.orchestrator: SupervisorOrchestrator
    - self._tasks: dict[str, BackgroundTask]
    - self._tasks_lock: threading.Lock
    - self._search_providers: dict[str, SearchProvider]
    - self._write_audit_event(): from AuditMixin
    - self._flush_state(): from PersistenceMixin
    - self._record_workflow_error(): from ErrorHandlingMixin
    - self._safe_orchestrator_transition(): from ErrorHandlingMixin
    - self._check_cancellation(): defined here
    - Phase execution methods: clarification, brief, gathering, supervision, synthesis
    """

    config: ResearchConfig
    memory: ResearchMemory
    hooks: Any
    orchestrator: Any
    _tasks: dict[str, Any]
    _tasks_lock: threading.Lock
    _search_providers: dict[str, Any]

    # Stubs for Pyright — canonical signatures live in phases/_protocols.py
    if TYPE_CHECKING:

        def _write_audit_event(
            self,
            state: DeepResearchState | None,
            event_name: str,
            *,
            data: dict[str, Any] | None = ...,
            level: str = ...,
        ) -> None: ...
        def _flush_state(self, state: DeepResearchState) -> None: ...
        def _record_workflow_error(self, *args: Any, **kwargs: Any) -> None: ...
        def _safe_orchestrator_transition(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_clarification_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_brief_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_gathering_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_synthesis_async(self, *args: Any, **kwargs: Any) -> Any: ...
        async def _execute_supervision_async(self, *args: Any, **kwargs: Any) -> Any: ...

    def _get_extract_provider(self) -> Any:
        """Get a TavilyExtractProvider for source deepening, or None."""
        import os

        try:
            from foundry_mcp.core.research.providers.tavily_extract import (
                TavilyExtractProvider,
            )

            api_key = self.config.tavily_api_key or os.environ.get("TAVILY_API_KEY")
            if not api_key:
                return None
            return TavilyExtractProvider(api_key=api_key)
        except Exception as exc:
            logger.warning("Failed to initialize extract provider: %s", exc)
            return None

    async def _finalize_report(
        self, state: DeepResearchState, *, trigger: str = "normal"
    ) -> None:
        """Finalize citations and append confidence section.

        1. Run finalize_citations (renumber + append ## Sources)
        2. Run generate_confidence_section (append ## Research Confidence)

        Non-fatal for both steps. Idempotency-guarded.
        """
        if state.metadata.get("_report_finalized") or not state.report:
            return

        # Step 1: Citation finalization
        try:
            from foundry_mcp.core.research.workflows.deep_research.phases._citation_postprocess import (
                finalize_citations,
            )

            report, finalize_meta = finalize_citations(
                state.report,
                state,
                query_type=state.metadata.get("_query_type"),
            )
            state.report = report

            audit_data = {**finalize_meta}
            if trigger != "normal":
                audit_data["trigger"] = trigger
            self._write_audit_event(
                state,
                "citation_finalize_complete",
                data=audit_data,
            )
        except Exception as exc:
            logger.warning(
                "Citation finalize failed for research %s (%s): %s",
                state.id,
                trigger,
                exc,
            )
            self._write_audit_event(
                state,
                "citation_finalize_failed",
                data={"error": str(exc), "trigger": trigger},
                level="warning",
            )

        # Step 2: Confidence section (non-fatal, skipped when no verification data)
        if state.claim_verification is not None:
            try:
                from foundry_mcp.core.research.workflows.deep_research.phases._confidence_section import (
                    generate_confidence_section,
                )
                from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
                    LLMCallResult,
                    execute_llm_call,
                )

                confidence_provider, confidence_model = safe_resolve_model_for_role(
                    self.config, "confidence"
                )
                if confidence_provider is None:
                    confidence_provider = resolve_phase_provider(
                        self.config, "compression"
                    )

                async def _confidence_llm_call(
                    system_prompt: str, user_prompt: str
                ) -> str:
                    call_result = await execute_llm_call(
                        workflow=self,
                        state=state,
                        phase_name="confidence",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        provider_id=confidence_provider,
                        model=confidence_model,
                        temperature=0.3,
                        timeout=30.0,
                        role="confidence",
                    )
                    if isinstance(call_result, LLMCallResult):
                        return call_result.result.content or ""
                    raise RuntimeError(
                        f"Confidence LLM call failed: {getattr(call_result, 'error', 'unknown')}"
                    )

                section = await generate_confidence_section(
                    state,
                    _confidence_llm_call,
                    query_type=state.metadata.get("_query_type"),
                )

                if section and state.report:
                    state.report = state.report.rstrip() + "\n\n" + section + "\n"

                self._write_audit_event(
                    state,
                    "confidence_section_complete",
                    data={"trigger": trigger, "length": len(section or "")},
                )
            except Exception as exc:
                logger.warning(
                    "Confidence section failed for research %s (%s): %s",
                    state.id,
                    trigger,
                    exc,
                )
                self._write_audit_event(
                    state,
                    "confidence_section_failed",
                    data={"error": str(exc), "trigger": trigger},
                    level="warning",
                )

        # Mark finalized and save
        state.metadata["_report_finalized"] = True
        if state.report_output_path and state.report:
            validated = _validate_report_output_path(state.report_output_path)
            validated.write_text(state.report, encoding="utf-8")

    def _check_cancellation(self, state: DeepResearchState) -> None:
        """Check if cancellation has been requested for this research session.

        Raises:
            asyncio.CancelledError: If cancellation is detected
        """
        # Retrieve the background task for this research session
        with self._tasks_lock:
            bg_task = self._tasks.get(state.id)

        if bg_task:
            bg_task.touch()  # Record liveness for stale detection
        if bg_task and bg_task.is_cancelled:
            logger.info(
                "Cancellation detected for research %s at phase %s, iteration %d",
                state.id,
                state.phase.value,
                state.iteration,
            )
            raise asyncio.CancelledError("Cancellation requested")

    async def _run_phase(
        self,
        state: DeepResearchState,
        phase: DeepResearchPhase,
        executor: Any,
        *,
        skip_error_check: bool = False,
        skip_transition: bool = False,
    ) -> WorkflowResult | None:
        """Execute common phase lifecycle: cancel -> timer -> hooks -> audit -> execute -> error -> hooks -> audit -> transition.

        Encapsulates the boilerplate shared across phase dispatch blocks
        in ``_execute_workflow_async``.

        Args:
            state: Current research state.
            phase: The phase being executed (used for audit events and orchestrator transition).
            executor: An *unawaited* coroutine returned by ``_execute_<phase>_async(...)``.
            skip_error_check: If True, do not check ``result.success`` for failure.
            skip_transition: If True, skip the standard orchestrator transition
                (used by SUPERVISION/SYNTHESIS which have custom post-processing).

        Returns:
            ``WorkflowResult`` on phase failure (caller should ``return`` it),
            ``None`` on success (caller continues to next phase).
        """
        try:
            self._check_cancellation(state)
        except asyncio.CancelledError:
            # Close the unawaited executor coroutine to prevent
            # "coroutine was never awaited" RuntimeWarning.
            if asyncio.iscoroutine(executor):
                executor.close()
            raise
        phase_started = time.perf_counter()
        self.hooks.emit_phase_start(state)
        self._write_audit_event(
            state,
            "phase_start",
            data={"phase": state.phase.value},
        )

        result = await executor

        if not skip_error_check and not result.success:
            self._write_audit_event(
                state,
                "phase_error",
                data={"phase": state.phase.value, "error": result.error},
                level="error",
            )
            state.mark_failed(result.error or f"Phase {state.phase.value} failed")
            self._flush_state(state)
            return result

        self.hooks.emit_phase_complete(state)
        self._write_audit_event(
            state,
            "phase_complete",
            data={
                "phase": state.phase.value,
                "duration_ms": (time.perf_counter() - phase_started) * 1000,
            },
        )

        if not skip_transition:
            self._safe_orchestrator_transition(state, phase)

        return None

    async def _execute_workflow_async(
        self,
        state: DeepResearchState,
        provider_id: Optional[str],
        timeout_per_operation: float,
        max_concurrent: int,
    ) -> WorkflowResult:
        """Execute the full workflow asynchronously.

        This is the main async entry point that orchestrates all phases.
        """
        start_time = time.perf_counter()

        try:
            # Phase transition table (new workflows):
            #   CLARIFICATION → BRIEF → SUPERVISION → SYNTHESIS
            #
            # PLANNING and GATHERING are legacy-resume-only phases — new
            # workflows never enter them.  The supervisor handles both
            # decomposition (round 0) and gap-driven follow-up (rounds 1+).
            #
            # Legacy saved states may resume at GATHERING; this is handled
            # below by running GATHERING once and then advancing to
            # SUPERVISION.  The ``elif … not in (SUPERVISION, SYNTHESIS)``
            # guard after the GATHERING block skips any other intermediate
            # phases that may exist in the enum but are no longer active.

            if state.phase == DeepResearchPhase.CLARIFICATION:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.CLARIFICATION,
                    self._execute_clarification_async(
                        state=state,
                        provider_id=resolve_phase_provider(self.config, "clarification"),
                        timeout=self.config.get_phase_timeout("clarification"),
                    ),
                )
                if err:
                    return err

            if state.phase == DeepResearchPhase.BRIEF:
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.BRIEF,
                    self._execute_brief_async(
                        state=state,
                        provider_id=resolve_phase_provider(self.config, "brief"),
                        timeout=self.config.get_phase_timeout("brief"),
                    ),
                )
                if err:
                    return err

            # After BRIEF, jump directly to SUPERVISION (supervisor-owned decomposition).
            # PLANNING and GATHERING are legacy-resume-only phases — new workflows
            # never enter them.  The supervisor handles both decomposition (round 0)
            # and gap-driven follow-up (rounds 1+).
            if state.phase == DeepResearchPhase.PLANNING:
                # Legacy saved states may resume at PLANNING; skip to SUPERVISION.
                logger.warning(
                    "PLANNING phase running from legacy saved state (research %s) — advancing to SUPERVISION",
                    state.id,
                )
                self._write_audit_event(
                    state,
                    "legacy_phase_resume",
                    data={
                        "phase": "planning",
                        "deprecated_phase": True,
                    },
                    level="warning",
                )
                state.advance_phase()  # PLANNING → skips GATHERING → SUPERVISION

            if state.phase == DeepResearchPhase.GATHERING:
                # Legacy saved states may resume at GATHERING; run it once
                # then advance to SUPERVISION.
                logger.warning(
                    "GATHERING phase running from legacy saved state (research %s) "
                    "— new workflows use supervisor-owned decomposition via SUPERVISION phase",
                    state.id,
                )
                self._write_audit_event(
                    state,
                    "legacy_phase_resume",
                    data={
                        "phase": "gathering",
                        "message": "Legacy saved state resumed at GATHERING phase",
                        "deprecated_phase": True,
                    },
                    level="warning",
                )
                state.metadata["iteration_in_progress"] = True
                state.metadata["iteration_snapshot"] = _snapshot_iteration_counts(state)
                err = await self._run_phase(
                    state,
                    DeepResearchPhase.GATHERING,
                    self._execute_gathering_async(
                        state=state,
                        provider_id=provider_id,
                        timeout=timeout_per_operation,
                        max_concurrent=max_concurrent,
                    ),
                )
                if err:
                    return err
                state.advance_phase()  # GATHERING → SUPERVISION
            elif state.phase not in (DeepResearchPhase.SUPERVISION, DeepResearchPhase.SYNTHESIS):
                logger.warning(
                    "Unexpected phase %s at iteration entry for research %s, advancing to next active phase",
                    state.phase,
                    state.id,
                )
                state.advance_phase()  # Skip to next active phase

            # SUPERVISION → SYNTHESIS → CLAIM VERIFICATION iteration loop.
            # Fidelity-gated re-iteration: if claim verification yields a
            # fidelity score below threshold and iterations remain, loop back
            # to SUPERVISION with gap-focused directives.
            _iteration_complete = False
            while not _iteration_complete:
                # Snapshot source count before supervision so we can detect
                # zero-yield iterations (no new sources found).
                _sources_before_supervision = len(state.sources)

                # SUPERVISION (handles decomposition + research + gap-fill internally)
                if state.phase == DeepResearchPhase.SUPERVISION:
                    if not self.config.deep_research_enable_supervision:
                        logger.info(
                            "Supervision disabled via config for research %s, skipping to SYNTHESIS",
                            state.id,
                        )
                        self._write_audit_event(
                            state,
                            "supervision_skipped",
                            data={"reason": "disabled_via_config"},
                        )
                        state.phase = DeepResearchPhase.SYNTHESIS
                    else:
                        state.metadata["iteration_in_progress"] = True
                        state.metadata["iteration_snapshot"] = _snapshot_iteration_counts(state)

                        err = await self._run_phase(
                            state,
                            DeepResearchPhase.SUPERVISION,
                            self._execute_supervision_async(
                                state=state,
                                provider_id=resolve_phase_provider(self.config, "supervision", "reflection"),
                                timeout=self.config.get_phase_timeout("supervision"),
                            ),
                            skip_transition=True,
                        )
                        if err:
                            return err
                        state.phase = DeepResearchPhase.SYNTHESIS

                # --- ZERO-YIELD SHORT-CIRCUIT ---
                # If this is iteration 2+ and supervision added no new sources,
                # re-synthesizing won't improve the report. Keep the previous
                # iteration's report, finalize, and complete immediately.
                _sources_gained = len(state.sources) - _sources_before_supervision
                if state.iteration > 1 and _sources_gained == 0:
                    logger.info(
                        "Zero new sources on iteration %d for research %s — "
                        "short-circuiting (previous report preserved)",
                        state.iteration,
                        state.id,
                    )
                    _short_circuit_data: dict[str, Any] = {
                        "iteration": state.iteration,
                        "reason": "zero_source_yield",
                        "sources_total": len(state.sources),
                    }
                    _provider_health = state.metadata.get("_provider_health")
                    if _provider_health and _provider_health.get("all_degraded"):
                        _short_circuit_data["provider_health"] = "all_degraded"
                    self._write_audit_event(
                        state,
                        "iteration_short_circuit",
                        data=_short_circuit_data,
                    )
                    await self._finalize_report(
                        state, trigger="zero_yield_short_circuit"
                    )
                    state.metadata["iteration_in_progress"] = False
                    state.metadata["last_completed_iteration"] = state.iteration
                    state.metadata.pop("iteration_snapshot", None)
                    state.metadata["completion_reason"] = "zero_source_yield"
                    state.mark_completed(report=state.report)
                    _iteration_complete = True
                    continue
                # --- END ZERO-YIELD SHORT-CIRCUIT ---

                # SYNTHESIS
                if state.phase == DeepResearchPhase.SYNTHESIS:
                    # --- ZERO-SOURCE GUARD ---
                    if len(state.sources) == 0:
                        logger.warning(
                            "Zero sources collected for research %s — synthesis will be ungrounded",
                            state.id,
                        )
                        self._write_audit_event(
                            state,
                            "zero_source_synthesis_warning",
                            data={
                                "source_count": 0,
                                "phase": state.phase.value,
                                "iteration": state.iteration,
                            },
                            level="warning",
                        )
                        state.metadata["ungrounded_synthesis"] = True
                    # --- END ZERO-SOURCE GUARD ---

                    # Resume guard: skip synthesis if report exists and verification was in progress
                    _skip_to_verification = False
                    if state.report and state.metadata.get("claim_verification_started") and not state.claim_verification:
                        logger.info("Resuming claim verification (synthesis already complete) for research %s", state.id)
                        _skip_to_verification = True
                    elif state.report and state.claim_verification:
                        logger.info("Claim verification already complete for research %s, skipping", state.id)
                        # Both synthesis and verification done — skip to completion.
                    else:
                        err = await self._run_phase(
                            state,
                            DeepResearchPhase.SYNTHESIS,
                            self._execute_synthesis_async(
                                state=state,
                                provider_id=state.synthesis_provider,
                                timeout=self.config.get_phase_timeout("synthesis"),
                            ),
                            skip_transition=True,
                        )
                        if err:
                            return err

                        # Phase-specific: evaluate synthesis quality
                        try:
                            self.orchestrator.evaluate_phase_completion(state, DeepResearchPhase.SYNTHESIS)
                            prompt = self.orchestrator.get_reflection_prompt(state, DeepResearchPhase.SYNTHESIS)
                            self.hooks.think_pause(state, prompt)
                            self.orchestrator.record_to_state(state)
                        except Exception as exc:
                            logger.exception(
                                "Orchestrator transition failed for synthesis, research %s: %s",
                                state.id,
                                exc,
                            )
                            self._write_audit_event(
                                state,
                                "orchestrator_error",
                                data={
                                    "phase": "synthesis",
                                    "error": str(exc),
                                    "traceback": traceback.format_exc(),
                                },
                                level="error",
                            )
                            self._record_workflow_error(exc, state, "orchestrator_synthesis")
                            raise

                    # Repair heading-body fusions from synthesis.
                    if state.report:
                        from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
                            repair_heading_boundaries_global,
                        )
                        state.report = repair_heading_boundaries_global(state.report)

                    # --- UNGROUNDED SYNTHESIS DISCLAIMER ---
                    if state.metadata.get("ungrounded_synthesis") and state.report:
                        _disclaimer = (
                            "> **Note:** This report was generated without web sources "
                            "due to search failures. All claims are based on the model's "
                            "training data and may be outdated or inaccurate.\n\n"
                        )
                        state.report = _disclaimer + state.report

                        if state.report_output_path:
                            validated = _validate_report_output_path(state.report_output_path)
                            validated.write_text(state.report, encoding="utf-8")
                        elif state.report:
                            # Fallback: primary save failed, try research memory dir
                            try:
                                fallback_dir = self.memory.base_path / "deep_research"
                                fallback_dir.mkdir(parents=True, exist_ok=True)
                                fallback_path = fallback_dir / f"{state.id}.md"
                                fallback_path.write_text(state.report, encoding="utf-8")
                                state.report_output_path = str(fallback_path)
                                logger.info(
                                    "Fallback-saved report to %s", fallback_path
                                )
                            except Exception:
                                logger.warning(
                                    "Fallback report save also failed",
                                    exc_info=True,
                                )

                        self._write_audit_event(
                            state,
                            "ungrounded_synthesis_disclaimer_added",
                            data={
                                "report_length": len(state.report),
                                "report_saved": bool(state.report_output_path),
                            },
                        )
                    # --- END UNGROUNDED SYNTHESIS DISCLAIMER ---

                    # --- CLAIM VERIFICATION ---
                    _cv_enabled = self.config.deep_research_claim_verification_enabled and (
                        state.research_profile is None
                        or getattr(state.research_profile, "enable_claim_verification", True)
                    )
                    if _cv_enabled and not state.claim_verification:
                        from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
                            apply_corrections,
                            extract_and_verify_claims,
                            remap_unsupported_citations,
                        )

                        state.metadata["claim_verification_started"] = True
                        state.metadata["claim_verification_in_progress"] = True
                        self.memory.save_deep_research(state)

                        # Snapshot report before corrections for rollback on exception.
                        report_snapshot = state.report

                        try:
                            verification_result = await extract_and_verify_claims(
                                state=state,
                                config=self.config,
                                provider_id=resolve_phase_provider(self.config, "claim_verification", "synthesis"),
                                workflow=self,
                                timeout=self.config.deep_research_claim_verification_timeout,
                            )
                            state.claim_verification = verification_result

                            report_modified = False

                            if verification_result.claims_contradicted > 0:
                                await apply_corrections(
                                    state=state,
                                    config=self.config,
                                    verification_result=verification_result,
                                    workflow=self,
                                )
                                report_modified = True

                            if verification_result.claims_unsupported > 0:
                                await remap_unsupported_citations(
                                    state=state,
                                    verification_result=verification_result,
                                    workflow=self,
                                    provider_id=resolve_phase_provider(
                                        self.config, "claim_verification", "synthesis"
                                    ),
                                    timeout=self.config.deep_research_claim_verification_timeout,
                                    max_concurrent=self.config.deep_research_claim_verification_max_concurrent,
                                )
                                if verification_result.citations_remapped > 0:
                                    report_modified = True

                            if report_modified and state.report_output_path:
                                validated = _validate_report_output_path(state.report_output_path)
                                validated.write_text(state.report or "", encoding="utf-8")

                            # Persist corrected report + verification result.
                            self.memory.save_deep_research(state)

                            self._write_audit_event(
                                state,
                                "claim_verification_complete",
                                data={
                                    "claims_extracted": verification_result.claims_extracted,
                                    "claims_filtered": verification_result.claims_filtered,
                                    "claims_verified": verification_result.claims_verified,
                                    "claims_contradicted": verification_result.claims_contradicted,
                                    "corrections_applied": verification_result.corrections_applied,
                                    "citations_remapped": verification_result.citations_remapped,
                                },
                            )
                        except Exception as exc:
                            # Rollback to pre-correction report.
                            state.report = report_snapshot
                            logger.warning(
                                "Claim verification failed for research %s, delivering unverified report: %s",
                                state.id,
                                exc,
                            )
                            state.metadata["claim_verification_skipped"] = str(exc)
                            self._write_audit_event(
                                state,
                                "claim_verification_failed",
                                data={"error": str(exc)},
                                level="warning",
                            )
                        finally:
                            state.pop_metadata("claim_verification_in_progress")
                    # --- END CLAIM VERIFICATION ---

                    # --- SOURCE DEEPENING ---
                    # Re-verify UNSUPPORTED claims with expanded content windows
                    # before computing fidelity. This runs inline (no re-iteration)
                    # and can upgrade verdicts from UNSUPPORTED to SUPPORTED.
                    _deepening_classification = None
                    if (
                        state.claim_verification
                        and state.claim_verification.claims_unsupported > 0
                    ):
                        try:
                            from foundry_mcp.core.research.workflows.deep_research.phases._source_deepening import (
                                classify_unsupported_claims,
                                deepen_thin_sources,
                                reverify_with_expanded_window,
                            )
                            from foundry_mcp.core.research.workflows.deep_research.phases._lifecycle import (
                                LLMCallResult,
                                execute_llm_call,
                            )

                            _cv_citation_map = state.get_citation_map()
                            _deepening_classification = classify_unsupported_claims(
                                state.claim_verification, _cv_citation_map
                            )

                            # Shared LLM callback for re-verification (used by
                            # both deepen_window and deepen_extract paths).
                            _deepen_provider_id = resolve_phase_provider(
                                self.config, "claim_verification", "synthesis"
                            )

                            # NOTE: This closure captures self, state, and
                            # _deepen_provider_id by reference. Safe in the
                            # current sequential flow; do not parallelise
                            # without binding these as default arguments.
                            async def _deepen_llm_call(
                                system_prompt: str, user_prompt: str
                            ) -> str:
                                ret = await execute_llm_call(
                                    workflow=self,
                                    state=state,
                                    phase_name="claim_reverification",
                                    system_prompt=system_prompt,
                                    user_prompt=user_prompt,
                                    provider_id=_deepen_provider_id,
                                    model=None,
                                    temperature=0.0,
                                    timeout=self.config.deep_research_claim_verification_timeout,
                                    role="claim_verification",
                                )
                                if isinstance(ret, LLMCallResult):
                                    return ret.result.content or ""
                                raise RuntimeError("Deepening LLM call failed")

                            # Expanded-window re-verification for claims with rich sources.
                            if _deepening_classification.deepen_window:
                                await reverify_with_expanded_window(
                                    _deepening_classification.deepen_window,
                                    _cv_citation_map,
                                    _deepen_llm_call,
                                )

                                # Recompute aggregate counts after verdict upgrades.
                                vr = state.claim_verification
                                vr.claims_supported = sum(
                                    1 for c in vr.details if c.verdict == "SUPPORTED"
                                )
                                vr.claims_partially_supported = sum(
                                    1 for c in vr.details if c.verdict == "PARTIALLY_SUPPORTED"
                                )
                                vr.claims_unsupported = sum(
                                    1 for c in vr.details if c.verdict == "UNSUPPORTED"
                                )

                            # Re-extract thin sources and re-verify upgraded claims.
                            _deepen_extract_deepened = 0
                            _deepen_extract_upgraded = 0
                            if _deepening_classification.deepen_extract:
                                _extract_provider = self._get_extract_provider()
                                if _extract_provider is not None:
                                    _deepen_extract_deepened = await deepen_thin_sources(
                                        _deepening_classification.deepen_extract,
                                        _cv_citation_map,
                                        _extract_provider,
                                    )

                                    if _deepen_extract_deepened > 0:
                                        # Re-verify the deepened claims with enriched content
                                        await reverify_with_expanded_window(
                                            _deepening_classification.deepen_extract,
                                            _cv_citation_map,
                                            _deepen_llm_call,
                                        )

                                        # Recompute aggregate counts after deepen_extract upgrades.
                                        vr = state.claim_verification
                                        vr.claims_supported = sum(
                                            1 for c in vr.details if c.verdict == "SUPPORTED"
                                        )
                                        vr.claims_partially_supported = sum(
                                            1 for c in vr.details if c.verdict == "PARTIALLY_SUPPORTED"
                                        )
                                        vr.claims_unsupported = sum(
                                            1 for c in vr.details if c.verdict == "UNSUPPORTED"
                                        )

                                    _deepen_extract_upgraded = sum(
                                        1
                                        for c in _deepening_classification.deepen_extract
                                        if c.verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED")
                                    )

                            _deepen_window_upgraded = sum(
                                1
                                for c in _deepening_classification.deepen_window
                                if c.verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED")
                            )
                            self._write_audit_event(
                                state,
                                "source_deepening_complete",
                                data={
                                    "inferential": len(_deepening_classification.inferential),
                                    "deepen_window_total": len(_deepening_classification.deepen_window),
                                    "deepen_window_upgraded": _deepen_window_upgraded,
                                    "deepen_extract_total": len(_deepening_classification.deepen_extract),
                                    "deepen_extract_deepened": _deepen_extract_deepened,
                                    "deepen_extract_upgraded": _deepen_extract_upgraded,
                                    "widen": len(_deepening_classification.widen),
                                },
                            )
                            # Persist upgraded verdicts so a crash before fidelity
                            # computation doesn't lose deepening results.
                            self.memory.save_deep_research(state)
                        except Exception as exc:
                            logger.warning(
                                "Source deepening failed for research %s: %s",
                                state.id,
                                exc,
                            )
                            self._write_audit_event(
                                state,
                                "source_deepening_failed",
                                data={"error": str(exc)},
                                level="warning",
                            )
                    # --- END SOURCE DEEPENING ---

                    # --- FIDELITY-GATED ITERATION DECISION ---
                    _fidelity_score = (
                        state.claim_verification.fidelity_score
                        if state.claim_verification
                        else None
                    )
                    if _fidelity_score is not None:
                        state.fidelity_scores.append(_fidelity_score)

                    iteration_decision = self.orchestrator.decide_iteration(
                        state,
                        fidelity_score=_fidelity_score,
                        fidelity_iteration_enabled=self.config.deep_research_fidelity_iteration_enabled,
                        fidelity_threshold=self.config.deep_research_fidelity_threshold,
                        fidelity_min_improvement=self.config.deep_research_fidelity_min_improvement,
                    )
                    self.orchestrator.record_to_state(state)

                    _should_iterate = (
                        iteration_decision.outputs or {}
                    ).get("should_iterate", False)

                    if _should_iterate:
                        # Build gap queries for the next supervision round
                        gap_queries: list[str] = []
                        if state.claim_verification:
                            from foundry_mcp.core.research.workflows.deep_research.phases.claim_verification import (
                                build_gap_queries as _build_gap_queries,
                            )
                            # Exclude inferential claims and claims already
                            # resolved by expanded-window re-verification.
                            _exclude: set[str] = set()
                            if _deepening_classification is not None:
                                for c in _deepening_classification.inferential:
                                    _exclude.add(c.claim)
                                for c in _deepening_classification.deepen_window:
                                    if c.verdict in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                                        _exclude.add(c.claim)
                            gap_queries = _build_gap_queries(
                                state.claim_verification,
                                exclude_claims=_exclude or None,
                            )

                        state.iteration_gap_queries = gap_queries
                        state.iteration += 1
                        state.phase = DeepResearchPhase.SUPERVISION
                        state.claim_verification = None
                        state.supervision_round = 0
                        state.supervision_messages = []
                        state.pop_metadata("claim_verification_started")
                        state.pop_metadata("claim_verification_in_progress")

                        logger.info(
                            "Fidelity re-iteration triggered for research %s: "
                            "fidelity=%.2f < threshold=%.2f, starting iteration %d/%d "
                            "with %d gap queries",
                            state.id,
                            _fidelity_score or 0.0,
                            self.config.deep_research_fidelity_threshold,
                            state.iteration,
                            state.max_iterations,
                            len(gap_queries),
                        )
                        self._write_audit_event(
                            state,
                            "fidelity_reiteration",
                            data={
                                "fidelity_score": _fidelity_score,
                                "threshold": self.config.deep_research_fidelity_threshold,
                                "iteration": state.iteration,
                                "max_iterations": state.max_iterations,
                                "gap_queries": gap_queries,
                            },
                        )
                        self.memory.save_deep_research(state)
                        continue  # Loop back to SUPERVISION
                    # --- END FIDELITY-GATED ITERATION DECISION ---

                    # --- CITATION FINALIZE ---
                    await self._finalize_report(state)
                    # --- END CITATION FINALIZE ---

                    # Workflow complete
                    state.metadata["iteration_in_progress"] = False
                    state.metadata["last_completed_iteration"] = state.iteration
                    state.metadata.pop("iteration_snapshot", None)
                    state.mark_completed(report=state.report)
                    _iteration_complete = True

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            state.total_duration_ms += duration_ms

            # Flush final state (bypasses throttle to ensure completion is captured)
            self._flush_state(state)
            self._write_audit_event(
                state,
                "workflow_complete",
                data={
                    "success": True,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "report_length": len(state.report or ""),
                    # Existing totals
                    "total_tokens_used": state.total_tokens_used,
                    "total_duration_ms": state.total_duration_ms,
                    # Token breakdown totals
                    "total_input_tokens": sum(m.input_tokens for m in state.phase_metrics),
                    "total_output_tokens": sum(m.output_tokens for m in state.phase_metrics),
                    "total_cached_tokens": sum(m.cached_tokens for m in state.phase_metrics),
                    # Per-phase metrics
                    "phase_metrics": [
                        {
                            "phase": m.phase,
                            "duration_ms": m.duration_ms,
                            "input_tokens": m.input_tokens,
                            "output_tokens": m.output_tokens,
                            "cached_tokens": m.cached_tokens,
                            "provider_id": m.provider_id,
                            "model_used": m.model_used,
                        }
                        for m in state.phase_metrics
                    ],
                    # Search provider stats
                    "search_provider_stats": state.search_provider_stats,
                    "total_search_queries": sum(state.search_provider_stats.values()),
                    # Source hostnames
                    "source_hostnames": sorted(
                        set(h for s in state.sources if s.url and (h := _extract_hostname(s.url)))
                    ),
                    # Research mode
                    "research_mode": state.research_mode.value,
                },
            )

            return WorkflowResult(
                success=True,
                content=state.report or "Research completed",
                provider_id=provider_id,
                tokens_used=state.total_tokens_used,
                duration_ms=duration_ms,
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "sub_query_count": len(state.sub_queries),
                    "source_count": len(state.sources),
                    "finding_count": len(state.findings),
                    "gap_count": len(state.unresolved_gaps()),
                    "is_complete": state.completed_at is not None,
                },
            )

        except asyncio.CancelledError:
            # Handle cancellation: implement partial result policy
            # Discard incomplete iteration results, persist only completed iterations

            # Transition to "cancelling" state
            state.metadata["cancellation_state"] = "cancelling"
            logger.info(
                "Workflow entering cancelling state for research %s",
                state.id,
            )

            logger.warning(
                "Workflow cancelled at phase %s, iteration %d, research %s",
                state.phase.value,
                state.iteration,
                state.id,
            )
            state.metadata["cancelled"] = True

            # Check if current iteration is incomplete and roll back
            # partial data collected during it.
            if state.metadata.get("iteration_in_progress"):
                # Remove sources, findings, and topic results added during
                # the partial iteration using the snapshot taken at entry.
                rollback_counts = _rollback_partial_iteration(state)

                last_completed_iteration = state.metadata.get("last_completed_iteration")
                if last_completed_iteration is not None and last_completed_iteration < state.iteration:
                    # We have a safe checkpoint from a prior completed iteration
                    logger.warning(
                        "Cancellation rollback: resetting iteration %d → %d and "
                        "phase → SYNTHESIS for research %s. Removed %d sources, "
                        "%d findings, %d topic results from partial iteration.",
                        state.iteration,
                        last_completed_iteration,
                        state.id,
                        rollback_counts.get("sources_removed", 0),
                        rollback_counts.get("findings_removed", 0),
                        rollback_counts.get("topic_results_removed", 0),
                    )
                    state.metadata["discarded_iteration"] = state.iteration
                    state.metadata["rollback_counts"] = rollback_counts
                    state.iteration = last_completed_iteration
                    state.phase = DeepResearchPhase.SYNTHESIS

                    # Finalize citations on the rolled-back report
                    await self._finalize_report(state, trigger="cancellation_rollback")
                else:
                    # First iteration is incomplete - we cannot safely resume, must discard entire session
                    logger.warning(
                        "First iteration incomplete at cancellation, marking session "
                        "for discard, research %s. Removed %d sources, %d findings, "
                        "%d topic results.",
                        state.id,
                        rollback_counts.get("sources_removed", 0),
                        rollback_counts.get("findings_removed", 0),
                        rollback_counts.get("topic_results_removed", 0),
                    )
                    state.metadata["discarded_iteration"] = state.iteration
                    state.metadata["rollback_counts"] = rollback_counts
            else:
                # Iteration was successfully completed, safe to save
                logger.info(
                    "Cancelled after completed iteration %d, research %s",
                    state.iteration,
                    state.id,
                )

                # Finalize citations if not already done
                await self._finalize_report(state, trigger="cancellation_completed")

            # Save state with cancelling transition
            self.memory.save_deep_research(state)

            # Transition to "cleanup" state before cleanup phase
            state.metadata["cancellation_state"] = "cleanup"
            logger.info(
                "Workflow entering cleanup state for research %s",
                state.id,
            )

            # Mark the state as cancelled with phase context
            state.mark_cancelled(phase_state=f"phase={state.phase.value}, iteration={state.iteration}")
            self.memory.save_deep_research(state)

            self._write_audit_event(
                state,
                "workflow_cancelled",
                data={
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                    "iteration_in_progress": state.metadata.get("iteration_in_progress"),
                    "last_completed_iteration": state.metadata.get("last_completed_iteration"),
                    "discarded_iteration": state.metadata.get("discarded_iteration"),
                    "cancellation_state": state.metadata.get("cancellation_state"),
                    "terminal_status": "cancelled",
                },
                level="warning",
            )
            # Re-raise CancelledError to honour Python's cancellation
            # contract.  Callers using asyncio.wait_for() or Task.cancel()
            # need to see the exception propagate.  The outer handler in
            # background_tasks.py already guards on ``state.completed_at is
            # None`` so it won't overwrite the careful iteration rollback
            # and partial-result discard performed above (mark_cancelled
            # sets completed_at).
            raise
        except Exception as exc:
            tb_str = traceback.format_exc()
            logger.exception(
                "Workflow execution failed at phase %s, iteration %d: %s",
                state.phase.value,
                state.iteration,
                exc,
            )
            if not state.metadata.get("failed"):
                state.mark_failed(str(exc))
            self.memory.save_deep_research(state)
            self._write_audit_event(
                state,
                "workflow_error",
                data={
                    "error": str(exc),
                    "traceback": tb_str,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
                level="error",
            )
            self._record_workflow_error(exc, state, "workflow_execution")
            return WorkflowResult(
                success=False,
                content="",
                error=str(exc),
                metadata={
                    "research_id": state.id,
                    "phase": state.phase.value,
                    "iteration": state.iteration,
                },
            )
        finally:
            # Ensure resources are cleaned up on cancellation, timeout, or any other exit
            # This block runs regardless of exception type or successful completion,
            # but does not re-save state if already saved (to avoid duplicate saves)
            logger.debug(
                "Workflow cleanup phase for research %s at phase %s",
                state.id,
                state.phase.value,
            )

            # Close any open search provider connections
            # (Currently search providers don't maintain persistent connections,
            # but this is in place for future stateful provider implementations)
            for provider in list(self._search_providers.values()):
                try:
                    # Check if provider has async close method
                    if hasattr(provider, "aclose"):
                        await provider.aclose()
                    elif hasattr(provider, "close"):
                        provider.close()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Error closing search provider during cleanup: %s",
                        cleanup_exc,
                    )

            # After cleanup completes, mark cancellation as fully complete if transitioning through cleanup state
            if state.metadata.get("cancellation_state") == "cleanup":
                state.metadata["cancellation_state"] = "cancelled"
                logger.info(
                    "Workflow cancellation complete for research %s",
                    state.id,
                )
                self.memory.save_deep_research(state)
