"""Tests for fidelity review prompt improvements and tiebreaker logic.

Phase 1: Verifies that verdict criteria and severity-based synthesis resolution
are present in the prompt templates.
Phase 2: Verifies tiebreaker reviewer invocation on split verdicts.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.core.ai_consultation import (
    ConsensusResult,
    ConsultationResult,
    ConsultationWorkflow,
    ProviderResponse,
)
from foundry_mcp.core.prompts.fidelity_review import (
    FIDELITY_REVIEW_V1,
    FIDELITY_SYNTHESIS_PROMPT_V1,
)


class TestReviewPromptVerdictCriteria:
    """Change 1: FIDELITY_REVIEW_V1 must include verdict criteria."""

    def test_review_prompt_contains_verdict_criteria_header(self):
        assert "Verdict criteria (use these strictly):" in FIDELITY_REVIEW_V1.user_template

    def test_review_prompt_pass_criterion(self):
        assert '"pass": All spec requirements are implemented correctly' in FIDELITY_REVIEW_V1.user_template

    def test_review_prompt_partial_criterion(self):
        assert '"partial": Implementation has one or more HIGH-severity deviations' in FIDELITY_REVIEW_V1.user_template

    def test_review_prompt_fail_criterion(self):
        assert '"fail": Implementation has CRITICAL-severity deviations' in FIDELITY_REVIEW_V1.user_template

    def test_review_prompt_medium_low_do_not_downgrade(self):
        assert (
            "Medium/low-severity" in FIDELITY_REVIEW_V1.user_template
            and "do NOT warrant" in FIDELITY_REVIEW_V1.user_template
        )


class TestSynthesisPromptSeverityResolution:
    """Change 2: FIDELITY_SYNTHESIS_PROMPT_V1 uses severity-based tiebreaker."""

    def test_synthesis_prompt_no_blind_partial_fallback(self):
        """The old 'use partial and note disagreement' rule must be gone."""
        assert 'use "partial" and note disagreement' not in FIDELITY_SYNTHESIS_PROMPT_V1.user_template

    def test_synthesis_prompt_severity_tiebreaker_pass(self):
        assert "No critical or high-severity deviations in any review" in FIDELITY_SYNTHESIS_PROMPT_V1.user_template

    def test_synthesis_prompt_severity_tiebreaker_partial(self):
        assert "Any high-severity deviations" in FIDELITY_SYNTHESIS_PROMPT_V1.user_template

    def test_synthesis_prompt_severity_tiebreaker_fail(self):
        assert "Any critical-severity deviations" in FIDELITY_SYNTHESIS_PROMPT_V1.user_template

    def test_synthesis_prompt_agreement_levels(self):
        template = FIDELITY_SYNTHESIS_PROMPT_V1.user_template
        for level in ("strong", "moderate", "weak", "conflicted"):
            assert level in template, f"Missing agreement_level: {level}"


# ---------------------------------------------------------------------------
# Phase 2: Tiebreaker reviewer tests
# ---------------------------------------------------------------------------

_MODULE = "foundry_mcp.tools.unified.review"


def _make_response(provider_id: str, verdict: str, *, success: bool = True) -> ProviderResponse:
    """Create a ProviderResponse with a JSON verdict."""
    content = json.dumps({"verdict": verdict, "deviations": [], "summary": "ok"})
    return ProviderResponse(
        provider_id=provider_id,
        model_used=f"{provider_id}-model",
        content=content if success else "",
        success=success,
        error=None if success else "failed",
    )


def _make_raw_response(provider_id: str, raw_content: str) -> ProviderResponse:
    """Create a ProviderResponse with non-JSON content."""
    return ProviderResponse(
        provider_id=provider_id,
        model_used=f"{provider_id}-model",
        content=raw_content,
        success=True,
    )


def _make_synthesis_content(verdict: str = "pass", num_models: int = 2) -> str:
    """Return valid synthesis JSON content."""
    return json.dumps(
        {
            "verdict": verdict,
            "deviations": [],
            "summary": "Synthesized result",
            "verdict_consensus": {
                "votes": {"pass": ["provider-a"]},
                "agreement_level": "strong",
            },
        }
    )


def _run_fidelity(tmp_path, mock_orch):
    """Call _handle_fidelity with all non-orchestrator deps mocked."""
    from foundry_mcp.tools.unified.review import _handle_fidelity

    specs_dir = tmp_path / "specs"
    specs_dir.mkdir(exist_ok=True)
    (specs_dir / ".fidelity-reviews").mkdir(exist_ok=True)

    with (
        patch(f"{_MODULE}.find_specs_directory", return_value=str(specs_dir)),
        patch(f"{_MODULE}.find_spec_file", return_value=str(specs_dir / "test.yaml")),
        patch(f"{_MODULE}.load_spec", return_value={"title": "Test", "description": "Test"}),
        patch(f"{_MODULE}._build_spec_requirements", return_value="reqs"),
        patch(f"{_MODULE}._build_implementation_artifacts", return_value="artifacts"),
        patch(f"{_MODULE}._build_test_results", return_value="tests"),
        patch(f"{_MODULE}._build_journal_entries", return_value="journal"),
        patch(f"{_MODULE}.load_consultation_config"),
        patch(f"{_MODULE}.ConsultationOrchestrator", return_value=mock_orch),
        patch(f"{_MODULE}.is_prompt_injection", return_value=False),
    ):
        return _handle_fidelity(
            config=MagicMock(),
            payload={"spec_id": "test-spec", "workspace": str(tmp_path)},
        )


class TestTiebreakerInvocation:
    """Phase 2: Tiebreaker invoked on split verdicts."""

    def test_tiebreaker_invoked_on_split_verdict(self, tmp_path):
        """Split verdicts trigger tiebreaker; synthesis receives 3 reviews."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.get_available_providers.return_value = [
            "provider-a",
            "provider-b",
            "provider-c",
        ]
        mock_orch.consult.side_effect = [
            # Call 1: initial multi-model → split verdict
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "partial"),
                ],
            ),
            # Call 2: tiebreaker → pass
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=json.dumps({"verdict": "pass", "deviations": [], "summary": "ok"}),
                provider_id="provider-c",
                model_used="provider-c-model",
            ),
            # Call 3: synthesis
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 3
        # Tiebreaker call used the unused provider
        tiebreaker_call = mock_orch.consult.call_args_list[1]
        tiebreaker_req = tiebreaker_call.args[0]
        assert tiebreaker_req.provider_id == "provider-c"
        # Tiebreaker bypasses cache to get a fresh evaluation
        assert tiebreaker_call.kwargs.get("use_cache") is False
        # Synthesis received 3 reviews
        synthesis_req = mock_orch.consult.call_args_list[2].args[0]
        assert synthesis_req.context["num_models"] == 3

    def test_no_tiebreaker_when_verdicts_agree(self, tmp_path):
        """Matching verdicts skip tiebreaker; only 2 consult calls."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.consult.side_effect = [
            # Call 1: both agree
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "pass"),
                ],
            ),
            # Call 2: synthesis
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 2
        mock_orch.get_available_providers.assert_not_called()

    def test_no_tiebreaker_when_majority_exists(self, tmp_path):
        """3 reviewers with 2-1 split → majority exists, skip tiebreaker."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.consult.side_effect = [
            # Call 1: 3 reviewers, 2 pass + 1 partial → majority on "pass"
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "pass"),
                    _make_response("provider-c", "partial"),
                ],
            ),
            # Call 2: synthesis (no tiebreaker needed)
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass", num_models=3),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 2
        mock_orch.get_available_providers.assert_not_called()

    def test_no_tiebreaker_when_no_unused_providers(self, tmp_path):
        """Split detected but no unused providers → skip tiebreaker."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.get_available_providers.return_value = [
            "provider-a",
            "provider-b",  # both already used
        ]
        mock_orch.consult.side_effect = [
            # Call 1: split verdict
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "partial"),
                ],
            ),
            # Call 2: synthesis (no tiebreaker)
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 2

    def test_tiebreaker_retries_next_candidate_on_failure(self, tmp_path):
        """First tiebreaker candidate fails → retries with second candidate."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.get_available_providers.return_value = [
            "provider-a",
            "provider-b",
            "provider-c",
            "provider-d",
        ]
        mock_orch.consult.side_effect = [
            # Call 1: split verdict
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "partial"),
                ],
            ),
            # Call 2: first tiebreaker candidate (provider-c) fails
            RuntimeError("Provider unavailable"),
            # Call 3: second tiebreaker candidate (provider-d) succeeds
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=json.dumps({"verdict": "pass", "deviations": [], "summary": "ok"}),
                provider_id="provider-d",
                model_used="provider-d-model",
            ),
            # Call 4: synthesis with 3 reviews
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 4
        # Second tiebreaker candidate was tried
        tiebreaker_req_2 = mock_orch.consult.call_args_list[2].args[0]
        assert tiebreaker_req_2.provider_id == "provider-d"
        # Synthesis received 3 reviews
        synthesis_req = mock_orch.consult.call_args_list[3].args[0]
        assert synthesis_req.context["num_models"] == 3

    def test_tiebreaker_all_candidates_exhausted(self, tmp_path):
        """All tiebreaker candidates fail → synthesis proceeds with 2 reviews."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.get_available_providers.return_value = [
            "provider-a",
            "provider-b",
            "provider-c",
        ]
        mock_orch.consult.side_effect = [
            # Call 1: split verdict
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "partial"),
                ],
            ),
            # Call 2: only tiebreaker candidate (provider-c) fails
            RuntimeError("Provider unavailable"),
            # Call 3: synthesis with 2 reviews (no tiebreaker succeeded)
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 3
        # Synthesis only got 2 reviews
        synthesis_req = mock_orch.consult.call_args_list[2].args[0]
        assert synthesis_req.context["num_models"] == 2

    def test_unparseable_response_excluded_from_split_detection(self, tmp_path):
        """Non-JSON response excluded from split detection → no tiebreaker."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.consult.side_effect = [
            # Call 1: one valid JSON, one non-JSON
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_raw_response("provider-b", "This is not JSON at all"),
                ],
            ),
            # Call 2: synthesis (no tiebreaker since only 1 parseable verdict)
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 2
        mock_orch.get_available_providers.assert_not_called()

    def test_tiebreaker_returning_consensus_result(self, tmp_path):
        """Tiebreaker returns ConsensusResult → first successful response used."""
        mock_orch = MagicMock()
        mock_orch.is_available.return_value = True
        mock_orch.get_available_providers.return_value = [
            "provider-a",
            "provider-b",
            "provider-c",
        ]
        mock_orch.consult.side_effect = [
            # Call 1: split verdict
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-a", "pass"),
                    _make_response("provider-b", "partial"),
                ],
            ),
            # Call 2: tiebreaker returns ConsensusResult (unusual but possible)
            ConsensusResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                responses=[
                    _make_response("provider-c", "pass"),
                ],
            ),
            # Call 3: synthesis with 3 reviews
            ConsultationResult(
                workflow=ConsultationWorkflow.FIDELITY_REVIEW,
                content=_make_synthesis_content("pass"),
                provider_id="provider-a",
                model_used="provider-a-model",
            ),
        ]

        _run_fidelity(tmp_path, mock_orch)

        assert mock_orch.consult.call_count == 3
        synthesis_req = mock_orch.consult.call_args_list[2].args[0]
        assert synthesis_req.context["num_models"] == 3
