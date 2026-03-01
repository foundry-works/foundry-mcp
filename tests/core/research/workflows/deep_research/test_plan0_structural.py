"""PLAN-0 structural tests.

Validates the Phase 1 (supervision extraction) and Phase 2 (ResearchExtensions)
changes from PLAN-0. These are purely structural tests — no behavioral changes.
"""

from __future__ import annotations

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    ProvenanceLog,
    ResearchExtensions,
    ResearchProfile,
)


# =========================================================================
# Phase 1: supervision_first_round module structure
# =========================================================================


class TestSupervisionFirstRoundModule:
    """Verify supervision_first_round.py is importable and well-formed."""

    def test_module_importable(self):
        """supervision_first_round can be imported independently."""
        from foundry_mcp.core.research.workflows.deep_research.phases import (
            supervision_first_round,
        )

        assert hasattr(supervision_first_round, "first_round_decompose_critique_revise")
        assert hasattr(supervision_first_round, "run_first_round_generate")
        assert hasattr(supervision_first_round, "run_first_round_critique")
        assert hasattr(supervision_first_round, "run_first_round_revise")

    def test_functions_are_async(self):
        """All extracted functions are async coroutines."""
        import asyncio

        from foundry_mcp.core.research.workflows.deep_research.phases.supervision_first_round import (
            first_round_decompose_critique_revise,
            run_first_round_critique,
            run_first_round_generate,
            run_first_round_revise,
        )

        assert asyncio.iscoroutinefunction(first_round_decompose_critique_revise)
        assert asyncio.iscoroutinefunction(run_first_round_generate)
        assert asyncio.iscoroutinefunction(run_first_round_critique)
        assert asyncio.iscoroutinefunction(run_first_round_revise)

    def test_supervision_mixin_still_importable(self):
        """SupervisionPhaseMixin import path is unchanged."""
        from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
            SupervisionPhaseMixin,
        )

        assert hasattr(SupervisionPhaseMixin, "_first_round_decompose_critique_revise")

    def test_supervision_mixin_delegates_to_extracted(self):
        """SupervisionPhaseMixin._first_round_decompose_critique_revise delegates
        to the extracted module (not a reimplementation)."""
        import inspect

        from foundry_mcp.core.research.workflows.deep_research.phases.supervision import (
            SupervisionPhaseMixin,
        )

        source = inspect.getsource(
            SupervisionPhaseMixin._first_round_decompose_critique_revise,
        )
        assert "first_round_decompose_critique_revise" in source


# =========================================================================
# Phase 2: ResearchExtensions container model
# =========================================================================


class TestResearchExtensions:
    """Verify ResearchExtensions model behavior."""

    def test_default_serializes_empty(self):
        """Empty ResearchExtensions serializes to {} (exclude_none)."""
        ext = ResearchExtensions()
        data = ext.model_dump()
        # Only methodology_assessments should be present (empty list)
        assert data == {"methodology_assessments": []}

    def test_single_field_populated(self):
        """Extensions with one field set serializes only that field."""
        profile = ResearchProfile(name="test")
        ext = ResearchExtensions(research_profile=profile)
        data = ext.model_dump()
        assert "research_profile" in data
        assert data["research_profile"]["name"] == "test"
        # None fields excluded
        assert "provenance" not in data
        assert "citation_network" not in data

    def test_extra_fields_forbidden(self):
        """ResearchExtensions rejects unknown fields (extra=forbid)."""
        with pytest.raises(Exception):  # ValidationError
            ResearchExtensions(unknown_field="bad")

    def test_methodology_assessments_default_empty_list(self):
        """methodology_assessments defaults to empty list, not None."""
        ext = ResearchExtensions()
        assert ext.methodology_assessments == []

    def test_round_trip_serialization(self):
        """ResearchExtensions survives model_dump → model_validate."""
        profile = ResearchProfile(name="academic")
        provenance = ProvenanceLog(
            session_id="deepres-test",
            query="test",
            started_at="2024-01-01T00:00:00+00:00",
        )
        ext = ResearchExtensions(
            research_profile=profile,
            provenance=provenance,
        )
        data = ext.model_dump(mode="json")
        restored = ResearchExtensions.model_validate(data)
        assert restored.research_profile is not None
        assert restored.research_profile.name == "academic"
        assert restored.provenance is not None
        assert restored.provenance.session_id == "deepres-test"
        assert restored.structured_output is None


class TestDeepResearchStateExtensions:
    """Verify DeepResearchState integration with ResearchExtensions."""

    def test_extensions_field_present(self):
        """DeepResearchState has an extensions field."""
        state = DeepResearchState(original_query="test")
        assert hasattr(state, "extensions")
        assert isinstance(state.extensions, ResearchExtensions)

    def test_default_extensions_backward_compatible(self):
        """State with default extensions serializes without breaking existing schema."""
        state = DeepResearchState(original_query="test")
        data = state.model_dump(mode="json")
        # extensions should be present in serialized output
        assert "extensions" in data
        # Should be restorable
        restored = DeepResearchState.model_validate(data)
        assert isinstance(restored.extensions, ResearchExtensions)
        assert restored.original_query == "test"

    def test_deserialization_without_extensions_field(self):
        """Deserializing old state without extensions field works (backward compat)."""
        # Simulate an old serialized state that doesn't have 'extensions'
        old_data = {
            "original_query": "old query",
            "phase": "supervision",
        }
        state = DeepResearchState.model_validate(old_data)
        assert isinstance(state.extensions, ResearchExtensions)
        assert state.extensions.research_profile is None

    def test_convenience_accessor_research_profile(self):
        """state.research_profile delegates to state.extensions.research_profile."""
        state = DeepResearchState(original_query="test")
        # Default accessor returns PROFILE_GENERAL (not None)
        assert state.research_profile.name == "general"

        profile = ResearchProfile(name="academic")
        state.extensions.research_profile = profile
        assert state.research_profile.name == "academic"

    def test_convenience_accessor_provenance(self):
        """state.provenance delegates to state.extensions.provenance."""
        state = DeepResearchState(original_query="test")
        assert state.provenance is None

        provenance = ProvenanceLog(
            session_id=state.id,
            query="test",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = provenance
        assert state.provenance is not None
        assert state.provenance.session_id == state.id
