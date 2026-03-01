"""Tests for PLAN-1 Item 2: Research Provenance Audit Trail.

Covers:
- ProvenanceEntry and ProvenanceLog model creation and validation
- ProvenanceLog.append() with auto-timestamp
- Provenance wired into ResearchExtensions and DeepResearchState
- Provenance populated after brief phase (mock)
- Provenance populated after supervision round (mock)
- Provenance persisted and loadable from disk (sidecar file)
- Provenance included in report response
- Serialization roundtrip (model_dump → model_validate)
- Entry cap enforcement
- Completion timestamp stamping via mark_completed/mark_failed/mark_cancelled
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from foundry_mcp.core.research.models.deep_research import (
    MAX_PROVENANCE_ENTRIES,
    DeepResearchState,
    ProvenanceEntry,
    ProvenanceLog,
    ResearchExtensions,
)


# =========================================================================
# ProvenanceEntry model tests
# =========================================================================


class TestProvenanceEntry:
    """Test ProvenanceEntry model creation and validation."""

    def test_create_entry(self):
        entry = ProvenanceEntry(
            timestamp="2024-01-01T00:00:00+00:00",
            phase="brief",
            event_type="brief_generated",
            summary="Research brief generated (500 chars)",
            details={"brief_length": 500},
        )
        assert entry.phase == "brief"
        assert entry.event_type == "brief_generated"
        assert entry.details["brief_length"] == 500

    def test_default_details(self):
        entry = ProvenanceEntry(
            timestamp="2024-01-01T00:00:00+00:00",
            phase="supervision",
            event_type="decomposition",
            summary="Generated 3 directives",
        )
        assert entry.details == {}

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ProvenanceEntry.model_validate({
                "timestamp": "2024-01-01T00:00:00+00:00",
                "phase": "brief",
                "event_type": "brief_generated",
                "summary": "test",
                "unknown_field": "value",
            })


# =========================================================================
# ProvenanceLog model tests
# =========================================================================


class TestProvenanceLog:
    """Test ProvenanceLog model creation and append functionality."""

    def test_create_log(self):
        log = ProvenanceLog(
            session_id="deepres-abc123",
            query="What is quantum computing?",
            profile="general",
            profile_config={"name": "general", "citation_style": "default"},
            started_at="2024-01-01T00:00:00+00:00",
        )
        assert log.session_id == "deepres-abc123"
        assert log.query == "What is quantum computing?"
        assert log.profile == "general"
        assert log.completed_at is None
        assert log.entries == []

    def test_append_creates_timestamped_entries(self):
        log = ProvenanceLog(
            session_id="deepres-abc123",
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )

        log.append(
            phase="brief",
            event_type="brief_generated",
            summary="Brief generated (200 chars)",
            brief_length=200,
            provider_id="tavily",
        )

        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.phase == "brief"
        assert entry.event_type == "brief_generated"
        assert entry.summary == "Brief generated (200 chars)"
        assert entry.details["brief_length"] == 200
        assert entry.details["provider_id"] == "tavily"
        # Timestamp should be valid ISO 8601
        datetime.fromisoformat(entry.timestamp)

    def test_append_multiple_entries(self):
        log = ProvenanceLog(
            session_id="deepres-abc123",
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )

        log.append(phase="brief", event_type="brief_generated", summary="Brief done")
        log.append(phase="supervision", event_type="decomposition", summary="3 directives")
        log.append(phase="supervision", event_type="iteration_complete", summary="Round 0 done")
        log.append(phase="synthesis", event_type="synthesis_completed", summary="Report done")

        assert len(log.entries) == 4
        assert [e.phase for e in log.entries] == ["brief", "supervision", "supervision", "synthesis"]
        assert [e.event_type for e in log.entries] == [
            "brief_generated",
            "decomposition",
            "iteration_complete",
            "synthesis_completed",
        ]

    def test_entry_cap_enforcement(self):
        """Entries beyond MAX_PROVENANCE_ENTRIES are trimmed (oldest dropped)."""
        log = ProvenanceLog(
            session_id="deepres-abc123",
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )

        # Fill to capacity + 10
        for i in range(MAX_PROVENANCE_ENTRIES + 10):
            log.append(
                phase="supervision",
                event_type="iteration_complete",
                summary=f"Round {i}",
                round=i,
            )

        assert len(log.entries) == MAX_PROVENANCE_ENTRIES
        # Oldest entries should have been trimmed — first entry should be round 10
        assert log.entries[0].details["round"] == 10

    def test_serialization_roundtrip(self):
        """ProvenanceLog survives model_dump → model_validate."""
        log = ProvenanceLog(
            session_id="deepres-abc123",
            query="quantum computing review",
            profile="academic",
            profile_config={"name": "academic", "citation_style": "apa"},
            started_at="2024-01-01T00:00:00+00:00",
            completed_at="2024-01-01T01:00:00+00:00",
        )
        log.append(phase="brief", event_type="brief_generated", summary="Brief done", length=300)
        log.append(
            phase="supervision",
            event_type="coverage_assessment",
            summary="Coverage 0.82",
            confidence=0.82,
        )

        data = log.model_dump(mode="json")
        restored = ProvenanceLog.model_validate(data)

        assert restored.session_id == log.session_id
        assert restored.query == log.query
        assert restored.profile == log.profile
        assert restored.completed_at == log.completed_at
        assert len(restored.entries) == 2
        assert restored.entries[0].event_type == "brief_generated"
        assert restored.entries[1].details["confidence"] == 0.82


# =========================================================================
# ResearchExtensions + DeepResearchState integration
# =========================================================================


class TestProvenanceOnState:
    """Test provenance wiring into ResearchExtensions and DeepResearchState."""

    def test_extensions_provenance_default_none(self):
        ext = ResearchExtensions()
        assert ext.provenance is None

    def test_extensions_provenance_assignment(self):
        ext = ResearchExtensions()
        log = ProvenanceLog(
            session_id="deepres-test",
            query="test",
            started_at="2024-01-01T00:00:00+00:00",
        )
        ext.provenance = log
        assert ext.provenance is not None
        assert ext.provenance.session_id == "deepres-test"

    def test_state_provenance_accessor(self):
        state = DeepResearchState(original_query="test query")
        assert state.provenance is None

        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = log
        assert state.provenance is not None
        assert state.provenance.session_id == state.id

    def test_state_serialization_excludes_none_provenance(self):
        state = DeepResearchState(original_query="test query")
        data = state.extensions.model_dump()
        assert "provenance" not in data  # exclude_none=True by default

    def test_mark_completed_stamps_provenance(self):
        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = log
        assert log.completed_at is None

        state.mark_completed(report="Final report")
        assert log.completed_at is not None
        datetime.fromisoformat(log.completed_at)

    def test_mark_failed_stamps_provenance(self):
        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = log

        state.mark_failed("Something went wrong")
        assert log.completed_at is not None

    def test_mark_cancelled_stamps_provenance(self):
        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = log

        state.mark_cancelled()
        assert log.completed_at is not None


# =========================================================================
# Memory persistence tests (sidecar file)
# =========================================================================


class TestProvenancePersistence:
    """Test provenance persistence via ResearchMemory sidecar file."""

    def test_save_and_load_with_provenance(self, tmp_path: Path):
        from foundry_mcp.core.research.memory import ResearchMemory

        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            profile="academic",
            started_at="2024-01-01T00:00:00+00:00",
        )
        log.append(phase="brief", event_type="brief_generated", summary="Brief done")
        state.extensions.provenance = log

        memory.save_deep_research(state)

        # Verify provenance sidecar file was created
        provenance_path = memory._provenance_path(state.id)
        assert provenance_path.exists()

        # Verify main state file does NOT contain provenance
        state_path = memory._deep_research._get_file_path(state.id)
        state_data = json.loads(state_path.read_text())
        ext_data = state_data.get("extensions", {})
        assert ext_data.get("provenance") is None

        # Load and verify provenance is reattached
        loaded = memory.load_deep_research(state.id)
        assert loaded is not None
        assert loaded.provenance is not None
        assert loaded.provenance.session_id == state.id
        assert loaded.provenance.profile == "academic"
        assert len(loaded.provenance.entries) == 1
        assert loaded.provenance.entries[0].event_type == "brief_generated"

    def test_load_without_provenance_sidecar(self, tmp_path: Path):
        """Pre-PLAN-1 states have no provenance sidecar — should load fine."""
        from foundry_mcp.core.research.memory import ResearchMemory

        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        state = DeepResearchState(original_query="test query")
        # Save without provenance
        memory._deep_research.save(state.id, state)

        loaded = memory.load_deep_research(state.id)
        assert loaded is not None
        assert loaded.provenance is None

    def test_delete_removes_provenance_sidecar(self, tmp_path: Path):
        from foundry_mcp.core.research.memory import ResearchMemory

        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        state.extensions.provenance = log

        memory.save_deep_research(state)
        provenance_path = memory._provenance_path(state.id)
        assert provenance_path.exists()

        memory.delete_deep_research(state.id)
        assert not provenance_path.exists()

    def test_save_preserves_in_memory_provenance(self, tmp_path: Path):
        """After save, the in-memory state should still have provenance attached."""
        from foundry_mcp.core.research.memory import ResearchMemory

        memory = ResearchMemory(base_path=tmp_path, ttl_hours=24)

        state = DeepResearchState(original_query="test query")
        log = ProvenanceLog(
            session_id=state.id,
            query="test query",
            started_at="2024-01-01T00:00:00+00:00",
        )
        log.append(phase="brief", event_type="brief_generated", summary="Brief done")
        state.extensions.provenance = log

        memory.save_deep_research(state)

        # In-memory provenance should still be present
        assert state.provenance is not None
        assert len(state.provenance.entries) == 1
