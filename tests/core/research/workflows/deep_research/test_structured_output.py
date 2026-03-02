"""Tests for structured research output.

Validates the StructuredResearchOutput model, the _build_structured_output()
synthesis method, report response inclusion, and serialization roundtrip.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest

from foundry_mcp.core.research.models.deep_research import (
    PROFILE_ACADEMIC,
    PROFILE_GENERAL,
    Contradiction,
    DeepResearchPhase,
    DeepResearchState,
    ResearchExtensions,
    ResearchProfile,
    StructuredResearchOutput,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchGap,
    ResearchSource,
    SourceQuality,
    SourceType,
)


# =========================================================================
# Helpers
# =========================================================================


def _make_source(
    *,
    id: str = "src-001",
    title: str = "Test Source",
    url: str | None = "https://example.com/paper",
    source_type: SourceType = SourceType.ACADEMIC,
    quality: SourceQuality = SourceQuality.HIGH,
    citation_number: int | None = 1,
    metadata: dict[str, Any] | None = None,
) -> ResearchSource:
    meta = metadata or {}
    return ResearchSource(
        id=id,
        title=title,
        url=url,
        source_type=source_type,
        quality=quality,
        citation_number=citation_number,
        metadata=meta,
    )


def _make_finding(
    *,
    id: str = "find-001",
    content: str = "Key finding",
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
    source_ids: list[str] | None = None,
    category: str | None = "theme-a",
) -> ResearchFinding:
    return ResearchFinding(
        id=id,
        content=content,
        confidence=confidence,
        source_ids=source_ids or ["src-001"],
        category=category,
    )


def _make_gap(
    *,
    id: str = "gap-001",
    description: str = "Missing data on X",
    resolved: bool = False,
    priority: int = 1,
    suggested_queries: list[str] | None = None,
) -> ResearchGap:
    return ResearchGap(
        id=id,
        description=description,
        resolved=resolved,
        priority=priority,
        suggested_queries=suggested_queries or ["search for X"],
    )


def _make_contradiction(
    *,
    id: str = "contra-001",
    description: str = "Source A says X, Source B says Y",
    finding_ids: list[str] | None = None,
    severity: Literal["major", "minor"] = "major",
    resolution: str | None = None,
) -> Contradiction:
    return Contradiction(
        id=id,
        description=description,
        finding_ids=finding_ids or ["find-001", "find-002"],
        severity=severity,
        resolution=resolution,
    )


def _make_state(
    *,
    sources: list[ResearchSource] | None = None,
    findings: list[ResearchFinding] | None = None,
    gaps: list[ResearchGap] | None = None,
    contradictions: list[Contradiction] | None = None,
    profile: ResearchProfile | None = None,
) -> DeepResearchState:
    ext = ResearchExtensions()
    if profile is not None:
        ext.research_profile = profile
    return DeepResearchState(
        original_query="test query",
        phase=DeepResearchPhase.SYNTHESIS,
        sources=sources or [],
        findings=findings or [],
        gaps=gaps or [],
        contradictions=contradictions or [],
        extensions=ext,
    )


# =========================================================================
# Model tests
# =========================================================================


class TestStructuredResearchOutputModel:
    """Tests for the StructuredResearchOutput Pydantic model."""

    def test_defaults(self) -> None:
        out = StructuredResearchOutput()
        assert out.sources == []
        assert out.findings == []
        assert out.gaps == []
        assert out.contradictions == []
        assert out.query_type == "explanation"
        assert out.profile == "general"

    def test_custom_fields(self) -> None:
        out = StructuredResearchOutput(
            sources=[{"id": "src-1", "title": "Paper A"}],
            findings=[{"id": "find-1", "content": "Result"}],
            gaps=[{"id": "gap-1", "description": "Missing"}],
            contradictions=[{"id": "contra-1", "description": "Conflict"}],
            query_type="literature_review",
            profile="academic",
        )
        assert len(out.sources) == 1
        assert out.sources[0]["title"] == "Paper A"
        assert out.query_type == "literature_review"
        assert out.profile == "academic"

    def test_serialization_roundtrip(self) -> None:
        out = StructuredResearchOutput(
            sources=[{"id": "src-1", "title": "Paper", "year": 2024}],
            findings=[{"id": "find-1", "content": "Result", "confidence": "high"}],
            gaps=[{"id": "gap-1", "description": "Gap"}],
            contradictions=[{"id": "contra-1", "description": "Conflict"}],
            query_type="comparison",
            profile="technical",
        )
        dumped = out.model_dump(mode="json")
        restored = StructuredResearchOutput.model_validate(dumped)
        assert restored.sources == out.sources
        assert restored.findings == out.findings
        assert restored.gaps == out.gaps
        assert restored.contradictions == out.contradictions
        assert restored.query_type == out.query_type
        assert restored.profile == out.profile

    def test_stored_on_extensions(self) -> None:
        out = StructuredResearchOutput(query_type="howto", profile="general")
        ext = ResearchExtensions(structured_output=out)
        assert ext.structured_output is not None
        assert ext.structured_output.query_type == "howto"

    def test_extensions_model_dump_excludes_none(self) -> None:
        ext = ResearchExtensions()
        dumped = ext.model_dump()
        assert "structured_output" not in dumped

    def test_extensions_model_dump_includes_when_set(self) -> None:
        out = StructuredResearchOutput(profile="academic")
        ext = ResearchExtensions(structured_output=out)
        dumped = ext.model_dump()
        assert "structured_output" in dumped
        assert dumped["structured_output"]["profile"] == "academic"


# =========================================================================
# _build_structured_output tests (via synthesis mixin)
# =========================================================================


class TestBuildStructuredOutput:
    """Tests for the _build_structured_output method on the synthesis mixin."""

    @pytest.fixture()
    def mixin(self):
        """Create a minimal synthesis mixin instance for testing."""
        from unittest.mock import MagicMock

        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        class FakeSynthesis(SynthesisPhaseMixin):
            pass

        obj = FakeSynthesis()
        obj.config = MagicMock()
        obj.memory = MagicMock()
        return obj

    def test_empty_state(self, mixin: Any) -> None:
        state = _make_state()
        result = mixin._build_structured_output(state, "explanation")
        assert isinstance(result, StructuredResearchOutput)
        assert result.sources == []
        assert result.findings == []
        assert result.gaps == []
        assert result.contradictions == []
        assert result.query_type == "explanation"
        assert result.profile == "general"

    def test_sources_denormalized(self, mixin: Any) -> None:
        src = _make_source(
            id="src-42",
            title="Deep Learning Survey",
            url="https://arxiv.org/abs/1234",
            source_type=SourceType.ACADEMIC,
            quality=SourceQuality.HIGH,
            citation_number=3,
            metadata={
                "authors": ["Smith, J.", "Doe, A."],
                "year": 2023,
                "venue": "Nature",
                "doi": "10.1234/test",
                "citation_count": 150,
                "fields_of_study": ["Computer Science"],
            },
        )
        state = _make_state(sources=[src])
        result = mixin._build_structured_output(state, "explanation")
        assert len(result.sources) == 1
        s = result.sources[0]
        assert s["id"] == "src-42"
        assert s["title"] == "Deep Learning Survey"
        assert s["url"] == "https://arxiv.org/abs/1234"
        assert s["source_type"] == "academic"
        assert s["quality"] == "high"
        assert s["citation_number"] == 3
        assert s["authors"] == ["Smith, J.", "Doe, A."]
        assert s["year"] == 2023
        assert s["venue"] == "Nature"
        assert s["doi"] == "10.1234/test"
        assert s["citation_count"] == 150
        # Extra metadata keys get included too
        assert s["fields_of_study"] == ["Computer Science"]

    def test_sources_exclude_internal_metadata(self, mixin: Any) -> None:
        """Internal underscore-prefixed metadata should be excluded."""
        src = _make_source(
            metadata={
                "authors": ["Test"],
                "_token_cache": {"v": 1, "counts": {}},
            },
        )
        state = _make_state(sources=[src])
        result = mixin._build_structured_output(state, "explanation")
        s = result.sources[0]
        assert s["authors"] == ["Test"]
        assert "_token_cache" not in s

    def test_sources_web_type(self, mixin: Any) -> None:
        src = _make_source(
            id="src-web",
            title="Blog Post",
            url="https://blog.example.com",
            source_type=SourceType.WEB,
            quality=SourceQuality.MEDIUM,
            metadata={},
        )
        state = _make_state(sources=[src])
        result = mixin._build_structured_output(state, "howto")
        s = result.sources[0]
        assert s["source_type"] == "web"
        assert s["quality"] == "medium"
        assert s["authors"] is None
        assert s["year"] is None

    def test_findings_with_confidence(self, mixin: Any) -> None:
        findings = [
            _make_finding(id="find-1", confidence=ConfidenceLevel.HIGH, category="theme-a"),
            _make_finding(id="find-2", confidence=ConfidenceLevel.LOW, category="theme-b"),
        ]
        state = _make_state(findings=findings)
        result = mixin._build_structured_output(state, "explanation")
        assert len(result.findings) == 2
        assert result.findings[0]["id"] == "find-1"
        assert result.findings[0]["confidence"] == "high"
        assert result.findings[0]["source_ids"] == ["src-001"]
        assert result.findings[0]["category"] == "theme-a"
        assert result.findings[1]["confidence"] == "low"

    def test_gaps_unresolved_only(self, mixin: Any) -> None:
        gaps = [
            _make_gap(id="gap-1", resolved=False, description="Open gap"),
            _make_gap(id="gap-2", resolved=True, description="Resolved gap"),
            _make_gap(id="gap-3", resolved=False, description="Another open gap"),
        ]
        state = _make_state(gaps=gaps)
        result = mixin._build_structured_output(state, "explanation")
        assert len(result.gaps) == 2
        gap_ids = [g["id"] for g in result.gaps]
        assert "gap-1" in gap_ids
        assert "gap-3" in gap_ids
        assert "gap-2" not in gap_ids

    def test_gaps_include_suggested_queries(self, mixin: Any) -> None:
        gap = _make_gap(
            suggested_queries=["query A", "query B"],
            priority=2,
        )
        state = _make_state(gaps=[gap])
        result = mixin._build_structured_output(state, "explanation")
        g = result.gaps[0]
        assert g["suggested_queries"] == ["query A", "query B"]
        assert g["priority"] == 2

    def test_contradictions_included(self, mixin: Any) -> None:
        contra = _make_contradiction(
            id="contra-42",
            finding_ids=["find-1", "find-3"],
            severity="major",
            resolution="Source A is more recent",
        )
        state = _make_state(contradictions=[contra])
        result = mixin._build_structured_output(state, "comparison")
        assert len(result.contradictions) == 1
        c = result.contradictions[0]
        assert c["id"] == "contra-42"
        assert c["description"] == "Source A says X, Source B says Y"
        assert c["finding_ids"] == ["find-1", "find-3"]
        assert c["severity"] == "major"
        assert c["resolution"] == "Source A is more recent"

    def test_query_type_and_profile_propagated(self, mixin: Any) -> None:
        state = _make_state(profile=PROFILE_ACADEMIC)
        result = mixin._build_structured_output(state, "literature_review")
        assert result.query_type == "literature_review"
        assert result.profile == "academic"

    def test_diverse_state(self, mixin: Any) -> None:
        """Full integration test with all data types present."""
        sources = [
            _make_source(id="src-1", title="Paper A", metadata={"authors": ["Author A"], "year": 2023}),
            _make_source(id="src-2", title="Blog B", source_type=SourceType.WEB, metadata={}),
        ]
        findings = [
            _make_finding(id="find-1", source_ids=["src-1"], confidence=ConfidenceLevel.HIGH),
            _make_finding(id="find-2", source_ids=["src-2"], confidence=ConfidenceLevel.MEDIUM),
        ]
        gaps = [
            _make_gap(id="gap-1", resolved=False),
            _make_gap(id="gap-2", resolved=True),
        ]
        contradictions = [
            _make_contradiction(id="contra-1", finding_ids=["find-1", "find-2"]),
        ]
        state = _make_state(
            sources=sources,
            findings=findings,
            gaps=gaps,
            contradictions=contradictions,
            profile=PROFILE_GENERAL,
        )
        result = mixin._build_structured_output(state, "comparison")

        assert len(result.sources) == 2
        assert len(result.findings) == 2
        assert len(result.gaps) == 1  # only unresolved
        assert len(result.contradictions) == 1
        assert result.query_type == "comparison"
        assert result.profile == "general"

    def test_result_serialization_roundtrip(self, mixin: Any) -> None:
        """Structured output should survive model_dump → model_validate."""
        src = _make_source(metadata={"authors": ["A"], "year": 2024, "doi": "10.1/test"})
        finding = _make_finding()
        gap = _make_gap(resolved=False)
        contra = _make_contradiction()
        state = _make_state(
            sources=[src],
            findings=[finding],
            gaps=[gap],
            contradictions=[contra],
        )
        result = mixin._build_structured_output(state, "explanation")
        dumped = result.model_dump(mode="json")
        restored = StructuredResearchOutput.model_validate(dumped)
        assert restored.sources == result.sources
        assert restored.findings == result.findings
        assert restored.gaps == result.gaps
        assert restored.contradictions == result.contradictions


# =========================================================================
# Report response inclusion tests
# =========================================================================


class TestReportResponseInclusion:
    """Tests that structured output appears in the report handler response."""

    def test_structured_output_in_report_response(self) -> None:
        """Verify the handler includes 'structured' when available on state."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.research.workflows.base import WorkflowResult
        from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (
            _handle_deep_research_report,
        )

        # Build a state with structured output
        structured = StructuredResearchOutput(
            sources=[{"id": "src-1", "title": "Paper"}],
            findings=[{"id": "find-1", "content": "Result"}],
            query_type="literature_review",
            profile="academic",
        )
        state = _make_state(profile=PROFILE_ACADEMIC)
        state.extensions.structured_output = structured
        state.report = "# Report\nContent"

        mock_workflow = MagicMock()
        mock_workflow.execute.return_value = WorkflowResult(
            success=True,
            content="# Report\nContent",
            metadata={"research_id": "test-123"},
        )

        mock_memory = MagicMock()
        mock_memory.load_deep_research.return_value = state

        with (
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_config"
            ) as mock_config,
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_memory",
                return_value=mock_memory,
            ),
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research.DeepResearchWorkflow",
                return_value=mock_workflow,
            ),
        ):
            mock_config.return_value.research = MagicMock()

            result = _handle_deep_research_report(research_id="test-123")

        assert result["data"]["structured"]["query_type"] == "literature_review"
        assert result["data"]["structured"]["profile"] == "academic"
        assert len(result["data"]["structured"]["sources"]) == 1

    def test_no_structured_when_none(self) -> None:
        """Verify no 'structured' field when state has no structured output."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.research.workflows.base import WorkflowResult
        from foundry_mcp.tools.unified.research_handlers.handlers_deep_research import (
            _handle_deep_research_report,
        )

        state = _make_state()
        state.report = "# Report"

        mock_workflow = MagicMock()
        mock_workflow.execute.return_value = WorkflowResult(
            success=True,
            content="# Report",
            metadata={"research_id": "test-456"},
        )

        mock_memory = MagicMock()
        mock_memory.load_deep_research.return_value = state

        with (
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_config"
            ) as mock_config,
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research._get_memory",
                return_value=mock_memory,
            ),
            patch(
                "foundry_mcp.tools.unified.research_handlers.handlers_deep_research.DeepResearchWorkflow",
                return_value=mock_workflow,
            ),
        ):
            mock_config.return_value.research = MagicMock()

            result = _handle_deep_research_report(research_id="test-456")

        assert "structured" not in result["data"]
