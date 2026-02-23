"""Unit and integration tests for Phase 3: Contradiction Detection.

Tests cover:
1. Contradiction model — fields, defaults, serialization
2. _detect_contradictions() — LLM call, JSON parsing, finding ID validation
3. Edge cases — empty findings, malformed JSON, LLM failure, no contradictions
4. Severity handling — major/minor classification and invalid values
5. Integration with analysis phase — contradictions stored in state
6. Integration with synthesis phase — contradictions included in prompt
7. Audit events — contradictions_detected event emission
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_mcp.core.research.models.deep_research import (
    Contradiction,
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchSource,
    SourceQuality,
)
from foundry_mcp.core.research.workflows.deep_research.phases.analysis import (
    AnalysisPhaseMixin,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "What are the effects of caffeine?",
    num_findings: int = 3,
) -> DeepResearchState:
    """Create a DeepResearchState with findings for contradiction detection."""
    state = DeepResearchState(
        id="deepres-test-contra",
        original_query=query,
        phase=DeepResearchPhase.ANALYSIS,
        iteration=1,
        max_iterations=3,
    )
    for i in range(num_findings):
        finding = ResearchFinding(
            id=f"find-{i}",
            content=f"Finding {i} about caffeine effects",
            confidence=ConfidenceLevel.MEDIUM,
            source_ids=[f"src-{i}"],
            category="Health",
        )
        state.findings.append(finding)
    # Add sources to match
    for i in range(num_findings):
        state.sources.append(
            ResearchSource(
                id=f"src-{i}",
                title=f"Source {i}",
                url=f"https://example.com/{i}",
                quality=SourceQuality.MEDIUM,
                citation_number=i + 1,
            )
        )
    return state


class StubAnalysisMixin(AnalysisPhaseMixin):
    """Concrete class inheriting AnalysisPhaseMixin for testing.

    Provides the runtime attributes the mixin expects from DeepResearchWorkflow.
    """

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.default_provider = "test-provider"
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []
        self._provider_async_fn: Any = None

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass

    async def _execute_provider_async(self, **kwargs: Any) -> MagicMock:
        if self._provider_async_fn:
            return await self._provider_async_fn(**kwargs)
        result = MagicMock()
        result.success = True
        result.content = json.dumps({"contradictions": []})
        result.tokens_used = 50
        return result


# =============================================================================
# Unit tests: Contradiction model
# =============================================================================


class TestContradictionModel:
    """Tests for the Contradiction model."""

    def test_default_values(self) -> None:
        """Default values are correct."""
        c = Contradiction(
            finding_ids=["find-1", "find-2"],
            description="Conflicting claims about caffeine",
        )
        assert c.finding_ids == ["find-1", "find-2"]
        assert c.description == "Conflicting claims about caffeine"
        assert c.resolution is None
        assert c.preferred_source_id is None
        assert c.severity == "minor"
        assert c.id.startswith("contra-")

    def test_full_construction(self) -> None:
        """All fields populated correctly."""
        c = Contradiction(
            id="contra-test",
            finding_ids=["find-1", "find-2", "find-3"],
            description="Source A says X, source B says Y",
            resolution="Source A is more recent and authoritative",
            preferred_source_id="src-a",
            severity="major",
        )
        assert c.id == "contra-test"
        assert len(c.finding_ids) == 3
        assert c.resolution is not None
        assert c.preferred_source_id == "src-a"
        assert c.severity == "major"

    def test_serialization(self) -> None:
        """Model serializes to dict correctly."""
        c = Contradiction(
            finding_ids=["find-1", "find-2"],
            description="Conflict",
            severity="major",
        )
        d = c.model_dump()
        assert d["finding_ids"] == ["find-1", "find-2"]
        assert d["description"] == "Conflict"
        assert d["severity"] == "major"
        assert "id" in d
        assert "created_at" in d

    def test_unique_ids(self) -> None:
        """Each Contradiction gets a unique auto-generated ID."""
        c1 = Contradiction(finding_ids=["f1", "f2"], description="A")
        c2 = Contradiction(finding_ids=["f3", "f4"], description="B")
        assert c1.id != c2.id

    def test_state_contradictions_list(self) -> None:
        """Contradictions can be stored on DeepResearchState."""
        state = _make_state(num_findings=2)
        c = Contradiction(
            finding_ids=["find-0", "find-1"],
            description="Conflicting data",
        )
        state.contradictions.append(c)

        assert len(state.contradictions) == 1
        assert state.contradictions[0].finding_ids == ["find-0", "find-1"]


# =============================================================================
# Unit tests: _detect_contradictions
# =============================================================================


class TestDetectContradictions:
    """Tests for AnalysisPhaseMixin._detect_contradictions()."""

    @pytest.mark.asyncio
    async def test_valid_contradictions_detected(self) -> None:
        """Valid LLM response with contradictions returns Contradiction objects."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "Finding 0 says caffeine is harmful, finding 1 says it is beneficial",
                        "resolution": "Depends on dosage",
                        "preferred_source_id": "src-0",
                        "severity": "major",
                    }
                ]
            })
            result.tokens_used = 100
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 1
        assert contradictions[0].finding_ids == ["find-0", "find-1"]
        assert "harmful" in contradictions[0].description
        assert contradictions[0].severity == "major"
        assert contradictions[0].preferred_source_id == "src-0"
        assert contradictions[0].resolution == "Depends on dosage"

    @pytest.mark.asyncio
    async def test_no_contradictions_returns_empty(self) -> None:
        """LLM reports no contradictions returns empty list."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"contradictions": []})
            result.tokens_used = 50
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_fewer_than_two_findings_skips(self) -> None:
        """With fewer than 2 findings, returns empty without LLM call."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=1)

        call_count = 0

        async def mock_provider(**kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"contradictions": []})
            result.tokens_used = 50
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []
        assert call_count == 0  # No LLM call made

    @pytest.mark.asyncio
    async def test_invalid_finding_ids_filtered(self) -> None:
        """Contradiction with non-existent finding IDs is filtered out."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-nonexistent-a", "find-nonexistent-b"],
                        "description": "This references invalid findings",
                        "severity": "major",
                    },
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "This references valid findings",
                        "severity": "minor",
                    },
                ]
            })
            result.tokens_used = 80
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        # Only the valid contradiction should survive
        assert len(contradictions) == 1
        assert contradictions[0].finding_ids == ["find-0", "find-1"]

    @pytest.mark.asyncio
    async def test_single_finding_id_filtered(self) -> None:
        """Contradiction with only 1 valid finding ID is filtered (needs >= 2)."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-0", "find-nonexistent"],
                        "description": "Only one valid ID",
                        "severity": "minor",
                    },
                ]
            })
            result.tokens_used = 60
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_empty_description_filtered(self) -> None:
        """Contradiction with empty description is filtered out."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "",
                        "severity": "minor",
                    },
                    {
                        "finding_ids": ["find-1", "find-2"],
                        "description": "  ",
                        "severity": "minor",
                    },
                ]
            })
            result.tokens_used = 60
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 0

    @pytest.mark.asyncio
    async def test_severity_validation(self) -> None:
        """Invalid severity values default to 'minor'."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "Conflict with invalid severity",
                        "severity": "critical",  # Invalid — should default to minor
                    },
                    {
                        "finding_ids": ["find-1", "find-2"],
                        "description": "Conflict with valid severity",
                        "severity": "major",
                    },
                ]
            })
            result.tokens_used = 80
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 2
        assert contradictions[0].severity == "minor"  # Invalid "critical" → "minor"
        assert contradictions[1].severity == "major"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self) -> None:
        """LLM call failure returns empty list gracefully."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = False
            result.error = "Provider timeout"
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self) -> None:
        """Exception during detection returns empty list gracefully."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            raise RuntimeError("Network error")

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty(self) -> None:
        """Malformed JSON in response returns empty list."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = "This is not JSON at all"
            result.tokens_used = 30
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_json_in_code_block(self) -> None:
        """JSON wrapped in markdown code block is extracted."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = """```json
{"contradictions": [{"finding_ids": ["find-0", "find-2"], "description": "Conflict", "severity": "minor"}]}
```"""
            result.tokens_used = 70
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 1
        assert contradictions[0].description == "Conflict"

    @pytest.mark.asyncio
    async def test_tokens_tracked(self) -> None:
        """Tokens from contradiction detection are added to state."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)
        state.total_tokens_used = 200

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"contradictions": []})
            result.tokens_used = 85
            return result

        mixin._provider_async_fn = mock_provider

        await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert state.total_tokens_used == 285

    @pytest.mark.asyncio
    async def test_multiple_contradictions(self) -> None:
        """Multiple contradictions are all returned."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=4)
        # Add a 4th finding
        state.findings.append(
            ResearchFinding(
                id="find-3",
                content="Finding 3",
                confidence=ConfidenceLevel.HIGH,
                source_ids=["src-3"],
            )
        )

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "First contradiction",
                        "severity": "major",
                    },
                    {
                        "finding_ids": ["find-2", "find-3"],
                        "description": "Second contradiction",
                        "severity": "minor",
                    },
                ]
            })
            result.tokens_used = 120
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 2
        assert contradictions[0].severity == "major"
        assert contradictions[1].severity == "minor"

    @pytest.mark.asyncio
    async def test_non_list_contradictions_returns_empty(self) -> None:
        """If contradictions field is not a list, returns empty."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({"contradictions": "not a list"})
            result.tokens_used = 40
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_non_dict_entries_skipped(self) -> None:
        """Non-dict entries in contradictions array are skipped."""
        mixin = StubAnalysisMixin()
        state = _make_state(num_findings=3)

        async def mock_provider(**kwargs):
            result = MagicMock()
            result.success = True
            result.content = json.dumps({
                "contradictions": [
                    "not a dict",
                    42,
                    {
                        "finding_ids": ["find-0", "find-1"],
                        "description": "Valid one",
                        "severity": "minor",
                    },
                ]
            })
            result.tokens_used = 60
            return result

        mixin._provider_async_fn = mock_provider

        contradictions = await mixin._detect_contradictions(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert len(contradictions) == 1
        assert contradictions[0].description == "Valid one"


# =============================================================================
# Integration: synthesis prompt includes contradictions
# =============================================================================


class TestContradictionSynthesisIntegration:
    """Tests for contradiction inclusion in synthesis prompt."""

    def test_contradictions_in_synthesis_prompt(self) -> None:
        """Contradictions appear in the synthesis user prompt."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        class StubSynthesis(SynthesisPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.memory = MagicMock()

        state = _make_state(num_findings=3)
        state.contradictions = [
            Contradiction(
                finding_ids=["find-0", "find-1"],
                description="Source A says caffeine is harmful, source B says it is beneficial",
                resolution="Depends on dosage and individual factors",
                preferred_source_id="src-0",
                severity="major",
            ),
        ]

        stub = StubSynthesis()
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Contradictions Detected" in prompt
        assert "caffeine is harmful" in prompt
        assert "Depends on dosage" in prompt
        assert "MAJOR" in prompt
        assert "find-0" in prompt
        assert "find-1" in prompt

    def test_no_contradictions_omits_section(self) -> None:
        """Without contradictions, no contradictions section in prompt."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        class StubSynthesis(SynthesisPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.memory = MagicMock()

        state = _make_state(num_findings=3)
        state.contradictions = []

        stub = StubSynthesis()
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Contradictions Detected" not in prompt

    def test_preferred_source_citation_in_prompt(self) -> None:
        """Preferred source is mapped to citation number in prompt."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        class StubSynthesis(SynthesisPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.memory = MagicMock()

        state = _make_state(num_findings=3)
        # src-0 has citation_number=1
        state.contradictions = [
            Contradiction(
                finding_ids=["find-0", "find-1"],
                description="Conflict about dosage",
                preferred_source_id="src-0",
                severity="minor",
            ),
        ]

        stub = StubSynthesis()
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Preferred source: [1]" in prompt

    def test_synthesis_system_prompt_mentions_conflicting_info(self) -> None:
        """Synthesis system prompt includes 'Conflicting Information' section."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        class StubSynthesis(SynthesisPhaseMixin):
            def __init__(self) -> None:
                self.config = MagicMock()
                self.memory = MagicMock()

        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        assert "Conflicting Information" in prompt
