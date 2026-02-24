"""Tests for Phase 3: Global Note Compression Before Synthesis.

Tests cover:
1. COMPRESSION phase enum positioning (between ANALYSIS and SYNTHESIS)
2. DeepResearchState.compressed_digest field
3. Global compression deduplicates cross-topic findings
4. Cross-topic contradictions are preserved and flagged
5. Citation numbering is consistent (original numbers preserved)
6. Synthesis prompt uses compressed_digest when available
7. Synthesis falls back to raw findings when digest absent
8. Phase is skipped for single-topic research
9. Phase is skipped when config flag disabled
10. Graceful fallback when compression fails
11. Config flags and role resolution
12. Workflow execution wiring
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from foundry_mcp.core.research.models.deep_research import (
    Contradiction,
    DeepResearchPhase,
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchGap,
    ResearchSource,
    SourceQuality,
    SourceType,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.orchestration import (
    PHASE_TO_AGENT,
    AgentRole,
)
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    CompressionMixin,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "What are the environmental impacts of AI?",
    phase: DeepResearchPhase = DeepResearchPhase.COMPRESSION,
    num_topics: int = 3,
    sources_per_topic: int = 2,
    with_compressed_findings: bool = True,
    with_analysis_findings: bool = True,
) -> DeepResearchState:
    """Create a DeepResearchState populated for global compression tests."""
    state = DeepResearchState(
        id="deepres-global-compress-test",
        original_query=query,
        research_brief="Investigating the environmental impacts of AI systems",
        phase=phase,
        iteration=1,
        max_iterations=3,
        max_sources_per_query=5,
    )

    for i in range(num_topics):
        sq = SubQuery(
            id=f"sq-{i}",
            query=f"Topic {i}: aspect {i} of AI environmental impact",
            status="completed",
            priority=1,
        )
        state.sub_queries.append(sq)

        source_ids = []
        for j in range(sources_per_topic):
            src = state.add_source(
                title=f"Source {i}-{j}: AI Impact Study {j}",
                url=f"https://example{j}.com/ai-impact-{i}",
                source_type=SourceType.WEB,
                snippet=f"Study about AI environmental impact topic {i}, source {j}.",
                sub_query_id=sq.id,
            )
            source_ids.append(src.id)

        compressed = None
        if with_compressed_findings:
            compressed = (
                f"## Topic {i} Findings\n"
                f"AI has significant environmental impact in area {i}. "
                f"Data centers consume substantial energy [{state.sources[-2].citation_number}]. "
                f"Carbon footprint is growing [{state.sources[-1].citation_number}]."
            )

        state.topic_research_results.append(
            TopicResearchResult(
                sub_query_id=sq.id,
                searches_performed=2,
                sources_found=sources_per_topic,
                per_topic_summary=f"Summary for topic {i}",
                reflection_notes=[f"Reflection {i}"],
                refined_queries=[f"Refined query {i}"],
                source_ids=source_ids,
                compressed_findings=compressed,
                early_completion=False,
                completion_rationale="Max searches reached",
            )
        )

    if with_analysis_findings:
        for i in range(num_topics):
            state.findings.append(
                ResearchFinding(
                    content=f"AI systems in area {i} consume significant energy",
                    source_ids=[state.sources[i * sources_per_topic].id],
                    category="Environmental Impact",
                )
            )

        # Add a contradiction
        state.contradictions.append(
            Contradiction(
                finding_ids=["find-0", "find-1"],
                description="Conflicting estimates of AI energy consumption growth rate",
                resolution="Different measurement methodologies",
                severity="minor",
            )
        )

        # Add a gap
        state.gaps.append(
            ResearchGap(
                description="Long-term environmental projections beyond 2030",
                resolved=False,
            )
        )

    return state


class StubCompressor(CompressionMixin):
    """Concrete class for testing CompressionMixin global compression."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.config.default_provider = "test-provider"
        self.config.deep_research_global_compression_provider = None
        self.config.deep_research_global_compression_model = None
        self.config.deep_research_compression_provider = None
        self.config.deep_research_compression_model = None
        self.config.deep_research_research_provider = None
        self.config.deep_research_research_model = None
        self.config.deep_research_analysis_provider = None
        self.config.deep_research_analysis_model = None
        self.memory = MagicMock()
        self._audit_events: list[tuple[str, dict]] = []

    def _write_audit_event(self, state: Any, event: str, **kwargs: Any) -> None:
        self._audit_events.append((event, kwargs))

    def _check_cancellation(self, state: Any) -> None:
        pass


# =============================================================================
# Phase Enum Tests
# =============================================================================


class TestCompressionPhaseEnum:
    """COMPRESSION phase enum positioning and agent mapping."""

    def test_compression_phase_exists(self):
        """COMPRESSION phase exists in the enum."""
        assert hasattr(DeepResearchPhase, "COMPRESSION")
        assert DeepResearchPhase.COMPRESSION.value == "compression"

    def test_compression_between_analysis_and_synthesis(self):
        """COMPRESSION is positioned between ANALYSIS and SYNTHESIS."""
        phases = list(DeepResearchPhase)
        analysis_idx = phases.index(DeepResearchPhase.ANALYSIS)
        compression_idx = phases.index(DeepResearchPhase.COMPRESSION)
        synthesis_idx = phases.index(DeepResearchPhase.SYNTHESIS)
        assert analysis_idx < compression_idx < synthesis_idx

    def test_advance_phase_analysis_to_compression(self):
        """advance_phase() from ANALYSIS goes to COMPRESSION."""
        state = DeepResearchState(
            original_query="test",
            phase=DeepResearchPhase.ANALYSIS,
        )
        new_phase = state.advance_phase()
        assert new_phase == DeepResearchPhase.COMPRESSION

    def test_advance_phase_compression_to_synthesis(self):
        """advance_phase() from COMPRESSION goes to SYNTHESIS."""
        state = DeepResearchState(
            original_query="test",
            phase=DeepResearchPhase.COMPRESSION,
        )
        new_phase = state.advance_phase()
        assert new_phase == DeepResearchPhase.SYNTHESIS

    def test_phase_to_agent_mapping(self):
        """COMPRESSION maps to COMPRESSOR agent role."""
        assert DeepResearchPhase.COMPRESSION in PHASE_TO_AGENT
        assert PHASE_TO_AGENT[DeepResearchPhase.COMPRESSION] == AgentRole.COMPRESSOR

    def test_compressor_agent_role_exists(self):
        """COMPRESSOR role exists in AgentRole enum."""
        assert hasattr(AgentRole, "COMPRESSOR")
        assert AgentRole.COMPRESSOR.value == "compressor"


# =============================================================================
# State Model Tests
# =============================================================================


class TestCompressedDigestField:
    """Tests for DeepResearchState.compressed_digest field."""

    def test_compressed_digest_defaults_to_none(self):
        """compressed_digest is None by default."""
        state = DeepResearchState(original_query="test")
        assert state.compressed_digest is None

    def test_compressed_digest_can_be_set(self):
        """compressed_digest can be set to a string."""
        state = DeepResearchState(original_query="test")
        state.compressed_digest = "## Research Digest\nKey findings..."
        assert state.compressed_digest == "## Research Digest\nKey findings..."

    def test_compressed_digest_serialization(self):
        """compressed_digest survives serialization round-trip."""
        state = DeepResearchState(
            original_query="test",
            compressed_digest="Digest content here",
        )
        dumped = state.model_dump()
        restored = DeepResearchState(**dumped)
        assert restored.compressed_digest == "Digest content here"

    def test_compressed_digest_none_backward_compat(self):
        """State without compressed_digest field deserializes cleanly."""
        data = DeepResearchState(original_query="test").model_dump()
        data.pop("compressed_digest", None)
        # Should still deserialize without error (default=None)
        state = DeepResearchState(**data)
        assert state.compressed_digest is None


# =============================================================================
# Global Compression Logic Tests
# =============================================================================


class TestGlobalCompressionSkipConditions:
    """Tests for when global compression should be skipped."""

    @pytest.mark.asyncio
    async def test_skip_single_topic(self):
        """Global compression is skipped for single-topic research."""
        state = _make_state(num_topics=1)
        stub = StubCompressor()

        result = await stub._execute_global_compression_async(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert result["skipped"] is True
        assert result["reason"] == "single_topic"
        assert state.compressed_digest is None

    @pytest.mark.asyncio
    async def test_skip_no_content(self):
        """Global compression is skipped when no topic content exists."""
        state = _make_state(
            num_topics=2,
            sources_per_topic=0,
            with_compressed_findings=False,
            with_analysis_findings=False,
        )
        # Clear topic results to have no content at all
        for tr in state.topic_research_results:
            tr.compressed_findings = None
            tr.per_topic_summary = None
            tr.source_ids = []
        stub = StubCompressor()

        result = await stub._execute_global_compression_async(
            state=state,
            provider_id="test-provider",
            timeout=60.0,
        )

        assert result["skipped"] is True
        assert result["reason"] == "no_content"


class TestGlobalCompressionExecution:
    """Tests for global compression LLM call and result handling."""

    @pytest.mark.asyncio
    async def test_successful_compression(self):
        """Global compression produces a unified digest."""
        state = _make_state(num_topics=3)
        stub = StubCompressor()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = (
            "## Research Digest\n"
            "AI has significant environmental impact across energy, water, and hardware.\n\n"
            "## Thematic Findings\n"
            "### Energy Consumption\n"
            "Data centers consume substantial energy [1][3][5].\n\n"
            "## Cross-Topic Contradictions\n"
            "Growth rate estimates differ [2] vs [4].\n\n"
            "## Source Summary\n"
            "6 sources across 3 domains."
        )
        mock_result.input_tokens = 2000
        mock_result.output_tokens = 800
        mock_result.provider_id = "test-provider"
        mock_result.model_used = "test-model"

        mock_call_result = MagicMock()
        mock_call_result.result = mock_result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_call_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", "test-model"),
        ):
            result = await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result["success"] is True
        assert result["topic_count"] == 3
        assert result["tokens_used"] == 2800
        assert state.compressed_digest is not None
        assert "Research Digest" in state.compressed_digest

    @pytest.mark.asyncio
    async def test_compression_failure_returns_error(self):
        """Global compression failure returns error dict, state.compressed_digest stays None."""
        state = _make_state(num_topics=2)
        stub = StubCompressor()

        from foundry_mcp.core.research.workflows.base import WorkflowResult

        error_result = WorkflowResult(
            success=False,
            content="",
            error="Context window exceeded",
        )

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=error_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            result = await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result["success"] is False
        assert "error" in result
        assert state.compressed_digest is None

    @pytest.mark.asyncio
    async def test_empty_result_handled(self):
        """Empty LLM result returns failure."""
        state = _make_state(num_topics=2)
        stub = StubCompressor()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = ""  # Empty content
        mock_call_result = MagicMock()
        mock_call_result.result = mock_result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_call_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            result = await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        assert result["success"] is False
        assert state.compressed_digest is None


class TestGlobalCompressionPrompts:
    """Tests for prompt construction in global compression."""

    @pytest.mark.asyncio
    async def test_prompt_includes_all_topics(self):
        """Global compression prompt includes content from all topics."""
        state = _make_state(num_topics=3)
        stub = StubCompressor()

        captured_prompts: dict[str, str] = {}

        async def capture_llm_call(**kwargs: Any) -> MagicMock:
            captured_prompts["system"] = kwargs.get("system_prompt", "")
            captured_prompts["user"] = kwargs.get("user_prompt", "")
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.content = "## Research Digest\nMerged findings."
            mock_result.input_tokens = 100
            mock_result.output_tokens = 50
            mock_call = MagicMock()
            mock_call.result = mock_result
            return mock_call

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        # All topics should appear in user prompt
        user_prompt = captured_prompts["user"]
        for i in range(3):
            assert f"Topic {i}" in user_prompt

        # System prompt should contain merge/dedup instructions
        system_prompt = captured_prompts["system"]
        assert "DEDUPLICATE" in system_prompt
        assert "MERGE" in system_prompt
        assert "CONTRADICTIONS" in system_prompt.upper()

    @pytest.mark.asyncio
    async def test_prompt_includes_analysis_findings(self):
        """Prompt includes analysis findings when available."""
        state = _make_state(num_topics=2, with_analysis_findings=True)
        stub = StubCompressor()

        captured_prompts: dict[str, str] = {}

        async def capture_llm_call(**kwargs: Any) -> MagicMock:
            captured_prompts["user"] = kwargs.get("user_prompt", "")
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.content = "digest"
            mock_result.input_tokens = 100
            mock_result.output_tokens = 50
            mock_call = MagicMock()
            mock_call.result = mock_result
            return mock_call

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        user_prompt = captured_prompts["user"]
        assert "Analysis Findings" in user_prompt
        assert "Environmental Impact" in user_prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_contradictions(self):
        """Prompt includes detected contradictions."""
        state = _make_state(num_topics=2, with_analysis_findings=True)
        stub = StubCompressor()

        captured_prompts: dict[str, str] = {}

        async def capture_llm_call(**kwargs: Any) -> MagicMock:
            captured_prompts["user"] = kwargs.get("user_prompt", "")
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.content = "digest"
            mock_result.input_tokens = 100
            mock_result.output_tokens = 50
            mock_call = MagicMock()
            mock_call.result = mock_result
            return mock_call

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        user_prompt = captured_prompts["user"]
        assert "Detected Contradictions" in user_prompt
        assert "energy consumption growth rate" in user_prompt

    @pytest.mark.asyncio
    async def test_prompt_preserves_original_citations(self):
        """Prompt instructs to preserve original citation numbers."""
        state = _make_state(num_topics=2)
        stub = StubCompressor()

        captured_prompts: dict[str, str] = {}

        async def capture_llm_call(**kwargs: Any) -> MagicMock:
            captured_prompts["system"] = kwargs.get("system_prompt", "")
            captured_prompts["user"] = kwargs.get("user_prompt", "")
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.content = "digest"
            mock_result.input_tokens = 100
            mock_result.output_tokens = 50
            mock_call = MagicMock()
            mock_call.result = mock_result
            return mock_call

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        system_prompt = captured_prompts["system"]
        assert "do not renumber" in system_prompt.lower()
        assert "PRESERVE CITATIONS" in system_prompt

    @pytest.mark.asyncio
    async def test_fallback_to_per_topic_summary(self):
        """When compressed_findings is None, falls back to per_topic_summary."""
        state = _make_state(num_topics=2, with_compressed_findings=False)
        stub = StubCompressor()

        captured_prompts: dict[str, str] = {}

        async def capture_llm_call(**kwargs: Any) -> MagicMock:
            captured_prompts["user"] = kwargs.get("user_prompt", "")
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.content = "digest"
            mock_result.input_tokens = 100
            mock_result.output_tokens = 50
            mock_call = MagicMock()
            mock_call.result = mock_result
            return mock_call

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            side_effect=capture_llm_call,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        user_prompt = captured_prompts["user"]
        assert "Summary for topic 0" in user_prompt
        assert "Summary for topic 1" in user_prompt


# =============================================================================
# Synthesis Integration Tests
# =============================================================================


class TestSynthesisCompressedDigestIntegration:
    """Tests for synthesis using compressed_digest when available."""

    def test_synthesis_prompt_uses_digest_when_available(self):
        """_build_synthesis_user_prompt prefers compressed_digest."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        state = _make_state(num_topics=2, with_analysis_findings=True)
        state.compressed_digest = (
            "## Research Digest\n"
            "Unified findings about AI environmental impact [1][2][3][4]."
        )

        # Create a minimal mixin instance
        mixin = SynthesisPhaseMixin()
        prompt = mixin._build_synthesis_user_prompt(state, allocation_result=None)

        assert "Unified Research Digest" in prompt
        assert state.compressed_digest in prompt
        # Should still include instructions and source reference
        assert "Instructions" in prompt
        assert "Source Reference" in prompt

    def test_synthesis_prompt_falls_back_without_digest(self):
        """_build_synthesis_user_prompt uses raw findings when no digest."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        state = _make_state(num_topics=2, with_analysis_findings=True)
        state.compressed_digest = None  # No digest

        mixin = SynthesisPhaseMixin()
        prompt = mixin._build_synthesis_user_prompt(state, allocation_result=None)

        assert "Findings to Synthesize" in prompt
        assert "Unified Research Digest" not in prompt
        # Should include categorized findings
        assert "Environmental Impact" in prompt

    def test_synthesis_digest_prompt_smaller_than_raw(self):
        """Prompt with compressed digest should be smaller than raw findings."""
        from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
            SynthesisPhaseMixin,
        )

        state = _make_state(num_topics=3, with_analysis_findings=True)

        mixin = SynthesisPhaseMixin()

        # Measure without digest
        state.compressed_digest = None
        raw_prompt = mixin._build_synthesis_user_prompt(state, allocation_result=None)

        # Set a short digest
        state.compressed_digest = "## Research Digest\nKey merged findings [1][2][3]."
        digest_prompt = mixin._build_synthesis_user_prompt(state, allocation_result=None)

        # Digest prompt should be shorter (digest replaces verbose findings)
        assert len(digest_prompt) < len(raw_prompt)


# =============================================================================
# Config Tests
# =============================================================================


class TestGlobalCompressionConfig:
    """Tests for global compression configuration."""

    def test_config_field_defaults(self):
        """Global compression config fields have correct defaults."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig()
        assert config.deep_research_enable_global_compression is False  # Default off (Phase 3 PLAN)
        assert config.deep_research_global_compression_provider is None
        assert config.deep_research_global_compression_model is None
        assert config.deep_research_global_compression_timeout == 360.0

    def test_config_from_toml_dict(self):
        """Config fields parse from TOML dict."""
        from foundry_mcp.config.research import ResearchConfig

        data = {
            "deep_research_enable_global_compression": False,
            "deep_research_global_compression_provider": "openai",
            "deep_research_global_compression_model": "gpt-4",
            "deep_research_global_compression_timeout": 120.0,
        }
        config = ResearchConfig.from_toml_dict(data)
        assert config.deep_research_enable_global_compression is False
        assert config.deep_research_global_compression_provider == "openai"
        assert config.deep_research_global_compression_model == "gpt-4"
        assert config.deep_research_global_compression_timeout == 120.0

    def test_phase_timeout_compression(self):
        """get_phase_timeout returns global compression timeout."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(deep_research_global_compression_timeout=180.0)
        assert config.get_phase_timeout("compression") == 180.0

    def test_role_resolution_chain(self):
        """global_compression role resolution falls through correctly."""
        from foundry_mcp.config.research import ResearchConfig

        config = ResearchConfig(
            default_provider="fallback-provider",
        )
        chain = config._ROLE_RESOLUTION_CHAIN.get("global_compression")
        assert chain is not None
        assert "global_compression" in chain
        assert "compression" in chain
        assert "research" in chain


# =============================================================================
# Audit / Observability Tests
# =============================================================================


class TestGlobalCompressionAudit:
    """Tests for audit event emission during global compression."""

    @pytest.mark.asyncio
    async def test_audit_event_on_success(self):
        """Successful compression emits audit event with stats."""
        state = _make_state(num_topics=2)
        stub = StubCompressor()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.content = "## Digest\nMerged."
        mock_result.input_tokens = 500
        mock_result.output_tokens = 200
        mock_call_result = MagicMock()
        mock_call_result.result = mock_result

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=mock_call_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        audit_events = [e[0] for e in stub._audit_events]
        assert "global_compression_complete" in audit_events

    @pytest.mark.asyncio
    async def test_audit_event_on_failure(self):
        """Failed compression emits failure audit event."""
        state = _make_state(num_topics=2)
        stub = StubCompressor()

        from foundry_mcp.core.research.workflows.base import WorkflowResult

        error_result = WorkflowResult(success=False, content="", error="timeout")

        with patch(
            "foundry_mcp.core.research.workflows.deep_research.phases._lifecycle.execute_llm_call",
            new_callable=AsyncMock,
            return_value=error_result,
        ), patch(
            "foundry_mcp.core.research.workflows.deep_research._helpers.safe_resolve_model_for_role",
            return_value=("test-provider", None),
        ):
            await stub._execute_global_compression_async(
                state=state,
                provider_id="test-provider",
                timeout=60.0,
            )

        audit_events = [e[0] for e in stub._audit_events]
        assert "global_compression_failed" in audit_events
