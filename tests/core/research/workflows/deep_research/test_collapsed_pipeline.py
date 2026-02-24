"""Tests for Phase 3 PLAN: Collapse Post-Gathering Pipeline.

Tests cover:
1. Default pipeline skips ANALYSIS, COMPRESSION, REFINEMENT phases
2. Synthesis consumes per-topic compressed findings directly
3. Config flags enable optional phases (backward compat)
4. Citations preserved in direct-to-synthesis path
5. Graceful fallback when compressed_findings missing
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from foundry_mcp.config.research import ResearchConfig
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchSource,
    SourceQuality,
    SubQuery,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    *,
    query: str = "What are the effects of caffeine on sleep?",
    phase: DeepResearchPhase = DeepResearchPhase.SYNTHESIS,
    with_findings: bool = False,
    with_compressed_topics: bool = False,
    with_compressed_digest: bool = False,
    num_topics: int = 2,
) -> DeepResearchState:
    """Create a DeepResearchState for pipeline collapse testing."""
    state = DeepResearchState(
        id="deepres-test-collapse",
        original_query=query,
        phase=phase,
        iteration=1,
        max_iterations=3,
        research_brief="Research the effects of caffeine on sleep quality and duration.",
    )

    # Add sub-queries and sources
    for i in range(num_topics):
        sq = SubQuery(
            id=f"sq-{i}",
            query=f"Sub-query {i} about caffeine",
            rationale=f"Rationale {i}",
            priority=1,
            status="completed",
        )
        state.sub_queries.append(sq)
        src = ResearchSource(
            id=f"src-{i}",
            title=f"Source {i}: Caffeine Study",
            url=f"https://example.com/study-{i}",
            quality=SourceQuality.MEDIUM,
            citation_number=i + 1,
            snippet=f"Caffeine affects sleep latency by {10 + i} minutes on average.",
        )
        state.sources.append(src)

    state.next_citation_number = num_topics + 1

    if with_compressed_topics:
        for i in range(num_topics):
            tr = TopicResearchResult(
                sub_query_id=f"sq-{i}",
                searches_performed=3,
                sources_found=2,
                source_ids=[f"src-{i}"],
                compressed_findings=(
                    f"Caffeine consumption delays sleep onset by approximately "
                    f"{10 + i} minutes. Studies show dose-dependent effects on "
                    f"sleep architecture, particularly REM sleep reduction [1]. "
                    f"Key excerpt: 'A 200mg dose administered 6 hours before "
                    f"bedtime significantly reduced total sleep time.' [{i + 1}]"
                ),
            )
            state.topic_research_results.append(tr)

    if with_findings:
        for i in range(num_topics):
            state.findings.append(
                ResearchFinding(
                    id=f"find-{i}",
                    content=f"Caffeine delays sleep onset by {10 + i} minutes",
                    confidence=ConfidenceLevel.MEDIUM,
                    source_ids=[f"src-{i}"],
                    category="Sleep Effects",
                )
            )

    if with_compressed_digest:
        state.compressed_digest = (
            "## Unified Findings\n\n"
            "Caffeine significantly affects sleep quality through multiple mechanisms. "
            "Studies consistently show delayed sleep onset [1] and reduced REM sleep [2]."
        )

    return state


class StubSynthesis(SynthesisPhaseMixin):
    """Concrete class inheriting SynthesisPhaseMixin for testing."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.memory = MagicMock()


# =============================================================================
# Tests: Config defaults (3.4)
# =============================================================================


class TestConfigDefaults:
    """Tests for Phase 3 config flag defaults."""

    def test_enable_analysis_digest_default_false(self) -> None:
        """Analysis digest is disabled by default."""
        config = ResearchConfig()
        assert config.deep_research_enable_analysis_digest is False

    def test_enable_refinement_default_false(self) -> None:
        """Refinement is disabled by default."""
        config = ResearchConfig()
        assert config.deep_research_enable_refinement is False

    def test_enable_global_compression_default_false(self) -> None:
        """Global compression is disabled by default (changed from True)."""
        config = ResearchConfig()
        assert config.deep_research_enable_global_compression is False

    def test_from_toml_enables_analysis_digest(self) -> None:
        """from_toml_dict respects enable_analysis_digest=True."""
        config = ResearchConfig.from_toml_dict(
            {"deep_research_enable_analysis_digest": True}
        )
        assert config.deep_research_enable_analysis_digest is True

    def test_from_toml_enables_refinement(self) -> None:
        """from_toml_dict respects enable_refinement=True."""
        config = ResearchConfig.from_toml_dict(
            {"deep_research_enable_refinement": True}
        )
        assert config.deep_research_enable_refinement is True

    def test_from_toml_enables_global_compression(self) -> None:
        """from_toml_dict respects enable_global_compression=True."""
        config = ResearchConfig.from_toml_dict(
            {"deep_research_enable_global_compression": True}
        )
        assert config.deep_research_enable_global_compression is True

    def test_from_toml_defaults_match_class_defaults(self) -> None:
        """from_toml_dict with empty dict matches class defaults."""
        config = ResearchConfig.from_toml_dict({})
        assert config.deep_research_enable_analysis_digest is False
        assert config.deep_research_enable_refinement is False
        assert config.deep_research_enable_global_compression is False


# =============================================================================
# Tests: Phase flow (3.1)
# =============================================================================


class TestPhaseFlowSkipAnalysis:
    """Tests for ANALYSIS phase skip logic in workflow_execution."""

    def test_advance_phase_from_supervision_goes_to_analysis(self) -> None:
        """advance_phase from SUPERVISION goes to ANALYSIS (enum order)."""
        state = _make_state(phase=DeepResearchPhase.SUPERVISION)
        state.advance_phase()
        assert state.phase == DeepResearchPhase.ANALYSIS

    def test_advance_phase_from_analysis_goes_to_compression(self) -> None:
        """advance_phase from ANALYSIS goes to COMPRESSION (enum order)."""
        state = _make_state(phase=DeepResearchPhase.ANALYSIS)
        state.advance_phase()
        assert state.phase == DeepResearchPhase.COMPRESSION

    def test_advance_phase_from_compression_goes_to_synthesis(self) -> None:
        """advance_phase from COMPRESSION goes to SYNTHESIS."""
        state = _make_state(phase=DeepResearchPhase.COMPRESSION)
        state.advance_phase()
        assert state.phase == DeepResearchPhase.SYNTHESIS

    def test_skip_chain_analysis_compression_reaches_synthesis(self) -> None:
        """Two advance_phase calls from ANALYSIS (skip both) reach SYNTHESIS."""
        state = _make_state(phase=DeepResearchPhase.ANALYSIS)
        # Skip ANALYSIS → COMPRESSION
        state.advance_phase()
        assert state.phase == DeepResearchPhase.COMPRESSION
        # Skip COMPRESSION → SYNTHESIS
        state.advance_phase()
        assert state.phase == DeepResearchPhase.SYNTHESIS


class TestPhaseFlowRefinement:
    """Tests for REFINEMENT phase gating by config flag."""

    def test_should_continue_refinement_with_gaps(self) -> None:
        """should_continue_refinement returns True with unresolved gaps."""
        state = _make_state()
        state.iteration = 1
        state.max_iterations = 3
        state.add_gap("Missing information about dosage effects")
        assert state.should_continue_refinement() is True

    def test_should_continue_refinement_no_gaps(self) -> None:
        """should_continue_refinement returns False without gaps."""
        state = _make_state()
        state.iteration = 1
        state.max_iterations = 3
        assert state.should_continue_refinement() is False


# =============================================================================
# Tests: Synthesis from compressed findings (3.2)
# =============================================================================


class TestSynthesisFromCompressedFindings:
    """Tests for synthesis consuming per-topic compressed findings directly."""

    def test_compressed_findings_used_when_no_analysis_findings(self) -> None:
        """Synthesis prompt uses compressed_findings when state.findings is empty."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Research Findings by Topic" in prompt
        assert "Caffeine consumption delays sleep onset" in prompt
        # Should NOT contain "Findings to Synthesize" (that's the analysis path)
        assert "Findings to Synthesize" not in prompt

    def test_compressed_findings_show_topic_headers(self) -> None:
        """Each topic gets a section header from its sub-query text."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Sub-query 0 about caffeine" in prompt
        assert "Sub-query 1 about caffeine" in prompt

    def test_compressed_findings_include_research_brief(self) -> None:
        """Synthesis prompt includes the research brief as context."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Research Brief" in prompt
        assert "effects of caffeine on sleep quality" in prompt

    def test_compressed_findings_include_source_reference(self) -> None:
        """Synthesis prompt includes source reference section."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Source Reference" in prompt
        assert "Source 0: Caffeine Study" in prompt
        assert "Source 1: Caffeine Study" in prompt

    def test_compressed_findings_include_instructions(self) -> None:
        """Synthesis prompt includes instructions with query type hint."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Instructions" in prompt
        assert "Query type hint" in prompt

    def test_analysis_findings_path_still_works(self) -> None:
        """When analysis findings exist, standard path is used (backward compat)."""
        stub = StubSynthesis()
        state = _make_state(with_findings=True, with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        # Should use the standard findings path
        assert "Findings to Synthesize" in prompt
        # Should NOT use compressed findings path
        assert "Research Findings by Topic" not in prompt

    def test_compressed_digest_takes_priority(self) -> None:
        """Global compressed_digest takes priority over per-topic compressed findings."""
        stub = StubSynthesis()
        state = _make_state(
            with_compressed_topics=True,
            with_compressed_digest=True,
        )
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Unified Research Digest" in prompt
        # Should NOT use per-topic compressed findings
        assert "Research Findings by Topic" not in prompt

    def test_fallback_when_no_compressed_findings(self) -> None:
        """When neither findings nor compressed findings exist, synthesis gets empty report path."""
        stub = StubSynthesis()
        state = _make_state()
        # No findings, no compressed topics, no compressed digest
        # The _execute_synthesis_async would generate empty report
        # We test the user prompt builder handles this gracefully
        prompt = stub._build_synthesis_user_prompt(state)

        # Should use the standard path (empty findings section)
        assert "Findings to Synthesize" in prompt

    def test_citations_preserved_in_compressed_findings(self) -> None:
        """Citation references in compressed findings are preserved in the prompt."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        prompt = stub._build_synthesis_user_prompt(state)

        # The compressed findings contain inline citations like [1], [2]
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_contradictions_included_with_compressed_findings(self) -> None:
        """Contradictions are included in the compressed findings synthesis path."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        from foundry_mcp.core.research.models.deep_research import Contradiction

        state.contradictions.append(
            Contradiction(
                finding_ids=["find-0", "find-1"],
                description="Conflicting dosage thresholds reported",
                severity="major",
            )
        )
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Contradictions Detected" in prompt
        assert "Conflicting dosage thresholds" in prompt

    def test_gaps_included_with_compressed_findings(self) -> None:
        """Knowledge gaps are included in the compressed findings synthesis path."""
        stub = StubSynthesis()
        state = _make_state(with_compressed_topics=True)
        state.add_gap("Long-term effects of caffeine on sleep architecture")
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Knowledge Gaps Identified" in prompt
        assert "Long-term effects" in prompt


# =============================================================================
# Tests: Empty findings detection (3.2 — synthesis entry)
# =============================================================================


class TestSynthesisEntryCheck:
    """Tests for the updated empty-findings detection in synthesis."""

    def test_has_compressed_findings_detected(self) -> None:
        """Synthesis detects per-topic compressed findings as valid input."""
        state = _make_state(with_compressed_topics=True)
        has_compressed = any(
            tr.compressed_findings for tr in state.topic_research_results
        )
        assert has_compressed is True

    def test_no_material_detected(self) -> None:
        """Synthesis detects no material when all sources are empty."""
        state = _make_state()
        has_compressed = any(
            tr.compressed_findings for tr in state.topic_research_results
        )
        assert has_compressed is False
        assert len(state.findings) == 0
        assert state.compressed_digest is None

    def test_topic_results_without_compressed_findings(self) -> None:
        """Topic results without compressed_findings are not treated as valid input."""
        state = _make_state()
        state.topic_research_results.append(
            TopicResearchResult(
                sub_query_id="sq-0",
                searches_performed=2,
                sources_found=1,
                compressed_findings=None,  # Explicitly None
            )
        )
        has_compressed = any(
            tr.compressed_findings for tr in state.topic_research_results
        )
        assert has_compressed is False


# =============================================================================
# Tests: Backward compatibility — full pipeline (3.5)
# =============================================================================


class TestBackwardCompatFullPipeline:
    """Tests that all flags enabled restores the full pipeline."""

    def test_all_flags_enabled_config(self) -> None:
        """All three flags can be enabled simultaneously."""
        config = ResearchConfig.from_toml_dict({
            "deep_research_enable_analysis_digest": True,
            "deep_research_enable_global_compression": True,
            "deep_research_enable_refinement": True,
        })
        assert config.deep_research_enable_analysis_digest is True
        assert config.deep_research_enable_global_compression is True
        assert config.deep_research_enable_refinement is True

    def test_findings_path_used_with_analysis_enabled(self) -> None:
        """When analysis findings exist, standard synthesis path is used."""
        stub = StubSynthesis()
        state = _make_state(with_findings=True)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Findings to Synthesize" in prompt
        assert "Sleep Effects" in prompt  # Category header

    def test_system_prompt_unchanged(self) -> None:
        """System prompt is unaffected by pipeline collapse changes."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        # Core elements still present
        assert "research synthesizer" in prompt.lower()
        assert "Citations" in prompt
        assert "Conclusions" in prompt


# =============================================================================
# Tests: _append_contradictions_and_gaps helper
# =============================================================================


class TestAppendContradictionsAndGaps:
    """Tests for the extracted _append_contradictions_and_gaps helper."""

    def test_no_contradictions_no_gaps(self) -> None:
        """Helper produces no output when neither exists."""
        stub = StubSynthesis()
        state = _make_state()
        parts: list[str] = []
        stub._append_contradictions_and_gaps(state, parts, {})
        assert len(parts) == 0

    def test_contradictions_appended(self) -> None:
        """Helper appends contradictions section."""
        from foundry_mcp.core.research.models.deep_research import Contradiction

        stub = StubSynthesis()
        state = _make_state()
        state.contradictions.append(
            Contradiction(
                finding_ids=["f-1", "f-2"],
                description="Dosage conflict",
                severity="major",
            )
        )
        parts: list[str] = []
        stub._append_contradictions_and_gaps(state, parts, {})
        joined = "\n".join(parts)
        assert "Contradictions Detected" in joined
        assert "Dosage conflict" in joined
        assert "MAJOR" in joined

    def test_gaps_appended(self) -> None:
        """Helper appends knowledge gaps section."""
        stub = StubSynthesis()
        state = _make_state()
        state.add_gap("Missing dosage data")
        parts: list[str] = []
        stub._append_contradictions_and_gaps(state, parts, {})
        joined = "\n".join(parts)
        assert "Knowledge Gaps Identified" in joined
        assert "Missing dosage data" in joined
        assert "unresolved" in joined
