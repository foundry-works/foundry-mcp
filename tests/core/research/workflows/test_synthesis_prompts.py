"""Tests for Phase 3: Synthesis Prompt Engineering.

Tests cover:
1. Query-type classification — comparison, enumeration, howto, explanation
2. Language matching directive — system prompt instructs language detection
3. Structure-adaptive directives — query type selects structure guidance
4. Anti-pattern guardrails — system prompt bans meta-commentary and hedging
5. Citation format — system prompt enforces inline [N] + auto-appended sources
6. Query-type hint in user prompt — structural hint included in instructions
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.research.models.deep_research import (
    DeepResearchPhase,
    DeepResearchState,
)
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.models.sources import (
    ResearchFinding,
    ResearchSource,
    SourceQuality,
)
from foundry_mcp.core.research.workflows.deep_research.phases.synthesis import (
    SynthesisPhaseMixin,
    _classify_query_type,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_state(
    query: str = "What are the effects of caffeine on sleep?",
    num_findings: int = 2,
) -> DeepResearchState:
    """Create a DeepResearchState with findings for synthesis testing."""
    state = DeepResearchState(
        id="deepres-test-synth-prompt",
        original_query=query,
        phase=DeepResearchPhase.SYNTHESIS,
        iteration=1,
        max_iterations=3,
    )
    for i in range(num_findings):
        state.findings.append(
            ResearchFinding(
                id=f"find-{i}",
                content=f"Finding {i} about the topic",
                confidence=ConfidenceLevel.MEDIUM,
                source_ids=[f"src-{i}"],
                category="General",
            )
        )
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


class StubSynthesis(SynthesisPhaseMixin):
    """Concrete class inheriting SynthesisPhaseMixin for testing."""

    def __init__(self) -> None:
        self.config = MagicMock()
        self.memory = MagicMock()


# =============================================================================
# Unit tests: _classify_query_type
# =============================================================================


class TestClassifyQueryType:
    """Tests for the module-level _classify_query_type() function."""

    def test_comparison_vs(self) -> None:
        assert _classify_query_type("React vs Vue for web development") == "comparison"

    def test_comparison_versus(self) -> None:
        assert _classify_query_type("Python versus JavaScript") == "comparison"

    def test_comparison_compare(self) -> None:
        assert _classify_query_type("Compare PostgreSQL and MySQL") == "comparison"

    def test_comparison_differ(self) -> None:
        assert _classify_query_type("What are the differences between TCP and UDP?") == "comparison"

    def test_comparison_between_and(self) -> None:
        assert _classify_query_type("between Redis and Memcached") == "comparison"

    def test_enumeration_list(self) -> None:
        assert _classify_query_type("List the best Python frameworks") == "enumeration"

    def test_enumeration_top_n(self) -> None:
        assert _classify_query_type("Top 5 machine learning algorithms") == "enumeration"

    def test_enumeration_alternatives(self) -> None:
        assert _classify_query_type("Alternatives to Docker for containerization") == "enumeration"

    def test_enumeration_types_of(self) -> None:
        assert _classify_query_type("Types of database indexes") == "enumeration"

    def test_howto_how_to(self) -> None:
        assert _classify_query_type("How to deploy a FastAPI application") == "howto"

    def test_howto_steps_to(self) -> None:
        assert _classify_query_type("Steps to configure Nginx reverse proxy") == "howto"

    def test_howto_guide_to(self) -> None:
        assert _classify_query_type("Guide to setting up CI/CD") == "howto"

    def test_howto_implement(self) -> None:
        assert _classify_query_type("How to implement OAuth2 authentication") == "howto"

    def test_explanation_default(self) -> None:
        assert _classify_query_type("What are the effects of caffeine on sleep?") == "explanation"

    def test_explanation_general(self) -> None:
        assert _classify_query_type("Explain quantum computing") == "explanation"

    def test_explanation_overview(self) -> None:
        assert _classify_query_type("Overview of transformer architectures") == "explanation"

    def test_empty_query(self) -> None:
        assert _classify_query_type("") == "explanation"


# =============================================================================
# Tests: Language directive in system prompt (3.1, 3.6)
# =============================================================================


class TestLanguageDirective:
    """Tests for language matching in the synthesis system prompt."""

    def test_system_prompt_contains_language_directive(self) -> None:
        """System prompt instructs the LLM to detect and match query language."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        assert "language" in prompt.lower()
        assert "same language" in prompt.lower()

    def test_language_directive_mentions_non_english_example(self) -> None:
        """System prompt mentions a non-English language as example."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        # Should mention at least one non-English language as an example
        assert "Chinese" in prompt or "chinese" in prompt

    def test_non_english_query_still_builds_prompt(self) -> None:
        """Non-English query successfully builds system and user prompts."""
        stub = StubSynthesis()
        state = _make_state(query="咖啡因对睡眠有什么影响？")
        system_prompt = stub._build_synthesis_system_prompt(state)
        user_prompt = stub._build_synthesis_user_prompt(state)

        # System prompt should still be valid
        assert len(system_prompt) > 100
        # User prompt should include the Chinese query
        assert "咖啡因" in user_prompt


# =============================================================================
# Tests: Structure-adaptive directives (3.2, 3.7)
# =============================================================================


class TestStructureAdaptive:
    """Tests for query-type-aware structure guidance in system prompt."""

    def test_comparison_query_gets_comparison_structure(self) -> None:
        """Comparison query includes comparison-specific structure guidance."""
        stub = StubSynthesis()
        state = _make_state(query="React vs Vue for building web apps")
        prompt = stub._build_synthesis_system_prompt(state)

        assert "comparison" in prompt.lower()
        assert "Comparative Analysis" in prompt or "Overview of [Subject A]" in prompt

    def test_enumeration_query_gets_enumeration_structure(self) -> None:
        """Enumeration query includes list-specific structure guidance."""
        stub = StubSynthesis()
        state = _make_state(query="Top 5 Python web frameworks")
        prompt = stub._build_synthesis_system_prompt(state)

        assert "enumeration" in prompt.lower() or "list" in prompt.lower()

    def test_howto_query_gets_howto_structure(self) -> None:
        """How-to query includes step-by-step structure guidance."""
        stub = StubSynthesis()
        state = _make_state(query="How to deploy a FastAPI application")
        prompt = stub._build_synthesis_system_prompt(state)

        assert "Step" in prompt
        assert "Prerequisites" in prompt

    def test_explanation_query_gets_default_structure(self) -> None:
        """General explanation query includes thematic structure guidance."""
        stub = StubSynthesis()
        state = _make_state(query="What is quantum computing?")
        prompt = stub._build_synthesis_system_prompt(state)

        assert "Key Findings" in prompt
        assert "Theme/Category" in prompt

    def test_all_structures_include_conclusions(self) -> None:
        """Every query type's system prompt includes a Conclusions requirement."""
        stub = StubSynthesis()
        for query, expected_type in [
            ("React vs Vue", "comparison"),
            ("Top 5 frameworks", "enumeration"),
            ("How to deploy", "howto"),
            ("What is quantum computing?", "explanation"),
        ]:
            state = _make_state(query=query)
            prompt = stub._build_synthesis_system_prompt(state)
            assert "Conclusions" in prompt, f"Missing Conclusions for query type {expected_type}"


# =============================================================================
# Tests: Anti-pattern guardrails (3.3, 3.8)
# =============================================================================


class TestAntiPatternGuardrails:
    """Tests for anti-pattern directives in the synthesis system prompt."""

    def _get_system_prompt(self) -> str:
        stub = StubSynthesis()
        state = _make_state()
        return stub._build_synthesis_system_prompt(state)

    def test_bans_meta_commentary(self) -> None:
        """System prompt explicitly bans meta-commentary phrases."""
        prompt = self._get_system_prompt()
        assert "based on the research" in prompt.lower()
        assert "meta-commentary" in prompt.lower()

    def test_bans_hedging_openers(self) -> None:
        """System prompt explicitly bans hedging openers."""
        prompt = self._get_system_prompt()
        assert "it appears that" in prompt.lower()
        assert "it seems" in prompt.lower()

    def test_bans_self_reference(self) -> None:
        """System prompt explicitly bans self-referential language."""
        prompt = self._get_system_prompt()
        assert "as an ai" in prompt.lower()

    def test_instructs_direct_writing(self) -> None:
        """System prompt instructs direct, authoritative writing."""
        prompt = self._get_system_prompt()
        assert "direct" in prompt.lower() or "authoritative" in prompt.lower()


# =============================================================================
# Tests: Citation format (3.4, 3.9)
# =============================================================================


class TestCitationFormat:
    """Tests for citation formatting rules in the synthesis prompt."""

    def test_system_prompt_specifies_inline_numbered_citations(self) -> None:
        """System prompt instructs inline [N] citation format."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        assert "[N]" in prompt
        assert "inline" in prompt.lower()

    def test_system_prompt_prohibits_sources_section(self) -> None:
        """System prompt tells LLM NOT to generate Sources section."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        assert "do not" in prompt.lower()
        assert "Sources" in prompt

    def test_conflicting_information_section_preserved(self) -> None:
        """System prompt still mentions 'Conflicting Information' (existing test compat)."""
        stub = StubSynthesis()
        state = _make_state()
        prompt = stub._build_synthesis_system_prompt(state)

        assert "Conflicting Information" in prompt


# =============================================================================
# Tests: Query-type hint in user prompt (3.5)
# =============================================================================


class TestQueryTypeHintInUserPrompt:
    """Tests for query-type structural hint in the synthesis user prompt."""

    def test_comparison_hint_in_user_prompt(self) -> None:
        """Comparison query adds comparison hint to user prompt."""
        stub = StubSynthesis()
        state = _make_state(query="React vs Vue for web development")
        prompt = stub._build_synthesis_user_prompt(state)

        assert "comparison" in prompt.lower()
        assert "Query type hint" in prompt

    def test_enumeration_hint_in_user_prompt(self) -> None:
        """Enumeration query adds enumeration hint to user prompt."""
        stub = StubSynthesis()
        state = _make_state(query="List the best Python frameworks")
        prompt = stub._build_synthesis_user_prompt(state)

        assert "enumeration" in prompt.lower() or "list" in prompt.lower()
        assert "Query type hint" in prompt

    def test_howto_hint_in_user_prompt(self) -> None:
        """How-to query adds procedural hint to user prompt."""
        stub = StubSynthesis()
        state = _make_state(query="How to deploy a FastAPI application")
        prompt = stub._build_synthesis_user_prompt(state)

        assert "how-to" in prompt.lower() or "step-by-step" in prompt.lower()
        assert "Query type hint" in prompt

    def test_explanation_hint_in_user_prompt(self) -> None:
        """Explanation query adds overview hint to user prompt."""
        stub = StubSynthesis()
        state = _make_state(query="What are the effects of caffeine on sleep?")
        prompt = stub._build_synthesis_user_prompt(state)

        assert "explanation" in prompt.lower() or "overview" in prompt.lower()
        assert "Query type hint" in prompt

    def test_user_prompt_still_contains_findings(self) -> None:
        """User prompt still includes all findings (no regression)."""
        stub = StubSynthesis()
        state = _make_state(num_findings=3)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Finding 0" in prompt
        assert "Finding 1" in prompt
        assert "Finding 2" in prompt
        assert "Findings to Synthesize" in prompt

    def test_user_prompt_still_contains_sources(self) -> None:
        """User prompt still includes source references (no regression)."""
        stub = StubSynthesis()
        state = _make_state(num_findings=2)
        prompt = stub._build_synthesis_user_prompt(state)

        assert "Source Reference" in prompt
        assert "Source 0" in prompt
        assert "Source 1" in prompt
