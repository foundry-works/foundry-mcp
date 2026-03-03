"""Tests for Layer 1: Structured Data Preservation in Compression.

Covers:
- _detect_structured_blocks: markdown tables, definition-style bullet lists,
  mixed content, no structured data
- _validate_structured_data_survival: tables preserved/lost, numeric token
  presence/absence
- _compression_output_is_valid: structured_blocks parameter integration
- Prompt assembly: <Structured Data Preservation> block placement
- Wiring: _compress_topic_findings_async detects blocks and passes to validation
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from foundry_mcp.core.research.models.deep_research import (
    DeepResearchState,
    TopicResearchResult,
)
from foundry_mcp.core.research.workflows.deep_research.phases.compression import (
    CompressionMixin,
    _compression_output_is_valid,
    _detect_structured_blocks,
    _validate_structured_data_survival,
)

from tests.core.research.workflows.deep_research.conftest import make_gathering_state

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TABLE_TEXT = """\
| Partner | Ratio | Annual Fee |
|---------|-------|------------|
| Aeroplan | 1:1 | $95 |
| Hilton | 2:1 | $550 |
| Marriott | 1:1.25 | $250 |
"""

DEFINITION_LIST_TEXT = """\
- **Aeroplan**: 1:1 ratio transfer
- **Hilton Honors**: 2:1 ratio, $550 annual fee
- **Marriott Bonvoy**: 1:1.25 ratio transfer
"""

MIXED_TEXT = f"""\
Some introductory text about credit card transfer partners.

{TABLE_TEXT}
Here is additional context about each program.

{DEFINITION_LIST_TEXT}
And some concluding remarks.
"""

PROSE_ONLY_TEXT = """\
Credit card transfer partners provide excellent value for travelers.
There are many options available from various issuers, each with
different benefits and annual fees. The best choice depends on
your travel patterns and preferred airlines.
"""


# ============================================================================
# _detect_structured_blocks
# ============================================================================


class TestDetectStructuredBlocks:
    """Tests for _detect_structured_blocks()."""

    def test_detects_markdown_table(self) -> None:
        blocks = _detect_structured_blocks(TABLE_TEXT)
        assert len(blocks) == 1
        assert "| Partner | Ratio |" in blocks[0]
        assert "| Aeroplan | 1:1 |" in blocks[0]
        assert "| Hilton | 2:1 |" in blocks[0]
        assert "| Marriott | 1:1.25 |" in blocks[0]

    def test_detects_definition_list(self) -> None:
        blocks = _detect_structured_blocks(DEFINITION_LIST_TEXT)
        assert len(blocks) == 1
        assert "**Aeroplan**: 1:1" in blocks[0]
        assert "**Hilton Honors**: 2:1" in blocks[0]
        assert "**Marriott Bonvoy**: 1:1.25" in blocks[0]

    def test_detects_mixed_content(self) -> None:
        blocks = _detect_structured_blocks(MIXED_TEXT)
        # Should detect both the table and the definition list
        assert len(blocks) == 2
        table_block = next(b for b in blocks if "| Partner |" in b)
        defn_block = next(b for b in blocks if "**Aeroplan**" in b)
        assert "| Aeroplan | 1:1 |" in table_block
        assert "**Hilton Honors**: 2:1" in defn_block

    def test_no_structured_data_returns_empty(self) -> None:
        blocks = _detect_structured_blocks(PROSE_ONLY_TEXT)
        assert blocks == []

    def test_empty_string_returns_empty(self) -> None:
        blocks = _detect_structured_blocks("")
        assert blocks == []

    def test_single_pipe_line_not_detected_as_table(self) -> None:
        """A single |..| line is not enough — need at least 2 rows."""
        text = "| Header | Value |\nSome other text"
        blocks = _detect_structured_blocks(text)
        # Should not detect a table from a single row
        table_blocks = [b for b in blocks if "|" in b]
        assert table_blocks == []

    def test_bullet_without_numbers_not_detected(self) -> None:
        """Bullets without numeric values should not be detected."""
        text = "- **Aeroplan**: great for flights\n- **Hilton**: good hotels"
        blocks = _detect_structured_blocks(text)
        assert blocks == []

    def test_asterisk_bullets_detected(self) -> None:
        """Asterisk-prefixed bullets should also be detected."""
        text = "* **Chase**: 1:1 ratio\n* **Amex**: 2:1 ratio"
        blocks = _detect_structured_blocks(text)
        assert len(blocks) == 1
        assert "**Chase**: 1:1" in blocks[0]

    def test_dash_separator_detected(self) -> None:
        """Definition lists using em-dash separator should be detected."""
        text = "- Aeroplan — 1:1 ratio transfer\n- Hilton — 2:1 ratio"
        blocks = _detect_structured_blocks(text)
        assert len(blocks) == 1

    def test_table_with_no_trailing_newline(self) -> None:
        """Table at end of text without trailing newline."""
        text = "| A | B |\n| 1 | 2 |"
        blocks = _detect_structured_blocks(text)
        assert len(blocks) == 1


# ============================================================================
# _validate_structured_data_survival
# ============================================================================


class TestValidateStructuredDataSurvival:
    """Tests for _validate_structured_data_survival()."""

    def test_tables_preserved_returns_true(self) -> None:
        blocks = _detect_structured_blocks(TABLE_TEXT)
        assert _validate_structured_data_survival(TABLE_TEXT, TABLE_TEXT, blocks) is True

    def test_tables_paraphrased_returns_false(self) -> None:
        blocks = _detect_structured_blocks(TABLE_TEXT)
        compressed = (
            "Various transfer partners offer different ratios. "
            "Aeroplan provides good value while Hilton offers 2-to-1."
        )
        assert _validate_structured_data_survival(TABLE_TEXT, compressed, blocks) is False

    def test_numeric_tokens_missing_returns_false(self) -> None:
        """When table rows survive but numeric values are changed."""
        blocks = _detect_structured_blocks(DEFINITION_LIST_TEXT)
        # Keep structure but change numbers
        compressed = (
            "- **Aeroplan**: excellent ratio transfer\n"
            "- **Hilton Honors**: good ratio, high annual fee\n"
            "- **Marriott Bonvoy**: decent ratio transfer\n"
        )
        assert _validate_structured_data_survival(DEFINITION_LIST_TEXT, compressed, blocks) is False

    def test_empty_blocks_returns_true(self) -> None:
        """No blocks to check means validation passes."""
        assert _validate_structured_data_survival("any text", "any compressed", []) is True

    def test_partial_table_survival_returns_false(self) -> None:
        """If some table rows are lost, validation fails."""
        blocks = _detect_structured_blocks(TABLE_TEXT)
        # Only keep 2 of 3 data rows (drop Marriott)
        compressed = (
            "| Partner | Ratio | Annual Fee |\n"
            "|---------|-------|------------|\n"
            "| Aeroplan | 1:1 | $95 |\n"
            "| Hilton | 2:1 | $550 |\n"
        )
        assert _validate_structured_data_survival(TABLE_TEXT, compressed, blocks) is False

    def test_all_table_rows_present_returns_true(self) -> None:
        """All data rows and numeric tokens present passes validation."""
        blocks = _detect_structured_blocks(TABLE_TEXT)
        # Compressed output has all the rows plus some extra text
        compressed = (
            "The transfer partners are:\n\n"
            "| Partner | Ratio | Annual Fee |\n"
            "|---------|-------|------------|\n"
            "| Aeroplan | 1:1 | $95 |\n"
            "| Hilton | 2:1 | $550 |\n"
            "| Marriott | 1:1.25 | $250 |\n"
            "\nThese are the main options.\n"
        )
        assert _validate_structured_data_survival(TABLE_TEXT, compressed, blocks) is True

    def test_definition_list_numeric_tokens_preserved(self) -> None:
        blocks = _detect_structured_blocks(DEFINITION_LIST_TEXT)
        # Reproduced with different formatting but same numbers
        compressed = (
            "Transfer partners include:\n"
            "- Aeroplan offers 1:1 ratio\n"
            "- Hilton Honors has 2:1 ratio with $550 fee\n"
            "- Marriott Bonvoy provides 1:1.25 ratio\n"
        )
        assert _validate_structured_data_survival(DEFINITION_LIST_TEXT, compressed, blocks) is True


# ============================================================================
# _compression_output_is_valid (structured_blocks integration)
# ============================================================================


class TestCompressionOutputIsValidStructuredBlocks:
    """Tests for the structured_blocks parameter on _compression_output_is_valid."""

    def _make_message_history(self, content: str) -> list[dict[str, str]]:
        return [{"role": "tool", "content": content}]

    def test_passes_without_blocks(self) -> None:
        """Existing behavior: no blocks means no structured data check."""
        compressed = "Some compressed [1] findings with http://source.com"
        history = self._make_message_history("Some original content for testing")
        assert _compression_output_is_valid(compressed, history, "topic-1") is True

    def test_passes_with_empty_blocks(self) -> None:
        compressed = "Some compressed [1] findings with http://source.com"
        history = self._make_message_history("Some original content for testing")
        assert _compression_output_is_valid(compressed, history, "topic-1", structured_blocks=[]) is True

    def test_fails_when_structured_data_lost(self) -> None:
        """When blocks are provided and data is lost, validation fails."""
        original_content = TABLE_TEXT
        history = self._make_message_history(original_content)
        blocks = _detect_structured_blocks(original_content)
        # Compressed has source refs and good length but lost tables
        compressed = (
            "Transfer partners offer various ratios including good options "
            "for travelers seeking value. [1] Source about transfers "
            "http://example.com/partners\n" * 5
        )
        assert _compression_output_is_valid(compressed, history, "topic-1", structured_blocks=blocks) is False

    def test_passes_when_structured_data_preserved(self) -> None:
        """When blocks are provided and data survives, validation passes."""
        original_content = TABLE_TEXT
        history = self._make_message_history(original_content)
        blocks = _detect_structured_blocks(original_content)
        compressed = (
            "Transfer partner information:\n\n"
            "| Partner | Ratio | Annual Fee |\n"
            "|---------|-------|------------|\n"
            "| Aeroplan | 1:1 | $95 |\n"
            "| Hilton | 2:1 | $550 |\n"
            "| Marriott | 1:1.25 | $250 |\n\n"
            "[1] Source: http://example.com/partners\n"
        )
        assert _compression_output_is_valid(compressed, history, "topic-1", structured_blocks=blocks) is True

    def test_existing_checks_still_apply_with_blocks(self) -> None:
        """Empty compressed output still fails even if blocks would pass."""
        history = self._make_message_history("content")
        assert _compression_output_is_valid(None, history, "topic-1", structured_blocks=["| A | B |"]) is False
        assert _compression_output_is_valid("", history, "topic-1", structured_blocks=["| A | B |"]) is False


# ============================================================================
# Prompt assembly
# ============================================================================


class TestPromptAssembly:
    """Verify the structured data preservation instruction is in the right place."""

    def test_preservation_block_in_system_prompt(self) -> None:
        """The <Structured Data Preservation> block must appear after
        </Citation Rules> and before 'Critical Reminder'."""
        # We read the system prompt from the source to check placement.
        # The system prompt is built inline in _compress_single_topic_async,
        # so we check the source file content.
        import inspect

        from foundry_mcp.core.research.workflows.deep_research.phases import compression

        source = inspect.getsource(compression)

        # Check that the block exists
        assert "<Structured Data Preservation>" in source
        assert "</Structured Data Preservation>" in source

        # Check ordering: Citation Rules -> Structured Data -> Critical Reminder
        citation_end = source.index("</Citation Rules>")
        struct_start = source.index("<Structured Data Preservation>")
        critical = source.index("Critical Reminder: It is extremely important")

        assert citation_end < struct_start < critical, (
            "Structured Data Preservation block must be between Citation Rules and Critical Reminder"
        )

    def test_preservation_block_content(self) -> None:
        """Verify the preservation instructions mention tables and bullet lists."""
        import inspect

        from foundry_mcp.core.research.workflows.deep_research.phases import compression

        source = inspect.getsource(compression)

        # Extract the block content
        start = source.index("<Structured Data Preservation>")
        end = source.index("</Structured Data Preservation>") + len("</Structured Data Preservation>")
        block = source[start:end]

        assert "Markdown tables" in block
        assert "VERBATIM" in block
        assert "Bulleted lists" in block
        assert "numeric values" in block


# ============================================================================
# Wiring: _compress_topic_findings_async integration
# ============================================================================


def _make_state_with_topic_results(
    message_history: list[dict[str, str]] | None = None,
) -> tuple[DeepResearchState, TopicResearchResult]:
    """Create a state with sub-queries, sources, and a TopicResearchResult."""
    state = make_gathering_state(num_sub_queries=1, sources_per_query=2)
    sq = state.sub_queries[0]
    source_ids = [s.id for s in state.sources if s.sub_query_id == sq.id]
    tr = TopicResearchResult(
        sub_query_id=sq.id,
        searches_performed=1,
        sources_found=len(source_ids),
        source_ids=source_ids,
    )
    if message_history is not None:
        tr.message_history = message_history
    state.topic_research_results.append(tr)
    return state, tr


class TestCompressionWiring:
    """Test that _compress_topic_findings_async detects blocks and passes
    them to _compression_output_is_valid."""

    def _make_workflow(self) -> Any:
        """Create a minimal mock workflow with CompressionMixin."""

        class FakeWorkflow(CompressionMixin):
            def __init__(self):
                self.config = MagicMock()
                self.config.default_provider = "test-provider"
                self.config.deep_research_compression_max_content_length = 50_000
                self.memory = MagicMock()
                self.audit_events: list[dict] = []

            def _write_audit_event(self, state, event_name, *, data=None, level="info"):
                self.audit_events.append({"event": event_name, "data": data})

            def _check_cancellation(self, state):
                pass

            async def _execute_provider_async(self, **kwargs):
                pass

        return FakeWorkflow()

    @pytest.mark.asyncio
    async def test_detect_blocks_called_for_topics_with_message_history(self) -> None:
        """Verify _detect_structured_blocks is called on message history content."""
        state, tr = _make_state_with_topic_results(
            message_history=[{"role": "tool", "content": TABLE_TEXT}]
        )

        workflow = self._make_workflow()

        # Side effect: simulate _compress_single_topic_async populating compressed_findings
        async def _fake_compress(topic_result, state, timeout):
            topic_result.compressed_findings = "compressed [1] output http://example.com"
            return (100, 50, True)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._detect_structured_blocks",
                wraps=_detect_structured_blocks,
            ) as mock_detect,
            patch.object(
                workflow, "_compress_single_topic_async", side_effect=_fake_compress
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._compression_output_is_valid",
                return_value=True,
            ) as mock_valid,
        ):
            await workflow._compress_topic_findings_async(state, max_concurrent=2, timeout=30.0)

            # _detect_structured_blocks should have been called
            mock_detect.assert_called_once()
            call_text = mock_detect.call_args[0][0]
            assert TABLE_TEXT.strip() in call_text

            # _compression_output_is_valid should receive the blocks
            mock_valid.assert_called_once()
            _, kwargs = mock_valid.call_args
            assert kwargs.get("structured_blocks") is not None
            assert len(kwargs["structured_blocks"]) > 0

    @pytest.mark.asyncio
    async def test_no_blocks_when_no_message_history(self) -> None:
        """Topics without message_history should not produce blocks."""
        state, tr = _make_state_with_topic_results(message_history=[])

        workflow = self._make_workflow()

        async def _fake_compress(topic_result, state, timeout):
            topic_result.compressed_findings = "compressed [1] output http://example.com"
            return (100, 50, True)

        with (
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._detect_structured_blocks"
            ) as mock_detect,
            patch.object(
                workflow, "_compress_single_topic_async", side_effect=_fake_compress
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._compression_output_is_valid",
                return_value=True,
            ) as mock_valid,
        ):
            await workflow._compress_topic_findings_async(state, max_concurrent=2, timeout=30.0)

            # _detect_structured_blocks should NOT have been called (no message_history)
            mock_detect.assert_not_called()

            # _compression_output_is_valid should be called with structured_blocks=None
            mock_valid.assert_called_once()
            _, kwargs = mock_valid.call_args
            assert kwargs.get("structured_blocks") is None

    @pytest.mark.asyncio
    async def test_message_history_not_cleared_when_structured_data_lost(self) -> None:
        """When structured data validation fails, message_history is retained."""
        state, tr = _make_state_with_topic_results(
            message_history=[{"role": "tool", "content": TABLE_TEXT}]
        )

        workflow = self._make_workflow()

        async def _fake_compress(topic_result, state, timeout):
            topic_result.compressed_findings = "Paraphrased text without tables"
            return (100, 50, True)

        with (
            patch.object(
                workflow, "_compress_single_topic_async", side_effect=_fake_compress
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._compression_output_is_valid",
                return_value=False,  # Simulate structured data loss
            ),
        ):
            await workflow._compress_topic_findings_async(state, max_concurrent=2, timeout=30.0)

            # message_history should NOT have been cleared
            assert len(tr.message_history) > 0
            assert tr.message_history[0]["content"] == TABLE_TEXT

    @pytest.mark.asyncio
    async def test_message_history_cleared_when_structured_data_preserved(self) -> None:
        """When validation passes, message_history is cleared normally."""
        state, tr = _make_state_with_topic_results(
            message_history=[{"role": "tool", "content": TABLE_TEXT}]
        )

        workflow = self._make_workflow()

        async def _fake_compress(topic_result, state, timeout):
            topic_result.compressed_findings = "Compressed with tables [1] http://example.com"
            return (100, 50, True)

        with (
            patch.object(
                workflow, "_compress_single_topic_async", side_effect=_fake_compress
            ),
            patch(
                "foundry_mcp.core.research.workflows.deep_research.phases.compression._compression_output_is_valid",
                return_value=True,
            ),
        ):
            await workflow._compress_topic_findings_async(state, max_concurrent=2, timeout=30.0)

            # message_history should have been cleared
            assert len(tr.message_history) == 0


# ============================================================================
# Integration: detection → validation end-to-end
# ============================================================================


class TestDetectionValidationIntegration:
    """End-to-end tests combining detection and validation."""

    def test_table_detected_and_validated_pass(self) -> None:
        """Detect a table, then validate it survives in compressed output."""
        blocks = _detect_structured_blocks(TABLE_TEXT)
        assert len(blocks) >= 1

        compressed = (
            "Transfer partners:\n\n"
            "| Partner | Ratio | Annual Fee |\n"
            "|---------|-------|------------|\n"
            "| Aeroplan | 1:1 | $95 |\n"
            "| Hilton | 2:1 | $550 |\n"
            "| Marriott | 1:1.25 | $250 |\n"
        )
        assert _validate_structured_data_survival(TABLE_TEXT, compressed, blocks) is True

    def test_table_detected_and_validated_fail(self) -> None:
        """Detect a table, validate fails when paraphrased."""
        blocks = _detect_structured_blocks(TABLE_TEXT)
        compressed = (
            "Aeroplan offers a 1:1 transfer ratio for $95/year. "
            "Hilton has a 2:1 ratio at $550/year. "
            "Marriott provides 1:1.25 ratio for $250/year."
        )
        assert _validate_structured_data_survival(TABLE_TEXT, compressed, blocks) is False

    def test_mixed_content_detection_and_validation(self) -> None:
        """Mixed content: both table and list detected, validated together."""
        blocks = _detect_structured_blocks(MIXED_TEXT)
        assert len(blocks) == 2

        # Compressed preserves everything
        compressed = MIXED_TEXT
        assert _validate_structured_data_survival(MIXED_TEXT, compressed, blocks) is True

    def test_no_false_positive_on_prose(self) -> None:
        """Prose-only text should produce no blocks and validation should pass."""
        blocks = _detect_structured_blocks(PROSE_ONLY_TEXT)
        assert blocks == []
        assert _validate_structured_data_survival(PROSE_ONLY_TEXT, "anything", blocks) is True
