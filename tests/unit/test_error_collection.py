"""Unit tests for error collection infrastructure."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from foundry_mcp.core.error_collection import (
    ErrorRecord,
    ErrorFingerprinter,
    ErrorCollector,
    get_error_collector,
    reset_error_collector,
)
from foundry_mcp.core.error_store import (
    FileErrorStore,
    get_error_store,
    reset_error_store,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for error storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def error_store(temp_storage_dir):
    """Create a FileErrorStore with temporary storage."""
    return FileErrorStore(temp_storage_dir)


@pytest.fixture
def error_collector(error_store):
    """Create an ErrorCollector with a test store."""
    collector = ErrorCollector(
        store=error_store,
        enabled=True,
        include_stack_traces=True,
        redact_inputs=True,
    )
    return collector


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global singletons before and after each test."""
    reset_error_collector()
    reset_error_store()
    yield
    reset_error_collector()
    reset_error_store()


# =============================================================================
# ErrorRecord Tests
# =============================================================================


class TestErrorRecord:
    """Tests for ErrorRecord dataclass."""

    def test_create_error_record(self):
        """Test creating an error record with required fields."""
        record = ErrorRecord(
            id="err_abc123",
            fingerprint="fp_xyz789",
            error_code="VALIDATION_ERROR",
            error_type="validation",
            tool_name="test-tool",
            correlation_id="corr_123",
            message="Test error message",
        )

        assert record.id == "err_abc123"
        assert record.fingerprint == "fp_xyz789"
        assert record.error_code == "VALIDATION_ERROR"
        assert record.tool_name == "test-tool"
        assert record.count == 1
        assert record.timestamp  # Should have auto-set timestamp

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = ErrorRecord(
            id="err_test",
            fingerprint="fp_test",
            error_code="INTERNAL_ERROR",
            error_type="internal",
            tool_name="my-tool",
            correlation_id="corr_test",
            message="Error occurred",
            exception_type="ValueError",
        )

        data = record.to_dict()
        assert data["id"] == "err_test"
        assert data["fingerprint"] == "fp_test"
        assert data["exception_type"] == "ValueError"
        # None values should not be in dict
        assert "stack_trace" not in data or data.get("stack_trace") is None

    def test_record_from_dict(self):
        """Test creating record from dictionary."""
        data = {
            "id": "err_from_dict",
            "fingerprint": "fp_from_dict",
            "error_code": "NOT_FOUND",
            "error_type": "not_found",
            "tool_name": "dict-tool",
            "correlation_id": "corr_dict",
            "message": "Not found",
            "count": 5,
        }

        record = ErrorRecord.from_dict(data)
        assert record.id == "err_from_dict"
        assert record.fingerprint == "fp_from_dict"
        assert record.count == 5


# =============================================================================
# ErrorFingerprinter Tests
# =============================================================================


class TestErrorFingerprinter:
    """Tests for error fingerprinting."""

    def test_fingerprint_basic(self):
        """Test basic fingerprint generation."""
        fingerprinter = ErrorFingerprinter()
        fp = fingerprinter.fingerprint(
            error_code="VALIDATION_ERROR",
            error_type="validation",
            tool_name="test-tool",
            exception_type="ValueError",
            message="Invalid value",
        )

        assert fp  # Non-empty
        assert len(fp) == 16  # SHA256[:16]
        assert fp.isalnum()  # Hex characters only

    def test_fingerprint_consistency(self):
        """Test that same inputs produce same fingerprint."""
        fingerprinter = ErrorFingerprinter()

        fp1 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Same message",
        )
        fp2 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Same message",
        )

        assert fp1 == fp2

    def test_fingerprint_different_tools(self):
        """Test that different tools produce different fingerprints."""
        fingerprinter = ErrorFingerprinter()

        fp1 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool-a",
            exception_type="Exception",
            message="Message",
        )
        fp2 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool-b",
            exception_type="Exception",
            message="Message",
        )

        assert fp1 != fp2

    def test_message_normalization_uuids(self):
        """Test that UUIDs in messages are normalized."""
        fingerprinter = ErrorFingerprinter()

        fp1 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Error for id 550e8400-e29b-41d4-a716-446655440000",
        )
        fp2 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Error for id 123e4567-e89b-12d3-a456-426614174000",
        )

        # Should be same after normalization
        assert fp1 == fp2

    def test_message_normalization_timestamps(self):
        """Test that timestamps in messages are normalized."""
        fingerprinter = ErrorFingerprinter()

        fp1 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Error at 2024-01-15T10:30:00Z",
        )
        fp2 = fingerprinter.fingerprint(
            error_code="ERROR",
            error_type="internal",
            tool_name="tool",
            exception_type="Exception",
            message="Error at 2024-12-25T23:59:59Z",
        )

        assert fp1 == fp2


# =============================================================================
# FileErrorStore Tests
# =============================================================================


class TestFileErrorStore:
    """Tests for file-based error storage."""

    def test_append_and_get(self, error_store):
        """Test appending and retrieving an error."""
        record = ErrorRecord(
            id="err_append_test",
            fingerprint="fp_append_test",
            error_code="TEST_ERROR",
            error_type="test",
            tool_name="test-tool",
            correlation_id="corr_test",
            message="Test error",
        )

        error_store.append(record)
        retrieved = error_store.get("err_append_test")

        assert retrieved is not None
        assert retrieved.id == "err_append_test"
        assert retrieved.fingerprint == "fp_append_test"
        assert retrieved.message == "Test error"

    def test_get_nonexistent(self, error_store):
        """Test retrieving non-existent error returns None."""
        result = error_store.get("err_nonexistent")
        assert result is None

    def test_query_by_tool_name(self, error_store):
        """Test querying errors by tool name."""
        # Add some errors
        for i in range(3):
            error_store.append(
                ErrorRecord(
                    id=f"err_tool_a_{i}",
                    fingerprint=f"fp_{i}",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool-a",
                    correlation_id="corr",
                    message=f"Error {i}",
                )
            )
        for i in range(2):
            error_store.append(
                ErrorRecord(
                    id=f"err_tool_b_{i}",
                    fingerprint=f"fp_b_{i}",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool-b",
                    correlation_id="corr",
                    message=f"Error B {i}",
                )
            )

        # Query tool-a only
        results = error_store.query(tool_name="tool-a")
        assert len(results) == 3
        assert all(r.tool_name == "tool-a" for r in results)

    def test_query_by_error_code(self, error_store):
        """Test querying errors by error code."""
        error_store.append(
            ErrorRecord(
                id="err_val",
                fingerprint="fp_val",
                error_code="VALIDATION_ERROR",
                error_type="validation",
                tool_name="tool",
                correlation_id="corr",
                message="Validation error",
            )
        )
        error_store.append(
            ErrorRecord(
                id="err_int",
                fingerprint="fp_int",
                error_code="INTERNAL_ERROR",
                error_type="internal",
                tool_name="tool",
                correlation_id="corr",
                message="Internal error",
            )
        )

        results = error_store.query(error_code="VALIDATION_ERROR")
        assert len(results) == 1
        assert results[0].error_code == "VALIDATION_ERROR"

    def test_query_pagination(self, error_store):
        """Test query pagination."""
        # Add 10 errors
        for i in range(10):
            error_store.append(
                ErrorRecord(
                    id=f"err_page_{i}",
                    fingerprint=f"fp_page_{i}",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool",
                    correlation_id="corr",
                    message=f"Error {i}",
                )
            )

        # First page
        page1 = error_store.query(limit=3)
        assert len(page1) == 3

        # Second page
        page2 = error_store.query(limit=3, offset=3)
        assert len(page2) == 3

        # Ensure different records
        page1_ids = {r.id for r in page1}
        page2_ids = {r.id for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_count(self, error_store):
        """Test error count."""
        assert error_store.count() == 0

        for i in range(5):
            error_store.append(
                ErrorRecord(
                    id=f"err_count_{i}",
                    fingerprint=f"fp_{i}",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool",
                    correlation_id="corr",
                    message=f"Error {i}",
                )
            )

        assert error_store.count() == 5

    def test_get_stats(self, error_store):
        """Test getting aggregated statistics."""
        error_store.append(
            ErrorRecord(
                id="err_1",
                fingerprint="fp_shared",
                error_code="ERROR_A",
                error_type="type_a",
                tool_name="tool-1",
                correlation_id="corr",
                message="Error",
            )
        )
        error_store.append(
            ErrorRecord(
                id="err_2",
                fingerprint="fp_shared",
                error_code="ERROR_A",
                error_type="type_a",
                tool_name="tool-1",
                correlation_id="corr",
                message="Error",
            )
        )
        error_store.append(
            ErrorRecord(
                id="err_3",
                fingerprint="fp_other",
                error_code="ERROR_B",
                error_type="type_b",
                tool_name="tool-2",
                correlation_id="corr",
                message="Other",
            )
        )

        stats = error_store.get_stats()
        assert stats["total_errors"] == 3
        assert stats["unique_patterns"] == 2
        assert stats["by_tool"]["tool-1"] == 2
        assert stats["by_tool"]["tool-2"] == 1

    def test_get_patterns(self, error_store):
        """Test getting recurring patterns."""
        # Add pattern with 5 occurrences
        for i in range(5):
            error_store.append(
                ErrorRecord(
                    id=f"err_pattern_{i}",
                    fingerprint="fp_recurring",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool",
                    correlation_id="corr",
                    message="Recurring error",
                )
            )

        # Add pattern with 2 occurrences
        for i in range(2):
            error_store.append(
                ErrorRecord(
                    id=f"err_rare_{i}",
                    fingerprint="fp_rare",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool",
                    correlation_id="corr",
                    message="Rare error",
                )
            )

        # min_count=3 should only return the recurring pattern
        patterns = error_store.get_patterns(min_count=3)
        assert len(patterns) == 1
        assert patterns[0]["fingerprint"] == "fp_recurring"
        assert patterns[0]["count"] == 5

    def test_cleanup_by_max_errors(self, error_store):
        """Test cleanup enforces max_errors limit."""
        for i in range(10):
            error_store.append(
                ErrorRecord(
                    id=f"err_cleanup_{i}",
                    fingerprint=f"fp_{i}",
                    error_code="ERROR",
                    error_type="test",
                    tool_name="tool",
                    correlation_id="corr",
                    message=f"Error {i}",
                )
            )

        assert error_store.count() == 10

        # Cleanup to keep only 5
        deleted = error_store.cleanup(retention_days=365, max_errors=5)
        assert deleted == 5
        assert error_store.count() == 5


# =============================================================================
# ErrorCollector Tests
# =============================================================================


class TestErrorCollector:
    """Tests for ErrorCollector."""

    def test_collect_tool_error(self, error_collector):
        """Test collecting a tool error."""
        try:
            raise ValueError("Test validation error")
        except ValueError as e:
            error_collector.collect_tool_error(
                tool_name="test-tool",
                error=e,
                input_params={"key": "value"},
                duration_ms=100.5,
            )

        # Verify error was stored
        store = error_collector.store
        assert store.count() == 1

        records = store.query()
        assert len(records) == 1
        record = records[0]
        assert record.tool_name == "test-tool"
        assert record.exception_type == "ValueError"
        assert "Test validation error" in record.message

    def test_collect_provider_error(self, error_collector):
        """Test collecting an AI provider error."""
        error_collector.collect_provider_error(
            provider_id="gemini",
            error=TimeoutError("Request timed out"),
            request_context={
                "workflow": "plan_review",
                "prompt_id": "analyze",
            },
        )

        store = error_collector.store
        assert store.count() == 1

        records = store.query(provider_id="gemini")
        assert len(records) == 1
        record = records[0]
        assert record.provider_id == "gemini"
        assert record.exception_type == "TimeoutError"

    def test_collector_disabled(self, error_store):
        """Test that disabled collector doesn't collect errors."""
        collector = ErrorCollector(
            store=error_store,
            enabled=False,
        )

        try:
            raise RuntimeError("Test error")
        except RuntimeError as e:
            collector.collect_tool_error(
                tool_name="tool",
                error=e,
                input_params={},
                duration_ms=0,
            )

        # Should not have collected
        assert error_store.count() == 0

    def test_input_redaction(self, error_collector):
        """Test that sensitive inputs are redacted."""
        try:
            raise ValueError("Error")
        except ValueError as e:
            error_collector.collect_tool_error(
                tool_name="tool",
                error=e,
                input_params={
                    "api_key": "secret-key-12345",
                    "password": "my-password",
                    "normal_param": "visible",
                },
                duration_ms=0,
            )

        records = error_collector.store.query()
        record = records[0]

        # Check input_summary was created and redacted
        assert record.input_summary is not None
        summary = record.input_summary

        # Sensitive keys should be fully redacted
        assert summary["api_key"] == "<REDACTED>"
        assert summary["password"] == "<REDACTED>"

        # Normal params should be visible
        assert summary["normal_param"] == "visible"

    def test_exception_to_error_code_mapping(self, error_collector):
        """Test that exceptions are mapped to correct error codes."""
        try:
            raise FileNotFoundError("File not found")
        except FileNotFoundError as e:
            error_collector.collect_tool_error(
                tool_name="tool",
                error=e,
                input_params={},
                duration_ms=0,
            )

        records = error_collector.store.query()
        record = records[0]
        assert record.error_code == "NOT_FOUND"

    def test_initialize_method(self, error_store):
        """Test the initialize method updates collector settings."""
        collector = ErrorCollector(enabled=False)

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.include_stack_traces = False
        mock_config.redact_inputs = True

        collector.initialize(error_store, mock_config)

        assert collector.is_enabled()
        assert collector._include_stack_traces is False
        assert collector._redact_inputs is True


# =============================================================================
# Global Function Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for module-level functions."""

    def test_get_error_collector_singleton(self, temp_storage_dir):
        """Test that get_error_collector returns singleton."""
        collector1 = get_error_collector()
        collector2 = get_error_collector()
        assert collector1 is collector2

    def test_get_error_store_singleton(self, temp_storage_dir):
        """Test that get_error_store returns singleton."""
        store1 = get_error_store(temp_storage_dir)
        store2 = get_error_store()  # Should return same instance
        assert store1 is store2

    def test_reset_error_collector(self):
        """Test that reset_error_collector clears singleton."""
        collector1 = get_error_collector()
        reset_error_collector()
        collector2 = get_error_collector()
        # After reset, should be a new instance
        assert collector1 is not collector2
