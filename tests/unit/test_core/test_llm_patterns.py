"""Tests for foundry_mcp.core.llm_patterns module."""


from foundry_mcp.core.llm_patterns import (
    # Progressive disclosure
    DetailLevel,
    DisclosureConfig,
    progressive_disclosure,
    get_detail_level,
    # Batch operations
    BatchItemResult,
    BatchResult,
    batch_response,
    paginated_batch_response,
)


# =============================================================================
# DetailLevel Tests
# =============================================================================


class TestDetailLevel:
    """Tests for DetailLevel enum."""

    def test_values(self):
        """Test enum values."""
        assert DetailLevel.SUMMARY.value == "summary"
        assert DetailLevel.STANDARD.value == "standard"
        assert DetailLevel.FULL.value == "full"

    def test_string_enum(self):
        """Test DetailLevel is a string enum."""
        assert str(DetailLevel.SUMMARY) == "DetailLevel.SUMMARY"
        assert DetailLevel.SUMMARY == "summary"


class TestGetDetailLevel:
    """Tests for get_detail_level function."""

    def test_parse_valid_levels(self):
        """Test parsing valid level strings."""
        assert get_detail_level("summary") == DetailLevel.SUMMARY
        assert get_detail_level("standard") == DetailLevel.STANDARD
        assert get_detail_level("full") == DetailLevel.FULL

    def test_case_insensitive(self):
        """Test parsing is case-insensitive."""
        assert get_detail_level("SUMMARY") == DetailLevel.SUMMARY
        assert get_detail_level("Summary") == DetailLevel.SUMMARY

    def test_default_on_none(self):
        """Test returns default when None."""
        assert get_detail_level(None) == DetailLevel.STANDARD
        assert get_detail_level(None, default=DetailLevel.FULL) == DetailLevel.FULL

    def test_default_on_invalid(self):
        """Test returns default on invalid input."""
        assert get_detail_level("invalid") == DetailLevel.STANDARD
        assert get_detail_level("xyz", default=DetailLevel.SUMMARY) == DetailLevel.SUMMARY


# =============================================================================
# DisclosureConfig Tests
# =============================================================================


class TestDisclosureConfig:
    """Tests for DisclosureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DisclosureConfig()

        assert "id" in config.summary_fields
        assert "name" in config.summary_fields
        assert "description" in config.standard_fields
        assert "metadata" in config.full_fields

    def test_custom_config(self):
        """Test custom configuration."""
        config = DisclosureConfig(
            summary_fields=["id", "title"],
            max_list_items={DetailLevel.SUMMARY: 3},
        )

        assert config.summary_fields == ["id", "title"]
        assert config.max_list_items[DetailLevel.SUMMARY] == 3


# =============================================================================
# Progressive Disclosure Tests
# =============================================================================


class TestProgressiveDisclosure:
    """Tests for progressive_disclosure function."""

    def test_summary_level_filters_fields(self):
        """Test SUMMARY level only includes summary fields."""
        data = {
            "id": "123",
            "name": "Test",
            "status": "active",
            "description": "Long description...",
            "metadata": {"complex": "data"},
        }

        result = progressive_disclosure(data, level=DetailLevel.SUMMARY)

        assert "id" in result
        assert "name" in result
        assert "status" in result
        assert "description" not in result or result.get("_truncated")
        assert "metadata" not in result or result.get("_truncated")

    def test_standard_level_includes_more(self):
        """Test STANDARD level includes standard fields."""
        data = {
            "id": "123",
            "name": "Test",
            "description": "A description",
            "created_at": "2025-01-01",
            "metadata": {"hidden": True},
        }

        result = progressive_disclosure(data, level=DetailLevel.STANDARD)

        assert "id" in result
        assert "description" in result
        assert "created_at" in result

    def test_full_level_includes_all(self):
        """Test FULL level includes all fields."""
        data = {
            "id": "123",
            "name": "Test",
            "description": "Description",
            "metadata": {"complex": "data"},
            "custom_field": "value",
        }

        result = progressive_disclosure(data, level=DetailLevel.FULL)

        assert "id" in result
        assert "metadata" in result
        assert "custom_field" in result

    def test_truncation_info_included(self):
        """Test _truncated metadata is included."""
        data = {
            "id": "123",
            "name": "Test",
            "hidden_field": "should be omitted",
        }

        result = progressive_disclosure(
            data,
            level=DetailLevel.SUMMARY,
            include_truncation_info=True,
        )

        assert "_truncated" in result
        assert "hidden_field" in result["_truncated"]["omitted_fields"]

    def test_truncation_info_can_be_disabled(self):
        """Test _truncated can be disabled."""
        data = {"id": "123", "hidden": "data"}

        result = progressive_disclosure(
            data,
            level=DetailLevel.SUMMARY,
            include_truncation_info=False,
        )

        assert "_truncated" not in result

    def test_list_input_truncated(self):
        """Test list input is truncated by max_list_items."""
        items = [{"id": i, "name": f"Item {i}"} for i in range(100)]

        result = progressive_disclosure(items, level=DetailLevel.SUMMARY)

        assert "items" in result
        assert len(result["items"]) <= 5  # Default summary max is 5
        assert result["total"] == 100

    def test_string_truncation(self):
        """Test long strings are truncated."""
        data = {
            "id": "123",
            "name": "Short",
            "status": "x" * 500,  # Long string
        }

        result = progressive_disclosure(data, level=DetailLevel.SUMMARY)

        # At summary level, long strings should be truncated
        if "status" in result:
            assert len(result["status"]) <= 103  # 100 + "..."

    def test_custom_config(self):
        """Test with custom DisclosureConfig."""
        config = DisclosureConfig(
            summary_fields=["title"],
            max_list_items={DetailLevel.SUMMARY: 2},
        )
        data = {"title": "Test", "other": "ignored"}

        result = progressive_disclosure(
            data,
            level=DetailLevel.SUMMARY,
            config=config,
        )

        assert "title" in result


# =============================================================================
# BatchItemResult Tests
# =============================================================================


class TestBatchItemResult:
    """Tests for BatchItemResult dataclass."""

    def test_success_result(self):
        """Test successful result to_dict."""
        result = BatchItemResult(
            item_id="123",
            success=True,
            result={"data": "value"},
        )
        d = result.to_dict()

        assert d["item_id"] == "123"
        assert d["success"] is True
        assert d["result"] == {"data": "value"}
        assert "error" not in d

    def test_error_result(self):
        """Test error result to_dict."""
        result = BatchItemResult(
            item_id="456",
            success=False,
            error="Not found",
            error_code="NOT_FOUND",
        )
        d = result.to_dict()

        assert d["item_id"] == "456"
        assert d["success"] is False
        assert d["error"] == "Not found"
        assert d["error_code"] == "NOT_FOUND"
        assert "result" not in d


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_all_succeeded(self):
        """Test all_succeeded property."""
        result = BatchResult(total=5, succeeded=5, failed=0)
        assert result.all_succeeded

        result_with_fail = BatchResult(total=5, succeeded=4, failed=1)
        assert not result_with_fail.all_succeeded

    def test_success_rate(self):
        """Test success_rate calculation."""
        result = BatchResult(total=10, succeeded=8, failed=2)
        assert result.success_rate == 80.0

    def test_success_rate_empty(self):
        """Test success_rate with empty batch."""
        result = BatchResult(total=0, succeeded=0, failed=0)
        assert result.success_rate == 100.0

    def test_to_response(self):
        """Test to_response format."""
        batch = BatchResult(
            total=3,
            succeeded=2,
            failed=1,
            results=[
                BatchItemResult(item_id="1", success=True, result="ok"),
                BatchItemResult(item_id="2", success=True, result="ok"),
            ],
            errors=[
                BatchItemResult(item_id="3", success=False, error="Failed"),
            ],
        )

        response = batch.to_response()

        assert "summary" in response
        assert "counts" in response
        assert response["counts"]["total"] == 3
        assert response["counts"]["succeeded"] == 2
        assert response["counts"]["failed"] == 1
        assert "results" in response
        assert "errors" in response

    def test_to_response_without_details(self):
        """Test to_response with include_details=False."""
        batch = BatchResult(
            total=2,
            succeeded=2,
            failed=0,
            results=[BatchItemResult(item_id="1", success=True, result="ok")],
        )

        response = batch.to_response(include_details=False)

        assert "results" not in response
        assert "errors" not in response


# =============================================================================
# batch_response Tests
# =============================================================================


class TestBatchResponse:
    """Tests for batch_response function."""

    def test_basic_response(self):
        """Test basic batch response creation."""
        results = [{"id": "1", "data": "a"}, {"id": "2", "data": "b"}]
        errors = [{"item_id": "3", "error": "Failed", "error_code": "ERROR"}]

        response = batch_response(results, errors)

        assert response["summary"] == "Processed 2/3 items successfully"
        assert response["counts"]["total"] == 3
        assert response["counts"]["succeeded"] == 2
        assert response["counts"]["failed"] == 1
        assert len(response["results"]) == 2
        assert len(response["errors"]) == 1

    def test_all_success(self):
        """Test batch with all successes."""
        results = [{"id": i} for i in range(5)]
        response = batch_response(results)

        assert response["counts"]["succeeded"] == 5
        assert response["counts"]["failed"] == 0
        assert "errors" not in response or response["errors"] == []

    def test_with_warnings(self):
        """Test batch with warnings."""
        response = batch_response(
            results=[{"id": "1"}],
            warnings=["Some items skipped"],
        )

        assert "warnings" in response
        assert "Some items skipped" in response["warnings"]

    def test_custom_total(self):
        """Test custom total count."""
        response = batch_response(results=[{"id": "1"}], total=10)

        assert response["counts"]["total"] == 10


# =============================================================================
# paginated_batch_response Tests
# =============================================================================


class TestPaginatedBatchResponse:
    """Tests for paginated_batch_response function."""

    def test_includes_pagination(self):
        """Test pagination metadata is included."""
        response = paginated_batch_response(
            results=[{"id": i} for i in range(10)],
            page_size=10,
            offset=0,
            total=50,
        )

        assert "pagination" in response
        assert response["pagination"]["offset"] == 0
        assert response["pagination"]["page_size"] == 10
        assert response["pagination"]["total"] == 50
        assert response["pagination"]["has_more"] is True
        assert response["pagination"]["next_offset"] == 10

    def test_last_page(self):
        """Test last page has_more=False."""
        response = paginated_batch_response(
            results=[{"id": i} for i in range(5)],
            page_size=10,
            offset=45,
            total=50,
        )

        assert response["pagination"]["has_more"] is False
        assert response["pagination"]["next_offset"] is None

    def test_adds_warning_for_more(self):
        """Test warning is added when more results available."""
        response = paginated_batch_response(
            results=[{"id": i} for i in range(10)],
            page_size=10,
            offset=0,
            total=100,
        )

        assert "warnings" in response
        assert any("90 more available" in w for w in response["warnings"])

    def test_no_warning_on_last_page(self):
        """Test no pagination warning on last page."""
        response = paginated_batch_response(
            results=[{"id": i} for i in range(10)],
            page_size=10,
            offset=90,
            total=100,
        )

        # Should not have pagination warning (might have other warnings)
        warnings = response.get("warnings", [])
        assert not any("more available" in str(w) for w in warnings)
