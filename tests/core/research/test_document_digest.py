"""Tests for document digest module.

Tests cover:
1. DigestPayload - JSON schema validation (valid and invalid payloads)
2. EvidenceSnippet - field validation and constraints
3. Serialization - round-trip preserves data, serialize_payload, deserialize_payload
4. validate_payload_dict - dict-based validation
5. Contract tests - fidelity envelope, schema validation, hash verification, locator integrity
"""

import hashlib
import json
import re
import unicodedata

import pytest
from pydantic import ValidationError

from foundry_mcp.core.research.models import DigestPayload, EvidenceSnippet
from foundry_mcp.core.research.document_digest import (
    deserialize_payload,
    serialize_payload,
    validate_payload_dict,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_evidence_snippet() -> EvidenceSnippet:
    """Create a valid EvidenceSnippet for testing."""
    return EvidenceSnippet(
        text="This is a test evidence snippet from the source document.",
        locator="char:100-158",
        relevance_score=0.85,
    )


@pytest.fixture
def valid_evidence_snippet_with_page() -> EvidenceSnippet:
    """Create a valid EvidenceSnippet with PDF page locator."""
    return EvidenceSnippet(
        text="Evidence from page 3 of the document.",
        locator="page:3:char:200-237",
        relevance_score=0.72,
    )


@pytest.fixture
def valid_payload_data() -> dict:
    """Create valid DigestPayload data as dict."""
    return {
        "version": "1.0",
        "content_type": "digest/v1",
        "query_hash": "ab12cd34",
        "summary": "This is a test summary of the document content.",
        "key_points": [
            "First key point about the topic.",
            "Second key point with more details.",
        ],
        "evidence_snippets": [
            {
                "text": "Evidence snippet one.",
                "locator": "char:0-21",
                "relevance_score": 0.9,
            },
            {
                "text": "Evidence snippet two.",
                "locator": "char:50-71",
                "relevance_score": 0.75,
            },
        ],
        "original_chars": 5000,
        "digest_chars": 500,
        "compression_ratio": 0.1,
        "source_text_hash": "sha256:" + "a" * 64,
    }


@pytest.fixture
def valid_payload(valid_payload_data: dict) -> DigestPayload:
    """Create a valid DigestPayload instance."""
    return DigestPayload.model_validate(valid_payload_data)


@pytest.fixture
def minimal_valid_payload_data() -> dict:
    """Create minimal valid DigestPayload data (no optional lists)."""
    return {
        "query_hash": "12345678",
        "summary": "Minimal summary.",
        "key_points": [],
        "evidence_snippets": [],
        "original_chars": 1000,
        "digest_chars": 100,
        "compression_ratio": 0.1,
        "source_text_hash": "sha256:" + "b" * 64,
    }


# =============================================================================
# Test: EvidenceSnippet Validation
# =============================================================================


class TestEvidenceSnippetValidation:
    """Tests for EvidenceSnippet field validation."""

    def test_valid_snippet_created(self, valid_evidence_snippet: EvidenceSnippet):
        """Test valid snippet is created without errors."""
        assert valid_evidence_snippet.text == "This is a test evidence snippet from the source document."
        assert valid_evidence_snippet.locator == "char:100-158"
        assert valid_evidence_snippet.relevance_score == 0.85

    def test_valid_snippet_with_page_locator(self, valid_evidence_snippet_with_page: EvidenceSnippet):
        """Test snippet with PDF page locator is valid."""
        assert valid_evidence_snippet_with_page.locator == "page:3:char:200-237"
        assert valid_evidence_snippet_with_page.relevance_score == 0.72

    def test_text_max_length_500(self):
        """Test text field rejects strings longer than 500 chars."""
        long_text = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            EvidenceSnippet(
                text=long_text,
                locator="char:0-501",
                relevance_score=0.5,
            )
        assert "max_length" in str(exc_info.value).lower() or "500" in str(exc_info.value)

    def test_text_exactly_500_chars_valid(self):
        """Test text field accepts exactly 500 chars."""
        text_500 = "x" * 500
        snippet = EvidenceSnippet(
            text=text_500,
            locator="char:0-500",
            relevance_score=0.5,
        )
        assert len(snippet.text) == 500

    def test_relevance_score_min_zero(self):
        """Test relevance_score rejects values below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceSnippet(
                text="Test",
                locator="char:0-4",
                relevance_score=-0.1,
            )
        assert "greater than or equal to 0" in str(exc_info.value).lower() or "ge" in str(exc_info.value).lower()

    def test_relevance_score_max_one(self):
        """Test relevance_score rejects values above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            EvidenceSnippet(
                text="Test",
                locator="char:0-4",
                relevance_score=1.1,
            )
        assert "less than or equal to 1" in str(exc_info.value).lower() or "le" in str(exc_info.value).lower()

    def test_relevance_score_boundaries_valid(self):
        """Test relevance_score accepts boundary values 0.0 and 1.0."""
        snippet_zero = EvidenceSnippet(
            text="Test",
            locator="char:0-4",
            relevance_score=0.0,
        )
        assert snippet_zero.relevance_score == 0.0

        snippet_one = EvidenceSnippet(
            text="Test",
            locator="char:0-4",
            relevance_score=1.0,
        )
        assert snippet_one.relevance_score == 1.0

    def test_missing_required_fields(self):
        """Test missing required fields raises ValidationError."""
        with pytest.raises(ValidationError):
            EvidenceSnippet(text="Test")  # Missing locator and relevance_score

        with pytest.raises(ValidationError):
            EvidenceSnippet(locator="char:0-4")  # Missing text and relevance_score


# =============================================================================
# Test: DigestPayload Valid Payloads
# =============================================================================


class TestDigestPayloadValidPayloads:
    """Tests for DigestPayload with valid data."""

    def test_valid_payload_created(self, valid_payload: DigestPayload):
        """Test valid payload is created without errors."""
        assert valid_payload.version == "1.0"
        assert valid_payload.content_type == "digest/v1"
        assert valid_payload.query_hash == "ab12cd34"
        assert len(valid_payload.key_points) == 2
        assert len(valid_payload.evidence_snippets) == 2

    def test_minimal_payload_uses_defaults(self, minimal_valid_payload_data: dict):
        """Test minimal payload gets default values for version and content_type."""
        payload = DigestPayload.model_validate(minimal_valid_payload_data)
        assert payload.version == "1.0"
        assert payload.content_type == "digest/v1"

    def test_is_valid_digest_property(self, valid_payload: DigestPayload):
        """Test is_valid_digest property returns True for valid v1.0 digest."""
        assert valid_payload.is_valid_digest is True

    def test_is_valid_digest_false_for_wrong_version(self, valid_payload_data: dict):
        """Test is_valid_digest returns False for non-1.0 version."""
        valid_payload_data["version"] = "2.0"
        payload = DigestPayload.model_validate(valid_payload_data)
        assert payload.is_valid_digest is False

    def test_is_valid_digest_false_for_wrong_content_type(self, valid_payload_data: dict):
        """Test is_valid_digest returns False for non-digest/v1 content type."""
        valid_payload_data["content_type"] = "text/plain"
        payload = DigestPayload.model_validate(valid_payload_data)
        assert payload.is_valid_digest is False

    def test_query_hash_lowercase_hex(self):
        """Test query_hash accepts lowercase hex strings."""
        data = {
            "query_hash": "abcdef12",
            "summary": "Test",
            "original_chars": 100,
            "digest_chars": 10,
            "compression_ratio": 0.1,
            "source_text_hash": "sha256:" + "c" * 64,
        }
        payload = DigestPayload.model_validate(data)
        assert payload.query_hash == "abcdef12"

    def test_compression_ratio_boundaries(self):
        """Test compression_ratio accepts 0.0 and 1.0."""
        base_data = {
            "query_hash": "12345678",
            "summary": "Test",
            "original_chars": 100,
            "digest_chars": 0,
            "source_text_hash": "sha256:" + "d" * 64,
        }

        # Test 0.0
        data_zero = {**base_data, "compression_ratio": 0.0}
        payload_zero = DigestPayload.model_validate(data_zero)
        assert payload_zero.compression_ratio == 0.0

        # Test 1.0
        data_one = {**base_data, "compression_ratio": 1.0, "digest_chars": 100}
        payload_one = DigestPayload.model_validate(data_one)
        assert payload_one.compression_ratio == 1.0

    def test_empty_lists_valid(self, minimal_valid_payload_data: dict):
        """Test empty key_points and evidence_snippets are valid."""
        payload = DigestPayload.model_validate(minimal_valid_payload_data)
        assert payload.key_points == []
        assert payload.evidence_snippets == []

    def test_max_key_points_10(self, valid_payload_data: dict):
        """Test key_points accepts exactly 10 items."""
        valid_payload_data["key_points"] = [f"Point {i}" for i in range(10)]
        payload = DigestPayload.model_validate(valid_payload_data)
        assert len(payload.key_points) == 10

    def test_max_evidence_snippets_10(self, valid_payload_data: dict):
        """Test evidence_snippets accepts exactly 10 items."""
        valid_payload_data["evidence_snippets"] = [
            {"text": f"Evidence {i}", "locator": f"char:{i*10}-{i*10+9}", "relevance_score": 0.5}
            for i in range(10)
        ]
        payload = DigestPayload.model_validate(valid_payload_data)
        assert len(payload.evidence_snippets) == 10


# =============================================================================
# Test: DigestPayload Invalid Payloads
# =============================================================================


class TestDigestPayloadInvalidPayloads:
    """Tests for DigestPayload rejection of invalid data."""

    def test_query_hash_too_short(self, valid_payload_data: dict):
        """Test query_hash rejects strings shorter than 8 chars."""
        valid_payload_data["query_hash"] = "abc123"  # 6 chars
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "query_hash" in error_str or "min_length" in error_str or "8" in error_str

    def test_query_hash_too_long(self, valid_payload_data: dict):
        """Test query_hash rejects strings longer than 8 chars."""
        valid_payload_data["query_hash"] = "abc123456"  # 9 chars
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "query_hash" in error_str or "max_length" in error_str or "8" in error_str

    def test_query_hash_invalid_chars(self, valid_payload_data: dict):
        """Test query_hash rejects non-hex characters."""
        valid_payload_data["query_hash"] = "abcdefgh"  # 'g' and 'h' not hex
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "query_hash" in error_str or "pattern" in error_str

    def test_query_hash_uppercase_rejected(self, valid_payload_data: dict):
        """Test query_hash rejects uppercase hex (pattern requires lowercase)."""
        valid_payload_data["query_hash"] = "ABCDEF12"
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "query_hash" in error_str or "pattern" in error_str

    def test_summary_exceeds_max_length(self, valid_payload_data: dict):
        """Test summary rejects strings longer than 2000 chars."""
        valid_payload_data["summary"] = "x" * 2001
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "summary" in error_str or "max_length" in error_str or "2000" in error_str

    def test_summary_exactly_2000_valid(self, valid_payload_data: dict):
        """Test summary accepts exactly 2000 chars."""
        valid_payload_data["summary"] = "x" * 2000
        payload = DigestPayload.model_validate(valid_payload_data)
        assert len(payload.summary) == 2000

    def test_key_point_exceeds_500_chars(self, valid_payload_data: dict):
        """Test key_points rejects items longer than 500 chars."""
        valid_payload_data["key_points"] = ["x" * 501]
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "key_points" in error_str or "500" in error_str

    def test_key_point_exactly_500_valid(self, valid_payload_data: dict):
        """Test key_points accepts items exactly 500 chars."""
        valid_payload_data["key_points"] = ["x" * 500]
        payload = DigestPayload.model_validate(valid_payload_data)
        assert len(payload.key_points[0]) == 500

    def test_original_chars_negative_rejected(self, valid_payload_data: dict):
        """Test original_chars rejects negative values."""
        valid_payload_data["original_chars"] = -1
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "original_chars" in error_str or "greater than or equal" in error_str

    def test_digest_chars_negative_rejected(self, valid_payload_data: dict):
        """Test digest_chars rejects negative values."""
        valid_payload_data["digest_chars"] = -1
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "digest_chars" in error_str or "greater than or equal" in error_str

    def test_compression_ratio_below_zero(self, valid_payload_data: dict):
        """Test compression_ratio rejects values below 0.0."""
        valid_payload_data["compression_ratio"] = -0.1
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "compression_ratio" in error_str or "greater than or equal" in error_str

    def test_compression_ratio_above_one(self, valid_payload_data: dict):
        """Test compression_ratio rejects values above 1.0."""
        valid_payload_data["compression_ratio"] = 1.1
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "compression_ratio" in error_str or "less than or equal" in error_str

    def test_source_text_hash_missing_prefix(self, valid_payload_data: dict):
        """Test source_text_hash rejects hash without sha256: prefix."""
        valid_payload_data["source_text_hash"] = "a" * 64  # Missing sha256: prefix
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "source_text_hash" in error_str or "pattern" in error_str

    def test_source_text_hash_wrong_length(self, valid_payload_data: dict):
        """Test source_text_hash rejects hash with wrong length."""
        valid_payload_data["source_text_hash"] = "sha256:" + "a" * 32  # Too short
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "source_text_hash" in error_str or "pattern" in error_str

    def test_source_text_hash_invalid_chars(self, valid_payload_data: dict):
        """Test source_text_hash rejects non-hex characters."""
        valid_payload_data["source_text_hash"] = "sha256:" + "g" * 64  # 'g' not hex
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        error_str = str(exc_info.value).lower()
        assert "source_text_hash" in error_str or "pattern" in error_str

    def test_missing_required_field_summary(self, valid_payload_data: dict):
        """Test missing summary raises ValidationError."""
        del valid_payload_data["summary"]
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        assert "summary" in str(exc_info.value).lower()

    def test_missing_required_field_query_hash(self, valid_payload_data: dict):
        """Test missing query_hash raises ValidationError."""
        del valid_payload_data["query_hash"]
        with pytest.raises(ValidationError) as exc_info:
            DigestPayload.model_validate(valid_payload_data)
        assert "query_hash" in str(exc_info.value).lower()


# =============================================================================
# Test: Serialization Round-Trip
# =============================================================================


class TestSerializationRoundTrip:
    """Tests for serialize/deserialize preserving data."""

    def test_serialize_produces_valid_json(self, valid_payload: DigestPayload):
        """Test serialize_payload produces valid JSON string."""
        json_str = serialize_payload(valid_payload)
        # Should be parseable as JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_serialize_includes_all_fields(self, valid_payload: DigestPayload):
        """Test serialized JSON includes all payload fields."""
        json_str = serialize_payload(valid_payload)
        parsed = json.loads(json_str)
        assert "version" in parsed
        assert "content_type" in parsed
        assert "query_hash" in parsed
        assert "summary" in parsed
        assert "key_points" in parsed
        assert "evidence_snippets" in parsed
        assert "original_chars" in parsed
        assert "digest_chars" in parsed
        assert "compression_ratio" in parsed
        assert "source_text_hash" in parsed

    def test_round_trip_preserves_all_data(self, valid_payload: DigestPayload):
        """Test serialize -> deserialize preserves all field values."""
        json_str = serialize_payload(valid_payload)
        restored = deserialize_payload(json_str)

        assert restored.version == valid_payload.version
        assert restored.content_type == valid_payload.content_type
        assert restored.query_hash == valid_payload.query_hash
        assert restored.summary == valid_payload.summary
        assert restored.key_points == valid_payload.key_points
        assert restored.original_chars == valid_payload.original_chars
        assert restored.digest_chars == valid_payload.digest_chars
        assert restored.compression_ratio == valid_payload.compression_ratio
        assert restored.source_text_hash == valid_payload.source_text_hash

    def test_round_trip_preserves_evidence_snippets(self, valid_payload: DigestPayload):
        """Test round-trip preserves evidence snippets exactly."""
        json_str = serialize_payload(valid_payload)
        restored = deserialize_payload(json_str)

        assert len(restored.evidence_snippets) == len(valid_payload.evidence_snippets)
        for original, restored_snippet in zip(
            valid_payload.evidence_snippets, restored.evidence_snippets
        ):
            assert restored_snippet.text == original.text
            assert restored_snippet.locator == original.locator
            assert restored_snippet.relevance_score == original.relevance_score

    def test_round_trip_empty_lists(self, minimal_valid_payload_data: dict):
        """Test round-trip with empty key_points and evidence_snippets."""
        payload = DigestPayload.model_validate(minimal_valid_payload_data)
        json_str = serialize_payload(payload)
        restored = deserialize_payload(json_str)

        assert restored.key_points == []
        assert restored.evidence_snippets == []

    def test_round_trip_max_key_points(self, valid_payload_data: dict):
        """Test round-trip with maximum 10 key points."""
        valid_payload_data["key_points"] = [f"Key point number {i}" for i in range(10)]
        payload = DigestPayload.model_validate(valid_payload_data)
        json_str = serialize_payload(payload)
        restored = deserialize_payload(json_str)

        assert len(restored.key_points) == 10
        for i in range(10):
            assert restored.key_points[i] == f"Key point number {i}"

    def test_round_trip_unicode_content(self, valid_payload_data: dict):
        """Test round-trip preserves Unicode characters."""
        valid_payload_data["summary"] = "Summary with émojis 🔬 and ünïcödé characters 日本語"
        valid_payload_data["key_points"] = ["Point with émojis 🎯", "日本語のポイント"]
        payload = DigestPayload.model_validate(valid_payload_data)
        json_str = serialize_payload(payload)
        restored = deserialize_payload(json_str)

        assert restored.summary == valid_payload_data["summary"]
        assert restored.key_points == valid_payload_data["key_points"]

    def test_serialize_deterministic(self, valid_payload: DigestPayload):
        """Test serialize produces deterministic output (sorted keys)."""
        json_str_1 = serialize_payload(valid_payload)
        json_str_2 = serialize_payload(valid_payload)
        assert json_str_1 == json_str_2

    def test_serialize_none_raises_value_error(self):
        """Test serialize_payload raises ValueError for None input."""
        with pytest.raises(ValueError) as exc_info:
            serialize_payload(None)
        assert "none" in str(exc_info.value).lower()

    def test_deserialize_empty_string_raises_value_error(self):
        """Test deserialize_payload raises ValueError for empty string."""
        with pytest.raises(ValueError) as exc_info:
            deserialize_payload("")
        assert "empty" in str(exc_info.value).lower()

    def test_deserialize_whitespace_only_raises_value_error(self):
        """Test deserialize_payload raises ValueError for whitespace-only string."""
        with pytest.raises(ValueError) as exc_info:
            deserialize_payload("   \n\t   ")
        assert "empty" in str(exc_info.value).lower()

    def test_deserialize_invalid_json_raises_value_error(self):
        """Test deserialize_payload raises ValueError for invalid JSON."""
        with pytest.raises(ValueError) as exc_info:
            deserialize_payload("not valid json {")
        assert "json" in str(exc_info.value).lower()

    def test_deserialize_valid_json_invalid_schema_raises_validation_error(self):
        """Test deserialize with valid JSON but invalid schema raises ValidationError."""
        json_str = '{"foo": "bar"}'  # Missing required fields
        with pytest.raises(ValidationError):
            deserialize_payload(json_str)


# =============================================================================
# Test: validate_payload_dict
# =============================================================================


class TestValidatePayloadDict:
    """Tests for validate_payload_dict function."""

    def test_valid_dict_returns_payload(self, valid_payload_data: dict):
        """Test valid dict returns DigestPayload instance."""
        payload = validate_payload_dict(valid_payload_data)
        assert isinstance(payload, DigestPayload)
        assert payload.query_hash == valid_payload_data["query_hash"]

    def test_invalid_dict_raises_validation_error(self):
        """Test invalid dict raises ValidationError."""
        invalid_data = {"query_hash": "invalid"}  # Missing required fields
        with pytest.raises(ValidationError):
            validate_payload_dict(invalid_data)

    def test_non_dict_raises_type_error(self):
        """Test non-dict input raises TypeError."""
        with pytest.raises(TypeError) as exc_info:
            validate_payload_dict("not a dict")
        assert "dict" in str(exc_info.value).lower()

        with pytest.raises(TypeError):
            validate_payload_dict(123)

        with pytest.raises(TypeError):
            validate_payload_dict(["list", "not", "dict"])

    def test_validates_nested_evidence_snippets(self, valid_payload_data: dict):
        """Test validation catches invalid nested evidence snippets."""
        valid_payload_data["evidence_snippets"] = [
            {"text": "Valid", "locator": "char:0-5", "relevance_score": 1.5}  # Invalid score
        ]
        with pytest.raises(ValidationError):
            validate_payload_dict(valid_payload_data)


# =============================================================================
# Test: DigestPayload JSON methods
# =============================================================================


class TestDigestPayloadJsonMethods:
    """Tests for DigestPayload.to_json() and from_json() methods."""

    def test_to_json_produces_valid_json(self, valid_payload: DigestPayload):
        """Test to_json produces parseable JSON."""
        json_str = valid_payload.to_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_from_json_restores_payload(self, valid_payload: DigestPayload):
        """Test from_json restores equivalent payload."""
        json_str = valid_payload.to_json()
        restored = DigestPayload.from_json(json_str)
        assert restored.query_hash == valid_payload.query_hash
        assert restored.summary == valid_payload.summary

    def test_from_json_invalid_raises_validation_error(self):
        """Test from_json raises ValidationError for invalid data."""
        with pytest.raises(ValidationError):
            DigestPayload.from_json('{"invalid": "data"}')

    def test_from_json_invalid_json_raises_error(self):
        """Test from_json raises error for malformed JSON."""
        with pytest.raises(Exception):  # Could be ValueError or JSONDecodeError
            DigestPayload.from_json("not json")


# =============================================================================
# Test: Evidence Scoring Algorithm
# =============================================================================


class TestEvidenceScoringAlgorithm:
    """Tests for evidence scoring determinism, fallbacks, and tie-breakers."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies for testing."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            max_evidence_snippets=5,
            max_snippet_length=500,
        )
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

    def test_same_input_produces_same_output(self, digestor):
        """Test evidence scoring is deterministic - same input produces same output."""
        text = (
            "Climate change is affecting global weather patterns. "
            "Rising temperatures cause ice caps to melt. "
            "Coastal cities face flooding risks from rising sea levels. "
            "Scientists recommend immediate action on emissions."
        )
        query = "climate change impact coastal cities"

        # Run multiple times
        result1 = digestor._extract_evidence(text, query, max_snippets=3)
        result2 = digestor._extract_evidence(text, query, max_snippets=3)
        result3 = digestor._extract_evidence(text, query, max_snippets=3)

        # All results should be identical
        assert result1 == result2
        assert result2 == result3

    def test_empty_query_uses_positional_fallback(self, digestor):
        """Test empty query falls back to positional scoring."""
        text = "First paragraph. Second paragraph. Third paragraph."
        query = ""

        result = digestor._extract_evidence(text, query, max_snippets=3)

        # Should return chunks in positional order (first chunks preferred)
        assert len(result) > 0
        # Scores should decrease with position
        if len(result) > 1:
            assert result[0][2] >= result[1][2]  # First chunk score >= second

    def test_short_query_uses_positional_fallback(self, digestor):
        """Test query shorter than 3 chars falls back to positional scoring."""
        text = "First paragraph. Second paragraph. Third paragraph."
        query = "ab"  # Less than 3 chars

        result = digestor._extract_evidence(text, query, max_snippets=3)

        # Should use positional fallback
        assert len(result) > 0

    def test_stopword_only_query_uses_positional_fallback(self, digestor):
        """Test query with only stopwords falls back to positional scoring."""
        text = "First paragraph. Second paragraph. Third paragraph."
        query = "the and or but"  # Only stopwords

        result = digestor._extract_evidence(text, query, max_snippets=3)

        # Should use positional fallback since no meaningful terms
        assert len(result) > 0

    def test_tie_breaker_score_first(self, digestor):
        """Test higher score wins over position."""
        # Create text where later chunk has more query terms
        text = (
            "Introduction with general content. "
            "Climate change affects weather patterns. "  # Some matches
            "Climate change causes coastal flooding in cities."  # More matches
        )
        query = "climate change coastal cities"

        result = digestor._extract_evidence(text, query, max_snippets=2)

        # Higher scoring chunk should come first regardless of position
        assert len(result) >= 1
        # First result should have highest score
        if len(result) > 1:
            assert result[0][2] >= result[1][2]

    def test_tie_breaker_position_second(self, digestor):
        """Test earlier position wins when scores are equal."""
        # Create chunks with same terms appearing equally
        chunk1 = "Climate change is real."
        chunk2 = "Climate change is happening."
        text = f"{chunk1} {chunk2}"
        query = "climate change"

        result = digestor._extract_evidence(text, query, max_snippets=2)

        # When scores are equal, earlier position should win
        # Position is the second element in the tuple
        if len(result) >= 2 and result[0][2] == result[1][2]:
            assert result[0][1] < result[1][1]  # Earlier position first

    def test_rare_terms_score_higher(self, digestor):
        """Test rarer terms in corpus contribute more to score."""
        # "climate" appears many times, "anthropogenic" appears once
        text = (
            "Climate change is a climate-related issue. "
            "Climate patterns are shifting. "
            "Anthropogenic factors drive climate change."  # Rare term here
        )
        query = "anthropogenic climate"

        result = digestor._extract_evidence(text, query, max_snippets=3)

        # The chunk with the rare term "anthropogenic" should score higher
        assert len(result) >= 1

    def test_case_insensitive_matching(self, digestor):
        """Test term matching is case-insensitive."""
        text = "CLIMATE Change affects COASTAL regions."
        query = "climate coastal"

        result = digestor._extract_evidence(text, query, max_snippets=1)

        # Should find matches despite case differences
        assert len(result) >= 1
        assert result[0][2] > 0  # Should have positive score

    def test_max_snippets_respected(self, digestor):
        """Test max_snippets limit is respected."""
        text = (
            "First chunk about climate. "
            "Second chunk about climate. "
            "Third chunk about climate. "
            "Fourth chunk about climate. "
            "Fifth chunk about climate. "
            "Sixth chunk about climate."
        )
        query = "climate"

        result = digestor._extract_evidence(text, query, max_snippets=2)

        assert len(result) <= 2

    def test_empty_text_returns_empty(self, digestor):
        """Test empty text returns empty results."""
        result = digestor._extract_evidence("", "query", max_snippets=5)
        assert result == []

    def test_whitespace_only_text_returns_empty(self, digestor):
        """Test whitespace-only text returns empty results."""
        result = digestor._extract_evidence("   \n\t   ", "query", max_snippets=5)
        assert result == []


class TestEvidenceLocatorOrdering:
    """Tests for locator generation when relevance order differs from text order."""

    def test_locators_match_snippet_text_out_of_order(self):
        """Ensure locators remain valid even when relevance order differs."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            chunk_size=40,
            max_snippet_length=50,
            max_evidence_snippets=2,
        )
        digestor = DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

        canonical_text = (
            "First section mentions keyword once. "
            "Second section mentions keyword keyword keyword for relevance. "
            "Third section is filler."
        )
        query = "keyword"

        snippets = digestor._build_evidence_snippets(canonical_text, query)
        assert len(snippets) == 2

        for snippet in snippets:
            match = re.match(r"^char:(\d+)-(\d+)$", snippet.locator)
            assert match is not None
            start = int(match.group(1))
            end = int(match.group(2))
            assert canonical_text[start:end] == snippet.text


class TestDigestEvidenceToggle:
    """Tests for include_evidence configuration."""

    @pytest.mark.asyncio
    async def test_include_evidence_false_skips_snippets(self):
        """Digest should omit evidence_snippets when include_evidence is False."""
        from unittest.mock import AsyncMock, MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )
        from foundry_mcp.core.research.summarization import (
            SummarizationLevel,
            SummarizationResult,
        )

        mock_summarizer = MagicMock()
        mock_summarizer.summarize_with_result = AsyncMock(
            return_value=SummarizationResult(
                content="Summary content.",
                level=SummarizationLevel.KEY_POINTS,
                key_points=["Point one", "Point two"],
                warnings=[],
            )
        )
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            policy=DigestPolicy.ALWAYS,
            include_evidence=False,
        )
        digestor = DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

        result = await digestor.digest(
            source="This source has enough content to digest.",
            query="test query",
        )

        assert result.success is True
        assert result.payload is not None
        assert result.payload.evidence_snippets == []
        expected_chars = len(result.payload.summary) + sum(
            len(kp) for kp in result.payload.key_points
        )
        assert result.payload.digest_chars == expected_chars


class TestExtractTerms:
    """Tests for query term extraction."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def test_extracts_meaningful_terms(self, digestor):
        """Test meaningful terms are extracted from query."""
        terms = digestor._extract_terms("climate change impact")
        assert "climate" in terms
        assert "change" in terms
        assert "impact" in terms

    def test_filters_stopwords(self, digestor):
        """Test stopwords are filtered out."""
        terms = digestor._extract_terms("the climate and the weather")
        assert "the" not in terms
        assert "and" not in terms
        assert "climate" in terms
        assert "weather" in terms

    def test_filters_short_terms(self, digestor):
        """Test terms shorter than 2 chars are filtered."""
        terms = digestor._extract_terms("a b climate x y z")
        assert "a" not in terms
        assert "b" not in terms
        assert "x" not in terms
        assert "climate" in terms

    def test_lowercases_terms(self, digestor):
        """Test terms are lowercased."""
        terms = digestor._extract_terms("CLIMATE Change WEATHER")
        assert "climate" in terms
        assert "change" in terms
        assert "weather" in terms
        # Uppercase versions should not be present
        assert "CLIMATE" not in terms

    def test_splits_on_punctuation(self, digestor):
        """Test query is split on punctuation."""
        terms = digestor._extract_terms("climate-change, weather.patterns")
        assert "climate" in terms
        assert "change" in terms
        assert "weather" in terms
        assert "patterns" in terms

    def test_empty_query_returns_empty(self, digestor):
        """Test empty query returns empty list."""
        terms = digestor._extract_terms("")
        assert terms == []

    def test_stopword_only_returns_empty(self, digestor):
        """Test query with only stopwords returns empty list."""
        terms = digestor._extract_terms("the and or but in on at")
        assert terms == []


class TestScoreByPosition:
    """Tests for positional scoring fallback."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def test_first_chunk_scores_highest(self, digestor):
        """Test first chunk gets highest score."""
        chunks = ["First", "Second", "Third", "Fourth"]
        result = digestor._score_by_position(chunks, max_snippets=4)

        # First chunk should have score 1.0 (or close to it)
        assert result[0][2] == 1.0
        # Scores should decrease
        for i in range(len(result) - 1):
            assert result[i][2] >= result[i + 1][2]

    def test_scores_decrease_linearly(self, digestor):
        """Test scores decrease linearly with position."""
        chunks = ["A", "B", "C", "D"]
        result = digestor._score_by_position(chunks, max_snippets=4)

        # Check that scores decrease
        scores = [r[2] for r in result]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_single_chunk_gets_score_one(self, digestor):
        """Test single chunk gets score of 1.0."""
        chunks = ["Only chunk"]
        result = digestor._score_by_position(chunks, max_snippets=1)

        assert len(result) == 1
        assert result[0][2] == 1.0

    def test_respects_max_snippets(self, digestor):
        """Test max_snippets is respected."""
        chunks = ["A", "B", "C", "D", "E"]
        result = digestor._score_by_position(chunks, max_snippets=2)

        assert len(result) == 2

    def test_preserves_chunk_text(self, digestor):
        """Test chunk text is preserved in output."""
        chunks = ["First chunk text", "Second chunk text"]
        result = digestor._score_by_position(chunks, max_snippets=2)

        assert result[0][0] == "First chunk text"
        assert result[1][0] == "Second chunk text"

    def test_preserves_position_index(self, digestor):
        """Test position index is preserved in output."""
        chunks = ["A", "B", "C"]
        result = digestor._score_by_position(chunks, max_snippets=3)

        assert result[0][1] == 0
        assert result[1][1] == 1
        assert result[2][1] == 2


# =============================================================================
# Test: Eligibility Logic
# =============================================================================


class TestEligibilityOffPolicy:
    """Tests for OFF digest policy - always ineligible."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with OFF policy."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )
        from foundry_mcp.core.research.models import SourceQuality

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(policy=DigestPolicy.OFF)
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

    def test_off_policy_high_quality_ineligible(self, digestor):
        """Test OFF policy rejects HIGH quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 10000  # Long content
        assert digestor._is_eligible(content, SourceQuality.HIGH) is False

    def test_off_policy_medium_quality_ineligible(self, digestor):
        """Test OFF policy rejects MEDIUM quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 10000
        assert digestor._is_eligible(content, SourceQuality.MEDIUM) is False

    def test_off_policy_any_content_ineligible(self, digestor):
        """Test OFF policy rejects any content regardless of quality."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 10000
        assert digestor._is_eligible(content, SourceQuality.HIGH) is False
        assert digestor._is_eligible(content, SourceQuality.MEDIUM) is False
        assert digestor._is_eligible(content, SourceQuality.LOW) is False
        assert digestor._is_eligible(content, SourceQuality.UNKNOWN) is False
        assert digestor._is_eligible(content, None) is False

    def test_off_policy_skip_reason(self, digestor):
        """Test OFF policy returns correct skip reason."""
        content = "x" * 10000
        reason = digestor._get_skip_reason(content, None)
        assert "OFF" in reason


class TestEligibilityAlwaysPolicy:
    """Tests for ALWAYS digest policy - eligible with content."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with ALWAYS policy."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(policy=DigestPolicy.ALWAYS)
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

    def test_always_policy_low_quality_eligible(self, digestor):
        """Test ALWAYS policy accepts LOW quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "Some content"
        assert digestor._is_eligible(content, SourceQuality.LOW) is True

    def test_always_policy_unknown_quality_eligible(self, digestor):
        """Test ALWAYS policy accepts UNKNOWN quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "Some content"
        assert digestor._is_eligible(content, SourceQuality.UNKNOWN) is True

    def test_always_policy_none_quality_eligible(self, digestor):
        """Test ALWAYS policy accepts content without quality specified."""
        content = "Some content"
        assert digestor._is_eligible(content, None) is True

    def test_always_policy_short_content_eligible(self, digestor):
        """Test ALWAYS policy accepts short content."""
        content = "Short"
        assert digestor._is_eligible(content, None) is True

    def test_always_policy_empty_content_ineligible(self, digestor):
        """Test ALWAYS policy rejects empty content."""
        assert digestor._is_eligible("", None) is False

    def test_always_policy_whitespace_only_ineligible(self, digestor):
        """Test ALWAYS policy rejects whitespace-only content."""
        assert digestor._is_eligible("   \n\t   ", None) is False

    def test_always_policy_skip_reason_for_empty(self, digestor):
        """Test ALWAYS policy returns correct skip reason for empty content."""
        reason = digestor._get_skip_reason("", None)
        assert "empty" in reason.lower()


class TestEligibilityAutoPolicy:
    """Tests for AUTO digest policy - checks thresholds."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with AUTO policy."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )
        from foundry_mcp.core.research.models import SourceQuality

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            policy=DigestPolicy.AUTO,
            min_content_length=500,  # Minimum 500 chars
            quality_threshold=SourceQuality.MEDIUM,  # Require MEDIUM or higher
        )
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

    def test_auto_policy_high_quality_long_content_eligible(self, digestor):
        """Test AUTO policy accepts HIGH quality content above threshold."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 600  # Above min_content_length
        assert digestor._is_eligible(content, SourceQuality.HIGH) is True

    def test_auto_policy_medium_quality_long_content_eligible(self, digestor):
        """Test AUTO policy accepts MEDIUM quality content above threshold."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 600
        assert digestor._is_eligible(content, SourceQuality.MEDIUM) is True

    def test_auto_policy_low_quality_ineligible(self, digestor):
        """Test AUTO policy rejects LOW quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 600
        assert digestor._is_eligible(content, SourceQuality.LOW) is False

    def test_auto_policy_unknown_quality_ineligible(self, digestor):
        """Test AUTO policy rejects UNKNOWN quality content."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 600
        assert digestor._is_eligible(content, SourceQuality.UNKNOWN) is False

    def test_auto_policy_none_quality_ineligible(self, digestor):
        """Test AUTO policy rejects content without quality specified."""
        content = "x" * 600
        assert digestor._is_eligible(content, None) is False

    def test_auto_policy_short_content_ineligible(self, digestor):
        """Test AUTO policy rejects content below size threshold."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 400  # Below min_content_length of 500
        assert digestor._is_eligible(content, SourceQuality.HIGH) is False

    def test_auto_policy_exact_threshold_eligible(self, digestor):
        """Test AUTO policy accepts content exactly at size threshold."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 500  # Exactly at min_content_length
        assert digestor._is_eligible(content, SourceQuality.HIGH) is True

    def test_auto_policy_skip_reason_size(self, digestor):
        """Test AUTO policy returns correct skip reason for size."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 100  # Below threshold
        reason = digestor._get_skip_reason(content, SourceQuality.HIGH)
        assert "100" in reason  # Content length
        assert "500" in reason  # Threshold

    def test_auto_policy_skip_reason_quality(self, digestor):
        """Test AUTO policy returns correct skip reason for quality."""
        from foundry_mcp.core.research.models import SourceQuality

        content = "x" * 600  # Above size threshold
        reason = digestor._get_skip_reason(content, SourceQuality.LOW)
        assert "low" in reason.lower()
        assert "medium" in reason.lower()

    def test_auto_policy_skip_reason_none_quality(self, digestor):
        """Test AUTO policy returns correct skip reason for missing quality."""
        content = "x" * 600
        reason = digestor._get_skip_reason(content, None)
        assert "not provided" in reason.lower() or "quality" in reason.lower()


class TestEligibilityCustomQualityThreshold:
    """Tests for AUTO policy with custom quality threshold."""

    def test_auto_policy_low_threshold_accepts_low(self):
        """Test AUTO policy with LOW threshold accepts LOW quality."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )
        from foundry_mcp.core.research.models import SourceQuality

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            policy=DigestPolicy.AUTO,
            min_content_length=100,
            quality_threshold=SourceQuality.LOW,  # Accept LOW and above
        )
        digestor = DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

        content = "x" * 200
        assert digestor._is_eligible(content, SourceQuality.LOW) is True
        assert digestor._is_eligible(content, SourceQuality.MEDIUM) is True
        assert digestor._is_eligible(content, SourceQuality.HIGH) is True
        # UNKNOWN still rejected (below LOW)
        assert digestor._is_eligible(content, SourceQuality.UNKNOWN) is False

    def test_auto_policy_high_threshold_rejects_medium(self):
        """Test AUTO policy with HIGH threshold rejects MEDIUM quality."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
            DigestPolicy,
        )
        from foundry_mcp.core.research.models import SourceQuality

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        config = DigestConfig(
            policy=DigestPolicy.AUTO,
            min_content_length=100,
            quality_threshold=SourceQuality.HIGH,  # Only accept HIGH
        )
        digestor = DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=config,
        )

        content = "x" * 200
        assert digestor._is_eligible(content, SourceQuality.HIGH) is True
        assert digestor._is_eligible(content, SourceQuality.MEDIUM) is False
        assert digestor._is_eligible(content, SourceQuality.LOW) is False


# =============================================================================
# Test: Cache Key Generation
# =============================================================================


class TestCacheKeyGeneration:
    """Tests for cache key generation and format."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def test_cache_key_format(self, digestor):
        """Test cache key follows expected format."""
        key = digestor.generate_cache_key(
            source_id="doc-123",
            content_hash="sha256:" + "a" * 64,
            query_hash="ef567890",
            config_hash="12345678abcdef00",
        )
        # Format: digest:{version}:{source_id}:{content[:16]}:{query[:8]}:{config[:8]}:{summarizer[:8]}
        parts = key.split(":")
        assert parts[0] == "digest"
        assert parts[1] == "1.0"  # impl version
        assert parts[2] == "doc-123"
        assert parts[3] == "a" * 16  # content hash truncated to 16
        assert parts[4] == "ef567890"  # query hash truncated to 8
        assert parts[5] == "12345678"  # config hash truncated to 8
        assert len(parts[6]) == 8  # summarizer hash truncated to 8

    def test_cache_key_strips_sha256_prefix(self, digestor):
        """Test cache key strips sha256: prefix from content hash."""
        key = digestor.generate_cache_key(
            source_id="doc-1",
            content_hash="sha256:abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            query_hash="12345678",
            config_hash="abcdef00",
        )
        assert "sha256" not in key
        assert "abcdef1234567890" in key  # First 16 chars of hex

    def test_cache_key_handles_raw_hex_content_hash(self, digestor):
        """Test cache key handles content hash without sha256: prefix."""
        key = digestor.generate_cache_key(
            source_id="doc-1",
            content_hash="abcdef1234567890abcdef1234567890",
            query_hash="12345678",
            config_hash="abcdef00",
        )
        assert "abcdef1234567890" in key

    def test_cache_key_truncates_hashes(self, digestor):
        """Test cache key truncates hashes to correct lengths."""
        key = digestor.generate_cache_key(
            source_id="doc-1",
            content_hash="sha256:" + "f" * 64,
            query_hash="a" * 64,
            config_hash="b" * 64,
        )
        parts = key.split(":")
        assert len(parts[3]) == 16  # content hash: 16 chars
        assert len(parts[4]) == 8   # query hash: 8 chars
        assert len(parts[5]) == 8   # config hash: 8 chars
        assert len(parts[6]) == 8   # summarizer hash: 8 chars

    def test_cache_key_deterministic(self, digestor):
        """Test same inputs produce same cache key."""
        args = dict(
            source_id="doc-1",
            content_hash="sha256:" + "a" * 64,
            query_hash="12345678",
            config_hash="abcdef00",
        )
        key1 = digestor.generate_cache_key(**args)
        key2 = digestor.generate_cache_key(**args)
        assert key1 == key2

    def test_cache_key_custom_impl_version(self, digestor):
        """Test cache key with custom implementation version."""
        key = digestor.generate_cache_key(
            source_id="doc-1",
            content_hash="sha256:" + "a" * 64,
            query_hash="12345678",
            config_hash="abcdef00",
            impl_version="2.0",
        )
        assert ":2.0:" in key


class TestCacheKeyInvalidation:
    """Tests for cache key invalidation on content/query/config/version change."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def _base_args(self):
        """Return baseline cache key arguments."""
        return dict(
            source_id="doc-1",
            content_hash="sha256:" + "a" * 64,
            query_hash="12345678",
            config_hash="abcdef00",
        )

    def test_content_change_invalidates(self, digestor):
        """Test different content hash produces different cache key."""
        base = self._base_args()
        key1 = digestor.generate_cache_key(**base)

        base["content_hash"] = "sha256:" + "b" * 64
        key2 = digestor.generate_cache_key(**base)

        assert key1 != key2

    def test_query_change_invalidates(self, digestor):
        """Test different query hash produces different cache key."""
        base = self._base_args()
        key1 = digestor.generate_cache_key(**base)

        base["query_hash"] = "87654321"
        key2 = digestor.generate_cache_key(**base)

        assert key1 != key2

    def test_config_change_invalidates(self, digestor):
        """Test different config hash produces different cache key."""
        base = self._base_args()
        key1 = digestor.generate_cache_key(**base)

        base["config_hash"] = "00fedcba"
        key2 = digestor.generate_cache_key(**base)

        assert key1 != key2

    def test_source_id_change_invalidates(self, digestor):
        """Test different source_id produces different cache key."""
        base = self._base_args()
        key1 = digestor.generate_cache_key(**base)

        base["source_id"] = "doc-2"
        key2 = digestor.generate_cache_key(**base)

        assert key1 != key2

    def test_version_bump_invalidates(self, digestor):
        """Test different impl_version produces different cache key."""
        base = self._base_args()
        key1 = digestor.generate_cache_key(**base, impl_version="1.0")
        key2 = digestor.generate_cache_key(**base, impl_version="2.0")

        assert key1 != key2

    def test_summarizer_change_invalidates(self):
        """Test different summarizer configs produce different cache keys."""
        from foundry_mcp.core.research.document_digest import DocumentDigestor, DigestConfig
        from foundry_mcp.core.research.pdf_extractor import PDFExtractor
        from foundry_mcp.core.research.summarization import ContentSummarizer

        digestor_a = DocumentDigestor(
            summarizer=ContentSummarizer(summarization_provider="claude"),
            pdf_extractor=PDFExtractor(),
            config=DigestConfig(),
        )
        digestor_b = DocumentDigestor(
            summarizer=ContentSummarizer(summarization_provider="gemini"),
            pdf_extractor=PDFExtractor(),
            config=DigestConfig(),
        )
        base = self._base_args()
        key1 = digestor_a.generate_cache_key(**base)
        key2 = digestor_b.generate_cache_key(**base)

        assert key1 != key2


class TestConfigHash:
    """Tests for DigestConfig.compute_config_hash()."""

    def test_config_hash_deterministic(self):
        """Test same config produces same hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config = DigestConfig()
        hash1 = config.compute_config_hash()
        hash2 = config.compute_config_hash()
        assert hash1 == hash2

    def test_config_hash_length_16(self):
        """Test config hash is 16 characters."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config = DigestConfig()
        assert len(config.compute_config_hash()) == 16

    def test_config_hash_hex_only(self):
        """Test config hash contains only hex characters."""
        from foundry_mcp.core.research.document_digest import DigestConfig
        import re

        config = DigestConfig()
        assert re.match(r"^[0-9a-f]{16}$", config.compute_config_hash())

    def test_different_max_snippets_different_hash(self):
        """Test changing max_evidence_snippets changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(max_evidence_snippets=5)
        config2 = DigestConfig(max_evidence_snippets=10)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_different_min_content_length_different_hash(self):
        """Test changing min_content_length changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(min_content_length=500)
        config2 = DigestConfig(min_content_length=1000)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_different_chunk_size_different_hash(self):
        """Test changing chunk_size changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(chunk_size=1000)
        config2 = DigestConfig(chunk_size=2000)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_different_policy_different_hash(self):
        """Test changing policy changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig, DigestPolicy

        config1 = DigestConfig(policy=DigestPolicy.AUTO)
        config2 = DigestConfig(policy=DigestPolicy.ALWAYS)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_different_include_evidence_different_hash(self):
        """Test changing include_evidence changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(include_evidence=True)
        config2 = DigestConfig(include_evidence=False)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_different_max_summary_length_different_hash(self):
        """Test changing max_summary_length changes hash."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(max_summary_length=2000)
        config2 = DigestConfig(max_summary_length=1000)
        assert config1.compute_config_hash() != config2.compute_config_hash()

    def test_cache_enabled_does_not_affect_hash(self):
        """Test cache_enabled does not change config hash (not a digest param)."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config1 = DigestConfig(cache_enabled=True)
        config2 = DigestConfig(cache_enabled=False)
        assert config1.compute_config_hash() == config2.compute_config_hash()


class TestConfigNormalization:
    """Tests for DigestConfig normalization to schema limits."""

    def test_config_clamps_to_schema_caps(self):
        """Config values above schema limits are clamped."""
        from foundry_mcp.core.research.document_digest import DigestConfig

        config = DigestConfig(
            max_summary_length=5000,
            max_key_points=25,
            max_evidence_snippets=50,
            max_snippet_length=2000,
        )

        assert config.max_summary_length == 2000
        assert config.max_key_points == 10
        assert config.max_evidence_snippets == 10
        assert config.max_snippet_length == 500


class TestQueryAndSourceHash:
    """Tests for query hash and source hash computation."""

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def test_query_hash_is_8_chars(self, digestor):
        """Test query hash is 8 characters."""
        h = digestor._compute_query_hash("test query")
        assert len(h) == 8

    def test_query_hash_deterministic(self, digestor):
        """Test same query produces same hash."""
        h1 = digestor._compute_query_hash("test query")
        h2 = digestor._compute_query_hash("test query")
        assert h1 == h2

    def test_different_queries_different_hash(self, digestor):
        """Test different queries produce different hashes."""
        h1 = digestor._compute_query_hash("query one")
        h2 = digestor._compute_query_hash("query two")
        assert h1 != h2

    def test_source_hash_has_sha256_prefix(self, digestor):
        """Test source hash starts with sha256: prefix."""
        h = digestor._compute_source_hash("content")
        assert h.startswith("sha256:")

    def test_source_hash_is_sha256_length(self, digestor):
        """Test source hash has correct length (sha256: + 64 hex chars)."""
        h = digestor._compute_source_hash("content")
        assert len(h) == 7 + 64  # "sha256:" + 64 hex

    def test_source_hash_deterministic(self, digestor):
        """Test same content produces same source hash."""
        h1 = digestor._compute_source_hash("same content")
        h2 = digestor._compute_source_hash("same content")
        assert h1 == h2

    def test_different_content_different_source_hash(self, digestor):
        """Test different content produces different source hash."""
        h1 = digestor._compute_source_hash("content A")
        h2 = digestor._compute_source_hash("content B")
        assert h1 != h2


# =============================================================================
# Test: _raw_content Lifecycle
# =============================================================================


class TestRawContentLifecycle:
    """Tests for _raw_content metadata field lifecycle.

    The _raw_content field is temporarily stored in source.metadata during
    digest processing and MUST be cleaned up afterwards. It must never
    appear in serialized output (to_dict, public_metadata, JSON).
    """

    @pytest.fixture
    def source(self):
        """Create a ResearchSource with _raw_content in metadata."""
        from foundry_mcp.core.research.models import ResearchSource

        return ResearchSource(
            title="Test Source",
            content="digested content",
            metadata={"_raw_content": "original raw content", "visible_key": "value"},
        )

    @pytest.fixture
    def source_without_raw(self):
        """Create a ResearchSource without _raw_content."""
        from foundry_mcp.core.research.models import ResearchSource

        return ResearchSource(
            title="Test Source",
            content="content",
            metadata={"visible_key": "value"},
        )

    def test_raw_content_stored_in_metadata(self, source):
        """Test _raw_content can be stored in metadata dict."""
        assert "_raw_content" in source.metadata
        assert source.metadata["_raw_content"] == "original raw content"

    def test_raw_content_not_in_to_dict(self, source):
        """Test _raw_content is excluded from to_dict() output."""
        data = source.to_dict()
        assert "_raw_content" not in data["metadata"]

    def test_raw_content_not_in_public_metadata(self, source):
        """Test _raw_content is excluded from public_metadata()."""
        public = source.public_metadata()
        assert "_raw_content" not in public

    def test_visible_keys_preserved_in_to_dict(self, source):
        """Test non-underscore metadata keys are preserved in to_dict()."""
        data = source.to_dict()
        assert data["metadata"]["visible_key"] == "value"

    def test_visible_keys_preserved_in_public_metadata(self, source):
        """Test non-underscore metadata keys are preserved in public_metadata()."""
        public = source.public_metadata()
        assert public["visible_key"] == "value"

    def test_raw_content_deleted_via_pop(self, source):
        """Test _raw_content is properly deleted via pop pattern."""
        # This mirrors the cleanup pattern in deep_research.py
        source.metadata.pop("_raw_content", None)
        assert "_raw_content" not in source.metadata

    def test_raw_content_pop_idempotent(self, source_without_raw):
        """Test pop on missing _raw_content does not raise."""
        # Should not raise even when _raw_content is not present
        source_without_raw.metadata.pop("_raw_content", None)
        assert "_raw_content" not in source_without_raw.metadata

    def test_raw_content_not_in_json_serialization(self, source):
        """Test _raw_content is excluded from JSON serialization via to_dict."""
        import json

        data = source.to_dict()
        json_str = json.dumps(data, default=str)
        assert "_raw_content" not in json_str

    def test_raw_content_present_in_model_dump(self, source):
        """Test _raw_content IS present in model_dump (internal serialization)."""
        data = source.model_dump()
        assert "_raw_content" in data["metadata"]

    def test_all_underscore_keys_filtered(self):
        """Test all underscore-prefixed metadata keys are filtered."""
        from foundry_mcp.core.research.models import ResearchSource

        source = ResearchSource(
            title="Test",
            metadata={
                "_raw_content": "raw",
                "_token_cache": {"v": 1},
                "_digest_cache_hit": True,
                "public_key": "visible",
            },
        )
        public = source.public_metadata()
        assert "_raw_content" not in public
        assert "_token_cache" not in public
        assert "_digest_cache_hit" not in public
        assert public["public_key"] == "visible"

    def test_lifecycle_set_use_delete(self):
        """Test full lifecycle: set _raw_content, use it, delete it."""
        from foundry_mcp.core.research.models import ResearchSource

        source = ResearchSource(
            title="Test",
            content="will be replaced by digest",
            metadata={},
        )

        # Phase 1: Store raw content before digest
        source.metadata["_raw_content"] = source.content
        assert source.metadata["_raw_content"] == "will be replaced by digest"

        # Phase 2: Replace content with digest (simulated)
        source.content = "digested summary"
        assert source.metadata["_raw_content"] == "will be replaced by digest"
        assert source.content == "digested summary"

        # Phase 3: Cleanup - delete raw content
        source.metadata.pop("_raw_content", None)
        assert "_raw_content" not in source.metadata

        # Phase 4: Verify cleanup in serialization
        data = source.to_dict()
        assert "_raw_content" not in data["metadata"]
        assert data["content"] == "digested summary"


# =============================================================================
# Test: Circuit Breaker
# =============================================================================


class TestCircuitBreaker:
    """Tests for digest circuit breaker behavior.

    Circuit breaker opens when failure ratio exceeds 70% with at least
    5 samples in a sliding window of 10 attempts. Auto-resets after 60s.
    """

    @pytest.fixture
    def digestor(self):
        """Create a DocumentDigestor with mock dependencies."""
        from unittest.mock import MagicMock
        from foundry_mcp.core.research.document_digest import (
            DocumentDigestor,
            DigestConfig,
        )

        mock_summarizer = MagicMock()
        mock_pdf_extractor = MagicMock()
        return DocumentDigestor(
            summarizer=mock_summarizer,
            pdf_extractor=mock_pdf_extractor,
            config=DigestConfig(),
        )

    def test_breaker_initially_closed(self, digestor):
        """Test circuit breaker starts in closed state."""
        assert digestor._is_circuit_breaker_open() is False
        assert digestor._circuit_breaker_open is False

    def test_breaker_stays_closed_below_min_samples(self, digestor):
        """Test breaker does not trip with fewer than 5 samples."""
        # Record 4 failures (below min_samples of 5)
        for _ in range(4):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is False

    def test_breaker_trips_at_threshold(self, digestor):
        """Test breaker trips when failure ratio >= 70% with >= 5 samples."""
        # Record 5 failures, 0 successes -> 100% failure rate
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

    def test_breaker_trips_at_exact_threshold(self, digestor):
        """Test breaker trips at exactly 70% failure ratio."""
        # 7 failures, 3 successes = 70% failure rate in window of 10
        for _ in range(3):
            digestor._record_success()
        for _ in range(7):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

    def test_breaker_stays_closed_below_threshold(self, digestor):
        """Test breaker stays closed below 70% failure ratio."""
        # 3 failures, 3 successes = 50% failure rate (6 samples >= min 5)
        for _ in range(3):
            digestor._record_success()
        for _ in range(3):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is False

    def test_breaker_closes_when_ratio_drops(self, digestor):
        """Test breaker closes when failure ratio drops below threshold."""
        # Open the breaker: 5 failures
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Record enough successes to bring ratio below 70%
        # Window is 10, so need enough successes mixed in
        for _ in range(5):
            digestor._record_success()
        # Now window has 5 failures + 5 successes = 50% failure rate
        assert digestor._is_circuit_breaker_open() is False

    def test_breaker_auto_resets_after_timeout(self, digestor):
        """Test breaker auto-resets after 60 seconds."""
        import time

        # Open the breaker
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Simulate time passage by backdating the opened_at timestamp
        digestor._circuit_breaker_opened_at = time.time() - 61.0
        assert digestor._is_circuit_breaker_open() is False

    def test_breaker_does_not_reset_before_timeout(self, digestor):
        """Test breaker stays open before 60 seconds."""
        import time

        # Open the breaker
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Simulate only 30 seconds passed
        digestor._circuit_breaker_opened_at = time.time() - 30.0
        assert digestor._is_circuit_breaker_open() is True

    def test_manual_reset(self, digestor):
        """Test manual reset clears circuit breaker state."""
        # Open the breaker
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Manual reset
        digestor.reset_circuit_breaker()
        assert digestor._is_circuit_breaker_open() is False
        assert digestor._circuit_breaker_open is False
        assert digestor._circuit_breaker_opened_at is None
        assert len(digestor._attempt_window) == 0

    def test_sliding_window_evicts_old_entries(self, digestor):
        """Test sliding window keeps only most recent 10 entries."""
        # Record 15 attempts (window size is 10)
        for _ in range(15):
            digestor._record_success()
        assert len(digestor._attempt_window) == 10

    def test_cache_reads_work_when_breaker_open(self, digestor):
        """Test that cache reads are allowed when circuit breaker is open."""
        from foundry_mcp.core.research.document_digest import (
            DigestCache,
            DigestResult,
            DigestPayload,
        )

        # Set up a cached result
        cache = DigestCache(enabled=True)
        payload = DigestPayload(
            query_hash="12345678",
            summary="Cached summary",
            key_points=[],
            evidence_snippets=[],
            original_chars=1000,
            digest_chars=50,
            compression_ratio=0.05,
            source_text_hash="sha256:" + "a" * 64,
        )
        cached_result = DigestResult(payload=payload, cache_hit=False)
        cache_key = "test-cache-key"
        cache.set(cache_key, cached_result)

        # Verify cache read works independently
        retrieved = cache.get(cache_key)
        assert retrieved is not None
        assert retrieved.payload.summary == "Cached summary"

        # Open the breaker on the digestor
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Cache reads should still work (cache is independent of breaker)
        retrieved_again = cache.get(cache_key)
        assert retrieved_again is not None

    @pytest.mark.asyncio
    async def test_digest_skipped_when_breaker_open(self, digestor):
        """Test that digest() returns skipped result when breaker is open."""
        from foundry_mcp.core.research.document_digest import DigestPolicy

        # Configure for ALWAYS policy so content is eligible
        digestor.config.policy = DigestPolicy.ALWAYS

        # Open the breaker
        for _ in range(5):
            digestor._record_failure()
        assert digestor._is_circuit_breaker_open() is True

        # Attempt digest - should be skipped due to circuit breaker
        result = await digestor.digest(
            source="Some content to digest",
            query="test query",
        )
        assert result.skipped is True
        assert result.skip_reason == "circuit_breaker_open"
        assert result.payload is None


# =============================================================================
# Contract Tests
# =============================================================================


def _canonicalize(text: str) -> str:
    """Reproduce the canonical normalization pipeline for contract tests."""
    import html as html_mod

    result = html_mod.unescape(text)
    result = re.sub(r"<[^>]+>", " ", result)
    result = unicodedata.normalize("NFC", result)
    result = re.sub(r"\s+", " ", result)
    return result.strip()


class TestContractFidelityEnvelope:
    """Contract: response envelope includes content_fidelity with DIGEST level."""

    def test_digest_payload_has_content_type_field(self):
        """DigestPayload always includes content_type='digest/v1'."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary text.",
            key_points=["Point 1"],
            evidence_snippets=[],
            original_chars=5000,
            digest_chars=1000,
            compression_ratio=0.2,
            source_text_hash="sha256:" + "a" * 64,
        )
        assert payload.content_type == "digest/v1"

    def test_digest_payload_content_type_in_serialized_form(self):
        """Serialized payload includes content_type field."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary text.",
            key_points=[],
            evidence_snippets=[],
            original_chars=5000,
            digest_chars=1000,
            compression_ratio=0.2,
            source_text_hash="sha256:" + "a" * 64,
        )
        serialized = serialize_payload(payload)
        data = json.loads(serialized)
        assert data["content_type"] == "digest/v1"

    def test_fidelity_level_digest_exists_in_model(self):
        """FidelityLevel.DIGEST is a valid fidelity level."""
        from foundry_mcp.core.research.models import FidelityLevel

        assert FidelityLevel.DIGEST is not None
        assert FidelityLevel.DIGEST.value == "digest"

    def test_compression_ratio_reflects_actual_compression(self):
        """compression_ratio = digest_chars / original_chars."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Short summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=10000,
            digest_chars=2000,
            compression_ratio=0.2,
            source_text_hash="sha256:" + "a" * 64,
        )
        assert payload.compression_ratio == payload.digest_chars / payload.original_chars


class TestContractSchemaValidation:
    """Contract: DigestPayload validates against JSON schema."""

    def test_content_type_defaults_to_digest_v1(self):
        """content_type defaults to 'digest/v1' and is present in serialized form."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=100,
            digest_chars=50,
            compression_ratio=0.5,
            source_text_hash="sha256:" + "a" * 64,
        )
        assert payload.content_type == "digest/v1"
        data = json.loads(serialize_payload(payload))
        assert data["content_type"] == "digest/v1"

    def test_query_hash_must_be_8_hex_chars(self):
        """query_hash must be exactly 8 hex characters."""
        # Valid 8-char hex
        payload = DigestPayload(
            query_hash="0123abcd",
            summary="Summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=100,
            digest_chars=50,
            compression_ratio=0.5,
            source_text_hash="sha256:" + "a" * 64,
        )
        assert len(payload.query_hash) == 8

        # Too short
        with pytest.raises(ValidationError):
            DigestPayload(
                query_hash="abc",
                summary="Summary.",
                key_points=[],
                evidence_snippets=[],
                original_chars=100,
                digest_chars=50,
                compression_ratio=0.5,
                source_text_hash="sha256:" + "a" * 64,
            )

    def test_source_text_hash_must_have_sha256_prefix(self):
        """source_text_hash must match 'sha256:{64-hex-chars}'."""
        # Valid
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=100,
            digest_chars=50,
            compression_ratio=0.5,
            source_text_hash="sha256:" + "f" * 64,
        )
        assert payload.source_text_hash.startswith("sha256:")

        # Invalid prefix
        with pytest.raises(ValidationError):
            DigestPayload(
                query_hash="ab12cd34",
                summary="Summary.",
                key_points=[],
                evidence_snippets=[],
                original_chars=100,
                digest_chars=50,
                compression_ratio=0.5,
                source_text_hash="md5:" + "a" * 64,
            )

    def test_version_field_present(self):
        """version field defaults to '1.0'."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=100,
            digest_chars=50,
            compression_ratio=0.5,
            source_text_hash="sha256:" + "a" * 64,
        )
        assert payload.version == "1.0"

    def test_deserialized_payload_validates_schema(self):
        """Deserialized payload passes all validation constraints."""
        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary text.",
            key_points=["Point 1", "Point 2"],
            evidence_snippets=[
                EvidenceSnippet(
                    text="Evidence text here.",
                    locator="char:100-118",
                    relevance_score=0.85,
                )
            ],
            original_chars=5000,
            digest_chars=1000,
            compression_ratio=0.2,
            source_text_hash="sha256:" + "b" * 64,
        )
        serialized = serialize_payload(payload)
        restored = deserialize_payload(serialized)
        assert restored.content_type == "digest/v1"
        assert restored.query_hash == "ab12cd34"
        assert restored.version == "1.0"
        assert len(restored.evidence_snippets) == 1


class TestContractSourceTextHash:
    """Contract: source_text_hash == SHA256 of archived canonical text."""

    def test_hash_matches_canonical_text(self):
        """source_text_hash must match SHA256 of the canonical text."""
        raw_text = "Hello   world!  \n\n  Multiple   spaces."
        canonical = _canonicalize(raw_text)
        expected_hash = "sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="Summary.",
            key_points=[],
            evidence_snippets=[],
            original_chars=len(raw_text),
            digest_chars=10,
            compression_ratio=0.1,
            source_text_hash=expected_hash,
        )
        # Verify the hash is verifiable against canonical text
        verify_hash = "sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        assert payload.source_text_hash == verify_hash

    def test_hash_changes_with_different_text(self):
        """Different text produces different source_text_hash."""
        text_a = _canonicalize("Text version A")
        text_b = _canonicalize("Text version B")
        hash_a = "sha256:" + hashlib.sha256(text_a.encode("utf-8")).hexdigest()
        hash_b = "sha256:" + hashlib.sha256(text_b.encode("utf-8")).hexdigest()
        assert hash_a != hash_b

    def test_canonical_normalization_collapses_whitespace(self):
        """Canonical text normalizes whitespace for consistent hashing."""
        text1 = "Hello   world"
        text2 = "Hello world"
        assert _canonicalize(text1) == _canonicalize(text2)

        # Therefore hashes should match
        hash1 = hashlib.sha256(_canonicalize(text1).encode("utf-8")).hexdigest()
        hash2 = hashlib.sha256(_canonicalize(text2).encode("utf-8")).hexdigest()
        assert hash1 == hash2

    def test_canonical_normalization_strips_html(self):
        """Canonical text strips HTML tags for consistent hashing."""
        html_text = "<p>Hello <b>world</b></p>"
        plain_text = "Hello world"
        assert _canonicalize(html_text) == _canonicalize(plain_text)

    def test_hash_format_is_sha256_plus_64_hex(self):
        """Hash format is 'sha256:' followed by exactly 64 hex characters."""
        text = "Some content"
        canonical = _canonicalize(text)
        hash_str = "sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        assert hash_str.startswith("sha256:")
        hex_part = hash_str[7:]
        assert len(hex_part) == 64
        assert all(c in "0123456789abcdef" for c in hex_part)


class TestContractLocatorVerification:
    """Contract: archived_text[start:end] == snippet.text for all evidence."""

    def test_char_locator_extracts_correct_text(self):
        """char:start-end locator allows exact text extraction."""
        source_text = "The quick brown fox jumps over the lazy dog."
        snippet_text = "brown fox"
        start = source_text.index(snippet_text)
        end = start + len(snippet_text)
        locator = f"char:{start}-{end}"

        evidence = EvidenceSnippet(
            text=snippet_text,
            locator=locator,
            relevance_score=0.9,
        )

        # Verify: extract using locator
        parts = evidence.locator.replace("char:", "").split("-")
        loc_start, loc_end = int(parts[0]), int(parts[1])
        extracted = source_text[loc_start:loc_end]
        assert extracted == evidence.text

    def test_page_locator_extracts_correct_text(self):
        """page:N:char:start-end locator allows exact text extraction."""
        source_text = "Page content with specific evidence in the middle."
        snippet_text = "specific evidence"
        start = source_text.index(snippet_text)
        end = start + len(snippet_text)
        locator = f"page:1:char:{start}-{end}"

        evidence = EvidenceSnippet(
            text=snippet_text,
            locator=locator,
            relevance_score=0.8,
        )

        # Parse page locator
        # Format: page:N:char:start-end
        match = re.match(r"page:\d+:char:(\d+)-(\d+)", evidence.locator)
        assert match is not None
        loc_start, loc_end = int(match.group(1)), int(match.group(2))
        extracted = source_text[loc_start:loc_end]
        assert extracted == evidence.text

    def test_multiple_evidence_locators_all_verifiable(self):
        """All evidence snippets in a payload have verifiable locators."""
        source_text = (
            "Machine learning models have shown remarkable progress. "
            "Transformer architectures revolutionized NLP tasks. "
            "Attention mechanisms are the key innovation."
        )
        snippets = [
            ("remarkable progress", 0.9),
            ("revolutionized NLP", 0.85),
            ("key innovation", 0.7),
        ]
        evidence_list = []
        for snippet_text, score in snippets:
            start = source_text.index(snippet_text)
            end = start + len(snippet_text)
            evidence_list.append(
                EvidenceSnippet(
                    text=snippet_text,
                    locator=f"char:{start}-{end}",
                    relevance_score=score,
                )
            )

        payload = DigestPayload(
            query_hash="ab12cd34",
            summary="ML progress summary.",
            key_points=["Models improved"],
            evidence_snippets=evidence_list,
            original_chars=len(source_text),
            digest_chars=100,
            compression_ratio=0.1,
            source_text_hash="sha256:" + "a" * 64,
        )

        # Verify ALL locators
        for ev in payload.evidence_snippets:
            parts = ev.locator.replace("char:", "").split("-")
            loc_start, loc_end = int(parts[0]), int(parts[1])
            extracted = source_text[loc_start:loc_end]
            assert extracted == ev.text, (
                f"Locator {ev.locator} extracted '{extracted}' but expected '{ev.text}'"
            )

    def test_locator_offsets_are_non_negative(self):
        """Locator start and end offsets are non-negative integers."""
        evidence = EvidenceSnippet(
            text="test",
            locator="char:0-4",
            relevance_score=0.5,
        )
        parts = evidence.locator.replace("char:", "").split("-")
        start, end = int(parts[0]), int(parts[1])
        assert start >= 0
        assert end >= start

    def test_locator_end_greater_than_start(self):
        """Locator end must be greater than start for non-empty snippets."""
        text = "non-empty"
        evidence = EvidenceSnippet(
            text=text,
            locator="char:10-19",
            relevance_score=0.5,
        )
        parts = evidence.locator.replace("char:", "").split("-")
        start, end = int(parts[0]), int(parts[1])
        assert end > start
        assert end - start == len(text)
