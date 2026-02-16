"""Unit tests for the declarative parameter validation framework."""

from __future__ import annotations

import pytest

from foundry_mcp.core.responses import ErrorCode
from foundry_mcp.tools.unified.param_schema import (
    AtLeastOne,
    Bool,
    Dict_,
    List_,
    Num,
    Str,
    validate_payload,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL = "test-tool"
ACTION = "test-action"
RID = "req_test_123"


def _validate(payload, schema, **kw):
    """Shorthand for validate_payload with test defaults."""
    return validate_payload(
        payload,
        schema,
        tool_name=kw.pop("tool_name", TOOL),
        action=kw.pop("action", ACTION),
        request_id=kw.pop("request_id", RID),
        **kw,
    )


def _assert_error(result, *, field, code=None):
    """Assert *result* is a validation error dict for *field*."""
    assert result is not None, "Expected a validation error, got None"
    assert result["success"] is False
    assert field in result["error"]
    details = result["data"].get("details", {})
    assert details["field"] == field
    assert details["action"] == f"{TOOL}.{ACTION}"
    assert result["meta"]["version"] == "response-v2"
    if code is not None:
        assert result["data"]["error_code"] == code


# ===================================================================
# Str
# ===================================================================

class TestStr:
    def test_required_present(self):
        payload = {"name": "hello"}
        assert _validate(payload, {"name": Str(required=True)}) is None

    def test_required_missing(self):
        result = _validate({}, {"name": Str(required=True)})
        _assert_error(result, field="name", code="MISSING_REQUIRED")

    def test_required_none(self):
        result = _validate({"name": None}, {"name": Str(required=True)})
        _assert_error(result, field="name", code="MISSING_REQUIRED")

    def test_required_empty_string(self):
        result = _validate({"name": ""}, {"name": Str(required=True)})
        _assert_error(result, field="name", code="MISSING_REQUIRED")

    def test_required_whitespace_only(self):
        result = _validate({"name": "   "}, {"name": Str(required=True)})
        _assert_error(result, field="name", code="MISSING_REQUIRED")

    def test_wrong_type_int(self):
        result = _validate({"name": 42}, {"name": Str()})
        _assert_error(result, field="name")

    def test_wrong_type_bool(self):
        result = _validate({"name": True}, {"name": Str()})
        _assert_error(result, field="name")

    def test_optional_absent(self):
        assert _validate({}, {"name": Str()}) is None

    def test_optional_none(self):
        assert _validate({"name": None}, {"name": Str()}) is None

    def test_strip_normalisation(self):
        payload = {"name": "  hello  "}
        assert _validate(payload, {"name": Str(strip=True)}) is None
        assert payload["name"] == "hello"

    def test_strip_disabled(self):
        payload = {"name": "  hello  "}
        assert _validate(payload, {"name": Str(strip=False)}) is None
        assert payload["name"] == "  hello  "

    def test_choices_valid(self):
        schema = {"cat": Str(choices=frozenset({"a", "b", "c"}))}
        assert _validate({"cat": "b"}, schema) is None

    def test_choices_invalid(self):
        schema = {"cat": Str(choices=frozenset({"a", "b", "c"}))}
        result = _validate({"cat": "d"}, schema)
        _assert_error(result, field="cat", code="VALIDATION_ERROR")

    def test_choices_after_strip(self):
        schema = {"cat": Str(choices=frozenset({"a", "b"}))}
        assert _validate({"cat": "  a  "}, schema) is None

    def test_min_length(self):
        schema = {"name": Str(min_length=3)}
        assert _validate({"name": "abc"}, schema) is None
        result = _validate({"name": "ab"}, schema)
        _assert_error(result, field="name", code="VALIDATION_ERROR")

    def test_max_length(self):
        schema = {"name": Str(max_length=5)}
        assert _validate({"name": "hello"}, schema) is None
        result = _validate({"name": "toolong"}, schema)
        _assert_error(result, field="name", code="VALIDATION_ERROR")

    def test_custom_error_code(self):
        schema = {"x": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED)}
        result = _validate({}, schema)
        _assert_error(result, field="x", code="MISSING_REQUIRED")

    def test_custom_remediation(self):
        schema = {"x": Str(required=True, remediation="Do the thing")}
        result = _validate({}, schema)
        assert result["data"]["remediation"] == "Do the thing"

    def test_allow_empty(self):
        schema = {"x": Str(required=True, allow_empty=True)}
        # allow_empty=True means empty string passes the format check.
        # The value is present (not None), so required check passes too.
        payload = {"x": ""}
        result = _validate(payload, schema)
        assert result is None


# ===================================================================
# Num
# ===================================================================

class TestNum:
    def test_valid_int(self):
        payload = {"n": 5}
        assert _validate(payload, {"n": Num()}) is None

    def test_valid_float(self):
        payload = {"n": 3.14}
        assert _validate(payload, {"n": Num()}) is None

    def test_bool_rejected(self):
        """bool is subclass of int but must be rejected for numeric fields."""
        result = _validate({"n": True}, {"n": Num()})
        _assert_error(result, field="n")

    def test_string_rejected(self):
        result = _validate({"n": "5"}, {"n": Num()})
        _assert_error(result, field="n")

    def test_required_missing(self):
        result = _validate({}, {"n": Num(required=True)})
        _assert_error(result, field="n", code="MISSING_REQUIRED")

    def test_optional_absent(self):
        assert _validate({}, {"n": Num()}) is None

    def test_min_val(self):
        schema = {"n": Num(min_val=0)}
        assert _validate({"n": 0}, schema) is None
        assert _validate({"n": 1}, schema) is None
        result = _validate({"n": -1}, schema)
        _assert_error(result, field="n")

    def test_max_val(self):
        schema = {"n": Num(max_val=10)}
        assert _validate({"n": 10}, schema) is None
        result = _validate({"n": 11}, schema)
        _assert_error(result, field="n")

    def test_min_and_max(self):
        schema = {"n": Num(min_val=1, max_val=10)}
        assert _validate({"n": 5}, schema) is None
        result = _validate({"n": 0}, schema)
        _assert_error(result, field="n")
        result = _validate({"n": 11}, schema)
        _assert_error(result, field="n")

    def test_integer_only_accepts_int(self):
        schema = {"n": Num(integer_only=True)}
        assert _validate({"n": 5}, schema) is None

    def test_integer_only_rejects_float(self):
        schema = {"n": Num(integer_only=True)}
        result = _validate({"n": 3.14}, schema)
        _assert_error(result, field="n")

    def test_float_coercion(self):
        """Non-integer_only Num should coerce int to float."""
        payload = {"n": 5}
        assert _validate(payload, {"n": Num()}) is None
        assert isinstance(payload["n"], float)
        assert payload["n"] == 5.0

    def test_integer_only_no_coercion(self):
        """integer_only Num should NOT coerce to float."""
        payload = {"n": 5}
        assert _validate(payload, {"n": Num(integer_only=True)}) is None
        assert isinstance(payload["n"], int)

    def test_negative_zero(self):
        """Negative zero should pass min_val=0."""
        assert _validate({"n": -0.0}, {"n": Num(min_val=0)}) is None

    def test_custom_remediation(self):
        schema = {"n": Num(min_val=0, remediation="Set hours to zero or greater")}
        result = _validate({"n": -1}, schema)
        assert result["data"]["remediation"] == "Set hours to zero or greater"


# ===================================================================
# Bool
# ===================================================================

class TestBool:
    def test_true(self):
        payload = {"flag": True}
        assert _validate(payload, {"flag": Bool()}) is None

    def test_false(self):
        payload = {"flag": False}
        assert _validate(payload, {"flag": Bool()}) is None

    def test_wrong_type_int(self):
        result = _validate({"flag": 1}, {"flag": Bool()})
        _assert_error(result, field="flag")

    def test_wrong_type_str(self):
        result = _validate({"flag": "true"}, {"flag": Bool()})
        _assert_error(result, field="flag")

    def test_default_applied(self):
        payload = {}
        schema = {"flag": Bool(default=False)}
        assert _validate(payload, schema) is None
        assert payload["flag"] is False

    def test_default_not_overridden(self):
        payload = {"flag": True}
        schema = {"flag": Bool(default=False)}
        assert _validate(payload, schema) is None
        assert payload["flag"] is True

    def test_required_missing(self):
        result = _validate({}, {"flag": Bool(required=True)})
        _assert_error(result, field="flag", code="MISSING_REQUIRED")

    def test_optional_absent_no_default(self):
        assert _validate({}, {"flag": Bool()}) is None


# ===================================================================
# List_
# ===================================================================

class TestList:
    def test_valid(self):
        assert _validate({"items": [1, 2]}, {"items": List_()}) is None

    def test_empty_list(self):
        assert _validate({"items": []}, {"items": List_()}) is None

    def test_wrong_type(self):
        result = _validate({"items": "not a list"}, {"items": List_()})
        _assert_error(result, field="items")

    def test_required_missing(self):
        result = _validate({}, {"items": List_(required=True)})
        _assert_error(result, field="items", code="MISSING_REQUIRED")

    def test_min_items(self):
        schema = {"items": List_(min_items=2)}
        assert _validate({"items": [1, 2]}, schema) is None
        result = _validate({"items": [1]}, schema)
        _assert_error(result, field="items")

    def test_max_items(self):
        schema = {"items": List_(max_items=2)}
        assert _validate({"items": [1, 2]}, schema) is None
        result = _validate({"items": [1, 2, 3]}, schema)
        _assert_error(result, field="items")

    def test_optional_absent(self):
        assert _validate({}, {"items": List_()}) is None


# ===================================================================
# Dict_
# ===================================================================

class TestDict:
    def test_valid(self):
        assert _validate({"meta": {"k": "v"}}, {"meta": Dict_()}) is None

    def test_empty_dict(self):
        assert _validate({"meta": {}}, {"meta": Dict_()}) is None

    def test_wrong_type(self):
        result = _validate({"meta": [1]}, {"meta": Dict_()})
        _assert_error(result, field="meta")

    def test_required_missing(self):
        result = _validate({}, {"meta": Dict_(required=True)})
        _assert_error(result, field="meta", code="MISSING_REQUIRED")

    def test_optional_absent(self):
        assert _validate({}, {"meta": Dict_()}) is None


# ===================================================================
# AtLeastOne cross-field rule
# ===================================================================

class TestAtLeastOne:
    def test_one_present(self):
        rule = AtLeastOne(fields=("a", "b", "c"))
        result = _validate({"a": "x"}, {"a": Str(), "b": Str(), "c": Str()},
                           cross_field_rules=[rule])
        assert result is None

    def test_all_present(self):
        rule = AtLeastOne(fields=("a", "b"))
        result = _validate({"a": "x", "b": "y"}, {"a": Str(), "b": Str()},
                           cross_field_rules=[rule])
        assert result is None

    def test_none_present(self):
        rule = AtLeastOne(fields=("a", "b", "c"),
                          remediation="Provide at least one metadata field")
        result = _validate({}, {"a": Str(), "b": Str(), "c": Str()},
                           cross_field_rules=[rule])
        _assert_error(result, field="a")
        assert result["data"]["remediation"] == "Provide at least one metadata field"

    def test_custom_error_code(self):
        rule = AtLeastOne(fields=("x", "y"), error_code=ErrorCode.MISSING_REQUIRED)
        result = _validate({}, {"x": Str(), "y": Str()},
                           cross_field_rules=[rule])
        _assert_error(result, field="x", code="MISSING_REQUIRED")


# ===================================================================
# Error envelope structure
# ===================================================================

class TestErrorEnvelope:
    """Verify the error dict matches the canonical make_validation_error_fn output."""

    def test_envelope_fields(self):
        result = _validate({}, {"x": Str(required=True)})
        assert result is not None
        # Top-level keys
        assert set(result.keys()) == {"success", "data", "error", "meta"}
        assert result["success"] is False
        assert isinstance(result["error"], str)

        # Data section
        data = result["data"]
        assert "error_code" in data
        assert "error_type" in data
        assert data["error_type"] == "validation"
        assert "remediation" in data
        details = data["details"]
        assert "field" in details
        assert "action" in details

        # Meta section
        meta = result["meta"]
        assert meta["version"] == "response-v2"

    def test_request_id_propagated(self):
        result = _validate({}, {"x": Str(required=True)}, request_id="req_abc")
        assert result["meta"].get("request_id") == "req_abc"

    def test_error_message_format(self):
        result = _validate({}, {"myfield": Str(required=True)},
                           tool_name="authoring", action="phase-add")
        assert "myfield" in result["error"]
        assert "authoring.phase-add" in result["error"]

    def test_details_field_and_action(self):
        result = _validate({}, {"x": Str(required=True)},
                           tool_name="task", action="add")
        assert result["data"]["details"]["field"] == "x"
        assert result["data"]["details"]["action"] == "task.add"


# ===================================================================
# Multi-field schemas
# ===================================================================

class TestMultiField:
    """Test schemas with multiple fields to verify ordering and short-circuit."""

    def test_first_error_returned(self):
        """validate_payload returns on first error (dict ordering)."""
        schema = {
            "a": Str(required=True),
            "b": Str(required=True),
            "c": Str(required=True),
        }
        result = _validate({}, schema)
        # Should fail on "a" (first in dict order)
        _assert_error(result, field="a")

    def test_all_valid(self):
        schema = {
            "spec_id": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
            "title": Str(required=True, error_code=ErrorCode.MISSING_REQUIRED),
            "description": Str(),
            "estimated_hours": Num(min_val=0),
            "position": Num(integer_only=True, min_val=0),
            "link_previous": Bool(default=True),
            "dry_run": Bool(default=False),
            "path": Str(),
        }
        payload = {
            "spec_id": "my-spec",
            "title": "Phase 1",
            "description": "A phase",
            "estimated_hours": 4,
            "position": 0,
        }
        assert _validate(payload, schema) is None
        # Check normalisations
        assert payload["link_previous"] is True  # default applied
        assert payload["dry_run"] is False  # default applied
        assert isinstance(payload["estimated_hours"], float)  # coerced

    def test_second_field_error(self):
        schema = {
            "spec_id": Str(required=True),
            "title": Str(required=True),
        }
        result = _validate({"spec_id": "ok"}, schema)
        _assert_error(result, field="title")


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    def test_unicode_whitespace_string(self):
        """Non-breaking spaces and other unicode whitespace should be stripped."""
        payload = {"name": "\u00a0\u2003hello\u00a0"}
        assert _validate(payload, {"name": Str(required=True)}) is None
        assert payload["name"] == "\u00a0\u2003hello\u00a0".strip()

    def test_empty_schema(self):
        """Empty schema passes any payload."""
        assert _validate({"anything": "goes"}, {}) is None

    def test_extra_fields_ignored(self):
        """Fields not in schema are left untouched."""
        payload = {"known": "value", "extra": 42}
        assert _validate(payload, {"known": Str()}) is None
        assert payload["extra"] == 42

    def test_none_request_id(self):
        """None request_id should still produce valid envelope."""
        result = validate_payload(
            {}, {"x": Str(required=True)},
            tool_name=TOOL, action=ACTION, request_id=None,
        )
        assert result is not None
        assert result["success"] is False

    def test_bool_default_false_not_treated_as_missing(self):
        """Bool(default=False) should apply the default, not trigger required check."""
        payload = {}
        schema = {"flag": Bool(default=False)}
        assert _validate(payload, schema) is None
        assert payload["flag"] is False

    def test_num_zero_not_treated_as_missing(self):
        """0 is a valid value for optional Num fields."""
        payload = {"n": 0}
        assert _validate(payload, {"n": Num(min_val=0)}) is None

    def test_required_num_zero_is_valid(self):
        """0 is a valid value for required Num fields."""
        payload = {"n": 0}
        assert _validate(payload, {"n": Num(required=True)}) is None
