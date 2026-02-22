"""Declarative parameter validation for unified tool handlers.

Replaces imperative per-field validation boilerplate with schema-driven
validation.  Each handler declares a schema dict mapping field names to
type descriptors, then calls :func:`validate_payload` once.

The engine produces error envelopes identical to those from
:func:`~foundry_mcp.tools.unified.common.make_validation_error_fn` so the
change is fully backward-compatible.

Example::

    _SCHEMA = {
        "spec_id": Str(required=True, remediation="Pass the spec identifier"),
        "title": Str(required=True),
        "hours": Num(min_val=0),
        "dry_run": Bool(default=False),
    }


    def _handle(*, config, **payload):
        err = validate_payload(payload, _SCHEMA, tool_name="authoring", action="do-thing", request_id=rid)
        if err:
            return err
        # payload values are now validated and normalised in-place
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

from foundry_mcp.core.responses.types import ErrorCode

# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Str:
    """String parameter."""

    required: bool = False
    strip: bool = True
    allow_empty: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    choices: Optional[FrozenSet[str]] = None
    error_code: ErrorCode = ErrorCode.INVALID_FORMAT
    remediation: Optional[str] = None


@dataclass(frozen=True)
class Num:
    """Numeric parameter (int or float)."""

    required: bool = False
    integer_only: bool = False
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None
    error_code: ErrorCode = ErrorCode.INVALID_FORMAT
    remediation: Optional[str] = None


@dataclass(frozen=True)
class Bool:
    """Boolean parameter."""

    required: bool = False
    default: Optional[bool] = None
    error_code: ErrorCode = ErrorCode.INVALID_FORMAT
    remediation: Optional[str] = None


@dataclass(frozen=True)
class List_:
    """List parameter."""

    required: bool = False
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    error_code: ErrorCode = ErrorCode.INVALID_FORMAT
    remediation: Optional[str] = None


@dataclass(frozen=True)
class Dict_:
    """Dict parameter."""

    required: bool = False
    error_code: ErrorCode = ErrorCode.INVALID_FORMAT
    remediation: Optional[str] = None


@dataclass(frozen=True)
class AtLeastOne:
    """Cross-field rule: at least one of *fields* must be non-None."""

    fields: Tuple[str, ...]
    error_code: ErrorCode = ErrorCode.VALIDATION_ERROR
    remediation: Optional[str] = None


# Union of all field-level schema types.
FieldSchema = Union[Str, Num, Bool, List_, Dict_]


# ---------------------------------------------------------------------------
# Validation engine
# ---------------------------------------------------------------------------


def validate_payload(
    payload: Dict[str, Any],
    schema: Mapping[str, FieldSchema],
    *,
    tool_name: str,
    action: str,
    request_id: Optional[str] = None,
    cross_field_rules: Optional[List[AtLeastOne]] = None,
) -> Optional[dict]:
    """Validate *payload* against *schema*, returning an error dict or ``None``.

    On success, payload values are **normalised in-place** (strings stripped,
    numeric types coerced, boolean defaults applied).

    Validation order matches existing imperative handlers:
      1. Required field presence
      2. Type check (with ``bool``-is-not-``int`` guard)
      3. Format checks (empty string, length, choices, range)
      4. Normalisation (strip strings, coerce to float/int)
      5. Cross-field rules (:class:`AtLeastOne`)
    """
    from dataclasses import asdict as _asdict

    from foundry_mcp.core.responses.builders import error_response
    from foundry_mcp.core.responses.types import ErrorType

    def _error(
        field: str,
        message: str,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        remediation: Optional[str] = None,
    ) -> dict:
        effective_remediation = remediation or f"Provide a valid '{field}' value"
        return _asdict(
            error_response(
                f"Invalid field '{field}' for {tool_name}.{action}: {message}",
                error_code=code,
                error_type=ErrorType.VALIDATION,
                remediation=effective_remediation,
                details={"field": field, "action": f"{tool_name}.{action}"},
                request_id=request_id,
            )
        )

    for field_name, spec in schema.items():
        value = payload.get(field_name)

        # --- Bool: apply defaults early so presence check works ----
        if isinstance(spec, Bool) and spec.default is not None and value is None:
            payload[field_name] = spec.default
            value = spec.default

        # 1. Required presence -----------------------------------
        if spec.required and value is None:
            return _error(
                field_name,
                f"Provide a non-empty {field_name} parameter",
                code=ErrorCode.MISSING_REQUIRED,
                remediation=spec.remediation,
            )

        # Nothing more to validate for absent optional fields.
        if value is None:
            continue

        # 2. Type check ------------------------------------------
        err = _check_type(field_name, value, spec, _error)
        if err is not None:
            return err

        # 3. Format / range checks --------------------------------
        err = _check_format(field_name, value, spec, _error)
        if err is not None:
            return err

        # 4. Normalisation ----------------------------------------
        _normalise(field_name, payload, spec)

    # 5. Cross-field rules ----------------------------------------
    if cross_field_rules:
        for rule in cross_field_rules:
            if all(payload.get(f) is None for f in rule.fields):
                names = ", ".join(f"'{f}'" for f in rule.fields)
                return _error(
                    rule.fields[0],
                    f"At least one of {names} must be provided",
                    code=rule.error_code,
                    remediation=rule.remediation,
                )

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ErrorFn = Any  # Callable[[str, str, ...], dict]


def _check_type(
    field: str,
    value: Any,
    spec: FieldSchema,
    _error: _ErrorFn,
) -> Optional[dict]:
    """Return an error dict if *value* fails the type check for *spec*."""
    if isinstance(spec, Str):
        if not isinstance(value, str):
            return _error(field, f"{field} must be a string", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, Num):
        # bool is a subclass of int — reject booleans explicitly.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = "Provide an integer value" if spec.integer_only else "Provide a numeric value"
            return _error(field, msg, code=spec.error_code, remediation=spec.remediation)
        if spec.integer_only and not isinstance(value, int):
            return _error(field, f"{field} must be an integer", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, Bool):
        if not isinstance(value, bool):
            return _error(field, "Expected a boolean value", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, List_):
        if not isinstance(value, list):
            return _error(field, f"{field} must be a list", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, Dict_):
        if not isinstance(value, dict):
            return _error(field, f"{field} must be a dict", code=spec.error_code, remediation=spec.remediation)

    return None


def _check_format(
    field: str,
    value: Any,
    spec: FieldSchema,
    _error: _ErrorFn,
) -> Optional[dict]:
    """Return an error dict if *value* fails format/range checks for *spec*."""
    if isinstance(spec, Str):
        text = value.strip() if spec.strip else value
        if not spec.allow_empty and spec.required and not text:
            return _error(
                field,
                f"Provide a non-empty {field} parameter",
                code=ErrorCode.MISSING_REQUIRED,
                remediation=spec.remediation,
            )
        if spec.min_length is not None and len(text) < spec.min_length:
            return _error(
                field,
                f"{field} must be at least {spec.min_length} characters",
                code=spec.error_code,
                remediation=spec.remediation,
            )
        if spec.max_length is not None and len(text) > spec.max_length:
            return _error(
                field,
                f"{field} must be at most {spec.max_length} characters",
                code=spec.error_code,
                remediation=spec.remediation,
            )
        if spec.choices is not None and text not in spec.choices:
            allowed = ", ".join(sorted(spec.choices))
            return _error(field, f"Must be one of: {allowed}", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, Num):
        if spec.min_val is not None and value < spec.min_val:
            return _error(field, f"Value must be >= {spec.min_val}", code=spec.error_code, remediation=spec.remediation)
        if spec.max_val is not None and value > spec.max_val:
            return _error(field, f"Value must be <= {spec.max_val}", code=spec.error_code, remediation=spec.remediation)

    elif isinstance(spec, List_):
        if spec.min_items is not None and len(value) < spec.min_items:
            return _error(
                field,
                f"{field} must have at least {spec.min_items} items",
                code=spec.error_code,
                remediation=spec.remediation,
            )
        if spec.max_items is not None and len(value) > spec.max_items:
            return _error(
                field,
                f"{field} must have at most {spec.max_items} items",
                code=spec.error_code,
                remediation=spec.remediation,
            )

    return None


def _normalise(
    field: str,
    payload: Dict[str, Any],
    spec: FieldSchema,
) -> None:
    """Normalise *payload[field]* in-place according to *spec*."""
    value = payload[field]

    if isinstance(spec, Str) and spec.strip and isinstance(value, str):
        payload[field] = value.strip()

    elif isinstance(spec, Num) and not spec.integer_only and isinstance(value, (int, float)):
        payload[field] = float(value)

    elif isinstance(spec, Bool) and spec.default is not None and value is None:
        payload[field] = spec.default
