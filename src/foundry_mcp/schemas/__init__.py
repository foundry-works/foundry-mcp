"""Bundled JSON schemas for SDD specifications."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def load_schema(name: str = "sdd-spec-schema.json") -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load a bundled JSON schema by name.

    Args:
        name: Schema filename (default: sdd-spec-schema.json).

    Returns:
        Tuple of (schema_dict, error_message). On success, error is None.
        On failure, schema is None and error contains the reason.
    """
    schema_path = Path(__file__).parent / name

    if not schema_path.exists():
        return None, f"Schema file not found: {name}"

    try:
        with open(schema_path, "r") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON in schema: {e}"
    except Exception as e:
        return None, f"Error loading schema: {e}"


def get_spec_schema() -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Load the SDD spec JSON schema.

    Returns:
        Tuple of (schema_dict, error_message).
    """
    return load_schema("sdd-spec-schema.json")
