"""JSON extraction utilities for deep research workflows."""

from __future__ import annotations

import re
from typing import Optional


def extract_json(content: str) -> Optional[str]:
    """Extract JSON object from content that may contain other text.

    Handles cases where JSON is wrapped in markdown code blocks
    or mixed with explanatory text.

    Args:
        content: Raw content that may contain JSON

    Returns:
        Extracted JSON string or None if not found
    """
    # First, try to find JSON in code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(code_block_pattern, content)
    for match in matches:
        match = match.strip()
        if match.startswith("{"):
            return match

    # Try to find raw JSON object
    # Look for the outermost { ... } pair
    brace_start = content.find("{")
    if brace_start == -1:
        return None

    # Find matching closing brace, skipping braces inside JSON strings.
    depth = 0
    in_string = False
    escape = False
    for i, char in enumerate(content[brace_start:], brace_start):
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[brace_start : i + 1]

    return None
