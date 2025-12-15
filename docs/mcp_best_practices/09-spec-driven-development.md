# 9. Spec-Driven Development & Documentation

> Treat specifications as the source of truth and keep them synchronized with code.

## Overview

Specifications define tool contracts before implementation. Keeping specs synchronized with code ensures documentation accuracy and enables automated validation.

## Requirements

### MUST

- **Create specs before implementation** - design-first approach
- **Update specs with code changes** - same PR/commit
- **Document all public interfaces** - inputs, outputs, errors
- **Include version information** - in both spec and response

### SHOULD

- **Use machine-readable formats** - JSON Schema, OpenAPI
- **Automate spec validation** - CI/CD checks
- **Include examples** - for each operation
- **Document edge cases** - empty results, error conditions

### MAY

- **Generate documentation from specs** - automated doc sites
- **Generate client code from specs** - SDKs, type definitions
- **Track spec history** - changelog in spec file

## Spec Structure

### Tool Specification Format

```json
{
    "name": "get_user",
    "version": "1.2.0",
    "description": "Retrieve user details by ID",
    "category": "users",

    "inputs": {
        "type": "object",
        "required": ["user_id"],
        "properties": {
            "user_id": {
                "type": "string",
                "description": "Unique user identifier",
                "pattern": "^usr_[a-zA-Z0-9]+$",
                "examples": ["usr_abc123"]
            },
            "include_profile": {
                "type": "boolean",
                "default": false,
                "description": "Include extended profile data"
            }
        }
    },

    "outputs": {
        "success": {
            "description": "User found",
            "schema": {
                "type": "object",
                "properties": {
                    "user": {
                        "$ref": "#/definitions/User"
                    }
                }
            },
            "example": {
                "success": true,
                "data": {
                    "user": {
                        "id": "usr_abc123",
                        "name": "Alice",
                        "email": "alice@example.com"
                    }
                },
                "meta": {"version": "response-v2"}
            }
        },
        "errors": [
            {
                "code": "USER_NOT_FOUND",
                "description": "User ID does not exist",
                "example": {
                    "success": false,
                    "error": "User 'usr_999' not found",
                    "data": {"error_code": "USER_NOT_FOUND"},
                    "meta": {"version": "response-v2"}
                }
            }
        ]
    },

    "idempotency": "naturally_idempotent",
    "rate_limit": "100/minute",

    "changelog": [
        {"version": "1.2.0", "date": "2025-11-26", "changes": "Added include_profile parameter"},
        {"version": "1.1.0", "date": "2025-11-01", "changes": "Added email to response"},
        {"version": "1.0.0", "date": "2025-10-15", "changes": "Initial release"}
    ]
}
```

## Spec-Code Synchronization

### Same-PR Rule

Always update specs and code in the same pull request:

```
PR: Add include_profile parameter to get_user

Files changed:
  - specs/users/get_user.json    # Spec update
  - src/tools/users.py           # Implementation
  - tests/test_users.py          # Tests
```

### CI Validation

```yaml
# .github/workflows/spec-validation.yml
name: Spec Validation

on: [push, pull_request]

jobs:
  validate-specs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate JSON schemas
        run: |
          for spec in specs/**/*.json; do
            jsonschema --instance "$spec" schemas/tool-spec.schema.json
          done

      - name: Check spec-code sync
        run: python scripts/check_spec_sync.py

      - name: Verify examples
        run: python scripts/validate_examples.py
```

### Sync Checker Script

```python
#!/usr/bin/env python3
"""Check that specs are in sync with implementation."""

import json
import ast
from pathlib import Path

def extract_tool_signature(source_file: Path) -> dict:
    """Extract tool signature from Python source."""
    tree = ast.parse(source_file.read_text())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if getattr(decorator, 'attr', None) == 'tool':
                    return {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    }
    return None

def validate_spec_matches_code(spec_file: Path, source_file: Path) -> list:
    """Validate spec matches implementation."""
    errors = []

    spec = json.loads(spec_file.read_text())
    signature = extract_tool_signature(source_file)

    if not signature:
        errors.append(f"No tool found in {source_file}")
        return errors

    # Check name matches
    if spec["name"] != signature["name"]:
        errors.append(f"Name mismatch: spec={spec['name']}, code={signature['name']}")

    # Check required params exist in code
    spec_required = spec.get("inputs", {}).get("required", [])
    for param in spec_required:
        if param not in signature["args"]:
            errors.append(f"Required param '{param}' not in function signature")

    return errors

# Run validation
if __name__ == "__main__":
    errors = []
    for spec_file in Path("specs").rglob("*.json"):
        source_file = Path("src/tools") / f"{spec_file.stem}.py"
        if source_file.exists():
            errors.extend(validate_spec_matches_code(spec_file, source_file))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        exit(1)
    print("All specs in sync with code")
```

## Documentation Generation

### From Spec to Markdown

```python
def generate_tool_docs(spec: dict) -> str:
    """Generate markdown documentation from tool spec."""
    lines = [
        f"# {spec['name']}",
        "",
        spec.get("description", ""),
        "",
        "## Parameters",
        "",
        "| Name | Type | Required | Description |",
        "|------|------|----------|-------------|",
    ]

    inputs = spec.get("inputs", {})
    required = set(inputs.get("required", []))
    for name, prop in inputs.get("properties", {}).items():
        req = "Yes" if name in required else "No"
        lines.append(
            f"| `{name}` | `{prop['type']}` | {req} | {prop.get('description', '')} |"
        )

    lines.extend([
        "",
        "## Response",
        "",
        "### Success",
        "",
        "```json",
        json.dumps(spec["outputs"]["success"]["example"], indent=2),
        "```",
        "",
        "### Errors",
        "",
    ])

    for error in spec["outputs"].get("errors", []):
        lines.extend([
            f"#### {error['code']}",
            "",
            error.get("description", ""),
            "",
            "```json",
            json.dumps(error["example"], indent=2),
            "```",
            "",
        ])

    return "\n".join(lines)
```

### Auto-Generated Type Definitions

```python
def generate_typescript_types(spec: dict) -> str:
    """Generate TypeScript types from spec."""
    lines = [
        f"// Auto-generated from {spec['name']} spec v{spec['version']}",
        "",
    ]

    # Input type
    lines.append(f"export interface {pascal_case(spec['name'])}Input {{")
    inputs = spec.get("inputs", {})
    required = set(inputs.get("required", []))
    for name, prop in inputs.get("properties", {}).items():
        optional = "" if name in required else "?"
        ts_type = json_type_to_ts(prop["type"])
        lines.append(f"  {name}{optional}: {ts_type};")
    lines.append("}")
    lines.append("")

    # Output type
    lines.append(f"export interface {pascal_case(spec['name'])}Output {{")
    lines.append("  success: boolean;")
    lines.append("  data: Record<string, unknown>;")
    lines.append("  error: string | null;")
    lines.append("  meta: { version: string; [key: string]: unknown };")
    lines.append("}")

    return "\n".join(lines)
```

## Documenting Rationale

Include reasoning in specs for future maintainers:

```json
{
    "name": "batch_delete",
    "inputs": {
        "properties": {
            "item_ids": {
                "type": "array",
                "maxItems": 100,
                "_rationale": "Limited to 100 to prevent timeout. For larger batches, use pagination."
            }
        }
    },
    "outputs": {
        "_design_decisions": [
            {
                "decision": "Return partial success instead of failing entire batch",
                "rationale": "Users prefer knowing what succeeded vs all-or-nothing",
                "date": "2025-10-15",
                "alternatives_considered": ["Transactional all-or-nothing", "Queue-based async"]
            }
        ]
    }
}
```

## Keeping READMEs Synchronized

### Link to Specs (Source of Truth)

```markdown
<!-- README.md -->
# User Tools

See the authoritative specs:
- `./specs/active/<spec_id>.json`
- `./specs/completed/<spec_id>.json`

> **Note**: Specs are the source of truth. If you publish rendered docs, treat them as derived artifacts and regenerate them from specs in CI.
```

For runtime tool discovery, prefer the unified router manifest in `mcp/capabilities_manifest.json` (or call `server(action="tools")`).

### Include Spec Badges

```markdown
![Spec Version](https://img.shields.io/badge/spec-v1.2.0-blue)
![Last Updated](https://img.shields.io/badge/updated-2025--11--26-green)
```

## Anti-Patterns

### Don't: Document After Implementation

```
# Bad sequence:
1. Write code
2. Manually write docs (often forgotten)
3. Docs drift from code

# Good sequence:
1. Write spec
2. Implement to spec
3. Generate docs from spec
4. Update spec and code together
```

### Don't: Keep Specs Separate from Code Reviews

```
# Bad: Separate PRs
PR #101: Add new parameter (code only)
PR #102: Update spec (days later, maybe)

# Good: Same PR
PR #101: Add new parameter
  - src/tools/users.py
  - specs/users/get_user.json
  - tests/test_users.py
```

### Don't: Use Prose-Only Documentation

```json
// Bad: Just prose
{
    "description": "Takes a user ID and returns user data"
}

// Good: Structured with examples
{
    "description": "Retrieve user details by ID",
    "inputs": {
        "properties": {
            "user_id": {
                "type": "string",
                "pattern": "^usr_[a-zA-Z0-9]+$",
                "examples": ["usr_abc123"]
            }
        }
    }
}
```

## Related Documents

- [Versioned Contracts](./01-versioned-contracts.md) - Schema versioning
- [Testing & Fixtures](./10-testing-fixtures.md) - Testing specs
- [Tool Discovery](./13-tool-discovery.md) - Runtime spec access

---

**Navigation:** [← Security & Trust Boundaries](./08-security-trust-boundaries.md) | [Index](./README.md) | [Next: Testing & Fixtures →](./10-testing-fixtures.md)
