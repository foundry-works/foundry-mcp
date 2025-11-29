# Golden Fixtures

This directory contains golden fixture files that capture the expected JSON output
format for SDD CLI commands. These fixtures are used for regression testing and
to validate that output schemas remain stable across releases.

## Fixture Categories

- `success_*.json` - Successful command outputs
- `error_*.json` - Error response formats
- `validation_*.json` - Validation command outputs

## Usage

Golden fixtures are compared against actual CLI output in tests to detect
unintended changes to the output schema.

## Regenerating Fixtures

To regenerate fixtures after intentional schema changes:

```bash
pytest tests/unit/test_golden_fixtures.py --regenerate-fixtures
```
