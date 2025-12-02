# Golden Fixtures

This directory contains golden fixture files that capture the expected JSON output
format for SDD CLI commands and MCP tools. These fixtures are used for regression
testing and to validate that output schemas remain stable across releases.

## Fixture Categories

### Spec/CLI Fixtures
- `success_*.json` - Successful command outputs
- `error_*.json` - Error response formats
- `validation_*.json` - Validation command outputs

### Provider Tool Fixtures
- `provider_list_*.json` - Provider listing tool responses
- `provider_status_*.json` - Provider status tool responses
- `provider_execute_*.json` - Provider execution tool responses

## Provider Fixtures

| Fixture | Description |
|---------|-------------|
| `provider_list_success.json` | Successful list of available providers |
| `provider_list_with_unavailable.json` | List including unavailable providers |
| `provider_status_success.json` | Successful provider status with metadata |
| `provider_status_not_found.json` | Error when provider not found |
| `provider_status_missing_id.json` | Validation error for missing provider_id |
| `provider_execute_success.json` | Successful prompt execution |
| `provider_execute_missing_prompt.json` | Validation error for missing prompt |
| `provider_execute_unavailable.json` | Error when provider unavailable |
| `provider_execute_timeout.json` | Timeout error response |

## Usage

Golden fixtures are compared against actual CLI/MCP output in tests to detect
unintended changes to the output schema.

## Regenerating Fixtures

To regenerate fixtures after intentional schema changes:

```bash
pytest tests/unit/test_golden_fixtures.py --regenerate-fixtures
```
