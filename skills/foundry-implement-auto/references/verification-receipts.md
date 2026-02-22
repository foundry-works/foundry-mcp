# Verification Receipts

Under strict gate policy, the orchestrator validates every field of the verification receipt. Malformed receipts are rejected.

## Contents

- [Receipt Fields](#receipt-fields)
- [Constructing Receipts](#constructing-receipts)
- [Error Codes](#error-codes)

## Receipt Fields

| Field | Type | Requirements |
|-------|------|--------------|
| `command_hash` | string | SHA-256 hex digest of the verification command string. Lowercase, exactly 64 characters. |
| `exit_code` | integer | Exit code from the verification process. |
| `output_digest` | string | SHA-256 hex digest of the verification output. Lowercase, exactly 64 characters. |
| `issued_at` | string | UTC timestamp with timezone info (ISO 8601). Naive datetimes are rejected. |
| `step_id` | string | Must match the current step's `step_id`. Mismatched binding is rejected. |

## Constructing Receipts

Use the canonical helper — do not hand-construct receipt objects:

```python
from foundry_mcp.core.autonomy.models import issue_verification_receipt

receipt = issue_verification_receipt(
    step_id=step_id,
    command=command_string,
    exit_code=exit_code,
    output=output_string
)
```

The helper enforces:
- Hash format (lowercase hex, 64 chars)
- Timezone-aware timestamps
- Step ID binding

Include the receipt in the step report:

```json
{
  "last_step_result": {
    "verification_receipt": { ... },
    "step_proof": "...",
    "outcome": "pass"
  }
}
```

## Error Codes

| Error | Cause | Signal |
|-------|-------|--------|
| `ERROR_VERIFICATION_RECEIPT_MISSING` | Receipt not included in report | `blocked_runtime` |
| `ERROR_VERIFICATION_RECEIPT_INVALID` | Receipt malformed or field validation failed | `blocked_runtime` |
