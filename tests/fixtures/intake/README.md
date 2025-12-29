# Intake Test Fixtures

Test fixtures for the intake module (`src/foundry_mcp/core/intake.py`).

## Files

### sample_items.jsonl
Contains valid intake items for basic testing:
- Various priority levels (p0-p4)
- Items with and without optional fields
- One dismissed item for status filtering tests
- Items with different sources and requesters

### edge_cases.jsonl
Contains items that test boundary conditions:
- Maximum title length (140 chars)
- Maximum description length (~2000 chars)
- Maximum number of tags (20)
- Maximum tag length (32 chars)
- Maximum source/requester length (100 chars)
- Maximum idempotency key length (64 chars)
- All priority levels (p0-p4)
- Unicode characters in title and description
- Newlines and tabs in description

### malformed.jsonl
Contains invalid items for error handling tests:
- Invalid ID format (not matching UUID pattern)
- Missing required field (title)
- Missing timestamps
- Invalid status value
- Invalid priority value
- Wrong type for tags (string instead of array)
- Wrong type for description (number instead of string)
- Wrong type for priority (number instead of string)
- Wrong schema version
- Invalid JSON syntax
- Incomplete JSON
- Empty title

### empty.jsonl
An empty file for testing empty store scenarios.

## Usage

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_items_path():
    return Path(__file__).parent / "intake" / "sample_items.jsonl"

def test_load_sample_items(sample_items_path):
    with open(sample_items_path) as f:
        items = [json.loads(line) for line in f]
    assert len(items) == 6
```
