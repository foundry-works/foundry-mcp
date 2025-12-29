# Intake Guide - Bikelane Fast-Capture System

A guide to using the bikelane intake system for rapid idea capture and triage.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Actions](#actions)
  - [intake-add](#intake-add)
  - [intake-list](#intake-list)
  - [intake-dismiss](#intake-dismiss)
- [Pagination](#pagination)
- [Storage and File Rotation](#storage-and-file-rotation)
- [Validation Constraints](#validation-constraints)
- [Common Workflows](#common-workflows)
- [Feature Flag](#feature-flag)

---

## Overview

The **bikelane** is a fast-capture queue for ideas, tasks, and requests that need quick recording without immediate triage. Think of it as a "bike lane" - a fast path for capturing items that can be processed later.

### Key Concepts

- **Intake Items**: Quick-capture records with title, description, priority, and tags
- **FIFO Queue**: Items are processed in first-in, first-out order
- **Two States**: Items are either `new` (active) or `dismissed` (processed/rejected)
- **Append-Only Storage**: Items are stored in a JSONL file with atomic writes

### Storage Location

Intake items are stored in:
```
specs/.bikelane/
    intake.jsonl       - Active intake items
    .intake.lock       - Lock file for synchronization
    intake.YYYY-MM.jsonl - Archived files (after rotation)
```

---

## Quick Start

### Add an Item

```json
{
  "action": "intake-add",
  "title": "Add dark mode support",
  "description": "Users have requested a dark theme option",
  "priority": "p2",
  "tags": ["feature", "ui"]
}
```

### List New Items

```json
{
  "action": "intake-list",
  "limit": 10
}
```

### Dismiss an Item

```json
{
  "action": "intake-dismiss",
  "intake_id": "intake-12345678-1234-1234-1234-123456789012",
  "reason": "Duplicate of existing issue"
}
```

---

## Actions

### intake-add

Add a new item to the intake queue.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `title` | string | Yes | - | Brief title (1-140 characters) |
| `description` | string | No | null | Detailed description (max 2000 characters) |
| `priority` | string | No | "p2" | Priority level: p0 (highest) to p4 (lowest) |
| `tags` | array | No | [] | List of tags (max 20, each 1-32 chars, lowercase) |
| `source` | string | No | null | Origin of the item (max 100 characters) |
| `requester` | string | No | null | Who requested the work (max 100 characters) |
| `idempotency_key` | string | No | null | Key for deduplication (max 64 characters) |
| `dry_run` | boolean | No | false | Validate without persisting |

#### Example Request

```json
{
  "action": "intake-add",
  "title": "Implement JWT authentication",
  "description": "Replace session-based auth with JWT tokens for the API",
  "priority": "p1",
  "tags": ["security", "api", "auth"],
  "source": "security-review",
  "requester": "security-team@example.com",
  "idempotency_key": "auth-migration-2024"
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "item": {
      "schema_version": "intake-v1",
      "id": "intake-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "title": "Implement JWT authentication",
      "description": "Replace session-based auth with JWT tokens for the API",
      "status": "new",
      "priority": "p1",
      "tags": ["security", "api", "auth"],
      "source": "security-review",
      "requester": "security-team@example.com",
      "idempotency_key": "auth-migration-2024",
      "created_at": "2024-01-15T10:30:00.000Z",
      "updated_at": "2024-01-15T10:30:00.000Z"
    },
    "was_duplicate": false,
    "intake_path": "/workspace/specs/.bikelane/intake.jsonl"
  }
}
```

#### Idempotency

When an `idempotency_key` is provided:
- If a matching key exists in the last 100 items, the existing item is returned
- `was_duplicate: true` indicates the item already existed
- Use idempotency keys to prevent duplicate submissions from retries

---

### intake-list

List intake items with status `new` in FIFO order (oldest first).

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `cursor` | string | No | null | Pagination cursor from previous response |
| `limit` | integer | No | 50 | Items per page (1-200) |

#### Example Request

```json
{
  "action": "intake-list",
  "limit": 10
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "intake-11111111-1111-1111-1111-111111111111",
        "title": "First item",
        "status": "new",
        "priority": "p2",
        "created_at": "2024-01-15T10:00:00.000Z"
      },
      {
        "id": "intake-22222222-2222-2222-2222-222222222222",
        "title": "Second item",
        "status": "new",
        "priority": "p1",
        "created_at": "2024-01-15T11:00:00.000Z"
      }
    ],
    "total_count": 25,
    "has_more": true,
    "next_cursor": "eyJsYXN0X2lkIjoiaW50YWtlLTIyMjIyMjIy..."
  }
}
```

#### Notes

- Only items with `status: "new"` are returned
- Items are ordered by creation time (FIFO)
- Use the `next_cursor` value to fetch the next page

---

### intake-dismiss

Dismiss an intake item by changing its status to `dismissed`.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `intake_id` | string | Yes | - | The intake item ID to dismiss |
| `reason` | string | No | null | Reason for dismissal (max 200 characters) |
| `dry_run` | boolean | No | false | Find item but don't modify |

#### Example Request

```json
{
  "action": "intake-dismiss",
  "intake_id": "intake-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "reason": "Converted to spec: feature-auth-2024-001"
}
```

#### Example Response

```json
{
  "success": true,
  "data": {
    "item": {
      "id": "intake-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "title": "Implement JWT authentication",
      "status": "dismissed",
      "updated_at": "2024-01-16T09:00:00.000Z"
    }
  }
}
```

#### Notes

- Dismissed items remain in the file but are excluded from `intake-list`
- Use the `reason` field to document why the item was dismissed
- Common reasons: "Converted to spec", "Duplicate", "Won't fix", "Out of scope"

---

## Pagination

The intake list uses cursor-based pagination for efficient traversal of large datasets.

### How It Works

1. **Initial Request**: Call `intake-list` without a cursor
2. **Check for More**: If `has_more: true`, save the `next_cursor` value
3. **Continue**: Call `intake-list` with `cursor: <next_cursor>` to get the next page
4. **Complete**: When `has_more: false`, you've reached the end

### Example: Paginating Through All Items

```python
cursor = None
all_items = []

while True:
    response = call_intake_list(cursor=cursor, limit=50)
    all_items.extend(response["data"]["items"])

    if not response["data"]["has_more"]:
        break
    cursor = response["data"]["next_cursor"]

print(f"Total items: {len(all_items)}")
```

### Cursor Internals

Cursors are base64-encoded JSON containing:
- `last_id`: ID of the last item on the previous page
- `line_hint`: File line position hint for efficient seeking
- `version`: Cursor format version for forward compatibility

Cursors may become stale if the file is rotated. The system handles this gracefully by falling back to a full scan.

---

## Storage and File Rotation

### JSONL Format

Items are stored as JSON Lines (one JSON object per line):

```jsonl
{"schema_version":"intake-v1","id":"intake-...","title":"First item",...}
{"schema_version":"intake-v1","id":"intake-...","title":"Second item",...}
```

### File Rotation

To prevent unbounded file growth, the intake file is rotated when:
- Item count exceeds **1000 items**, OR
- File size exceeds **1 MB**

When rotation occurs:
1. The current file is renamed to `intake.YYYY-MM.jsonl` (based on oldest item date)
2. A new empty `intake.jsonl` is created
3. If an archive with the same name exists, a numeric suffix is added

### Concurrency Safety

- File operations use `fcntl` locking for cross-process safety
- Thread-level locking ensures thread safety within a process
- Lock acquisition times out after 5 seconds to prevent deadlocks

---

## Validation Constraints

### Field Constraints

| Field | Constraint |
|-------|------------|
| `title` | Required, 1-140 characters |
| `description` | Optional, max 2000 characters |
| `priority` | One of: p0, p1, p2, p3, p4 |
| `tags` | Max 20 tags, each 1-32 chars, pattern: `[a-z0-9_-]+` |
| `source` | Optional, max 100 characters |
| `requester` | Optional, max 100 characters |
| `idempotency_key` | Optional, max 64 characters |
| `reason` (dismiss) | Optional, max 200 characters |

### Input Sanitization

All string inputs are sanitized:
- Control characters (0x00-0x1F) are stripped, except `\n`, `\r`, `\t`
- Prompt injection patterns are detected and sanitized
- Tags are normalized to lowercase

### ID Format

Intake IDs follow the pattern: `intake-<uuid4>`

Example: `intake-a1b2c3d4-e5f6-7890-abcd-ef1234567890`

---

## Common Workflows

### Workflow 1: Quick Capture During Meeting

```json
// Rapidly capture action items
{"action": "intake-add", "title": "Review API rate limits", "priority": "p2"}
{"action": "intake-add", "title": "Update onboarding docs", "priority": "p3"}
{"action": "intake-add", "title": "Fix login timeout bug", "priority": "p1"}
```

### Workflow 2: Triage Session

```python
# 1. List all new items
items = intake_list(limit=100)

# 2. For each item, decide:
for item in items:
    if should_create_spec(item):
        # Create a spec from the item
        create_spec_from_intake(item)
        intake_dismiss(item["id"], reason="Converted to spec")
    elif is_duplicate(item):
        intake_dismiss(item["id"], reason="Duplicate of existing issue")
    elif out_of_scope(item):
        intake_dismiss(item["id"], reason="Out of scope for current cycle")
    # else: leave as new for further discussion
```

### Workflow 3: Deduplication with Idempotency Keys

```json
// First submission
{
  "action": "intake-add",
  "title": "Add search feature",
  "idempotency_key": "search-feature-2024"
}
// Response: was_duplicate: false

// Retry (network hiccup, etc.)
{
  "action": "intake-add",
  "title": "Add search feature",
  "idempotency_key": "search-feature-2024"
}
// Response: was_duplicate: true, returns original item
```

---

## Feature Flag

Intake tools are gated behind the `intake_tools` feature flag.

### Enabling the Feature

The flag must be enabled before using intake actions. If disabled, you'll receive:

```json
{
  "success": false,
  "error": {
    "code": "FEATURE_DISABLED",
    "message": "Feature 'intake_tools' is not enabled",
    "remediation": "Enable the 'intake_tools' feature flag to use intake actions."
  }
}
```

### Configuration

Enable the feature flag via the feature flag system:
- State: `EXPERIMENTAL` (opt-in, default disabled)
- Enable globally or per-workspace as needed

---

## Best Practices

1. **Keep titles concise**: Use the description for details
2. **Use consistent tags**: Establish a tag taxonomy for your team
3. **Set meaningful priorities**: Reserve p0 for true emergencies
4. **Use idempotency keys**: Prevent duplicates from retries
5. **Triage regularly**: Don't let the intake queue grow too large
6. **Document dismissal reasons**: Future reference for why items were dropped
