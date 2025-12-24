# Bikelane Intake Plan

This document defines the fast-intake ("bikelane") capability for foundry-mcp, aligned with the docs/mcp_best_practices and codebase standards.

---

## 1) Summary

Goal: Add a lightweight intake queue that captures work ideas quickly without forcing a full spec up front. Intake items live in a separate JSONL store and are exposed via new authoring actions (`intake-add`, `intake-list`, `intake-dismiss`). The bikelane is not a spec; specs remain the source of truth for implementation.

Key constraints:
- Storage lives at `specs/.bikelane/intake.jsonl`
- `intake-list` returns only `status="new"` items (sorted FIFO)
- Use standard response envelopes (`response-v2`) via `success_response` and `error_response` helpers from `foundry_mcp.core.responses`
- Use cursor-based pagination with documented O(n) characteristics
- File locking for concurrent write safety
- Synchronous execution model (file I/O with blocking locks)

---

## 2) Revision History

| Issue | Original | Revised |
|-------|----------|---------|
| Scalability | No archival strategy | Add file rotation at 1000 items |
| Cursor perf | Undocumented O(n) | Document limitation, add line hints |
| Concurrency | Not addressed | Add fcntl file locking |
| Exit path | None | Add `intake-dismiss` action |
| Idempotency | Non-idempotent only | Add optional `idempotency_key` |
| Sort order | Unspecified | Explicit FIFO (oldest first) |
| dry_run | Not supported | Add `dry_run` parameter |
| Tag case | Mixed case allowed | Normalize to lowercase on write |
| total_count | Optional | Always include (already scanning) |

---

## 3) Goals

- Provide a fast, low-friction capture mechanism for incoming work
- Keep intake data separate from spec lifecycle state
- Offer a concise, reliable list of new items for triage
- Maintain strict input validation and predictable tool responses
- Ensure safe concurrent access and bounded file growth

---

## 4) Non-Goals (for this phase)

- No automatic promotion from intake to spec
- No triage or planning phases added to specs
- No cross-linking between intake and specs
- No background indexing, search, or tag analytics
- No full CRUD (update action deferred)

---

## 5) Storage

### Location
- Directory: `specs/.bikelane/`
- Active file: `specs/.bikelane/intake.jsonl`
- Archive pattern: `specs/.bikelane/intake.YYYY-MM.jsonl`

### Format
- JSON Lines (one JSON object per line)
- Append-only writes for `intake-add`
- In-place status update for `intake-dismiss` (single field change)
- Lock file: `specs/.bikelane/.intake.lock`

### File Rotation
When `intake.jsonl` exceeds **1000 items** OR **1MB**:
1. Rename to `intake.YYYY-MM.jsonl` (based on oldest item's created_at)
2. Create fresh `intake.jsonl`
3. Archive files are read-only (not scanned by `intake-list`)

Rationale:
- Bounds scan time for `intake-list`
- Preserves history without impacting performance
- Archives can be searched separately if needed later

---

## 6) Intake Item Schema

```json
{
  "schema_version": "intake-v1",
  "id": "intake-<uuid>",
  "title": "...",
  "description": "...",
  "status": "new",
  "priority": "p2",
  "tags": ["tag-a", "tag-b"],
  "source": "user-note",
  "requester": "tyler",
  "idempotency_key": "optional-client-key",
  "created_at": "2025-01-30T12:34:56Z",
  "updated_at": "2025-01-30T12:34:56Z"
}
```

Field rules:
- `schema_version`: Required. Fixed value `intake-v1`
- `id`: Required. Format `intake-<uuid4>`
- `title`: Required. 1-140 chars
- `description`: Optional. Max 2000 chars
- `status`: Required. Enum: `new`, `dismissed`
- `priority`: Optional. Enum `p0|p1|p2|p3|p4`. Default `p2`
- `tags`: Optional. 0-20 items; each 1-32 chars; pattern `^[a-z0-9_-]+$` (lowercase normalized)
- `source`: Optional. Max 100 chars
- `requester`: Optional. Max 100 chars
- `idempotency_key`: Optional. Max 64 chars. Used for deduplication
- `created_at`/`updated_at`: Required. ISO 8601 UTC with `Z` suffix

---

## 7) New Tool Actions

Router: `authoring`

### 7.1 authoring(action="intake-add")

Purpose: Capture a new intake item and append to `intake.jsonl`

**Inputs:**
| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `title` | string | Yes | - | 1-140 chars |
| `description` | string | No | null | max 2000 chars |
| `priority` | string | No | "p2" | enum p0-p4 |
| `tags` | array | No | [] | max 20, each 1-32 chars |
| `source` | string | No | null | max 100 chars |
| `requester` | string | No | null | max 100 chars |
| `idempotency_key` | string | No | null | max 64 chars |
| `path` | string | No | null | workspace override |
| `dry_run` | boolean | No | false | preview without writing |

**Behavior:**
1. Validate all inputs per validation rules
2. Acquire exclusive file lock (timeout 5s)
3. If `idempotency_key` provided, scan last 100 items for duplicate
4. Normalize tags to lowercase
5. Check file rotation threshold; rotate if needed
6. Append JSONL record with `status="new"`
7. Release lock
8. Return created item

**Response `data`:**
```json
{
  "item": { ... },
  "intake_path": "/abs/path/to/intake.jsonl",
  "was_duplicate": false
}
```

**Full response envelope:**
```json
{
  "success": true,
  "data": {
    "item": { ... },
    "intake_path": "/abs/path/to/intake.jsonl",
    "was_duplicate": false
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_abc123",
    "telemetry": { "duration_ms": 45, "lock_wait_ms": 2 }
  }
}
```

**Idempotency:**
- Without `idempotency_key`: Non-idempotent (each call creates new item)
- With `idempotency_key`: If key matches within last 100 items, returns **success** with the existing item and `was_duplicate: true` (this is NOT an error response)

**dry_run behavior:**
- Validates inputs, returns what would be written
- Sets `meta.dry_run: true`
- Does not write to file

---

### 7.2 authoring(action="intake-list")

Purpose: List `status="new"` intake items for triage

**Idempotency:** Naturally idempotent. Safe to retry any number of times.

**Inputs:**
| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `cursor` | string | No | null | opaque pagination cursor |
| `limit` | integer | No | 50 | 1-200 |
| `path` | string | No | null | workspace override |

**Behavior:**
1. Validate inputs
2. Acquire shared file lock
3. Scan `intake.jsonl` line-by-line
4. Filter items where `status="new"`
5. Apply cursor (skip to line hint, verify id match)
6. Return up to `limit` items in **FIFO order** (oldest first)
7. Release lock

**Response `data`:**
```json
{
  "items": [ ... ],
  "intake_path": "/abs/path/to/intake.jsonl",
  "total_count": 42
}
```

**Full response envelope:**
```json
{
  "success": true,
  "data": {
    "items": [ ... ],
    "intake_path": "/abs/path/to/intake.jsonl",
    "total_count": 42
  },
  "error": null,
  "meta": {
    "version": "response-v2",
    "request_id": "req_def456",
    "pagination": { "cursor": "...", "has_more": true, "page_size": 50 },
    "telemetry": { "duration_ms": 120, "lock_wait_ms": 0 }
  }
}
```

**Pagination `meta.pagination`:**
```json
{
  "cursor": "eyJsYXN0X2lkIjogIi4uLiIsICJsaW5lX2hpbnQiOiA0Mn0=",
  "has_more": true,
  "page_size": 50
}
```

**Cursor format (base64 JSON):**
```json
{
  "last_id": "intake-...",
  "line_hint": 42,
  "version": 1
}
```

The `line_hint` allows O(1) seek to approximate position. If the id at that line doesn't match `last_id`, fall back to full scan from beginning.

**Performance note:** Document that `intake-list` is O(n) where n = total items in file. File rotation bounds n to ~1000.

---

### 7.3 authoring(action="intake-dismiss")

Purpose: Mark an intake item as dismissed (removes from triage list)

**Inputs:**
| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `intake_id` | string | Yes | - | format `intake-<uuid>` |
| `reason` | string | No | null | max 200 chars |
| `path` | string | No | null | workspace override |
| `dry_run` | boolean | No | false | preview without writing |

**Behavior:**
1. Validate inputs
2. Acquire exclusive file lock
3. Scan file for matching `intake_id`
4. If not found, return `NOT_FOUND` error
5. Rewrite file with item's `status` changed to `dismissed` and `updated_at` refreshed
6. Release lock
7. Return updated item

**Response `data`:**
```json
{
  "item": { ... },
  "intake_path": "/abs/path/to/intake.jsonl"
}
```

**Implementation note:** This is the only action that rewrites the file. Use atomic write-rename pattern:
1. Write to `intake.jsonl.tmp`
2. Rename to `intake.jsonl`

---

## 8) Concurrency & Locking

**Execution Model:** Synchronous. All intake operations use blocking file I/O with advisory locks. This is appropriate for the file-based storage backend.

### Lock Strategy
- Lock file: `specs/.bikelane/.intake.lock`
- Use `fcntl.flock()` for advisory locking
- Shared lock for reads (`intake-list`)
- Exclusive lock for writes (`intake-add`, `intake-dismiss`)

### Timeouts
- Lock acquisition timeout: 5 seconds
- On timeout, return `RESOURCE_BUSY` error with remediation "Retry after a moment"

### Atomic Writes
For `intake-dismiss` (file rewrite):
1. Write complete content to `intake.jsonl.tmp`
2. `os.rename()` atomically replaces `intake.jsonl`
3. Guarantees no partial writes on crash

---

## 9) Validation

Order: required fields, type checks, format checks, business rules

| Field | Validation |
|-------|------------|
| `title` | required, string, 1-140 chars |
| `description` | optional, string, max 2000 chars |
| `priority` | optional, enum p0-p4, default p2 |
| `tags` | optional, array, max 20 items, each 1-32 chars, pattern `^[a-z0-9_-]+$` |
| `source` | optional, string, max 100 chars |
| `requester` | optional, string, max 100 chars |
| `idempotency_key` | optional, string, max 64 chars |
| `intake_id` | required (dismiss), pattern `^intake-[a-f0-9-]{36}$` |
| `reason` | optional, string, max 200 chars |
| `limit` | optional, integer, 1-200, default 50 |
| `cursor` | optional, string, must decode to valid cursor payload |

**Tag normalization:** Tags are lowercased on write. Input `["Bug", "HIGH-PRIORITY"]` becomes `["bug", "high-priority"]`.

---

## 10) Error Semantics

All errors use `error_response` helper with standard codes:

| Code | When |
|------|------|
| `MISSING_REQUIRED` | Missing `title` or `intake_id` |
| `INVALID_FORMAT` | Invalid cursor, bad tag pattern, malformed id |
| `VALUE_OUT_OF_RANGE` | Length/limit violations |
| `NOT_FOUND` | Specs directory not found, or intake_id not found |
| `RESOURCE_BUSY` | Lock acquisition timeout |
| `DUPLICATE_ENTRY` | N/A - idempotency key match returns **success** with `was_duplicate: true`, not an error |
| `INTERNAL_ERROR` | IO failures, corruption |

Error responses include:
- `error_code`, `error_type`, `remediation`, `details.field`

---

## 11) Security & Trust Boundaries

- Treat all inputs as untrusted (LLM-generated)
- Resolve specs directory via existing helpers to prevent path traversal
- Validate `path` parameter against allowed workspace roots
- Avoid logging full description text (truncate to 100 chars in logs, use `redact_sensitive()` helper)
- Strip control characters from string inputs (title, description, source, requester, reason)
- Sanitize `title` and `description` for potential prompt injection patterns before storage
- Lock file prevents race conditions that could corrupt data

---

## 12) Observability

**Telemetry (in response `meta.telemetry`):**
- `duration_ms`: Total operation time
- `lock_wait_ms`: Time spent waiting for file lock

**Request ID:**
- Propagate `request_id` through all operations
- Include in response `meta.request_id` and all log entries

**Structured Logging (JSON format):**
- `INFO`: Operation start, operation complete (success)
- `WARN`: Lock wait > 1s, cursor fallback to full scan
- `ERROR`: Lock timeout, file corruption, IO failures

**Audit log entries:**
- action, intake_id (if applicable), timestamp, request_id
- No full descriptions in logs (truncate to 100 chars)

---

## 13) Feature Flags

- Flag name: `intake_tools`
- Initial state: `experimental`
- Gate all three actions behind this flag
- Manifest shows actions only when flag enabled

---

## 14) Discovery / Manifest Updates

Update `mcp/capabilities_manifest.json`:

```json
{
  "authoring": {
    "actions": [..., "intake-add", "intake-list", "intake-dismiss"],
    "parameters": {
      "title": { "type": "string", "max_length": 140 },
      "description": { "type": "string", "max_length": 2000 },
      "priority": { "type": "string", "enum": ["p0","p1","p2","p3","p4"] },
      "tags": { "type": "array", "max_items": 20 },
      "source": { "type": "string", "max_length": 100 },
      "requester": { "type": "string", "max_length": 100 },
      "idempotency_key": { "type": "string", "max_length": 64 },
      "intake_id": { "type": "string", "pattern": "^intake-[a-f0-9-]{36}$" },
      "reason": { "type": "string", "max_length": 200 }
    },
    "examples": [
      {
        "action": "intake-add",
        "title": "Add dark mode support",
        "priority": "p1",
        "tags": ["ui", "feature"]
      },
      {
        "action": "intake-list",
        "limit": 10
      },
      {
        "action": "intake-dismiss",
        "intake_id": "intake-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "reason": "Converted to spec"
      }
    ]
  }
}
```

---

## 15) Implementation Files

| File | Changes |
|------|---------|
| `src/foundry_mcp/tools/unified/authoring.py` | Add intake-add, intake-list, intake-dismiss handlers |
| `src/foundry_mcp/core/intake.py` | **NEW** - Core logic for intake operations |
| `src/foundry_mcp/schemas/intake-schema.json` | **NEW** - JSON Schema for intake items |
| `tests/unit/test_core/test_intake.py` | **NEW** - Unit tests |
| `tests/fixtures/intake/` | **NEW** - Test fixtures |
| `mcp/capabilities_manifest.json` | Add actions and parameters |
| `docs/guides/intake.md` | **NEW** - User documentation |

---

## 16) Specs, Docs, and Tests

**Spec:**
- Create `specs/pending/bikelane-intake.json` defining all three actions

**Docs:**
- Create `docs/guides/intake.md` explaining bikelane concept
- Update authoring action reference

**Tests:**
- `test_intake_add_basic` - happy path
- `test_intake_add_validation` - all validation rules
- `test_intake_add_idempotency` - duplicate key handling
- `test_intake_add_dry_run` - preview mode
- `test_intake_list_empty` - empty file
- `test_intake_list_pagination` - cursor navigation
- `test_intake_list_filters_dismissed` - status filtering
- `test_intake_dismiss_basic` - happy path
- `test_intake_dismiss_not_found` - missing id
- `test_intake_concurrency` - lock contention
- `test_intake_rotation` - file rotation threshold

---

## 17) Open Questions (Deferred)

- Promotion workflow from intake to spec
- `intake-update` for editing items
- Tag search/filtering
- Archive search capability
- Spec linkage or provenance tracking

---

## 18) Acceptance Criteria

1. `authoring(intake-add)` writes valid JSONL to `specs/.bikelane/intake.jsonl`
2. `authoring(intake-list)` returns only `status="new"` items in FIFO order
3. `authoring(intake-dismiss)` changes item status to `dismissed`
4. All responses use response-v2 envelope
5. Concurrent access is safe (file locking works)
6. File rotation triggers at 1000 items or 1MB
7. Idempotency key prevents duplicates within last 100 items
8. `dry_run` works for add and dismiss
9. Tags are normalized to lowercase
10. All validation rules enforced with clear error messages
11. Feature flag gates all actions
12. Tests pass with >90% coverage on new code
