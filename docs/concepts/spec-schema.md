# Spec Schema Documentation

This document describes the JSON structure of SDD specification files.

## Overview

A specification file is a JSON document that defines:
- **Frontmatter**: Metadata about the spec (title, version, status)
- **Phases**: Major work stages containing tasks
- **Tasks**: Individual work items with status tracking
- **Hierarchy**: Parent-child relationships and dependencies

## File Location

Specs are stored in the `specs/` directory organized by lifecycle status:

```
specs/
├── pending/           # Not yet started
├── active/            # Currently in progress
├── completed/         # Finished successfully (all tasks done)
└── archived/          # Abandoned, superseded, or deprioritized
```

### Folder Semantics

| Folder | Status | Meaning |
|--------|--------|---------|
| `pending/` | `pending` | Spec created but work not started |
| `active/` | `in_progress` | Work is actively being done |
| `completed/` | `completed` | All work finished successfully |
| `archived/` | `archived` | Decided not to pursue (abandoned, superseded, deprioritized) |

**Key distinction**: Use `completed/` when all tasks are done. Use `archived/` when you decide not to finish the spec, regardless of progress.

## Spec ID Format

Spec IDs follow the pattern: `{name}-{date}-{sequence}`

Example: `user-authentication-2025-01-15-001`

---

## Top-Level Structure

```json
{
  "id": "my-feature-2025-01-15-001",
  "title": "My Feature Implementation",
  "description": "Brief description of the feature",
  "version": "1.0.0",
  "status": "active",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T14:22:00Z",
  "author": "developer@example.com",
  "category": "implementation",
  "mission": "Optional mission statement",
  "tags": ["feature", "api"],
  "hierarchy": { ... },
  "assumptions": [ ... ],
  "revisions": [ ... ],
  "metadata": { ... }
}
```

### Frontmatter Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier |
| `title` | string | Yes | Human-readable title |
| `description` | string | No | Brief description |
| `version` | string | No | Semantic version |
| `status` | string | Yes | Current status |
| `created_at` | datetime | Yes | Creation timestamp (ISO 8601) |
| `updated_at` | datetime | Yes | Last update timestamp |
| `author` | string | No | Creator identifier |
| `category` | string | No | Default task category |
| `mission` | string | No | Mission statement |
| `tags` | array | No | Categorization tags |

### Status Values

| Status | Description |
|--------|-------------|
| `draft` | Initial creation, not validated |
| `pending` | Ready to start, not yet active |
| `active` | Currently in progress |
| `completed` | All tasks finished |
| `archived` | Historical, no longer active |

---

## Hierarchy Structure

The `hierarchy` object contains all phases, tasks, and their relationships.

### Node Types

| Type | Description |
|------|-------------|
| `spec-root` | Root node (auto-created) |
| `phase` | Major work stage |
| `task` | Work item |
| `subtask` | Child of a task |
| `verify` | Verification step |

### Phase Node

```json
{
  "phase-1": {
    "type": "phase",
    "title": "Phase 1: Setup",
    "description": "Initial setup and configuration",
    "status": "pending",
    "parent": "spec-root",
    "position": 1,
    "children": ["task-1-1", "task-1-2"],
    "metadata": {
      "estimated_hours": 4,
      "priority": "high"
    }
  }
}
```

### Task Node

```json
{
  "task-1-1": {
    "type": "task",
    "title": "Create database schema",
    "description": "Design and implement the database schema",
    "status": "pending",
    "parent": "phase-1",
    "position": 1,
    "children": [],
    "category": "implementation",
    "acceptance_criteria": [
      "Schema supports all required entities",
      "Migrations are reversible"
    ],
    "dependencies": {
      "blocked_by": [],
      "blocks": ["task-1-2"]
    },
    "verification": {
      "type": "test",
      "command": "pytest tests/test_schema.py"
    },
    "metadata": {
      "estimated_hours": 2,
      "complexity": "medium",
      "owners": ["backend-team"]
    }
  }
}
```

### Node Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Node type |
| `title` | string | Yes | Display title |
| `description` | string | No | Detailed description |
| `status` | string | Yes | Current status |
| `parent` | string | Yes | Parent node ID |
| `position` | integer | Yes | Order within parent (1-based) |
| `children` | array | Yes | Child node IDs |

### Task-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `category` | string | Task category |
| `acceptance_criteria` | array | Completion requirements |
| `dependencies` | object | Dependency relationships |
| `verification` | object | Verification definition |
| `metadata` | object | Custom metadata |

---

## Task Status

| Status | Description |
|--------|-------------|
| `pending` | Not started |
| `in_progress` | Currently being worked on |
| `completed` | Finished successfully |
| `blocked` | Cannot proceed (has blocker) |
| `archived` | Abandoned or deprioritized (spec-root only) |

### Valid Transitions

```
pending → in_progress → completed
pending → blocked → pending → in_progress
in_progress → blocked → in_progress
```

---

## Dependencies

Dependencies define relationships between tasks.

```json
{
  "dependencies": {
    "blocked_by": ["task-1-1"],
    "blocks": ["task-1-3", "task-2-1"]
  }
}
```

| Field | Description |
|-------|-------------|
| `blocked_by` | Tasks that must complete first |
| `blocks` | Tasks waiting for this task |

### Dependency Rules

1. A task with `blocked_by` items cannot start until all are completed
2. Circular dependencies are invalid
3. The system validates dependency integrity on save

---

## Verification

Verification defines how to confirm task completion.

```json
{
  "verification": {
    "type": "test",
    "command": "pytest tests/test_feature.py",
    "expected_output": "All tests passed"
  }
}
```

### Verification Types

| Type | Description |
|------|-------------|
| `test` | Run automated tests |
| `manual` | Manual verification required |
| `review` | Code review required |
| `build` | Build must succeed |
| `deploy` | Deployment verification |

---

## Categories

Task categories help classify work:

| Category | Description |
|----------|-------------|
| `investigation` | Research and analysis |
| `implementation` | Code implementation |
| `refactoring` | Code improvement |
| `decision` | Decision point |
| `research` | External research |
| `documentation` | Documentation work |
| `testing` | Test creation |

---

## Assumptions and Revisions

### Assumptions

```json
{
  "assumptions": [
    {
      "id": "assumption-1",
      "text": "Database supports JSON columns",
      "type": "technical",
      "created_at": "2025-01-15T10:30:00Z"
    }
  ]
}
```

### Revisions

```json
{
  "revisions": [
    {
      "version": "1.1.0",
      "date": "2025-01-16T09:00:00Z",
      "author": "developer@example.com",
      "changes": "Added phase 3 for deployment"
    }
  ]
}
```

---

## Metadata

Custom metadata can be added at any level:

```json
{
  "metadata": {
    "estimated_hours": 2.5,
    "complexity": "high",
    "priority": "p1",
    "owners": ["alice", "bob"],
    "labels": {
      "team": "platform",
      "quarter": "Q1"
    }
  }
}
```

Common metadata fields:

| Field | Type | Description |
|-------|------|-------------|
| `estimated_hours` | number | Time estimate |
| `actual_hours` | number | Time spent |
| `complexity` | string | `low`, `medium`, `high` |
| `priority` | string | `p0`-`p4` |
| `owners` | array | Responsible parties |
| `labels` | object | Custom labels |

---

## Example: Complete Spec

```json
{
  "id": "user-auth-2025-01-15-001",
  "title": "User Authentication System",
  "description": "Implement JWT-based authentication",
  "version": "1.0.0",
  "status": "active",
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T14:30:00Z",
  "author": "alice@example.com",
  "category": "implementation",
  "tags": ["auth", "security", "api"],
  "hierarchy": {
    "spec-root": {
      "type": "spec-root",
      "children": ["phase-1", "phase-2"]
    },
    "phase-1": {
      "type": "phase",
      "title": "Phase 1: Core Auth",
      "status": "in_progress",
      "parent": "spec-root",
      "position": 1,
      "children": ["task-1-1", "task-1-2"]
    },
    "task-1-1": {
      "type": "task",
      "title": "Implement JWT token generation",
      "status": "completed",
      "parent": "phase-1",
      "position": 1,
      "children": [],
      "category": "implementation",
      "dependencies": {
        "blocked_by": [],
        "blocks": ["task-1-2"]
      }
    },
    "task-1-2": {
      "type": "task",
      "title": "Add token validation middleware",
      "status": "in_progress",
      "parent": "phase-1",
      "position": 2,
      "children": [],
      "category": "implementation",
      "dependencies": {
        "blocked_by": ["task-1-1"],
        "blocks": []
      }
    },
    "phase-2": {
      "type": "phase",
      "title": "Phase 2: User Management",
      "status": "pending",
      "parent": "spec-root",
      "position": 2,
      "children": ["task-2-1"]
    },
    "task-2-1": {
      "type": "task",
      "title": "Create user registration endpoint",
      "status": "pending",
      "parent": "phase-2",
      "position": 1,
      "children": [],
      "category": "implementation"
    }
  },
  "assumptions": [
    {
      "id": "assumption-1",
      "text": "Using RS256 for JWT signing",
      "type": "technical"
    }
  ]
}
```

---

## Validation

Use the `validate` command to check spec validity:

```bash
foundry-cli validate check my-spec
foundry-cli validate fix my-spec --dry-run
```

Common validation issues:
- Missing required fields
- Invalid status values
- Circular dependencies
- Orphaned nodes
- Invalid parent references

---

## Related

- [CLI Command Reference](../04-cli-command-reference.md) - Spec commands
- [MCP Tool Reference](../05-mcp-tool-reference.md) - spec/authoring tools
- [Workflow Guide](../03-workflow-guide.md) - Using specs in practice
