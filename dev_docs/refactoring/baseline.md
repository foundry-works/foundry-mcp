# Refactoring Baseline Metrics

Captured: 2026-02-06
Spec: unified-refactoring-2026-02-06-001

---

## 1. Module Line Counts

### Unified Tool Routers (`src/foundry_mcp/tools/unified/`)

| File | Lines |
|------|------:|
| `__init__.py` | 88 |
| `authoring.py` | 3,645 |
| `context_helpers.py` | 97 |
| `documentation_helpers.py` | 268 |
| `environment.py` | 1,409 |
| `error.py` | 491 |
| `health.py` | 237 |
| `journal.py` | 853 |
| `lifecycle.py` | 652 |
| `plan.py` | 888 |
| `pr.py` | 306 |
| `provider.py` | 601 |
| `research.py` | 1,732 |
| `review.py` | 1,054 |
| `review_helpers.py` | 314 |
| `router.py` | 102 |
| `server.py` | 573 |
| `spec.py` | 1,295 |
| `task.py` | 3,887 |
| `test.py` | 443 |
| `verification.py` | 532 |
| **Total** | **19,467** |

### Core LLM Providers (`src/foundry_mcp/core/providers/`)

| File | Lines |
|------|------:|
| `__init__.py` | 238 |
| `base.py` | 583 |
| `claude.py` | 474 |
| `claude_zai.py` | 477 |
| `codex.py` | 639 |
| `cursor_agent.py` | 632 |
| `detectors.py` | 522 |
| `gemini.py` | 428 |
| `opencode.py` | 723 |
| `registry.py` | 607 |
| `test_provider.py` | 171 |
| `validation.py` | 857 |
| **Total** | **6,351** |

### Research Providers (`src/foundry_mcp/core/research/providers/`)

| File | Lines |
|------|------:|
| `__init__.py` | 73 |
| `base.py` | 356 |
| `google.py` | 650 |
| `perplexity.py` | 729 |
| `resilience.py` | 847 |
| `semantic_scholar.py` | 672 |
| `tavily.py` | 674 |
| `tavily_extract.py` | 733 |
| **Total** | **4,734** |

### Research Workflows (`src/foundry_mcp/core/research/workflows/`)

| File | Lines |
|------|------:|
| `__init__.py` | 25 |
| `base.py` | 548 |
| `chat.py` | 289 |
| `consensus.py` | 562 |
| `deep_research.py` | 6,994 |
| `ideate.py` | 695 |
| `thinkdeep.py` | 418 |
| **Total** | **9,531** |

### Key Standalone Modules

| File | Lines |
|------|------:|
| `src/foundry_mcp/config.py` | 2,012 |
| `src/foundry_mcp/server.py` | 248 |

### Summary

| Category | Files | Lines |
|----------|------:|------:|
| Unified tool routers | 21 | 19,467 |
| Core LLM providers | 12 | 6,351 |
| Research providers | 8 | 4,734 |
| Research workflows | 7 | 9,531 |
| Key standalone modules | 2 | 2,260 |
| **Grand total** | **50** | **42,343** |

---

## 2. Duplicate Helper Inventory

### `_validation_error` (10 definitions)

| File | Line | Signature variant |
|------|-----:|-------------------|
| `authoring.py` | 88 | `(*, field, action, message, request_id, code=..., remediation=None)` |
| `environment.py` | 250 | `(*, action, ...)` |
| `journal.py` | 84 | `(field, action, ...)` |
| `lifecycle.py` | 73 | `(*, action, ...)` |
| `provider.py` | 60 | `(*, action, ...)` |
| `research.py` | 105 | `(field, action, ...)` |
| `server.py` | 91 | `(*, message, request_id, remediation=None)` |
| `task.py` | 110 | `(*, field, ...)` |
| `test.py` | 77 | `(*, message, request_id, remediation=None)` |
| `verification.py` | 48 | `(*, action, field, message, request_id, remediation=None, code=...)` |

Note: Signatures vary between files — some use positional `field`, others keyword-only `*`.

### `_request_id` (8 definitions)

| File | Line | Prefix |
|------|-----:|--------|
| `authoring.py` | 84 | `"authoring"` |
| `environment.py` | 241 | `"environment"` |
| `lifecycle.py` | 57 | `"lifecycle"` |
| `provider.py` | 56 | `"provider"` |
| `server.py` | 55 | `"server"` |
| `task.py` | 90 | `"task"` |
| `test.py` | 42 | `"test"` |
| `verification.py` | 44 | `"verification"` |

All follow `get_correlation_id() or generate_correlation_id(prefix=<prefix>)`.

### `_metric_name` (5 definitions)

| File | Line | Pattern |
|------|-----:|---------|
| `authoring.py` | 80 | `f"authoring.{action.replace('-', '_')}"` |
| `environment.py` | 237 | `f"environment.{action.replace('-', '_')}"` |
| `lifecycle.py` | 53 | `f"lifecycle.{action.replace('-', '_')}"` |
| `provider.py` | 52 | `f"provider.{action}"` (no replace) |
| `verification.py` | 40 | `f"verification.{action}"` (no replace) |

Note: Inconsistency — some apply `.replace('-', '_')`, others do not.

### `_metric` (3 definitions)

| File | Line | Pattern |
|------|-----:|---------|
| `server.py` | 59 | `f"unified_tools.server.{action.replace('-', '_')}"` |
| `task.py` | 94 | `f"unified_tools.task.{action.replace('-', '_')}"` |
| `test.py` | 46 | `f"unified_tools.test.{action.replace('-', '_')}"` |

Note: Different prefix scheme (`unified_tools.<router>`) vs `_metric_name` (`<router>`).

### `_resolve_specs_dir` (5 definitions)

| File | Line | Return type |
|------|-----:|-------------|
| `authoring.py` | 121 | `Optional[Path]` |
| `journal.py` | 113 | `Tuple[Optional[Path], Optional[dict]]` |
| `lifecycle.py` | 94 | `Optional[Path]` |
| `spec.py` | 74 | `Optional[Path]` |
| `task.py` | 132 | `Optional[Path]` |

Note: journal.py returns a tuple (path + error dict), others return just the path.

### `_build_router` (4 definitions)

| File | Line |
|------|-----:|
| `health.py` | 160 |
| `research.py` | 1,432 |
| `server.py` | 472 |
| `test.py` | 358 |

### Duplication Summary

| Helper | Definitions | Files |
|--------|----------:|-------|
| `_validation_error` | 10 | authoring, environment, journal, lifecycle, provider, research, server, task, test, verification |
| `_request_id` | 8 | authoring, environment, lifecycle, provider, server, task, test, verification |
| `_metric_name` | 5 | authoring, environment, lifecycle, provider, verification |
| `_resolve_specs_dir` | 5 | authoring, journal, lifecycle, spec, task |
| `_build_router` | 4 | health, research, server, test |
| `_metric` | 3 | server, task, test |
| **Total duplicated definitions** | **35** | |

---

## 3. Metric Name Inventory Per Router

### `task.py` (38 metrics — highest)

Uses `_metric(action)` -> `f"unified_tools.task.{action}"`

- `unified_tools.task.prepare` (+`.duration_ms`)
- `unified_tools.task.prepare_batch` (+`.duration_ms`)
- `unified_tools.task.add` (+`.duration_ms`)
- `unified_tools.task.update` (+`.duration_ms`)
- `unified_tools.task.move` (+`.duration_ms`)
- `unified_tools.task.set_status` (+`.duration_ms`)
- `unified_tools.task.mark_blocked` (+`.duration_ms`)
- `unified_tools.task.unblock` (+`.duration_ms`)
- `unified_tools.task.update_estimate` (+`.duration_ms`)
- `unified_tools.task.list` (+`.duration_ms`)
- `unified_tools.task.get` (+`.duration_ms`)
- `unified_tools.task.next` (+`.duration_ms`)
- `unified_tools.task.dependencies` (+`.duration_ms`)
- `unified_tools.task.blocked` (+`.duration_ms`)
- `unified_tools.task.start_batch` (+`.duration_ms`)
- `unified_tools.task.complete_batch` (+`.duration_ms`)
- `unified_tools.task.reset_batch` (+`.duration_ms`)
- `unified_tools.task.get_batch_status` (+`.duration_ms`)
- `unified_tools.task.edit` (+`.duration_ms`)

### `environment.py` (7 metrics)

Uses `_metric_name(action)` -> `f"environment.{action}"`

- `environment.verify_toolchain`
- `environment.init`
- `environment.detect`
- `environment.detect_test_runner`
- `environment.verify_env`
- `environment.setup`
- `environment.get_config`

### `authoring.py` (8+ metrics)

Uses `_metric_name(action)` -> `f"authoring.{action}"`

- `authoring.create`
- `authoring.add_phase`
- `authoring.update_phase`
- `authoring.delete_phase`
- `authoring.add_assumption`
- `authoring.update_assumption`
- `authoring.add_revision`
- `authoring.create_from_template`
- (plus dynamic variants per action)

### `test.py` (4 metrics)

Uses `_metric(action)` -> `f"unified_tools.test.{action}"`

- `unified_tools.test.run` (+`.duration_ms`)
- `unified_tools.test.discover` (+`.duration_ms`)

### `server.py` (1 metric)

Uses `_metric(action)` -> `f"unified_tools.server.{action}"`

- `unified_tools.server.tools` (+`.duration_ms`)

### `spec.py` (2 metrics)

Dynamic via `f"analysis.{tool_name}"`:

- `analysis.spec_analyze` (+`.duration_ms`)
- `analysis.spec_analyze_deps` (+`.duration_ms`)

### `plan.py` (3 metrics)

- `plan_review.security_blocked`
- `plan_review.errors`
- `plan_review.completed`
- `plan_create.security_blocked`
- `plan_create.completed`
- `plan_list.completed`

### `pr.py` (1 metric)

- `pr_workflow.pr_get_spec_context.duration_ms`

### `verification.py` (2 metrics)

Uses `_metric_name(action)` -> `f"verification.{action}"`

- `verification.run`
- `verification.list`

### `provider.py` (3 metrics)

Uses `_metric_name(action)` -> `f"provider.{action}"`

- `provider.list`
- `provider.status`
- `provider.execute`

### `lifecycle.py` (1 metric)

Uses `_metric_name(action)` -> `f"lifecycle.{action}"`

- `lifecycle.move`

### Routers with NO metrics

- `health.py`
- `journal.py`
- `error.py`
- `research.py`

### Naming Inconsistencies

| Pattern | Used by | Example |
|---------|---------|---------|
| `unified_tools.<router>.<action>` | task, test, server | `unified_tools.task.prepare` |
| `<router>.<action>` | authoring, environment, lifecycle, provider, verification | `environment.detect` |
| `<workflow>.<action>` | plan, pr, spec | `plan_review.completed` |

Three distinct prefix schemes across routers.

---

## 4. Key Log Fields Per Dispatch Function

### Routers with Substantive Structured Logging

#### `plan.py`

| Function | Level | Fields |
|----------|-------|--------|
| `_handle_consultation_result()` | info | `len(successful_responses)`, `successful_providers` |
| `_handle_consultation_result()` | error | exception `e`, `exc_info=True` |
| `_handle_consultation_result()` | warning | `error_detail` (values: "synthesis crashed", "empty response", "empty synthesis content") |
| `_handle_consultation_result()` | error | `type(result)` |

#### `review.py`

| Function | Level | Fields |
|----------|-------|--------|
| `execute_fidelity_review()` | warning | `response.provider_id` |
| `execute_fidelity_review()` | warning | exception `e` |
| `execute_fidelity_review()` | info | `len(successful_responses)`, `successful_providers` |
| `execute_fidelity_review()` | error | exception `e`, `exc_info=True` |
| `execute_fidelity_review()` | warning | `error_detail` |
| `execute_fidelity_review()` | warning | exception `e` |

#### `research.py`

| Function | Level | Fields |
|----------|-------|--------|
| `register_unified_research_tool()` | info | (none — message-only: "Research tools disabled in config") |

### Common Debug-Only Logging (All Other Routers)

All remaining routers contain a single debug log in their registration function:

```
logger.debug("Registered unified <toolname> tool")
```

Files: authoring.py, environment.py, error.py, health.py, journal.py, lifecycle.py, pr.py, provider.py, server.py, task.py, test.py, verification.py

### Helper File Logging

| File | Function | Level | Fields |
|------|----------|-------|--------|
| `review_helpers.py` | (helper) | debug | exception `exc` |
| `context_helpers.py` | (helper) | debug | exception `exc` |

### Observations

1. Only `plan.py` and `review.py` have substantive structured logging in dispatch paths
2. No router logs `action`, `spec_id`, or `task_id` at the dispatch level
3. No request/response logging at the router entry/exit points
4. Error handling uses `exc_info=True` for crashes, `error_detail` strings for known failures
