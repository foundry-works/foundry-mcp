# Make Deep Research a Blocking Operation

## Context

Currently, `research(action="deep-research")` runs in a background daemon thread and returns immediately with a `research_id` + polling guidance. The caller must then poll via `deep-research-status` (up to 5 times) and finally retrieve results via `deep-research-report`. This 3+ tool-call dance is unnecessary friction — all other research workflows (chat, consensus, thinkdeep, ideate) are already blocking/synchronous.

The goal is to make deep research block until completion and return the full report in a single tool call, matching the pattern used by fidelity review and all other research workflows.

## Current Flow (3+ tool calls)

1. `research(action="deep-research", query="...")` → returns immediately with `{research_id, status: "started", polling_guidance}`
2. `research(action="deep-research-status", research_id="...")` → returns progress (repeat up to 5x)
3. `research(action="deep-research-report", research_id="...")` → returns final report

## Desired Flow (1 tool call, blocking)

1. `research(action="deep-research", query="...")` → blocks until complete → returns `{report, research_id, source_count, ...}`

## Architecture

The synchronous execution path already exists in the workflow layer (`action_handlers.py:_start_research` with `background=False`). The only file that needs changing is the MCP handler layer that currently hardcodes `background=True`.

### Key code paths

- **Handler (change here):** `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py` — `_handle_deep_research()` line 114 sets `background=True`
- **Workflow sync path (already works):** `src/foundry_mcp/core/research/workflows/deep_research/action_handlers.py` — `_start_research()` lines 171-206 handle `background=False`
- **Workflow completion return:** `src/foundry_mcp/core/research/workflows/deep_research/workflow_execution.py` — lines 560-576 return `WorkflowResult(success=True, content=state.report, metadata={research_id, phase, iteration, source_count, ...})`
- **Report builder (reuse):** `handlers_deep_research.py:_handle_deep_research_report()` — loads persisted state and builds rich response with content fidelity, allocation warnings

## Changes

### `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

#### `_handle_deep_research()`

1. Change `background=True` to `background=False` at line 114
2. Replace the success response block (lines 118-144) — instead of returning polling guidance, call `_handle_deep_research_report(research_id=result.metadata.get("research_id"))` to reuse the existing report-building logic (content fidelity, allocation warnings, dropped content IDs)
3. Update docstring to reflect blocking behavior

### No changes to:

- `action_handlers.py` — synchronous execution path already works
- `background_tasks.py` — infrastructure stays in case background mode is ever re-enabled
- `workflow_execution.py` — async workflow engine is the same regardless of blocking mode
- `handlers_workflows.py` — other research workflows already blocking
- Lifecycle handlers (`deep-research-status`, `deep-research-report`, `deep-research-list`, `deep-research-delete`, `deep-research-evaluate`) — all remain for inspecting past sessions
