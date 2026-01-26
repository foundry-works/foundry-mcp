# Deep Research Status Visibility Plan

## Goals
- Make it clear whether a deep-research run is actively progressing or likely stalled.
- Always include structured, machine-readable progress signals in status responses.
- Avoid changing research outputs or content fidelity.

## Non-Goals
- No new human-readable status guidance strings in the status payload.
- No changes to phase logic, provider behavior, or prompt content.
- No new action variants or detail levels for status.

## Proposed Status Fields (Always Included)
Add these fields to `deep-research-status` responses:
- `phase_started_at` (ISO 8601 UTC)
- `phase_last_update_at` (ISO 8601 UTC)
- `last_progress_at` (ISO 8601 UTC)
- `last_progress_event` (string enum; examples below)
- `phase_elapsed_ms` (int)
- `progress_age_ms` (int)
- `progress_signal` (`active|stale|unknown`)
- `stale_threshold_seconds` (int; fixed at 300)
- `task_status` (`running|completed|failed|timeout|cancelled`)
- `background_thread_alive` (bool or null if no background task)
- `task_timeout_seconds` (number or null)
- `time_remaining_seconds` (number or null)

## Progress Events (Structured Enum)
Standardize `last_progress_event` values:
- `phase_start:<phase>`
- `phase_complete:<phase>`
- `planning_parsed`
- `gathering_complete`
- `analysis_parsed`
- `synthesis_report_saved`
- `refinement_queries_generated`
- `workflow_completed`
- `workflow_failed`

## Data Capture Points
Persist progress markers on:
- Phase start and phase completion hooks.
- After parsing phase results (planning/analysis/refinement).
- After synthesis report is saved to state.
- When workflow completes or fails.

## Staleness Signal (Fixed Threshold)
- `progress_signal` is computed at status time.
- `active` if `progress_age_ms <= 300_000`
- `stale` if `progress_age_ms > 300_000`
- `unknown` if `last_progress_at` is missing

## State Storage
Add new fields to `DeepResearchState`:
- `phase_started_at`
- `phase_last_update_at`
- `last_progress_at`
- `last_progress_event`

These are updated and persisted by the workflow as progress occurs.

## Status Computation (At Request Time)
Compute derived fields without modifying stored state:
- `phase_elapsed_ms = now - phase_started_at`
- `progress_age_ms = now - last_progress_at`
- `task_status` and `background_thread_alive` from task registry
- `time_remaining_seconds` if a task timeout is configured

## Contract Updates (Specs + Docs + Tests)
- Update deep-research status contract to include new fields.
- Add examples showing stalled vs active signals.
- Update troubleshooting docs to interpret `progress_signal`.
- Add tests covering:
  - New fields present in status response
  - Progress markers updated on phase transitions
  - `progress_signal` thresholds

## Files to Touch (Planning Only)
- `src/foundry_mcp/core/research/models.py`
- `src/foundry_mcp/core/research/workflows/deep_research.py`
- `src/foundry_mcp/tools/unified/research.py`
- `specs/...` (deep-research status contract)
- `docs/concepts/deep_research_workflow.md`
- `docs/examples/deep-research/README.md`
- `tests/core/research/workflows/test_deep_research.py`

## Acceptance Criteria
- Status responses always include the new structured fields.
- `progress_signal` reliably differentiates active vs stale runs using 300s threshold.
- No changes to research outputs or content fidelity.
