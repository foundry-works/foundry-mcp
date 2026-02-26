# Blocking Deep Research — Implementation Checklist

## Implementation

- [ ] In `_handle_deep_research()`: change `background=True` → `background=False` (line 114)
- [ ] In `_handle_deep_research()`: replace success response block (lines 118-144) to return full report via `_handle_deep_research_report(research_id=...)`
- [ ] In `_handle_deep_research()`: update docstring to reflect blocking behavior
- [ ] In `_handle_deep_research()`: handle timeout/error cases — return error_response with research_id and failure details

## Verification

- [ ] Run existing deep research tests: `pytest tests/ -k deep_research`
- [ ] Manual MCP test: `research(action="deep-research", query="test")` blocks and returns full report
- [ ] Confirm `deep-research-report` still works for retrieving past sessions
- [ ] Confirm `deep-research-status` still works (though no longer needed for primary flow)
- [ ] Confirm `deep-research-list` still lists completed sessions

## Not Changed (intentionally)

- [ ] `action_handlers.py` sync path — already works, no changes needed
- [ ] `background_tasks.py` — infrastructure preserved
- [ ] `workflow_execution.py` — engine unchanged
- [ ] Other research handlers (chat, consensus, thinkdeep, ideate) — already blocking
- [ ] Lifecycle handlers (status, report, list, delete, evaluate) — preserved
