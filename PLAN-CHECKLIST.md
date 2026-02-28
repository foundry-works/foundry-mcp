# Plan 2 Implementation Checklist

## Model Change
- [ ] Add `report_output_path: Optional[str] = None` field to `DeepResearchState` in `src/foundry_mcp/core/research/models/deep_research.py`

## Auto-save on Synthesis Complete
- [ ] Add `_slugify_query()` helper in `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py`
- [ ] Add `_save_report_markdown(state)` helper in same file
  - [ ] Slugify `state.original_query` for filename
  - [ ] Truncate slug to 80 chars
  - [ ] Default output dir = `Path.cwd()`
  - [ ] Collision handling: append research ID suffix if file exists
  - [ ] try/except — non-fatal on failure
- [ ] Call `_save_report_markdown()` after `state.report = report` (~line 739)
- [ ] Persist `state.report_output_path` back to state

## Handler: `deep-research-report`
- [ ] Add `output_path: Optional[str] = None` param to `_handle_deep_research_report()` in `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`
- [ ] If `output_path` provided, save report to that path (override auto-save location)
- [ ] Include `output_path` (or `state.report_output_path`) in `response_data`

## Handler: `deep-research-status`
- [ ] When research is complete, include `report_output_path` from state in status response

## Verification
- [ ] Run deep research and confirm `.md` file appears in CWD
- [ ] Call `deep-research-report` and confirm `output_path` in response data
- [ ] Call `deep-research-status` on completed research and confirm `report_output_path`
- [ ] Test collision: same query twice produces unique filenames
- [ ] Test custom `output_path` override on `deep-research-report`
- [ ] Run existing deep research tests — no regressions
