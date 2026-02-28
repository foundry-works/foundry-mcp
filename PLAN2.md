# Plan: Auto-save Deep Research Reports as Markdown Files

## Context

When deep research completes, the report is stored only inside a JSON state file (`~/.foundry-mcp/research/deep_research/{id}.json`). Users have no standalone markdown artifact — they must call `deep-research-report` and then manually extract/save the content. The goal is to automatically save the report as a `.md` file in the working directory (by default) so users get a ready-to-read artifact.

## Approach

Add automatic markdown file saving at two points:
1. **When synthesis completes** (background task finishes) — auto-save to default location
2. **When `deep-research-report` is called** — optionally re-save/override with custom `output_path`

Both return the `output_path` in the response so the caller knows where the file landed.

## Files to Modify

### 1. `src/foundry_mcp/core/research/workflows/deep_research/phases/synthesis.py` (~line 739-742)

After `state.report = report` and `self.memory.save_deep_research(state)`, add auto-save logic:

```python
# Auto-save report as markdown file
output_path = _save_report_markdown(state)
if output_path:
    state.report_output_path = str(output_path)
    self.memory.save_deep_research(state)
```

The `_save_report_markdown()` helper:
- Slugifies `state.original_query` to create a filename (e.g., `conversation-based-assessment-in-education.md`)
- Truncates slug to ~80 chars for sanity
- Resolves output directory: use `Path.cwd()` as default (the directory the server was launched from)
- Writes `state.report` via `Path.write_text(content, encoding="utf-8")`
- Wraps in try/except — failure is non-fatal (log warning, don't break the workflow)
- If file already exists, append research_id suffix to avoid overwriting (e.g., `conversation-based-assessment-deepres-8bc677.md`)

### 2. `src/foundry_mcp/core/research/models/deep_research.py` (DeepResearchState model)

Add optional field:
```python
report_output_path: Optional[str] = None  # Path to saved markdown report file
```

### 3. `src/foundry_mcp/tools/unified/research_handlers/handlers_deep_research.py`

**In `_handle_deep_research_report()`** (~line 195):
- Add `output_path: Optional[str] = None` parameter
- After getting the report, if `output_path` is provided, save to that path (overriding the auto-saved location)
- If no `output_path` but `state.report_output_path` exists, include it in response
- Add `output_path` to `response_data` so the caller sees where the file is

**In `_handle_deep_research_status()`** (~line 146):
- When research is complete, include `report_output_path` from state in the status response so the user gets pointed to the file immediately

### 4. Reuse `_slugify()` pattern

Extract inline helper in synthesis.py (4 lines, matches existing pattern in `plan.py:195-200`).

## Implementation Details

### Filename generation
```python
def _slugify_query(query: str, max_len: int = 80) -> str:
    slug = query.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    return slug[:max_len].rstrip("-")
```

### Default output directory
- Use `Path.cwd()` — the working directory of the MCP server process, which matches "the folder from which it was called"

### Collision handling
- If `{slug}.md` exists, use `{slug}-{research_id_suffix}.md` (last 8 chars of ID)

### Error handling
- All file I/O wrapped in try/except
- Failure logs a warning but does NOT fail the research
- If save fails, `report_output_path` remains None

## Response Changes

`deep-research-status` (when complete) will include:
```json
{"report_output_path": "/home/user/project/conversation-based-assessment.md"}
```

`deep-research-report` will include:
```json
{"report": "...", "output_path": "/home/user/project/conversation-based-assessment.md"}
```
