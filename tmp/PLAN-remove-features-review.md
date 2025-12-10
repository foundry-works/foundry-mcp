To ensure the `PLAN-remove-features` execution is safe and valid, I have conducted a review of your plan against the codebase state.

### Review Summary

#### Critical Blockers
Issues that MUST be fixed before this becomes a spec.

- **[Sequencing]** Duplicate `task-list` registration in Phase 1/2
  - **Description:** Phase 1 adds `task-list` to `tools/tasks.py`, but Phase 3 removes the old `tools/rendering.py`. In the interim (Phases 1 & 2), `server.py` will register both modules, causing a tool name collision for `task-list` which will likely crash the server or cause undefined behavior.
  - **Impact:** Server fails to start or behaves unpredictably after Phase 1.
  - **Fix:** In Phase 1, add a task to **remove** the `task-list` function (and its registration decorator) from `src/foundry_mcp/tools/rendering.py`.

- **[Completeness]** Missing imports in `tools/tasks.py`
  - **Description:** `task-list` relies on `foundry_mcp.core.pagination` (specifically `encode_cursor`, `decode_cursor`, `paginated_response`, `normalize_page_size`, `CursorError`) and `get_status_icon`. `tools/tasks.py` does not currently import these.
  - **Impact:** `task-list` will fail with `NameError` immediately upon use.
  - **Fix:** In Phase 1, explicit instruction to add `from foundry_mcp.core.pagination import ...` and `from foundry_mcp.core.progress import get_status_icon` to `src/foundry_mcp/tools/tasks.py`.

#### Major Suggestions
Significant improvements to strengthen the plan.

- **[Clarity]** Import cleanup in `tools/documentation.py`
  - **Description:** Phase 1 removes `spec-doc` and `spec-doc-llm` from `tools/documentation.py`. However, it must also explicitly remove the imports for `foundry_mcp.core.docgen` and `foundry_mcp.core.rendering`. If these imports remain, `tools/documentation.py` will break in Phase 3/4 when those core modules are deleted.
  - **Impact:** Import errors in later phases even though the tools were removed.
  - **Fix:** Add a specific task in Phase 1 or 5: "Remove unused imports (`core.docgen`, `core.rendering`) from `src/foundry_mcp/tools/documentation.py`".

#### Minor Suggestions
Smaller refinements.

- **[Clarity]** `get_status_icon` source
  - **Description:** Phase 1 adds `get_status_icon` to `core/progress.py`. It should specify whether this is a copy from `core/rendering.py` (resulting in duplication until Phase 3) or a move.
  - **Fix:** Clarify that it is a copy/re-implementation to avoid modifying `core/rendering.py` before Phase 3 (which would break `tools/rendering.py` imports).

#### Questions
Clarifications needed before proceeding.

- **[Testing]** Verification of `spec-review-fidelity` dependencies
  - **Context:** We are keeping `spec-review-fidelity`. Does it rely on any helpers in `core/docs.py` (which is being deleted in Phase 2)? My quick check suggests it's safe (it uses `ai_consultation` and internal helpers), but confirmation is good.
  - **Needed:** Confirm `spec-review-fidelity` is self-contained or only depends on `core/ai_consultation.py` and `core/spec.py`.

#### Praise
What the plan does well.

- **[Strategy]** Good phasing strategy
  - **Why:** Relocating preserved functionality *before* destructive removal is the correct safe approach.
- **[Scope]** Accurate file identification
  - **Why:** correctly identified all related files including tests and CLI commands.