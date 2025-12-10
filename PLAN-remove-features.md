# Phase Plan: Remove Doc-Query, Spec Rendering, and Codebase Doc Generation

## Objective

Remove three feature sets from foundry-mcp while preserving two critical tools:
- **Keep**: `task-list` tool (relocate to tools/tasks.py)
- **Keep**: `spec-review-fidelity` tool (stays in documentation.py)

## Scope

### Features to Remove

| Feature | MCP Tools | Core Modules | CLI Commands |
|---------|-----------|--------------|--------------|
| Doc-Query | 7 tools (code-find-class, code-find-function, code-trace-calls, code-impact-analysis, code-get-callers, code-get-callees, doc-stats) | core/docs.py | cli/commands/docquery.py |
| Spec Rendering | 2 tools (spec-render, spec-render-progress) | core/rendering.py | cli/commands/render.py |
| Doc Generation | 2 tools (spec-doc, spec-doc-llm) | core/docgen.py, core/prompts/doc_generation.py | cli/commands/docgen.py |

### Tools to Preserve

| Tool | Current Location | Action |
|------|------------------|--------|
| `task-list` | tools/rendering.py | Move to tools/tasks.py |
| `spec-review-fidelity` | tools/documentation.py | Keep in place, remove sibling tools |
| `get_status_icon()` | core/rendering.py | Move to core/progress.py |

## Phases

### Phase 1: Relocate Preserved Functionality

**Purpose**: Move tools and helpers that must be preserved before deleting their source modules.

**Tasks**:
1. Add `get_status_icon()` function to `core/progress.py`
2. Add `task-list` tool to `tools/tasks.py` (import from new location)
3. Update `tools/documentation.py` to remove spec-doc and spec-doc-llm while keeping spec-review-fidelity

**Files Modified**:
- `src/foundry_mcp/core/progress.py`
- `src/foundry_mcp/tools/tasks.py`
- `src/foundry_mcp/tools/documentation.py`

**Verification**: Server starts, task-list tool works, spec-review-fidelity tool works

---

### Phase 2: Remove Doc-Query Feature

**Purpose**: Remove all doc-query MCP tools, core module, and CLI commands.

**Tasks**:
1. Remove `register_docs_tools` import and call from `server.py`
2. Delete `src/foundry_mcp/tools/docs.py`
3. Delete `src/foundry_mcp/core/docs.py`
4. Delete `src/foundry_mcp/cli/commands/docquery.py`
5. Delete `tests/doc_query/` directory

**Files Deleted**:
- `src/foundry_mcp/tools/docs.py`
- `src/foundry_mcp/core/docs.py`
- `src/foundry_mcp/cli/commands/docquery.py`
- `tests/doc_query/*`

**Verification**: Server starts, no doc-query tools registered, tests pass

---

### Phase 3: Remove Spec Rendering Feature

**Purpose**: Remove rendering MCP tools, core module, and CLI commands (task-list already relocated).

**Tasks**:
1. Remove `register_rendering_tools` import and call from `server.py`
2. Delete `src/foundry_mcp/tools/rendering.py`
3. Delete `src/foundry_mcp/core/rendering.py`
4. Delete `src/foundry_mcp/cli/commands/render.py`
5. Delete `tests/unit/test_core/test_rendering.py`

**Files Deleted**:
- `src/foundry_mcp/tools/rendering.py`
- `src/foundry_mcp/core/rendering.py`
- `src/foundry_mcp/cli/commands/render.py`
- `tests/unit/test_core/test_rendering.py`

**Verification**: Server starts, task-list still works (from new location), tests pass

---

### Phase 4: Remove Doc Generation Feature

**Purpose**: Remove docgen core module, prompts, and CLI commands.

**Tasks**:
1. Delete `src/foundry_mcp/core/docgen.py`
2. Delete `src/foundry_mcp/core/prompts/doc_generation.py`
3. Update `src/foundry_mcp/core/prompts/__init__.py` to remove DOC_GENERATION workflow
4. Delete `src/foundry_mcp/cli/commands/docgen.py`
5. Delete `tests/integration/test_llm_docs.py`

**Files Deleted**:
- `src/foundry_mcp/core/docgen.py`
- `src/foundry_mcp/core/prompts/doc_generation.py`
- `src/foundry_mcp/cli/commands/docgen.py`
- `tests/integration/test_llm_docs.py`

**Verification**: Server starts, prompts module loads, tests pass

---

### Phase 5: CLI and Test Cleanup

**Purpose**: Remove all CLI registrations and update remaining tests.

**Tasks**:
1. Update `src/foundry_mcp/cli/commands/__init__.py` - remove docquery, render, docgen imports
2. Update `src/foundry_mcp/cli/main.py` - remove command group registrations
3. Update `tests/unit/test_documentation.py` - keep only spec-review-fidelity tests
4. Update `tests/integration/test_mcp_smoke.py` - remove rendering tool tests
5. Update `tests/integration/test_mcp_tools.py` - update tool count assertions
6. Update `tests/integration/test_llm_tools.py` - update documentation tool tests

**Verification**: Full test suite passes, CLI help shows updated commands

---

### Phase 6: Documentation Cleanup

**Purpose**: Update documentation to reflect removed features.

**Tasks**:
1. Update `docs/guides/llm-configuration.md` - remove spec-doc-llm references
2. Update `docs/guides/development-guide.md` - remove doc-query/rendering references
3. Update `docs/codebase_standards/naming-conventions.md` - remove removed tool references
4. Consider removing `docs/generated/` directory

**Verification**: Documentation builds, no broken references

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking task-list during relocation | High | Relocate first, verify before deleting source |
| Breaking spec-review-fidelity | High | Carefully edit documentation.py, verify before other removals |
| Missing import cleanup | Medium | Run full test suite after each phase |
| Test failures from tool count assertions | Low | Update assertions as part of phase 5 |

## Success Criteria

1. MCP server starts without errors
2. `task-list` tool functions correctly from new location
3. `spec-review-fidelity` tool functions correctly
4. All remaining tests pass
5. No references to removed modules in codebase
6. Documentation updated
