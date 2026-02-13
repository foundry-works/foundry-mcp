# remove-mcp-tools-pr-code-test

## Objective

Remove `pr`, `code`, and `test` entirely — MCP tools, CLI commands, and supporting core modules — in a single pull request. No staged deprecation window.

## Mission

Remove pr, code, and test entirely — MCP tools, CLI commands, and core/testing.py — eliminating runtime registration, discovery exposure, CLI entry points, and all references across docs, prompts, and tests in one atomic PR.

## Glossary

| Term | Meaning | Tracked In |
|------|---------|------------|
| **Unified router module** | Python file in `tools/unified/` implementing a tool's dispatch | `tools/unified/*.py` |
| **Runtime registered tool** | Tool registered with MCP server at startup via `register_unified_tools` | `tools/unified/__init__.py` |
| **Manifest-advertised tool** | Tool returned by `server(action="tools")` for LLM discovery | `tools/unified/server.py` |
| **Capabilities manifest entry** | Static tool metadata in the JSON capabilities file | `mcp/capabilities_manifest.json` |
| **CLI command group** | Click command group registered in CLI registry | `cli/commands/*.py`, `cli/registry.py` |

## Counts & Surfaces (Before → After)

| Surface | Before | Remove | After |
|---------|--------|--------|-------|
| Unified router modules (`tools/unified/*.py`) | 16 | `pr.py`, `test.py` (2) | **14** |
| Runtime registered tools (`register_unified_tools`) | 16 | `pr`, `test` (2) | **14** |
| `server(action="tools")` manifest | 15 (excludes `research`) | `pr`, `test` (2) | **13** |
| `capabilities_manifest.json` `unified_tools.tools` list | 16 (excludes `research`) | `pr`, `code`, `test` (3) | **13** |
| CLI command groups | includes `pr`, `test` | `pr`, `test` (2) | updated |
| Core modules | includes `core/testing.py` | `core/testing.py` (1) | updated |

Notes:
- `code` has no runtime router module and no CLI command — removal is discovery/manifest only
- `research` is runtime-registered but excluded from both manifest surfaces (unchanged)
- Docstrings referencing "16-router" refer to unified router modules → update to **14**
- `core/review.py` is NOT removed — it is used by the `review` unified tool and CLI
- `core/testing.py` IS removed — its only consumers are `tools/unified/test.py` and `cli/commands/testing.py`, both being deleted

## Scope

### In Scope
- Runtime unified tool registration removal (`pr`, `test`)
- CLI command group removal (`pr`, `test`) and registry cleanup
- Core module removal (`core/testing.py`)
- Discovery/manifest removal (`pr`, `code`, `test`)
- Prompt/config reference cleanup
- Documentation updates (tool reference, CLI reference, README, naming conventions, development guide, import audit)
- Count-reference docstring/comment updates (per surface-specific counts above)
- Test updates and deletions (including review-identified missing files)
- Manifest version bump (breaking contract change)
- Cleanup of PLAN_TOOL_REMOVAL.md

### Out of Scope
- `core/review.py` — used by the review tool/CLI, must stay
- `code` runtime/CLI module (never existed)

## Contract Change: Version Bump

**This is a breaking change.** Removing tool entries from the capabilities manifest requires a version bump.

- **Field**: `schema_version` in `mcp/capabilities_manifest.json` (currently `"1.1.0"`)
- **Bump**: `"1.1.0"` → `"2.0.0"` (semver major — tools removed = breaking)
- **Field**: `server.version` (currently `"0.5.1"`) → `"0.6.0"` (0.x minor bump; per semver, 0.x minor bumps may include breaking changes)
- **Field**: `unified_tools.description` — update count text `"(16 tools)"` → `"(13 tools)"`
- **Compatibility note**: Consumers calling `pr`, `test`, or `code` will receive a standard MCP "tool not found" error.

## Phases

### Phase 1: Remove Runtime Tool Registration

**Purpose**: Remove `pr` and `test` from the unified tool runtime so the MCP server no longer registers them.

**Tasks**:
1. Update `src/foundry_mcp/tools/unified/__init__.py` — remove imports, registration branches, `__all__` entries
2. Delete `src/foundry_mcp/tools/unified/pr.py`
3. Delete `src/foundry_mcp/tools/unified/test.py`

**Verification**: `python -c "from foundry_mcp.tools.unified import register_unified_tools"` succeeds.

### Phase 2: Remove CLI Commands and Core Modules

**Purpose**: Remove CLI pr and test command groups, their registry entries, and core/testing.py.

**Tasks**:
1. Delete `src/foundry_mcp/cli/commands/pr.py`
2. Delete `src/foundry_mcp/cli/commands/testing.py`
3. Update `src/foundry_mcp/cli/commands/__init__.py` — remove pr_group, test_group
4. Update `src/foundry_mcp/cli/registry.py` — remove pr_group, test_group imports and registration
5. Delete `src/foundry_mcp/core/testing.py`

**Verification**: `python -c "from foundry_mcp.cli.registry import register_all_commands"` succeeds.

### Phase 3: Remove Discovery Exposure

**Purpose**: Remove `pr`, `code`, and `test` from all discovery surfaces. Bump manifest version.

**Tasks**:
1. Update `src/foundry_mcp/tools/unified/server.py` — remove _PR_ROUTER/_TEST_ROUTER imports and dict entries
2. Update `mcp/capabilities_manifest.json` — remove tool objects, list entries, bump versions
3. Validate JSON

**Verification**: JSON validates. No pr/code/test tool entries in manifest.

### Phase 4: Update Prompt and Config References

**Tasks**:
1. Update `src/foundry_mcp/prompts/workflows.py` — remove test/pr MCP guidance
2. Update `src/foundry_mcp/tools/unified/environment.py` — remove pr/test from TOML template

**Verification**: `rg` sweep returns no matches for removed tool action patterns.

### Phase 5: Update Documentation

**Tasks**:
1. Update `docs/05-mcp-tool-reference.md` — remove pr/code/test
2. Update `docs/README.md` — remove MCP and CLI mappings
3. Update `docs/04-cli-command-reference.md` — remove pr/test CLI sections and MCP equivalents
4. Update `dev_docs/codebase_standards/naming-conventions.md` — update router lists/counts
5. Update `dev_docs/guides/development-guide.md` — remove test/code tool examples
6. Update `dev_docs/refactoring/import_consumer_audit.md` — remove pr.py consumer, update router inventory

**Verification**: No stale MCP/CLI tool references in docs.

### Phase 6: Update and Remove Tests

**Tasks**:
1. Update `tests/integration/test_mcp_smoke.py` — remove pr/test from _UNIFIED_TOOL_NAMES
2. Update `tests/tools/unified/test_tool_registration_parity.py` — remove pr/test
3. Update `tests/tools/unified/test_dispatch_common.py` — remove baselines, update count 16→14
4. Update `tests/tools/unified/test_telemetry_invariants.py` — remove from ROUTER_BASELINES, _DISPATCH_SIGNATURES, _KEYWORD_ONLY_DISPATCH, _call_dispatch(); rename test method, update count assertions
5. Update `tests/unit/test_contracts/test_dispatch_contracts.py` — update "16 routers" → 14
6. Delete `tests/tools/unified/test_pr.py`
7. Delete `tests/tools/unified/test_test.py`
8. Remove `foundry_mcp.tools.unified.pr` entry from `tests/unit/test_core/test_spec_public_api.py`

**Verification**: All unified tool tests pass.

### Phase 7: Count-Reference Sweep and Final Validation

**Tasks**:
1. Update `src/foundry_mcp/server.py` docstring — "16-router" → "14-router"
2. Update `src/foundry_mcp/tools/__init__.py` docstring — "16-router" → "14-router"
3. Sweep for remaining 16-router count references (including fractional like "13/16", "all 16")
4. Sweep for stale pr/test/code import references (including _PR_ROUTER, _TEST_ROUTER, _dispatch_pr_action, _dispatch_test_action)
5. Delete PLAN_TOOL_REMOVAL.md
6. Full test suite run

**Verification**: Zero stale references. All tests green.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Broken imports after deleting modules | High | `rg` sweep for all import paths before and after deletion |
| Runtime vs discovery count drift | Medium | Use Counts & Surfaces table; verify each surface independently |
| Stale prompt/docs/CLI guidance | Medium | Targeted `rg` sweeps for removed tool patterns |
| Contract change without versioning note | Low | Explicit version bumps; migration note in PR |
| core/review.py accidentally removed | High | Out-of-scope constraint; only core/testing.py deleted |

## Success Criteria

- [ ] MCP server runtime registers 14 tools (pr and test removed)
- [ ] `server(action="tools")` advertises 13 tools (pr and test removed)
- [ ] `mcp/capabilities_manifest.json` no longer includes pr, code, or test; schema_version bumped to 2.0.0
- [ ] CLI no longer has pr or test command groups
- [ ] core/testing.py deleted; no remaining imports
- [ ] Docs/prompts/config no longer present removed tools as available options
- [ ] Unified tool tests pass with updated router inventory/count assumptions
- [ ] No stale imports/references to removed modules remain
- [ ] PLAN_TOOL_REMOVAL.md deleted
