# autonomy-session-observability-dx

## Mission

Add real-time monitoring, graceful stopping, and tmux-based launching for unattended autonomous sessions, plus rename the CLI from `sdd`/`foundry-cli` to `foundry`.

## Objective

Running unattended autonomous sessions (`claude -p "/foundry-implement-auto <spec>" --dangerously-skip-permissions`) currently has two critical DX gaps: (1) no observability — no live feedback on task progress, current step, or session health; (2) no clean intervention — killing the process leaves the session orphaned in RUNNING state. Rich session infrastructure already exists (session state files, audit ledger, journal, heartbeat tracking), but there's no user-facing tool that surfaces it. This plan adds three CLI commands (`foundry watch`, `foundry stop`, `foundry run`), a server-side signal file mechanism in the orchestrator, and a comprehensive rename of all legacy `sdd`/`SDD` references to `foundry`.

## Scope

### In Scope
- Add `foundry` as primary CLI entry point (keep `foundry-cli` as alias)
- Add `rich>=13.0.0` dependency for live dashboard
- `foundry watch <spec-id>` — Rich Live dashboard + `--simple` streaming mode
- `foundry stop <spec-id>` — signal file mechanism + optional `--force` process kill
- Signal file check in orchestrator (Step 7b in `compute_next_step()`)
- `foundry run <spec-id>` — tmux launcher with agent + watcher split panes
- Rename all `SDD_*` env vars to `FOUNDRY_*` (with backward-compat fallback)
- Rename all `sdd-*` MCP tool names to `foundry-*`
- Rename schema file, session ID prefix, backup suffix, docstrings, test files, docs

### Out of Scope
- `CHANGELOG.md` historical entries (keep as-is)
- Existing `specs/.reports/sdd-*` and `specs/.backups/sdd-*` files (historical artifacts)
- WebSocket/SSE streaming (polling is sufficient for v1)
- Full TUI framework (Rich Live panels are enough)

## Phases

### Phase 1: CLI Rename & Configuration

**Purpose**: Establish `foundry` as the primary CLI entry point and add the `rich` dependency. This unblocks all subsequent phases that create new commands.

**Tasks**:
1. Add `foundry` entry point and keep `foundry-cli` alias in `pyproject.toml` `[project.scripts]`
2. Add `rich>=13.0.0` to dependencies in `pyproject.toml`
3. Update `pyproject.toml` keywords: `"sdd"` → `"foundry"`
4. Update `src/foundry_mcp/cli/main.py`: rename docstring from "SDD CLI" to "Foundry CLI", accept `FOUNDRY_SPECS_DIR` envvar (with `SDD_SPECS_DIR` fallback)
5. Update `src/foundry_mcp/cli/registry.py` docstrings: "SDD CLI" → "Foundry CLI"

**Verification**: `pip install -e .` succeeds, `foundry --help` works, `foundry-cli --help` still works as alias.

### Phase 2: Signal File Infrastructure (Orchestrator)

**Purpose**: Add server-side signal file checking to the orchestrator so that `foundry stop` can trigger a clean PAUSE without modifying session state directly. This is the critical enabler for graceful intervention.

**Tasks**:
1. Add Step 7b to `compute_next_step()` in `src/foundry_mcp/core/autonomy/orchestrator.py`: check for `specs/.autonomy/signals/{spec_id}.stop`, consume signal file, transition to PAUSED with `PauseReason.USER`, emit audit event
2. Write unit tests for signal file detection (mock file exists → returns PAUSE), signal file cleanup after consumption, and no-op when signal file absent

**Verification**: `python -m pytest tests/unit/test_core/autonomy/test_orchestrator.py --tb=short` — new tests pass, existing tests unaffected.

### Phase 3: `foundry stop` Command

**Purpose**: Provide an operator-facing CLI command to gracefully stop a running autonomous session. Uses the signal file mechanism from Phase 2.

**Tasks**:
1. Create `src/foundry_mcp/cli/commands/stop.py`: Click command accepting `spec_id` argument, `--force` flag (pgrep + SIGTERM), `--wait` flag (poll until terminal state), `--timeout` option
2. Implement signal file write logic: create `specs/.autonomy/signals/` dir, write JSON signal file with `requested_at`, `requested_by`, `reason`
3. Implement `--force` mode: find claude processes via `pgrep -f "foundry-implement-auto.*{spec_id}"`, send SIGTERM
4. Implement `--wait` mode: poll `AutonomyMemory.load()` every 2s until session status is terminal or timeout
5. Register command in `src/foundry_mcp/cli/commands/__init__.py` and `src/foundry_mcp/cli/registry.py`
6. Write unit tests for stop command (signal file creation, force mode mock, wait mode mock)

**Verification**: `foundry stop --help` works. Unit tests pass. Manual test: start a session, run `foundry stop <spec-id>`, confirm signal file created and session pauses on next step.

### Phase 4: `foundry watch` Command

**Purpose**: Provide a Rich Live dashboard for real-time monitoring of autonomous sessions. Reads session state, audit ledger, and spec progress directly from disk (no MCP calls needed).

**Tasks**:
1. Create `src/foundry_mcp/cli/commands/watch.py` with Click command accepting `spec_id`, `--interval` (default 2.0s), `--events` (default 10), `--simple` flag
2. Implement session data assembly: resolve spec_id → session_id via `AutonomyMemory.get_active_session()`, load session state via `AutonomyMemory.load()`, load audit events via `AuditLedger.get_entries()`, load spec progress via `recalculate_progress()`
3. Implement Rich Live dashboard layout: status panel (session ID, status, phase, context %, heartbeat age), current step panel (type, task ID, step ID, time since issued), recent events table (timestamp, event type, task, metadata)
4. Implement keyboard handling: `q` to quit, `s` to write stop signal file (reuse logic from stop command)
5. Implement `--simple` streaming mode: tail audit ledger by last-seen sequence number, print one line per new event
6. Implement terminal state detection: auto-exit when session reaches COMPLETED/FAILED/ENDED/PAUSED
7. Register command in `__init__.py` and `registry.py`
8. Write unit tests for data assembly (mock AutonomyMemory + AuditLedger), dashboard rendering (mock Rich console)

**Verification**: `foundry watch --help` works. Unit tests pass. Manual test: start a session, run `foundry watch <spec-id>`, confirm dashboard updates in real-time.

### Phase 5: `foundry run` Command (tmux Launcher)

**Purpose**: One-command launch that creates a tmux session with the agent in one pane and the watcher in another. Best possible DX for running autonomous sessions.

**Tasks**:
1. Create `src/foundry_mcp/cli/commands/run.py` with Click command accepting `spec_id`, `--posture` (default "unattended"), `--detach`/`-d` flag, `--layout` (horizontal/vertical)
2. Implement tmux availability check (`shutil.which("tmux")`) with clear error message if missing
3. Implement tmux session creation: generate session name `foundry-{spec_id[:30]}`, build agent command with env vars, build watcher command with 5s delay, create session with split panes
4. Implement attach/detach behavior: attach by default, print reattach command when detached
5. Register command in `__init__.py` and `registry.py`
6. Write unit tests (mock subprocess calls, verify tmux command construction)

**Verification**: `foundry run --help` works. Unit tests pass. Manual test: `foundry run <spec-id>` creates tmux session with two panes.

### Phase 6: SDD → Foundry Rename (Environment Variables & Tool Names)

**Purpose**: Rename all `SDD_*` environment variables to `FOUNDRY_*` and all `sdd-*` MCP tool names to `foundry-*`. Maintain backward compatibility for env vars.

**Tasks**:
1. Update `src/foundry_mcp/cli/config.py`: accept `FOUNDRY_SPECS_DIR` with `SDD_SPECS_DIR` fallback
2. Update `src/foundry_mcp/cli/context.py`: rename `SDD_MAX_CONSULTATIONS` → `FOUNDRY_MAX_CONSULTATIONS`, `SDD_MAX_CONTEXT_TOKENS` → `FOUNDRY_MAX_CONTEXT_TOKENS`, `SDD_WARN_PERCENTAGE` → `FOUNDRY_WARN_PERCENTAGE` (all with fallbacks); change session ID prefix from `sdd_` to `foundry_`
3. Update 7 CLI command files (`modify.py`, `tasks.py`, `lifecycle.py`, `journal.py`, `specs.py`, `validate.py`, `review.py`): replace `SDD_SPECS_DIR` in error/remediation strings with `FOUNDRY_SPECS_DIR`
4. Update `src/foundry_mcp/core/discovery/metadata/environment.py`: rename 4 tool metadata entries from `sdd-*` to `foundry-*`, update `related_tools` arrays
5. Update `src/foundry_mcp/tools/unified/environment.py`: rename tool aliases (`sdd-detect-test-runner` → `foundry-detect-test-runner`, `sdd-setup` → `foundry-setup`), update audit log reference from `sdd_setup` to `foundry_setup`
6. Update tests in `tests/unit/test_environment.py`: update 5 test class docstrings referencing `sdd_*` function names, update any tool name assertions

**Verification**: `python -m pytest tests/unit/test_environment.py --tb=short` passes. `foundry --help` shows no `sdd` references. Env var backward compat works: setting `SDD_SPECS_DIR` still works.

### Phase 7: SDD → Foundry Rename (Schema, Files, Docs)

**Purpose**: Complete the rename by updating schema files, test file names, documentation, and remaining internal references.

**Tasks**:
1. Rename `src/foundry_mcp/schemas/sdd-spec-schema.json` → `foundry-spec-schema.json`
2. Update `src/foundry_mcp/schemas/__init__.py`: change `load_schema()` default param from `sdd-spec-schema.json` to `foundry-spec-schema.json`
3. Update `src/foundry_mcp/core/providers/cursor_agent.py`: change `.sdd-backup` → `.foundry-backup`
4. Update `src/foundry_mcp/cli/commands/session.py` module docstring
5. Update all remaining CLI command file docstrings referencing "SDD"
6. Rename test files: `test_sdd_cli_core.py` → `test_foundry_cli_core.py`, `test_sdd_cli_runtime.py` → `test_foundry_cli_runtime.py`, `test_sdd_cli_parity.py` → `test_foundry_cli_parity.py`
7. Update `tests/unit/test_core/test_spec.py`: change `modified_by="sdd-cli"` markers to `"foundry-cli"`
8. Rename `docs/concepts/sdd-philosophy.md` → `docs/concepts/foundry-philosophy.md`
9. Update links in `docs/README.md` and `README.md` pointing to the renamed doc
10. Update `docs/04-cli-command-reference.md` (env var table), `docs/07-troubleshooting.md` (`grep -i sdd` example), `docs/reference/error-codes.md` (`SDD_SPECS_DIR` reference)

**Verification**: `python -m pytest tests/ --tb=short -x` — full test suite passes. `grep -ri "sdd" src/ tests/ docs/ --include="*.py" --include="*.md" --include="*.json" | grep -v CHANGELOG | grep -v .backups | grep -v .reports` returns no results (all references cleaned up except historical artifacts).

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| `rich` dependency adds weight | Low | `rich` is widely used, well-maintained, pure Python. Minimal footprint. |
| Signal file race condition | Low | Signal file is write-once/read-once. Orchestrator consumes atomically. No version conflict with session state. |
| tmux not available on all systems | Medium | `foundry run` checks for tmux and shows clear error. `foundry watch` and `foundry stop` work independently without tmux. |
| Env var rename breaks existing users | Medium | Backward-compatible fallback: read `FOUNDRY_*` first, fall back to `SDD_*`. No breaking change. |
| Tool name rename breaks MCP clients | Low | Environment tools are rarely called directly by external clients. Internal-only impact. |
| Large number of files touched in rename | Medium | Rename phases (6-7) are low-risk string replacements. Full test suite run validates nothing is broken. |

## Success Criteria

- [ ] `foundry watch <spec-id>` shows live dashboard with session state, progress, audit events
- [ ] `foundry watch <spec-id> --simple` streams events in tail-like mode
- [ ] `foundry stop <spec-id>` creates signal file and session pauses on next orchestrator step
- [ ] `foundry stop <spec-id> --force` also kills the claude process
- [ ] `foundry stop <spec-id> --wait` blocks until session reaches terminal state
- [ ] `foundry run <spec-id>` creates tmux session with agent + watcher panes
- [ ] Signal file check works server-side in orchestrator (Step 7b)
- [ ] All `sdd`/`SDD` references cleaned up (except CHANGELOG and historical artifacts)
- [ ] `FOUNDRY_*` env vars work, `SDD_*` still accepted as fallback
- [ ] Full test suite passes after all changes
