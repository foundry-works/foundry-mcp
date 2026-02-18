# Feature Flag Removal — Remaining Work

## What's Done

All **source code** changes are complete. Feature flag runtime gating has been removed from:

- **Session handlers** (`handlers_session_lifecycle.py`, `handlers_session_query.py`, `handlers_session_step.py`, `handlers_session_rebase.py`) — removed 14 `_is_feature_enabled` guard blocks and imports
- **Review handler** (`review.py`) — removed `autonomy_fidelity_gates` check
- **Helper functions** (`_helpers.py`) — removed `_is_feature_enabled()` and `_feature_disabled_response()`
- **ServerConfig** (`config/server.py`) — removed `feature_flags: Dict[str, bool]` field
- **Config parsing** (`config/parsing.py`) — removed `_FEATURE_FLAG_*` constants, `_normalize_feature_flag_name()`, `_parse_feature_flags_mapping()`, `_parse_feature_flags_env()`
- **Config loading** (`config/loader.py`) — removed TOML `[feature_flags]` loading, env var loading (`FOUNDRY_MCP_FEATURE_FLAGS`, `FOUNDRY_MCP_FEATURE_FLAG_*`), and dependency validation
- **Config re-exports** (`config/__init__.py`) — removed feature flag symbol re-exports
- **Capabilities** (`core/discovery/capabilities.py`) — simplified to always report `autonomy_sessions=True`, `autonomy_fidelity_gates=True`, `autonomy_gate_invariants=True`; removed `feature_flags` parameter from `get_capabilities()`
- **Discovery flags** (`core/discovery/flags.py`) — removed `AUTONOMY_FEATURE_FLAGS` dict and helper functions; kept `FeatureFlagDescriptor` (used by metadata modules)
- **Discovery re-exports** (`core/discovery/__init__.py`) — removed autonomy flag re-exports
- **Error codes** (`core/responses/types.py`) — removed `ErrorCode.FEATURE_DISABLED` and `ErrorType.FEATURE_FLAG`
- **Signals** (`core/autonomy/models/signals.py`) — removed `FEATURE_DISABLED` from `_BLOCKED_RUNTIME_ERROR_CODES` and its recommended action handler
- **Server handler** (`tools/unified/server.py`) — removed `FEATURE_DISABLED` from `fail_fast_on`, updated `get_capabilities()` call
- **Context helpers** (`tools/unified/context_helpers.py`) — updated `get_capabilities()` call
- **Skill** (`skills/foundry_implement_v2.py`) — removed `_read_runtime_autonomy_flags()`, removed flag checks from preflight, removed `autonomy_sessions_enabled`/`autonomy_fidelity_gates_enabled` from `StartupPreflightResult`
- **Comments** (`task_handlers/__init__.py`) — removed "feature-flag guarded" comments

## What Remains — Test Updates

The following test files reference the removed feature flag infrastructure and need updating:

### Tests to DELETE (test entire feature flag system that no longer exists)

1. **`tests/unit/test_config_hierarchy.py`** — Remove these test methods:
   - `test_feature_flags_loaded_from_toml` (line ~572)
   - `test_feature_flags_env_overrides_toml` (line ~597)
   - `test_feature_flag_per_flag_env_overrides_bulk_env` (line ~628)
   - `test_feature_flag_dependency_error` (line ~654)
   - `test_feature_flag_dependency_satisfied` (line ~676)

2. **`tests/unit/test_core/test_discovery.py`** — Update/remove:
   - `test_runtime_feature_flags_override_defaults` (line ~374) — DELETE (no more feature_flags param)
   - `test_runtime_dependency_warnings_for_inconsistent_flags` — DELETE
   - Update `test_server_capabilities_defaults` — remove `feature_flags_enabled` assertion, update `autonomy_sessions`/`autonomy_fidelity_gates` to expect `True`
   - Update `test_server_capabilities_to_dict` — remove `feature_flags` key assertion

3. **`tests/unit/test_review.py`** — Remove:
   - `test_fidelity_gate_uses_runtime_feature_flag` (line ~77) — DELETE entire test

4. **`tests/tools/unified/test_task_action_shapes.py`** — Remove:
   - `test_dispatch_session_step_feature_disabled_includes_loop_signal` (line ~206) — DELETE
   - `test_dispatch_legacy_action_feature_disabled_includes_deprecation_metadata` (line ~233) — DELETE

5. **`tests/tools/unified/test_server.py`** — Update:
   - `test_capabilities_reflect_runtime_feature_flags` (line ~87) — DELETE or rewrite (no more feature_flags param)
   - Remove `feature_flags={...}` from any `ServerConfig()` constructors (lines ~92, ~114, ~134, ~148)

6. **`tests/unit/test_skills/test_foundry_implement_v2.py`** — Update:
   - Remove test asserting `FEATURE_DISABLED` error code (line ~145)
   - Remove assertions on `autonomy_sessions_enabled` / `autonomy_fidelity_gates_enabled` (lines ~189-190)

7. **`tests/tools/unified/test_research.py`** — Update:
   - `test_feature_flag_error_response_format` (line ~692) — update or DELETE (references `ErrorCode.FEATURE_DISABLED`)
   - `test_dispatch_without_feature_flag_check` (line ~719) — update name/assertions

### Tests to UPDATE (remove `feature_flags` from fixtures)

8. **`tests/unit/test_core/autonomy/conftest.py`** — Remove `config.feature_flags = {"autonomy_sessions": True}` from fixture (line ~131)

9. **`tests/unit/test_core/autonomy/test_handlers_session_step.py`** — Remove feature_flags fixture setup (line ~63)

10. **`tests/unit/test_core/autonomy/test_handlers_session.py`** — Remove feature_flags fixture setup (line ~55)

11. **`tests/unit/test_core/autonomy/test_error_paths.py`** — Remove feature_flags fixture setup (line ~50)

12. **`tests/unit/test_core/autonomy/test_integration.py`** — Remove feature_flags fixture setup (line ~90)

13. **`tests/unit/test_core/autonomy/test_concurrency_scale.py`** — Remove feature_flags fixture setup (line ~33)

### Contract schema update

14. **`tests/contract/response_schema.json`** — Remove `FEATURE_DISABLED` from ErrorCode enum (line ~169) and `feature_flag` from ErrorType enum (line ~184)

### Response test updates

15. **`tests/test_responses.py`** — Remove:
    - `assert ErrorCode.FEATURE_DISABLED.value == "FEATURE_DISABLED"` (line ~338)
    - `assert ErrorType.FEATURE_FLAG.value == "feature_flag"` (line ~377)

### Integration tests

16. **`tests/integration/test_foundry_implement_v2_unattended.py`** — Remove `feature_flags={...}` from config (line ~80)

17. **`tests/integration/test_environment_tools.py`** — These tests reference metadata `ENVIRONMENT_FEATURE_FLAGS` which still exist, so they should be fine. Just verify they pass.

## Kept Intentionally

- **`FeatureFlagDescriptor`** class in `core/discovery/flags.py` — used by metadata modules
- **`ENVIRONMENT_FEATURE_FLAGS`**, **`LLM_FEATURE_FLAGS`**, **`PROVIDER_FEATURE_FLAGS`** — informational discovery metadata, not runtime gates
