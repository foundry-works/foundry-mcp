# Plan: Default Role from `observer` to `maintainer`

## Files to Change

### 1. `src/foundry_mcp/config/autonomy.py`
- **Line 33**: Docstring `Default: "observer" (fail-closed to read-only)` → `Default: "maintainer" (full interactive access; autonomous sessions use posture-driven role override)`
- **Line 45**: `role: str = "observer"` → `role: str = "maintainer"`

### 2. `src/foundry_mcp/core/authorization.py`
- **Line 9**: Docstring `- observer: Read-only operations (default)` → `- observer: Read-only operations` and `- maintainer: Full mutation access (wildcard)` → `- maintainer: Full mutation access (wildcard, default)`
- **Line 127**: `_configured_server_role: str = Role.OBSERVER.value` → `Role.MAINTAINER.value`
- **Line 135**: Docstring `(default: "observer")` → `(default: "maintainer")`
- **Line 156**: Warning message keep as-is (still falls back to observer for invalid → actually change to maintainer)
- **Line 160**: `role = Role.OBSERVER.value` → `role = Role.MAINTAINER.value`
- **Line 189**: `configured_role: str = Role.OBSERVER.value` → `Role.MAINTAINER.value`
- **Lines 383-384**: Default branch `Role.OBSERVER.value` → `Role.MAINTAINER.value`, comment `# Default to observer (fail-closed)` → `# Default to maintainer (full interactive access)`
- **Line 364**: Docstring `3. Default: "observer"` → `3. Default: "maintainer"`

### 3. `tests/unit/test_core/test_authorization.py`
- **Line 73** (`TestServerRoleVar.teardown_method`): `set_server_role("observer")` → `set_server_role("maintainer")`
- **Lines 75-76**: Rename `test_default_role_is_observer` → `test_default_role_is_maintainer`, assert `== "maintainer"`
- **Lines 84-86**: Rename `test_set_invalid_role_falls_back_to_observer` → `test_set_invalid_role_falls_back_to_maintainer`, assert `== "maintainer"`
- **Lines 187-190** (`TestInitializeRoleFromConfig.test_default_is_observer`): Rename → `test_default_is_maintainer`, assert `== "maintainer"`
- **Line 432** (`TestRunIsolatedSubprocess.teardown_method`): `set_server_role("observer")` → `set_server_role("maintainer")`
- **Line 483** (`TestValidateRunnerPath.teardown_method`): `set_server_role("observer")` → `set_server_role("maintainer")`

## Verification
1. `pytest tests/unit/test_core/test_authorization.py -v`
2. `pytest tests/ -x`
