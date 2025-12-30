"""Unified environment tool with action routing and feature-flag enforcement."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig, _PACKAGE_VERSION
from foundry_mcp.core.context import generate_correlation_id, get_correlation_id
from foundry_mcp.core.feature_flags import FeatureFlag, FlagState, get_flag_service
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics, mcp_tool
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
    to_json,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()
_flag_service = get_flag_service()
try:
    _flag_service.register(
        FeatureFlag(
            name="environment_tools",
            description="Environment readiness and workspace tooling",
            state=FlagState.BETA,
            default_enabled=True,
        )
    )
except ValueError:
    pass

_DEFAULT_TOML_TEMPLATE = """[workspace]
specs_dir = "./specs"

[logging]
level = "INFO"
structured = true

[server]
name = "foundry-mcp"
version = "{version}"

[workflow]
mode = "single"
auto_validate = true
journal_enabled = true

[consultation]
# priority = []  # Appended by setup based on detected providers
default_timeout = 300
max_retries = 2
retry_delay = 5.0
fallback_enabled = true
cache_ttl = 3600

[consultation.workflows.fidelity_review]
min_models = 2
timeout_override = 600.0
default_review_type = "full"

[consultation.workflows.plan_review]
min_models = 2
timeout_override = 180.0
default_review_type = "full"

[consultation.workflows.markdown_plan_review]
min_models = 2
timeout_override = 180.0
default_review_type = "full"
"""


# ---------------------------------------------------------------------------
# Helper functions used by unified surface
# ---------------------------------------------------------------------------


def _update_permissions(
    settings_file: Path, preset: str, dry_run: bool
) -> Dict[str, Any]:
    """Update .claude/settings.local.json with additive permission merge."""

    changes: List[str] = []
    preset_perms = {
        "minimal": [
            "mcp__foundry-mcp__server",
            "mcp__foundry-mcp__spec",
            "mcp__foundry-mcp__task",
        ],
        "standard": [
            "mcp__foundry-mcp__authoring",
            "mcp__foundry-mcp__environment",
            "mcp__foundry-mcp__journal",
            "mcp__foundry-mcp__lifecycle",
            "mcp__foundry-mcp__review",
            "mcp__foundry-mcp__server",
            "mcp__foundry-mcp__spec",
            "mcp__foundry-mcp__task",
            "mcp__foundry-mcp__test",
            "Read(//**/specs/**)",
            "Write(//**/specs/active/**)",
            "Write(//**/specs/pending/**)",
            "Edit(//**/specs/active/**)",
            "Edit(//**/specs/pending/**)",
        ],
        "full": [
            "mcp__foundry-mcp__*",
            "Read(//**/specs/**)",
            "Write(//**/specs/**)",
            "Edit(//**/specs/**)",
        ],
    }[preset]

    if settings_file.exists():
        with open(settings_file, "r") as handle:
            settings = cast(Dict[str, Any], json.load(handle))
    else:
        settings = cast(
            Dict[str, Any], {"permissions": {"allow": [], "deny": [], "ask": []}}
        )
        changes.append(f"Created {settings_file}")

    permissions_cfg = settings.get("permissions")
    if not isinstance(permissions_cfg, dict):
        permissions_cfg = {"allow": [], "deny": [], "ask": []}
        settings["permissions"] = permissions_cfg

    allow_list = permissions_cfg.get("allow")
    if not isinstance(allow_list, list):
        allow_list = []
        permissions_cfg["allow"] = allow_list

    existing = set(allow_list)
    new_perms = set(preset_perms) - existing

    if new_perms:
        allow_list.extend(sorted(new_perms))
        changes.append(f"Added {len(new_perms)} permissions to allow list")

    settings["enableAllProjectMcpServers"] = True
    enabled_servers = settings.get("enabledMcpjsonServers")
    if not isinstance(enabled_servers, list):
        enabled_servers = []
        settings["enabledMcpjsonServers"] = enabled_servers
    if "foundry-mcp" not in enabled_servers:
        enabled_servers.append("foundry-mcp")
        changes.append("Enabled foundry-mcp server")

    if not dry_run and changes:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_file, "w") as handle:
            json.dump(settings, handle, indent=2)

    return {"changes": changes}


def _get_default_toml_content() -> str:
    """Get default TOML content with current package version."""
    return _DEFAULT_TOML_TEMPLATE.format(version=_PACKAGE_VERSION)


def _write_default_toml(toml_path: Path) -> None:
    """Write default foundry-mcp.toml configuration file."""

    with open(toml_path, "w") as handle:
        handle.write(_get_default_toml_content())


def _init_specs_directory(base_path: Path, dry_run: bool) -> Dict[str, Any]:
    """Initialize specs directory structure."""

    specs_dir = base_path / "specs"
    subdirs = ["active", "pending", "completed", "archived"]
    changes: List[str] = []

    if not dry_run:
        if not specs_dir.exists():
            specs_dir.mkdir(parents=True)
            changes.append(f"Created {specs_dir}")
        for subdir in subdirs:
            subdir_path = specs_dir / subdir
            if not subdir_path.exists():
                subdir_path.mkdir(parents=True)
                changes.append(f"Created {subdir_path}")
    else:
        if not specs_dir.exists():
            changes.append(f"Would create {specs_dir}")
        for subdir in subdirs:
            subdir_path = specs_dir / subdir
            if not subdir_path.exists():
                changes.append(f"Would create {subdir_path}")

    return {"changes": changes}


# ---------------------------------------------------------------------------
# Unified action helpers
# ---------------------------------------------------------------------------

_ACTION_SUMMARY = {
    "verify-toolchain": "Validate CLI/toolchain availability",
    "verify-env": "Validate runtimes, packages, and workspace environment",
    "init": "Initialize the standard specs/ workspace structure",
    "detect": "Detect repository topology (project type, specs/docs)",
    "detect-test-runner": "Detect appropriate test runner for the project",
    "setup": "Complete SDD setup with permissions + config",
}


def _metric_name(action: str) -> str:
    return f"environment.{action.replace('-', '_')}"


def _request_id() -> str:
    return get_correlation_id() or generate_correlation_id(prefix="environment")


def _feature_flag_blocked(request_id: str) -> Optional[dict]:
    if _flag_service.is_enabled("environment_tools"):
        return None

    return to_json(
        error_response(
            "Environment tools are disabled by feature flag",
            error_code=ErrorCode.FEATURE_DISABLED,
            error_type=ErrorType.FEATURE_FLAG,
            data={"feature": "environment_tools"},
            remediation="Enable the 'environment_tools' feature flag to call environment actions.",
            request_id=request_id,
        )
    )


def _validation_error(
    *,
    action: str,
    field: str,
    message: str,
    request_id: str,
    remediation: Optional[str] = None,
    code: ErrorCode = ErrorCode.VALIDATION_ERROR,
) -> dict:
    return to_json(
        error_response(
            f"Invalid field '{field}' for environment.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            details={"field": field, "action": f"environment.{action}"},
            request_id=request_id,
        )
    )


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_verify_toolchain(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    include_optional: Optional[bool] = True,
    **_: Any,
) -> dict:
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if include_optional is not None and not isinstance(include_optional, bool):
        return _validation_error(
            action="verify-toolchain",
            field="include_optional",
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    include = True if include_optional is None else include_optional
    metric_key = _metric_name("verify-toolchain")

    try:
        required_tools = ["python", "git"]
        optional_tools = ["grep", "cat", "find", "node", "npm"]

        def check_tool(tool_name: str) -> bool:
            return shutil.which(tool_name) is not None

        required_status: Dict[str, bool] = {}
        missing_required: List[str] = []
        for tool in required_tools:
            available = check_tool(tool)
            required_status[tool] = available
            if not available:
                missing_required.append(tool)

        optional_status: Dict[str, bool] = {}
        if include:
            for tool in optional_tools:
                optional_status[tool] = check_tool(tool)

        data: Dict[str, Any] = {
            "required": required_status,
            "all_available": not missing_required,
        }
        if include:
            data["optional"] = optional_status
        if missing_required:
            data["missing"] = missing_required

        warnings: List[str] = []
        if include:
            missing_optional = [
                tool for tool, available in optional_status.items() if not available
            ]
            if missing_optional:
                warnings.append(
                    f"Optional tools not found: {', '.join(sorted(missing_optional))}"
                )

        if missing_required:
            _metrics.counter(metric_key, labels={"status": "missing_required"})
            return to_json(
                error_response(
                    f"Required tools missing: {', '.join(missing_required)}",
                    error_code=ErrorCode.MISSING_REQUIRED,
                    error_type=ErrorType.VALIDATION,
                    data=data,
                    remediation="Install missing tools before continuing with SDD workflows.",
                    request_id=request_id,
                )
            )

        _metrics.counter(metric_key, labels={"status": "success"})
        return to_json(
            success_response(
                data=data,
                warnings=warnings or None,
                request_id=request_id,
            )
        )
    except Exception:
        logger.exception("Error verifying toolchain")
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                "Failed to verify toolchain",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check PATH configuration and retry",
                request_id=request_id,
            )
        )


def _handle_init_workspace(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    path: Optional[str] = None,
    create_subdirs: bool = True,
    **_: Any,
) -> dict:
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action="init",
            field="path",
            message="Workspace path must be a string",
            request_id=request_id,
        )
    if not isinstance(create_subdirs, bool):
        return _validation_error(
            action="init",
            field="create_subdirs",
            message="Expected a boolean value",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    metric_key = _metric_name("init")
    try:
        base_path = Path(path) if path else Path.cwd()
        specs_dir = base_path / "specs"
        subdirs = ["active", "pending", "completed", "archived"]

        created_dirs: List[str] = []
        existing_dirs: List[str] = []

        if not specs_dir.exists():
            specs_dir.mkdir(parents=True)
            created_dirs.append(str(specs_dir))
        else:
            existing_dirs.append(str(specs_dir))

        if create_subdirs:
            for subdir in subdirs:
                subdir_path = specs_dir / subdir
                if not subdir_path.exists():
                    subdir_path.mkdir(parents=True)
                    created_dirs.append(str(subdir_path))
                else:
                    existing_dirs.append(str(subdir_path))

        warnings: List[str] = []
        if not created_dirs:
            warnings.append("All directories already existed, no changes made")

        audit_log(
            "workspace_init",
            tool="environment.init",
            path=str(base_path),
            created_count=len(created_dirs),
            success=True,
        )
        _metrics.counter(metric_key, labels={"status": "success"})

        data: Dict[str, Any] = {
            "specs_dir": str(specs_dir),
            "active_dir": str(specs_dir / "active"),
            "created_dirs": created_dirs,
            "existing_dirs": existing_dirs,
        }
        return to_json(
            success_response(
                data=data,
                warnings=warnings or None,
                request_id=request_id,
            )
        )
    except PermissionError as exc:
        logger.exception("Permission denied during workspace initialization")
        _metrics.counter(metric_key, labels={"status": "forbidden"})
        return to_json(
            error_response(
                f"Permission denied: {exc}",
                error_code=ErrorCode.FORBIDDEN,
                error_type=ErrorType.AUTHORIZATION,
                remediation="Check write permissions for the target directory.",
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception("Error initializing workspace")
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                f"Failed to initialize workspace: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify the path exists and retry",
                request_id=request_id,
            )
        )


def _handle_detect_topology(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    path: Optional[str] = None,
    **_: Any,
) -> dict:
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action="detect",
            field="path",
            message="Directory path must be a string",
            request_id=request_id,
        )

    metric_key = _metric_name("detect")
    try:
        base_path = Path(path) if path else Path.cwd()

        project_type = "unknown"
        detected_files: List[str] = []

        python_markers = ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"]
        for marker in python_markers:
            if (base_path / marker).exists():
                project_type = "python"
                detected_files.append(marker)
                break

        if project_type == "unknown":
            node_markers = ["package.json", "yarn.lock", "pnpm-lock.yaml"]
            for marker in node_markers:
                if (base_path / marker).exists():
                    project_type = "node"
                    detected_files.append(marker)
                    break

        if project_type == "unknown" and (base_path / "Cargo.toml").exists():
            project_type = "rust"
            detected_files.append("Cargo.toml")

        if project_type == "unknown" and (base_path / "go.mod").exists():
            project_type = "go"
            detected_files.append("go.mod")

        specs_dir = None
        for candidate in ["specs", ".specs", "specifications"]:
            candidate_path = base_path / candidate
            if candidate_path.is_dir():
                specs_dir = str(candidate_path)
                break

        docs_dir = None
        for candidate in ["docs", "documentation", "doc"]:
            candidate_path = base_path / candidate
            if candidate_path.is_dir():
                docs_dir = str(candidate_path)
                break

        has_git = (base_path / ".git").is_dir()

        data: Dict[str, Any] = {
            "project_type": project_type,
            "has_git": has_git,
        }
        if specs_dir:
            data["specs_dir"] = specs_dir
        if docs_dir:
            data["docs_dir"] = docs_dir
        if detected_files:
            data["detected_files"] = detected_files

        warnings: List[str] = []
        if project_type == "unknown":
            warnings.append("Could not detect project type from standard marker files")
        if not specs_dir:
            warnings.append(
                "No specs directory found - run environment(action=init) to create one"
            )

        _metrics.counter(metric_key, labels={"status": "success"})
        return to_json(
            success_response(
                data=data,
                warnings=warnings or None,
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception("Error detecting topology")
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                f"Failed to detect topology: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify the directory exists and retry",
                request_id=request_id,
            )
        )


def _handle_detect_test_runner(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    path: Optional[str] = None,
    **_: Any,
) -> dict:
    """Detect appropriate test runner based on project type and configuration files.

    Returns a structured response with detected runners, confidence levels, and
    a recommended default runner.

    Detection rules:
    - Python: pyproject.toml, setup.py, requirements.txt, Pipfile → pytest
    - Go: go.mod → go
    - Jest: jest.config.* or package.json with "jest" key → jest (precedence over npm)
    - Node: package.json with "test" script → npm
    - Rust: Cargo.toml + Makefile present → make
    """
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action="detect-test-runner",
            field="path",
            message="Directory path must be a string",
            request_id=request_id,
        )

    metric_key = _metric_name("detect-test-runner")
    try:
        base_path = Path(path) if path else Path.cwd()

        detected_runners: List[Dict[str, Any]] = []

        # Python detection (highest precedence for Python projects)
        python_primary = ["pyproject.toml", "setup.py"]
        python_secondary = ["requirements.txt", "Pipfile"]

        for marker in python_primary:
            if (base_path / marker).exists():
                detected_runners.append({
                    "runner_name": "pytest",
                    "project_type": "python",
                    "confidence": "high",
                    "reason": f"{marker} found",
                })
                break
        else:
            # Check secondary markers only if no primary found
            for marker in python_secondary:
                if (base_path / marker).exists():
                    detected_runners.append({
                        "runner_name": "pytest",
                        "project_type": "python",
                        "confidence": "medium",
                        "reason": f"{marker} found",
                    })
                    break

        # Go detection
        if (base_path / "go.mod").exists():
            detected_runners.append({
                "runner_name": "go",
                "project_type": "go",
                "confidence": "high",
                "reason": "go.mod found",
            })

        # Node detection - Jest takes precedence over npm
        jest_configs = [
            "jest.config.js",
            "jest.config.ts",
            "jest.config.mjs",
            "jest.config.cjs",
            "jest.config.json",
        ]

        jest_detected = False
        for jest_config in jest_configs:
            if (base_path / jest_config).exists():
                detected_runners.append({
                    "runner_name": "jest",
                    "project_type": "node",
                    "confidence": "high",
                    "reason": f"{jest_config} found",
                })
                jest_detected = True
                break

        # Check package.json for jest config or test script
        package_json_path = base_path / "package.json"
        if package_json_path.exists():
            try:
                with open(package_json_path, "r") as f:
                    pkg = json.load(f)

                # Jest config in package.json takes precedence
                if not jest_detected and "jest" in pkg:
                    detected_runners.append({
                        "runner_name": "jest",
                        "project_type": "node",
                        "confidence": "high",
                        "reason": "jest key in package.json",
                    })
                    jest_detected = True

                # npm test script (only if jest not already detected)
                if not jest_detected:
                    scripts = pkg.get("scripts", {})
                    if "test" in scripts:
                        detected_runners.append({
                            "runner_name": "npm",
                            "project_type": "node",
                            "confidence": "high",
                            "reason": "test script in package.json",
                        })
            except (json.JSONDecodeError, OSError):
                # If package.json is invalid, skip Node detection
                pass

        # Rust detection - only if BOTH Cargo.toml and Makefile exist
        cargo_exists = (base_path / "Cargo.toml").exists()
        makefile_exists = (base_path / "Makefile").exists() or (
            base_path / "makefile"
        ).exists()

        if cargo_exists and makefile_exists:
            detected_runners.append({
                "runner_name": "make",
                "project_type": "rust",
                "confidence": "medium",
                "reason": "Cargo.toml + Makefile found",
            })

        # Determine recommended default based on precedence order from plan
        # Priority: python (1) > go (2) > jest (3) > npm (4) > make (5)
        precedence_order = ["pytest", "go", "jest", "npm", "make"]
        recommended_default: Optional[str] = None

        for runner_name in precedence_order:
            for runner in detected_runners:
                if runner["runner_name"] == runner_name:
                    recommended_default = runner_name
                    break
            if recommended_default:
                break

        data: Dict[str, Any] = {
            "detected_runners": detected_runners,
            "recommended_default": recommended_default,
        }

        warnings: List[str] = []
        if not detected_runners:
            warnings.append(
                "No test runners detected. Configure [test] section manually in "
                "foundry-mcp.toml if tests are needed."
            )

        _metrics.counter(metric_key, labels={"status": "success"})
        return to_json(
            success_response(
                data=data,
                warnings=warnings or None,
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception("Error detecting test runner")
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                f"Failed to detect test runner: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify the directory exists and retry",
                request_id=request_id,
            )
        )


def _handle_verify_environment(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    path: Optional[str] = None,
    check_python: bool = True,
    check_git: bool = True,
    check_node: bool = False,
    required_packages: Optional[str] = None,
    **_: Any,
) -> dict:
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action="verify-env",
            field="path",
            message="Directory path must be a string",
            request_id=request_id,
        )
    for field_name, value in (
        ("check_python", check_python),
        ("check_git", check_git),
        ("check_node", check_node),
    ):
        if not isinstance(value, bool):
            return _validation_error(
                action="verify-env",
                field=field_name,
                message="Expected a boolean value",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    if required_packages is not None and not isinstance(required_packages, str):
        return _validation_error(
            action="verify-env",
            field="required_packages",
            message="Provide a comma-separated string",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )

    metric_key = _metric_name("verify-env")
    try:
        Path(path) if path else Path.cwd()

        runtimes: Dict[str, Any] = {}
        issues: List[str] = []
        packages: Dict[str, bool] = {}

        if check_python:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            runtimes["python"] = {
                "available": True,
                "version": python_version,
                "executable": sys.executable,
            }
            if sys.version_info < (3, 9):
                issues.append(f"Python 3.9+ required, found {python_version}")

        if check_git:
            git_path = shutil.which("git")
            if git_path:
                try:
                    result = subprocess.run(
                        ["git", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    version_str = result.stdout.strip().replace("git version ", "")
                except Exception:
                    version_str = "unknown"
                runtimes["git"] = {
                    "available": True,
                    "version": version_str,
                    "executable": git_path,
                }
            else:
                runtimes["git"] = {"available": False}
                issues.append("Git not found in PATH")

        if check_node:
            node_path = shutil.which("node")
            if node_path:
                try:
                    result = subprocess.run(
                        ["node", "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    node_version = result.stdout.strip()
                except Exception:
                    node_version = "unknown"
                runtimes["node"] = {
                    "available": True,
                    "version": node_version,
                    "executable": node_path,
                }
            else:
                runtimes["node"] = {"available": False}
                issues.append("Node.js not found in PATH")

        if required_packages:
            pkg_list = [
                pkg.strip() for pkg in required_packages.split(",") if pkg.strip()
            ]
            for pkg in pkg_list:
                try:
                    __import__(pkg.replace("-", "_"))
                    packages[pkg] = True
                except ImportError:
                    packages[pkg] = False
                    issues.append(f"Required package not found: {pkg}")

        all_valid = not issues
        data: Dict[str, Any] = {"runtimes": runtimes, "all_valid": all_valid}
        if packages:
            data["packages"] = packages
        if issues:
            data["issues"] = issues

        if not all_valid:
            _metrics.counter(metric_key, labels={"status": "invalid"})
            return to_json(
                error_response(
                    f"Environment validation failed: {len(issues)} issue(s) found",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    error_type=ErrorType.VALIDATION,
                    data=data,
                    remediation="Resolve the listed issues and retry the validation.",
                    request_id=request_id,
                )
            )

        _metrics.counter(metric_key, labels={"status": "success"})
        return to_json(
            success_response(
                data=data,
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception("Error verifying environment", extra={"path": path})
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                f"Failed to verify environment: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check system configuration and retry",
                request_id=request_id,
            )
        )


def _handle_setup(
    *,
    config: ServerConfig,  # noqa: ARG001 - reserved for future hooks
    path: Optional[str] = None,
    permissions_preset: str = "full",
    create_toml: bool = True,
    dry_run: bool = False,
    **_: Any,
) -> dict:
    request_id = _request_id()
    blocked = _feature_flag_blocked(request_id)
    if blocked:
        return blocked

    if path is not None and not isinstance(path, str):
        return _validation_error(
            action="setup",
            field="path",
            message="Project path must be a string",
            request_id=request_id,
        )
    if permissions_preset not in {"minimal", "standard", "full"}:
        return _validation_error(
            action="setup",
            field="permissions_preset",
            message="Invalid preset. Use 'minimal', 'standard', or 'full'",
            request_id=request_id,
            code=ErrorCode.INVALID_FORMAT,
        )
    for field_name, value in (("create_toml", create_toml), ("dry_run", dry_run)):
        if not isinstance(value, bool):
            return _validation_error(
                action="setup",
                field=field_name,
                message="Expected a boolean value",
                request_id=request_id,
                code=ErrorCode.INVALID_FORMAT,
            )

    metric_key = _metric_name("setup")
    try:
        base_path = Path(path) if path else Path.cwd()
        if not base_path.exists():
            return to_json(
                error_response(
                    f"Path does not exist: {base_path}",
                    error_code=ErrorCode.NOT_FOUND,
                    error_type=ErrorType.NOT_FOUND,
                    remediation="Provide a valid project directory path",
                    request_id=request_id,
                )
            )

        changes: List[str] = []
        warnings: List[str] = []

        specs_result = _init_specs_directory(base_path, dry_run)
        changes.extend(specs_result["changes"])

        claude_dir = base_path / ".claude"
        settings_file = claude_dir / "settings.local.json"
        settings_result = _update_permissions(
            settings_file, permissions_preset, dry_run
        )
        changes.extend(settings_result["changes"])

        config_file = None
        if create_toml:
            toml_path = base_path / "foundry-mcp.toml"
            if not toml_path.exists():
                config_file = str(toml_path)
                if not dry_run:
                    _write_default_toml(toml_path)
                changes.append(f"Created {toml_path}")
            else:
                warnings.append("foundry-mcp.toml already exists, skipping")

        audit_log(
            "sdd_setup",
            tool="environment.setup",
            path=str(base_path),
            preset=permissions_preset,
            dry_run=dry_run,
        )
        _metrics.counter(
            metric_key,
            labels={
                "status": "success",
                "preset": permissions_preset,
                "dry_run": str(dry_run),
            },
        )

        return to_json(
            success_response(
                data={
                    "specs_dir": str(base_path / "specs"),
                    "permissions_file": str(settings_file),
                    "config_file": config_file,
                    "changes": changes,
                    "dry_run": dry_run,
                },
                warnings=warnings or None,
                request_id=request_id,
            )
        )
    except PermissionError as exc:
        logger.exception("Permission denied during environment setup")
        _metrics.counter(metric_key, labels={"status": "forbidden"})
        return to_json(
            error_response(
                f"Permission denied: {exc}",
                error_code=ErrorCode.FORBIDDEN,
                error_type=ErrorType.AUTHORIZATION,
                remediation="Check write permissions for the target directory.",
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception("Error in environment setup")
        _metrics.counter(metric_key, labels={"status": "error"})
        return to_json(
            error_response(
                f"Setup failed: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect the logged errors and retry",
                request_id=request_id,
            )
        )


_ENVIRONMENT_ROUTER = ActionRouter(
    tool_name="environment",
    actions=[
        ActionDefinition(
            name="verify-toolchain",
            handler=_handle_verify_toolchain,
            summary=_ACTION_SUMMARY["verify-toolchain"],
            aliases=(
                "verify_toolchain",
                "sdd-verify-toolchain",
                "sdd_verify_toolchain",
            ),
        ),
        ActionDefinition(
            name="verify-env",
            handler=_handle_verify_environment,
            summary=_ACTION_SUMMARY["verify-env"],
            aliases=("verify_env", "sdd-verify-environment", "sdd_verify_environment"),
        ),
        ActionDefinition(
            name="init",
            handler=_handle_init_workspace,
            summary=_ACTION_SUMMARY["init"],
            aliases=("sdd-init-workspace", "sdd_init_workspace"),
        ),
        ActionDefinition(
            name="detect",
            handler=_handle_detect_topology,
            summary=_ACTION_SUMMARY["detect"],
            aliases=("sdd-detect-topology", "sdd_detect_topology"),
        ),
        ActionDefinition(
            name="detect-test-runner",
            handler=_handle_detect_test_runner,
            summary=_ACTION_SUMMARY["detect-test-runner"],
            aliases=(
                "detect_test_runner",
                "sdd-detect-test-runner",
                "sdd_detect_test_runner",
            ),
        ),
        ActionDefinition(
            name="setup",
            handler=_handle_setup,
            summary=_ACTION_SUMMARY["setup"],
            aliases=("sdd-setup", "sdd_setup"),
        ),
    ],
)


def _dispatch_environment_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _ENVIRONMENT_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        request_id = _request_id()
        allowed = ", ".join(exc.allowed_actions)
        return to_json(
            error_response(
                f"Unsupported environment action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
                request_id=request_id,
            )
        )


def register_unified_environment_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated environment tool."""

    @canonical_tool(mcp, canonical_name="environment")
    @mcp_tool(tool_name="environment", emit_metrics=True, audit=True)
    def environment(  # noqa: PLR0913 - composite signature spanning actions
        action: str,
        path: Optional[str] = None,
        include_optional: Optional[bool] = True,
        create_subdirs: bool = True,
        check_python: bool = True,
        check_git: bool = True,
        check_node: bool = False,
        required_packages: Optional[str] = None,
        permissions_preset: str = "full",
        create_toml: bool = True,
        dry_run: bool = False,
    ) -> dict:
        payload = {
            "path": path,
            "include_optional": include_optional,
            "create_subdirs": create_subdirs,
            "check_python": check_python,
            "check_git": check_git,
            "check_node": check_node,
            "required_packages": required_packages,
            "permissions_preset": permissions_preset,
            "create_toml": create_toml,
            "dry_run": dry_run,
        }
        return _dispatch_environment_action(
            action=action, payload=payload, config=config
        )

    logger.debug("Registered unified environment tool")


__all__ = [
    "register_unified_environment_tool",
]
