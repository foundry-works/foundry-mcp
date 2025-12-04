"""
Environment tools for foundry-mcp.

Provides MCP tools for environment verification, workspace initialization,
and topology detection.
"""

import json
import logging
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.responses import success_response, error_response
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import audit_log, get_metrics
from foundry_mcp.core.resilience import FAST_TIMEOUT

logger = logging.getLogger(__name__)

# Metrics singleton for environment tools
_metrics = get_metrics()

# Permission presets for sdd-setup tool
_MINIMAL_PERMISSIONS = [
    # Read-only spec operations
    "mcp__foundry-mcp__spec-list",
    "mcp__foundry-mcp__spec-get",
    "mcp__foundry-mcp__spec-get-hierarchy",
    "mcp__foundry-mcp__spec-render",
    "mcp__foundry-mcp__spec-stats",
    # Read-only task operations
    "mcp__foundry-mcp__task-list",
    "mcp__foundry-mcp__task-info",
    "mcp__foundry-mcp__task-progress",
    # Server info
    "mcp__foundry-mcp__get-server-context",
    "mcp__foundry-mcp__tool-list",
]

_STANDARD_PERMISSIONS = _MINIMAL_PERMISSIONS + [
    # Environment tools
    "mcp__foundry-mcp__sdd-init-workspace",
    "mcp__foundry-mcp__sdd-verify-environment",
    "mcp__foundry-mcp__sdd-verify-toolchain",
    "mcp__foundry-mcp__sdd-setup",
    # Spec lifecycle
    "mcp__foundry-mcp__spec-create",
    "mcp__foundry-mcp__spec-validate",
    "mcp__foundry-mcp__spec-lifecycle-activate",
    "mcp__foundry-mcp__spec-lifecycle-complete",
    # Task workflow
    "mcp__foundry-mcp__task-start",
    "mcp__foundry-mcp__task-complete",
    "mcp__foundry-mcp__task-update-status",
    # Journal
    "mcp__foundry-mcp__journal-add",
    "mcp__foundry-mcp__journal-list",
    # File patterns for specs
    "Read(//**/specs/**)",
    "Write(//**/specs/active/**)",
    "Write(//**/specs/pending/**)",
    "Edit(//**/specs/active/**)",
    "Edit(//**/specs/pending/**)",
]

_FULL_PERMISSIONS = [
    "mcp__foundry-mcp__*",
    "Read(//**/specs/**)",
    "Write(//**/specs/**)",
    "Edit(//**/specs/**)",
]

# Default TOML content for foundry-mcp.toml
_DEFAULT_TOML_CONTENT = """[workspace]
specs_dir = "./specs"

[workflow]
mode = "single"
auto_validate = true

[logging]
level = "INFO"
"""


def _update_permissions(settings_file: Path, preset: str, dry_run: bool) -> Dict[str, Any]:
    """
    Update .claude/settings.local.json with additive permission merge.

    Args:
        settings_file: Path to settings.local.json
        preset: Permission preset name (minimal, standard, full)
        dry_run: If True, don't write changes

    Returns:
        Dict with changes list
    """
    changes: List[str] = []
    preset_perms = {
        "minimal": _MINIMAL_PERMISSIONS,
        "standard": _STANDARD_PERMISSIONS,
        "full": _FULL_PERMISSIONS,
    }[preset]

    # Load existing or create new
    if settings_file.exists():
        with open(settings_file, "r") as f:
            settings = json.load(f)
    else:
        settings = {"permissions": {"allow": [], "deny": [], "ask": []}}
        changes.append(f"Created {settings_file}")

    # Merge permissions (additive - preserve user's custom entries)
    existing = set(settings.get("permissions", {}).get("allow", []))
    new_perms = set(preset_perms) - existing

    if new_perms:
        settings.setdefault("permissions", {}).setdefault("allow", [])
        settings["permissions"]["allow"].extend(sorted(new_perms))
        changes.append(f"Added {len(new_perms)} permissions to allow list")

    # Ensure MCP server enabled
    settings["enableAllProjectMcpServers"] = True
    if "foundry-mcp" not in settings.get("enabledMcpjsonServers", []):
        settings.setdefault("enabledMcpjsonServers", []).append("foundry-mcp")
        changes.append("Enabled foundry-mcp server")

    if not dry_run and changes:
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_file, "w") as f:
            json.dump(settings, f, indent=2)

    return {"changes": changes}


def _write_default_toml(toml_path: Path) -> None:
    """Write default foundry-mcp.toml configuration file."""
    with open(toml_path, "w") as f:
        f.write(_DEFAULT_TOML_CONTENT)


def _init_specs_directory(base_path: Path, dry_run: bool) -> Dict[str, Any]:
    """
    Initialize specs directory structure.

    Args:
        base_path: Project root path
        dry_run: If True, don't create directories

    Returns:
        Dict with changes list
    """
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
        # Dry run - report what would be created
        if not specs_dir.exists():
            changes.append(f"Would create {specs_dir}")
        for subdir in subdirs:
            subdir_path = specs_dir / subdir
            if not subdir_path.exists():
                changes.append(f"Would create {subdir_path}")

    return {"changes": changes}


def register_environment_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """
    Register environment tools with the FastMCP server.

    Args:
        mcp: FastMCP server instance
        config: Server configuration
    """

    @canonical_tool(
        mcp,
        canonical_name="sdd-verify-toolchain",
    )
    def sdd_verify_toolchain(
        include_optional: bool = True,
    ) -> dict:
        """
        Verify local CLI and toolchain availability.

        Performs a sanity check of required and optional binaries needed
        for SDD workflows. Returns availability status for each tool.

        WHEN TO USE:
        - Before starting a new SDD workflow
        - Diagnosing environment issues
        - Validating CI/CD environment setup
        - Troubleshooting missing dependencies

        Args:
            include_optional: Include optional tools in check (default: True)

        Returns:
            JSON object with tool availability status:
            - required: Dict of required tools and their availability
            - optional: Dict of optional tools and their availability (if requested)
            - all_available: Boolean indicating if all required tools are present
            - missing: List of missing required tools (if any)
        """
        try:
            # Define required and optional tools
            required_tools = ["python", "git"]
            optional_tools = ["grep", "cat", "find", "node", "npm"]

            def check_tool(tool_name: str) -> bool:
                """Check if a tool is available in PATH."""
                return shutil.which(tool_name) is not None

            # Check required tools
            required_status: Dict[str, bool] = {}
            missing_required: List[str] = []
            for tool in required_tools:
                available = check_tool(tool)
                required_status[tool] = available
                if not available:
                    missing_required.append(tool)

            # Check optional tools if requested
            optional_status: Dict[str, bool] = {}
            if include_optional:
                for tool in optional_tools:
                    optional_status[tool] = check_tool(tool)

            # Build response data
            all_available = len(missing_required) == 0
            data: Dict[str, Any] = {
                "required": required_status,
                "all_available": all_available,
            }

            if include_optional:
                data["optional"] = optional_status

            if missing_required:
                data["missing"] = missing_required

            # Add warnings for missing optional tools
            warnings: List[str] = []
            if include_optional:
                missing_optional = [
                    tool for tool, available in optional_status.items() if not available
                ]
                if missing_optional:
                    warnings.append(
                        f"Optional tools not found: {', '.join(missing_optional)}"
                    )

            if not all_available:
                return asdict(
                    error_response(
                        f"Required tools missing: {', '.join(missing_required)}",
                        error_code="MISSING_REQUIRED",
                        error_type="validation",
                        data=data,
                        remediation="Install missing tools before proceeding with SDD workflows.",
                    )
                )

            return asdict(
                success_response(
                    data=data,
                    warnings=warnings if warnings else None,
                )
            )

        except Exception as e:
            logger.exception("Error verifying toolchain")
            return asdict(error_response(f"Failed to verify toolchain: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="sdd-init-workspace",
    )
    def sdd_init_workspace(
        path: Optional[str] = None,
        create_subdirs: bool = True,
    ) -> dict:
        """
        Bootstrap working directory for SDD workflows.

        Initializes the specs/ directory structure needed for SDD workflows.
        Creates the standard folder hierarchy (active, pending, completed, archived).

        WHEN TO USE:
        - Setting up a new project for SDD
        - Onboarding an existing repository to SDD
        - Recreating missing directory structure

        Args:
            path: Project root path (default: current directory)
            create_subdirs: Create standard subdirectories (default: True)

        Returns:
            JSON object with initialization status:
            - success: Boolean indicating if initialization succeeded
            - specs_dir: Path to the specs directory
            - created_dirs: List of directories that were created
            - existing_dirs: List of directories that already existed
        """
        import os
        from pathlib import Path

        try:
            # Determine base path
            base_path = Path(path) if path else Path.cwd()
            specs_dir = base_path / "specs"

            # Standard subdirectories
            subdirs = ["active", "pending", "completed", "archived"]

            created_dirs: List[str] = []
            existing_dirs: List[str] = []

            # Create specs directory if needed
            if not specs_dir.exists():
                specs_dir.mkdir(parents=True)
                created_dirs.append(str(specs_dir))
            else:
                existing_dirs.append(str(specs_dir))

            # Create subdirectories if requested
            if create_subdirs:
                for subdir in subdirs:
                    subdir_path = specs_dir / subdir
                    if not subdir_path.exists():
                        subdir_path.mkdir(parents=True)
                        created_dirs.append(str(subdir_path))
                    else:
                        existing_dirs.append(str(subdir_path))

            data: Dict[str, Any] = {
                "specs_dir": str(specs_dir),
                "active_dir": str(specs_dir / "active"),
            }

            if created_dirs:
                data["created_dirs"] = created_dirs
            if existing_dirs:
                data["existing_dirs"] = existing_dirs

            # Add warning if nothing was created
            warnings: List[str] = []
            if not created_dirs:
                warnings.append("All directories already existed, no changes made")

            # Audit log workspace initialization
            audit_log(
                "workspace_init",
                tool="sdd-init-workspace",
                path=str(base_path),
                created_count=len(created_dirs),
                success=True,
            )
            _metrics.counter("environment.workspace_init", labels={"status": "success"})

            return asdict(
                success_response(
                    data=data,
                    warnings=warnings if warnings else None,
                )
            )

        except PermissionError as e:
            logger.exception("Permission denied during workspace initialization")
            return asdict(
                error_response(
                    f"Permission denied: {e}",
                    error_code="FORBIDDEN",
                    error_type="authorization",
                    remediation="Check write permissions for the target directory.",
                )
            )
        except Exception as e:
            logger.exception("Error initializing workspace")
            return asdict(error_response(f"Failed to initialize workspace: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="sdd-detect-topology",
    )
    def sdd_detect_topology(
        path: Optional[str] = None,
    ) -> dict:
        """
        Auto-detect repository layout for specs and documentation.

        Analyzes the project structure to identify project type, specs directory,
        docs paths, and other relevant topology information.

        WHEN TO USE:
        - Before initializing SDD in an existing project
        - Understanding project structure for tool configuration
        - Diagnosing configuration issues
        - Auto-configuring MCP server settings

        Args:
            path: Directory to analyze (default: current directory)

        Returns:
            JSON object with detected topology:
            - project_type: Detected project type (python, node, rust, etc.)
            - specs_dir: Path to specs directory if found
            - docs_dir: Path to documentation directory if found
            - has_git: Whether the directory is a git repository
            - detected_files: Key configuration files found
        """
        from pathlib import Path
        import os

        try:
            base_path = Path(path) if path else Path.cwd()

            # Detect project type based on marker files
            project_type = "unknown"
            detected_files: List[str] = []

            # Python markers
            python_markers = ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"]
            for marker in python_markers:
                if (base_path / marker).exists():
                    project_type = "python"
                    detected_files.append(marker)
                    break

            # Node markers (only if not already Python)
            if project_type == "unknown":
                node_markers = ["package.json", "yarn.lock", "pnpm-lock.yaml"]
                for marker in node_markers:
                    if (base_path / marker).exists():
                        project_type = "node"
                        detected_files.append(marker)
                        break

            # Rust markers
            if project_type == "unknown":
                if (base_path / "Cargo.toml").exists():
                    project_type = "rust"
                    detected_files.append("Cargo.toml")

            # Go markers
            if project_type == "unknown":
                if (base_path / "go.mod").exists():
                    project_type = "go"
                    detected_files.append("go.mod")

            # Check for specs directory
            specs_dir = None
            specs_candidates = ["specs", ".specs", "specifications"]
            for candidate in specs_candidates:
                candidate_path = base_path / candidate
                if candidate_path.is_dir():
                    specs_dir = str(candidate_path)
                    break

            # Check for docs directory
            docs_dir = None
            docs_candidates = ["docs", "documentation", "doc"]
            for candidate in docs_candidates:
                candidate_path = base_path / candidate
                if candidate_path.is_dir():
                    docs_dir = str(candidate_path)
                    break

            # Check for git
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
                warnings.append("No specs directory found - run sdd-init-workspace to create one")

            return asdict(
                success_response(
                    data=data,
                    warnings=warnings if warnings else None,
                )
            )

        except Exception as e:
            logger.exception("Error detecting topology")
            return asdict(error_response(f"Failed to detect topology: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="sdd-verify-environment",
    )
    def sdd_verify_environment(
        path: Optional[str] = None,
        check_python: bool = True,
        check_git: bool = True,
        check_node: bool = False,
        required_packages: Optional[str] = None,
    ) -> dict:
        """
        Validate OS packages, runtimes, and environment for SDD workflows.

        Performs comprehensive environment validation including runtime versions,
        package availability, and optional credential checks.

        WHEN TO USE:
        - Before starting development work
        - Diagnosing CI/CD environment issues
        - Validating team member setup
        - Pre-flight checks before deployments

        Args:
            path: Directory context for checks (default: current directory)
            check_python: Validate Python runtime (default: True)
            check_git: Validate Git availability (default: True)
            check_node: Validate Node.js runtime (default: False)
            required_packages: Comma-separated list of required Python packages

        Returns:
            JSON object with environment validation results:
            - runtimes: Dict of runtime versions and availability
            - packages: Dict of package installation status (if checked)
            - all_valid: Boolean indicating if all checks passed
            - issues: List of validation issues found
        """
        import sys
        from pathlib import Path

        try:
            base_path = Path(path) if path else Path.cwd()

            runtimes: Dict[str, Any] = {}
            issues: List[str] = []
            packages: Dict[str, bool] = {}

            # Check Python
            if check_python:
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                runtimes["python"] = {
                    "available": True,
                    "version": python_version,
                    "executable": sys.executable,
                }

                # Check minimum version (3.9+)
                if sys.version_info < (3, 9):
                    issues.append(f"Python 3.9+ required, found {python_version}")

            # Check Git
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
                        runtimes["git"] = {
                            "available": True,
                            "version": version_str,
                            "executable": git_path,
                        }
                    except Exception:
                        runtimes["git"] = {"available": True, "version": "unknown"}
                else:
                    runtimes["git"] = {"available": False}
                    issues.append("Git not found in PATH")

            # Check Node.js
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
                        runtimes["node"] = {
                            "available": True,
                            "version": result.stdout.strip(),
                            "executable": node_path,
                        }
                    except Exception:
                        runtimes["node"] = {"available": True, "version": "unknown"}
                else:
                    runtimes["node"] = {"available": False}
                    issues.append("Node.js not found in PATH")

            # Check required packages
            if required_packages:
                pkg_list = [p.strip() for p in required_packages.split(",")]
                for pkg in pkg_list:
                    try:
                        __import__(pkg.replace("-", "_"))
                        packages[pkg] = True
                    except ImportError:
                        packages[pkg] = False
                        issues.append(f"Required package not found: {pkg}")

            all_valid = len(issues) == 0
            data: Dict[str, Any] = {
                "runtimes": runtimes,
                "all_valid": all_valid,
            }

            if packages:
                data["packages"] = packages
            if issues:
                data["issues"] = issues

            if not all_valid:
                return asdict(
                    error_response(
                        f"Environment validation failed: {len(issues)} issue(s) found",
                        error_code="VALIDATION_ERROR",
                        error_type="validation",
                        data=data,
                        remediation="Install missing dependencies and ensure runtimes are properly configured.",
                    )
                )

            return asdict(success_response(data=data))

        except Exception as e:
            logger.exception("Error verifying environment")
            return asdict(error_response(f"Failed to verify environment: {e}"))

    @canonical_tool(
        mcp,
        canonical_name="sdd-setup",
    )
    def sdd_setup(
        path: Optional[str] = None,
        permissions_preset: str = "full",
        create_toml: bool = True,
        dry_run: bool = False,
    ) -> dict:
        """
        Initialize a project for SDD workflows.

        Performs comprehensive setup including workspace directories,
        Claude Code permissions, and configuration files.

        WHEN TO USE:
        - First-time setup of a project for SDD
        - Adding foundry-mcp permissions after plugin installation
        - Onboarding an existing project to spec-driven development

        WHEN NOT TO USE:
        - If you only need to create specs directory (use sdd-init-workspace)
        - If you only need to verify environment (use sdd-verify-environment)

        Args:
            path: Project root path (default: current directory)
            permissions_preset: Permission level - "minimal", "standard", or "full" (default: full)
            create_toml: Create foundry-mcp.toml config file (default: True)
            dry_run: Preview changes without writing files (default: False)

        Returns:
            JSON object with:
            - specs_dir: Path to specs directory created/verified
            - permissions_file: Path to .claude/settings.local.json
            - config_file: Path to foundry-mcp.toml (if created)
            - changes: List of changes made (or would be made if dry_run)
            - warnings: Any non-fatal issues encountered
        """
        try:
            # 1. Input validation
            base_path = Path(path) if path else Path.cwd()
            if not base_path.exists():
                return asdict(
                    error_response(
                        f"Path does not exist: {base_path}",
                        error_code="PATH_NOT_FOUND",
                        error_type="validation",
                        remediation="Provide a valid project directory path",
                    )
                )

            if permissions_preset not in ("minimal", "standard", "full"):
                return asdict(
                    error_response(
                        f"Invalid preset: {permissions_preset}",
                        error_code="INVALID_PRESET",
                        error_type="validation",
                        remediation="Use 'minimal', 'standard', or 'full'",
                    )
                )

            changes: List[str] = []
            warnings: List[str] = []

            # 2. Initialize specs/ directory
            specs_result = _init_specs_directory(base_path, dry_run)
            changes.extend(specs_result["changes"])

            # 3. Create/update .claude/settings.local.json
            claude_dir = base_path / ".claude"
            settings_file = claude_dir / "settings.local.json"
            settings_result = _update_permissions(settings_file, permissions_preset, dry_run)
            changes.extend(settings_result["changes"])

            # 4. Create foundry-mcp.toml (if requested and doesn't exist)
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

            # 5. Audit log and metrics
            audit_log(
                "sdd_setup",
                tool="sdd-setup",
                path=str(base_path),
                preset=permissions_preset,
                dry_run=dry_run,
            )
            _metrics.counter(
                "environment.sdd_setup",
                labels={"preset": permissions_preset, "dry_run": str(dry_run)},
            )

            return asdict(
                success_response(
                    data={
                        "specs_dir": str(base_path / "specs"),
                        "permissions_file": str(settings_file),
                        "config_file": config_file,
                        "changes": changes,
                        "dry_run": dry_run,
                    },
                    warnings=warnings if warnings else None,
                )
            )

        except PermissionError as e:
            logger.exception("Permission denied during setup")
            return asdict(
                error_response(
                    f"Permission denied: {e}",
                    error_code="FORBIDDEN",
                    error_type="authorization",
                    remediation="Check write permissions for the target directory.",
                )
            )
        except Exception as e:
            logger.exception("Error in sdd_setup")
            return asdict(
                error_response(
                    f"Setup failed: {e}",
                    error_code="INTERNAL_ERROR",
                    error_type="internal",
                )
            )
