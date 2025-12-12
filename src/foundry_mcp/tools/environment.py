"""
Environment tools for foundry-mcp.

These wrappers preserve the legacy sdd-* tool surface by delegating to the
unified environment(action=...) router.
"""

from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.tools.unified.environment import (
    _DEFAULT_TOML_CONTENT,
    _init_specs_directory,
    _update_permissions,
    _write_default_toml,
    legacy_environment_action,
)

_MINIMAL_PERMISSIONS = [
    "mcp__foundry-mcp__server",
    "mcp__foundry-mcp__spec",
    "mcp__foundry-mcp__task",
]

_STANDARD_PERMISSIONS = [
    *_MINIMAL_PERMISSIONS,
    "mcp__foundry-mcp__authoring",
    "mcp__foundry-mcp__environment",
    "mcp__foundry-mcp__journal",
    "mcp__foundry-mcp__lifecycle",
    "mcp__foundry-mcp__review",
    "mcp__foundry-mcp__test",
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

__all__ = [
    "register_environment_tools",
    "_DEFAULT_TOML_CONTENT",
    "_MINIMAL_PERMISSIONS",
    "_STANDARD_PERMISSIONS",
    "_FULL_PERMISSIONS",
    "_update_permissions",
    "_write_default_toml",
    "_init_specs_directory",
]


def register_environment_tools(mcp: FastMCP, config: ServerConfig) -> None:
    """Register environment tools with the FastMCP server."""

    @canonical_tool(
        mcp,
        canonical_name="sdd-verify-toolchain",
    )
    def sdd_verify_toolchain(
        include_optional: bool = True,
    ) -> dict:
        """Verify local CLI and toolchain availability."""

        return legacy_environment_action(
            "verify-toolchain",
            config=config,
            include_optional=include_optional,
        )

    @canonical_tool(
        mcp,
        canonical_name="sdd-init-workspace",
    )
    def sdd_init_workspace(
        path: Optional[str] = None,
        create_subdirs: bool = True,
    ) -> dict:
        """Bootstrap working directory for SDD workflows."""

        return legacy_environment_action(
            "init",
            config=config,
            path=path,
            create_subdirs=create_subdirs,
        )

    @canonical_tool(
        mcp,
        canonical_name="sdd-detect-topology",
    )
    def sdd_detect_topology(
        path: Optional[str] = None,
    ) -> dict:
        """Auto-detect repository layout for specs and documentation."""

        return legacy_environment_action(
            "detect",
            config=config,
            path=path,
        )

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
        """Validate OS packages, runtimes, and environment for SDD workflows."""

        return legacy_environment_action(
            "verify-env",
            config=config,
            path=path,
            check_python=check_python,
            check_git=check_git,
            check_node=check_node,
            required_packages=required_packages,
        )

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
        """Initialize a project for SDD workflows."""

        return legacy_environment_action(
            "setup",
            config=config,
            path=path,
            permissions_preset=permissions_preset,
            create_toml=create_toml,
            dry_run=dry_run,
        )
