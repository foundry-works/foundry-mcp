"""State migration module for DeepResearchState versioning.

Provides versioned state schema migrations to ensure backwards compatibility
when loading persisted DeepResearchState from older schema versions.

Schema Versions:
    v0: Original schema (implicit, pre-versioning)
    v1: Adds content_fidelity, dropped_content_ids, content_archive_hashes

Migration Strategy:
    - Each version bump has a dedicated migration function
    - Migrations are applied sequentially (v0 -> v1 -> v2 -> ...)
    - Failed migrations trigger recovery with STATE_MIGRATION_RECOVERED warning
    - Recovery creates a valid v1 state with default values

Usage:
    from foundry_mcp.core.research.state_migrations import (
        migrate_state,
        CURRENT_SCHEMA_VERSION,
    )

    # Load raw state dict from storage
    raw_state = load_from_disk()

    # Migrate to current version
    migrated_state, warnings = migrate_state(raw_state)

    # Create DeepResearchState from migrated dict
    state = DeepResearchState(**migrated_state)
"""

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Current schema version for DeepResearchState
CURRENT_SCHEMA_VERSION = 1

# Schema version field name in state dict
SCHEMA_VERSION_KEY = "_schema_version"


# Error class (canonical definition in foundry_mcp.core.errors.storage)
from foundry_mcp.core.errors.storage import MigrationError  # noqa: E402


class MigrationWarning:
    """Structured warning for migration issues.

    Attributes:
        code: Warning code (e.g., STATE_MIGRATION_RECOVERED)
        severity: Warning severity (info, warning, error)
        message: Human-readable warning message
        context: Additional context about the warning
    """

    def __init__(
        self,
        code: str,
        severity: str,
        message: str,
        context: Optional[dict[str, Any]] = None,
    ):
        self.code = code
        self.severity = severity
        self.message = message
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to warning_details format."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


def get_schema_version(state: dict[str, Any]) -> int:
    """Get the schema version from a state dict.

    Args:
        state: Raw state dictionary

    Returns:
        Schema version (0 if not present, indicating pre-versioning state)
    """
    return state.get(SCHEMA_VERSION_KEY, 0)


def set_schema_version(state: dict[str, Any], version: int) -> None:
    """Set the schema version in a state dict.

    Args:
        state: State dictionary to modify
        version: Schema version to set
    """
    state[SCHEMA_VERSION_KEY] = version


# =============================================================================
# Migration Functions
# =============================================================================


def migrate_v0_to_v1(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate state from v0 (pre-versioning) to v1.

    V1 adds content fidelity tracking fields:
    - content_fidelity: Fidelity level of serialized content
    - dropped_content_ids: IDs of content items dropped during serialization
    - content_archive_hashes: Hashes for retrieving archived content

    Args:
        state: State dict at v0 schema

    Returns:
        State dict migrated to v1 schema
    """
    migrated = deepcopy(state)

    # Add content fidelity fields with defaults
    if "content_fidelity" not in migrated:
        migrated["content_fidelity"] = "full"

    if "dropped_content_ids" not in migrated:
        migrated["dropped_content_ids"] = []

    if "content_archive_hashes" not in migrated:
        migrated["content_archive_hashes"] = {}

    # Update schema version
    set_schema_version(migrated, 1)

    logger.debug("Migrated state from v0 to v1: added content fidelity fields")
    return migrated


# Registry of migration functions: (from_version, to_version) -> migration_fn
MIGRATIONS: dict[tuple[int, int], Callable[[dict[str, Any]], dict[str, Any]]] = {
    (0, 1): migrate_v0_to_v1,
}


# =============================================================================
# Main Migration Entry Point
# =============================================================================


def migrate_state(
    state: dict[str, Any],
    target_version: Optional[int] = None,
) -> tuple[dict[str, Any], list[MigrationWarning]]:
    """Migrate a state dict to the target schema version.

    Applies sequential migrations from the state's current version to the
    target version. If any migration fails, attempts recovery by creating
    a valid state with default values and emits STATE_MIGRATION_RECOVERED
    warning.

    Args:
        state: Raw state dictionary (may be any schema version)
        target_version: Target schema version (defaults to CURRENT_SCHEMA_VERSION)

    Returns:
        Tuple of (migrated_state, warnings):
        - migrated_state: State dict at target schema version
        - warnings: List of MigrationWarning objects for any issues

    Raises:
        MigrationError: If migration fails and recovery is not possible
    """
    if target_version is None:
        target_version = CURRENT_SCHEMA_VERSION

    warnings: list[MigrationWarning] = []
    current_version = get_schema_version(state)

    # Already at target version
    if current_version == target_version:
        return state, warnings

    # Validate migration path exists
    if current_version > target_version:
        raise MigrationError(
            f"Cannot downgrade state from v{current_version} to v{target_version}"
        )

    # Apply migrations sequentially
    migrated = deepcopy(state)
    version = current_version

    while version < target_version:
        migration_key = (version, version + 1)

        if migration_key not in MIGRATIONS:
            raise MigrationError(
                f"No migration path from v{version} to v{version + 1}"
            )

        migration_fn = MIGRATIONS[migration_key]

        try:
            migrated = migration_fn(migrated)
            version = get_schema_version(migrated)
            logger.info(f"Successfully migrated state to v{version}")

        except Exception as e:
            # Migration failed - attempt recovery
            logger.warning(
                f"Migration v{version} -> v{version + 1} failed: {e}. "
                "Attempting recovery with defaults."
            )

            try:
                migrated = _recover_state(state, target_version)
                warnings.append(
                    MigrationWarning(
                        code="STATE_MIGRATION_RECOVERED",
                        severity="info",
                        message=f"State recovered from v{current_version} migration failure",
                        context={
                            "original_version": current_version,
                            "target_version": target_version,
                            "failed_at_version": version,
                            "error": str(e),
                            "recovered_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                logger.info(
                    f"State recovery successful: v{current_version} -> v{target_version}"
                )
                return migrated, warnings

            except Exception as recovery_error:
                raise MigrationError(
                    f"Migration failed at v{version} -> v{version + 1} and recovery "
                    f"failed: {recovery_error}"
                ) from e

    return migrated, warnings


def _recover_state(
    state: dict[str, Any],
    target_version: int,
) -> dict[str, Any]:
    """Attempt to recover a state by applying default values.

    Creates a valid state at the target version by:
    1. Preserving all existing valid fields from the original state
    2. Adding missing required fields with safe defaults
    3. Setting the schema version to target

    Args:
        state: Original state that failed migration
        target_version: Target schema version

    Returns:
        Recovered state dict at target version

    Raises:
        MigrationError: If recovery is not possible (e.g., missing essential fields)
    """
    recovered = deepcopy(state)

    # Essential fields that must exist for a valid DeepResearchState
    essential_fields = ["id", "original_query"]

    for field in essential_fields:
        if field not in recovered:
            raise MigrationError(
                f"Cannot recover state: missing essential field '{field}'"
            )

    # Apply all migrations' default values up to target version
    if target_version >= 1:
        # V1 defaults
        if "content_fidelity" not in recovered:
            recovered["content_fidelity"] = "full"
        if "dropped_content_ids" not in recovered:
            recovered["dropped_content_ids"] = []
        if "content_archive_hashes" not in recovered:
            recovered["content_archive_hashes"] = {}

    # Set schema version
    set_schema_version(recovered, target_version)

    return recovered


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_state_version(state: dict[str, Any]) -> tuple[bool, Optional[str]]:
    """Validate that a state dict has a valid schema version.

    Args:
        state: State dictionary to validate

    Returns:
        Tuple of (is_valid, error_message):
        - is_valid: True if version is valid
        - error_message: Description of issue if invalid, None if valid
    """
    version = get_schema_version(state)

    if version < 0:
        return False, f"Invalid schema version: {version} (must be >= 0)"

    if version > CURRENT_SCHEMA_VERSION:
        return False, (
            f"Schema version {version} is newer than current version "
            f"{CURRENT_SCHEMA_VERSION}. Update foundry-mcp to load this state."
        )

    return True, None


def needs_migration(state: dict[str, Any]) -> bool:
    """Check if a state dict needs migration to current version.

    Args:
        state: State dictionary to check

    Returns:
        True if migration is needed, False if already at current version
    """
    return get_schema_version(state) < CURRENT_SCHEMA_VERSION
