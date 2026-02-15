"""State migration module for AutonomousSessionState versioning.

Provides versioned state schema migrations to ensure backwards compatibility
when loading persisted AutonomousSessionState from older schema versions.

Schema Versions:
    v1: Initial schema version (current)

Migration Strategy:
    - Each version bump has a dedicated migration function
    - Migrations are applied sequentially (v0 -> v1 -> v2 -> ...)
    - Failed migrations trigger recovery with STATE_MIGRATION_RECOVERED warning
    - Recovery creates a valid state with default values

Usage:
    from foundry_mcp.core.autonomy.state_migrations import (
        migrate_state,
        CURRENT_SCHEMA_VERSION,
    )

    # Load raw state dict from storage
    raw_state = load_from_disk()

    # Migrate to current version
    migrated_state, warnings = migrate_state(raw_state)

    # Create AutonomousSessionState from migrated dict
    state = AutonomousSessionState.model_validate(migrated_state)
"""

import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Current schema version for AutonomousSessionState
CURRENT_SCHEMA_VERSION = 3

# Schema version field name in state dict (matches ADR-002 specification)
SCHEMA_VERSION_KEY = "_schema_version"


class MigrationError(Exception):
    """Raised when a state migration fails."""

    pass


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
        Schema version (1 if not present, since v1 is the initial version)
    """
    return state.get(SCHEMA_VERSION_KEY, 1)


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

# Registry of migration functions: (from_version, to_version) -> migration_fn
MIGRATIONS: dict[tuple[int, int], Callable[[dict[str, Any]], dict[str, Any]]] = {}


def migrate_v1_to_v2(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate state from v1 to v2.

    Adds context tracking fields to SessionContext and new limit fields
    to SessionLimits for robust context usage tracking.
    """
    migrated = deepcopy(state)

    # Add new context tracking fields
    context = migrated.setdefault("context", {})
    context.setdefault("context_source", None)
    context.setdefault("last_context_report_at", None)
    context.setdefault("last_context_report_pct", None)
    context.setdefault("consecutive_same_reports", 0)
    context.setdefault("steps_since_last_report", 0)

    # Add new limit fields
    limits = migrated.setdefault("limits", {})
    limits.setdefault("avg_pct_per_step", 3)
    limits.setdefault("context_staleness_threshold", 5)
    limits.setdefault("context_staleness_penalty_pct", 5)

    set_schema_version(migrated, 2)
    return migrated


MIGRATIONS[(1, 2)] = migrate_v1_to_v2


def migrate_v2_to_v3(state: dict[str, Any]) -> dict[str, Any]:
    """Migrate state from v2 to v3.

    Adds required_phase_gates and satisfied_gates fields for gate obligation
    and satisfaction tracking. Defaults to one fidelity gate required per
    existing phase (based on phase_gates keys), with no gates satisfied.
    """
    migrated = deepcopy(state)

    # Get existing phase IDs from phase_gates
    existing_phases = set(migrated.get("phase_gates", {}).keys())

    # Also check active_phase_id
    active_phase = migrated.get("active_phase_id")
    if active_phase:
        existing_phases.add(active_phase)

    # Initialize required_phase_gates with default fidelity gate per phase
    required_gates = migrated.setdefault("required_phase_gates", {})
    for phase_id in existing_phases:
        # Default: one fidelity gate required per phase
        if phase_id not in required_gates:
            required_gates[phase_id] = ["fidelity"]

    # Initialize satisfied_gates as empty (no gates satisfied yet)
    migrated.setdefault("satisfied_gates", {})

    set_schema_version(migrated, 3)
    return migrated


MIGRATIONS[(2, 3)] = migrate_v2_to_v3


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
            logger.info(f"Successfully migrated autonomous session state to v{version}")

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

    # Essential fields that must exist for a valid AutonomousSessionState
    essential_fields = ["id", "spec_id", "spec_structure_hash", "created_at", "updated_at"]

    for field in essential_fields:
        if field not in recovered:
            raise MigrationError(
                f"Cannot recover state: missing essential field '{field}'"
            )

    # Apply default values for v1 fields if missing
    if target_version >= 1:
        # V1 defaults - only add if missing
        if "status" not in recovered:
            recovered["status"] = "paused"  # Safe default for recovered state
        if "counters" not in recovered:
            recovered["counters"] = {
                "tasks_completed": 0,
                "consecutive_errors": 0,
                "fidelity_review_cycles_in_active_phase": 0,
            }
        if "limits" not in recovered:
            recovered["limits"] = {
                "max_tasks_per_session": 100,
                "max_consecutive_errors": 3,
                "context_threshold_pct": 85,
                "heartbeat_stale_minutes": 10,
                "heartbeat_grace_minutes": 5,
                "step_stale_minutes": 60,
                "max_fidelity_review_cycles_per_phase": 3,
            }
        if "stop_conditions" not in recovered:
            recovered["stop_conditions"] = {
                "stop_on_phase_completion": False,
                "auto_retry_fidelity_gate": True,
            }
        if "context" not in recovered:
            recovered["context"] = {
                "context_usage_pct": 0,
            }
        if "phase_gates" not in recovered:
            recovered["phase_gates"] = {}
        if "completed_task_ids" not in recovered:
            recovered["completed_task_ids"] = []
        if "state_version" not in recovered:
            recovered["state_version"] = 1
        if "write_lock_enforced" not in recovered:
            recovered["write_lock_enforced"] = True
        if "gate_policy" not in recovered:
            recovered["gate_policy"] = "strict"

    # Apply v2 defaults
    if target_version >= 2:
        context = recovered.setdefault("context", {})
        context.setdefault("context_source", None)
        context.setdefault("last_context_report_at", None)
        context.setdefault("last_context_report_pct", None)
        context.setdefault("consecutive_same_reports", 0)
        context.setdefault("steps_since_last_report", 0)

        limits = recovered.setdefault("limits", {})
        limits.setdefault("avg_pct_per_step", 3)
        limits.setdefault("context_staleness_threshold", 5)
        limits.setdefault("context_staleness_penalty_pct", 5)

    # Apply v3 defaults
    if target_version >= 3:
        # Get existing phase IDs to populate required_phase_gates defaults
        existing_phases = set(recovered.get("phase_gates", {}).keys())
        active_phase = recovered.get("active_phase_id")
        if active_phase:
            existing_phases.add(active_phase)

        # Initialize required_phase_gates with fidelity gate per phase
        required_gates = recovered.setdefault("required_phase_gates", {})
        for phase_id in existing_phases:
            if phase_id not in required_gates:
                required_gates[phase_id] = ["fidelity"]

        # Initialize satisfied_gates as empty
        recovered.setdefault("satisfied_gates", {})

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

    if version < 1:
        return False, f"Invalid schema version: {version} (must be >= 1)"

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
