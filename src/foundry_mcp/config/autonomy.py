"""Autonomy configuration dataclasses and posture defaults.

Contains configuration for autonomy security controls, session defaults,
and posture profile management (unattended, supervised, debug).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AutonomySecurityConfig:
    """Configuration for autonomy session security controls.

    These settings control hardening behaviors for autonomous execution,
    including write-lock bypass restrictions and authorization controls.

    Attributes:
        allow_lock_bypass: Whether bypass_autonomy_lock=true is permitted.
            When False (default), bypass attempts are rejected regardless of
            caller input. This provides fail-closed protection against
            unauthorized lock bypasses.
        allow_gate_waiver: Whether privileged gate waiver is permitted.
            When False (default), waiver attempts are rejected regardless of
            role. This provides fail-closed protection against gate bypasses.
            Even when True, only maintainer role can waive gates.
        enforce_required_phase_gates: Whether required phase gates are enforced.
            When True (default), transitions are blocked if required gates
            are not satisfied. This provides fail-closed protection against
            skipping quality checkpoints.
        role: Server role for authorization decisions. Determines what
            actions are allowed. Can be overridden by FOUNDRY_MCP_ROLE env var.
            Default: "maintainer" (full interactive access; autonomous
            sessions use posture-driven role override).
        rate_limit_max_consecutive_denials: Maximum consecutive authorization
            denials before rate limiting kicks in. Default: 10.
        rate_limit_denial_window_seconds: Sliding window in seconds for
            counting consecutive denials. Default: 60.
        rate_limit_retry_after_seconds: Seconds to wait before allowing
            retries after rate limit is triggered. Default: 5.
    """

    allow_lock_bypass: bool = False
    allow_gate_waiver: bool = False
    enforce_required_phase_gates: bool = True
    role: str = "maintainer"
    rate_limit_max_consecutive_denials: int = 10
    rate_limit_denial_window_seconds: int = 60
    rate_limit_retry_after_seconds: int = 5


@dataclass
class AutonomySessionDefaultsConfig:
    """Default session-start configuration for autonomous runs."""

    gate_policy: str = "strict"
    stop_on_phase_completion: bool = False
    auto_retry_fidelity_gate: bool = True
    max_tasks_per_session: int = 100
    max_consecutive_errors: int = 3
    max_fidelity_review_cycles_per_phase: int = 3


@dataclass
class AutonomyPostureConfig:
    """Operator-facing posture profile selection for autonomy controls."""

    profile: Optional[str] = None


# ---------------------------------------------------------------------------
# Posture constants and env var names
# ---------------------------------------------------------------------------

_AUTONOMY_POSTURE_ENV_VAR = "FOUNDRY_MCP_AUTONOMY_POSTURE"
_AUTONOMY_DEFAULT_GATE_POLICY_ENV_VAR = "FOUNDRY_MCP_AUTONOMY_DEFAULT_GATE_POLICY"
_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION_ENV_VAR = (
    "FOUNDRY_MCP_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION"
)
_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE_ENV_VAR = (
    "FOUNDRY_MCP_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE"
)
_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION_ENV_VAR = (
    "FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION"
)
_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS_ENV_VAR = (
    "FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS"
)
_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_ENV_VAR = (
    "FOUNDRY_MCP_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_PER_PHASE"
)
_VALID_AUTONOMY_POSTURES = frozenset({"unattended", "supervised", "debug"})
_VALID_GATE_POLICIES = frozenset({"strict", "lenient", "manual"})
_AUTONOMY_POSTURE_DEFAULTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "unattended": {
        "autonomy_security": {
            "role": "autonomy_runner",
            "allow_lock_bypass": False,
            "allow_gate_waiver": False,
            "enforce_required_phase_gates": True,
        },
        "autonomy_session_defaults": {
            "gate_policy": "strict",
            "stop_on_phase_completion": True,
            "auto_retry_fidelity_gate": True,
            "max_tasks_per_session": 100,
            "max_consecutive_errors": 3,
            "max_fidelity_review_cycles_per_phase": 3,
        },
    },
    "supervised": {
        "autonomy_security": {
            "role": "maintainer",
            "allow_lock_bypass": True,
            "allow_gate_waiver": True,
            "enforce_required_phase_gates": True,
        },
        "autonomy_session_defaults": {
            "gate_policy": "strict",
            "stop_on_phase_completion": True,
            "auto_retry_fidelity_gate": True,
            "max_tasks_per_session": 100,
            "max_consecutive_errors": 5,
            "max_fidelity_review_cycles_per_phase": 3,
        },
    },
    "debug": {
        "autonomy_security": {
            "role": "maintainer",
            "allow_lock_bypass": True,
            "allow_gate_waiver": True,
            "enforce_required_phase_gates": False,
        },
        "autonomy_session_defaults": {
            "gate_policy": "manual",
            "stop_on_phase_completion": False,
            "auto_retry_fidelity_gate": False,
            "max_tasks_per_session": 250,
            "max_consecutive_errors": 20,
            "max_fidelity_review_cycles_per_phase": 10,
        },
    },
}
