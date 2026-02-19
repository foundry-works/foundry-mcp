"""Main autonomous session state model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import (
    FailureReason,
    GatePolicy,
    PauseReason,
    SessionStatus,
)
from .gates import PendingGateEvidence, PendingManualGateAck, PhaseGateRecord
from .session_config import SessionContext, SessionCounters, SessionLimits, StopConditions
from .steps import LastStepIssued
from .verification import PendingVerificationReceipt


class AutonomousSessionState(BaseModel):
    """
    Complete autonomous session state for persistence.

    This model represents the full state of an autonomous execution session,
    designed for file-backed persistence with schema versioning support.

    IMPORTANT: ``schema_version`` uses ``alias="_schema_version"`` so that
    persisted JSON contains the underscore-prefixed key expected by
    ``state_migrations``.  Always serialize with ``by_alias=True`` (the
    overridden ``model_dump`` defaults to this) to avoid breaking migration
    detection.
    """

    # Schema versioning (serialized as _schema_version for ADR compliance)
    schema_version: int = Field(default=3, alias="_schema_version", description="Schema version")

    # Core identifiers
    id: str = Field(..., description="ULID-format session identifier")
    spec_id: str = Field(..., description="Spec ID this session is for")
    idempotency_key: Optional[str] = Field(None, max_length=128, description="Client-provided idempotency key")

    # Spec integrity tracking
    spec_structure_hash: str = Field(..., description="SHA-256 hash of spec structure")
    spec_file_mtime: Optional[float] = Field(None, description="Cached spec file mtime")
    spec_file_size: Optional[int] = Field(None, description="Cached spec file size")

    # Session lifecycle
    status: SessionStatus = Field(default=SessionStatus.RUNNING, description="Session status")
    created_at: datetime = Field(..., description="When session was created")
    updated_at: datetime = Field(..., description="When session was last updated")
    paused_at: Optional[datetime] = Field(None, description="When session was paused")
    pause_reason: Optional[PauseReason] = Field(None, description="Why session is paused")
    failure_reason: Optional[FailureReason] = Field(None, description="Why session failed")

    # Execution state
    active_phase_id: Optional[str] = Field(None, description="Currently active phase")
    last_task_id: Optional[str] = Field(None, description="Last task that was worked on")
    last_step_issued: Optional[LastStepIssued] = Field(None, description="Last step issued to caller")
    last_issued_response: Optional[Dict[str, Any]] = Field(
        None, description="Cached last response for replay-safe exactly-once semantics"
    )

    # Pending state
    pending_gate_evidence: Optional[PendingGateEvidence] = Field(
        None, description="Pending gate evidence awaiting consumption"
    )
    pending_manual_gate_ack: Optional[PendingManualGateAck] = Field(
        None, description="Pending manual gate acknowledgment"
    )
    pending_verification_receipt: Optional[PendingVerificationReceipt] = Field(
        None, description="Pending verification receipt for execute_verification steps"
    )

    # Counters and limits
    counters: SessionCounters = Field(default_factory=SessionCounters, description="Session counters")
    limits: SessionLimits = Field(default_factory=SessionLimits, description="Session limits")
    stop_conditions: StopConditions = Field(default_factory=StopConditions, description="Stop conditions")

    # Configuration
    write_lock_enforced: bool = Field(default=True, description="Whether write lock is enforced")
    gate_policy: GatePolicy = Field(default=GatePolicy.STRICT, description="Gate evaluation policy")

    # Runtime context
    context: SessionContext = Field(default_factory=lambda: SessionContext(), description="Runtime context")

    # Phase gates (phase_id -> PhaseGateRecord)
    phase_gates: Dict[str, PhaseGateRecord] = Field(default_factory=dict, description="Phase gate records")

    # Required gate obligations and satisfaction tracking (v3 schema)
    # Maps phase_id -> list of required gate types (e.g., ["fidelity"])
    required_phase_gates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required gate types per phase (phase_id -> gate_types)",
    )
    # Maps phase_id -> list of satisfied gate types (passed or waived)
    satisfied_gates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Satisfied gate types per phase (phase_id -> gate_types)",
    )

    # Completed tasks
    completed_task_ids: List[str] = Field(default_factory=list, description="IDs of completed tasks")

    # State version for observability
    state_version: int = Field(default=1, ge=1, description="Monotonic state version")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate idempotency key format: alphanumeric, hyphens, underscores only."""
        if v is None:
            return v
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("idempotency_key must contain only alphanumeric characters, hyphens, and underscores")
        return v

    @model_validator(mode="after")
    def validate_satisfied_gates_subset(self) -> "AutonomousSessionState":
        """Verify satisfied_gates is a subset of required_phase_gates per phase."""
        for phase_id, satisfied in self.satisfied_gates.items():
            if phase_id not in self.required_phase_gates:
                raise ValueError(f"satisfied_gates references unknown phase '{phase_id}' not in required_phase_gates")
            required = set(self.required_phase_gates[phase_id])
            satisfied_set = set(satisfied)
            extra = satisfied_set - required
            if extra:
                raise ValueError(
                    f"satisfied_gates for phase '{phase_id}' contains gates "
                    f"not in required_phase_gates: {sorted(extra)}"
                )
        return self

    model_config = {
        "populate_by_name": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
        },
    }

    def model_dump(self, *, by_alias: bool = True, **kwargs: Any) -> Dict[str, Any]:
        """Override to default ``by_alias=True`` for correct _schema_version serialization."""
        return super().model_dump(by_alias=by_alias, **kwargs)
