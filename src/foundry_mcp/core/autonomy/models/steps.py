"""Step-related models for autonomous session execution."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import StepOutcome, StepType
from .verification import VerificationReceipt


class LastStepIssued(BaseModel):
    """Record of the last step issued to the caller."""

    step_id: str = Field(..., description="ULID-format step identifier")
    type: StepType = Field(..., description="Type of step")
    task_id: Optional[str] = Field(None, description="Task ID if step involves a task")
    phase_id: Optional[str] = Field(None, description="Phase ID if step involves a phase")
    issued_at: datetime = Field(..., description="When the step was issued")
    step_proof: Optional[str] = Field(
        None, description="One-time proof token for this step (consumed on report)"
    )


class StepInstruction(BaseModel):
    """A structured instruction for step execution."""

    tool: str = Field(..., description="Tool to use")
    action: str = Field(..., description="Action to perform")
    description: str = Field(..., description="Human-readable description")


class LastStepResult(BaseModel):
    """Result of a step execution reported by the caller."""

    step_id: str = Field(..., description="Step ID being reported")
    step_type: StepType = Field(..., description="Type of step")
    task_id: Optional[str] = Field(None, description="Task ID if step involved a task")
    phase_id: Optional[str] = Field(None, description="Phase ID if step involved a phase")
    outcome: StepOutcome = Field(..., description="Step outcome")
    note: Optional[str] = Field(None, description="Free-text note about outcome")
    files_touched: Optional[List[str]] = Field(None, description="Files modified during step")
    gate_attempt_id: Optional[str] = Field(None, description="Gate attempt ID for gate steps")
    step_proof: Optional[str] = Field(
        None, description="One-time proof token from the issued step (required for proof-enforced steps)"
    )
    verification_receipt: Optional[VerificationReceipt] = Field(
        None,
        description="Server-issued receipt for execute_verification steps (required when outcome='success')",
    )

    @model_validator(mode="after")
    def validate_fields_by_step_type(self) -> "LastStepResult":
        """Validate required fields per step type (ADR section 4)."""
        if self.step_type in (StepType.IMPLEMENT_TASK, StepType.EXECUTE_VERIFICATION):
            if not self.task_id:
                raise ValueError(
                    f"task_id is required for step type {self.step_type.value}"
                )
        if self.step_type == StepType.RUN_FIDELITY_GATE:
            if not self.phase_id:
                raise ValueError(
                    "phase_id is required for step type run_fidelity_gate"
                )
            if not self.gate_attempt_id:
                raise ValueError(
                    "gate_attempt_id is required for step type run_fidelity_gate"
                )
        if self.step_type == StepType.ADDRESS_FIDELITY_FEEDBACK:
            if not self.phase_id:
                raise ValueError(
                    "phase_id is required for step type address_fidelity_feedback"
                )
        return self


class StepProofRecord(BaseModel):
    """Record of a consumed step proof for replay protection.

    Persisted atomically to enable replay-safe exactly-once semantics
    across process restarts.
    """

    step_proof: str = Field(..., description="One-time proof token")
    step_id: str = Field(..., description="Step ID this proof is bound to")
    payload_hash: str = Field(..., description="SHA-256 hash of the original request payload")
    consumed_at: datetime = Field(..., description="When this proof was consumed")
    grace_expires_at: datetime = Field(..., description="When the replay grace window expires")
    response_hash: Optional[str] = Field(
        None, description="SHA-256 hash of the response for replay validation"
    )
    cached_response: Optional[Dict[str, Any]] = Field(
        None, description="Cached response for idempotent replay within grace window"
    )

    @model_validator(mode="after")
    def validate_grace_after_consumed(self) -> "StepProofRecord":
        """Assert grace_expires_at > consumed_at."""
        if self.grace_expires_at <= self.consumed_at:
            raise ValueError(
                f"grace_expires_at ({self.grace_expires_at.isoformat()}) must be "
                f"after consumed_at ({self.consumed_at.isoformat()})"
            )
        return self
