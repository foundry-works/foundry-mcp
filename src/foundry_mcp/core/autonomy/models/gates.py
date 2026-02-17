"""Gate-related models for phase gate evaluation and evidence tracking."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .enums import GateVerdict, OverrideReasonCode, PhaseGateStatus


class PendingGateEvidence(BaseModel):
    """Pending gate evidence awaiting consumption by next command."""

    gate_attempt_id: str = Field(..., description="Unique gate attempt identifier")
    step_id: str = Field(..., description="Step ID this evidence is bound to")
    phase_id: str = Field(..., description="Phase ID this gate is for")
    verdict: GateVerdict = Field(..., description="Gate verdict from review")
    issued_at: datetime = Field(..., description="When this evidence was created")
    integrity_checksum: Optional[str] = Field(
        None,
        description="HMAC-SHA256 integrity checksum keyed by server secret (gate_attempt_id + step_id + phase_id + verdict)",
    )


class PendingManualGateAck(BaseModel):
    """Pending manual gate acknowledgment required for resume."""

    gate_attempt_id: str = Field(..., description="Gate attempt that requires acknowledgment")
    phase_id: str = Field(..., description="Phase ID for this gate")
    issued_at: datetime = Field(..., description="When this acknowledgment was created")


class PhaseGateRecord(BaseModel):
    """Record of a phase gate evaluation."""

    required: bool = Field(default=True, description="Whether gate is required")
    status: PhaseGateStatus = Field(default=PhaseGateStatus.PENDING, description="Gate status")
    verdict: Optional[GateVerdict] = Field(None, description="Gate verdict")
    gate_attempt_id: Optional[str] = Field(None, description="Gate attempt ID")
    review_path: Optional[str] = Field(None, description="Path to review file")
    evaluated_at: Optional[datetime] = Field(None, description="When gate was evaluated")
    # Waiver metadata (set when status is WAIVED)
    waiver_reason_code: Optional[OverrideReasonCode] = Field(
        None, description="Structured reason code for gate waiver"
    )
    waiver_reason_detail: Optional[str] = Field(
        None, description="Free-text detail for waiver reason"
    )
    waived_at: Optional[datetime] = Field(None, description="When gate was waived")
    waived_by_role: Optional[str] = Field(None, description="Role that waived the gate")
