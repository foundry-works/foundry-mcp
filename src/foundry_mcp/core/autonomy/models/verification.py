"""Verification receipt models for proof-carrying verification (P1.2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class VerificationReceipt(BaseModel):
    """Server-issued verification receipt for execute_verification steps.

    Contains cryptographic hashes of command, exit code, and output to enable
    tamper-evident verification of test execution results. The receipt is
    issued by the server when verification completes and must be included
    in last_step_result when reporting outcome='success'.
    """

    command_hash: str = Field(
        ..., description="SHA-256 hash of the verification command executed"
    )
    exit_code: int = Field(..., description="Exit code from the verification command")
    output_digest: str = Field(
        ..., description="SHA-256 hash of the combined stdout/stderr output"
    )
    issued_at: datetime = Field(..., description="When this receipt was issued")
    step_id: str = Field(..., description="Step ID this receipt is bound to")

    @field_validator("command_hash", "output_digest")
    @classmethod
    def validate_sha256_hex(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("must be a 64-character lowercase hex sha256 digest")
        return normalized

    @field_validator("step_id")
    @classmethod
    def validate_step_id_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("step_id must be non-empty")
        return normalized

    @field_validator("issued_at")
    @classmethod
    def validate_issued_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("issued_at must include timezone information")
        return value


class PendingVerificationReceipt(BaseModel):
    """Pending verification receipt data stored in session state.

    When the orchestrator issues an EXECUTE_VERIFICATION step, it stores
    the expected receipt data here. When the caller reports the step result,
    the receipt validation checks that the provided receipt matches the
    expected values.
    """

    step_id: str = Field(..., description="Step ID this pending receipt is bound to")
    task_id: str = Field(..., description="Verification task ID")
    expected_command_hash: str = Field(
        ..., description="Expected SHA-256 hash of the verification command"
    )
    issued_at: datetime = Field(..., description="When this pending receipt was created")

    @field_validator("expected_command_hash")
    @classmethod
    def validate_expected_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("expected_command_hash must be a 64-character lowercase hex sha256 digest")
        return normalized


def issue_verification_receipt(
    step_id: str,
    command: str,
    exit_code: int,
    output: str,
) -> VerificationReceipt:
    """Issue a verification receipt for an execute_verification step.

    This function creates a tamper-evident receipt containing cryptographic
    hashes of the verification command, exit code, and output. The receipt
    should be included in last_step_result when reporting outcome='success'.

    Args:
        step_id: Step ID this receipt is bound to
        command: The verification command that was executed
        exit_code: Exit code from the verification command
        output: Combined stdout/stderr output from the command

    Returns:
        VerificationReceipt with command_hash, exit_code, output_digest

    Example:
        >>> receipt = issue_verification_receipt(
        ...     step_id="step_01HXYZ",
        ...     command="pytest tests/",
        ...     exit_code=0,
        ...     output="3 passed",
        ... )
        >>> # Include receipt in last_step_result
        >>> result.verification_receipt = receipt
    """
    import hashlib

    command_hash = hashlib.sha256(command.encode()).hexdigest()
    output_digest = hashlib.sha256(output.encode()).hexdigest()

    return VerificationReceipt(
        command_hash=command_hash,
        exit_code=exit_code,
        output_digest=output_digest,
        issued_at=datetime.now(timezone.utc),
        step_id=step_id,
    )
