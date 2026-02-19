"""Append-only audit ledger for autonomous execution.

This module provides tamper-detection audit logging with hash chain integrity.
On single-host deployment, append-only is tamper detection not prevention -
the value is post-hoc auditability.

Key features:
- Hash-linked entries (prev_hash, event_hash)
- Events: step issued/consumed, pause/resume, bypass/override, reset/end
- Auto-verification on session-start
- CLI and programmatic verification share implementation

Usage:
    from foundry_mcp.core.autonomy.audit import (
        AuditLedger,
        AuditEventType,
        append_event,
        verify_chain,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from filelock import BaseFileLock, FileLock
from pydantic import BaseModel, Field

from foundry_mcp.core.authorization import get_server_role

logger = logging.getLogger(__name__)

# Constants
LOCK_TIMEOUT = 5  # seconds
LEDGER_FILENAME = "ledger.jsonl"
GENESIS_HASH = "0" * 64  # SHA-256 of nothing for first entry


class AuditEventType(str, Enum):
    """Types of auditable events in autonomous execution."""

    STEP_ISSUED = "step_issued"
    STEP_CONSUMED = "step_consumed"
    PAUSE = "pause"
    RESUME = "resume"
    BYPASS = "bypass"
    OVERRIDE = "override"
    RESET = "reset"
    END = "end"
    GATE_PASSED = "gate_passed"
    GATE_FAILED = "gate_failed"
    GATE_WAIVED = "gate_waived"
    SPEC_COMPLETE = "spec_complete"


class AuditEvent(BaseModel):
    """A single entry in the audit ledger.

    Each entry is hash-linked to the previous entry for tamper detection.
    """

    sequence: int = Field(..., description="Monotonic sequence number")
    timestamp: str = Field(..., description="ISO-8601 timestamp when event occurred")
    event_type: AuditEventType = Field(..., description="Type of event")
    spec_id: str = Field(..., description="Spec ID this event relates to")
    session_id: Optional[str] = Field(None, description="Session ID if applicable")
    step_id: Optional[str] = Field(None, description="Step ID if applicable")
    phase_id: Optional[str] = Field(None, description="Phase ID if applicable")
    task_id: Optional[str] = Field(None, description="Task ID if applicable")

    # Server context
    server_role: str = Field(default="runner", description="Configured server role")
    instance_id: str = Field(..., description="Unique server instance identifier")

    # Event details
    action: str = Field(..., description="Action taken (e.g., 'issue_step', 'consume_step')")
    payload_digest: Optional[str] = Field(None, description="SHA-256 hash of relevant payload data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event context")

    # Hash chain
    prev_hash: str = Field(..., description="Hash of previous entry")
    event_hash: str = Field(..., description="Hash of this entry (computed)")

    model_config = {
        "populate_by_name": True,
        "use_enum_values": False,
    }

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this event (excluding event_hash field)."""
        data = {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else self.event_type,
            "spec_id": self.spec_id,
            "session_id": self.session_id,
            "step_id": self.step_id,
            "phase_id": self.phase_id,
            "task_id": self.task_id,
            "server_role": self.server_role,
            "instance_id": self.instance_id,
            "action": self.action,
            "payload_digest": self.payload_digest,
            "metadata": self.metadata,
            "prev_hash": self.prev_hash,
        }
        # Sort keys for deterministic hashing
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class VerificationResult(BaseModel):
    """Result of hash chain verification."""

    valid: bool = Field(..., description="Whether the chain is valid")
    total_entries: int = Field(..., description="Total entries checked")
    divergence_point: Optional[int] = Field(None, description="Sequence number where divergence detected")
    divergence_type: Optional[str] = Field(
        None, description="Type of divergence (hash_mismatch, sequence_gap, corrupted_entry)"
    )
    divergence_detail: Optional[str] = Field(None, description="Detailed explanation of divergence")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues found")


class AuditLedger:
    """Append-only audit ledger with hash chain integrity.

    Stores events as JSONL (one JSON object per line) for easy append
    operations while maintaining hash chain integrity.

    Storage path: {storage_path}/ledger.jsonl
    Lock file: {storage_path}/.ledger.lock
    """

    def __init__(
        self,
        spec_id: str,
        storage_path: Optional[Path] = None,
        workspace_path: Optional[Path] = None,
        server_role: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> None:
        """Initialize audit ledger.

        Args:
            spec_id: Spec ID this ledger is for
            storage_path: Custom storage path (overrides default)
            workspace_path: Workspace root (default: cwd)
            server_role: Configured server role override (default: authorization role)
            instance_id: Server instance ID (default: generated)
        """
        self.spec_id = spec_id
        self.workspace_path = workspace_path or Path.cwd()

        # Storage path: specs/.autonomy/audit/{spec_id}/
        if storage_path:
            self.storage_dir = storage_path
        else:
            self.storage_dir = self.workspace_path / "specs" / ".autonomy" / "audit" / spec_id

        self.ledger_path = self.storage_dir / LEDGER_FILENAME
        self.lock_path = self.storage_dir / ".ledger.lock"

        # Server context
        self.server_role = server_role or get_server_role()
        self.instance_id = instance_id or self._generate_instance_id()

        self._ensure_directories()

    def _generate_instance_id(self) -> str:
        """Generate a unique instance identifier."""
        return f"inst_{secrets.token_hex(8)}"

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _acquire_lock(self) -> "BaseFileLock":
        """Get file lock for ledger operations."""
        return FileLock(self.lock_path, timeout=LOCK_TIMEOUT)

    def _read_entries(self) -> List[AuditEvent]:
        """Read all entries from ledger file."""
        entries = []
        if not self.ledger_path.exists():
            return entries

        with open(self.ledger_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(AuditEvent.model_validate(data))
                except (json.JSONDecodeError, Exception) as e:
                    logger.warning(
                        "Corrupted entry at line %d in %s: %s",
                        line_num,
                        self.ledger_path,
                        e,
                    )
                    # Store raw line for forensic analysis
                    entries.append(None)  # type: ignore

        return entries

    def _get_last_entry(self) -> Optional[AuditEvent]:
        """Get the last valid entry in the ledger."""
        if not self.ledger_path.exists():
            return None

        entries = self._read_entries()
        for entry in reversed(entries):
            if entry is not None:
                return entry
        return None

    def _get_next_sequence(self) -> int:
        """Get the next sequence number."""
        last = self._get_last_entry()
        return (last.sequence + 1) if last else 1

    def _compute_payload_digest(self, payload: Optional[Dict[str, Any]]) -> Optional[str]:
        """Compute SHA-256 digest of payload."""
        if not payload:
            return None
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        event_type: AuditEventType,
        action: str,
        session_id: Optional[str] = None,
        step_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Append a new event to the ledger.

        Args:
            event_type: Type of event
            action: Action taken
            session_id: Session ID if applicable
            step_id: Step ID if applicable
            phase_id: Phase ID if applicable
            task_id: Task ID if applicable
            payload: Payload data to hash (not stored, only digest)
            metadata: Additional metadata

        Returns:
            The created audit event

        Raises:
            Timeout: If lock cannot be acquired
        """
        with self._acquire_lock():
            # Get previous hash
            last_entry = self._get_last_entry()
            prev_hash = last_entry.event_hash if last_entry else GENESIS_HASH

            # Create event
            event = AuditEvent(
                sequence=self._get_next_sequence(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                spec_id=self.spec_id,
                session_id=session_id,
                step_id=step_id,
                phase_id=phase_id,
                task_id=task_id,
                server_role=self.server_role,
                instance_id=self.instance_id,
                action=action,
                payload_digest=self._compute_payload_digest(payload),
                metadata=metadata or {},
                prev_hash=prev_hash,
                event_hash="",  # Placeholder, computed next
            )

            # Compute and set event hash
            event.event_hash = event.compute_hash()

            # Append to file
            with open(self.ledger_path, "a") as f:
                f.write(event.model_dump_json() + "\n")

            logger.debug(
                "Appended audit event: seq=%d type=%s spec=%s",
                event.sequence,
                event.event_type.value,
                self.spec_id,
            )

            return event

    def verify(self) -> VerificationResult:
        """Verify hash chain integrity.

        Walks the chain from first to last entry, verifying:
        - Each entry's event_hash matches computed hash
        - Each entry's prev_hash matches previous entry's event_hash
        - Sequence numbers are monotonic

        Returns:
            VerificationResult with details of any divergence
        """
        entries = self._read_entries()

        if not entries:
            return VerificationResult(
                valid=True,
                total_entries=0,
                divergence_point=None,
                divergence_type=None,
                divergence_detail=None,
                warnings=["Empty ledger - nothing to verify"],
            )

        warnings: List[str] = []
        prev_hash = GENESIS_HASH
        expected_sequence = 1
        divergence_point: Optional[int] = None
        divergence_type: Optional[str] = None
        divergence_detail: Optional[str] = None

        for i, entry in enumerate(entries):
            if entry is None:
                # Corrupted entry from _read_entries
                divergence_point = i + 1
                divergence_type = "corrupted_entry"
                divergence_detail = f"Entry at line {i + 1} could not be parsed"
                break

            # Check sequence
            if entry.sequence != expected_sequence:
                divergence_point = entry.sequence
                divergence_type = "sequence_gap"
                divergence_detail = f"Expected sequence {expected_sequence}, got {entry.sequence}"
                break

            # Check prev_hash
            if entry.prev_hash != prev_hash:
                divergence_point = entry.sequence
                divergence_type = "hash_mismatch"
                divergence_detail = (
                    f"prev_hash mismatch at sequence {entry.sequence}: "
                    f"expected {prev_hash[:16]}..., got {entry.prev_hash[:16]}..."
                )
                break

            # Check event_hash
            computed_hash = entry.compute_hash()
            if entry.event_hash != computed_hash:
                divergence_point = entry.sequence
                divergence_type = "hash_mismatch"
                divergence_detail = (
                    f"event_hash mismatch at sequence {entry.sequence}: "
                    f"stored {entry.event_hash[:16]}..., computed {computed_hash[:16]}..."
                )
                break

            # Check timestamp ordering (warning only)
            if i > 0 and entries[i - 1] is not None:
                prev_entry = entries[i - 1]
                if prev_entry.timestamp > entry.timestamp:
                    warnings.append(f"Timestamp out of order at sequence {entry.sequence}")

            prev_hash = entry.event_hash
            expected_sequence += 1

        valid = divergence_point is None

        if not valid:
            logger.warning(
                "Audit chain verification failed: %s at sequence %d",
                divergence_type,
                divergence_point,
            )

        return VerificationResult(
            valid=valid,
            total_entries=len([e for e in entries if e is not None]),
            divergence_point=divergence_point,
            divergence_type=divergence_type,
            divergence_detail=divergence_detail,
            warnings=warnings,
        )

    def get_entries(
        self,
        limit: int = 100,
        event_type: Optional[AuditEventType] = None,
        since_sequence: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Get entries from the ledger.

        Args:
            limit: Maximum entries to return
            event_type: Filter by event type
            since_sequence: Get entries after this sequence number

        Returns:
            List of audit events
        """
        entries = self._read_entries()

        # Filter
        result = []
        for entry in entries:
            if entry is None:
                continue
            if event_type and entry.event_type != event_type:
                continue
            if since_sequence and entry.sequence <= since_sequence:
                continue
            result.append(entry)

        # Sort by sequence descending and limit
        result.sort(key=lambda e: e.sequence, reverse=True)
        return result[:limit]

    def get_entry_count(self) -> int:
        """Get total number of valid entries."""
        entries = self._read_entries()
        return len([e for e in entries if e is not None])


# =============================================================================
# Convenience Functions
# =============================================================================


def append_event(
    spec_id: str,
    event_type: AuditEventType,
    action: str,
    workspace_path: Optional[Path] = None,
    **kwargs: Any,
) -> AuditEvent:
    """Convenience function to append an event.

    Args:
        spec_id: Spec ID
        event_type: Type of event
        action: Action taken
        workspace_path: Workspace root
        **kwargs: Additional arguments passed to ledger.append()

    Returns:
        The created audit event
    """
    ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
    return ledger.append(event_type=event_type, action=action, **kwargs)


def verify_chain(
    spec_id: str,
    workspace_path: Optional[Path] = None,
) -> VerificationResult:
    """Verify hash chain integrity for a spec's audit ledger.

    Args:
        spec_id: Spec ID to verify
        workspace_path: Workspace root

    Returns:
        VerificationResult with details
    """
    ledger = AuditLedger(spec_id=spec_id, workspace_path=workspace_path)
    return ledger.verify()


def get_ledger_path(spec_id: str, workspace_path: Optional[Path] = None) -> Path:
    """Get the path to a spec's audit ledger.

    Args:
        spec_id: Spec ID
        workspace_path: Workspace root

    Returns:
        Path to the ledger file
    """
    ws = workspace_path or Path.cwd()
    return ws / "specs" / ".autonomy" / "audit" / spec_id / LEDGER_FILENAME


__all__ = [
    "AuditEventType",
    "AuditEvent",
    "AuditLedger",
    "VerificationResult",
    "append_event",
    "verify_chain",
    "get_ledger_path",
    "GENESIS_HASH",
]
