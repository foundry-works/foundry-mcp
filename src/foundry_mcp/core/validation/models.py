"""Validation data models for SDD spec files."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Diagnostic:
    """
    Structured diagnostic for MCP consumption.

    Provides a machine-readable format for validation findings
    that can be easily processed by MCP tools.
    """

    code: str  # Diagnostic code (e.g., "MISSING_FILE_PATH", "INVALID_STATUS")
    message: str  # Human-readable description
    severity: str  # "error", "warning", "info"
    category: str  # Category for grouping (e.g., "metadata", "structure", "counts")
    location: Optional[str] = None  # Node ID or path where issue occurred
    suggested_fix: Optional[str] = None  # Suggested fix description
    auto_fixable: bool = False  # Whether this can be auto-fixed


@dataclass
class ValidationResult:
    """
    Complete validation result for a spec file.
    """

    spec_id: str
    is_valid: bool
    diagnostics: List[Diagnostic] = field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass
class FixAction:
    """
    Represents a candidate auto-fix operation.
    """

    id: str
    description: str
    category: str
    severity: str
    auto_apply: bool
    preview: str
    apply: Callable[[Dict[str, Any]], None]


@dataclass
class FixReport:
    """
    Outcome of applying a set of fix actions.
    """

    spec_path: Optional[str] = None
    backup_path: Optional[str] = None
    applied_actions: List[FixAction] = field(default_factory=list)
    skipped_actions: List[FixAction] = field(default_factory=list)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None


@dataclass
class SpecStats:
    """
    Statistics for a spec file.
    """

    spec_id: str
    title: str
    version: str
    status: str
    totals: Dict[str, int] = field(default_factory=dict)
    status_counts: Dict[str, int] = field(default_factory=dict)
    max_depth: int = 0
    avg_tasks_per_phase: float = 0.0
    verification_coverage: float = 0.0
    progress: float = 0.0
    file_size_kb: float = 0.0
