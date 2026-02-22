"""
Validation operations for SDD spec files.
Provides spec validation, auto-fix capabilities, and statistics.

Security Note:
    This module uses size limits from foundry_mcp.core.security to protect
    against resource exhaustion attacks. See dev_docs/mcp_best_practices/04-validation-input-hygiene.md
"""

from foundry_mcp.core.validation.application import apply_fixes
from foundry_mcp.core.validation.constants import (
    FIELD_NAME_SUGGESTIONS,
    RESEARCH_BLOCKING_MODES,
    STATUS_FIELDS,
    VALID_NODE_TYPES,
    VALID_RESEARCH_RESULTS,
    VALID_RESEARCH_TYPES,
    VALID_STATUSES,
    VALID_TASK_CATEGORIES,
    VALID_VERIFICATION_TYPES,
    VERIFICATION_RESULTS,
)
from foundry_mcp.core.validation.fixes import get_fix_actions
from foundry_mcp.core.validation.input import validate_spec_input
from foundry_mcp.core.validation.models import (
    Diagnostic,
    FixAction,
    FixReport,
    SpecStats,
    ValidationResult,
)
from foundry_mcp.core.validation.rules import validate_spec
from foundry_mcp.core.validation.stats import calculate_stats
from foundry_mcp.core.validation.verification import (
    add_verification,
    execute_verification,
    format_verification_summary,
)

__all__ = [
    # Constants
    "FIELD_NAME_SUGGESTIONS",
    "RESEARCH_BLOCKING_MODES",
    "STATUS_FIELDS",
    "VALID_NODE_TYPES",
    "VALID_RESEARCH_RESULTS",
    "VALID_RESEARCH_TYPES",
    "VALID_STATUSES",
    "VALID_TASK_CATEGORIES",
    "VALID_VERIFICATION_TYPES",
    "VERIFICATION_RESULTS",
    # Models
    "Diagnostic",
    "FixAction",
    "FixReport",
    "SpecStats",
    "ValidationResult",
    # Functions
    "add_verification",
    "apply_fixes",
    "calculate_stats",
    "execute_verification",
    "format_verification_summary",
    "get_fix_actions",
    "validate_spec",
    "validate_spec_input",
]
