"""
Standard response contracts for MCP tool operations.

Re-exports all public symbols from sub-modules for backward compatibility.
Callers can use ``from foundry_mcp.core.responses import success_response``
or import from canonical sub-module paths like ``responses.builders``.

Sub-modules:
    types           - ErrorCode, ErrorType, ToolResponse, _build_meta
    builders        - success_response, error_response
    errors_generic  - validation_error, not_found_error, rate_limit_error, etc.
    errors_spec     - circular_dependency_error, invalid_parent_error, etc.
    errors_ai       - ai_no_provider_error, ai_provider_timeout_error, etc.
    sanitization    - sanitize_error_message
    batch_schemas   - Pydantic models for batch operations
"""

# --- Core types ---
from foundry_mcp.core.responses.types import (  # noqa: F401
    ErrorCode,
    ErrorType,
    ToolResponse,
    _build_meta,
)

# --- Response builders ---
from foundry_mcp.core.responses.builders import (  # noqa: F401
    error_response,
    success_response,
)

# --- Generic error helpers ---
from foundry_mcp.core.responses.errors_generic import (  # noqa: F401
    conflict_error,
    forbidden_error,
    internal_error,
    not_found_error,
    rate_limit_error,
    unauthorized_error,
    unavailable_error,
    validation_error,
)

# --- Spec-domain error helpers ---
from foundry_mcp.core.responses.errors_spec import (  # noqa: F401
    backup_corrupted_error,
    backup_not_found_error,
    circular_dependency_error,
    comparison_failed_error,
    dependency_not_found_error,
    invalid_parent_error,
    invalid_position_error,
    invalid_regex_error,
    no_matches_error,
    pattern_too_broad_error,
    rollback_failed_error,
    self_reference_error,
)

# --- AI/LLM provider error helpers ---
from foundry_mcp.core.responses.errors_ai import (  # noqa: F401
    ai_cache_stale_error,
    ai_context_too_large_error,
    ai_no_provider_error,
    ai_prompt_not_found_error,
    ai_provider_error,
    ai_provider_timeout_error,
)

# --- Sanitization ---
from foundry_mcp.core.responses.sanitization import (  # noqa: F401
    sanitize_error_message,
)

# --- Batch operation schemas ---
from foundry_mcp.core.responses.batch_schemas import (  # noqa: F401
    PYDANTIC_AVAILABLE,
    BatchCompleteResponse,
    BatchPrepareResponse,
    BatchStartResponse,
    BatchTaskCompletion,
    BatchTaskContext,
    BatchTaskDependencies,
    BatchTaskResult,
    DependencyEdge,
    DependencyGraph,
    DependencyNode,
    StaleTaskInfo,
    __all_pydantic__,
)
