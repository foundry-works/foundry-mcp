"""Shared pure utility functions used by multiple phases.

These are stateless functions with no instance access. Called as
module-level functions (not via ``self``).

This module is a backward-compatibility re-export shim. The actual
implementations have been split into focused modules:

- ``_json_parsing`` — JSON extraction from LLM responses
- ``_token_budget`` — fidelity scoring, truncation, structured source dropping
- ``_model_resolution`` — model/provider lookup, reflection/clarification parsing
- ``_content_dedup`` — content similarity, novelty tagging
- ``_injection_protection`` — SSRF validation, prompt injection sanitization
"""

# Re-export everything so existing ``from ._helpers import X`` continues to work.

from foundry_mcp.core.research.workflows.deep_research._content_dedup import (  # noqa: F401
    NoveltyTag,
    compute_novelty_tag,
    content_similarity,
)
from foundry_mcp.core.research.workflows.deep_research._injection_protection import (  # noqa: F401
    build_novelty_summary,
    build_sanitized_context,
    sanitize_external_content,
    validate_extract_url,
)
from foundry_mcp.core.research.workflows.deep_research._json_parsing import (  # noqa: F401
    extract_json,
)
from foundry_mcp.core.research.workflows.deep_research._model_resolution import (  # noqa: F401
    ClarificationDecision,
    TopicReflectionDecision,
    estimate_token_limit_for_model,
    parse_clarification_decision,
    parse_reflection_decision,
    resolve_phase_provider,
    safe_resolve_model_for_role,
)
from foundry_mcp.core.research.workflows.deep_research._token_budget import (  # noqa: F401
    _split_prompt_sections,
    fidelity_level_from_score,
    structured_drop_sources,
    structured_truncate_blocks,
    truncate_at_boundary,
    truncate_to_token_estimate,
)

# _extract_domain was imported from source_quality in the original module
# and is used by some tests via _helpers. Re-export for backward compat.
from foundry_mcp.core.research.workflows.deep_research.source_quality import (  # noqa: F401
    _extract_domain,
)
