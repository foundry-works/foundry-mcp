"""
AI Consultation Layer for foundry-mcp.

This module provides a unified interface for AI-assisted operations including
plan review and fidelity checking. It integrates with the provider registry
to support multiple LLM backends while providing caching, timeout handling,
and consistent result structures.

Design Principles:
    - Workflow-specific prompt templates (plan_review, fidelity_review)
    - Provider-agnostic orchestration via the provider registry
    - Filesystem-based caching for consultation results
    - Consistent result structures across all workflows
    - Graceful degradation when providers are unavailable

Example Usage:
    from foundry_mcp.core.ai_consultation import (
        ConsultationOrchestrator,
        ConsultationRequest,
        ConsultationWorkflow,
    )
    from foundry_mcp.core.providers import ProviderHooks

    orchestrator = ConsultationOrchestrator()

    # Check availability
    if orchestrator.is_available():
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="spec_review",
            context={"spec_content": "..."},
            provider_id="gemini",
        )
        result = orchestrator.consult(request)
        if result.content:
            print(result.content)
"""

from foundry_mcp.core.ai_consultation.types import (
    AgreementMetadata,
    ConsensusResult,
    ConsultationOutcome,
    ConsultationRequest,
    ConsultationResult,
    ConsultationWorkflow,
    ProviderResponse,
    ResolvedProvider,
)
from foundry_mcp.core.ai_consultation.cache import ResultCache
from foundry_mcp.core.ai_consultation.orchestrator import ConsultationOrchestrator

__all__ = [
    # Workflow types
    "ConsultationWorkflow",
    # Request/Response
    "ConsultationRequest",
    "ConsultationResult",
    "ProviderResponse",
    "AgreementMetadata",
    "ConsensusResult",
    "ConsultationOutcome",
    "ResolvedProvider",
    # Cache
    "ResultCache",
    # Orchestrator
    "ConsultationOrchestrator",
]
