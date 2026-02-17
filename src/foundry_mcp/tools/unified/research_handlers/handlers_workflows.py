"""Core AI workflow handlers: chat, consensus, thinkdeep, ideate."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from foundry_mcp.core.research.models import ConsensusStrategy
from foundry_mcp.core.research.workflows import (
    ChatWorkflow,
    ConsensusWorkflow,
    IdeateWorkflow,
    ThinkDeepWorkflow,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)

from ._helpers import _get_config, _get_memory, _validation_error


def _handle_chat(
    *,
    prompt: Optional[str] = None,
    thread_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    title: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """Handle chat action."""
    if not prompt:
        return _validation_error(field="prompt", action="chat", message="Required non-empty string")

    config = _get_config()
    workflow = ChatWorkflow(config.research, _get_memory())

    result = workflow.execute(
        prompt=prompt,
        thread_id=thread_id,
        system_prompt=system_prompt,
        provider_id=provider_id,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        title=title,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "thread_id": result.metadata.get("thread_id"),
                    "message_count": result.metadata.get("message_count"),
                    "provider_id": result.provider_id,
                    "model_used": result.model_used,
                    "tokens_used": result.tokens_used,
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Chat failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider availability and retry",
            )
        )


def _handle_consensus(
    *,
    prompt: Optional[str] = None,
    providers: Optional[list[str]] = None,
    strategy: Optional[str] = None,
    synthesis_provider: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout_per_provider: float = 360.0,
    max_concurrent: int = 3,
    require_all: bool = False,
    min_responses: int = 1,
    **kwargs: Any,
) -> dict:
    """Handle consensus action."""
    if not prompt:
        return _validation_error(field="prompt", action="consensus", message="Required non-empty string")

    # Parse strategy
    consensus_strategy = ConsensusStrategy.SYNTHESIZE
    if strategy:
        try:
            consensus_strategy = ConsensusStrategy(strategy)
        except ValueError:
            valid = [s.value for s in ConsensusStrategy]
            return _validation_error(
                field="strategy",
                action="consensus",
                message=f"Invalid value. Valid: {valid}",
                remediation=f"Use one of: {', '.join(valid)}",
            )

    config = _get_config()
    workflow = ConsensusWorkflow(config.research, _get_memory())

    result = workflow.execute(
        prompt=prompt,
        providers=providers,
        strategy=consensus_strategy,
        synthesis_provider=synthesis_provider,
        system_prompt=system_prompt,
        timeout_per_provider=timeout_per_provider,
        max_concurrent=max_concurrent,
        require_all=require_all,
        min_responses=min_responses,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "consensus_id": result.metadata.get("consensus_id"),
                    "providers_consulted": result.metadata.get("providers_consulted"),
                    "strategy": result.metadata.get("strategy"),
                    "response_count": result.metadata.get("response_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Consensus failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider availability and retry",
                details=result.metadata,
            )
        )


def _handle_thinkdeep(
    *,
    topic: Optional[str] = None,
    investigation_id: Optional[str] = None,
    query: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    max_depth: Optional[int] = None,
    **kwargs: Any,
) -> dict:
    """Handle thinkdeep action."""
    if not topic and not investigation_id:
        return _validation_error(
            field="topic/investigation_id",
            action="thinkdeep",
            message="Either 'topic' (new) or 'investigation_id' (continue) required",
        )

    config = _get_config()
    workflow = ThinkDeepWorkflow(config.research, _get_memory())

    result = workflow.execute(
        topic=topic,
        investigation_id=investigation_id,
        query=query,
        system_prompt=system_prompt,
        provider_id=provider_id,
        max_depth=max_depth,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "investigation_id": result.metadata.get("investigation_id"),
                    "current_depth": result.metadata.get("current_depth"),
                    "max_depth": result.metadata.get("max_depth"),
                    "converged": result.metadata.get("converged"),
                    "hypothesis_count": result.metadata.get("hypothesis_count"),
                    "step_count": result.metadata.get("step_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "ThinkDeep failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check investigation ID or topic validity",
            )
        )


def _handle_ideate(
    *,
    topic: Optional[str] = None,
    ideation_id: Optional[str] = None,
    ideate_action: str = "generate",
    perspective: Optional[str] = None,
    cluster_ids: Optional[list[str]] = None,
    system_prompt: Optional[str] = None,
    provider_id: Optional[str] = None,
    perspectives: Optional[list[str]] = None,
    scoring_criteria: Optional[list[str]] = None,
    **kwargs: Any,
) -> dict:
    """Handle ideate action."""
    if not topic and not ideation_id:
        return _validation_error(
            field="topic/ideation_id",
            action="ideate",
            message="Either 'topic' (new) or 'ideation_id' (continue) required",
        )

    config = _get_config()
    workflow = IdeateWorkflow(config.research, _get_memory())

    result = workflow.execute(
        topic=topic,
        ideation_id=ideation_id,
        action=ideate_action,
        perspective=perspective,
        cluster_ids=cluster_ids,
        system_prompt=system_prompt,
        provider_id=provider_id,
        perspectives=perspectives,
        scoring_criteria=scoring_criteria,
    )

    if result.success:
        return asdict(
            success_response(
                data={
                    "content": result.content,
                    "ideation_id": result.metadata.get("ideation_id"),
                    "phase": result.metadata.get("phase"),
                    "idea_count": result.metadata.get("idea_count"),
                    "cluster_count": result.metadata.get("cluster_count"),
                }
            )
        )
    else:
        return asdict(
            error_response(
                result.error or "Ideate failed",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check ideation ID or topic validity",
            )
        )
