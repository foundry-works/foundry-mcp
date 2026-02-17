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
from foundry_mcp.tools.unified.param_schema import AtLeastOne, Str, validate_payload

from ._helpers import _get_config, _get_memory

# ---------------------------------------------------------------------------
# Declarative validation schemas
# ---------------------------------------------------------------------------

_CHAT_SCHEMA = {
    "prompt": Str(required=True),
}

_CONSENSUS_SCHEMA = {
    "prompt": Str(required=True),
    "strategy": Str(choices=frozenset(s.value for s in ConsensusStrategy)),
}

_THINKDEEP_SCHEMA: dict = {}
_THINKDEEP_CROSS_FIELD = [AtLeastOne(fields=("topic", "investigation_id"))]

_IDEATE_SCHEMA: dict = {}
_IDEATE_CROSS_FIELD = [AtLeastOne(fields=("topic", "ideation_id"))]


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
    payload = {"prompt": prompt}
    err = validate_payload(payload, _CHAT_SCHEMA, tool_name="research", action="chat")
    if err:
        return err

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
    payload = {"prompt": prompt, "strategy": strategy}
    err = validate_payload(payload, _CONSENSUS_SCHEMA, tool_name="research", action="consensus")
    if err:
        return err

    # Convert strategy string to enum (schema already validated choices)
    consensus_strategy = ConsensusStrategy(strategy) if strategy else ConsensusStrategy.SYNTHESIZE

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
    payload = {"topic": topic, "investigation_id": investigation_id}
    err = validate_payload(
        payload, _THINKDEEP_SCHEMA,
        tool_name="research", action="thinkdeep",
        cross_field_rules=_THINKDEEP_CROSS_FIELD,
    )
    if err:
        return err

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
    payload = {"topic": topic, "ideation_id": ideation_id}
    err = validate_payload(
        payload, _IDEATE_SCHEMA,
        tool_name="research", action="ideate",
        cross_field_rules=_IDEATE_CROSS_FIELD,
    )
    if err:
        return err

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
