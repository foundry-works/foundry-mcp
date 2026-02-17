"""Unified provider tool backed by ActionRouter."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config.server import ServerConfig
from foundry_mcp.core.errors.llm import RateLimitError
from foundry_mcp.core.errors.provider import (
    ProviderExecutionError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.observability import get_metrics, mcp_tool
from foundry_mcp.core.providers import (
    ProviderHooks,
    ProviderRequest,
    check_provider_available,
    describe_providers,
    get_provider_metadata,
    get_provider_statuses,
    resolve_provider,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    sanitize_error_message,
    success_response,
)
from foundry_mcp.tools.unified.common import (
    build_request_id,
    dispatch_with_standard_errors,
    make_metric_name,
)
from foundry_mcp.tools.unified.param_schema import Bool, Num, Str, validate_payload
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
)

logger = logging.getLogger(__name__)
_metrics = get_metrics()

_ACTION_SUMMARY = {
    "list": "List registered providers with optional unavailable entries",
    "status": "Fetch metadata and health for a provider",
    "execute": "Run prompts through providers with validation and telemetry",
}


def _metric_name(action: str) -> str:
    return make_metric_name("provider", action)


def _request_id() -> str:
    return build_request_id("provider")


# ---------------------------------------------------------------------------
# Declarative parameter schemas
# ---------------------------------------------------------------------------

_LIST_SCHEMA = {
    "include_unavailable": Bool(default=False),
}

_STATUS_SCHEMA = {
    "provider_id": Str(required=True, remediation="Call provider(action=list) to discover valid providers"),
}

_EXECUTE_SCHEMA = {
    "provider_id": Str(required=True, remediation="Call provider(action=list) to discover valid providers"),
    "prompt": Str(required=True, remediation="Supply the text you want to send to the provider"),
    "model": Str(),
    "max_tokens": Num(min_val=1, integer_only=True),
    "temperature": Num(min_val=0, max_val=2),
    "timeout": Num(min_val=1, integer_only=True),
}


def _handle_list(*, config: ServerConfig, **payload: Any) -> dict:  # noqa: ARG001
    request_id = _request_id()

    err = validate_payload(payload, _LIST_SCHEMA,
                           tool_name="provider", action="list",
                           request_id=request_id)
    if err:
        return err

    include = payload.get("include_unavailable", False)

    try:
        providers = describe_providers()
    except Exception:
        logger.exception("Failed to describe providers")
        _metrics.counter(_metric_name("list"), labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to list providers",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect provider registry configuration",
                request_id=request_id,
            )
        )

    total_count = len(providers)
    available_count = sum(
        1 for provider in providers if provider.get("available", False)
    )
    visible = (
        providers
        if include
        else [provider for provider in providers if provider.get("available", False)]
    )

    warnings: List[str] = []
    if not include and available_count < total_count:
        missing = total_count - available_count
        warnings.append(
            f"{missing} provider(s) filtered out because they are unavailable"
        )

    _metrics.counter(_metric_name("list"), labels={"status": "success"})
    return asdict(
        success_response(
            data={
                "providers": visible,
                "available_count": available_count,
                "total_count": total_count,
            },
            warnings=warnings or None,
            request_id=request_id,
        )
    )


def _handle_status(*, config: ServerConfig, **payload: Any) -> dict:  # noqa: ARG001
    request_id = _request_id()

    err = validate_payload(payload, _STATUS_SCHEMA,
                           tool_name="provider", action="status",
                           request_id=request_id)
    if err:
        return err

    provider_id = payload["provider_id"]

    try:
        availability = check_provider_available(provider_id)
        metadata = get_provider_metadata(provider_id)
        statuses = get_provider_statuses()
    except Exception:
        logger.exception(
            "Failed to load provider status", extra={"provider_id": provider_id}
        )
        _metrics.counter(_metric_name("status"), labels={"status": "error"})
        return asdict(
            error_response(
                f"Failed to retrieve status for provider '{provider_id}'",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect provider registry configuration",
                request_id=request_id,
            )
        )

    metadata_dict: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    if metadata:
        metadata_dict = {
            "name": metadata.display_name or metadata.provider_id,
            "version": metadata.extra.get("version") if metadata.extra else None,
            "default_model": metadata.default_model,
            "supported_models": [
                {
                    "id": model.id,
                    "name": model.display_name or model.id,
                    "context_window": model.routing_hints.get("context_window")
                    if model.routing_hints
                    else None,
                    "is_default": model.id == metadata.default_model,
                }
                for model in (metadata.models or [])
            ],
            "documentation_url": metadata.extra.get("documentation_url")
            if metadata.extra
            else None,
            "tags": metadata.extra.get("tags", []) if metadata.extra else [],
        }
        capabilities = [cap.value for cap in (metadata.capabilities or [])]

    health = statuses.get(provider_id)
    health_dict = None
    if health is not None:
        health_dict = {
            "status": "available" if health else "unavailable",
            "available": health,
        }

    if not availability and not metadata_dict and health_dict is None:
        _metrics.counter(_metric_name("status"), labels={"status": "not_found"})
        return asdict(
            error_response(
                f"Provider '{provider_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use provider(action=list) to see registered providers",
                request_id=request_id,
            )
        )

    _metrics.counter(_metric_name("status"), labels={"status": "success"})
    return asdict(
        success_response(
            data={
                "provider_id": provider_id,
                "available": availability,
                "metadata": metadata_dict,
                "capabilities": capabilities,
                "health": health_dict,
            },
            request_id=request_id,
        )
    )


def _handle_execute(*, config: ServerConfig, **payload: Any) -> dict:  # noqa: ARG001
    request_id = _request_id()
    action = "execute"

    err = validate_payload(payload, _EXECUTE_SCHEMA,
                           tool_name="provider", action=action,
                           request_id=request_id)
    if err:
        return err

    provider_id = payload["provider_id"]
    prompt_text = payload["prompt"]
    model_name = payload.get("model")
    max_tokens = payload.get("max_tokens")
    temp_value = payload.get("temperature")
    timeout_value = payload.get("timeout")

    try:
        provider_summaries = describe_providers()
    except Exception:
        logger.exception("Failed to describe providers before execution")
        _metrics.counter(_metric_name(action), labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to resolve provider registry",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect provider registry configuration",
                request_id=request_id,
            )
        )

    known_providers = {
        entry.get("id") for entry in provider_summaries if entry.get("id")
    }
    if provider_id not in known_providers:
        _metrics.counter(_metric_name(action), labels={"status": "not_found"})
        return asdict(
            error_response(
                f"Provider '{provider_id}' not found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Use provider(action=list) to see available providers",
                request_id=request_id,
            )
        )

    try:
        if not check_provider_available(provider_id):
            _metrics.counter(_metric_name(action), labels={"status": "unavailable"})
            return asdict(
                error_response(
                    f"Provider '{provider_id}' is not available",
                    error_code=ErrorCode.UNAVAILABLE,
                    error_type=ErrorType.UNAVAILABLE,
                    data={"provider_id": provider_id},
                    remediation="Verify provider credentials and availability",
                    request_id=request_id,
                )
            )
    except Exception:
        logger.exception(
            "Failed to check provider availability", extra={"provider_id": provider_id}
        )
        _metrics.counter(_metric_name(action), labels={"status": "error"})
        return asdict(
            error_response(
                "Failed to validate provider availability",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect provider detector configuration",
                request_id=request_id,
            )
        )

    hooks = ProviderHooks()
    try:
        provider_ctx = resolve_provider(provider_id, hooks=hooks, model=model_name)
    except ProviderUnavailableError as exc:
        _metrics.counter(_metric_name(action), labels={"status": "unavailable"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="providers"),
                error_code=ErrorCode.UNAVAILABLE,
                error_type=ErrorType.UNAVAILABLE,
                data={"provider_id": provider_id},
                remediation="Verify provider configuration and retry",
                request_id=request_id,
            )
        )

    request = ProviderRequest(
        prompt=prompt_text,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temp_value,
        timeout=timeout_value or 300,
        stream=False,
    )

    metric_key = _metric_name(action)
    start_time = time.perf_counter()
    try:
        result = provider_ctx.generate(request)
    except RateLimitError as exc:
        _metrics.counter(metric_key, labels={"status": "rate_limited"})
        retry_after = exc.retry_after if exc.retry_after is not None else 0
        return asdict(
            error_response(
                f"Provider '{provider_id}' rate limited the request",
                error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
                error_type=ErrorType.RATE_LIMIT,
                data={"provider_id": provider_id, "retry_after_seconds": retry_after},
                remediation="Wait before retrying or reduce concurrent executions",
                request_id=request_id,
                rate_limit={
                    "status": "rate_limited",
                    "retry_after_seconds": retry_after,
                    "provider": provider_id,
                },
            )
        )
    except ProviderTimeoutError:
        _metrics.counter(metric_key, labels={"status": "timeout"})
        return asdict(
            error_response(
                f"Provider '{provider_id}' timed out",
                error_code=ErrorCode.AI_PROVIDER_TIMEOUT,
                error_type=ErrorType.UNAVAILABLE,
                data={"provider_id": provider_id},
                remediation="Increase timeout or simplify the prompt",
                request_id=request_id,
            )
        )
    except ProviderExecutionError:
        _metrics.counter(metric_key, labels={"status": "provider_error"})
        return asdict(
            error_response(
                f"Provider '{provider_id}' execution failed",
                error_code=ErrorCode.AI_PROVIDER_ERROR,
                error_type=ErrorType.AI_PROVIDER,
                data={"provider_id": provider_id},
                remediation="Inspect provider logs and retry after resolving the issue",
                request_id=request_id,
            )
        )
    except Exception as exc:
        logger.exception(
            "Unexpected provider execution failure", extra={"provider_id": provider_id}
        )
        _metrics.counter(metric_key, labels={"status": "error"})
        return asdict(
            error_response(
                sanitize_error_message(exc, context="providers"),
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check provider configuration and retry",
                request_id=request_id,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response_data: Dict[str, Any] = {
        "provider_id": provider_id,
        "model": result.model_used or model_name or "default",
        "content": result.content,
        "finish_reason": result.status.value if result.status else None,
    }
    if result.tokens and result.tokens.total_tokens > 0:
        response_data["token_usage"] = {
            "prompt_tokens": result.tokens.input_tokens,
            "completion_tokens": result.tokens.output_tokens,
            "total_tokens": result.tokens.total_tokens,
        }

    _metrics.counter(metric_key, labels={"status": "success"})
    return asdict(
        success_response(
            data=response_data,
            telemetry={"duration_ms": round(elapsed_ms, 2)},
            request_id=request_id,
        )
    )


_PROVIDER_ROUTER = ActionRouter(
    tool_name="provider",
    actions=[
        ActionDefinition(
            name="list",
            handler=_handle_list,
            summary=_ACTION_SUMMARY["list"],
            aliases=("provider_list",),
        ),
        ActionDefinition(
            name="status",
            handler=_handle_status,
            summary=_ACTION_SUMMARY["status"],
            aliases=("provider_status",),
        ),
        ActionDefinition(
            name="execute",
            handler=_handle_execute,
            summary=_ACTION_SUMMARY["execute"],
            aliases=("provider_execute",),
        ),
    ],
)


def _dispatch_provider_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    return dispatch_with_standard_errors(
        _PROVIDER_ROUTER, "provider", action, config=config, **payload
    )


def register_unified_provider_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated provider tool."""

    @canonical_tool(mcp, canonical_name="provider")
    @mcp_tool(tool_name="provider", emit_metrics=True, audit=True)
    def provider(  # noqa: PLR0913 - unified signature spans multiple actions
        action: str,
        include_unavailable: Optional[bool] = False,
        provider_id: Optional[str] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> dict:
        payload = {
            "include_unavailable": include_unavailable,
            "provider_id": provider_id,
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": timeout,
        }
        return _dispatch_provider_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified provider tool")


__all__ = [
    "register_unified_provider_tool",
]
