"""Unified error introspection tool with action routing."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
    paginated_response,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)

_ACTION_SUMMARY = {
    "list": "Query collected errors with filters + pagination",
    "get": "Retrieve a single error record by identifier",
    "stats": "Aggregate error counts across dimensions",
    "patterns": "List recurring error fingerprints",
    "cleanup": "Apply retention limits to error storage",
}


def _error_collection_disabled_response() -> dict:
    return asdict(
        error_response(
            "Error collection is disabled",
            error_code=ErrorCode.UNAVAILABLE,
            error_type=ErrorType.UNAVAILABLE,
            details={"config_key": "error_collection.enabled"},
            remediation="Set error_collection.enabled=true in server configuration",
        )
    )


def _collector_unavailable_response() -> dict:
    return asdict(
        error_response(
            "Error collector is not enabled",
            error_code=ErrorCode.UNAVAILABLE,
            error_type=ErrorType.UNAVAILABLE,
            remediation="Initialize the error collector before querying records",
        )
    )


def _invalid_cursor_response(exc: CursorError) -> dict:
    return asdict(
        error_response(
            f"Invalid cursor: {exc}",
            error_code=ErrorCode.VALIDATION_ERROR,
            error_type=ErrorType.VALIDATION,
            remediation="Pass a cursor value returned by a previous response",
        )
    )


def _missing_parameter_response(param: str, action: str) -> dict:
    return asdict(
        error_response(
            f"Missing required parameter '{param}' for error.{action}",
            error_code=ErrorCode.MISSING_REQUIRED,
            error_type=ErrorType.VALIDATION,
            remediation=f"Provide '{param}' when action={action}",
        )
    )


def _resolve_error_store(
    config: ServerConfig,
) -> Tuple[Any | None, Optional[dict]]:
    if (
        not getattr(config, "error_collection", None)
        or not config.error_collection.enabled
    ):
        return None, _error_collection_disabled_response()

    try:
        from foundry_mcp.core.error_collection import get_error_collector

        collector = get_error_collector()
    except Exception as exc:  # pragma: no cover - defensive import guard
        logger.exception("Failed to initialize error collector")
        return None, asdict(
            error_response(
                f"Failed to initialize error collector: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect server logs for error collection issues",
            )
        )

    if not collector.is_enabled():
        return None, _collector_unavailable_response()

    return collector.store, None


def perform_error_list(
    *,
    config: ServerConfig,
    tool_name: Optional[str] = None,
    error_code: Optional[str] = None,
    error_type: Optional[str] = None,
    fingerprint: Optional[str] = None,
    provider_id: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: Optional[int] = None,
    cursor: Optional[str] = None,
) -> dict:
    store, error = _resolve_error_store(config)
    if error:
        return error
    assert store is not None

    page_size = normalize_page_size(limit)
    offset = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            offset = cursor_data.get("offset", 0)
        except CursorError as exc:
            return _invalid_cursor_response(exc)

    try:
        records = store.query(
            tool_name=tool_name,
            error_code=error_code,
            error_type=error_type,
            fingerprint=fingerprint,
            provider_id=provider_id,
            since=since,
            until=until,
            limit=page_size + 1,
            offset=offset,
        )
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error querying errors")
        return asdict(
            error_response(
                f"Failed to query errors: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check error collector logs",
            )
        )

    has_more = len(records) > page_size
    visible_records = records[:page_size] if has_more else records
    next_cursor = encode_cursor({"offset": offset + page_size}) if has_more else None
    error_dicts = [record.to_dict() for record in visible_records]

    data = {
        "errors": error_dicts,
        "count": len(error_dicts),
    }

    return paginated_response(
        data=data,
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=store.count(),
    )


def perform_error_get(*, config: ServerConfig, error_id: Optional[str] = None) -> dict:
    if not error_id:
        return _missing_parameter_response("error_id", "get")

    store, error = _resolve_error_store(config)
    if error:
        return error
    assert store is not None

    try:
        record = store.get(error_id)
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error retrieving error record")
        return asdict(
            error_response(
                f"Failed to retrieve error: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check error collector logs",
            )
        )

    if record is None:
        return asdict(
            error_response(
                f"Error record not found: {error_id}",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Verify the error ID via error.list",
            )
        )

    return asdict(success_response(data={"error": record.to_dict()}))


def perform_error_stats(*, config: ServerConfig) -> dict:
    store, error = _resolve_error_store(config)
    if error:
        return error
    assert store is not None

    try:
        stats = store.get_stats()
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error retrieving error stats")
        return asdict(
            error_response(
                f"Failed to get error stats: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect error collector logs",
            )
        )

    return asdict(success_response(data=stats))


def perform_error_patterns(*, config: ServerConfig, min_count: int = 3) -> dict:
    store, error = _resolve_error_store(config)
    if error:
        return error
    assert store is not None

    effective_min = max(1, min_count or 1)

    try:
        patterns = store.get_patterns(min_count=effective_min)

    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error retrieving error patterns")
        return asdict(
            error_response(
                f"Failed to get error patterns: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect error collector logs",
            )
        )

    return asdict(
        success_response(
            data={
                "patterns": patterns,
                "pattern_count": len(patterns),
                "min_count_filter": effective_min,
            }
        )
    )


def perform_error_cleanup(
    *,
    config: ServerConfig,
    retention_days: Optional[int] = None,
    max_errors: Optional[int] = None,
    dry_run: bool = False,
) -> dict:
    store, error = _resolve_error_store(config)
    if error:
        return error
    assert store is not None

    effective_retention = retention_days or config.error_collection.retention_days
    effective_max = max_errors or config.error_collection.max_errors

    try:
        if dry_run:
            current_count = store.count()
            return asdict(
                success_response(
                    data={
                        "current_count": current_count,
                        "retention_days": effective_retention,
                        "max_errors": effective_max,
                        "dry_run": True,
                        "message": "Dry run - no records deleted",
                    }
                )
            )

        deleted_count = store.cleanup(
            retention_days=effective_retention,
            max_errors=effective_max,
        )
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error cleaning up error records")
        return asdict(
            error_response(
                f"Failed to cleanup errors: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Inspect error collector logs",
            )
        )

    return asdict(
        success_response(
            data={
                "deleted_count": deleted_count,
                "retention_days": effective_retention,
                "max_errors": effective_max,
                "dry_run": False,
            }
        )
    )


def _handle_error_list(*, config: ServerConfig, **payload: Any) -> dict:
    return perform_error_list(config=config, **payload)


def _handle_error_get(*, config: ServerConfig, **payload: Any) -> dict:
    return perform_error_get(config=config, error_id=payload.get("error_id"))


def _handle_error_stats(*, config: ServerConfig, **_: Any) -> dict:
    return perform_error_stats(config=config)


def _handle_error_patterns(*, config: ServerConfig, **payload: Any) -> dict:
    return perform_error_patterns(config=config, min_count=payload.get("min_count", 3))


def _handle_error_cleanup(*, config: ServerConfig, **payload: Any) -> dict:
    return perform_error_cleanup(
        config=config,
        retention_days=payload.get("retention_days"),
        max_errors=payload.get("max_errors"),
        dry_run=payload.get("dry_run", False),
    )


_ERROR_ROUTER = ActionRouter(
    tool_name="error",
    actions=[
        ActionDefinition(
            name="list",
            handler=_handle_error_list,
            summary=_ACTION_SUMMARY["list"],
        ),
        ActionDefinition(
            name="get",
            handler=_handle_error_get,
            summary=_ACTION_SUMMARY["get"],
        ),
        ActionDefinition(
            name="stats",
            handler=_handle_error_stats,
            summary=_ACTION_SUMMARY["stats"],
        ),
        ActionDefinition(
            name="patterns",
            handler=_handle_error_patterns,
            summary=_ACTION_SUMMARY["patterns"],
        ),
        ActionDefinition(
            name="cleanup",
            handler=_handle_error_cleanup,
            summary=_ACTION_SUMMARY["cleanup"],
        ),
    ],
)


def _dispatch_error_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _ERROR_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported error action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_error_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated error tool."""

    @canonical_tool(
        mcp,
        canonical_name="error",
    )
    def error(
        action: str,
        error_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        fingerprint: Optional[str] = None,
        provider_id: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        min_count: int = 3,
        retention_days: Optional[int] = None,
        max_errors: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
        """Execute error workflows via the action router."""

        payload = {
            "error_id": error_id,
            "tool_name": tool_name,
            "error_code": error_code,
            "error_type": error_type,
            "fingerprint": fingerprint,
            "provider_id": provider_id,
            "since": since,
            "until": until,
            "limit": limit,
            "cursor": cursor,
            "min_count": min_count,
            "retention_days": retention_days,
            "max_errors": max_errors,
            "dry_run": dry_run,
        }
        return _dispatch_error_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified error tool")


__all__ = [
    "register_unified_error_tool",
    "perform_error_list",
    "perform_error_get",
    "perform_error_stats",
    "perform_error_patterns",
    "perform_error_cleanup",
]
