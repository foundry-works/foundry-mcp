"""Unified metrics tool with action routing and validation."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Tuple, TypedDict

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


class MetricsQueryPayload(TypedDict, total=False):
    """Typed definition for query inputs."""

    metric_name: str
    labels: Mapping[str, str]
    label_selectors: Mapping[str, str]
    since: str
    until: str
    limit: int
    cursor: str


class MetricsListPayload(TypedDict, total=False):
    """Typed definition for list inputs."""

    limit: int
    cursor: str


class MetricsSummaryPayload(TypedDict, total=False):
    """Typed definition for summary inputs."""

    metric_name: str
    labels: Mapping[str, str]
    since: str
    until: str


class MetricsCleanupPayload(TypedDict, total=False):
    """Typed definition for cleanup inputs."""

    retention_days: int
    max_records: int
    dry_run: bool


_ACTION_SUMMARY = {
    "query": "Query persisted metrics with optional filters",
    "list": "List persisted metric series with pagination",
    "summary": "Return aggregate statistics for a metric",
    "stats": "Surface global metrics persistence statistics",
    "cleanup": "Apply retention policy or preview cleanup",
}


def _metrics_disabled_response() -> dict:
    return asdict(
        error_response(
            "Metrics persistence is disabled",
            error_code=ErrorCode.UNAVAILABLE,
            error_type=ErrorType.UNAVAILABLE,
            remediation="Enable metrics_persistence.enabled in server configuration",
            details={"config_key": "metrics_persistence.enabled"},
        )
    )


def _invalid_cursor_response(exc: CursorError) -> dict:
    return asdict(
        error_response(
            f"Invalid cursor: {exc}",
            error_code=ErrorCode.INVALID_FORMAT,
            error_type=ErrorType.VALIDATION,
            remediation="Use the cursor value returned by the previous response",
        )
    )


def _validation_error(
    field: str,
    action: str,
    message: str,
    *,
    code: ErrorCode = ErrorCode.INVALID_FORMAT,
    remediation: Optional[str] = None,
) -> dict:
    return asdict(
        error_response(
            f"Invalid field '{field}' for metrics.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            details={"field": field, "action": f"metrics.{action}"},
        )
    )


def _resolve_metrics_store(config: ServerConfig) -> Tuple[Any | None, Optional[dict]]:
    persistence = getattr(config, "metrics_persistence", None)
    if not persistence or not persistence.enabled:
        return None, _metrics_disabled_response()

    try:
        from foundry_mcp.core.metrics_store import get_metrics_store

        store = get_metrics_store(persistence.get_storage_path())
    except Exception as exc:  # pragma: no cover - defensive import/runtime guard
        logger.exception("Failed to initialize metrics store")
        return None, asdict(
            error_response(
                f"Failed to initialize metrics store: {exc}",
                error_code=ErrorCode.UNAVAILABLE,
                error_type=ErrorType.UNAVAILABLE,
                remediation="Verify metrics persistence configuration",
                details={"storage_path": str(persistence.get_storage_path())},
            )
        )

    return store, None


def _normalize_labels(
    *,
    action: str,
    labels: Any = None,
    label_selectors: Any = None,
) -> Tuple[Optional[Dict[str, str]], Optional[dict]]:
    source = labels if labels is not None else label_selectors
    if source is None:
        return None, None
    if not isinstance(source, Mapping):
        return None, _validation_error(
            "labels",
            action,
            "Expected an object with label key/value pairs",
        )

    normalized: Dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not key.strip():
            return None, _validation_error(
                "labels",
                action,
                "Label names must be non-empty strings",
            )
        if not isinstance(value, str):
            return None, _validation_error(
                "labels",
                action,
                "Label values must be strings",
            )
        normalized[key] = value

    return normalized or None, None


def _validate_optional_str(
    value: Any,
    *,
    field: str,
    action: str,
    allow_empty: bool = False,
) -> Tuple[Optional[str], Optional[dict]]:
    if value is None:
        return None, None
    if isinstance(value, str) and (allow_empty or value.strip()):
        return value, None
    return None, _validation_error(field, action, "Expected a non-empty string")


def _validate_required_str(
    value: Any, *, field: str, action: str
) -> Tuple[str, Optional[dict]]:
    normalized, error = _validate_optional_str(value, field=field, action=action)
    if error:
        return "", error
    if normalized is None:
        return "", _validation_error(
            field,
            action,
            "Value is required",
            code=ErrorCode.MISSING_REQUIRED,
            remediation=f"Provide '{field}' for metrics.{action}",
        )
    return normalized, None


def _validate_optional_int(
    value: Any,
    *,
    field: str,
    action: str,
    minimum: int = 1,
) -> Tuple[Optional[int], Optional[dict]]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, _validation_error(field, action, "Boolean values are not allowed")
    if not isinstance(value, int):
        return None, _validation_error(field, action, "Expected an integer")
    if value < minimum:
        return None, _validation_error(
            field,
            action,
            f"Value must be >= {minimum}",
            remediation=f"Provide a {field} that is at least {minimum}",
        )
    return value, None


def _validate_optional_bool(
    value: Any, *, field: str, action: str
) -> Tuple[Optional[bool], Optional[dict]]:
    if value is None:
        return None, None
    if isinstance(value, bool):
        return value, None
    return None, _validation_error(field, action, "Expected a boolean value")


def _validate_optional_cursor(
    value: Any, *, action: str
) -> Tuple[Optional[str], Optional[dict]]:
    if value is None:
        return None, None
    if isinstance(value, str) and value.strip():
        return value, None
    return None, _validation_error(
        "cursor", action, "Cursor must be a non-empty string"
    )


def _validate_query_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[dict]]:
    action = "query"
    metric_name, error = _validate_optional_str(
        payload.get("metric_name"), field="metric_name", action=action
    )
    if error:
        return {}, error

    labels, error = _normalize_labels(
        action=action,
        labels=payload.get("labels"),
        label_selectors=payload.get("label_selectors"),
    )
    if error:
        return {}, error

    since, error = _validate_optional_str(
        payload.get("since"), field="since", action=action
    )
    if error:
        return {}, error
    until, error = _validate_optional_str(
        payload.get("until"), field="until", action=action
    )
    if error:
        return {}, error

    limit, error = _validate_optional_int(
        payload.get("limit"), field="limit", action=action
    )
    if error:
        return {}, error

    cursor, error = _validate_optional_cursor(payload.get("cursor"), action=action)
    if error:
        return {}, error

    return {
        "metric_name": metric_name,
        "labels": labels,
        "since": since,
        "until": until,
        "limit": limit,
        "cursor": cursor,
    }, None


def _validate_list_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[dict]]:
    action = "list"
    limit, error = _validate_optional_int(
        payload.get("limit"), field="limit", action=action
    )
    if error:
        return {}, error
    cursor, error = _validate_optional_cursor(payload.get("cursor"), action=action)
    if error:
        return {}, error
    return {"limit": limit, "cursor": cursor}, None


def _validate_summary_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[dict]]:
    action = "summary"
    metric_name, error = _validate_required_str(
        payload.get("metric_name"), field="metric_name", action=action
    )
    if error:
        return {}, error

    labels, error = _normalize_labels(
        action=action,
        labels=payload.get("labels"),
        label_selectors=payload.get("label_selectors"),
    )
    if error:
        return {}, error

    since, error = _validate_optional_str(
        payload.get("since"), field="since", action=action
    )
    if error:
        return {}, error
    until, error = _validate_optional_str(
        payload.get("until"), field="until", action=action
    )
    if error:
        return {}, error

    return {
        "metric_name": metric_name,
        "labels": labels,
        "since": since,
        "until": until,
    }, None


def _validate_cleanup_payload(
    payload: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Optional[dict]]:
    action = "cleanup"
    retention_days, error = _validate_optional_int(
        payload.get("retention_days"), field="retention_days", action=action
    )
    if error:
        return {}, error
    max_records, error = _validate_optional_int(
        payload.get("max_records"), field="max_records", action=action
    )
    if error:
        return {}, error
    dry_run, error = _validate_optional_bool(
        payload.get("dry_run"), field="dry_run", action=action
    )
    if error:
        return {}, error
    return {
        "retention_days": retention_days,
        "max_records": max_records,
        "dry_run": dry_run if dry_run is not None else False,
    }, None


def perform_metrics_query(
    *,
    config: ServerConfig,
    metric_name: Optional[str],
    labels: Optional[Mapping[str, str]],
    since: Optional[str],
    until: Optional[str],
    limit: Optional[int],
    cursor: Optional[str],
) -> dict:
    store, error = _resolve_metrics_store(config)
    if error:
        return error
    assert store is not None

    page_size = normalize_page_size(limit)
    offset = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            offset_value = cursor_data.get("offset", 0)
            offset = (
                int(offset_value)
                if isinstance(offset_value, int)
                else int(offset_value or 0)
            )
        except CursorError as exc:
            return _invalid_cursor_response(exc)
        except (TypeError, ValueError):
            return _invalid_cursor_response(
                CursorError("Cursor offset must be an integer", cursor=cursor)
            )

    try:
        records = store.query(
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
            limit=page_size + 1,
            offset=offset,
        )
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error querying metrics")
        return asdict(
            error_response(
                f"Failed to query metrics: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    has_more = len(records) > page_size
    visible_records = records[:page_size] if has_more else records
    next_cursor = encode_cursor({"offset": offset + page_size}) if has_more else None
    metrics_dicts = [record.to_dict() for record in visible_records]

    data = {
        "metrics": metrics_dicts,
        "count": len(metrics_dicts),
    }

    total_count = None
    try:
        total_count = store.count()
    except Exception:  # pragma: no cover - defensive guard
        logger.debug("Metrics store count failed; omitting total_count", exc_info=True)

    return paginated_response(
        data=data,
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=total_count,
    )


def perform_metrics_list(
    *, config: ServerConfig, limit: Optional[int], cursor: Optional[str]
) -> dict:
    store, error = _resolve_metrics_store(config)
    if error:
        return error
    assert store is not None

    page_size = normalize_page_size(limit)
    offset = 0
    if cursor:
        try:
            cursor_data = decode_cursor(cursor)
            offset = int(cursor_data.get("offset", 0))
        except CursorError as exc:
            return _invalid_cursor_response(exc)
        except (TypeError, ValueError):
            return _invalid_cursor_response(
                CursorError("Cursor offset must be an integer", cursor=cursor)
            )

    try:
        all_metrics = store.list_metrics()
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error listing metrics")
        return asdict(
            error_response(
                f"Failed to list metrics: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    total_count = len(all_metrics)
    end_idx = offset + page_size
    metrics_page = all_metrics[offset:end_idx]
    has_more = end_idx < total_count
    next_cursor = encode_cursor({"offset": end_idx}) if has_more else None

    data = {
        "metrics": metrics_page,
        "count": len(metrics_page),
    }

    return paginated_response(
        data=data,
        cursor=next_cursor,
        has_more=has_more,
        page_size=page_size,
        total_count=total_count,
    )


def perform_metrics_summary(
    *,
    config: ServerConfig,
    metric_name: str,
    labels: Optional[Mapping[str, str]],
    since: Optional[str],
    until: Optional[str],
) -> dict:
    store, error = _resolve_metrics_store(config)
    if error:
        return error
    assert store is not None

    try:
        summary = store.get_summary(
            metric_name=metric_name,
            labels=labels,
            since=since,
            until=until,
        )
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error getting metrics summary")
        return asdict(
            error_response(
                f"Failed to get metrics summary: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    return asdict(success_response(data={"summary": summary}))


def perform_metrics_stats(*, config: ServerConfig) -> dict:
    store, error = _resolve_metrics_store(config)
    if error:
        return error
    assert store is not None

    try:
        metrics_list = store.list_metrics()
        total_records = store.count()
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error getting metrics stats")
        return asdict(
            error_response(
                f"Failed to get metrics stats: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    unique_metrics = len(metrics_list)
    total_samples = sum(metric.get("count", 0) for metric in metrics_list)

    return asdict(
        success_response(
            data={
                "total_records": total_records,
                "unique_metrics": unique_metrics,
                "total_samples": total_samples,
                "metrics_by_name": {
                    metric.get("metric_name"): metric.get("count", 0)
                    for metric in metrics_list
                    if metric.get("metric_name")
                },
                "storage_path": str(config.metrics_persistence.get_storage_path()),
                "retention_days": config.metrics_persistence.retention_days,
                "max_records": config.metrics_persistence.max_records,
            }
        )
    )


def perform_metrics_cleanup(
    *,
    config: ServerConfig,
    retention_days: Optional[int],
    max_records: Optional[int],
    dry_run: bool,
) -> dict:
    store, error = _resolve_metrics_store(config)
    if error:
        return error
    assert store is not None

    effective_retention = (
        retention_days
        if retention_days is not None
        else config.metrics_persistence.retention_days
    )
    effective_max = (
        max_records
        if max_records is not None
        else config.metrics_persistence.max_records
    )

    try:
        if dry_run:
            current_count = store.count()
            return asdict(
                success_response(
                    data={
                        "current_count": current_count,
                        "retention_days": effective_retention,
                        "max_records": effective_max,
                        "dry_run": True,
                        "message": "Dry run - no records deleted",
                    }
                )
            )

        deleted_count = store.cleanup(
            retention_days=effective_retention,
            max_records=effective_max,
        )
    except Exception as exc:  # pragma: no cover - backend failure guard
        logger.exception("Error cleaning up metrics")
        return asdict(
            error_response(
                f"Failed to cleanup metrics: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    return asdict(
        success_response(
            data={
                "deleted_count": deleted_count,
                "retention_days": effective_retention,
                "max_records": effective_max,
                "dry_run": False,
            }
        )
    )


def _handle_metrics_query(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_query_payload(payload)
    if error:
        return error
    return perform_metrics_query(config=config, **validated)


def _handle_metrics_list(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_list_payload(payload)
    if error:
        return error
    return perform_metrics_list(config=config, **validated)


def _handle_metrics_summary(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_summary_payload(payload)
    if error:
        return error
    return perform_metrics_summary(config=config, **validated)


def _handle_metrics_stats(*, config: ServerConfig, **_: Any) -> dict:
    return perform_metrics_stats(config=config)


def _handle_metrics_cleanup(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_cleanup_payload(payload)
    if error:
        return error
    return perform_metrics_cleanup(config=config, **validated)


_METRICS_ROUTER = ActionRouter(
    tool_name="metrics",
    actions=[
        ActionDefinition(
            name="query",
            handler=_handle_metrics_query,
            summary=_ACTION_SUMMARY["query"],
        ),
        ActionDefinition(
            name="list",
            handler=_handle_metrics_list,
            summary=_ACTION_SUMMARY["list"],
        ),
        ActionDefinition(
            name="summary",
            handler=_handle_metrics_summary,
            summary=_ACTION_SUMMARY["summary"],
        ),
        ActionDefinition(
            name="stats",
            handler=_handle_metrics_stats,
            summary=_ACTION_SUMMARY["stats"],
        ),
        ActionDefinition(
            name="cleanup",
            handler=_handle_metrics_cleanup,
            summary=_ACTION_SUMMARY["cleanup"],
        ),
    ],
)


def _dispatch_metrics_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _METRICS_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported metrics action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_metrics_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated metrics tool."""

    @canonical_tool(
        mcp,
        canonical_name="metrics",
    )
    def metrics(
        action: str,
        metric_name: Optional[str] = None,
        labels: Optional[Mapping[str, str]] = None,
        label_selectors: Optional[Mapping[str, str]] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        retention_days: Optional[int] = None,
        max_records: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
        payload = {
            "metric_name": metric_name,
            "labels": labels,
            "label_selectors": label_selectors,
            "since": since,
            "until": until,
            "limit": limit,
            "cursor": cursor,
            "retention_days": retention_days,
            "max_records": max_records,
            "dry_run": dry_run,
        }
        return _dispatch_metrics_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified metrics tool")


__all__ = [
    "register_unified_metrics_tool",
    "perform_metrics_query",
    "perform_metrics_list",
    "perform_metrics_summary",
    "perform_metrics_stats",
    "perform_metrics_cleanup",
]
