"""Unified journal tool family with action routing and validation."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, cast

from mcp.server.fastmcp import FastMCP

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.journal import (
    add_journal_entry,
    find_unjournaled_tasks,
    get_journal_entries,
)
from foundry_mcp.core.naming import canonical_tool
from foundry_mcp.core.pagination import (
    CursorError,
    decode_cursor,
    encode_cursor,
    normalize_page_size,
)
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    error_response,
    success_response,
)
from foundry_mcp.core.spec import find_specs_directory, load_spec, save_spec
from foundry_mcp.tools.unified.router import (
    ActionDefinition,
    ActionRouter,
    ActionRouterError,
)

logger = logging.getLogger(__name__)

_ALLOWED_ENTRY_TYPES = (
    "status_change",
    "deviation",
    "blocker",
    "decision",
    "note",
)


@dataclass
class JournalAddInput:
    spec_id: str
    title: str
    content: str
    entry_type: str
    task_id: Optional[str]
    workspace: Optional[str]


@dataclass
class JournalListInput:
    spec_id: str
    task_id: Optional[str]
    entry_type: Optional[str]
    cursor: Optional[str]
    limit: Optional[int]
    workspace: Optional[str]


@dataclass
class JournalListUnjournaledInput:
    spec_id: str
    cursor: Optional[str]
    limit: Optional[int]
    workspace: Optional[str]


_ACTION_SUMMARY = {
    "add": "Add a journal entry to a specification",
    "list": "List journal entries with pagination",
    "list-unjournaled": "List completed tasks missing journal entries",
}


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
            f"Invalid field '{field}' for journal.{action}: {message}",
            error_code=code,
            error_type=ErrorType.VALIDATION,
            remediation=remediation,
            details={"field": field, "action": f"journal.{action}"},
        )
    )


def _missing_field(field: str, action: str) -> dict:
    return _validation_error(
        field,
        action,
        "Value is required",
        code=ErrorCode.MISSING_REQUIRED,
        remediation=f"Provide '{field}' when calling journal.{action}",
    )


def _resolve_specs_dir(
    config: ServerConfig, workspace: Optional[str]
) -> Tuple[Optional[Path], Optional[dict]]:
    try:
        specs_dir: Optional[Path] = (
            find_specs_directory(workspace)
            if workspace
            else (config.specs_dir or find_specs_directory())
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to resolve specs directory")
        return None, asdict(
            error_response(
                f"Failed to resolve specs directory: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify specs_dir configuration or pass a workspace path",
            )
        )

    if not specs_dir:
        return None, asdict(
            error_response(
                "No specs directory found",
                error_code=ErrorCode.NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation="Set SDD_SPECS_DIR or provide a workspace path",
            )
        )

    return specs_dir, None


def _load_spec_data(
    *, spec_id: str, specs_dir: Path, action: str
) -> Tuple[Optional[Dict[str, Any]], Optional[dict]]:
    try:
        spec_data = load_spec(spec_id, specs_dir)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to load spec %s", spec_id)
        return None, asdict(
            error_response(
                f"Failed to load spec '{spec_id}': {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Verify the spec file is accessible",
            )
        )

    if not spec_data:
        return None, asdict(
            error_response(
                f"Specification '{spec_id}' not found",
                error_code=ErrorCode.SPEC_NOT_FOUND,
                error_type=ErrorType.NOT_FOUND,
                remediation='Run spec(action="list") to verify the spec ID',
                details={"spec_id": spec_id, "action": f"journal.{action}"},
            )
        )

    return cast(Dict[str, Any], spec_data), None


def _persist_spec(
    *, spec_id: str, spec_data: Dict[str, Any], specs_dir: Path
) -> Optional[dict]:
    try:
        if not save_spec(spec_id, spec_data, specs_dir):
            return asdict(
                error_response(
                    f"Failed to save spec '{spec_id}'",
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_type=ErrorType.INTERNAL,
                    remediation="Check filesystem permissions and retry",
                )
            )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Failed to persist spec %s", spec_id)
        return asdict(
            error_response(
                f"Failed to save spec '{spec_id}': {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check filesystem permissions and retry",
            )
        )

    return None


def _validate_string(
    value: Any,
    *,
    field: str,
    action: str,
    required: bool = False,
    allow_empty: bool = False,
) -> Tuple[Optional[str], Optional[dict]]:
    if value is None:
        if required:
            return None, _missing_field(field, action)
        return None, None

    if not isinstance(value, str):
        return None, _validation_error(field, action, "Expected a string value")

    if not allow_empty and not value.strip():
        return None, _validation_error(
            field, action, "Value must be a non-empty string"
        )

    return value, None


def _validate_entry_type(value: Any, *, action: str) -> Tuple[str, Optional[dict]]:
    entry_type, error = _validate_string(
        value, field="entry_type", action=action, required=False
    )
    if error:
        return "", error

    normalized = (entry_type or "note").strip()
    if normalized not in _ALLOWED_ENTRY_TYPES:
        allowed = ", ".join(_ALLOWED_ENTRY_TYPES)
        return "", _validation_error(
            "entry_type",
            action,
            f"Must be one of: {allowed}",
            remediation=f"Provide one of: {allowed}",
        )

    return normalized, None


def _validate_limit(
    value: Any,
    *,
    field: str,
    action: str,
) -> Tuple[Optional[int], Optional[dict]]:
    if value is None:
        return None, None

    if isinstance(value, bool) or not isinstance(value, int):
        return None, _validation_error(field, action, "Expected an integer value")

    if value <= 0:
        return None, _validation_error(
            field,
            action,
            "Value must be greater than zero",
            remediation="Provide a positive integer",
        )

    return value, None


def _validate_cursor(
    value: Any,
    *,
    field: str,
    action: str,
) -> Tuple[Optional[str], Optional[dict]]:
    if value is None:
        return None, None

    if not isinstance(value, str) or not value.strip():
        return None, _validation_error(
            field,
            action,
            "Cursor must be a non-empty string",
        )

    return value, None


def _validate_add_payload(
    payload: Mapping[str, Any],
) -> Tuple[Optional[JournalAddInput], Optional[dict]]:
    action = "add"
    spec_id, error = _validate_string(
        payload.get("spec_id"), field="spec_id", action=action, required=True
    )
    if error:
        return None, error
    if spec_id is None:
        return None, _missing_field("spec_id", action)
    spec_id = cast(str, spec_id)

    title, error = _validate_string(
        payload.get("title"), field="title", action=action, required=True
    )
    if error:
        return None, error
    if title is None:
        return None, _missing_field("title", action)
    title = cast(str, title)

    content, error = _validate_string(
        payload.get("content"), field="content", action=action, required=True
    )
    if error:
        return None, error
    if content is None:
        return None, _missing_field("content", action)
    content = cast(str, content)

    entry_type, error = _validate_entry_type(payload.get("entry_type"), action=action)
    if error:
        return None, error

    task_id, error = _validate_string(
        payload.get("task_id"),
        field="task_id",
        action=action,
        required=False,
        allow_empty=False,
    )
    if error:
        return None, error

    workspace, error = _validate_string(
        payload.get("workspace"),
        field="workspace",
        action=action,
        required=False,
        allow_empty=False,
    )
    if error:
        return None, error

    return (
        JournalAddInput(
            spec_id=spec_id,
            title=title,
            content=content,
            entry_type=entry_type,
            task_id=task_id,
            workspace=workspace,
        ),
        None,
    )


def _validate_list_payload(
    payload: Mapping[str, Any],
) -> Tuple[Optional[JournalListInput], Optional[dict]]:
    action = "list"
    spec_id, error = _validate_string(
        payload.get("spec_id"), field="spec_id", action=action, required=True
    )
    if error:
        return None, error
    if spec_id is None:
        return None, _missing_field("spec_id", action)

    task_id, error = _validate_string(
        payload.get("task_id"), field="task_id", action=action, required=False
    )
    if error:
        return None, error

    entry_type_raw = payload.get("entry_type")
    entry_type = None
    if entry_type_raw is not None:
        entry_type, error = _validate_entry_type(entry_type_raw, action=action)
        if error:
            return None, error

    limit, error = _validate_limit(payload.get("limit"), field="limit", action=action)
    if error:
        return None, error

    cursor, error = _validate_cursor(
        payload.get("cursor"), field="cursor", action=action
    )
    if error:
        return None, error

    workspace, error = _validate_string(
        payload.get("workspace"), field="workspace", action=action, required=False
    )
    if error:
        return None, error

    return (
        JournalListInput(
            spec_id=spec_id,
            task_id=task_id,
            entry_type=entry_type,
            cursor=cursor,
            limit=limit,
            workspace=workspace,
        ),
        None,
    )


def _validate_list_unjournaled_payload(
    payload: Mapping[str, Any],
) -> Tuple[Optional[JournalListUnjournaledInput], Optional[dict]]:
    action = "list-unjournaled"
    spec_id, error = _validate_string(
        payload.get("spec_id"), field="spec_id", action=action, required=True
    )
    if error:
        return None, error
    if spec_id is None:
        return None, _missing_field("spec_id", action)
    spec_id = cast(str, spec_id)

    limit, error = _validate_limit(payload.get("limit"), field="limit", action=action)
    if error:
        return None, error

    cursor, error = _validate_cursor(
        payload.get("cursor"), field="cursor", action=action
    )
    if error:
        return None, error

    workspace, error = _validate_string(
        payload.get("workspace"), field="workspace", action=action, required=False
    )
    if error:
        return None, error

    return (
        JournalListUnjournaledInput(
            spec_id=spec_id,
            cursor=cursor,
            limit=limit,
            workspace=workspace,
        ),
        None,
    )


def _serialize_entry(entry: Any) -> Dict[str, Any]:
    return {
        "timestamp": getattr(entry, "timestamp", None),
        "entry_type": getattr(entry, "entry_type", None),
        "title": getattr(entry, "title", None),
        "content": getattr(entry, "content", None),
        "author": getattr(entry, "author", None),
        "task_id": getattr(entry, "task_id", None),
    }


def perform_journal_add(
    *,
    config: ServerConfig,
    spec_id: str,
    title: str,
    content: str,
    entry_type: str,
    task_id: Optional[str],
    workspace: Optional[str],
) -> dict:
    specs_dir, error = _resolve_specs_dir(config, workspace)
    if error:
        return error
    assert specs_dir is not None

    spec_data, error = _load_spec_data(
        spec_id=spec_id, specs_dir=specs_dir, action="add"
    )
    if error:
        return error
    assert spec_data is not None

    try:
        entry = add_journal_entry(
            spec_data,
            title=title,
            content=content,
            entry_type=entry_type,
            task_id=task_id,
            author="foundry-mcp",
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Error adding journal entry for %s", spec_id)
        return asdict(
            error_response(
                f"Failed to add journal entry: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
                remediation="Check spec contents and retry",
            )
        )

    error = _persist_spec(spec_id=spec_id, spec_data=spec_data, specs_dir=specs_dir)
    if error:
        return error

    data = {
        "spec_id": spec_id,
        "entry": {
            "timestamp": entry.timestamp,
            "entry_type": entry.entry_type,
            "title": entry.title,
            "task_id": entry.task_id,
        },
    }

    return asdict(success_response(data=data))


def perform_journal_list(
    *,
    config: ServerConfig,
    spec_id: str,
    task_id: Optional[str],
    entry_type: Optional[str],
    cursor: Optional[str],
    limit: Optional[int],
    workspace: Optional[str],
) -> dict:
    specs_dir, error = _resolve_specs_dir(config, workspace)
    if error:
        return error
    assert specs_dir is not None

    spec_data, error = _load_spec_data(
        spec_id=spec_id, specs_dir=specs_dir, action="list"
    )
    if error:
        return error
    assert spec_data is not None

    page_size = normalize_page_size(limit)
    start_after_ts = None
    if cursor:
        try:
            decoded = decode_cursor(cursor)
            start_after_ts = decoded.get("last_ts")
        except CursorError as exc:
            return asdict(
                error_response(
                    f"Invalid cursor: {exc}",
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use the cursor returned by the previous response",
                )
            )

    try:
        entries = get_journal_entries(
            spec_data,
            task_id=task_id,
            entry_type=entry_type,
            limit=None,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Error retrieving journal entries for %s", spec_id)
        return asdict(
            error_response(
                f"Failed to fetch journal entries: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    entries.sort(key=lambda e: getattr(e, "timestamp", ""), reverse=True)

    if start_after_ts:
        start_index = 0
        for idx, entry in enumerate(entries):
            if getattr(entry, "timestamp", None) == start_after_ts:
                start_index = idx + 1
                break
        entries = entries[start_index:]

    page_entries = entries[: page_size + 1]
    has_more = len(page_entries) > page_size
    if has_more:
        page_entries = page_entries[:page_size]

    next_cursor = None
    if has_more and page_entries:
        next_cursor = encode_cursor(
            {"last_ts": getattr(page_entries[-1], "timestamp", None)}
        )

    warnings = None
    if has_more:
        warnings = [
            f"Results truncated after {page_size} entries. Use the returned cursor to continue."
        ]

    data = {
        "spec_id": spec_id,
        "count": len(page_entries),
        "entries": [_serialize_entry(entry) for entry in page_entries],
    }

    pagination = {
        "cursor": next_cursor,
        "has_more": has_more,
        "page_size": page_size,
    }

    return asdict(
        success_response(
            data=data,
            pagination=pagination,
            warnings=warnings,
        )
    )


def perform_journal_list_unjournaled(
    *,
    config: ServerConfig,
    spec_id: str,
    cursor: Optional[str],
    limit: Optional[int],
    workspace: Optional[str],
) -> dict:
    specs_dir, error = _resolve_specs_dir(config, workspace)
    if error:
        return error
    assert specs_dir is not None

    spec_data, error = _load_spec_data(
        spec_id=spec_id, specs_dir=specs_dir, action="list-unjournaled"
    )
    if error:
        return error
    assert spec_data is not None

    page_size = normalize_page_size(limit)
    start_after_id = None
    if cursor:
        try:
            decoded = decode_cursor(cursor)
            start_after_id = decoded.get("last_id")
        except CursorError as exc:
            return asdict(
                error_response(
                    f"Invalid cursor: {exc}",
                    error_code=ErrorCode.INVALID_FORMAT,
                    error_type=ErrorType.VALIDATION,
                    remediation="Use the cursor returned by the previous response",
                )
            )

    try:
        tasks = find_unjournaled_tasks(spec_data)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Error listing unjournaled tasks for %s", spec_id)
        return asdict(
            error_response(
                f"Failed to list unjournaled tasks: {exc}",
                error_code=ErrorCode.INTERNAL_ERROR,
                error_type=ErrorType.INTERNAL,
            )
        )

    tasks.sort(key=lambda task: task.get("task_id", ""))

    if start_after_id:
        start_index = 0
        for idx, task in enumerate(tasks):
            if task.get("task_id") == start_after_id:
                start_index = idx + 1
                break
        tasks = tasks[start_index:]

    page_tasks = tasks[: page_size + 1]
    has_more = len(page_tasks) > page_size
    if has_more:
        page_tasks = page_tasks[:page_size]

    next_cursor = None
    if has_more and page_tasks:
        next_cursor = encode_cursor({"last_id": page_tasks[-1].get("task_id")})

    warnings = None
    if has_more:
        warnings = [
            f"Results truncated after {page_size} tasks. Use the returned cursor to continue."
        ]

    data = {
        "spec_id": spec_id,
        "count": len(page_tasks),
        "unjournaled_tasks": page_tasks,
    }

    pagination = {
        "cursor": next_cursor,
        "has_more": has_more,
        "page_size": page_size,
    }

    return asdict(
        success_response(
            data=data,
            pagination=pagination,
            warnings=warnings,
        )
    )


def _handle_journal_add(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_add_payload(payload)
    if error:
        return error

    assert validated is not None
    return perform_journal_add(
        config=config,
        spec_id=validated.spec_id,
        title=validated.title,
        content=validated.content,
        entry_type=validated.entry_type,
        task_id=validated.task_id,
        workspace=validated.workspace,
    )


def _handle_journal_list(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_list_payload(payload)
    if error:
        return error

    assert validated is not None
    return perform_journal_list(
        config=config,
        spec_id=validated.spec_id,
        task_id=validated.task_id,
        entry_type=validated.entry_type,
        cursor=validated.cursor,
        limit=validated.limit,
        workspace=validated.workspace,
    )


def _handle_journal_list_unjournaled(*, config: ServerConfig, **payload: Any) -> dict:
    validated, error = _validate_list_unjournaled_payload(payload)
    if error:
        return error

    assert validated is not None
    return perform_journal_list_unjournaled(
        config=config,
        spec_id=validated.spec_id,
        cursor=validated.cursor,
        limit=validated.limit,
        workspace=validated.workspace,
    )


_JOURNAL_ROUTER = ActionRouter(
    tool_name="journal",
    actions=[
        ActionDefinition(
            name="add",
            handler=_handle_journal_add,
            summary=_ACTION_SUMMARY["add"],
        ),
        ActionDefinition(
            name="list",
            handler=_handle_journal_list,
            summary=_ACTION_SUMMARY["list"],
        ),
        ActionDefinition(
            name="list-unjournaled",
            handler=_handle_journal_list_unjournaled,
            summary=_ACTION_SUMMARY["list-unjournaled"],
        ),
    ],
)


def _dispatch_journal_action(
    *, action: str, payload: Dict[str, Any], config: ServerConfig
) -> dict:
    try:
        return _JOURNAL_ROUTER.dispatch(action=action, config=config, **payload)
    except ActionRouterError as exc:
        allowed = ", ".join(exc.allowed_actions)
        return asdict(
            error_response(
                f"Unsupported journal action '{action}'. Allowed actions: {allowed}",
                error_code=ErrorCode.VALIDATION_ERROR,
                error_type=ErrorType.VALIDATION,
                remediation=f"Use one of: {allowed}",
            )
        )


def register_unified_journal_tool(mcp: FastMCP, config: ServerConfig) -> None:
    """Register the consolidated journal tool."""

    @canonical_tool(
        mcp,
        canonical_name="journal",
    )
    def journal(
        action: str,
        spec_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        entry_type: Optional[str] = None,
        task_id: Optional[str] = None,
        workspace: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> dict:
        payload = {
            "spec_id": spec_id,
            "title": title,
            "content": content,
            "entry_type": entry_type,
            "task_id": task_id,
            "workspace": workspace,
            "cursor": cursor,
            "limit": limit,
        }
        return _dispatch_journal_action(action=action, payload=payload, config=config)

    logger.debug("Registered unified journal tool")


__all__ = [
    "register_unified_journal_tool",
]
