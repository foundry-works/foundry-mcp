"""Tests for canonical task session action-shape adapters."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from foundry_mcp.tools.unified.task_handlers._helpers import (
    _normalize_task_action_shape,
)


@pytest.mark.parametrize(
    ("action", "command", "expected"),
    [
        ("session", "start", "session-start"),
        ("session", "status", "session-status"),
        ("session", "pause", "session-pause"),
        ("session", "resume", "session-resume"),
        ("session", "rebase", "session-rebase"),
        ("session", "end", "session-end"),
        ("session", "list", "session-list"),
        ("session", "reset", "session-reset"),
        ("session-step", "next", "session-step-next"),
        ("session-step", "report", "session-step-report"),
        ("session-step", "replay", "session-step-replay"),
        ("session-step", "heartbeat", "session-step-heartbeat"),
    ],
)
def test_normalize_canonical_actions(action: str, command: str, expected: str) -> None:
    mapped_action, _, deprecation, error = _normalize_task_action_shape(
        action=action,
        payload={"command": command},
        request_id="req-test",
    )
    assert error is None
    assert deprecation is None
    assert mapped_action == expected


@pytest.mark.parametrize("action", ["session", "session-step"])
def test_normalize_canonical_action_requires_command(action: str) -> None:
    _, _, _, error = _normalize_task_action_shape(
        action=action,
        payload={},
        request_id="req-test",
    )
    assert error is not None
    assert error["success"] is False
    assert error["data"]["error_code"] == "MISSING_REQUIRED"


@pytest.mark.parametrize(
    ("action", "command"),
    [("session", "invalid"), ("session-step", "invalid")],
)
def test_normalize_canonical_action_rejects_unknown_command(action: str, command: str) -> None:
    _, _, _, error = _normalize_task_action_shape(
        action=action,
        payload={"command": command},
        request_id="req-test",
    )
    assert error is not None
    assert error["success"] is False
    assert error["data"]["error_code"] == "INVALID_FORMAT"


def _mock_config() -> MagicMock:
    config = MagicMock()
    config.specs_dir = None
    return config


def test_dispatch_session_canonical_maps_to_legacy_action() -> None:
    from foundry_mcp.tools.unified.task_handlers import _dispatch_task_action

    with patch(
        "foundry_mcp.tools.unified.task_handlers.dispatch_with_standard_errors"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "success": True,
            "data": {"ok": True},
            "error": None,
            "meta": {"version": "response-v2"},
        }

        response = _dispatch_task_action(
            action="session",
            payload={"command": "start", "spec_id": "spec-001"},
            config=_mock_config(),
        )

    args, kwargs = mock_dispatch.call_args
    assert args[2] == "session-start"
    assert kwargs["spec_id"] == "spec-001"
    assert response["success"] is True
    assert "deprecated" not in response["meta"]


def test_dispatch_legacy_action_includes_deprecation_metadata() -> None:
    from foundry_mcp.tools.unified.task_handlers import _dispatch_task_action

    with patch(
        "foundry_mcp.tools.unified.task_handlers.dispatch_with_standard_errors"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "success": True,
            "data": {"ok": True},
            "error": None,
            "meta": {"version": "response-v2"},
        }

        response = _dispatch_task_action(
            action="session-start",
            payload={"spec_id": "spec-001"},
            config=_mock_config(),
        )

    deprecated = response["meta"].get("deprecated")
    assert deprecated is not None
    assert deprecated["action"] == "session-start"
    assert deprecated["replacement"] == 'task(action="session", command="start")'
    assert deprecated["removal_target"] == "2026-05-16_or_2_minor_releases"


def test_dispatch_legacy_action_emits_warn_log(caplog) -> None:
    from foundry_mcp.tools.unified.task_handlers import _dispatch_task_action

    with patch(
        "foundry_mcp.tools.unified.task_handlers.dispatch_with_standard_errors"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "success": True,
            "data": {"ok": True},
            "error": None,
            "meta": {"version": "response-v2"},
        }

        with caplog.at_level("WARNING"):
            _dispatch_task_action(
                action="session-start",
                payload={"spec_id": "spec-001"},
                config=_mock_config(),
            )

    assert "Deprecated task action invoked" in caplog.text
    assert "session-start" in caplog.text
    assert 'task(action="session", command="start")' in caplog.text


def test_dispatch_shim_includes_deprecation_metadata() -> None:
    from foundry_mcp.tools.unified.task import _dispatch_task_action

    with patch(
        "foundry_mcp.tools.unified.task.dispatch_with_standard_errors"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "success": False,
            "data": {"error_code": "CONFLICT", "error_type": "conflict"},
            "error": "conflict",
            "meta": {"version": "response-v2"},
        }

        response = _dispatch_task_action(
            action="session-step-next",
            payload={"session_id": "sess-001"},
            config=_mock_config(),
        )

    deprecated = response["meta"].get("deprecated")
    assert deprecated is not None
    assert deprecated["action"] == "session-step-next"
    assert deprecated["replacement"] == 'task(action="session-step", command="next")'


def test_dispatch_session_step_authorization_denial_includes_loop_signal() -> None:
    from foundry_mcp.tools.unified.task_handlers import _dispatch_task_action

    with patch(
        "foundry_mcp.tools.unified.task_handlers.dispatch_with_standard_errors"
    ) as mock_dispatch:
        mock_dispatch.return_value = {
            "success": False,
            "data": {
                "error_code": "AUTHORIZATION",
                "error_type": "authorization",
                "details": {"action": "task.session-step-next"},
            },
            "error": "Authorization denied",
            "meta": {"version": "response-v2"},
        }

        response = _dispatch_task_action(
            action="session-step-next",
            payload={"session_id": "sess-001"},
            config=_mock_config(),
        )

    assert response["data"]["loop_signal"] == "blocked_runtime"
    assert response["data"]["recommended_actions"]


@pytest.mark.parametrize(
    "example_input",
    [
        {"action": "session", "command": "start", "spec_id": "my-feature-spec-001"},
        {"action": "session-step", "command": "next", "session_id": "01HXEXAMPLESESSION"},
        {"action": "session-events", "session_id": "01HXEXAMPLESESSION", "limit": 25},
        {"action": "session-start", "spec_id": "my-feature-spec-001"},
    ],
)
def test_docs_task_examples_supported_by_action_shape_adapter(example_input: dict) -> None:
    payload = dict(example_input)
    action = payload.pop("action")
    mapped_action, _, _, error = _normalize_task_action_shape(
        action=action,
        payload=payload,
        request_id="req-doc-example",
    )
    assert error is None
    assert mapped_action


def test_manifest_task_examples_supported_by_action_shape_adapter() -> None:
    manifest_path = Path(__file__).resolve().parents[3] / "mcp" / "capabilities_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_tool = next(
        tool for tool in manifest["tools"]["unified"] if tool["name"] == "task"
    )

    for example in task_tool.get("examples", []):
        input_payload = dict(example["input"])
        action = input_payload.pop("action")
        mapped_action, _, _, error = _normalize_task_action_shape(
            action=action,
            payload=input_payload,
            request_id="req-manifest-example",
        )
        assert error is None, example["description"]
        assert mapped_action, example["description"]
