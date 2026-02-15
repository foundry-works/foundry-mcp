"""Unit tests for unified review surface.

We keep these tests lightweight: verify that helper functions behave and that
`review(action=...)` is registered on the server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_mcp.config import ServerConfig
from foundry_mcp.core.authorization import set_server_role
from foundry_mcp.server import create_server
from tests.conftest import extract_response_dict


@pytest.fixture(autouse=True)
def maintainer_role():
    set_server_role("maintainer")
    yield
    set_server_role("observer")


def test_get_llm_status_handles_import_error(monkeypatch):
    from foundry_mcp.tools.unified.review_helpers import _get_llm_status

    def _raise():
        raise ImportError("missing")

    monkeypatch.setattr("foundry_mcp.core.llm_config.get_llm_config", _raise)
    status = _get_llm_status()
    assert status["configured"] is False


@pytest.fixture
def test_config(tmp_path: Path) -> ServerConfig:
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()

    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=specs_dir,
        log_level="WARNING",
    )


def test_review_tool_registered(test_config: ServerConfig):
    server = create_server(test_config)
    tools = server._tool_manager._tools
    assert "review" in tools
    assert callable(tools["review"].fn)


def test_review_list_tools_returns_envelope(test_config: ServerConfig):
    server = create_server(test_config)
    tools = server._tool_manager._tools

    set_server_role("maintainer")
    try:
        result = extract_response_dict(tools["review"].fn(action="list-tools"))
    finally:
        set_server_role("observer")
    assert result["success"] is True
    assert "tools" in result["data"]
    assert "review_types" in result["data"]


def test_fidelity_gate_uses_runtime_feature_flag(test_config: ServerConfig):
    from foundry_mcp.tools.unified.review import _dispatch_review_action

    disabled = _dispatch_review_action(
        action="fidelity-gate",
        payload={},
        config=test_config,
    )
    assert disabled["success"] is False
    assert disabled["data"]["error_code"] == "FEATURE_DISABLED"

    test_config.feature_flags["autonomy_fidelity_gates"] = True
    enabled = _dispatch_review_action(
        action="fidelity-gate",
        payload={"spec_id": "spec-001"},
        config=test_config,
    )
    assert enabled["success"] is False
    assert enabled["data"]["error_code"] != "FEATURE_DISABLED"
