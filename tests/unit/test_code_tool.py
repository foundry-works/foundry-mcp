"""Unit tests for the unified `code(action=...)` router."""

from __future__ import annotations

from pathlib import Path

from foundry_mcp.config import ServerConfig
from foundry_mcp.tools.unified.code import _dispatch_code_action


def test_python_call_graph_respects_depth(tmp_path: Path, response_validator) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        """

def foo():
    return 1


def bar():
    return foo()


def baz():
    return bar()
""".lstrip()
    )

    config = ServerConfig()

    response_depth_1 = _dispatch_code_action(
        action="callees",
        config=config,
        payload={
            "symbol": "baz",
            "language": "python",
            "workspace": str(tmp_path),
            "path_allowlist": ["."],
            "depth": 1,
            "max_nodes": 50,
        },
    )
    response_validator(response_depth_1)
    assert response_depth_1["success"] is True

    symbols_depth_1 = {node["symbol"] for node in response_depth_1["data"]["nodes"]}
    assert "baz" in symbols_depth_1
    assert "bar" in symbols_depth_1
    assert "foo" not in symbols_depth_1

    response_depth_2 = _dispatch_code_action(
        action="callees",
        config=config,
        payload={
            "symbol": "baz",
            "language": "python",
            "workspace": str(tmp_path),
            "path_allowlist": ["."],
            "depth": 2,
            "max_nodes": 50,
        },
    )
    response_validator(response_depth_2)
    assert response_depth_2["success"] is True

    symbols_depth_2 = {node["symbol"] for node in response_depth_2["data"]["nodes"]}
    assert "foo" in symbols_depth_2


def test_invalid_depth_is_validation_error(tmp_path: Path, response_validator) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("def foo():\n    return 1\n")

    response = _dispatch_code_action(
        action="callees",
        config=ServerConfig(),
        payload={
            "symbol": "foo",
            "language": "python",
            "workspace": str(tmp_path),
            "path_allowlist": ["."],
            "depth": "nope",
        },
    )

    response_validator(response)
    assert response["success"] is False
    assert response["meta"].get("request_id")
    assert response["error"]
