"""Integration tests for unified MCP tool surface.

Covers:
- Tool registration + callable schemas
- Basic tool IO shape for a few core actions
- Resource access patterns (foundry:// URIs)
- Prompt expansion with various arguments
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foundry_mcp.config import ServerConfig
from foundry_mcp.server import create_server


@pytest.fixture
def test_specs_dir(tmp_path: Path) -> Path:
    specs_dir = tmp_path / "specs"
    specs_dir.mkdir()
    (specs_dir / "active").mkdir()
    (specs_dir / "pending").mkdir()
    (specs_dir / "completed").mkdir()
    (specs_dir / "archived").mkdir()
    (specs_dir / "templates").mkdir()

    sample_spec = {
        "spec_id": "test-spec-001",
        "title": "Test Specification",
        "metadata": {
            "title": "Test Specification",
            "description": "A test spec for integration testing",
            "created_at": "2025-01-25T00:00:00Z",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "Test Specification",
                "status": "in_progress",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Implementation",
                "status": "in_progress",
                "parent": "spec-root",
                "children": ["task-1-1", "task-1-2"],
            },
            "task-1-1": {
                "type": "task",
                "title": "First task",
                "status": "completed",
                "parent": "phase-1",
                "children": [],
            },
            "task-1-2": {
                "type": "task",
                "title": "Second task",
                "status": "pending",
                "parent": "phase-1",
                "children": [],
            },
        },
        "journal": [
            {
                "timestamp": "2025-01-25T00:00:00Z",
                "entry_type": "status_change",
                "title": "Task completed",
                "content": "Completed first task",
                "task_id": "task-1-1",
            },
        ],
    }

    spec_file = specs_dir / "active" / "test-spec-001.json"
    spec_file.write_text(json.dumps(sample_spec), encoding="utf-8")

    template = {
        "spec_id": "{{spec_id}}",
        "title": "Custom Template",
        "metadata": {
            "title": "{{title}}",
            "description": "Custom template for testing",
        },
        "hierarchy": {
            "spec-root": {
                "type": "spec",
                "title": "{{title}}",
                "status": "pending",
                "children": ["phase-1"],
            },
            "phase-1": {
                "type": "phase",
                "title": "Custom Phase",
                "status": "pending",
                "parent": "spec-root",
                "children": [],
            },
        },
        "journal": [],
    }

    template_file = specs_dir / "templates" / "custom.json"
    template_file.write_text(json.dumps(template), encoding="utf-8")

    return specs_dir


@pytest.fixture
def test_config(test_specs_dir: Path) -> ServerConfig:
    return ServerConfig(
        server_name="foundry-mcp-test",
        server_version="0.1.0",
        specs_dir=test_specs_dir,
        log_level="WARNING",
    )


@pytest.fixture
def mcp_server(test_config: ServerConfig):
    return create_server(test_config)


def test_tools_registered(mcp_server):
    tools = mcp_server._tool_manager._tools
    assert "spec" in tools
    assert "task" in tools
    assert "server" in tools


def test_all_tools_have_callable(mcp_server):
    tools = mcp_server._tool_manager._tools
    for tool_name, tool in tools.items():
        assert callable(tool.fn), f"Tool {tool_name} should be callable"


def test_spec_list_returns_envelope(mcp_server):
    tools = mcp_server._tool_manager._tools
    result = tools["spec"].fn(action="list", status="all")
    assert result["success"] is True
    assert "specs" in result["data"]
    assert "count" in result["data"]


def test_task_hierarchy_returns_hierarchy(mcp_server):
    tools = mcp_server._tool_manager._tools
    result = tools["task"].fn(action="hierarchy", spec_id="test-spec-001")
    assert result["success"] is True
    assert "hierarchy" in result["data"]
    assert "spec-root" in result["data"]["hierarchy"]


class TestResourceAccess:
    def test_foundry_specs_resource_registered(self, mcp_server):
        resources = mcp_server._resource_manager._resources
        assert any("foundry://specs/" in str(uri) for uri in resources)

    def test_specs_list_resource_returns_json(self, mcp_server):
        resources = mcp_server._resource_manager._resources
        for uri, resource in resources.items():
            if (
                "foundry://specs/" in str(uri)
                and resource.fn.__name__ == "resource_specs_list"
            ):
                payload = json.loads(resource.fn())
                assert "success" in payload
                assert "schema_version" in payload
                return
        raise AssertionError("resource_specs_list not found")


class TestPromptExpansion:
    def test_prompts_registered(self, mcp_server):
        prompts = mcp_server._prompt_manager._prompts
        assert len(prompts) > 0

    def test_start_feature_prompt_exists(self, mcp_server):
        assert "start_feature" in mcp_server._prompt_manager._prompts
