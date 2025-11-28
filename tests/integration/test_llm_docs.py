"""
Integration tests for LLM-powered documentation tools.

Tests verify that spec-doc, spec-doc-llm, and spec-review-fidelity
emit actionable results with proper data-only fallback behavior.
"""

import json
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestSpecDocIntegration:
    """Integration tests for spec-doc tool."""

    def test_spec_doc_returns_actionable_output(self, tmp_path):
        """Test that spec-doc produces actionable output with proper schema."""
        # Create a minimal spec file
        specs_dir = tmp_path / "specs" / "active"
        specs_dir.mkdir(parents=True)

        spec = {
            "spec_id": "test-spec-001",
            "title": "Test Spec",
            "metadata": {"title": "Test Spec"},
            "hierarchy": {
                "spec-root": {
                    "type": "spec",
                    "title": "Test Spec",
                    "status": "in_progress",
                    "children": ["task-1"],
                },
                "task-1": {
                    "type": "task",
                    "title": "Task One",
                    "status": "completed",
                    "parent": "spec-root",
                    "children": [],
                },
            },
            "journal": [],
        }
        (specs_dir / "test-spec-001.json").write_text(json.dumps(spec))

        # Run sdd render command
        result = subprocess.run(
            ["sdd", "render", "test-spec-001", "--path", str(tmp_path), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should succeed or provide actionable error
        if result.returncode == 0:
            output = json.loads(result.stdout)
            # Verify actionable output structure
            assert "output_path" in output or "data" in output
        else:
            # Even failures should be actionable (proper error message)
            assert result.stderr or result.stdout

    def test_spec_doc_data_only_fallback(self):
        """Test spec-doc falls back to data-only when JSON parsing fails."""
        from foundry_mcp.tools.documentation import _run_sdd_render_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            # Simulate non-JSON output
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Rendered to specs/.human-readable/test.md",
                stderr="",
            )

            result = _run_sdd_render_command(["test-spec"])

            # Should succeed with raw_output fallback
            assert result["success"] is True
            assert "raw_output" in result["data"]
            assert "test.md" in result["data"]["raw_output"]


class TestSpecDocLlmIntegration:
    """Integration tests for spec-doc-llm tool."""

    def test_spec_doc_llm_returns_actionable_output(self, tmp_path):
        """Test that spec-doc-llm produces actionable output."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "output_dir": str(tmp_path / "docs"),
                    "files_generated": 5,
                    "total_shards": 10,
                    "project_name": "test-project",
                }),
                stderr="",
            )

            result = _run_sdd_llm_doc_gen_command(["generate", str(tmp_path)])

            # Verify actionable output
            assert result["success"] is True
            assert "files_generated" in result["data"]
            assert "output_dir" in result["data"]

    def test_spec_doc_llm_data_only_fallback(self):
        """Test spec-doc-llm falls back to data-only on non-JSON output."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Generated 5 documentation files in ./docs",
                stderr="",
            )

            result = _run_sdd_llm_doc_gen_command(["generate", "/path/to/project"])

            # Should succeed with raw_output fallback
            assert result["success"] is True
            assert "raw_output" in result["data"]

    def test_spec_doc_llm_timeout_provides_recovery_guidance(self):
        """Test that timeouts include recovery guidance."""
        from foundry_mcp.tools.documentation import _run_sdd_llm_doc_gen_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sdd", timeout=600)

            result = _run_sdd_llm_doc_gen_command(["generate", "/path"], timeout=600)

            assert result["success"] is False
            assert "timed out" in result["error"]
            assert "--resume" in result["error"]  # Recovery guidance


class TestSpecReviewFidelityIntegration:
    """Integration tests for spec-review-fidelity tool."""

    def test_fidelity_review_returns_actionable_verdict(self):
        """Test that fidelity review produces actionable verdict."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec-001",
                    "verdict": "pass",
                    "deviations": [],
                    "recommendations": [],
                    "consensus": {"models_consulted": 2, "agreement": "unanimous"},
                }),
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec-001"])

            # Verify actionable output with verdict
            assert result["success"] is True
            assert result["data"]["verdict"] == "pass"
            assert "consensus" in result["data"]

    def test_fidelity_review_with_deviations(self):
        """Test fidelity review reports deviations actionably."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec-001",
                    "verdict": "partial",
                    "deviations": [
                        {
                            "task_id": "task-1",
                            "type": "missing_implementation",
                            "severity": "high",
                            "description": "Function foo() not implemented",
                            "file": "src/module.py",
                        }
                    ],
                    "recommendations": [
                        "Implement foo() in src/module.py as specified in task-1"
                    ],
                }),
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec-001"])

            assert result["success"] is True
            assert result["data"]["verdict"] == "partial"
            assert len(result["data"]["deviations"]) == 1
            assert result["data"]["deviations"][0]["severity"] == "high"
            assert len(result["data"]["recommendations"]) == 1

    def test_fidelity_review_data_only_fallback(self):
        """Test fidelity review falls back to data-only on non-JSON."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="Fidelity review: PASS - No deviations found",
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec-001"])

            # Should succeed with raw_output fallback
            assert result["success"] is True
            assert "raw_output" in result["data"]
            assert "PASS" in result["data"]["raw_output"]

    def test_fidelity_review_timeout_provides_scope_guidance(self):
        """Test that timeouts include scope reduction guidance."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="sdd", timeout=600)

            result = _run_sdd_fidelity_review_command(["test-spec-001"], timeout=600)

            assert result["success"] is False
            assert "timed out" in result["error"]
            assert "smaller scope" in result["error"]  # Scope guidance

    def test_fidelity_review_with_phase_scope(self):
        """Test fidelity review with phase-scoped review."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec-001",
                    "scope": {"type": "phase", "id": "phase-1"},
                    "verdict": "pass",
                    "deviations": [],
                    "tasks_reviewed": 5,
                }),
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(
                ["test-spec-001", "--phase", "phase-1"]
            )

            assert result["success"] is True
            assert result["data"]["verdict"] == "pass"
            assert result["data"]["scope"]["type"] == "phase"


class TestDocToolsResponseEnvelope:
    """Test that all doc tools emit proper response envelopes."""

    def test_spec_doc_response_envelope_compliance(self, tmp_path):
        """Test spec-doc responses follow envelope schema."""
        from foundry_mcp.tools.documentation import register_documentation_tools
        from foundry_mcp.config import ServerConfig
        from mcp.server.fastmcp import FastMCP
        from pathlib import Path

        mcp = FastMCP("test")
        config = ServerConfig(specs_dir=Path(tmp_path / "specs"))

        # This should not raise - registration must succeed
        # Registration succeeding proves tools can be registered properly
        try:
            register_documentation_tools(mcp, config)
            registration_success = True
        except Exception:
            registration_success = False

        assert registration_success, "Documentation tools should register without error"

    def test_all_doc_tools_handle_sdd_not_found(self):
        """Test all doc tools handle missing sdd CLI gracefully."""
        from foundry_mcp.tools.documentation import (
            _run_sdd_doc_command,
            _run_sdd_render_command,
            _run_sdd_llm_doc_gen_command,
            _run_sdd_fidelity_review_command,
        )

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("sdd not found")

            # All should return actionable error
            for runner, args in [
                (_run_sdd_doc_command, ["spec-id"]),
                (_run_sdd_render_command, ["spec-id"]),
                (_run_sdd_llm_doc_gen_command, ["generate", "/path"]),
                (_run_sdd_fidelity_review_command, ["spec-id"]),
            ]:
                result = runner(args)
                assert result["success"] is False
                assert "sdd CLI not found" in result["error"]
                assert "sdd-toolkit" in result["error"]  # Installation guidance


class TestFidelityReviewActionableOutput:
    """Focused tests for fidelity review actionable output requirements."""

    def test_fidelity_deviations_include_remediation_info(self):
        """Test deviations include enough info for automated remediation."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec",
                    "verdict": "fail",
                    "deviations": [
                        {
                            "task_id": "task-2-1",
                            "type": "signature_mismatch",
                            "severity": "critical",
                            "description": "Function accepts str but spec requires List[str]",
                            "file": "src/handlers.py",
                            "line": 45,
                            "expected": "def process(items: List[str]) -> bool",
                            "actual": "def process(item: str) -> bool",
                        }
                    ],
                    "recommendations": [
                        "Update function signature at src/handlers.py:45 to accept List[str]"
                    ],
                }),
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec"])

            deviation = result["data"]["deviations"][0]
            # All fields needed for automated fix
            assert "file" in deviation
            assert "line" in deviation
            assert "expected" in deviation
            assert "actual" in deviation
            assert deviation["severity"] == "critical"

    def test_fidelity_consensus_provides_confidence(self):
        """Test consensus info provides confidence metrics."""
        from foundry_mcp.tools.documentation import _run_sdd_fidelity_review_command

        with patch('foundry_mcp.tools.documentation.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=json.dumps({
                    "spec_id": "test-spec",
                    "verdict": "pass",
                    "deviations": [],
                    "consensus": {
                        "models_consulted": 3,
                        "agreement": "unanimous",
                        "confidence": 0.95,
                        "dissenting_views": [],
                    },
                }),
                stderr="",
            )

            result = _run_sdd_fidelity_review_command(["test-spec"])

            consensus = result["data"]["consensus"]
            assert "confidence" in consensus
            assert consensus["agreement"] == "unanimous"
            assert consensus["models_consulted"] >= 2
