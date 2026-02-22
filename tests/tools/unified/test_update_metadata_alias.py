"""Tests for update_metadata → custom_metadata alias remap in task update-metadata action."""

from unittest.mock import MagicMock, patch

import pytest


def _make_payload(**overrides):
    """Build a minimal update-metadata payload."""
    base = {
        "spec_id": "test-spec",
        "task_id": "task-1",
        "dry_run": False,
        "title": "Updated title",
    }
    base.update(overrides)
    return base


class TestUpdateMetadataAlias:
    """Tests for the update_metadata → custom_metadata alias remap."""

    def test_alias_remapped_when_custom_metadata_absent(self, mock_config):
        """update_metadata should be remapped to custom_metadata."""
        from foundry_mcp.tools.unified.task_handlers.handlers_mutation import _handle_update_metadata

        captured_payloads = []

        original_validate = None

        def capture_validate(payload, *a, **kw):
            captured_payloads.append(dict(payload))
            # Return an error to short-circuit the handler after validation
            from dataclasses import asdict

            from foundry_mcp.core.responses.builders import error_response

            return asdict(error_response("short-circuit"))

        payload = _make_payload(update_metadata={"key": "value"})

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_mutation.validate_payload",
            side_effect=capture_validate,
        ):
            _handle_update_metadata(config=mock_config, **payload)

        assert len(captured_payloads) == 1
        p = captured_payloads[0]
        assert "custom_metadata" in p
        assert p["custom_metadata"] == {"key": "value"}
        assert "update_metadata" not in p

    def test_custom_metadata_takes_precedence_when_both_present(self, mock_config):
        """When both present, custom_metadata is preserved and update_metadata is not popped."""
        from foundry_mcp.tools.unified.task_handlers.handlers_mutation import _handle_update_metadata

        captured_payloads = []

        def capture_validate(payload, *a, **kw):
            captured_payloads.append(dict(payload))
            from dataclasses import asdict

            from foundry_mcp.core.responses.builders import error_response

            return asdict(error_response("short-circuit"))

        payload = _make_payload(
            update_metadata={"alias": True},
            custom_metadata={"original": True},
        )

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_mutation.validate_payload",
            side_effect=capture_validate,
        ):
            _handle_update_metadata(config=mock_config, **payload)

        assert len(captured_payloads) == 1
        p = captured_payloads[0]
        # custom_metadata should be the original, not overwritten
        assert p["custom_metadata"] == {"original": True}
        # update_metadata should still be present (remap skipped)
        assert "update_metadata" in p

    def test_normal_custom_metadata_usage_unchanged(self, mock_config):
        """Normal custom_metadata usage (no update_metadata) works unchanged."""
        from foundry_mcp.tools.unified.task_handlers.handlers_mutation import _handle_update_metadata

        captured_payloads = []

        def capture_validate(payload, *a, **kw):
            captured_payloads.append(dict(payload))
            from dataclasses import asdict

            from foundry_mcp.core.responses.builders import error_response

            return asdict(error_response("short-circuit"))

        payload = _make_payload(custom_metadata={"normal": True})

        with patch(
            "foundry_mcp.tools.unified.task_handlers.handlers_mutation.validate_payload",
            side_effect=capture_validate,
        ):
            _handle_update_metadata(config=mock_config, **payload)

        assert len(captured_payloads) == 1
        p = captured_payloads[0]
        assert p["custom_metadata"] == {"normal": True}
        assert "update_metadata" not in p
