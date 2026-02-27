"""Audit trail for deep research workflow.

Writes JSONL audit events for observability and debugging of
deep research sessions, with configurable verbosity levels.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from foundry_mcp.config.research import ResearchConfig
    from foundry_mcp.core.research.memory import ResearchMemory
from uuid import uuid4

from foundry_mcp.core.research.models.deep_research import DeepResearchState

logger = logging.getLogger(__name__)


class AuditMixin:
    """Mixin providing audit trail capabilities for deep research.

    Requires the composing class to provide:
    - self.config: ResearchConfig (with audit_verbosity, deep_research_audit_artifacts)
    - self.memory: ResearchMemory (with base_path)

    See ``DeepResearchWorkflowProtocol`` in ``phases/_protocols.py`` for
    the full structural contract.
    """

    config: ResearchConfig
    memory: ResearchMemory

    def _audit_enabled(self) -> bool:
        """Return True if audit artifacts are enabled."""
        return bool(self.config.deep_research_audit_artifacts)

    def _audit_path(self, research_id: str) -> Path:
        """Resolve audit artifact path for a research session."""
        # Use memory's base_path which is set from ServerConfig.get_research_dir()
        return self.memory.base_path / "deep_research" / f"{research_id}.audit.jsonl"

    def _prepare_audit_payload(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare audit payload based on configured verbosity level.

        In 'full' mode: Returns data unchanged for complete audit trail.
        In 'minimal' mode: Sets large text fields to null while preserving
        metrics and schema shape for analysis compatibility.

        Nulled fields in minimal mode:
        - Top-level: system_prompt, user_prompt, raw_response, report, error, traceback
        - Nested: findings[*].content, gaps[*].description

        Preserved fields (always included):
        - provider_id, model_used, tokens_used, duration_ms
        - sources_added, report_length, parse_success
        - All other scalar metrics

        Args:
            data: Original audit event data dictionary

        Returns:
            Processed data dictionary (same schema shape, potentially nulled values)
        """
        verbosity = self.config.audit_verbosity

        # Full mode: return unchanged
        if verbosity == "full":
            return data

        # Minimal mode: null out large text fields while preserving schema
        result = dict(data)  # Shallow copy

        # Top-level fields to null
        text_fields = {
            "system_prompt",
            "user_prompt",
            "raw_response",
            "report",
            "error",
            "traceback",
        }
        for field in text_fields:
            if field in result:
                result[field] = None

        # Handle nested findings array
        if "findings" in result and isinstance(result["findings"], list):
            result["findings"] = [
                {**f, "content": None} if isinstance(f, dict) and "content" in f else f for f in result["findings"]
            ]

        # Handle nested gaps array
        if "gaps" in result and isinstance(result["gaps"], list):
            result["gaps"] = [
                {**g, "description": None} if isinstance(g, dict) and "description" in g else g for g in result["gaps"]
            ]

        return result

    def _write_audit_event(
        self,
        state: Optional[DeepResearchState],
        event_type: str,
        data: Optional[dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """Write a JSONL audit event for deep research observability."""
        if not self._audit_enabled():
            return

        research_id = state.id if state else None
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_id": uuid4().hex,
            "event_type": event_type,
            "level": level,
            "research_id": research_id,
            "phase": state.phase.value if state else None,
            "iteration": state.iteration if state else None,
            "data": self._prepare_audit_payload(data or {}),
        }

        try:
            if research_id is None:
                return
            path = self._audit_path(research_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")
        except Exception as exc:
            logger.error("Failed to write audit event: %s", exc)
            # Fallback to stderr for crash visibility
            print(
                f"AUDIT_FALLBACK: {event_type} for {research_id} - {exc}",
                file=sys.stderr,
                flush=True,
            )
