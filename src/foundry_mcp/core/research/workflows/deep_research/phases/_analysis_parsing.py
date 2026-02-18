"""Response-parsing mixin for the analysis phase.

Parses LLM JSON responses into structured findings, gaps, and quality updates.
Split from ``analysis.py`` to keep each module focused.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.workflows.deep_research._helpers import extract_json

if TYPE_CHECKING:
    from foundry_mcp.core.research.workflows.deep_research.core import (
        DeepResearchWorkflow,
    )

logger = logging.getLogger(__name__)


class AnalysisParsingMixin:
    """Response parsing methods for analysis. Mixed into AnalysisPhaseMixin.

    At runtime, ``self`` is a DeepResearchWorkflow instance.
    """

    def _parse_analysis_response(
        self: DeepResearchWorkflow,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured analysis data.

        Args:
            content: Raw LLM response content
            state: Current research state (reserved for context-aware parsing)

        Returns:
            Dict with 'success', 'findings', 'gaps', and 'quality_updates' keys
        """
        # state is reserved for future context-aware parsing
        _ = state
        result = {
            "success": False,
            "findings": [],
            "gaps": [],
            "quality_updates": [],
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in analysis response")
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from analysis response: %s", e)
            return result

        # Parse findings
        raw_findings = data.get("findings", [])
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if not isinstance(f, dict):
                    continue
                content_text = f.get("content", "").strip()
                if not content_text:
                    continue

                # Map confidence string to enum
                confidence_str = f.get("confidence", "medium").lower()
                confidence_map = {
                    "low": ConfidenceLevel.LOW,
                    "medium": ConfidenceLevel.MEDIUM,
                    "high": ConfidenceLevel.HIGH,
                    "confirmed": ConfidenceLevel.CONFIRMED,
                    "speculation": ConfidenceLevel.SPECULATION,
                }
                confidence = confidence_map.get(confidence_str, ConfidenceLevel.MEDIUM)

                result["findings"].append({
                    "content": content_text,
                    "confidence": confidence,
                    "source_ids": f.get("source_ids", []),
                    "category": f.get("category"),
                })

        # Parse gaps
        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if not isinstance(g, dict):
                    continue
                description = g.get("description", "").strip()
                if not description:
                    continue

                result["gaps"].append({
                    "description": description,
                    "suggested_queries": g.get("suggested_queries", []),
                    "priority": min(max(int(g.get("priority", 1)), 1), 10),
                })

        # Parse quality updates
        raw_quality = data.get("quality_updates", [])
        if isinstance(raw_quality, list):
            for q in raw_quality:
                if not isinstance(q, dict):
                    continue
                source_id = q.get("source_id", "").strip()
                quality = q.get("quality", "").lower()
                if source_id and quality in ("low", "medium", "high", "unknown"):
                    result["quality_updates"].append({
                        "source_id": source_id,
                        "quality": quality,
                    })

        # Mark success if we got at least one finding
        result["success"] = len(result["findings"]) > 0

        return result
