"""Response-parsing mixin for the analysis phase.

Parses LLM JSON responses into structured findings, gaps, and quality updates.
Split from ``analysis.py`` to keep each module focused.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from foundry_mcp.core.research.models.deep_research import DeepResearchState
from foundry_mcp.core.research.models.enums import ConfidenceLevel
from foundry_mcp.core.research.workflows.deep_research._json_parsing import extract_json

logger = logging.getLogger(__name__)


# --- Structured output models for analysis response validation ---

_CONFIDENCE_MAP = {
    "low": ConfidenceLevel.LOW,
    "medium": ConfidenceLevel.MEDIUM,
    "high": ConfidenceLevel.HIGH,
    "confirmed": ConfidenceLevel.CONFIRMED,
    "speculation": ConfidenceLevel.SPECULATION,
}


class AnalysisFinding(BaseModel):
    """A single finding from source analysis."""

    content: str = Field(..., description="A clear, specific finding")
    confidence: str = Field(default="medium", description="low|medium|high|confirmed|speculation")
    source_ids: list[str] = Field(default_factory=list, description="Source IDs supporting this finding")
    category: Optional[str] = Field(default=None, description="Category/theme")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Finding content must not be empty")
        return v


class AnalysisGap(BaseModel):
    """A knowledge gap identified during analysis."""

    description: str = Field(..., description="Description of missing information")
    suggested_queries: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=10)

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Gap description must not be empty")
        return v


class AnalysisQualityUpdate(BaseModel):
    """Quality assessment for a source."""

    source_id: str = Field(..., description="Source ID")
    quality: Literal["low", "medium", "high", "unknown"] = Field(...)


class AnalysisResponse(BaseModel):
    """Complete structured analysis response from the LLM."""

    findings: list[AnalysisFinding] = Field(default_factory=list)
    gaps: list[AnalysisGap] = Field(default_factory=list)
    quality_updates: list[AnalysisQualityUpdate] = Field(default_factory=list)


class AnalysisParsingMixin:
    """Response parsing methods for analysis. Mixed into AnalysisPhaseMixin.

    At runtime, ``self`` is a DeepResearchWorkflow instance.
    """

    def _parse_analysis_response(
        self,
        content: str,
        state: DeepResearchState,
    ) -> dict[str, Any]:
        """Parse LLM response into structured analysis data.

        Tries Pydantic-validated JSON first, falls back to manual dict extraction.

        Args:
            content: Raw LLM response content
            state: Current research state (reserved for context-aware parsing)

        Returns:
            Dict with 'success', 'findings', 'gaps', 'quality_updates',
            and 'parse_method' keys
        """
        _ = state
        result: dict[str, Any] = {
            "success": False,
            "findings": [],
            "gaps": [],
            "quality_updates": [],
            "parse_method": None,
        }

        if not content:
            return result

        # Try to extract JSON from the response
        json_str = extract_json(content)
        if not json_str:
            logger.warning("No JSON found in analysis response, attempting markdown fallback")
            self._parse_analysis_markdown_fallback(content, result)
            if result["findings"]:
                result["success"] = True
                result["parse_method"] = "fallback_markdown"
            return result

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON from analysis response: %s", e)
            self._parse_analysis_markdown_fallback(content, result)
            if result["findings"]:
                result["success"] = True
                result["parse_method"] = "fallback_markdown"
            return result

        # Try Pydantic validation first
        try:
            parsed = AnalysisResponse.model_validate(data)
            for f in parsed.findings:
                result["findings"].append(
                    {
                        "content": f.content,
                        "confidence": _CONFIDENCE_MAP.get(f.confidence.lower(), ConfidenceLevel.MEDIUM),
                        "source_ids": f.source_ids,
                        "category": f.category,
                    }
                )
            for g in parsed.gaps:
                result["gaps"].append(
                    {
                        "description": g.description,
                        "suggested_queries": g.suggested_queries,
                        "priority": g.priority,
                    }
                )
            for q in parsed.quality_updates:
                result["quality_updates"].append(
                    {
                        "source_id": q.source_id,
                        "quality": q.quality,
                    }
                )
            result["success"] = len(result["findings"]) > 0
            result["parse_method"] = "json"
            return result

        except Exception as exc:
            logger.warning("Pydantic validation failed, falling back to manual dict extraction: %s", exc)

        # Fallback: manual dict extraction (original behavior)
        self._parse_analysis_dict_fallback(data, result)
        result["success"] = len(result["findings"]) > 0
        result["parse_method"] = "fallback_dict"
        return result

    @staticmethod
    def _parse_analysis_dict_fallback(data: dict, result: dict[str, Any]) -> None:
        """Manual dict extraction fallback (original parsing logic)."""
        raw_findings = data.get("findings", [])
        if isinstance(raw_findings, list):
            for f in raw_findings:
                if not isinstance(f, dict):
                    continue
                content_text = f.get("content", "").strip()
                if not content_text:
                    continue

                confidence_str = f.get("confidence", "medium").lower()
                confidence = _CONFIDENCE_MAP.get(confidence_str, ConfidenceLevel.MEDIUM)

                result["findings"].append(
                    {
                        "content": content_text,
                        "confidence": confidence,
                        "source_ids": f.get("source_ids", []),
                        "category": f.get("category"),
                    }
                )

        raw_gaps = data.get("gaps", [])
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if not isinstance(g, dict):
                    continue
                description = g.get("description", "").strip()
                if not description:
                    continue

                result["gaps"].append(
                    {
                        "description": description,
                        "suggested_queries": g.get("suggested_queries", []),
                        "priority": min(max(int(g.get("priority", 1)), 1), 10),
                    }
                )

        raw_quality = data.get("quality_updates", [])
        if isinstance(raw_quality, list):
            for q in raw_quality:
                if not isinstance(q, dict):
                    continue
                source_id = q.get("source_id", "").strip()
                quality = q.get("quality", "").lower()
                if source_id and quality in ("low", "medium", "high", "unknown"):
                    result["quality_updates"].append(
                        {
                            "source_id": source_id,
                            "quality": quality,
                        }
                    )

    @staticmethod
    def _parse_analysis_markdown_fallback(content: str, result: dict[str, Any]) -> None:
        """Extract findings from markdown-formatted analysis responses.

        Looks for bullet points, numbered items, or heading-based findings.
        """
        # Look for bullet-point or numbered findings
        finding_patterns = [
            # "- Finding: ..." or "- **Finding**: ..."
            re.compile(r"^[-*]\s+\*?\*?(?:Finding|Key\s+(?:finding|insight))s?\*?\*?:?\s*(.+)", re.IGNORECASE),
            # "1. ..." numbered findings
            re.compile(r"^\d+\.\s+(.+)"),
            # "- ..." generic bullets (only if preceded by a findings header)
            re.compile(r"^[-*]\s+(.+)"),
        ]

        lines = content.split("\n")
        in_findings_section = False

        for line in lines:
            stripped = line.strip()
            # Detect findings section headers
            if re.match(r"^#{1,3}\s*(?:Key\s+)?Findings?", stripped, re.IGNORECASE):
                in_findings_section = True
                continue
            # Detect section change
            if re.match(r"^#{1,3}\s+", stripped) and in_findings_section:
                in_findings_section = False
                continue

            if in_findings_section:
                for pattern in finding_patterns:
                    m = pattern.match(stripped)
                    if m:
                        finding_text = m.group(1).strip()
                        if finding_text and len(finding_text) > 10:
                            result["findings"].append(
                                {
                                    "content": finding_text,
                                    "confidence": ConfidenceLevel.MEDIUM,
                                    "source_ids": [],
                                    "category": None,
                                }
                            )
                        break
