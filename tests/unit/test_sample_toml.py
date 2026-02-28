"""Validation tests for sample TOML configuration files.

Ensures that both the quick-start sample and the comprehensive reference
parse as valid TOML, contain only known fields, and include no deprecated
field names.
"""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

from foundry_mcp.config.research import ResearchConfig

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"
SAMPLE_PATH = SAMPLES_DIR / "foundry-mcp.toml"
REFERENCE_PATH = SAMPLES_DIR / "foundry-mcp-reference.toml"

DEPRECATED_FIELDS = set(ResearchConfig._DEPRECATED_FIELDS.keys()) | {
    # Backward-compat aliases that should not appear as primary keys
    "deep_research_topic_max_searches",
    # Phantom fields that never existed in the dataclass
    "notes_dir",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _uncommented_keys_in_section(
    path: Path, section: str
) -> set[str]:
    """Extract uncommented keys from a TOML section.

    Returns keys that are *not* commented out (no leading '#') within
    the specified top-level [section].
    """
    data = tomllib.loads(path.read_text())
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        return set()
    # Recursively collect leaf keys (skip nested dicts which are sub-tables)
    keys: set[str] = set()
    for key, value in section_data.items():
        if not isinstance(value, dict):
            keys.add(key)
    return keys


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestSampleToml:
    """Tests for the quick-start sample (samples/foundry-mcp.toml)."""

    def test_parses_as_valid_toml(self) -> None:
        """Sample file must be syntactically valid TOML."""
        data = tomllib.loads(SAMPLE_PATH.read_text())
        assert isinstance(data, dict)

    def test_no_deprecated_fields_uncommented(self) -> None:
        """Sample must not contain deprecated field names as active keys."""
        data = tomllib.loads(SAMPLE_PATH.read_text())
        research = data.get("research", {})
        active_keys = {
            k for k, v in research.items() if not isinstance(v, dict)
        }
        overlap = active_keys & DEPRECATED_FIELDS
        assert not overlap, f"Deprecated fields found in sample [research]: {overlap}"

    def test_no_orphaned_sections(self) -> None:
        """Sample must not contain orphaned sections that are never parsed."""
        data = tomllib.loads(SAMPLE_PATH.read_text())
        orphaned = {"implement", "workflow", "feature_flags"}
        present = set(data.keys()) & orphaned
        assert not present, f"Orphaned sections found in sample: {present}"

    def test_research_fields_known(self) -> None:
        """All uncommented [research] keys must be known to ResearchConfig."""
        known = {f.name for f in dataclasses.fields(ResearchConfig)}
        # Include sub-table names and aliases that from_toml_dict accepts
        known |= {
            "deep_research", "tavily", "perplexity", "semantic_scholar",
            "timeouts", "fallback_chains", "phase_fallbacks", "model_tiers",
            "per_provider_rate_limits", "model_context_overrides",
            "deep_research_topic_max_searches",
        }
        active = _uncommented_keys_in_section(SAMPLE_PATH, "research")
        unknown = active - known
        assert not unknown, f"Unknown [research] keys in sample: {unknown}"

    def test_from_toml_dict_accepts_sample(self) -> None:
        """ResearchConfig.from_toml_dict() must accept the sample without errors."""
        data = tomllib.loads(SAMPLE_PATH.read_text())
        research_data = dict(data.get("research", {}))
        # Should not raise
        config = ResearchConfig.from_toml_dict(research_data)
        assert config.enabled is True


class TestReferenceToml:
    """Tests for the comprehensive reference (samples/foundry-mcp-reference.toml)."""

    def test_parses_as_valid_toml(self) -> None:
        """Reference file must be syntactically valid TOML."""
        data = tomllib.loads(REFERENCE_PATH.read_text())
        assert isinstance(data, dict)

    def test_no_deprecated_fields_uncommented(self) -> None:
        """Reference must not contain deprecated field names as active keys."""
        data = tomllib.loads(REFERENCE_PATH.read_text())
        research = data.get("research", {})
        active_keys = {
            k for k, v in research.items() if not isinstance(v, dict)
        }
        overlap = active_keys & DEPRECATED_FIELDS
        assert not overlap, f"Deprecated fields found in reference [research]: {overlap}"

    def test_no_orphaned_sections(self) -> None:
        """Reference must not contain orphaned sections."""
        data = tomllib.loads(REFERENCE_PATH.read_text())
        orphaned = {"implement", "workflow", "feature_flags"}
        present = set(data.keys()) & orphaned
        assert not present, f"Orphaned sections found in reference: {present}"

    def test_no_phantom_notes_dir(self) -> None:
        """Neither file should have a notes_dir field."""
        for path in (SAMPLE_PATH, REFERENCE_PATH):
            data = tomllib.loads(path.read_text())
            workspace = data.get("workspace", {})
            assert "notes_dir" not in workspace, (
                f"Phantom field 'notes_dir' found in {path.name} [workspace]"
            )

    def test_reference_from_toml_dict_accepts(self) -> None:
        """ResearchConfig.from_toml_dict() must accept the reference research section."""
        data = tomllib.loads(REFERENCE_PATH.read_text())
        research_data = dict(data.get("research", {}))
        # Should not raise
        config = ResearchConfig.from_toml_dict(research_data)
        assert isinstance(config, ResearchConfig)
