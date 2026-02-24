"""Tests for evaluation dimension definitions."""

from __future__ import annotations

from foundry_mcp.core.research.evaluation.dimensions import (
    DIMENSION_BY_NAME,
    DIMENSIONS,
    Dimension,
)


class TestDimensionDefinitions:
    """Verify dimension metadata and completeness."""

    def test_six_dimensions_defined(self):
        assert len(DIMENSIONS) == 6

    def test_dimension_names(self):
        expected = {"depth", "source_quality", "analytical_rigor", "completeness", "groundedness", "structure"}
        actual = {d.name for d in DIMENSIONS}
        assert actual == expected

    def test_all_dimensions_have_rubrics(self):
        for dim in DIMENSIONS:
            assert dim.rubric, f"Dimension {dim.name} has empty rubric"
            # Rubric should contain score markers 1-5
            for score in range(1, 6):
                assert f"{score}:" in dim.rubric, f"Dimension {dim.name} rubric missing score {score}"

    def test_all_dimensions_have_display_names(self):
        for dim in DIMENSIONS:
            assert dim.display_name, f"Dimension {dim.name} has empty display_name"

    def test_all_dimensions_have_descriptions(self):
        for dim in DIMENSIONS:
            assert dim.description, f"Dimension {dim.name} has empty description"

    def test_dimension_by_name_lookup(self):
        assert len(DIMENSION_BY_NAME) == 6
        for dim in DIMENSIONS:
            assert DIMENSION_BY_NAME[dim.name] is dim

    def test_dimension_is_frozen(self):
        dim = DIMENSIONS[0]
        try:
            dim.name = "modified"  # type: ignore[misc]
            assert False, "Expected FrozenInstanceError"
        except AttributeError:
            pass  # Expected — frozen dataclass

    def test_dimension_names_are_unique(self):
        names = [d.name for d in DIMENSIONS]
        assert len(names) == len(set(names))
