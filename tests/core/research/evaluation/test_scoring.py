"""Tests for scoring normalization and composite calculation."""

from __future__ import annotations

import pytest

from foundry_mcp.core.research.evaluation.scoring import (
    DimensionScore,
    EvaluationResult,
    build_dimension_score,
    compute_composite,
    normalize_score,
)


class TestNormalizeScore:
    """Verify 1-5 to 0-1 normalization."""

    def test_score_1_maps_to_0(self):
        assert normalize_score(1) == 0.0

    def test_score_3_maps_to_05(self):
        assert normalize_score(3) == 0.5

    def test_score_5_maps_to_1(self):
        assert normalize_score(5) == 1.0

    def test_score_2_maps_to_025(self):
        assert normalize_score(2) == 0.25

    def test_score_4_maps_to_075(self):
        assert normalize_score(4) == 0.75

    def test_score_below_1_raises(self):
        with pytest.raises(ValueError, match="1-5"):
            normalize_score(0)

    def test_score_above_5_raises(self):
        with pytest.raises(ValueError, match="1-5"):
            normalize_score(6)


class TestBuildDimensionScore:
    """Verify DimensionScore construction with auto-normalization."""

    def test_basic_construction(self):
        ds = build_dimension_score("depth", 4, "Good depth")
        assert ds.name == "depth"
        assert ds.raw_score == 4
        assert ds.normalized_score == 0.75
        assert ds.rationale == "Good depth"

    def test_clamps_below_1(self):
        ds = build_dimension_score("depth", 0)
        assert ds.raw_score == 1
        assert ds.normalized_score == 0.0

    def test_clamps_above_5(self):
        ds = build_dimension_score("depth", 8)
        assert ds.raw_score == 5
        assert ds.normalized_score == 1.0

    def test_empty_rationale(self):
        ds = build_dimension_score("depth", 3)
        assert ds.rationale == ""


class TestComputeComposite:
    """Verify composite score computation."""

    def _make_scores(self, raw_scores: list[tuple[str, int]]) -> list[DimensionScore]:
        return [build_dimension_score(name, score) for name, score in raw_scores]

    def test_equal_weights_uniform_scores(self):
        scores = self._make_scores([("a", 3), ("b", 3), ("c", 3)])
        composite, variance, weights = compute_composite(scores)
        assert composite == pytest.approx(0.5)
        assert variance == pytest.approx(0.0)
        assert len(weights) == 3
        for w in weights.values():
            assert w == pytest.approx(1.0 / 3)

    def test_equal_weights_varied_scores(self):
        scores = self._make_scores([("a", 1), ("b", 5)])
        composite, variance, weights = compute_composite(scores)
        # (0.0 + 1.0) / 2 = 0.5
        assert composite == pytest.approx(0.5)
        # Variance of [0.0, 1.0]: mean=0.5, var=0.25
        assert variance == pytest.approx(0.25)

    def test_all_fives(self):
        scores = self._make_scores([("a", 5), ("b", 5), ("c", 5)])
        composite, variance, weights = compute_composite(scores)
        assert composite == pytest.approx(1.0)
        assert variance == pytest.approx(0.0)

    def test_all_ones(self):
        scores = self._make_scores([("a", 1), ("b", 1), ("c", 1)])
        composite, variance, weights = compute_composite(scores)
        assert composite == pytest.approx(0.0)
        assert variance == pytest.approx(0.0)

    def test_custom_weights(self):
        scores = self._make_scores([("a", 5), ("b", 1)])
        custom_weights = {"a": 3.0, "b": 1.0}
        composite, variance, weights = compute_composite(scores, weights=custom_weights)
        # a: 1.0 * 0.75 + b: 0.0 * 0.25 = 0.75
        assert composite == pytest.approx(0.75)

    def test_empty_scores_raises(self):
        with pytest.raises(ValueError, match="no dimension scores"):
            compute_composite([])

    def test_composite_always_between_0_and_1(self):
        """Property: composite is always in [0, 1] for any valid raw scores."""
        import itertools

        for combo in itertools.product(range(1, 6), repeat=3):
            scores = self._make_scores([(f"d{i}", s) for i, s in enumerate(combo)])
            composite, _, _ = compute_composite(scores)
            assert 0.0 <= composite <= 1.0, f"Composite {composite} out of range for {combo}"


class TestEvaluationResult:
    """Verify EvaluationResult serialization."""

    def test_to_dict_round_trip(self):
        scores = [
            build_dimension_score("depth", 4, "Good"),
            build_dimension_score("structure", 3, "OK"),
        ]
        composite, variance, weights = compute_composite(scores)
        result = EvaluationResult(
            dimension_scores=scores,
            composite_score=composite,
            score_variance=variance,
            weights=weights,
            metadata={"provider_id": "gemini"},
        )

        d = result.to_dict()
        assert d["composite_score"] == composite
        assert d["score_variance"] == variance
        assert len(d["dimension_scores"]) == 2
        assert d["dimension_scores"][0]["name"] == "depth"
        assert d["dimension_scores"][0]["raw_score"] == 4
        assert d["dimension_scores"][0]["normalized_score"] == 0.75
        assert d["dimension_scores"][0]["rationale"] == "Good"
        assert d["weights"] == weights
        assert d["metadata"]["provider_id"] == "gemini"

    def test_to_dict_empty(self):
        result = EvaluationResult()
        d = result.to_dict()
        assert d["dimension_scores"] == []
        assert d["composite_score"] == 0.0
