"""Score normalization and composite calculation.

Handles the conversion from raw 1-5 dimension scores to normalized 0-1
values and computation of the composite quality score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension.

    Attributes:
        name: Dimension identifier (e.g. ``"depth"``).
        raw_score: LLM-assigned score on the 1-5 rubric scale.
        normalized_score: Score mapped to 0-1 range via ``(raw - 1) / 4``.
        rationale: LLM's explanation for the assigned score.
    """

    name: str
    raw_score: int
    normalized_score: float
    rationale: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a research report.

    Attributes:
        dimension_scores: Per-dimension scores with rationales.
        composite_score: Weighted average of normalized scores (0-1).
        score_variance: Variance across normalized dimension scores.
        weights: Weights used for composite calculation (dimension_name -> weight).
        metadata: Additional evaluation context (provider, model, etc.).
    """

    dimension_scores: list[DimensionScore] = field(default_factory=list)
    composite_score: float = 0.0
    score_variance: float = 0.0
    weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict for storage in state metadata."""
        return {
            "dimension_scores": [
                {
                    "name": ds.name,
                    "raw_score": ds.raw_score,
                    "normalized_score": ds.normalized_score,
                    "rationale": ds.rationale,
                }
                for ds in self.dimension_scores
            ],
            "composite_score": self.composite_score,
            "score_variance": self.score_variance,
            "weights": self.weights,
            "metadata": self.metadata,
        }


def normalize_score(raw: int) -> float:
    """Normalize a 1-5 raw score to the 0-1 range.

    Maps: 1 -> 0.0, 2 -> 0.25, 3 -> 0.5, 4 -> 0.75, 5 -> 1.0

    Args:
        raw: Integer score on the 1-5 scale.

    Returns:
        Normalized float in [0.0, 1.0].

    Raises:
        ValueError: If raw is outside [1, 5].
    """
    if not 1 <= raw <= 5:
        raise ValueError(f"Raw score must be 1-5, got {raw}")
    return (raw - 1) / 4.0


def compute_composite(
    dimension_scores: list[DimensionScore],
    weights: Optional[dict[str, float]] = None,
) -> tuple[float, float, dict[str, float]]:
    """Compute weighted composite score and variance.

    Args:
        dimension_scores: Per-dimension scores.
        weights: Optional dimension_name -> weight mapping.
            If ``None``, equal weights are used.

    Returns:
        Tuple of (composite_score, score_variance, effective_weights).

    Raises:
        ValueError: If dimension_scores is empty.
    """
    if not dimension_scores:
        raise ValueError("Cannot compute composite with no dimension scores")

    # Default to equal weights
    if weights is None:
        w = 1.0 / len(dimension_scores)
        effective_weights = {ds.name: w for ds in dimension_scores}
    else:
        # Normalize weights to sum to 1.0
        total = sum(weights.get(ds.name, 1.0) for ds in dimension_scores)
        if total == 0:
            total = 1.0
        effective_weights = {ds.name: weights.get(ds.name, 1.0) / total for ds in dimension_scores}

    # Weighted average
    composite = sum(ds.normalized_score * effective_weights[ds.name] for ds in dimension_scores)

    # Variance of normalized scores
    scores = [ds.normalized_score for ds in dimension_scores]
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    return composite, variance, effective_weights


def build_dimension_score(
    name: str,
    raw_score: int,
    rationale: str = "",
) -> DimensionScore:
    """Create a DimensionScore with automatic normalization.

    Clamps raw_score to [1, 5] to handle LLM edge cases.

    Args:
        name: Dimension identifier.
        raw_score: LLM-assigned score (clamped to 1-5).
        rationale: LLM's explanation.

    Returns:
        DimensionScore with normalized_score computed.
    """
    clamped = max(1, min(5, raw_score))
    return DimensionScore(
        name=name,
        raw_score=clamped,
        normalized_score=normalize_score(clamped),
        rationale=rationale,
    )
