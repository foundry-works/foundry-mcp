"""Research quality evaluation framework.

Provides LLM-as-judge evaluation of completed deep research reports
across six quality dimensions: Depth, Source Quality, Analytical Rigor,
Completeness, Groundedness, and Structure.

Usage::

    from foundry_mcp.core.research.evaluation import (
        DIMENSIONS,
        EvaluationResult,
        evaluate_report,
    )

    result = await evaluate_report(
        workflow=workflow,
        state=state,
        provider_id="gemini",
        model=None,
        timeout=360.0,
    )
    print(result.composite_score)  # 0.0 - 1.0
"""

from foundry_mcp.core.research.evaluation.dimensions import DIMENSIONS, Dimension
from foundry_mcp.core.research.evaluation.evaluator import evaluate_report
from foundry_mcp.core.research.evaluation.scoring import EvaluationResult

__all__ = [
    "DIMENSIONS",
    "Dimension",
    "EvaluationResult",
    "evaluate_report",
]
