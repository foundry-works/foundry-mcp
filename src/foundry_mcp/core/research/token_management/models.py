"""Token budgeting mode and model context limit definitions.

Provides:
    - BudgetingMode: Enum for input-only vs combined budgeting strategies
    - ModelContextLimits: Dataclass defining model token constraints
"""

from dataclasses import dataclass
from enum import Enum


class BudgetingMode(str, Enum):
    """Token budgeting strategies for different model architectures.

    Different models handle input/output token budgets differently:
    - INPUT_ONLY: Context window is for input only; output is separate
      (e.g., Claude, GPT-4). Use full context_window for input.
    - COMBINED: Context window includes both input and output
      (e.g., some Gemini modes). Must reserve space for output.

    The budgeting mode affects how get_effective_context() calculates
    available input space.
    """

    INPUT_ONLY = "input_only"
    COMBINED = "combined"


@dataclass(frozen=True)
class ModelContextLimits:
    """Token limits for a specific model.

    Defines the token constraints for a model including context window size,
    maximum output tokens, and how to budget between input and output.

    Attributes:
        context_window: Maximum input context tokens the model accepts
        max_output_tokens: Maximum tokens the model can generate in output
        budgeting_mode: How to allocate tokens between input and output
        output_reserved: Tokens to reserve for output when mode is COMBINED

    Example:
        # Claude Opus limits
        limits = ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        )
    """

    context_window: int
    max_output_tokens: int
    budgeting_mode: BudgetingMode = BudgetingMode.INPUT_ONLY
    output_reserved: int = 0

    def __post_init__(self) -> None:
        """Validate limits after initialization."""
        if self.context_window <= 0:
            raise ValueError(f"context_window must be positive, got {self.context_window}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")
        if self.output_reserved < 0:
            raise ValueError(f"output_reserved must be non-negative, got {self.output_reserved}")
        if self.budgeting_mode == BudgetingMode.COMBINED:
            if self.output_reserved > self.context_window:
                raise ValueError(
                    f"output_reserved ({self.output_reserved}) cannot exceed context_window ({self.context_window})"
                )
