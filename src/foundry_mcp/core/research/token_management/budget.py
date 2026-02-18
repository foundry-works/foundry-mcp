"""Mutable token budget tracker with allocation and safety margins.

Provides:
    - TokenBudget: Tracks token budget allocation and usage for a workflow
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TokenBudget:
    """Tracks token budget allocation and usage for a workflow.

    Provides methods to check available budget, allocate tokens, and
    track usage with a configurable safety margin.

    Attributes:
        total_budget: Total tokens available for the workflow
        reserved_output: Tokens reserved for output generation
        safety_margin: Fraction of budget to keep as buffer (0.0-1.0)
        used_tokens: Tokens already consumed (mutable, updated by allocate())

    Example:
        budget = TokenBudget(
            total_budget=100_000,
            reserved_output=8_000,
            safety_margin=0.1,
        )
        # Effective budget: (100_000 - 8_000) * (1 - 0.1) = 82_800

        if budget.can_fit(5_000):
            budget.allocate(5_000)
    """

    total_budget: int
    reserved_output: int = 0
    safety_margin: float = 0.1
    used_tokens: int = 0

    def __post_init__(self) -> None:
        """Validate budget parameters after initialization."""
        if self.total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {self.total_budget}")
        if self.reserved_output < 0:
            raise ValueError(f"reserved_output must be non-negative, got {self.reserved_output}")
        if self.reserved_output >= self.total_budget:
            raise ValueError(
                f"reserved_output ({self.reserved_output}) must be less than "
                f"total_budget ({self.total_budget})"
            )
        if not 0.0 <= self.safety_margin < 1.0:
            raise ValueError(
                f"safety_margin must be in [0.0, 1.0), got {self.safety_margin}"
            )
        if self.used_tokens < 0:
            raise ValueError(f"used_tokens must be non-negative, got {self.used_tokens}")

    def effective_budget(self) -> int:
        """Calculate the effective budget after reservations and safety margin.

        The effective budget is:
            (total_budget - reserved_output) * (1 - safety_margin)

        Returns:
            Effective token budget available for allocation
        """
        available = self.total_budget - self.reserved_output
        return int(available * (1.0 - self.safety_margin))

    def remaining(self) -> int:
        """Calculate remaining tokens available for allocation.

        Returns:
            Tokens remaining (effective_budget - used_tokens), minimum 0
        """
        return max(0, self.effective_budget() - self.used_tokens)

    def can_fit(self, tokens: int) -> bool:
        """Check if a given number of tokens can fit in the remaining budget.

        Args:
            tokens: Number of tokens to check

        Returns:
            True if tokens fit within remaining budget, False otherwise
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        return tokens <= self.remaining()

    def allocate(self, tokens: int) -> bool:
        """Allocate tokens from the budget.

        Attempts to allocate the specified tokens. If successful, updates
        used_tokens and returns True. If insufficient budget, returns False
        without modifying state.

        Args:
            tokens: Number of tokens to allocate

        Returns:
            True if allocation succeeded, False if insufficient budget

        Raises:
            ValueError: If tokens is negative
        """
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        if not self.can_fit(tokens):
            logger.debug(
                f"Token allocation failed: requested {tokens}, "
                f"remaining {self.remaining()}"
            )
            return False
        self.used_tokens += tokens
        return True

    def usage_fraction(self) -> float:
        """Calculate the fraction of effective budget used.

        Returns:
            Fraction of budget used (0.0 to 1.0+)
        """
        effective = self.effective_budget()
        if effective <= 0:
            return 1.0 if self.used_tokens > 0 else 0.0
        return self.used_tokens / effective
