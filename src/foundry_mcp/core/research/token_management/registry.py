"""Default model limits registry.

Pre-configured token limits for common providers and models.
"""

from .models import BudgetingMode, ModelContextLimits

# Conservative fallback for unknown models
_DEFAULT_FALLBACK = ModelContextLimits(
    context_window=128_000,
    max_output_tokens=8_000,
    budgeting_mode=BudgetingMode.INPUT_ONLY,
)

# Default limits by provider and model
# Format: {provider: {model: ModelContextLimits}}
DEFAULT_MODEL_LIMITS: dict[str, dict[str, ModelContextLimits]] = {
    # Anthropic Claude models
    "claude": {
        "opus": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "sonnet": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "haiku": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=8_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        # Default for claude provider without specific model
        "_default": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # Google Gemini models
    "gemini": {
        "flash": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "pro": ModelContextLimits(
            context_window=2_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        # Gemini 2.x variants
        "gemini-2.5-flash": ModelContextLimits(
            context_window=1_048_576,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "2.0-flash": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "_default": ModelContextLimits(
            context_window=1_000_000,
            max_output_tokens=8_192,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # OpenAI Codex models (hypothetical future models)
    "codex": {
        "gpt-5.2-codex": ModelContextLimits(
            context_window=256_000,
            max_output_tokens=32_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "gpt-4.1": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "o3": ModelContextLimits(
            context_window=200_000,
            max_output_tokens=100_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "o4-mini": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=65_536,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # Cursor Agent (IDE integration)
    "cursor-agent": {
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
    # OpenCode provider
    "opencode": {
        "_default": ModelContextLimits(
            context_window=128_000,
            max_output_tokens=16_000,
            budgeting_mode=BudgetingMode.INPUT_ONLY,
        ),
    },
}
