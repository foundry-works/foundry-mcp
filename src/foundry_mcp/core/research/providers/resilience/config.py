"""Provider-specific resilience configurations.

Maps provider names to tuned ProviderResilienceConfig instances and
provides a lookup function with sensible defaults.
"""

from foundry_mcp.core.research.providers.resilience.models import (
    ProviderResilienceConfig,
)

# Provider-specific configurations with tuned defaults
PROVIDER_CONFIGS: dict[str, ProviderResilienceConfig] = {
    "tavily": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "google": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "perplexity": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "semantic_scholar": ProviderResilienceConfig(
        # Slightly under 1 RPS to stay within Semantic Scholar's rate limits
        requests_per_second=0.9,
        burst_limit=2,
        max_retries=3,
        base_delay=1.5,  # Slightly higher base delay for academic API
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "tavily_extract": ProviderResilienceConfig(
        requests_per_second=1.0,
        burst_limit=3,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
    "openalex": ProviderResilienceConfig(
        # Conservative 50 RPS (100 hard cap); high burst for batch lookups
        requests_per_second=50.0,
        burst_limit=10,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        jitter=0.5,
        circuit_failure_threshold=5,
        circuit_recovery_timeout=30.0,
    ),
}


def get_provider_config(provider_name: str) -> ProviderResilienceConfig:
    """Get resilience configuration for a provider.

    Args:
        provider_name: Name of the provider (e.g., 'tavily', 'google')

    Returns:
        Provider-specific config or default config if provider not found
    """
    return PROVIDER_CONFIGS.get(provider_name, ProviderResilienceConfig())
