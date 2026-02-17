"""
LLM configuration parsing for foundry-mcp.

Parses the [llm] section from foundry-mcp.toml to configure LLM provider settings.

TOML Configuration Example:
    [llm]
    provider = "openai"           # Required: "openai", "anthropic", or "local"
    api_key = "sk-..."            # Optional: defaults to env var based on provider
    model = "gpt-4.1"             # Optional: provider-specific default
    timeout = 30                  # Optional: request timeout in seconds (default: 30)

Environment Variables (fallback if not in TOML):
    - FOUNDRY_MCP_LLM_PROVIDER: LLM provider type ("openai", "anthropic", "local")
    - FOUNDRY_MCP_LLM_API_KEY: API key (takes precedence over provider-specific keys)
    - FOUNDRY_MCP_LLM_MODEL: Model identifier
    - FOUNDRY_MCP_LLM_TIMEOUT: Request timeout in seconds
    - FOUNDRY_MCP_LLM_BASE_URL: Custom API base URL
    - FOUNDRY_MCP_LLM_MAX_TOKENS: Default max tokens
    - FOUNDRY_MCP_LLM_TEMPERATURE: Default temperature
    - FOUNDRY_MCP_LLM_ORGANIZATION: Organization ID (OpenAI)

Provider-specific API key fallbacks:
    - OPENAI_API_KEY: OpenAI API key (if FOUNDRY_MCP_LLM_API_KEY not set)
    - ANTHROPIC_API_KEY: Anthropic API key (if FOUNDRY_MCP_LLM_API_KEY not set)
"""

# Re-export from sub-modules
from .provider_spec import (
    ProviderSpec,
    LLMProviderType,
    DEFAULT_MODELS,
    API_KEY_ENV_VARS,
)
from .llm import (
    LLMConfig,
    load_llm_config,
    get_llm_config,
    set_llm_config,
    reset_llm_config,
)
from .workflow import (
    WorkflowMode,
    WorkflowConfig,
    load_workflow_config,
    get_workflow_config,
    set_workflow_config,
    reset_workflow_config,
)
from .consultation import (
    WorkflowConsultationConfig,
    ConsultationConfig,
    load_consultation_config,
    get_consultation_config,
    set_consultation_config,
    reset_consultation_config,
)

__all__ = [
    # Provider Spec (unified priority notation)
    "ProviderSpec",
    # LLM Config
    "LLMProviderType",
    "LLMConfig",
    "load_llm_config",
    "get_llm_config",
    "set_llm_config",
    "reset_llm_config",
    "DEFAULT_MODELS",
    "API_KEY_ENV_VARS",
    # Workflow Config
    "WorkflowMode",
    "WorkflowConfig",
    "load_workflow_config",
    "get_workflow_config",
    "set_workflow_config",
    "reset_workflow_config",
    # Consultation Config
    "WorkflowConsultationConfig",
    "ConsultationConfig",
    "load_consultation_config",
    "get_consultation_config",
    "set_consultation_config",
    "reset_consultation_config",
]
