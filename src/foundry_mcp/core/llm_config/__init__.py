"""
LLM configuration parsing for foundry-mcp.

Provides consultation configuration for multi-provider AI workflows
and provider specification parsing for CLI-based providers.
"""

# Re-export from sub-modules
from .consultation import (
    ConsultationConfig,
    WorkflowConsultationConfig,
    get_consultation_config,
    load_consultation_config,
    reset_consultation_config,
    set_consultation_config,
)
from .provider_spec import (
    ProviderSpec,
)
from .workflow import (
    WorkflowConfig,
    WorkflowMode,
    get_workflow_config,
    load_workflow_config,
    reset_workflow_config,
    set_workflow_config,
)

__all__ = [
    # Provider Spec (unified priority notation)
    "ProviderSpec",
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
