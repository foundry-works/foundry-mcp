"""Configuration package for foundry-mcp.

Re-exports all public symbols for backward compatibility. Callers can
continue to use ``from foundry_mcp.config import ServerConfig`` etc.

Sub-modules:
    parsing    – Boolean/provider-spec parsing helpers
    research   – ResearchConfig dataclass
    domains    – GitSettings, ObservabilityConfig, HealthConfig, ErrorCollectionConfig,
                 MetricsPersistenceConfig, RunnerConfig, TestConfig
    autonomy   – AutonomySecurityConfig, AutonomySessionDefaultsConfig,
                 AutonomyPostureConfig, posture constants
    server     – ServerConfig dataclass, get_config/set_config globals
    loader     – ServerConfig loading/validation mixin (_ServerConfigLoader)
    decorators – log_call, timed, require_auth
"""

# ---------------------------------------------------------------------------
# Re-exports from sub-modules
# ---------------------------------------------------------------------------
from foundry_mcp.config.parsing import (  # noqa: F401
    _VALID_COMMIT_CADENCE,
    _normalize_commit_cadence,
    _parse_bool,
    _parse_provider_spec,
    _try_parse_bool,
)
from foundry_mcp.config.research import ResearchConfig  # noqa: F401
from foundry_mcp.config.domains import (  # noqa: F401
    ErrorCollectionConfig,
    GitSettings,
    HealthConfig,
    MetricsPersistenceConfig,
    ObservabilityConfig,
    RunnerConfig,
    TestConfig,
)
from foundry_mcp.config.autonomy import (  # noqa: F401
    AutonomyPostureConfig,
    AutonomySecurityConfig,
    AutonomySessionDefaultsConfig,
    _AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE_ENV_VAR,
    _AUTONOMY_DEFAULT_GATE_POLICY_ENV_VAR,
    _AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS_ENV_VAR,
    _AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_ENV_VAR,
    _AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION_ENV_VAR,
    _AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION_ENV_VAR,
    _AUTONOMY_POSTURE_DEFAULTS,
    _AUTONOMY_POSTURE_ENV_VAR,
    _VALID_AUTONOMY_POSTURES,
    _VALID_GATE_POLICIES,
)
from foundry_mcp.config.server import (  # noqa: F401
    _PACKAGE_VERSION,
    ServerConfig,
    T,
    get_config,
    set_config,
)
from foundry_mcp.config.decorators import (  # noqa: F401
    log_call,
    require_auth,
    timed,
)
