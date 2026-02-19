"""ServerConfig loading and validation logic.

Provides ``_ServerConfigLoader``, a mixin class whose methods are inherited by
``ServerConfig`` (defined in ``server.py``).  Splitting loading/validation
logic into its own module keeps ``server.py`` focused on field definitions and
simple accessor methods.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from foundry_mcp.config.server import ServerConfig

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python < 3.11 fallback

from foundry_mcp.config.autonomy import (
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
    AutonomySecurityConfig,
    AutonomySessionDefaultsConfig,
)
from foundry_mcp.config.domains import (
    ErrorCollectionConfig,
    HealthConfig,
    MetricsPersistenceConfig,
    ObservabilityConfig,
    TestConfig,
)
from foundry_mcp.config.parsing import (
    _normalize_commit_cadence,
    _parse_bool,
    _try_parse_bool,
)
from foundry_mcp.config.research import ResearchConfig

logger = logging.getLogger(__name__)


class _ServerConfigLoader:
    """Mixin providing config-loading methods for ``ServerConfig``.

    These methods are inherited by the ``ServerConfig`` dataclass defined in
    ``server.py``.  At runtime ``self`` is always a ``ServerConfig`` instance.
    """

    if TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs: Any) -> None: ...

        workspace_roots: List[Path]
        specs_dir: Optional[Path]
        notes_dir: Optional[Path]
        research_dir: Optional[Path]
        log_level: str
        structured_logging: bool
        api_keys: List[str]
        require_auth: bool
        server_name: str
        server_version: str
        disabled_tools: List[str]
        git: Any
        observability: ObservabilityConfig
        health: HealthConfig
        error_collection: ErrorCollectionConfig
        metrics_persistence: MetricsPersistenceConfig
        test: TestConfig
        research: ResearchConfig
        autonomy_posture: Any
        autonomy_session_defaults: AutonomySessionDefaultsConfig
        autonomy_security: AutonomySecurityConfig
        _startup_warnings: List[str]

        def _add_startup_warning(self, warning: str) -> None: ...

    @classmethod
    def from_env(cls, config_file: Optional[str] = None) -> "ServerConfig":
        """
        Create configuration from environment variables and optional TOML file.

        Priority (highest to lowest):
        1. Environment variables
        2. Project TOML config (./foundry-mcp.toml or ./.foundry-mcp.toml)
        3. User TOML config (~/.foundry-mcp.toml)
        4. XDG config (~/.config/foundry-mcp/config.toml)
        5. Default values
        """
        config = cls()

        # Load TOML config if available
        toml_path = config_file or os.environ.get("FOUNDRY_MCP_CONFIG_FILE")
        if toml_path:
            config._load_toml(Path(toml_path))
        else:
            # Layered config loading (lowest to highest priority):
            # 1. XDG config directory (system-wide user defaults)
            # Follows XDG Base Directory Specification
            xdg_config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
            xdg_config = Path(xdg_config_home) / "foundry-mcp" / "config.toml"
            if xdg_config.exists():
                config._load_toml(xdg_config)
                logger.debug(f"Loaded XDG config from {xdg_config}")

            # 2. Home directory config (user defaults, legacy location)
            home_config = Path.home() / ".foundry-mcp.toml"
            if home_config.exists():
                config._load_toml(home_config)
                logger.debug(f"Loaded user config from {home_config}")

            # 3. Project directory config (project overrides)
            # Try foundry-mcp.toml first, fall back to .foundry-mcp.toml for compatibility
            project_config = Path("foundry-mcp.toml")
            if project_config.exists():
                config._load_toml(project_config)
                logger.debug(f"Loaded project config from {project_config}")
            else:
                legacy_config = Path(".foundry-mcp.toml")
                if legacy_config.exists():
                    config._load_toml(legacy_config)
                    logger.debug(f"Loaded project config from {legacy_config}")

        # Override with environment variables
        config._load_env()
        config._validate_startup_configuration()

        return cast("ServerConfig", config)

    def _load_toml(self, path: Path) -> None:
        """Load configuration from TOML file."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            # Workspace settings
            if "workspace" in data:
                ws = data["workspace"]
                if "roots" in ws:
                    self.workspace_roots = [Path(p) for p in ws["roots"]]
                if "specs_dir" in ws:
                    self.specs_dir = Path(ws["specs_dir"])
                if "notes_dir" in ws:
                    self.notes_dir = Path(ws["notes_dir"])
                if "research_dir" in ws:
                    self.research_dir = Path(ws["research_dir"])

            # Logging settings
            if "logging" in data:
                log = data["logging"]
                if "level" in log:
                    self.log_level = log["level"].upper()
                if "structured" in log:
                    self.structured_logging = log["structured"]

            # Auth settings
            if "auth" in data:
                auth = data["auth"]
                if "api_keys" in auth:
                    self.api_keys = auth["api_keys"]
                if "require_auth" in auth:
                    self.require_auth = auth["require_auth"]

            # Server settings
            if "server" in data:
                srv = data["server"]
                if "name" in srv:
                    self.server_name = srv["name"]
                if "version" in srv:
                    self.server_version = srv["version"]
            # Tools configuration (preferred location for disabled_tools)
            if "tools" in data:
                tools_cfg = data["tools"]
                if "disabled_tools" in tools_cfg:
                    self.disabled_tools = tools_cfg["disabled_tools"]

            # Git workflow settings
            if "git" in data:
                git_cfg = data["git"]
                if "enabled" in git_cfg:
                    self.git.enabled = _parse_bool(git_cfg["enabled"])
                if "auto_commit" in git_cfg:
                    self.git.auto_commit = _parse_bool(git_cfg["auto_commit"])
                if "auto_push" in git_cfg:
                    self.git.auto_push = _parse_bool(git_cfg["auto_push"])
                if "auto_pr" in git_cfg:
                    self.git.auto_pr = _parse_bool(git_cfg["auto_pr"])
                if "show_before_commit" in git_cfg:
                    self.git.show_before_commit = _parse_bool(git_cfg["show_before_commit"])
                if "commit_cadence" in git_cfg:
                    self.git.commit_cadence = _normalize_commit_cadence(str(git_cfg["commit_cadence"]))

            # Observability settings
            if "observability" in data:
                self.observability = ObservabilityConfig.from_toml_dict(data["observability"])

            # Health check settings
            if "health" in data:
                self.health = HealthConfig.from_toml_dict(data["health"])

            # Error collection settings
            if "error_collection" in data:
                self.error_collection = ErrorCollectionConfig.from_toml_dict(data["error_collection"])

            # Metrics persistence settings
            if "metrics_persistence" in data:
                self.metrics_persistence = MetricsPersistenceConfig.from_toml_dict(data["metrics_persistence"])

            # Test runner settings
            if "test" in data:
                self.test = TestConfig.from_toml_dict(data["test"])

            # Research workflows settings
            if "research" in data:
                self.research = ResearchConfig.from_toml_dict(data["research"])

            # Autonomy posture profile (applies defaults that direct sections can override)
            if "autonomy_posture" in data:
                posture_data = data["autonomy_posture"]
                if isinstance(posture_data, dict):
                    self.apply_autonomy_posture_profile(
                        posture_data.get("profile"),
                        source=f"{path}: [autonomy_posture].profile",
                    )
                else:
                    self._add_startup_warning(
                        f"Ignoring [autonomy_posture] in {path}: expected table/dict, got {type(posture_data).__name__}"
                    )

            # Autonomy session-start defaults
            if "autonomy_session_defaults" in data:
                defaults_data = data["autonomy_session_defaults"]
                if isinstance(defaults_data, dict):
                    self._apply_autonomy_session_defaults(
                        defaults_data,
                        source=f"{path}: [autonomy_session_defaults]",
                    )
                else:
                    self._add_startup_warning(
                        f"Ignoring [autonomy_session_defaults] in {path}: expected table/dict, got {type(defaults_data).__name__}"
                    )

            # Autonomy security settings
            if "autonomy_security" in data:
                sec_data = data["autonomy_security"]
                current = self.autonomy_security
                self.autonomy_security = AutonomySecurityConfig(
                    allow_lock_bypass=sec_data.get(
                        "allow_lock_bypass",
                        current.allow_lock_bypass,
                    ),
                    allow_gate_waiver=sec_data.get(
                        "allow_gate_waiver",
                        current.allow_gate_waiver,
                    ),
                    enforce_required_phase_gates=sec_data.get(
                        "enforce_required_phase_gates",
                        current.enforce_required_phase_gates,
                    ),
                    role=sec_data.get("role", current.role),
                    rate_limit_max_consecutive_denials=sec_data.get(
                        "rate_limit_max_consecutive_denials",
                        current.rate_limit_max_consecutive_denials,
                    ),
                    rate_limit_denial_window_seconds=sec_data.get(
                        "rate_limit_denial_window_seconds",
                        current.rate_limit_denial_window_seconds,
                    ),
                    rate_limit_retry_after_seconds=sec_data.get(
                        "rate_limit_retry_after_seconds",
                        current.rate_limit_retry_after_seconds,
                    ),
                )

        except Exception as e:
            logger.error(f"Error loading config file {path}: {e}")

    def _load_env(self) -> None:
        """Load configuration from environment variables."""
        # Workspace roots
        if roots := os.environ.get("FOUNDRY_MCP_WORKSPACE_ROOTS"):
            self.workspace_roots = [Path(p.strip()) for p in roots.split(",")]

        # Specs directory
        if specs := os.environ.get("FOUNDRY_MCP_SPECS_DIR"):
            self.specs_dir = Path(specs)

        # Notes directory (intake queue storage)
        if notes := os.environ.get("FOUNDRY_MCP_NOTES_DIR"):
            self.notes_dir = Path(notes)

        # Research directory (research state storage)
        if research := os.environ.get("FOUNDRY_MCP_RESEARCH_DIR"):
            self.research_dir = Path(research)

        # Log level
        if level := os.environ.get("FOUNDRY_MCP_LOG_LEVEL"):
            self.log_level = level.upper()

        # API keys
        if keys := os.environ.get("FOUNDRY_MCP_API_KEYS"):
            self.api_keys = [k.strip() for k in keys.split(",") if k.strip()]

        # Require auth
        if require := os.environ.get("FOUNDRY_MCP_REQUIRE_AUTH"):
            self.require_auth = require.lower() in ("true", "1", "yes")

        # Git settings
        if git_enabled := os.environ.get("FOUNDRY_MCP_GIT_ENABLED"):
            self.git.enabled = _parse_bool(git_enabled)
        if git_auto_commit := os.environ.get("FOUNDRY_MCP_GIT_AUTO_COMMIT"):
            self.git.auto_commit = _parse_bool(git_auto_commit)
        if git_auto_push := os.environ.get("FOUNDRY_MCP_GIT_AUTO_PUSH"):
            self.git.auto_push = _parse_bool(git_auto_push)
        if git_auto_pr := os.environ.get("FOUNDRY_MCP_GIT_AUTO_PR"):
            self.git.auto_pr = _parse_bool(git_auto_pr)
        if git_show_preview := os.environ.get("FOUNDRY_MCP_GIT_SHOW_PREVIEW"):
            self.git.show_before_commit = _parse_bool(git_show_preview)
        if git_cadence := os.environ.get("FOUNDRY_MCP_GIT_COMMIT_CADENCE"):
            self.git.commit_cadence = _normalize_commit_cadence(git_cadence)

        # Observability settings
        if obs_enabled := os.environ.get("FOUNDRY_MCP_OBSERVABILITY_ENABLED"):
            self.observability.enabled = _parse_bool(obs_enabled)
        if otel_enabled := os.environ.get("FOUNDRY_MCP_OTEL_ENABLED"):
            self.observability.otel_enabled = _parse_bool(otel_enabled)
        if otel_endpoint := os.environ.get("FOUNDRY_MCP_OTEL_ENDPOINT"):
            self.observability.otel_endpoint = otel_endpoint
        if otel_service := os.environ.get("FOUNDRY_MCP_OTEL_SERVICE_NAME"):
            self.observability.otel_service_name = otel_service
        if otel_sample := os.environ.get("FOUNDRY_MCP_OTEL_SAMPLE_RATE"):
            try:
                self.observability.otel_sample_rate = float(otel_sample)
            except ValueError:
                pass
        if prom_enabled := os.environ.get("FOUNDRY_MCP_PROMETHEUS_ENABLED"):
            self.observability.prometheus_enabled = _parse_bool(prom_enabled)
        if prom_port := os.environ.get("FOUNDRY_MCP_PROMETHEUS_PORT"):
            try:
                self.observability.prometheus_port = int(prom_port)
            except ValueError:
                pass
        if prom_host := os.environ.get("FOUNDRY_MCP_PROMETHEUS_HOST"):
            self.observability.prometheus_host = prom_host
        if prom_ns := os.environ.get("FOUNDRY_MCP_PROMETHEUS_NAMESPACE"):
            self.observability.prometheus_namespace = prom_ns

        # Health check settings
        if health_enabled := os.environ.get("FOUNDRY_MCP_HEALTH_ENABLED"):
            self.health.enabled = _parse_bool(health_enabled)
        if health_liveness_timeout := os.environ.get("FOUNDRY_MCP_HEALTH_LIVENESS_TIMEOUT"):
            try:
                self.health.liveness_timeout = float(health_liveness_timeout)
            except ValueError:
                pass
        if health_readiness_timeout := os.environ.get("FOUNDRY_MCP_HEALTH_READINESS_TIMEOUT"):
            try:
                self.health.readiness_timeout = float(health_readiness_timeout)
            except ValueError:
                pass
        if health_timeout := os.environ.get("FOUNDRY_MCP_HEALTH_TIMEOUT"):
            try:
                self.health.health_timeout = float(health_timeout)
            except ValueError:
                pass
        if disk_threshold := os.environ.get("FOUNDRY_MCP_DISK_SPACE_THRESHOLD_MB"):
            try:
                self.health.disk_space_threshold_mb = int(disk_threshold)
            except ValueError:
                pass
        if disk_warning := os.environ.get("FOUNDRY_MCP_DISK_SPACE_WARNING_MB"):
            try:
                self.health.disk_space_warning_mb = int(disk_warning)
            except ValueError:
                pass

        # Error collection settings
        if err_enabled := os.environ.get("FOUNDRY_MCP_ERROR_COLLECTION_ENABLED"):
            self.error_collection.enabled = _parse_bool(err_enabled)
        if err_storage := os.environ.get("FOUNDRY_MCP_ERROR_STORAGE_PATH"):
            self.error_collection.storage_path = err_storage
        if err_retention := os.environ.get("FOUNDRY_MCP_ERROR_RETENTION_DAYS"):
            try:
                self.error_collection.retention_days = int(err_retention)
            except ValueError:
                pass
        if err_max := os.environ.get("FOUNDRY_MCP_ERROR_MAX_ERRORS"):
            try:
                self.error_collection.max_errors = int(err_max)
            except ValueError:
                pass
        if err_stack := os.environ.get("FOUNDRY_MCP_ERROR_INCLUDE_STACK_TRACES"):
            self.error_collection.include_stack_traces = _parse_bool(err_stack)
        if err_redact := os.environ.get("FOUNDRY_MCP_ERROR_REDACT_INPUTS"):
            self.error_collection.redact_inputs = _parse_bool(err_redact)

        # Metrics persistence settings
        if metrics_enabled := os.environ.get("FOUNDRY_MCP_METRICS_PERSISTENCE_ENABLED"):
            self.metrics_persistence.enabled = _parse_bool(metrics_enabled)
        if metrics_storage := os.environ.get("FOUNDRY_MCP_METRICS_STORAGE_PATH"):
            self.metrics_persistence.storage_path = metrics_storage
        if metrics_retention := os.environ.get("FOUNDRY_MCP_METRICS_RETENTION_DAYS"):
            try:
                self.metrics_persistence.retention_days = int(metrics_retention)
            except ValueError:
                pass
        if metrics_max := os.environ.get("FOUNDRY_MCP_METRICS_MAX_RECORDS"):
            try:
                self.metrics_persistence.max_records = int(metrics_max)
            except ValueError:
                pass
        if metrics_bucket := os.environ.get("FOUNDRY_MCP_METRICS_BUCKET_INTERVAL"):
            try:
                self.metrics_persistence.bucket_interval_seconds = int(metrics_bucket)
            except ValueError:
                pass
        if metrics_flush := os.environ.get("FOUNDRY_MCP_METRICS_FLUSH_INTERVAL"):
            try:
                self.metrics_persistence.flush_interval_seconds = int(metrics_flush)
            except ValueError:
                pass
        if persist_list := os.environ.get("FOUNDRY_MCP_METRICS_PERSIST_METRICS"):
            self.metrics_persistence.persist_metrics = [m.strip() for m in persist_list.split(",") if m.strip()]

        # Search provider API keys (direct env vars, no FOUNDRY_MCP_ prefix)
        # These use standard env var names that match provider documentation
        if tavily_key := os.environ.get("TAVILY_API_KEY"):
            self.research.tavily_api_key = tavily_key
        if perplexity_key := os.environ.get("PERPLEXITY_API_KEY"):
            self.research.perplexity_api_key = perplexity_key
        if google_key := os.environ.get("GOOGLE_API_KEY"):
            self.research.google_api_key = google_key
        if google_cse := os.environ.get("GOOGLE_CSE_ID"):
            self.research.google_cse_id = google_cse
        if semantic_scholar_key := os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
            self.research.semantic_scholar_api_key = semantic_scholar_key

        # Disabled tools (comma-separated list)
        if disabled := os.environ.get("FOUNDRY_MCP_DISABLED_TOOLS"):
            self.disabled_tools = [t.strip() for t in disabled.split(",") if t.strip()]

        # Autonomy posture profile defaults (can be overridden by explicit env vars below)
        if posture_profile := os.environ.get(_AUTONOMY_POSTURE_ENV_VAR):
            self.apply_autonomy_posture_profile(
                posture_profile,
                source=_AUTONOMY_POSTURE_ENV_VAR,
            )

        # Autonomy session-start defaults
        if gate_policy := os.environ.get(_AUTONOMY_DEFAULT_GATE_POLICY_ENV_VAR):
            normalized_policy = gate_policy.strip().lower()
            if normalized_policy in _VALID_GATE_POLICIES:
                self.autonomy_session_defaults.gate_policy = normalized_policy
            else:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_GATE_POLICY_ENV_VAR}: expected one of "
                    f"{', '.join(sorted(_VALID_GATE_POLICIES))}, got {gate_policy!r}"
                )

        if stop_on_phase_completion := os.environ.get(_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION_ENV_VAR):
            parsed_stop_on_phase_completion = _try_parse_bool(stop_on_phase_completion)
            if parsed_stop_on_phase_completion is None:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_STOP_ON_PHASE_COMPLETION_ENV_VAR}: "
                    f"expected true/false value, got {stop_on_phase_completion!r}"
                )
            else:
                self.autonomy_session_defaults.stop_on_phase_completion = parsed_stop_on_phase_completion

        if auto_retry_fidelity_gate := os.environ.get(_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE_ENV_VAR):
            parsed_auto_retry_fidelity_gate = _try_parse_bool(auto_retry_fidelity_gate)
            if parsed_auto_retry_fidelity_gate is None:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_AUTO_RETRY_FIDELITY_GATE_ENV_VAR}: "
                    f"expected true/false value, got {auto_retry_fidelity_gate!r}"
                )
            else:
                self.autonomy_session_defaults.auto_retry_fidelity_gate = parsed_auto_retry_fidelity_gate

        if max_tasks_per_session := os.environ.get(_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION_ENV_VAR):
            try:
                parsed = int(max_tasks_per_session)
                if parsed > 0:
                    self.autonomy_session_defaults.max_tasks_per_session = parsed
                else:
                    self._add_startup_warning(
                        f"Ignoring {_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION_ENV_VAR}: value must be > 0"
                    )
            except ValueError:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_MAX_TASKS_PER_SESSION_ENV_VAR}: "
                    f"expected integer > 0, got {max_tasks_per_session!r}"
                )

        if max_consecutive_errors := os.environ.get(_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS_ENV_VAR):
            try:
                parsed = int(max_consecutive_errors)
                if parsed > 0:
                    self.autonomy_session_defaults.max_consecutive_errors = parsed
                else:
                    self._add_startup_warning(
                        f"Ignoring {_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS_ENV_VAR}: value must be > 0"
                    )
            except ValueError:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_MAX_CONSECUTIVE_ERRORS_ENV_VAR}: "
                    f"expected integer > 0, got {max_consecutive_errors!r}"
                )

        if max_fidelity_cycles := os.environ.get(_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_ENV_VAR):
            try:
                parsed = int(max_fidelity_cycles)
                if parsed > 0:
                    self.autonomy_session_defaults.max_fidelity_review_cycles_per_phase = parsed
                else:
                    self._add_startup_warning(
                        f"Ignoring {_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_ENV_VAR}: value must be > 0"
                    )
            except ValueError:
                self._add_startup_warning(
                    f"Ignoring {_AUTONOMY_DEFAULT_MAX_FIDELITY_REVIEW_CYCLES_ENV_VAR}: "
                    f"expected integer > 0, got {max_fidelity_cycles!r}"
                )

        # Autonomy security settings (provenance-logged per H4)
        if allow_lock_bypass := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS"):
            self.autonomy_security.allow_lock_bypass = _parse_bool(allow_lock_bypass)
            logger.info(
                "Config: autonomy_security.allow_lock_bypass = %s (source: FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_LOCK_BYPASS env var)",
                self.autonomy_security.allow_lock_bypass,
            )
        if allow_gate_waiver := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_GATE_WAIVER"):
            self.autonomy_security.allow_gate_waiver = _parse_bool(allow_gate_waiver)
            logger.info(
                "Config: autonomy_security.allow_gate_waiver = %s (source: FOUNDRY_MCP_AUTONOMY_SECURITY_ALLOW_GATE_WAIVER env var)",
                self.autonomy_security.allow_gate_waiver,
            )
        if enforce_required_gates := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_ENFORCE_REQUIRED_PHASE_GATES"):
            self.autonomy_security.enforce_required_phase_gates = _parse_bool(enforce_required_gates)
            logger.info(
                "Config: autonomy_security.enforce_required_phase_gates = %s (source: FOUNDRY_MCP_AUTONOMY_SECURITY_ENFORCE_REQUIRED_PHASE_GATES env var)",
                self.autonomy_security.enforce_required_phase_gates,
            )
        if role := os.environ.get("FOUNDRY_MCP_ROLE"):
            self.autonomy_security.role = role.strip().lower()
            logger.info(
                "Config: autonomy_security.role = %s (source: FOUNDRY_MCP_ROLE env var)",
                self.autonomy_security.role,
            )
        if max_denials := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_MAX_CONSECUTIVE_DENIALS"):
            try:
                parsed = int(max_denials)
                if parsed > 0:
                    self.autonomy_security.rate_limit_max_consecutive_denials = parsed
            except ValueError:
                pass
        if denial_window := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_DENIAL_WINDOW_SECONDS"):
            try:
                parsed = int(denial_window)
                if parsed > 0:
                    self.autonomy_security.rate_limit_denial_window_seconds = parsed
            except ValueError:
                pass
        if retry_after := os.environ.get("FOUNDRY_MCP_AUTONOMY_SECURITY_RATE_LIMIT_RETRY_AFTER_SECONDS"):
            try:
                parsed = int(retry_after)
                if parsed > 0:
                    self.autonomy_security.rate_limit_retry_after_seconds = parsed
            except ValueError:
                pass

    def apply_autonomy_posture_profile(
        self,
        profile_value: Any,
        *,
        source: str,
    ) -> None:
        """Apply a fixed posture profile to autonomy defaults."""
        if profile_value is None:
            return
        if not isinstance(profile_value, str) or not profile_value.strip():
            self._add_startup_warning(
                f"Ignoring autonomy posture from {source}: expected non-empty string, got {profile_value!r}"
            )
            return

        normalized_profile = profile_value.strip().lower()
        profile_defaults = _AUTONOMY_POSTURE_DEFAULTS.get(normalized_profile)
        if profile_defaults is None:
            self._add_startup_warning(
                f"Ignoring autonomy posture from {source}: expected one of "
                f"{', '.join(sorted(_VALID_AUTONOMY_POSTURES))}, got {profile_value!r}"
            )
            return

        self.autonomy_posture.profile = normalized_profile
        logger.info(
            "Config: autonomy_posture.profile = %s (source: %s)",
            normalized_profile,
            source,
        )

        current_security = self.autonomy_security
        security_defaults = profile_defaults["autonomy_security"]

        # Log posture overrides for security-relevant settings
        for field_name in ("allow_lock_bypass", "allow_gate_waiver", "enforce_required_phase_gates", "role"):
            old_val = getattr(current_security, field_name)
            new_val = security_defaults[field_name]
            if isinstance(old_val, bool):
                new_val = bool(new_val)
            else:
                new_val = str(new_val)
            if old_val != new_val:
                logger.info(
                    "Config: autonomy_security.%s overridden by %s posture (%s \u2190 %s)",
                    field_name,
                    normalized_profile,
                    new_val,
                    old_val,
                )

        self.autonomy_security = AutonomySecurityConfig(
            allow_lock_bypass=bool(security_defaults["allow_lock_bypass"]),
            allow_gate_waiver=bool(security_defaults["allow_gate_waiver"]),
            enforce_required_phase_gates=bool(security_defaults["enforce_required_phase_gates"]),
            role=str(security_defaults["role"]),
            rate_limit_max_consecutive_denials=(current_security.rate_limit_max_consecutive_denials),
            rate_limit_denial_window_seconds=(current_security.rate_limit_denial_window_seconds),
            rate_limit_retry_after_seconds=(current_security.rate_limit_retry_after_seconds),
        )

        session_defaults = profile_defaults["autonomy_session_defaults"]
        self.autonomy_session_defaults = AutonomySessionDefaultsConfig(
            gate_policy=str(session_defaults["gate_policy"]),
            stop_on_phase_completion=bool(session_defaults["stop_on_phase_completion"]),
            auto_retry_fidelity_gate=bool(session_defaults["auto_retry_fidelity_gate"]),
            max_tasks_per_session=int(session_defaults["max_tasks_per_session"]),
            max_consecutive_errors=int(session_defaults["max_consecutive_errors"]),
            max_fidelity_review_cycles_per_phase=int(session_defaults["max_fidelity_review_cycles_per_phase"]),
        )

    def _apply_autonomy_session_defaults(
        self,
        defaults_data: Dict[Any, Any],
        *,
        source: str,
    ) -> None:
        """Apply autonomy session default overrides from config."""
        if not isinstance(defaults_data, dict):
            self._add_startup_warning(f"Ignoring autonomy session defaults from {source}: expected table/dict")
            return

        current = self.autonomy_session_defaults
        gate_policy = current.gate_policy
        if "gate_policy" in defaults_data:
            raw_gate_policy = defaults_data["gate_policy"]
            if isinstance(raw_gate_policy, str):
                normalized_gate_policy = raw_gate_policy.strip().lower()
                if normalized_gate_policy in _VALID_GATE_POLICIES:
                    gate_policy = normalized_gate_policy
                else:
                    self._add_startup_warning(
                        f"Ignoring gate_policy from {source}: expected one of "
                        f"{', '.join(sorted(_VALID_GATE_POLICIES))}, got {raw_gate_policy!r}"
                    )
            else:
                self._add_startup_warning(
                    f"Ignoring gate_policy from {source}: expected string, got {type(raw_gate_policy).__name__}"
                )

        stop_on_phase_completion = current.stop_on_phase_completion
        if "stop_on_phase_completion" in defaults_data:
            parsed = _try_parse_bool(defaults_data["stop_on_phase_completion"])
            if parsed is None:
                self._add_startup_warning(
                    f"Ignoring stop_on_phase_completion from {source}: expected true/false value, got {defaults_data['stop_on_phase_completion']!r}"
                )
            else:
                stop_on_phase_completion = parsed

        auto_retry_fidelity_gate = current.auto_retry_fidelity_gate
        if "auto_retry_fidelity_gate" in defaults_data:
            parsed = _try_parse_bool(defaults_data["auto_retry_fidelity_gate"])
            if parsed is None:
                self._add_startup_warning(
                    f"Ignoring auto_retry_fidelity_gate from {source}: expected true/false value, got {defaults_data['auto_retry_fidelity_gate']!r}"
                )
            else:
                auto_retry_fidelity_gate = parsed

        max_tasks_per_session = current.max_tasks_per_session
        if "max_tasks_per_session" in defaults_data:
            raw_value = defaults_data["max_tasks_per_session"]
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    max_tasks_per_session = parsed
                else:
                    self._add_startup_warning(f"Ignoring max_tasks_per_session from {source}: value must be > 0")
            except (TypeError, ValueError):
                self._add_startup_warning(
                    f"Ignoring max_tasks_per_session from {source}: expected integer > 0, got {raw_value!r}"
                )

        max_consecutive_errors = current.max_consecutive_errors
        if "max_consecutive_errors" in defaults_data:
            raw_value = defaults_data["max_consecutive_errors"]
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    max_consecutive_errors = parsed
                else:
                    self._add_startup_warning(f"Ignoring max_consecutive_errors from {source}: value must be > 0")
            except (TypeError, ValueError):
                self._add_startup_warning(
                    f"Ignoring max_consecutive_errors from {source}: expected integer > 0, got {raw_value!r}"
                )

        max_fidelity_review_cycles_per_phase = current.max_fidelity_review_cycles_per_phase
        if "max_fidelity_review_cycles_per_phase" in defaults_data:
            raw_value = defaults_data["max_fidelity_review_cycles_per_phase"]
            try:
                parsed = int(raw_value)
                if parsed > 0:
                    max_fidelity_review_cycles_per_phase = parsed
                else:
                    self._add_startup_warning(
                        f"Ignoring max_fidelity_review_cycles_per_phase from {source}: value must be > 0"
                    )
            except (TypeError, ValueError):
                self._add_startup_warning(
                    "Ignoring max_fidelity_review_cycles_per_phase from "
                    f"{source}: expected integer > 0, got {raw_value!r}"
                )

        self.autonomy_session_defaults = AutonomySessionDefaultsConfig(
            gate_policy=gate_policy,
            stop_on_phase_completion=stop_on_phase_completion,
            auto_retry_fidelity_gate=auto_retry_fidelity_gate,
            max_tasks_per_session=max_tasks_per_session,
            max_consecutive_errors=max_consecutive_errors,
            max_fidelity_review_cycles_per_phase=(max_fidelity_review_cycles_per_phase),
        )

    def _validate_startup_configuration(self) -> None:
        """Validate startup configuration. Raises on critical misconfigurations."""
        profile = self.autonomy_posture.profile
        if profile == "unattended":
            unsafe_conditions: List[str] = []
            if self.autonomy_security.role != "autonomy_runner":
                unsafe_conditions.append("role must be autonomy_runner for unattended posture")
            if self.autonomy_security.allow_lock_bypass:
                unsafe_conditions.append("allow_lock_bypass must be false")
            if self.autonomy_security.allow_gate_waiver:
                unsafe_conditions.append("allow_gate_waiver must be false")
            if not self.autonomy_security.enforce_required_phase_gates:
                unsafe_conditions.append("enforce_required_phase_gates must be true")
            if self.autonomy_session_defaults.gate_policy != "strict":
                unsafe_conditions.append("gate_policy should be strict")
            if not self.autonomy_session_defaults.stop_on_phase_completion:
                unsafe_conditions.append("stop_on_phase_completion should be true")
            if unsafe_conditions:
                self._add_startup_warning("UNSAFE unattended posture configuration: " + "; ".join(unsafe_conditions))

        if profile == "debug" and self.autonomy_security.role == "autonomy_runner":
            self._add_startup_warning(
                "Debug posture is intended for supervised/manual operation; autonomy_runner role should not be used."
            )
