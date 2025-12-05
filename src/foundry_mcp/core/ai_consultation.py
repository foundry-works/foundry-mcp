"""
AI Consultation Layer for foundry-mcp.

This module provides a unified interface for AI-assisted operations including
document generation, plan review, and fidelity checking. It integrates with
the provider registry to support multiple LLM backends while providing
caching, timeout handling, and consistent result structures.

Design Principles:
    - Workflow-specific prompt templates (doc_generation, plan_review, fidelity_review)
    - Provider-agnostic orchestration via the provider registry
    - Filesystem-based caching for consultation results
    - Consistent result structures across all workflows
    - Graceful degradation when providers are unavailable

Example Usage:
    from foundry_mcp.core.ai_consultation import (
        ConsultationOrchestrator,
        ConsultationRequest,
        ConsultationWorkflow,
    )
    from foundry_mcp.core.providers import ProviderHooks

    orchestrator = ConsultationOrchestrator()

    # Check availability
    if orchestrator.is_available():
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.DOC_GENERATION,
            prompt_id="analyze_module",
            context={"file_path": "src/main.py", "content": "..."},
            provider_id="gemini",
        )
        result = orchestrator.consult(request)
        if result.content:
            print(result.content)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from foundry_mcp.core.providers import (
    ProviderHooks,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    ProviderUnavailableError,
    available_providers,
    check_provider_available,
    resolve_provider,
)
from foundry_mcp.core.llm_config import ProviderSpec

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow Types
# =============================================================================


class ConsultationWorkflow(str, Enum):
    """
    Supported AI consultation workflows.

    Each workflow corresponds to a category of prompt templates and
    determines cache partitioning and result handling.

    Values:
        DOC_GENERATION: Generate documentation from code analysis
        PLAN_REVIEW: Review and critique SDD specifications
        FIDELITY_REVIEW: Compare implementation against specifications
    """

    DOC_GENERATION = "doc_generation"
    PLAN_REVIEW = "plan_review"
    FIDELITY_REVIEW = "fidelity_review"


# =============================================================================
# Request/Response Dataclasses
# =============================================================================


@dataclass
class ResolvedProvider:
    """
    Resolved provider information from a ProviderSpec.

    Contains the provider ID to use for registry lookup, along with
    model and override settings from the priority configuration.

    Attributes:
        provider_id: Provider ID for registry lookup (e.g., "gemini", "opencode")
        model: Model identifier to use (may include backend routing for CLI)
        overrides: Per-provider setting overrides from config
        spec_str: Original spec string for logging/debugging
    """

    provider_id: str
    model: Optional[str] = None
    overrides: Dict[str, Any] = field(default_factory=dict)
    spec_str: str = ""


@dataclass(frozen=True)
class ConsultationRequest:
    """
    Request payload for AI consultation.

    Encapsulates all parameters needed to execute a consultation workflow,
    including prompt selection, context data, and provider preferences.

    Attributes:
        workflow: The consultation workflow type
        prompt_id: Identifier for the prompt template within the workflow
        context: Structured context data to inject into the prompt
        provider_id: Optional preferred provider (uses first available if None)
        model: Optional model override for the provider
        cache_key: Optional explicit cache key (auto-generated if None)
        timeout: Request timeout in seconds (default: 120)
        temperature: Sampling temperature (default: provider default)
        max_tokens: Maximum output tokens (default: provider default)
        system_prompt_override: Optional system prompt override
    """

    workflow: ConsultationWorkflow
    prompt_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    provider_id: Optional[str] = None
    model: Optional[str] = None
    cache_key: Optional[str] = None
    timeout: float = 120.0
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt_override: Optional[str] = None


@dataclass
class ConsultationResult:
    """
    Result of an AI consultation.

    Provides a consistent structure for consultation outcomes across all
    workflows and providers, including metadata for debugging and analytics.

    Attributes:
        workflow: The workflow that produced this result
        content: The generated content (may be empty on failure)
        provider_id: Provider that handled the request
        model_used: Fully-qualified model identifier
        tokens: Token usage if reported by provider
        duration_ms: Total consultation duration in milliseconds
        cache_hit: Whether result was served from cache
        raw_payload: Provider-specific metadata and debug info
        warnings: Non-fatal issues encountered during consultation
        error: Error message if consultation failed
    """

    workflow: ConsultationWorkflow
    content: str
    provider_id: str
    model_used: str
    tokens: Dict[str, int] = field(default_factory=dict)
    duration_ms: float = 0.0
    cache_hit: bool = False
    raw_payload: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Return True if consultation succeeded (has content, no error)."""
        return bool(self.content) and self.error is None


# =============================================================================
# Cache Implementation
# =============================================================================


class ResultCache:
    """
    Filesystem-based cache for consultation results.

    Provides persistent caching of AI consultation results to reduce
    redundant API calls and improve response times for repeated queries.

    Cache Structure:
        .cache/foundry-mcp/consultations/{workflow}/{key}.json

    Each cached entry contains:
        - content: The consultation result
        - provider_id: Provider that generated the result
        - model_used: Model identifier
        - tokens: Token usage
        - timestamp: Cache entry creation time
        - ttl: Time-to-live in seconds

    Attributes:
        base_dir: Root directory for cache storage
        default_ttl: Default time-to-live in seconds (default: 3600 = 1 hour)
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        default_ttl: int = 3600,
    ):
        """
        Initialize the result cache.

        Args:
            base_dir: Root directory for cache (default: .cache/foundry-mcp/consultations)
            default_ttl: Default TTL in seconds (default: 3600)
        """
        if base_dir is None:
            base_dir = Path.cwd() / ".cache" / "foundry-mcp" / "consultations"
        self.base_dir = base_dir
        self.default_ttl = default_ttl

    def _get_cache_path(self, workflow: ConsultationWorkflow, key: str) -> Path:
        """Return the cache file path for a workflow and key."""
        # Sanitize key to be filesystem-safe
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        return self.base_dir / workflow.value / f"{safe_key}.json"

    def get(
        self,
        workflow: ConsultationWorkflow,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached result.

        Args:
            workflow: The consultation workflow
            key: The cache key

        Returns:
            Cached data dict if found and not expired, None otherwise
        """
        cache_path = self._get_cache_path(workflow, key)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check TTL
            timestamp = data.get("timestamp", 0)
            ttl = data.get("ttl", self.default_ttl)
            if time.time() - timestamp > ttl:
                # Expired - remove file
                cache_path.unlink(missing_ok=True)
                return None

            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read cache entry %s: %s", cache_path, exc)
            return None

    def set(
        self,
        workflow: ConsultationWorkflow,
        key: str,
        result: ConsultationResult,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a consultation result in the cache.

        Args:
            workflow: The consultation workflow
            key: The cache key
            result: The consultation result to cache
            ttl: Time-to-live in seconds (default: default_ttl)
        """
        cache_path = self._get_cache_path(workflow, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "content": result.content,
            "provider_id": result.provider_id,
            "model_used": result.model_used,
            "tokens": result.tokens,
            "timestamp": time.time(),
            "ttl": ttl if ttl is not None else self.default_ttl,
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to write cache entry %s: %s", cache_path, exc)

    def invalidate(
        self,
        workflow: Optional[ConsultationWorkflow] = None,
        key: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            workflow: If provided, only invalidate entries for this workflow
            key: If provided (with workflow), only invalidate this specific entry

        Returns:
            Number of entries invalidated
        """
        count = 0

        if workflow is not None and key is not None:
            # Invalidate specific entry
            cache_path = self._get_cache_path(workflow, key)
            if cache_path.exists():
                cache_path.unlink()
                count = 1
        elif workflow is not None:
            # Invalidate all entries for workflow
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                for cache_file in workflow_dir.glob("*.json"):
                    cache_file.unlink()
                    count += 1
        else:
            # Invalidate all entries
            for workflow_enum in ConsultationWorkflow:
                workflow_dir = self.base_dir / workflow_enum.value
                if workflow_dir.exists():
                    for cache_file in workflow_dir.glob("*.json"):
                        cache_file.unlink()
                        count += 1

        return count

    def stats(self) -> Dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with entry counts per workflow and total size
        """
        stats: Dict[str, Any] = {
            "total_entries": 0,
            "total_size_bytes": 0,
            "by_workflow": {},
        }

        for workflow in ConsultationWorkflow:
            workflow_dir = self.base_dir / workflow.value
            if workflow_dir.exists():
                entries = list(workflow_dir.glob("*.json"))
                size = sum(f.stat().st_size for f in entries if f.exists())
                stats["by_workflow"][workflow.value] = {
                    "entries": len(entries),
                    "size_bytes": size,
                }
                stats["total_entries"] += len(entries)
                stats["total_size_bytes"] += size
            else:
                stats["by_workflow"][workflow.value] = {
                    "entries": 0,
                    "size_bytes": 0,
                }

        return stats


# =============================================================================
# Consultation Orchestrator
# =============================================================================


class ConsultationOrchestrator:
    """
    Central orchestrator for AI consultation workflows.

    Coordinates between prompt templates, the provider registry, and
    the result cache to execute consultation requests. Handles provider
    selection, timeout management, and error handling.

    Attributes:
        cache: ResultCache instance for caching results
        preferred_providers: Ordered list of preferred provider IDs
        default_timeout: Default timeout in seconds

    Example:
        orchestrator = ConsultationOrchestrator()

        if orchestrator.is_available():
            request = ConsultationRequest(
                workflow=ConsultationWorkflow.DOC_GENERATION,
                prompt_id="analyze_module",
                context={"content": "def foo(): pass"},
            )
            result = orchestrator.consult(request)
    """

    def __init__(
        self,
        cache: Optional[ResultCache] = None,
        preferred_providers: Optional[Sequence[str]] = None,
        default_timeout: Optional[float] = None,
        config: Optional["ConsultationConfig"] = None,
    ):
        """
        Initialize the consultation orchestrator.

        Args:
            cache: ResultCache instance (creates default if None)
            preferred_providers: Ordered list of preferred provider IDs (legacy, use config.priority)
            default_timeout: Default timeout in seconds (uses config if None)
            config: ConsultationConfig instance (uses global config if None)
        """
        # Lazy import to avoid circular dependency
        from foundry_mcp.core.llm_config import ConsultationConfig, get_consultation_config

        self._config: ConsultationConfig = config or get_consultation_config()
        self.cache = cache or ResultCache(default_ttl=self._config.cache_ttl)
        self.default_timeout = (
            default_timeout if default_timeout is not None else self._config.default_timeout
        )

        # Parse priority list from config into ProviderSpec objects
        # Priority: 1) config.priority specs, 2) preferred_providers param (legacy)
        self._priority_specs: List[ProviderSpec] = []
        if self._config.priority:
            for spec_str in self._config.priority:
                try:
                    self._priority_specs.append(ProviderSpec.parse(spec_str))
                except ValueError as e:
                    logger.warning(f"Invalid provider spec in priority list: {spec_str}: {e}")

        # Legacy preferred_providers for backwards compatibility
        self.preferred_providers = list(preferred_providers) if preferred_providers else []

    def is_available(self, provider_id: Optional[str] = None) -> bool:
        """
        Check if consultation services are available.

        Args:
            provider_id: Check specific provider, or any available if None

        Returns:
            True if at least one provider is available
        """
        if provider_id:
            return check_provider_available(provider_id)

        # Check preferred providers first
        for prov_id in self.preferred_providers:
            if check_provider_available(prov_id):
                return True

        # Fall back to any available provider
        return len(available_providers()) > 0

    def get_available_providers(self) -> List[str]:
        """
        Return list of available provider IDs.

        Preferred providers are listed first (if available), followed by
        other available providers.

        Returns:
            List of available provider IDs
        """
        available = set(available_providers())
        result = []

        # Add preferred providers that are available
        for prov_id in self.preferred_providers:
            if prov_id in available:
                result.append(prov_id)
                available.discard(prov_id)

        # Add remaining available providers
        result.extend(sorted(available))
        return result

    def _select_provider(self, request: ConsultationRequest) -> str:
        """
        Select the provider to use for a request.

        Args:
            request: The consultation request

        Returns:
            Provider ID to use

        Raises:
            ProviderUnavailableError: If no providers are available
        """
        # Explicit provider requested
        if request.provider_id:
            if check_provider_available(request.provider_id):
                return request.provider_id
            raise ProviderUnavailableError(
                f"Requested provider '{request.provider_id}' is not available",
                provider=request.provider_id,
            )

        # Try preferred providers
        for prov_id in self.preferred_providers:
            if check_provider_available(prov_id):
                return prov_id

        # Fall back to first available
        providers = available_providers()
        if providers:
            return providers[0]

        raise ProviderUnavailableError(
            "No AI providers are currently available",
            provider=None,
        )

    def _generate_cache_key(self, request: ConsultationRequest) -> str:
        """
        Generate a cache key for a consultation request.

        Args:
            request: The consultation request

        Returns:
            Cache key string
        """
        if request.cache_key:
            return request.cache_key

        # Build a deterministic key from request parameters
        key_parts = [
            request.prompt_id,
            json.dumps(request.context, sort_keys=True),
            request.model or "default",
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _build_prompt(self, request: ConsultationRequest) -> str:
        """
        Build the full prompt from template and context.

        This method delegates to workflow-specific prompt builders.

        Args:
            request: The consultation request

        Returns:
            The rendered prompt string
        """
        # Import prompt builders lazily to avoid circular imports
        from foundry_mcp.core.prompts import get_prompt_builder

        builder = get_prompt_builder(request.workflow)
        return builder.build(request.prompt_id, request.context)

    def _resolve_spec_to_provider(self, spec: ProviderSpec) -> Optional[ResolvedProvider]:
        """
        Resolve a ProviderSpec to a ResolvedProvider if available.

        For CLI providers, checks registry availability.
        For API providers, logs a warning (not yet implemented).

        Args:
            spec: The provider specification to resolve

        Returns:
            ResolvedProvider if available, None otherwise
        """
        if spec.type == "api":
            # API providers not yet integrated into registry
            # TODO: Register API providers (openai, anthropic, local) in registry
            logger.debug(
                f"API provider spec '{spec}' skipped - API providers not yet "
                "integrated into consultation registry"
            )
            return None

        # CLI provider - check registry availability
        if not check_provider_available(spec.provider):
            return None

        # Build model string - include backend routing if specified
        model = None
        if spec.backend and spec.model:
            # Backend routing: "openai/gpt-5.1-codex"
            model = f"{spec.backend}/{spec.model}"
        elif spec.model:
            model = spec.model

        # Get overrides from config
        overrides = self._config.get_override(str(spec))

        return ResolvedProvider(
            provider_id=spec.provider,
            model=model,
            overrides=overrides,
            spec_str=str(spec),
        )

    def _get_providers_to_try(self, request: ConsultationRequest) -> List[ResolvedProvider]:
        """
        Get ordered list of providers to try for a request.

        Provider selection priority:
        1. Explicit provider_id in request (wraps to ResolvedProvider)
        2. Priority specs from config (parsed ProviderSpec list)
        3. Legacy preferred_providers (for backwards compatibility)
        4. Available providers from registry (fallback)

        Args:
            request: The consultation request

        Returns:
            Ordered list of ResolvedProvider instances to try
        """
        result: List[ResolvedProvider] = []
        seen_providers: set = set()

        # 1. Explicit provider requested - only try that one
        if request.provider_id:
            return [
                ResolvedProvider(
                    provider_id=request.provider_id,
                    model=request.model,
                    spec_str=f"explicit:{request.provider_id}",
                )
            ]

        # 2. Priority specs from config
        for spec in self._priority_specs:
            resolved = self._resolve_spec_to_provider(spec)
            if resolved and resolved.provider_id not in seen_providers:
                result.append(resolved)
                seen_providers.add(resolved.provider_id)

        # 3. Legacy preferred_providers (for backwards compatibility)
        for prov_id in self.preferred_providers:
            if prov_id not in seen_providers and check_provider_available(prov_id):
                result.append(
                    ResolvedProvider(
                        provider_id=prov_id,
                        spec_str=f"legacy:{prov_id}",
                    )
                )
                seen_providers.add(prov_id)

        # 4. Fallback to available providers from registry
        for prov_id in available_providers():
            if prov_id not in seen_providers:
                result.append(
                    ResolvedProvider(
                        provider_id=prov_id,
                        spec_str=f"fallback:{prov_id}",
                    )
                )
                seen_providers.add(prov_id)

        return result

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error warrants a retry.

        Retryable errors include timeouts and rate limits.
        Non-retryable errors include authentication failures and invalid prompts.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is transient and retry may succeed
        """
        error_str = str(error).lower()

        # Timeout errors are retryable
        if "timeout" in error_str or "timed out" in error_str:
            return True

        # Rate limit errors are retryable
        if "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str:
            return True

        # Connection errors may be transient
        if "connection" in error_str and ("reset" in error_str or "refused" in error_str):
            return True

        # Server errors (5xx) are potentially retryable
        if any(code in error_str for code in ["500", "502", "503", "504"]):
            return True

        return False

    def _should_try_next_provider(self, error: Exception) -> bool:
        """
        Determine if we should try the next provider after an error.

        Args:
            error: The exception that occurred

        Returns:
            True if fallback to next provider is appropriate
        """
        # Don't fallback if disabled
        if not self._config.fallback_enabled:
            return False

        error_str = str(error).lower()

        # Don't fallback for prompt-level errors (these will fail with any provider)
        if "prompt" in error_str and ("too long" in error_str or "invalid" in error_str):
            return False

        # Don't fallback for authentication errors specific to all providers
        if "api key" in error_str or "authentication" in error_str:
            # This might be provider-specific, so allow fallback
            return True

        # Fallback for most other errors
        return True

    def _try_provider_with_retries(
        self,
        request: ConsultationRequest,
        prompt: str,
        resolved: ResolvedProvider,
        warnings: List[str],
    ) -> Optional[ProviderResult]:
        """
        Try a single provider with retry logic.

        Args:
            request: The consultation request
            prompt: The rendered prompt
            resolved: Resolved provider information (includes model and overrides)
            warnings: List to append warnings to

        Returns:
            ProviderResult on success, None on failure
        """
        hooks = ProviderHooks()
        last_error: Optional[Exception] = None
        provider_id = resolved.provider_id

        max_attempts = self._config.max_retries + 1  # +1 for initial attempt

        # Determine model: request.model > resolved.model > None
        effective_model = request.model or resolved.model

        # Apply overrides from config
        effective_timeout = resolved.overrides.get("timeout", request.timeout) or self.default_timeout
        effective_temperature = resolved.overrides.get("temperature", request.temperature)
        effective_max_tokens = resolved.overrides.get("max_tokens", request.max_tokens)

        for attempt in range(max_attempts):
            try:
                provider = resolve_provider(provider_id, hooks=hooks, model=effective_model)
                provider_request = ProviderRequest(
                    prompt=prompt,
                    system_prompt=request.system_prompt_override,
                    model=effective_model,
                    timeout=effective_timeout,
                    temperature=effective_temperature,
                    max_tokens=effective_max_tokens,
                )
                result = provider.generate(provider_request)

                # Success
                if result.status == ProviderStatus.SUCCESS:
                    if attempt > 0:
                        warnings.append(
                            f"Provider {provider_id} succeeded on attempt {attempt + 1}"
                        )
                    return result

                # Non-success status from provider
                error_msg = f"Provider {provider_id} returned status: {result.status.value}"
                if result.stderr:
                    error_msg += f" - {result.stderr}"
                last_error = Exception(error_msg)

                # Check if this error type is retryable
                if not self._is_retryable_error(last_error):
                    break

            except ProviderUnavailableError as exc:
                last_error = exc
                # Provider unavailable - don't retry, move to fallback
                break

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not self._is_retryable_error(exc):
                    break

            # Retry delay
            if attempt < max_attempts - 1:
                warnings.append(
                    f"Provider {provider_id} attempt {attempt + 1} failed: {last_error}, "
                    f"retrying in {self._config.retry_delay}s..."
                )
                time.sleep(self._config.retry_delay)

        # All retries exhausted
        if last_error:
            warnings.append(
                f"Provider {provider_id} failed after {max_attempts} attempt(s): {last_error}"
            )
        return None

    def _execute_with_fallback(
        self,
        request: ConsultationRequest,
        prompt: str,
        providers: List[ResolvedProvider],
        warnings: List[str],
    ) -> tuple[Optional[ProviderResult], str, Optional[str]]:
        """
        Execute request with fallback across providers.

        Args:
            request: The consultation request
            prompt: The rendered prompt
            providers: Ordered list of ResolvedProvider instances to try
            warnings: List to append warnings to

        Returns:
            Tuple of (result, provider_id, error_message)
        """
        if not providers:
            return None, "none", "No AI providers are currently available"

        last_error: Optional[str] = None
        last_provider_id = providers[0].provider_id

        for i, resolved in enumerate(providers):
            provider_id = resolved.provider_id
            last_provider_id = provider_id

            # Check if provider is available (may have changed since _get_providers_to_try)
            if not check_provider_available(provider_id):
                warnings.append(f"Provider {provider_id} is not available, skipping")
                continue

            logger.debug(
                f"Trying provider {provider_id} (spec: {resolved.spec_str}, "
                f"model: {resolved.model})"
            )
            result = self._try_provider_with_retries(request, prompt, resolved, warnings)

            if result is not None:
                return result, provider_id, None

            # Determine if we should try next provider
            if i < len(providers) - 1:
                # Check the last warning for the error
                last_warning = warnings[-1] if warnings else ""
                # Create a pseudo-error from the warning to check fallback eligibility
                pseudo_error = Exception(last_warning)
                if self._should_try_next_provider(pseudo_error):
                    warnings.append(f"Falling back to next provider...")
                else:
                    last_error = f"Provider {provider_id} failed and fallback is not appropriate"
                    break
            else:
                last_error = f"All {len(providers)} provider(s) failed"

        return None, last_provider_id, last_error or "All providers failed"

    def consult(
        self,
        request: ConsultationRequest,
        *,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
    ) -> ConsultationResult:
        """
        Execute a consultation request with retry and fallback support.

        The consultation process:
        1. Check cache for existing result
        2. Build prompt from template and context
        3. Get ordered list of providers to try
        4. Execute with retries per provider and fallback across providers
        5. Cache successful results

        Retry behavior (configurable via ConsultationConfig):
        - max_retries: Number of retry attempts per provider (default: 2)
        - retry_delay: Delay between retries in seconds (default: 5.0)
        - Retries occur for transient errors (timeouts, rate limits, 5xx errors)

        Fallback behavior (configurable via ConsultationConfig):
        - fallback_enabled: Whether to try next provider on failure (default: True)
        - Fallback skipped for prompt-level errors that would fail with any provider

        Args:
            request: The consultation request
            use_cache: Whether to use cached results (default: True)
            cache_ttl: Cache TTL override in seconds

        Returns:
            ConsultationResult with the outcome
        """
        start_time = time.time()
        warnings: List[str] = []

        # Generate cache key
        cache_key = self._generate_cache_key(request)

        # Check cache
        if use_cache:
            cached = self.cache.get(request.workflow, cache_key)
            if cached:
                duration_ms = (time.time() - start_time) * 1000
                return ConsultationResult(
                    workflow=request.workflow,
                    content=cached.get("content", ""),
                    provider_id=cached.get("provider_id", "cached"),
                    model_used=cached.get("model_used", "cached"),
                    tokens=cached.get("tokens", {}),
                    duration_ms=duration_ms,
                    cache_hit=True,
                )

        # Build prompt
        try:
            prompt = self._build_prompt(request)
        except Exception as exc:  # noqa: BLE001 - wrap prompt build errors
            duration_ms = (time.time() - start_time) * 1000
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id="none",
                model_used="none",
                duration_ms=duration_ms,
                error=f"Failed to build prompt: {exc}",
            )

        # Get providers to try (respects explicit provider_id if set)
        providers = self._get_providers_to_try(request)

        # Execute with fallback and retries
        provider_result, provider_id, error_msg = self._execute_with_fallback(
            request, prompt, providers, warnings
        )

        # Build result
        duration_ms = (time.time() - start_time) * 1000

        if provider_result is None:
            # All providers failed
            return ConsultationResult(
                workflow=request.workflow,
                content="",
                provider_id=provider_id,
                model_used="none",
                duration_ms=duration_ms,
                warnings=warnings,
                error=error_msg or "AI consultation failed",
            )

        # Extract token counts
        tokens = {
            "input_tokens": provider_result.tokens.input_tokens,
            "output_tokens": provider_result.tokens.output_tokens,
            "total_tokens": provider_result.tokens.total_tokens,
        }

        result = ConsultationResult(
            workflow=request.workflow,
            content=provider_result.content,
            provider_id=provider_result.provider_id,
            model_used=provider_result.model_used,
            tokens=tokens,
            duration_ms=duration_ms,
            cache_hit=False,
            raw_payload=provider_result.raw_payload,
            warnings=warnings,
            error=None,
        )

        # Cache successful results
        if result.success and use_cache:
            self.cache.set(request.workflow, cache_key, result, ttl=cache_ttl)

        return result

    def consult_multiple(
        self,
        requests: Sequence[ConsultationRequest],
        *,
        use_cache: bool = True,
    ) -> List[ConsultationResult]:
        """
        Execute multiple consultation requests sequentially.

        Args:
            requests: Sequence of consultation requests
            use_cache: Whether to use cached results

        Returns:
            List of ConsultationResult objects in the same order as requests
        """
        return [self.consult(req, use_cache=use_cache) for req in requests]


# =============================================================================
# Module Exports
# =============================================================================


__all__ = [
    # Workflow types
    "ConsultationWorkflow",
    # Request/Response
    "ConsultationRequest",
    "ConsultationResult",
    # Cache
    "ResultCache",
    # Orchestrator
    "ConsultationOrchestrator",
]
