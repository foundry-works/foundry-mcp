"""Fixture-backed provider used for offline verification.

This provider is intentionally *disabled by default* and only becomes available
when explicitly enabled via environment variables. It exists to support:

- deterministic unit/integration tests
- offline generation of fidelity/plan review artifacts for SDD specs

Security notes:
- No network access
- No shell execution
- Reads only from a configured fixture directory
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from foundry_mcp.core.providers.base import (
    ModelDescriptor,
    ProviderCapability,
    ProviderContext,
    ProviderHooks,
    ProviderMetadata,
    ProviderRequest,
    ProviderResult,
    ProviderStatus,
    TokenUsage,
)
from foundry_mcp.core.providers.registry import register_provider


_PROVIDER_ID = "test-provider"
_ENABLE_ENV = "FOUNDRY_TEST_PROVIDER_ENABLED"
_FIXTURES_DIR_ENV = "FOUNDRY_TEST_PROVIDER_FIXTURES_DIR"
_DEFAULT_FIXTURES_DIR = Path("tests/fixtures/ai_responses")

_WORKFLOW_TO_DEFAULT_FILE = {
    "plan_review": "plan_review_response.json",
    "fidelity_review": "fidelity_review_response.json",
    "markdown_plan_review": "plan_review_response.json",
}


def _is_enabled() -> bool:
    value = os.environ.get(_ENABLE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _resolve_fixtures_dir() -> Path:
    override = os.environ.get(_FIXTURES_DIR_ENV)
    if override:
        return Path(override)
    return _DEFAULT_FIXTURES_DIR


def _load_fixture(*, workflow: str) -> Dict[str, Any]:
    fixtures_dir = _resolve_fixtures_dir()
    filename = _WORKFLOW_TO_DEFAULT_FILE.get(workflow)
    if not filename:
        raise FileNotFoundError(f"No fixture mapping for workflow '{workflow}'")

    fixture_path = fixtures_dir / filename
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    return json.loads(fixture_path.read_text(encoding="utf-8"))


_TEST_PROVIDER_MODELS = [
    ModelDescriptor(
        id="fixture",
        display_name="Fixture Provider (offline)",
        capabilities={ProviderCapability.TEXT},
        routing_hints={"offline": True, "fixtures": True},
    )
]

_TEST_PROVIDER_METADATA = ProviderMetadata(
    provider_id=_PROVIDER_ID,
    display_name="Test Fixture Provider",
    models=_TEST_PROVIDER_MODELS,
    default_model="fixture",
    capabilities={ProviderCapability.TEXT},
    security_flags={"writes_allowed": False, "read_only": True, "offline": True},
    extra={
        "enabled_env": _ENABLE_ENV,
        "fixtures_dir_env": _FIXTURES_DIR_ENV,
        "default_fixtures_dir": str(_DEFAULT_FIXTURES_DIR),
    },
)


class TestFixtureProvider(ProviderContext):
    """ProviderContext that returns canned responses from fixtures."""

    def __init__(
        self,
        metadata: ProviderMetadata,
        hooks: Optional[ProviderHooks] = None,
        *,
        model: Optional[str] = None,
    ):
        super().__init__(metadata, hooks)
        self._model = model or metadata.default_model or "fixture"

    def _execute(self, request: ProviderRequest) -> ProviderResult:
        start_time = time.perf_counter()

        workflow = None
        if isinstance(request.metadata, dict):
            workflow = request.metadata.get("workflow")

        if not isinstance(workflow, str) or not workflow:
            workflow = "plan_review"

        fixture = _load_fixture(workflow=workflow)
        tokens = fixture.get("tokens", {}) if isinstance(fixture, dict) else {}

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ProviderResult(
            content=str(fixture.get("content", "")),
            provider_id=self._metadata.provider_id,
            model_used=str(fixture.get("model", self._model)),
            status=ProviderStatus.SUCCESS,
            tokens=TokenUsage(
                input_tokens=int(tokens.get("prompt", 0) or 0),
                output_tokens=int(tokens.get("completion", 0) or 0),
                total_tokens=int(tokens.get("total", 0) or 0),
            ),
            duration_ms=round(duration_ms, 2),
            raw_payload={
                "fixture_version": fixture.get("fixture_version"),
                "workflow": fixture.get("workflow"),
                "prompt_id": fixture.get("prompt_id"),
                "cached": fixture.get("cached"),
                "timestamp": fixture.get("timestamp"),
            },
        )


def _factory(
    *,
    hooks: ProviderHooks,
    model: Optional[str] = None,
    dependencies: Optional[Dict[str, object]] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> ProviderContext:
    return TestFixtureProvider(_TEST_PROVIDER_METADATA, hooks, model=model)


def _availability_check() -> bool:
    if not _is_enabled():
        return False
    return _resolve_fixtures_dir().exists()


register_provider(
    _PROVIDER_ID,
    factory=_factory,
    metadata=_TEST_PROVIDER_METADATA,
    availability_check=_availability_check,
    priority=0,
    description="Offline fixture-backed provider (tests/verification only)",
    tags=("test", "offline"),
)
