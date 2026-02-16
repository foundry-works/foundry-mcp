"""Deterministic integration coverage for consultation fallback behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

from foundry_mcp.core.ai_consultation import (
    ConsultationOrchestrator,
    ConsultationRequest,
    ConsultationWorkflow,
    ProviderResponse,
    ResultCache,
)
from foundry_mcp.core.llm_config import ConsultationConfig, WorkflowConsultationConfig
from foundry_mcp.core.providers import ProviderResult, ProviderStatus


class _FakeProvider:
    def __init__(self, responses: List[ProviderResult]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def generate(self, _request) -> ProviderResult:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return ProviderResult(
            content="",
            status=ProviderStatus.ERROR,
            provider_id="unknown",
            model_used="unknown",
            stderr="exhausted fake responses",
        )


def _result(
    *,
    provider_id: str,
    success: bool,
    content: str = "",
    error: str = "simulated failure",
) -> ProviderResult:
    return ProviderResult(
        content=content if success else "",
        status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
        provider_id=provider_id,
        model_used=f"{provider_id}-model",
        stderr=None if success else error,
    )


def _make_orchestrator(
    tmp_path: Path,
    *,
    priority: List[str],
    max_retries: int = 0,
    workflows: Dict[str, WorkflowConsultationConfig] | None = None,
) -> ConsultationOrchestrator:
    priority_specs = [
        entry if entry.startswith("[") else f"[cli]{entry}"
        for entry in priority
    ]
    config = ConsultationConfig(
        priority=priority_specs,
        default_timeout=30.0,
        max_retries=max_retries,
        retry_delay=0.0,
        fallback_enabled=True,
        cache_ttl=60,
        workflows=workflows or {},
    )
    orchestrator = ConsultationOrchestrator(
        cache=ResultCache(base_dir=tmp_path / "cache", default_ttl=60),
        config=config,
    )
    orchestrator._build_prompt = lambda _request: "test prompt"  # type: ignore[method-assign]
    return orchestrator


def test_single_model_fallback_respects_priority_order(tmp_path):
    orchestrator = _make_orchestrator(
        tmp_path,
        priority=["provider-a", "provider-b", "provider-c"],
        max_retries=0,
    )
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.PLAN_REVIEW,
        prompt_id="TEST_PROMPT",
        context={},
    )

    providers = {
        "provider-a": _FakeProvider([
            _result(provider_id="provider-a", success=False, error="rate limit"),
        ]),
        "provider-b": _FakeProvider([
            _result(provider_id="provider-b", success=True, content="fallback success"),
        ]),
        "provider-c": _FakeProvider([
            _result(provider_id="provider-c", success=True, content="should not run"),
        ]),
    }
    resolve_order: List[str] = []

    def _resolve(provider_id, **_kwargs):
        resolve_order.append(provider_id)
        return providers[provider_id]

    with patch(
        "foundry_mcp.core.ai_consultation.orchestrator.check_provider_available",
        return_value=True,
    ), patch(
        "foundry_mcp.core.ai_consultation.orchestrator.resolve_provider",
        side_effect=_resolve,
    ):
        outcome = orchestrator.consult(request, use_cache=False)

    assert outcome.error is None
    assert outcome.provider_id == "provider-b"
    assert outcome.content == "fallback success"
    assert resolve_order == ["provider-a", "provider-b"]
    assert providers["provider-c"].calls == 0


def test_single_model_retries_before_fallback(tmp_path):
    orchestrator = _make_orchestrator(
        tmp_path,
        priority=["provider-a", "provider-b"],
        max_retries=2,
    )
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.PLAN_REVIEW,
        prompt_id="TEST_PROMPT",
        context={},
    )

    providers = {
        "provider-a": _FakeProvider([
            _result(provider_id="provider-a", success=False, error="timed out"),
            _result(provider_id="provider-a", success=False, error="timed out"),
            _result(provider_id="provider-a", success=True, content="retry success"),
        ]),
        "provider-b": _FakeProvider([
            _result(provider_id="provider-b", success=True, content="fallback success"),
        ]),
    }
    resolve_order: List[str] = []

    def _resolve(provider_id, **_kwargs):
        resolve_order.append(provider_id)
        return providers[provider_id]

    with patch(
        "foundry_mcp.core.ai_consultation.orchestrator.check_provider_available",
        return_value=True,
    ), patch(
        "foundry_mcp.core.ai_consultation.orchestrator.resolve_provider",
        side_effect=_resolve,
    ):
        outcome = orchestrator.consult(request, use_cache=False)

    assert outcome.error is None
    assert outcome.provider_id == "provider-a"
    assert outcome.content == "retry success"
    assert providers["provider-a"].calls == 3
    assert providers["provider-b"].calls == 0
    assert resolve_order == ["provider-a", "provider-a", "provider-a"]


def test_parallel_fallback_stops_once_min_models_met(tmp_path):
    workflows = {"fidelity_review": WorkflowConsultationConfig(min_models=2)}
    orchestrator = _make_orchestrator(
        tmp_path,
        priority=["provider-a", "provider-b", "provider-c", "provider-d"],
        workflows=workflows,
    )
    request = ConsultationRequest(
        workflow=ConsultationWorkflow.FIDELITY_REVIEW,
        prompt_id="TEST_PROMPT",
        context={},
    )

    responses = {
        "provider-a": ProviderResponse(
            provider_id="provider-a",
            model_used="provider-a-model",
            content="primary success",
            success=True,
        ),
        "provider-b": ProviderResponse(
            provider_id="provider-b",
            model_used="provider-b-model",
            content="",
            success=False,
            error="simulated failure",
        ),
        "provider-c": ProviderResponse(
            provider_id="provider-c",
            model_used="provider-c-model",
            content="fallback success",
            success=True,
        ),
        "provider-d": ProviderResponse(
            provider_id="provider-d",
            model_used="provider-d-model",
            content="should not run",
            success=True,
        ),
    }
    call_order: List[str] = []

    async def _fake_execute(request, prompt, resolved):
        call_order.append(resolved.provider_id)
        return replace(responses[resolved.provider_id])

    orchestrator._execute_single_provider_async = _fake_execute  # type: ignore[method-assign]

    with patch(
        "foundry_mcp.core.ai_consultation.orchestrator.check_provider_available",
        return_value=True,
    ):
        outcome = orchestrator.consult(request, use_cache=False)

    assert [r.provider_id for r in outcome.responses] == [
        "provider-a",
        "provider-b",
        "provider-c",
    ]
    assert sum(1 for r in outcome.responses if r.success and r.content) == 2
    assert "provider-d" not in call_order
    assert any("attempting fallback for 1 more" in w for w in outcome.warnings)
