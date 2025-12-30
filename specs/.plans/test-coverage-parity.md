# Test Coverage Parity: AI Consultation & Research Router

## Objective

Bring test coverage to parity between AI Consultation and Research Router modules, then add workflow/orchestrator tests and E2E tests with mocked providers for both.

## Status: ✅ COMPLETE

All planned tests have been implemented and are passing.

## Final State

### Research Router Tests ✅
- `tests/unit/test_core/research/test_models.py` - 53 tests (Pydantic models, enums)
- `tests/unit/test_core/research/test_memory.py` - 49 tests (storage CRUD, TTL, concurrency)
- `tests/unit/test_core/research/test_workflows.py` - 32 tests (workflow classes with mocked providers)
- `tests/tools/unified/test_research.py` - 47 tests (router dispatch, mocked workflows)
- `tests/integration/test_research_e2e.py` - 20 tests (NEW - E2E workflow tests)

### AI Consultation Tests ✅
- `tests/unit/test_ai_consultation.py` - 95 tests:
  - ✅ ConsultationWorkflow enum
  - ✅ ConsultationRequest dataclass
  - ✅ ConsultationResult dataclass
  - ✅ ProviderResponse dataclass
  - ✅ AgreementMetadata dataclass
  - ✅ ConsensusResult dataclass
  - ✅ ResolvedProvider dataclass (4 tests)
  - ✅ ResultCache class (14 tests)
  - ✅ ConsultationOrchestrator class (15 tests)
  - ✅ ConsultationOrchestratorMultiModel class (3 tests)
- `tests/integration/test_ai_consultation_e2e.py` - 13 tests (NEW - E2E workflow tests)

## Completed Work

### 1. ✅ Added Missing AI Consultation Dataclass Tests
File: `tests/unit/test_ai_consultation.py`

Added tests for:
- `ResolvedProvider` dataclass (creation minimal/full, overrides default, spec_str)
- `ResultCache` class (init, set/get, TTL expiration, invalidation, stats, corrupt file handling)

### 2. ✅ Added ConsultationOrchestrator Tests
File: `tests/unit/test_ai_consultation.py`

Added tests for:
- `ConsultationOrchestrator.is_available()` (no providers, with providers, specific provider)
- `ConsultationOrchestrator.get_available_providers()`
- `ConsultationOrchestrator.consult()` with mocked providers (cache hit, success, all fail, no providers)
- `ConsultationOrchestrator.consult()` fallback behavior
- `ConsultationOrchestrator._generate_cache_key()` (deterministic, explicit override)
- Timeout from request
- Duration tracking
- Multi-model mode (ConsensusResult, partial failures, all fail)

### 3. ✅ Added E2E Tests with Mocked Providers

#### For AI Consultation
File: `tests/integration/test_ai_consultation_e2e.py`

Tests full flow:
- Plan review workflow end-to-end
- Plan review cached response
- Plan review cache bypass
- Fidelity review workflow end-to-end
- Multi-model consensus plan review
- Multi-model partial failure with fallback
- Multi-model all providers fail
- Fallback to second provider
- No fallback when disabled
- No providers available error
- Invalid prompt ID error
- Single/multi-model response structure

#### For Research Router
File: `tests/integration/test_research_e2e.py`

Tests full flow through router:
- `research action=chat` -> ChatWorkflow -> response envelope (new thread, continue, failure)
- `research action=consensus` -> ConsensusWorkflow -> response envelope (synthesize, all_responses)
- `research action=thinkdeep` -> ThinkDeepWorkflow -> response envelope (new, continue, converged)
- `research action=ideate` -> IdeateWorkflow -> response envelope (generate, cluster)
- Route action recommends workflow
- Feature flag gating
- Thread operations (list, get, delete)
- Response envelope structure (success, error)
- Error handling (invalid action, missing param, workflow exception)

## Key Files

### AI Consultation
- Source: `src/foundry_mcp/core/ai_consultation.py`
- Tests: `tests/unit/test_ai_consultation.py`

### Research Router
- Source: `src/foundry_mcp/core/research/` (models, memory, workflows)
- Source: `src/foundry_mcp/tools/unified/research.py` (router)
- Tests: `tests/unit/test_core/research/` (models, memory, workflows)
- Tests: `tests/tools/unified/test_research.py` (router)

## Test Patterns to Follow

### Mocking Providers
```python
from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

mock_context = MagicMock()
mock_context.generate.return_value = ProviderResult(
    content="Mock response",
    status=ProviderStatus.SUCCESS,
    provider_id="gemini",
    model_used="gemini-2.0-flash",
    tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
    duration_ms=500.0,
)
```

### Response Structure Checks
```python
assert result.success is True
assert isinstance(result.content, str)
assert result.metadata["key"] == expected_value
assert result.meta["version"] == "response-v2"  # For router responses
```

## PR Info
- Branch: `sandbox/foundry-mcp-20251229-0843`
- PR: https://github.com/tylerburleigh/foundry-mcp/pull/9
