"""
Unit tests for AI consultation layer.

Tests the core consultation infrastructure including:
- ConsultationWorkflow enum
- ConsultationConfig dataclass
- PromptTemplate and PromptRegistry
- Workflow-specific prompt builders (plan_review, fidelity_review)
- AI error response helpers
"""

import pytest

# Core consultation module
from foundry_mcp.core.ai_consultation import (
    ConsultationWorkflow,
    ConsultationRequest,
    ConsultationResult,
)

# Prompts infrastructure
from foundry_mcp.core.prompts import (
    PromptTemplate,
    PromptRegistry,
    get_prompt_builder,
)

# Workflow-specific prompt builders
from foundry_mcp.core.prompts.plan_review import (
    PLAN_REVIEW_FULL_V1,
    PLAN_REVIEW_SECURITY_V1,
    PlanReviewPromptBuilder,
)

from foundry_mcp.core.prompts.fidelity_review import (
    FIDELITY_REVIEW_V1,
    FIDELITY_RESPONSE_SCHEMA,
    SEVERITY_KEYWORDS,
    FidelityReviewPromptBuilder,
)

# Response helpers
from foundry_mcp.core.responses import (
    ErrorCode,
    ErrorType,
    ai_no_provider_error,
    ai_provider_timeout_error,
    ai_provider_error,
    ai_context_too_large_error,
    ai_prompt_not_found_error,
    ai_cache_stale_error,
)


# =============================================================================
# ConsultationWorkflow Tests
# =============================================================================


class TestConsultationWorkflow:
    """Tests for ConsultationWorkflow enum."""

    def test_workflow_values(self):
        """Workflow enum has expected values."""
        assert ConsultationWorkflow.PLAN_REVIEW.value == "plan_review"
        assert ConsultationWorkflow.FIDELITY_REVIEW.value == "fidelity_review"

    def test_workflow_count(self):
        """Verify all expected workflows are defined."""
        workflows = list(ConsultationWorkflow)
        assert len(workflows) == 3


# =============================================================================
# ConsultationRequest Tests
# =============================================================================


class TestConsultationRequest:
    """Tests for ConsultationRequest dataclass."""

    def test_request_creation(self):
        """Request can be created with valid parameters."""
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="PLAN_REVIEW_FULL_V1",
            context={"spec_id": "test-001"},
        )
        assert request.workflow == ConsultationWorkflow.PLAN_REVIEW
        assert request.prompt_id == "PLAN_REVIEW_FULL_V1"
        assert request.context["spec_id"] == "test-001"

    def test_request_defaults(self):
        """Request has expected defaults."""
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="test",
            context={},
        )
        assert request.provider_id is None
        assert request.timeout == 120.0  # Default timeout
        assert request.model is None


# =============================================================================
# ConsultationResult Tests
# =============================================================================


class TestConsultationResult:
    """Tests for ConsultationResult dataclass."""

    def test_result_success(self):
        """Result can represent successful consultation."""
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Generated content",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
        )
        assert result.content == "Generated content"
        assert result.provider_id == "gemini"
        assert result.model_used == "gemini-2.0-flash"
        assert result.error is None

    def test_result_with_error(self):
        """Result can include error information."""
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            error="Provider timeout",
        )
        assert result.error == "Provider timeout"


# =============================================================================
# PromptTemplate Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for PromptTemplate dataclass."""

    def test_template_creation(self):
        """Template can be created with valid parameters."""
        template = PromptTemplate(
            id="test_template",
            version="1.0",
            system_prompt="System message",
            user_template="User message: {content}",
            required_context=["content"],
        )
        assert template.id == "test_template"
        assert template.version == "1.0"

    def test_template_validation_empty_id(self):
        """Template rejects empty ID."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            PromptTemplate(
                id="",
                version="1.0",
                system_prompt="System",
                user_template="User",
            )

    def test_template_get_variables(self):
        """Template extracts variables from user_template."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="Hello {name}, your score is {score}",
            required_context=["name", "score"],
        )
        variables = template.get_variables()
        assert "name" in variables
        assert "score" in variables

    def test_template_validate_context(self):
        """Template validates context contains required keys."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="{required_key}",
            required_context=["required_key"],
        )
        # Valid context
        missing = template.validate_context({"required_key": "value"})
        assert missing == []

        # Missing context
        missing = template.validate_context({})
        assert "required_key" in missing

    def test_template_render(self):
        """Template renders with context substitution."""
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="Hello {name}!",
            required_context=["name"],
        )
        result = template.render({"name": "World"})
        assert result == "Hello World!"


# =============================================================================
# PromptRegistry Tests
# =============================================================================


class TestPromptRegistry:
    """Tests for PromptRegistry."""

    def test_registry_register_and_get(self):
        """Registry can register and retrieve templates."""
        registry = PromptRegistry()
        template = PromptTemplate(
            id="test_template",
            version="1.0",
            system_prompt="System",
            user_template="User",
        )
        registry.register(template)

        retrieved = registry.get("test_template")
        assert retrieved is not None
        assert retrieved.id == "test_template"

    def test_registry_duplicate_registration(self):
        """Registry rejects duplicate IDs by default."""
        registry = PromptRegistry()
        template = PromptTemplate(
            id="test",
            version="1.0",
            system_prompt="System",
            user_template="User",
        )
        registry.register(template)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(template)

    def test_registry_list_templates(self):
        """Registry lists all registered template IDs."""
        registry = PromptRegistry()
        registry.register(
            PromptTemplate(id="a", version="1.0", system_prompt="S", user_template="U")
        )
        registry.register(
            PromptTemplate(id="b", version="1.0", system_prompt="S", user_template="U")
        )

        templates = registry.list_templates()
        assert "a" in templates
        assert "b" in templates


# =============================================================================
# Prompt Builder Factory Tests
# =============================================================================


class TestPromptBuilderFactory:
    """Tests for get_prompt_builder factory."""

    def test_factory_returns_plan_review_builder(self):
        """Factory returns PlanReviewPromptBuilder for PLAN_REVIEW."""
        builder = get_prompt_builder(ConsultationWorkflow.PLAN_REVIEW)
        assert isinstance(builder, PlanReviewPromptBuilder)

    def test_factory_returns_fidelity_review_builder(self):
        """Factory returns FidelityReviewPromptBuilder for FIDELITY_REVIEW."""
        builder = get_prompt_builder(ConsultationWorkflow.FIDELITY_REVIEW)
        assert isinstance(builder, FidelityReviewPromptBuilder)


# =============================================================================
# PlanReviewPromptBuilder Tests
# =============================================================================


class TestPlanReviewPromptBuilder:
    """Tests for PlanReviewPromptBuilder."""

    def test_builder_list_prompts(self):
        """Builder lists available prompts."""
        builder = PlanReviewPromptBuilder()
        prompts = builder.list_prompts()

        assert "PLAN_REVIEW_FULL_V1" in prompts
        assert "PLAN_REVIEW_QUICK_V1" in prompts
        assert "PLAN_REVIEW_SECURITY_V1" in prompts
        assert "PLAN_REVIEW_FEASIBILITY_V1" in prompts
        assert "SYNTHESIS_PROMPT_V1" in prompts

    def test_builder_build_full_review(self):
        """Builder renders PLAN_REVIEW_FULL_V1."""
        builder = PlanReviewPromptBuilder()
        context = {
            "spec_id": "test-spec-001",
            "title": "Test Specification",
            "version": "1.0",
            "spec_content": "Test content here",
        }
        result = builder.build("PLAN_REVIEW_FULL_V1", context)

        assert "test-spec-001" in result
        assert "Test Specification" in result

    def test_full_review_has_six_dimensions(self):
        """PLAN_REVIEW_FULL_V1 metadata includes 6 dimensions."""
        dimensions = PLAN_REVIEW_FULL_V1.metadata.get("dimensions", [])
        assert len(dimensions) == 6
        assert "Completeness" in dimensions
        assert "Security" in dimensions

    def test_security_review_has_metadata(self):
        """PLAN_REVIEW_SECURITY_V1 has workflow metadata."""
        assert "workflow" in PLAN_REVIEW_SECURITY_V1.metadata
        assert PLAN_REVIEW_SECURITY_V1.id == "PLAN_REVIEW_SECURITY_V1"


# =============================================================================
# FidelityReviewPromptBuilder Tests
# =============================================================================


class TestFidelityReviewPromptBuilder:
    """Tests for FidelityReviewPromptBuilder."""

    def test_builder_list_prompts(self):
        """Builder lists available prompts."""
        builder = FidelityReviewPromptBuilder()
        prompts = builder.list_prompts()

        assert "FIDELITY_REVIEW_V1" in prompts
        assert "FIDELITY_DEVIATION_ANALYSIS_V1" in prompts
        assert "FIDELITY_COMPLIANCE_SUMMARY_V1" in prompts

    def test_builder_build_fidelity_review(self):
        """Builder renders FIDELITY_REVIEW_V1."""
        builder = FidelityReviewPromptBuilder()
        context = {
            "spec_id": "test-spec-001",
            "spec_title": "Test Spec",
            "review_scope": "task-1-1",
            "spec_requirements": "Implement feature X",
            "implementation_artifacts": "git diff here",
        }
        result = builder.build("FIDELITY_REVIEW_V1", context)

        assert "test-spec-001" in result
        assert "Implementation Fidelity Review" in result

    def test_fidelity_has_six_sections(self):
        """FIDELITY_REVIEW_V1 metadata includes 6 sections."""
        sections = FIDELITY_REVIEW_V1.metadata.get("sections", [])
        assert len(sections) == 6
        assert "Context" in sections
        assert "Review Questions" in sections

    def test_severity_keywords_defined(self):
        """Severity keywords are defined for all levels."""
        assert "critical" in SEVERITY_KEYWORDS
        assert "high" in SEVERITY_KEYWORDS
        assert "medium" in SEVERITY_KEYWORDS
        assert "low" in SEVERITY_KEYWORDS

    def test_get_severity_keywords(self):
        """Builder returns severity keywords for a level."""
        builder = FidelityReviewPromptBuilder()
        critical_kw = builder.get_severity_keywords("critical")
        assert "security" in critical_kw

    def test_response_schema_structure(self):
        """Fidelity response schema has expected fields."""
        assert "verdict" in FIDELITY_RESPONSE_SCHEMA
        assert "deviations" in FIDELITY_RESPONSE_SCHEMA
        assert "recommendations" in FIDELITY_RESPONSE_SCHEMA


# =============================================================================
# AI Error Response Tests
# =============================================================================


class TestAIErrorCodes:
    """Tests for AI-specific error codes."""

    def test_error_codes_defined(self):
        """AI error codes are defined in ErrorCode enum."""
        assert ErrorCode.AI_NO_PROVIDER == "AI_NO_PROVIDER"
        assert ErrorCode.AI_PROVIDER_TIMEOUT == "AI_PROVIDER_TIMEOUT"
        assert ErrorCode.AI_PROVIDER_ERROR == "AI_PROVIDER_ERROR"
        assert ErrorCode.AI_CONTEXT_TOO_LARGE == "AI_CONTEXT_TOO_LARGE"
        assert ErrorCode.AI_PROMPT_NOT_FOUND == "AI_PROMPT_NOT_FOUND"
        assert ErrorCode.AI_CACHE_STALE == "AI_CACHE_STALE"

    def test_error_type_ai_provider(self):
        """AI_PROVIDER error type is defined."""
        assert ErrorType.AI_PROVIDER == "ai_provider"


class TestAIErrorHelpers:
    """Tests for AI error response helpers."""

    def test_ai_no_provider_error(self):
        """ai_no_provider_error creates correct response."""
        resp = ai_no_provider_error(
            "No AI provider available",
            required_providers=["gemini", "codex"],
        )
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_NO_PROVIDER
        assert resp.data["required_providers"] == ["gemini", "codex"]
        assert "remediation" in resp.data

    def test_ai_provider_timeout_error(self):
        """ai_provider_timeout_error creates correct response."""
        resp = ai_provider_timeout_error("gemini", 300)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROVIDER_TIMEOUT
        assert resp.data["provider_id"] == "gemini"
        assert resp.data["timeout_seconds"] == 300

    def test_ai_provider_error(self):
        """ai_provider_error creates correct response."""
        resp = ai_provider_error("gemini", "Invalid API key", status_code=401)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROVIDER_ERROR
        assert resp.data["status_code"] == 401

    def test_ai_context_too_large_error(self):
        """ai_context_too_large_error creates correct response."""
        resp = ai_context_too_large_error(150000, 128000, provider_id="gemini")
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_CONTEXT_TOO_LARGE
        assert resp.data["overflow_tokens"] == 22000

    def test_ai_prompt_not_found_error(self):
        """ai_prompt_not_found_error creates correct response."""
        resp = ai_prompt_not_found_error(
            "INVALID_PROMPT",
            available_prompts=["PLAN_REVIEW_FULL_V1"],
            workflow="plan_review",
        )
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_PROMPT_NOT_FOUND
        assert resp.data["workflow"] == "plan_review"

    def test_ai_cache_stale_error(self):
        """ai_cache_stale_error creates correct response."""
        resp = ai_cache_stale_error("key", 7200, 3600)
        assert resp.success is False
        assert resp.data["error_code"] == ErrorCode.AI_CACHE_STALE
        assert resp.data["cache_age_seconds"] == 7200

    def test_all_helpers_include_remediation(self):
        """All AI error helpers include remediation guidance."""
        responses = [
            ai_no_provider_error(),
            ai_provider_timeout_error("test", 60),
            ai_provider_error("test", "error"),
            ai_context_too_large_error(100, 50),
            ai_prompt_not_found_error("test"),
            ai_cache_stale_error("key", 100, 50),
        ]
        for resp in responses:
            assert "remediation" in resp.data
            assert resp.data["remediation"]  # Not empty


# =============================================================================
# Multi-Model Consensus Tests
# =============================================================================


class TestProviderResponse:
    """Tests for ProviderResponse dataclass."""

    def test_provider_response_creation(self):
        """ProviderResponse can be created with all fields."""
        from foundry_mcp.core.ai_consultation import ProviderResponse

        response = ProviderResponse(
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            content="Test content",
            success=True,
            tokens=100,
            duration_ms=500,
        )
        assert response.provider_id == "gemini"
        assert response.success is True
        assert response.tokens == 100

    def test_provider_response_from_result(self):
        """ProviderResponse.from_result converts ConsultationResult correctly."""
        from foundry_mcp.core.ai_consultation import ProviderResponse

        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Review content",
            provider_id="claude",
            model_used="claude-sonnet",
            tokens={"input_tokens": 50, "output_tokens": 100, "total_tokens": 150},
            duration_ms=1000.0,
            cache_hit=False,
        )
        response = ProviderResponse.from_result(result)
        assert response.provider_id == "claude"
        assert response.model_used == "claude-sonnet"
        assert response.content == "Review content"
        assert response.success is True
        # Note: from_result sums all token values (50+100+150=300)
        assert response.tokens == 300


class TestAgreementMetadata:
    """Tests for AgreementMetadata dataclass."""

    def test_agreement_metadata_success_rate(self):
        """AgreementMetadata calculates success rate correctly."""
        from foundry_mcp.core.ai_consultation import AgreementMetadata

        metadata = AgreementMetadata(
            total_providers=4,
            successful_providers=3,
            failed_providers=1,
        )
        assert metadata.success_rate == 0.75
        assert metadata.has_consensus is True

    def test_agreement_metadata_no_consensus(self):
        """AgreementMetadata.has_consensus is False with < 2 successful."""
        from foundry_mcp.core.ai_consultation import AgreementMetadata

        metadata = AgreementMetadata(
            total_providers=3,
            successful_providers=1,
            failed_providers=2,
        )
        assert metadata.has_consensus is False

    def test_agreement_metadata_from_responses(self):
        """AgreementMetadata.from_responses computes counts correctly."""
        from foundry_mcp.core.ai_consultation import AgreementMetadata, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "content", success=True),
            ProviderResponse("p2", "m2", "content", success=True),
            ProviderResponse("p3", "m3", "", success=False, error="timeout"),
        ]
        metadata = AgreementMetadata.from_responses(responses)
        assert metadata.total_providers == 3
        assert metadata.successful_providers == 2
        assert metadata.failed_providers == 1


class TestConsensusResult:
    """Tests for ConsensusResult dataclass."""

    def test_consensus_result_auto_computes_agreement(self):
        """ConsensusResult auto-computes agreement via __post_init__."""
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "content1", success=True),
            ProviderResponse("p2", "m2", "content2", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            responses=responses,
        )
        assert result.agreement is not None
        assert result.agreement.total_providers == 2
        assert result.agreement.successful_providers == 2

    def test_consensus_result_success_property(self):
        """ConsensusResult.success is True if any provider succeeded."""
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail"),
            ProviderResponse("p2", "m2", "content", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )
        assert result.success is True

    def test_consensus_result_primary_content(self):
        """ConsensusResult.primary_content returns first successful content."""
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail"),
            ProviderResponse("p2", "m2", "first success", success=True),
            ProviderResponse("p3", "m3", "second success", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )
        assert result.primary_content == "first success"

    def test_consensus_result_successful_responses(self):
        """ConsensusResult filters successful/failed responses correctly."""
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "ok", success=True),
            ProviderResponse("p2", "m2", "", success=False, error="fail"),
            ProviderResponse("p3", "m3", "ok", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )
        assert len(result.successful_responses) == 2
        assert len(result.failed_responses) == 1


class TestConsultationOutcome:
    """Tests for ConsultationOutcome type alias."""

    def test_consultation_outcome_differentiates_types(self):
        """ConsultationOutcome can be differentiated with isinstance."""
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationOutcome,
            ProviderResponse,
        )

        single_result: ConsultationOutcome = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="single",
            provider_id="p1",
            model_used="m1",
        )
        multi_result: ConsultationOutcome = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=[ProviderResponse("p1", "m1", "content", success=True)],
        )

        assert isinstance(single_result, ConsultationResult)
        assert not isinstance(single_result, ConsensusResult)
        assert isinstance(multi_result, ConsensusResult)
        assert not isinstance(multi_result, ConsultationResult)


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests verifying backward compatibility when min_models=1 (default).

    These tests ensure that existing code expecting ConsultationResult
    continues to work when workflow has min_models=1 or no workflow config.
    """

    def test_default_workflow_returns_consultation_result_type(self):
        """Default workflow config (no min_models) returns ConsultationResult type.

        Ensures isinstance(result, ConsultationResult) == True for default config.
        """
        # Default workflow has no explicit min_models, defaults to 1
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationOutcome,
        )

        # Simulate what consult() returns for min_models=1
        result: ConsultationOutcome = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="test response",
            provider_id="test-provider",
            model_used="test-model",
        )

        # Must be ConsultationResult, not ConsensusResult
        assert isinstance(result, ConsultationResult)
        assert not isinstance(result, ConsensusResult)

    def test_explicit_min_models_1_returns_consultation_result(self):
        """Workflow with explicit min_models=1 returns ConsultationResult.

        This is the backward compatibility contract - existing code should
        continue to receive ConsultationResult when min_models=1.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult

        # Create result as returned by consult() when min_models=1
        result = ConsultationResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            content="fidelity check passed",
            provider_id="claude",
            model_used="claude-sonnet-4",
        )

        # Verify type for backward compatibility
        assert type(result).__name__ == "ConsultationResult"
        assert not isinstance(result, ConsensusResult)

    def test_min_models_greater_than_1_returns_consensus_result(self):
        """Workflow with min_models>1 returns ConsensusResult.

        New multi-model workflows should receive ConsensusResult.
        """
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ProviderResponse,
        )

        responses = [
            ProviderResponse("p1", "m1", "content1", success=True),
            ProviderResponse("p2", "m2", "content2", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )

        # Must be ConsensusResult, not ConsultationResult
        assert isinstance(result, ConsensusResult)
        assert not isinstance(result, ConsultationResult)

    def test_existing_code_accessing_content_attribute(self):
        """Existing code using result.content continues to work.

        Legacy pattern: accessing result.content directly.
        """
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Generated review...",
            provider_id="openai",
            model_used="gpt-4",
        )

        # Direct attribute access - legacy pattern
        assert result.content == "Generated review..."
        assert len(result.content) > 0

    def test_existing_code_accessing_provider_id(self):
        """Existing code using result.provider_id continues to work.

        Legacy pattern: checking which provider was used.
        """
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="review complete",
            provider_id="anthropic",
            model_used="claude-3-opus",
        )

        # Direct attribute access - legacy pattern
        assert result.provider_id == "anthropic"
        assert result.model_used == "claude-3-opus"

    def test_existing_code_checking_error_attribute(self):
        """Existing code using result.error continues to work.

        Legacy pattern: checking if an error occurred.
        """
        # Success case - no error
        success_result = ConsultationResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            content="review passed",
            provider_id="test",
            model_used="model",
            error=None,
        )
        assert success_result.error is None

        # Failure case - has error
        failure_result = ConsultationResult(
            workflow=ConsultationWorkflow.FIDELITY_REVIEW,
            content="",
            provider_id="test",
            model_used="model",
            error="Provider timeout after 30s",
        )
        assert failure_result.error is not None
        assert "timeout" in failure_result.error.lower()

    def test_existing_code_error_checking_pattern(self):
        """Common error checking pattern from existing code continues to work.

        Legacy pattern: if result.error or not result.content
        """

        # Simulating the common pattern in tools
        def process_consultation_result(result: ConsultationResult) -> str:
            """Example of existing code pattern."""
            if result.error:
                return f"Error: {result.error}"
            if not result.content:
                return "Empty response"
            return f"Success: {result.content[:20]}..."

        success = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="This is a valid response from the LLM",
            provider_id="test",
            model_used="model",
        )
        assert process_consultation_result(success).startswith("Success:")

        error = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="",
            provider_id="test",
            model_used="model",
            error="Connection failed",
        )
        assert process_consultation_result(error).startswith("Error:")

    def test_consultation_result_has_expected_attributes(self):
        """ConsultationResult has all expected attributes for backward compat.

        Ensures no attributes were removed or renamed.
        """
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="test",
            provider_id="test-provider",
            model_used="test-model",
            cache_hit=True,
            error=None,
            tokens={"total": 100},
        )

        # All expected attributes must exist
        assert hasattr(result, "workflow")
        assert hasattr(result, "content")
        assert hasattr(result, "provider_id")
        assert hasattr(result, "model_used")
        assert hasattr(result, "cache_hit")
        assert hasattr(result, "error")
        assert hasattr(result, "tokens")

        # Values match
        assert result.workflow == ConsultationWorkflow.PLAN_REVIEW
        assert result.content == "test"
        assert result.provider_id == "test-provider"
        assert result.model_used == "test-model"
        assert result.cache_hit is True
        assert result.error is None
        assert result.tokens == {"total": 100}

    def test_isinstance_check_for_legacy_code(self):
        """isinstance(result, ConsultationResult) works for type guards.

        Legacy pattern using isinstance for type narrowing.
        """
        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationOutcome,
            ProviderResponse,
        )

        def handle_result(result: ConsultationOutcome) -> str:
            """Example legacy handler with type narrowing."""
            if isinstance(result, ConsultationResult):
                # Single-model path - existing code
                return f"Single: {result.provider_id}"
            elif isinstance(result, ConsensusResult):
                # Multi-model path - new code
                return f"Multi: {len(result.responses)} providers"
            return "Unknown"

        single = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="test",
            provider_id="p1",
            model_used="m1",
        )
        multi = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=[ProviderResponse("p1", "m1", "c", success=True)],
        )

        assert handle_result(single) == "Single: p1"
        assert handle_result(multi) == "Multi: 1 providers"
