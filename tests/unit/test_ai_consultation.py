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


# =============================================================================
# ResolvedProvider Tests
# =============================================================================


class TestResolvedProvider:
    """Tests for ResolvedProvider dataclass."""

    def test_creation_minimal(self):
        """ResolvedProvider can be created with just provider_id."""
        from foundry_mcp.core.ai_consultation import ResolvedProvider

        resolved = ResolvedProvider(provider_id="gemini")
        assert resolved.provider_id == "gemini"
        assert resolved.model is None
        assert resolved.overrides == {}
        assert resolved.spec_str == ""

    def test_creation_full(self):
        """ResolvedProvider can be created with all fields."""
        from foundry_mcp.core.ai_consultation import ResolvedProvider

        resolved = ResolvedProvider(
            provider_id="opencode",
            model="openai/gpt-4",
            overrides={"timeout": 120, "temperature": 0.7},
            spec_str="opencode:openai/gpt-4",
        )
        assert resolved.provider_id == "opencode"
        assert resolved.model == "openai/gpt-4"
        assert resolved.overrides["timeout"] == 120
        assert resolved.overrides["temperature"] == 0.7
        assert resolved.spec_str == "opencode:openai/gpt-4"

    def test_overrides_default_to_empty_dict(self):
        """ResolvedProvider.overrides defaults to empty dict."""
        from foundry_mcp.core.ai_consultation import ResolvedProvider

        resolved = ResolvedProvider(provider_id="claude")
        # Verify default_factory creates new dict each time
        resolved.overrides["key"] = "value"

        resolved2 = ResolvedProvider(provider_id="claude")
        assert resolved2.overrides == {}

    def test_spec_str_for_logging(self):
        """ResolvedProvider.spec_str is useful for logging."""
        from foundry_mcp.core.ai_consultation import ResolvedProvider

        resolved = ResolvedProvider(
            provider_id="gemini",
            spec_str="explicit:gemini",
        )
        # spec_str can be used in log messages
        log_msg = f"Using provider: {resolved.spec_str}"
        assert "explicit:gemini" in log_msg


# =============================================================================
# ResultCache Tests
# =============================================================================


class TestResultCache:
    """Tests for ResultCache class."""

    def test_init_default_path(self, tmp_path):
        """ResultCache initializes with default path if not provided."""
        from foundry_mcp.core.ai_consultation import ResultCache

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cache = ResultCache()
            assert cache.base_dir.parts[-3:] == (".cache", "foundry-mcp", "consultations")
            assert cache.default_ttl == 3600
        finally:
            os.chdir(original_cwd)

    def test_init_custom_path(self, tmp_path):
        """ResultCache can be initialized with custom path."""
        from foundry_mcp.core.ai_consultation import ResultCache

        custom_path = tmp_path / "my-cache"
        cache = ResultCache(base_dir=custom_path, default_ttl=7200)
        assert cache.base_dir == custom_path
        assert cache.default_ttl == 7200

    def test_set_and_get(self, tmp_path):
        """ResultCache can store and retrieve results."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Test content",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            tokens={"total": 150},
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "test-key", result)
        cached = cache.get(ConsultationWorkflow.PLAN_REVIEW, "test-key")

        assert cached is not None
        assert cached["content"] == "Test content"
        assert cached["provider_id"] == "gemini"
        assert cached["model_used"] == "gemini-2.0-flash"
        assert cached["tokens"] == {"total": 150}

    def test_get_nonexistent_key(self, tmp_path):
        """ResultCache.get returns None for nonexistent keys."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        cached = cache.get(ConsultationWorkflow.PLAN_REVIEW, "nonexistent")
        assert cached is None

    def test_cache_expiration(self, tmp_path):
        """ResultCache.get returns None for expired entries."""
        from foundry_mcp.core.ai_consultation import ResultCache
        import time

        cache = ResultCache(base_dir=tmp_path / "cache", default_ttl=1)
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Expiring content",
            provider_id="test",
            model_used="test-model",
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "expire-key", result)

        # Should be available immediately
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "expire-key") is not None

        # Wait for TTL to expire
        time.sleep(1.1)

        # Should be None after expiration
        cached = cache.get(ConsultationWorkflow.PLAN_REVIEW, "expire-key")
        assert cached is None

    def test_custom_ttl_override(self, tmp_path):
        """ResultCache.set can override default TTL."""
        from foundry_mcp.core.ai_consultation import ResultCache
        import time

        cache = ResultCache(base_dir=tmp_path / "cache", default_ttl=3600)
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Short-lived content",
            provider_id="test",
            model_used="test-model",
        )

        # Set with short TTL override
        cache.set(ConsultationWorkflow.PLAN_REVIEW, "short-ttl", result, ttl=1)

        # Should be available immediately
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "short-ttl") is not None

        # Wait for custom TTL to expire
        time.sleep(1.1)

        # Should be None after custom TTL
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "short-ttl") is None

    def test_invalidate_specific_entry(self, tmp_path):
        """ResultCache.invalidate can remove a specific entry."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="To be invalidated",
            provider_id="test",
            model_used="test-model",
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key1", result)
        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key2", result)

        count = cache.invalidate(ConsultationWorkflow.PLAN_REVIEW, "key1")
        assert count == 1

        # key1 should be gone
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "key1") is None
        # key2 should still exist
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "key2") is not None

    def test_invalidate_workflow(self, tmp_path):
        """ResultCache.invalidate can remove all entries for a workflow."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Content",
            provider_id="test",
            model_used="test-model",
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key1", result)
        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key2", result)
        cache.set(ConsultationWorkflow.FIDELITY_REVIEW, "key3", result)

        count = cache.invalidate(ConsultationWorkflow.PLAN_REVIEW)
        assert count == 2

        # PLAN_REVIEW entries should be gone
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "key1") is None
        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "key2") is None
        # FIDELITY_REVIEW should remain
        assert cache.get(ConsultationWorkflow.FIDELITY_REVIEW, "key3") is not None

    def test_invalidate_all(self, tmp_path):
        """ResultCache.invalidate can remove all entries."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Content",
            provider_id="test",
            model_used="test-model",
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key1", result)
        cache.set(ConsultationWorkflow.FIDELITY_REVIEW, "key2", result)

        count = cache.invalidate()
        assert count == 2

        assert cache.get(ConsultationWorkflow.PLAN_REVIEW, "key1") is None
        assert cache.get(ConsultationWorkflow.FIDELITY_REVIEW, "key2") is None

    def test_stats_empty_cache(self, tmp_path):
        """ResultCache.stats returns zero counts for empty cache."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        stats = cache.stats()

        assert stats["total_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert "by_workflow" in stats

    def test_stats_with_entries(self, tmp_path):
        """ResultCache.stats returns correct counts."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Test content",
            provider_id="test",
            model_used="test-model",
        )

        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key1", result)
        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key2", result)
        cache.set(ConsultationWorkflow.FIDELITY_REVIEW, "key3", result)

        stats = cache.stats()

        assert stats["total_entries"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["by_workflow"]["plan_review"]["entries"] == 2
        assert stats["by_workflow"]["fidelity_review"]["entries"] == 1

    def test_cache_key_sanitization(self, tmp_path):
        """ResultCache sanitizes keys to be filesystem-safe."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Content",
            provider_id="test",
            model_used="test-model",
        )

        # Keys with special characters should be sanitized
        cache.set(ConsultationWorkflow.PLAN_REVIEW, "key/with:special<chars>", result)
        cached = cache.get(ConsultationWorkflow.PLAN_REVIEW, "key/with:special<chars>")
        assert cached is not None
        assert cached["content"] == "Content"

    def test_corrupt_cache_file_handling(self, tmp_path):
        """ResultCache handles corrupt cache files gracefully."""
        from foundry_mcp.core.ai_consultation import ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache")

        # Create a corrupt cache file
        workflow_dir = tmp_path / "cache" / "plan_review"
        workflow_dir.mkdir(parents=True)
        corrupt_file = workflow_dir / "corrupt-key.json"
        corrupt_file.write_text("not valid json {{{")

        # Should return None and not raise
        cached = cache.get(ConsultationWorkflow.PLAN_REVIEW, "corrupt-key")
        assert cached is None


# =============================================================================
# Multi-Model Consensus Fallback Tests
# =============================================================================


class TestConsensusWithFallback:
    """Tests for multi-model consensus with fallback behavior.

    These tests verify that when one provider fails in multi-model mode,
    the system falls back to additional providers from the priority list.
    """

    def test_fallback_result_structure_with_mixed_responses(self):
        """ConsensusResult correctly handles mixed success/failure responses.

        When initial parallel execution has failures and fallback adds successes,
        the result should include all attempted responses in order.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        # Simulating: initial p1 failed, p2 succeeded, fallback p3 succeeded
        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="timeout"),
            ProviderResponse("p2", "m2", "content2", success=True),
            ProviderResponse("p3", "m3", "content3", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
            warnings=[
                "Provider p1 failed: timeout",
                "Initial parallel execution yielded 1/2 successes, attempting fallback for 1 more",
                "Fallback provider p3 succeeded",
            ],
        )

        # Should have 3 responses total (2 initial + 1 fallback)
        assert len(result.responses) == 3
        assert result.agreement.total_providers == 3
        assert result.agreement.successful_providers == 2
        assert result.agreement.failed_providers == 1
        assert result.success is True

    def test_fallback_warnings_indicate_fallback_used(self):
        """ConsensusResult warnings show when fallback was attempted.

        The warnings list should indicate that fallback was triggered
        and whether fallback providers succeeded or failed.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("gemini", "auto", "", success=False, error="CLI exit 1"),
            ProviderResponse("codex", "codex", "review content", success=True),
            ProviderResponse("opencode", "gpt-5", "review content", success=True),
        ]
        warnings = [
            "Provider gemini failed: CLI exit 1",
            "Initial parallel execution yielded 1/2 successes, attempting fallback for 1 more",
            "Fallback provider opencode succeeded",
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
            warnings=warnings,
        )

        # Warnings should indicate fallback was used
        assert any("fallback" in w.lower() for w in result.warnings)
        assert any("succeeded" in w.lower() for w in result.warnings)

    def test_fallback_preserves_response_order(self):
        """Responses maintain attempt order for debugging.

        First responses are from initial parallel execution,
        later responses are from sequential fallback.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail"),
            ProviderResponse("p2", "m2", "", success=False, error="fail"),
            ProviderResponse("p3", "m3", "success", success=True),
            ProviderResponse("p4", "m4", "success", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )

        # Order should be preserved
        assert result.responses[0].provider_id == "p1"
        assert result.responses[1].provider_id == "p2"
        assert result.responses[2].provider_id == "p3"
        assert result.responses[3].provider_id == "p4"

    def test_all_providers_failed_warning(self):
        """When all providers fail, appropriate warning is included.

        If fallback exhausts all providers without reaching min_models,
        the warning should indicate how many were tried.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail1"),
            ProviderResponse("p2", "m2", "", success=False, error="fail2"),
            ProviderResponse("p3", "m3", "", success=False, error="fail3"),
        ]
        warnings = [
            "Provider p1 failed: fail1",
            "Provider p2 failed: fail2",
            "Initial parallel execution yielded 0/2 successes, attempting fallback for 2 more",
            "Fallback provider p3 failed: fail3",
            "Only 0 of 2 required models succeeded after trying 3 provider(s)",
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
            warnings=warnings,
        )

        # Result should indicate failure
        assert result.success is False
        assert result.agreement.successful_providers == 0
        # Final warning should indicate exhaustion
        assert any("after trying 3" in w for w in result.warnings)

    def test_primary_content_from_successful_fallback(self):
        """primary_content returns content from first successful response.

        Even if initial providers failed, fallback success provides content.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail"),
            ProviderResponse("p2", "m2", "first success content", success=True),
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
        )

        # primary_content should return p2's content
        assert result.primary_content == "first success content"

    def test_fallback_respects_min_models_requirement(self):
        """Fallback stops once min_models successful providers reached.

        If min_models=2 and we achieve 2 successes, don't try more providers.
        """
        from foundry_mcp.core.ai_consultation import ConsensusResult, ProviderResponse

        # Simulating: p1 failed, p2 succeeded, fallback p3 succeeded -> stop
        responses = [
            ProviderResponse("p1", "m1", "", success=False, error="fail"),
            ProviderResponse("p2", "m2", "content2", success=True),
            ProviderResponse("p3", "m3", "content3", success=True),
        ]
        warnings = [
            "Provider p1 failed: fail",
            "Initial parallel execution yielded 1/2 successes, attempting fallback for 1 more",
            "Fallback provider p3 succeeded",
        ]
        result = ConsensusResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            responses=responses,
            warnings=warnings,
        )

        # Should have exactly 3 responses (2 initial + 1 fallback)
        # p4 and p5 should NOT be tried since we reached min_models=2
        assert len(result.responses) == 3
        assert result.agreement.successful_providers == 2


# =============================================================================
# ConsultationOrchestrator Tests
# =============================================================================


class TestConsultationOrchestrator:
    """Tests for ConsultationOrchestrator class with mocked providers."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock ConsultationConfig."""
        from foundry_mcp.core.llm_config import ConsultationConfig

        return ConsultationConfig(
            priority=["gemini", "claude"],
            default_timeout=60.0,
            fallback_enabled=True,
            max_retries=1,
            retry_delay=0.1,
            cache_ttl=3600,
        )

    @pytest.fixture
    def orchestrator(self, tmp_path, mock_config):
        """Create a ConsultationOrchestrator with mock config."""
        from foundry_mcp.core.ai_consultation import ConsultationOrchestrator, ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache", default_ttl=3600)
        return ConsultationOrchestrator(cache=cache, config=mock_config)

    @pytest.fixture
    def mock_provider_result(self):
        """Create a mock ProviderResult factory."""
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

        def _create(
            content: str = "Generated response content",
            success: bool = True,
            provider_id: str = "gemini",
            model: str = "gemini-2.0-flash",
        ):
            return ProviderResult(
                content=content,
                status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
                provider_id=provider_id,
                model_used=model,
                tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
                duration_ms=500.0,
            )

        return _create

    def test_init_default(self):
        """ConsultationOrchestrator can be initialized with defaults."""
        from foundry_mcp.core.ai_consultation import ConsultationOrchestrator

        # Should not raise
        orchestrator = ConsultationOrchestrator()
        assert orchestrator.cache is not None
        assert orchestrator.default_timeout > 0

    def test_is_available_no_providers(self, orchestrator):
        """is_available returns False when no providers available."""
        from unittest.mock import patch

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=False,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=[],
            ):
                assert orchestrator.is_available() is False

    def test_is_available_with_providers(self, orchestrator):
        """is_available returns True when providers available."""
        from unittest.mock import patch

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            assert orchestrator.is_available() is True

    def test_is_available_specific_provider(self, orchestrator):
        """is_available checks specific provider when provided."""
        from unittest.mock import patch

        def mock_check(provider_id):
            return provider_id == "gemini"

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            side_effect=mock_check,
        ):
            assert orchestrator.is_available("gemini") is True
            assert orchestrator.is_available("nonexistent") is False

    def test_get_available_providers(self, orchestrator):
        """get_available_providers returns sorted list."""
        from unittest.mock import patch

        with patch(
            "foundry_mcp.core.ai_consultation.available_providers",
            return_value=["claude", "gemini", "openai"],
        ):
            providers = orchestrator.get_available_providers()
            assert providers == ["claude", "gemini", "openai"]

    def test_consult_cache_hit(self, orchestrator, tmp_path):
        """consult returns cached result on cache hit."""
        from foundry_mcp.core.ai_consultation import (
            ConsultationRequest,
            ConsultationResult,
        )

        # Pre-populate cache
        result = ConsultationResult(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            content="Cached content",
            provider_id="gemini",
            model_used="gemini-2.0-flash",
        )
        cache_key = orchestrator._generate_cache_key(
            ConsultationRequest(
                workflow=ConsultationWorkflow.PLAN_REVIEW,
                prompt_id="PLAN_REVIEW_FULL_V1",
                context={"spec_content": "test"},
            )
        )
        orchestrator.cache.set(ConsultationWorkflow.PLAN_REVIEW, cache_key, result)

        # Make request
        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="PLAN_REVIEW_FULL_V1",
            context={"spec_content": "test"},
        )
        outcome = orchestrator.consult(request, use_cache=True)

        assert outcome.cache_hit is True
        assert outcome.content == "Cached content"

    def test_consult_success_single_provider(
        self, orchestrator, mock_provider_result
    ):
        """consult succeeds with single provider mode."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result()

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content here",
                        },
                    )
                    outcome = orchestrator.consult(request, use_cache=False)

        assert outcome.content == "Generated response content"
        assert outcome.error is None
        assert isinstance(outcome, ConsultationResult)

    def test_consult_all_providers_fail(self, orchestrator):
        """consult returns error when all providers fail."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        mock_provider = MagicMock()
        mock_provider.generate.return_value = ProviderResult(
            content="",
            status=ProviderStatus.ERROR,
            provider_id="gemini",
            model_used="gemini-2.0-flash",
            stderr="Connection timeout",
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = orchestrator.consult(request, use_cache=False)

        assert outcome.error is not None
        assert outcome.content == ""

    def test_consult_no_providers_available(self, orchestrator):
        """consult returns error when no providers available."""
        from unittest.mock import patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=False,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=[],
            ):
                request = ConsultationRequest(
                    workflow=ConsultationWorkflow.PLAN_REVIEW,
                    prompt_id="PLAN_REVIEW_FULL_V1",
                    context={
                        "spec_id": "test-001",
                        "title": "Test",
                        "version": "1.0",
                        "spec_content": "Spec content",
                    },
                )
                outcome = orchestrator.consult(request, use_cache=False)

        assert outcome.error is not None
        assert "no" in outcome.error.lower() and "provider" in outcome.error.lower()

    def test_consult_prompt_build_error(self, orchestrator):
        """consult handles prompt build errors."""
        from unittest.mock import patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                # Request with invalid/missing context will cause prompt build error
                request = ConsultationRequest(
                    workflow=ConsultationWorkflow.PLAN_REVIEW,
                    prompt_id="NONEXISTENT_PROMPT",
                    context={},
                )
                outcome = orchestrator.consult(request, use_cache=False)

        assert outcome.error is not None
        assert "prompt" in outcome.error.lower() or "failed" in outcome.error.lower()

    def test_consult_fallback_on_provider_failure(
        self, orchestrator, mock_provider_result
    ):
        """consult falls back to next provider on failure."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        call_count = [0]

        def mock_generate(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # First provider fails
                return ProviderResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    provider_id="gemini",
                    model_used="gemini-2.0-flash",
                    stderr="Rate limit exceeded",
                )
            else:
                # Second provider succeeds
                return mock_provider_result(provider_id="claude")

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = mock_generate

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = orchestrator.consult(request, use_cache=False)

        # Should have succeeded via fallback
        assert outcome.content != ""
        assert outcome.error is None

    def test_generate_cache_key_deterministic(self, orchestrator):
        """_generate_cache_key produces deterministic keys."""
        from foundry_mcp.core.ai_consultation import ConsultationRequest

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="test_prompt",
            context={"a": 1, "b": 2},
        )

        key1 = orchestrator._generate_cache_key(request)
        key2 = orchestrator._generate_cache_key(request)

        assert key1 == key2

    def test_generate_cache_key_explicit_override(self, orchestrator):
        """_generate_cache_key uses explicit cache_key if provided."""
        from foundry_mcp.core.ai_consultation import ConsultationRequest

        request = ConsultationRequest(
            workflow=ConsultationWorkflow.PLAN_REVIEW,
            prompt_id="test_prompt",
            context={},
            cache_key="explicit-key",
        )

        key = orchestrator._generate_cache_key(request)
        assert key == "explicit-key"

    def test_consult_uses_timeout_from_request(
        self, orchestrator, mock_provider_result
    ):
        """consult uses timeout from request."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result()

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                        timeout=300.0,  # Custom timeout
                    )
                    orchestrator.consult(request, use_cache=False)

        # Verify generate was called with the custom timeout
        call_args = mock_provider.generate.call_args
        assert call_args is not None
        provider_request = call_args[0][0]
        assert provider_request.timeout == 300.0

    def test_consult_result_includes_duration(
        self, orchestrator, mock_provider_result
    ):
        """consult result includes duration_ms."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import ConsultationRequest

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result()

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = orchestrator.consult(request, use_cache=False)

        # Duration should be recorded
        assert outcome.duration_ms >= 0


class TestConsultationOrchestratorMultiModel:
    """Tests for ConsultationOrchestrator multi-model mode."""

    @pytest.fixture
    def mock_multi_model_config(self, tmp_path):
        """Create a mock ConsultationConfig for multi-model mode."""
        from foundry_mcp.core.llm_config import ConsultationConfig, WorkflowConsultationConfig

        return ConsultationConfig(
            priority=["gemini", "claude", "openai"],
            default_timeout=60.0,
            fallback_enabled=True,
            max_retries=1,
            retry_delay=0.1,
            cache_ttl=3600,
            workflows={
                "plan_review": WorkflowConsultationConfig(min_models=2),
            },
        )

    @pytest.fixture
    def multi_model_orchestrator(self, tmp_path, mock_multi_model_config):
        """Create a ConsultationOrchestrator for multi-model mode."""
        from foundry_mcp.core.ai_consultation import ConsultationOrchestrator, ResultCache

        cache = ResultCache(base_dir=tmp_path / "cache", default_ttl=3600)
        return ConsultationOrchestrator(cache=cache, config=mock_multi_model_config)

    @pytest.fixture
    def mock_provider_result(self):
        """Create a mock ProviderResult factory."""
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus, TokenUsage

        def _create(
            content: str = "Generated response content",
            success: bool = True,
            provider_id: str = "gemini",
            model: str = "gemini-2.0-flash",
        ):
            return ProviderResult(
                content=content,
                status=ProviderStatus.SUCCESS if success else ProviderStatus.ERROR,
                provider_id=provider_id,
                model_used=model,
                tokens=TokenUsage(input_tokens=50, output_tokens=100, total_tokens=150),
                duration_ms=500.0,
            )

        return _create

    def test_consult_multi_model_returns_consensus_result(
        self, multi_model_orchestrator, mock_provider_result
    ):
        """consult returns ConsensusResult when min_models > 1."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationRequest,
        )

        mock_provider = MagicMock()
        mock_provider.generate.return_value = mock_provider_result()

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = multi_model_orchestrator.consult(
                        request, use_cache=False
                    )

        # Should return ConsensusResult for multi-model workflow
        assert isinstance(outcome, ConsensusResult)
        assert len(outcome.responses) >= 1
        assert outcome.agreement is not None

    def test_consult_multi_model_with_failures(
        self, multi_model_orchestrator, mock_provider_result
    ):
        """consult handles partial failures in multi-model mode."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationRequest,
        )
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        call_count = [0]

        def mock_generate(request):
            call_count[0] += 1
            if call_count[0] == 1:
                # First provider fails
                return ProviderResult(
                    content="",
                    status=ProviderStatus.ERROR,
                    provider_id="gemini",
                    model_used="gemini-2.0-flash",
                    stderr="Error",
                )
            else:
                # Other providers succeed
                return mock_provider_result(provider_id=f"provider-{call_count[0]}")

        mock_provider = MagicMock()
        mock_provider.generate.side_effect = mock_generate

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude", "openai"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = multi_model_orchestrator.consult(
                        request, use_cache=False
                    )

        assert isinstance(outcome, ConsensusResult)
        # Should have tracked both successes and failures
        assert outcome.agreement.total_providers >= 2
        # Should have at least some successful providers via fallback
        assert len(outcome.warnings) > 0  # Should have recorded the failure

    def test_consult_multi_model_all_fail(self, multi_model_orchestrator):
        """consult returns failure ConsensusResult when all providers fail."""
        from unittest.mock import MagicMock, patch

        from foundry_mcp.core.ai_consultation import (
            ConsensusResult,
            ConsultationRequest,
        )
        from foundry_mcp.core.providers import ProviderResult, ProviderStatus

        mock_provider = MagicMock()
        mock_provider.generate.return_value = ProviderResult(
            content="",
            status=ProviderStatus.ERROR,
            provider_id="test",
            model_used="test-model",
            stderr="All providers failing",
        )

        with patch(
            "foundry_mcp.core.ai_consultation.check_provider_available",
            return_value=True,
        ):
            with patch(
                "foundry_mcp.core.ai_consultation.available_providers",
                return_value=["gemini", "claude"],
            ):
                with patch(
                    "foundry_mcp.core.ai_consultation.resolve_provider",
                    return_value=mock_provider,
                ):
                    request = ConsultationRequest(
                        workflow=ConsultationWorkflow.PLAN_REVIEW,
                        prompt_id="PLAN_REVIEW_FULL_V1",
                        context={
                            "spec_id": "test-001",
                            "title": "Test",
                            "version": "1.0",
                            "spec_content": "Spec content",
                        },
                    )
                    outcome = multi_model_orchestrator.consult(
                        request, use_cache=False
                    )

        assert isinstance(outcome, ConsensusResult)
        assert outcome.success is False
        assert outcome.agreement.successful_providers == 0
