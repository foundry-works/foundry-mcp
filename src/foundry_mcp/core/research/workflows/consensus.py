"""CONSENSUS workflow for multi-model parallel consultation with synthesis.

Provides parallel execution across multiple providers with configurable
synthesis strategies for combining responses.
"""

import asyncio
import logging
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.providers import ProviderHooks, ProviderRequest, ProviderStatus
from foundry_mcp.core.providers.registry import available_providers, resolve_provider
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    ConsensusConfig,
    ConsensusState,
    ConsensusStrategy,
    ModelResponse,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


class ConsensusWorkflow(ResearchWorkflowBase):
    """Multi-model consensus workflow with synthesis strategies.

    Features:
    - Parallel execution across multiple providers
    - Concurrency limiting with semaphore
    - Multiple synthesis strategies (all_responses, synthesize, majority, first_valid)
    - Partial failure handling (continue on some provider errors)
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
    ) -> None:
        """Initialize consensus workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance
        """
        super().__init__(config, memory)

    def execute(
        self,
        prompt: str,
        providers: Optional[list[str]] = None,
        strategy: ConsensusStrategy = ConsensusStrategy.SYNTHESIZE,
        synthesis_provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        timeout_per_provider: float = 30.0,
        max_concurrent: int = 3,
        require_all: bool = False,
        min_responses: int = 1,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute consensus across multiple providers.

        Args:
            prompt: User prompt to send to all providers
            providers: List of provider IDs (uses config default if None)
            strategy: Synthesis strategy for combining responses
            synthesis_provider: Provider for synthesis (if strategy=synthesize)
            system_prompt: Optional system prompt
            timeout_per_provider: Timeout per provider in seconds
            max_concurrent: Maximum concurrent provider calls
            require_all: Require all providers to succeed
            min_responses: Minimum responses needed for success

        Returns:
            WorkflowResult with synthesized or combined response
        """
        # Resolve providers
        provider_ids = providers or self.config.consensus_providers
        available = available_providers()
        valid_providers = [p for p in provider_ids if p in available]

        if not valid_providers:
            return WorkflowResult(
                success=False,
                content="",
                error=f"No valid providers available. Requested: {provider_ids}, Available: {available}",
            )

        # Create consensus config and state
        consensus_config = ConsensusConfig(
            providers=valid_providers,
            strategy=strategy,
            synthesis_provider=synthesis_provider or self.config.default_provider,
            timeout_per_provider=timeout_per_provider,
            max_concurrent=max_concurrent,
            require_all=require_all,
            min_responses=min_responses,
        )

        state = ConsensusState(
            prompt=prompt,
            config=consensus_config,
            system_prompt=system_prompt,
        )

        # Execute parallel requests
        try:
            responses = asyncio.run(
                self._execute_parallel(
                    prompt=prompt,
                    providers=valid_providers,
                    system_prompt=system_prompt,
                    timeout=timeout_per_provider,
                    max_concurrent=max_concurrent,
                )
            )
        except Exception as exc:
            logger.error("Parallel execution failed: %s", exc)
            return WorkflowResult(
                success=False,
                content="",
                error=f"Parallel execution failed: {exc}",
            )

        # Add responses to state
        for response in responses:
            state.add_response(response)

        # Check if we have enough responses
        successful = state.successful_responses()
        if len(successful) < min_responses:
            failed_info = [
                f"{r.provider_id}: {r.error_message}"
                for r in state.failed_responses()
            ]
            return WorkflowResult(
                success=False,
                content="",
                error=f"Insufficient responses ({len(successful)}/{min_responses}). Failures: {failed_info}",
                metadata={
                    "successful_count": len(successful),
                    "failed_count": len(state.failed_responses()),
                    "responses": [r.model_dump() for r in responses],
                },
            )

        if require_all and len(state.failed_responses()) > 0:
            return WorkflowResult(
                success=False,
                content="",
                error=f"Not all providers succeeded (require_all=True). Failed: {[r.provider_id for r in state.failed_responses()]}",
            )

        # Apply synthesis strategy
        result = self._apply_strategy(state)

        # Persist state
        state.mark_completed(synthesis=result.content if result.success else None)
        self.memory.save_consensus(state)

        # Add consensus metadata
        result.metadata["consensus_id"] = state.id
        result.metadata["providers_consulted"] = [r.provider_id for r in successful]
        result.metadata["strategy"] = strategy.value
        result.metadata["response_count"] = len(successful)

        return result

    async def _execute_parallel(
        self,
        prompt: str,
        providers: list[str],
        system_prompt: Optional[str],
        timeout: float,
        max_concurrent: int,
    ) -> list[ModelResponse]:
        """Execute requests to multiple providers in parallel.

        Args:
            prompt: User prompt
            providers: Provider IDs to query
            system_prompt: Optional system prompt
            timeout: Timeout per provider
            max_concurrent: Max concurrent requests

        Returns:
            List of ModelResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def query_provider(provider_id: str) -> ModelResponse:
            async with semaphore:
                return await self._query_single_provider(
                    provider_id=provider_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    timeout=timeout,
                )

        tasks = [query_provider(pid) for pid in providers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed responses
        result = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                result.append(
                    ModelResponse(
                        provider_id=providers[i],
                        content="",
                        success=False,
                        error_message=str(response),
                    )
                )
            else:
                result.append(response)

        return result

    async def _query_single_provider(
        self,
        provider_id: str,
        prompt: str,
        system_prompt: Optional[str],
        timeout: float,
    ) -> ModelResponse:
        """Query a single provider asynchronously.

        Args:
            provider_id: Provider to query
            prompt: User prompt
            system_prompt: Optional system prompt
            timeout: Request timeout

        Returns:
            ModelResponse with result or error
        """
        import time

        start_time = time.perf_counter()

        try:
            provider = resolve_provider(provider_id, hooks=ProviderHooks())
            request = ProviderRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                timeout=timeout,
            )

            # Run synchronous generate in thread pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, provider.generate, request),
                timeout=timeout,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.status != ProviderStatus.SUCCESS:
                return ModelResponse(
                    provider_id=provider_id,
                    model_used=result.model_used,
                    content=result.content or "",
                    success=False,
                    error_message=f"Provider returned status: {result.status.value}",
                    duration_ms=duration_ms,
                )

            return ModelResponse(
                provider_id=provider_id,
                model_used=result.model_used,
                content=result.content,
                success=True,
                tokens_used=result.tokens.total_tokens if result.tokens else None,
                duration_ms=duration_ms,
            )

        except asyncio.TimeoutError:
            return ModelResponse(
                provider_id=provider_id,
                content="",
                success=False,
                error_message=f"Timeout after {timeout}s",
                duration_ms=timeout * 1000,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return ModelResponse(
                provider_id=provider_id,
                content="",
                success=False,
                error_message=str(exc),
                duration_ms=duration_ms,
            )

    def _apply_strategy(self, state: ConsensusState) -> WorkflowResult:
        """Apply synthesis strategy to responses.

        Args:
            state: ConsensusState with collected responses

        Returns:
            WorkflowResult with synthesized content
        """
        successful = state.successful_responses()
        strategy = state.config.strategy

        if strategy == ConsensusStrategy.ALL_RESPONSES:
            # Return all responses without synthesis
            content_parts = []
            for resp in successful:
                content_parts.append(f"### {resp.provider_id}\n\n{resp.content}")
            return WorkflowResult(
                success=True,
                content="\n\n---\n\n".join(content_parts),
                metadata={"strategy": "all_responses"},
            )

        elif strategy == ConsensusStrategy.FIRST_VALID:
            # Return first successful response
            first = successful[0]
            return WorkflowResult(
                success=True,
                content=first.content,
                provider_id=first.provider_id,
                model_used=first.model_used,
                tokens_used=first.tokens_used,
                metadata={"strategy": "first_valid"},
            )

        elif strategy == ConsensusStrategy.MAJORITY:
            # For factual questions, try to find majority agreement
            # Simple heuristic: if responses are similar, use first; otherwise synthesize
            # A more sophisticated implementation would compare semantic similarity
            return self._synthesize_responses(state, successful)

        elif strategy == ConsensusStrategy.SYNTHESIZE:
            # Use a model to synthesize all responses
            return self._synthesize_responses(state, successful)

        else:
            # Default to first valid
            first = successful[0]
            return WorkflowResult(
                success=True,
                content=first.content,
                provider_id=first.provider_id,
            )

    def _synthesize_responses(
        self,
        state: ConsensusState,
        responses: list[ModelResponse],
    ) -> WorkflowResult:
        """Synthesize multiple responses using a model.

        Args:
            state: ConsensusState with original prompt
            responses: Successful responses to synthesize

        Returns:
            WorkflowResult with synthesized content
        """
        # Build synthesis prompt
        response_text = "\n\n---\n\n".join(
            f"Response from {r.provider_id}:\n{r.content}"
            for r in responses
        )

        synthesis_prompt = f"""You are synthesizing multiple AI responses to the same question.

Original question: {state.prompt}

{response_text}

Please synthesize these responses into a single, comprehensive answer that:
1. Captures the key points from all responses
2. Resolves any contradictions by noting different perspectives
3. Provides a clear, well-structured response

Synthesized response:"""

        # Execute synthesis
        result = self._execute_provider(
            prompt=synthesis_prompt,
            provider_id=state.config.synthesis_provider,
            system_prompt="You are a helpful assistant that synthesizes multiple AI responses into a coherent, comprehensive answer.",
        )

        if result.success:
            result.metadata["strategy"] = "synthesize"
            result.metadata["synthesis_provider"] = state.config.synthesis_provider
            result.metadata["source_providers"] = [r.provider_id for r in responses]

        return result
