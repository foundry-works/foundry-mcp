"""CHAT workflow for single-model conversation with thread persistence.

Provides conversational interaction with context preservation across messages,
supporting thread creation, continuation, and message history management.
"""

import logging
from typing import Any, Optional

from foundry_mcp.config import ResearchConfig
from foundry_mcp.core.research.memory import ResearchMemory
from foundry_mcp.core.research.models import (
    ConversationMessage,
    ConversationThread,
    ThreadStatus,
)
from foundry_mcp.core.research.workflows.base import ResearchWorkflowBase, WorkflowResult

logger = logging.getLogger(__name__)


class ChatWorkflow(ResearchWorkflowBase):
    """Single-model conversation workflow with thread persistence.

    Features:
    - Create new conversation threads
    - Continue existing threads with full context
    - Token-aware context window management
    - Message persistence across invocations
    """

    def __init__(
        self,
        config: ResearchConfig,
        memory: Optional[ResearchMemory] = None,
    ) -> None:
        """Initialize chat workflow.

        Args:
            config: Research configuration
            memory: Optional memory instance
        """
        super().__init__(config, memory)

    def execute(
        self,
        prompt: str,
        thread_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Execute a chat turn.

        Creates a new thread or continues an existing one, sends the prompt
        to the provider, and persists the conversation.

        Args:
            prompt: User message
            thread_id: Existing thread to continue (creates new if None)
            system_prompt: System prompt (only used for new threads)
            provider_id: Provider to use (uses config default if None)
            model: Optional model override
            temperature: Optional temperature setting
            max_tokens: Optional max tokens
            title: Optional title for new threads

        Returns:
            WorkflowResult with assistant response and thread metadata
        """
        # Get or create thread
        thread = self._get_or_create_thread(
            thread_id=thread_id,
            system_prompt=system_prompt,
            provider_id=provider_id,
            title=title,
        )

        # Add user message
        thread.add_message(role="user", content=prompt)

        # Build context for provider
        context = self._build_context(thread)

        # Execute provider
        result = self._execute_provider(
            prompt=context,
            provider_id=thread.provider_id or provider_id,
            system_prompt=thread.system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if result.success:
            # Add assistant message
            thread.add_message(
                role="assistant",
                content=result.content,
                provider_id=result.provider_id,
                model_used=result.model_used,
                tokens_used=result.tokens_used,
            )

            # Persist thread
            self.memory.save_thread(thread)

            # Add thread info to result metadata
            result.metadata["thread_id"] = thread.id
            result.metadata["message_count"] = len(thread.messages)
            result.metadata["thread_title"] = thread.title

        return result

    def _get_or_create_thread(
        self,
        thread_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        provider_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> ConversationThread:
        """Get existing thread or create a new one.

        Args:
            thread_id: Existing thread ID to load
            system_prompt: System prompt for new threads
            provider_id: Provider ID for new threads
            title: Title for new threads

        Returns:
            ConversationThread instance
        """
        if thread_id:
            thread = self.memory.load_thread(thread_id)
            if thread:
                return thread
            logger.warning("Thread %s not found, creating new thread", thread_id)

        # Create new thread
        return ConversationThread(
            title=title,
            system_prompt=system_prompt,
            provider_id=provider_id or self.config.default_provider,
        )

    def _build_context(self, thread: ConversationThread) -> str:
        """Build conversation context for the provider.

        Formats message history with token-aware truncation to fit
        within context window limits.

        Args:
            thread: Conversation thread

        Returns:
            Formatted context string
        """
        # Get recent messages (respecting max_messages config)
        messages = thread.get_context_messages(
            max_messages=self.config.max_messages_per_thread
        )

        # Format messages for context
        parts = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"{role_label}: {msg.content}")

        return "\n\n".join(parts)

    def list_threads(
        self,
        status: Optional[ThreadStatus] = None,
        limit: Optional[int] = 50,
    ) -> list[dict[str, Any]]:
        """List conversation threads.

        Args:
            status: Filter by thread status
            limit: Maximum threads to return

        Returns:
            List of thread summaries
        """
        threads = self.memory.list_threads(status=status, limit=limit)

        return [
            {
                "id": t.id,
                "title": t.title,
                "status": t.status.value,
                "message_count": len(t.messages),
                "created_at": t.created_at.isoformat(),
                "updated_at": t.updated_at.isoformat(),
                "provider_id": t.provider_id,
            }
            for t in threads
        ]

    def get_thread(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Get full thread details including messages.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread data with messages or None if not found
        """
        thread = self.memory.load_thread(thread_id)
        if not thread:
            return None

        return {
            "id": thread.id,
            "title": thread.title,
            "status": thread.status.value,
            "system_prompt": thread.system_prompt,
            "provider_id": thread.provider_id,
            "created_at": thread.created_at.isoformat(),
            "updated_at": thread.updated_at.isoformat(),
            "messages": [
                {
                    "id": m.id,
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "provider_id": m.provider_id,
                    "model_used": m.model_used,
                    "tokens_used": m.tokens_used,
                }
                for m in thread.messages
            ],
            "metadata": thread.metadata,
        }

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread.

        Args:
            thread_id: Thread identifier

        Returns:
            True if deleted, False if not found
        """
        return self.memory.delete_thread(thread_id)

    def update_thread_status(
        self,
        thread_id: str,
        status: ThreadStatus,
    ) -> bool:
        """Update thread status.

        Args:
            thread_id: Thread identifier
            status: New status

        Returns:
            True if updated, False if not found
        """
        thread = self.memory.load_thread(thread_id)
        if not thread:
            return False

        thread.status = status
        self.memory.save_thread(thread)
        return True
