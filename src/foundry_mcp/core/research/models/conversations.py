"""Conversation and thread models for the CHAT workflow."""

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from foundry_mcp.core.research.models.enums import ThreadStatus


class ConversationMessage(BaseModel):
    """A single message in a conversation thread."""

    id: str = Field(default_factory=lambda: f"msg-{uuid4().hex[:8]}")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(
        default=None, description="Provider that generated this message"
    )
    model_used: Optional[str] = Field(
        default=None, description="Model that generated this message"
    )
    tokens_used: Optional[int] = Field(
        default=None, description="Tokens consumed for this message"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )


class ConversationThread(BaseModel):
    """A conversation thread with message history."""

    id: str = Field(default_factory=lambda: f"thread-{uuid4().hex[:12]}")
    title: Optional[str] = Field(default=None, description="Optional thread title")
    status: ThreadStatus = Field(default=ThreadStatus.ACTIVE)
    messages: list[ConversationMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    provider_id: Optional[str] = Field(
        default=None, description="Default provider for this thread"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for this thread"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional thread metadata"
    )

    def add_message(
        self,
        role: str,
        content: str,
        provider_id: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        **metadata: Any,
    ) -> ConversationMessage:
        """Add a message to the thread and update timestamp."""
        message = ConversationMessage(
            role=role,
            content=content,
            provider_id=provider_id,
            model_used=model_used,
            tokens_used=tokens_used,
            metadata=metadata,
        )
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message

    def get_context_messages(
        self, max_messages: Optional[int] = None
    ) -> list[ConversationMessage]:
        """Get messages for context, optionally limited to recent N messages."""
        if max_messages is None or max_messages >= len(self.messages):
            return self.messages
        return self.messages[-max_messages:]
