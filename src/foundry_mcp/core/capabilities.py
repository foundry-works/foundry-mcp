"""
MCP Capabilities module for foundry-mcp.

Provides support for MCP Notifications and Sampling features.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Schema version for capability responses
SCHEMA_VERSION = "1.0.0"


# Notification Types


@dataclass
class Notification:
    """
    MCP notification to be sent to clients.
    """

    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class ResourceUpdate:
    """
    Notification for resource updates.
    """

    uri: str
    update_type: str  # created, updated, deleted
    metadata: Dict[str, Any] = field(default_factory=dict)


# Notification Manager


class NotificationManager:
    """
    Manages MCP notifications for resource updates and other events.

    Supports registering handlers and emitting notifications when
    specs or other resources change.
    """

    def __init__(self):
        """Initialize notification manager."""
        self._handlers: List[Callable[[Notification], None]] = []
        self._pending: List[Notification] = []
        self._enabled = True

    def register_handler(self, handler: Callable[[Notification], None]) -> None:
        """
        Register a notification handler.

        Args:
            handler: Callback function to receive notifications
        """
        self._handlers.append(handler)
        logger.debug(f"Registered notification handler: {handler}")

    def unregister_handler(self, handler: Callable[[Notification], None]) -> None:
        """
        Unregister a notification handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._handlers:
            self._handlers.remove(handler)
            logger.debug(f"Unregistered notification handler: {handler}")

    def emit(self, notification: Notification) -> None:
        """
        Emit a notification to all registered handlers.

        Args:
            notification: Notification to emit
        """
        if not self._enabled:
            self._pending.append(notification)
            return

        for handler in self._handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")

    def emit_resource_update(self, uri: str, update_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit a resource update notification.

        Args:
            uri: Resource URI that was updated
            update_type: Type of update (created, updated, deleted)
            metadata: Optional additional metadata
        """
        notification = Notification(
            method="notifications/resources/updated",
            params={
                "uri": uri,
                "type": update_type,
                "metadata": metadata or {},
            },
        )
        self.emit(notification)

    def emit_spec_updated(
        self, spec_id: str, update_type: str = "updated", changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit a spec update notification.

        Args:
            spec_id: Spec ID that was updated
            update_type: Type of update (created, updated, deleted, status_changed)
            changes: Optional dict describing what changed
        """
        notification = Notification(
            method="foundry/specs/updated",
            params={
                "spec_id": spec_id,
                "type": update_type,
                "changes": changes or {},
            },
        )
        self.emit(notification)

    def emit_task_updated(
        self, spec_id: str, task_id: str, update_type: str = "updated", changes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Emit a task update notification.

        Args:
            spec_id: Spec ID containing the task
            task_id: Task ID that was updated
            update_type: Type of update (status_changed, journal_added, etc.)
            changes: Optional dict describing what changed
        """
        notification = Notification(
            method="foundry/tasks/updated",
            params={
                "spec_id": spec_id,
                "task_id": task_id,
                "type": update_type,
                "changes": changes or {},
            },
        )
        self.emit(notification)

    def pause(self) -> None:
        """Pause notification delivery (queue pending)."""
        self._enabled = False

    def resume(self) -> None:
        """Resume notification delivery and flush pending."""
        self._enabled = True
        pending = self._pending[:]
        self._pending.clear()
        for notification in pending:
            self.emit(notification)

    def clear_pending(self) -> None:
        """Clear pending notifications without sending."""
        self._pending.clear()


# Sampling Support


@dataclass
class SamplingRequest:
    """
    Request for server-side AI sampling.
    """

    messages: List[Dict[str, Any]]
    model_preferences: Optional[Dict[str, Any]] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SamplingResponse:
    """
    Response from server-side AI sampling.
    """

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SamplingManager:
    """
    Manages MCP sampling requests for server-side AI operations.

    Used for features like impact analysis that benefit from AI assistance.
    """

    def __init__(self):
        """Initialize sampling manager."""
        self._handler: Optional[Callable[[SamplingRequest], SamplingResponse]] = None
        self._enabled = False
        self._request_count = 0
        self._total_tokens = 0

    def set_handler(self, handler: Callable[[SamplingRequest], SamplingResponse]) -> None:
        """
        Set the sampling handler (typically provided by MCP client).

        Args:
            handler: Function to handle sampling requests
        """
        self._handler = handler
        self._enabled = True
        logger.info("Sampling handler registered")

    def is_available(self) -> bool:
        """Check if sampling is available."""
        return self._enabled and self._handler is not None

    def request(self, request: SamplingRequest) -> Optional[SamplingResponse]:
        """
        Make a sampling request.

        Args:
            request: Sampling request to process

        Returns:
            SamplingResponse if successful, None otherwise
        """
        if not self.is_available():
            logger.warning("Sampling not available")
            return None

        try:
            assert self._handler is not None
            response = self._handler(request)
            self._request_count += 1
            self._total_tokens += response.usage.get("total_tokens", 0)
            return response
        except Exception as e:
            logger.error(f"Sampling request failed: {e}")
            return None

    def analyze_impact(self, target: str, target_type: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Use sampling to analyze impact of a change.

        Args:
            target: Name of class/function being changed
            target_type: Type of target (class, function, module)
            context: Optional additional context about the change

        Returns:
            Analysis results or None if sampling unavailable
        """
        if not self.is_available():
            return None

        system_prompt = """You are analyzing code change impacts.
Given a target entity and its type, identify:
1. Direct impacts (what directly depends on this)
2. Indirect impacts (what might be affected transitively)
3. Risk level (low, medium, high)
4. Recommended actions before making changes

Respond in JSON format with keys: direct_impacts, indirect_impacts, risk_level, recommendations"""

        messages = [
            {
                "role": "user",
                "content": f"Analyze the impact of changing {target_type} '{target}'."
                + (f"\n\nAdditional context: {context}" if context else ""),
            }
        ]

        request = SamplingRequest(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=500,
            temperature=0.3,
            metadata={"operation": "impact_analysis", "target": target},
        )

        response = self.request(request)
        if response:
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {"raw_response": response.content}

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get sampling usage statistics."""
        return {
            "enabled": self._enabled,
            "available": self.is_available(),
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
        }


# Capabilities Registry


class CapabilitiesRegistry:
    """
    Central registry for server capabilities.

    Tracks what features are available and their configuration.
    """

    def __init__(self):
        """Initialize capabilities registry."""
        self.notifications = NotificationManager()
        self.sampling = SamplingManager()
        self._capabilities: Dict[str, bool] = {
            "notifications": True,
            "sampling": False,  # Requires handler
            "resources": True,
            "prompts": True,
            "tools": True,
        }
        self._metadata: Dict[str, Any] = {}

    def enable(self, capability: str) -> None:
        """Enable a capability."""
        self._capabilities[capability] = True

    def disable(self, capability: str) -> None:
        """Disable a capability."""
        self._capabilities[capability] = False

    def is_enabled(self, capability: str) -> bool:
        """Check if a capability is enabled."""
        return self._capabilities.get(capability, False)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set capability metadata."""
        self._metadata[key] = value

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get all capabilities and their status.

        Returns:
            Dict with capability information
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "capabilities": self._capabilities.copy(),
            "notifications": {
                "enabled": self._capabilities.get("notifications", False),
                "methods": [
                    "notifications/resources/updated",
                    "foundry/specs/updated",
                    "foundry/tasks/updated",
                ],
            },
            "sampling": self.sampling.get_stats(),
            "metadata": self._metadata,
        }

    def load_manifest(self, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Load capabilities from manifest file.

        Args:
            manifest_path: Path to capabilities manifest

        Returns:
            Loaded manifest data
        """
        if manifest_path is None:
            # Try default locations
            search_paths = [
                Path.cwd() / "mcp" / "capabilities_manifest.json",
                Path(__file__).parent.parent.parent.parent / "mcp" / "capabilities_manifest.json",
            ]
            for path in search_paths:
                if path.exists():
                    manifest_path = path
                    break

        if manifest_path and manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load manifest: {e}")

        return {}


# Global instance
_registry: Optional[CapabilitiesRegistry] = None


def get_capabilities_registry() -> CapabilitiesRegistry:
    """Get the global capabilities registry."""
    global _registry
    if _registry is None:
        _registry = CapabilitiesRegistry()
    return _registry


def get_notification_manager() -> NotificationManager:
    """Get the notification manager from global registry."""
    return get_capabilities_registry().notifications


def get_sampling_manager() -> SamplingManager:
    """Get the sampling manager from global registry."""
    return get_capabilities_registry().sampling
