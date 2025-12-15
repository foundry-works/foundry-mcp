"""Action routing utilities for unified MCP tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Dict, cast


class ActionRouterError(ValueError):
    """Raised when an unsupported action is requested."""

    def __init__(self, message: str, *, allowed_actions: Sequence[str]) -> None:
        super().__init__(message)
        self.allowed_actions = tuple(allowed_actions)


@dataclass(frozen=True)
class ActionDefinition:
    """Describe an action handler for a unified tool."""

    name: str
    handler: Callable[..., dict]
    summary: str | None = None
    aliases: Sequence[str] = ()


class ActionRouter:
    """Route action requests to the correct handler."""

    def __init__(
        self,
        *,
        tool_name: str,
        actions: Iterable[ActionDefinition] | Mapping[str, Callable[..., dict]],
        case_sensitive: bool = False,
    ) -> None:
        if isinstance(actions, Mapping):
            mapping_actions = cast(Mapping[str, Callable[..., dict]], actions)
            normalized_actions = [
                ActionDefinition(name=name, handler=handler)
                for name, handler in mapping_actions.items()
            ]
        else:
            iterable_actions = cast(Iterable[ActionDefinition], actions)
            normalized_actions = list(iterable_actions)

        if not normalized_actions:
            raise ValueError("ActionRouter requires at least one action")

        self._tool_name = tool_name
        self._case_sensitive = case_sensitive
        self._handlers: Dict[str, Callable[..., dict]] = {}
        self._canonical: Dict[str, str] = {}
        self._summaries: Dict[str, str | None] = {}

        for action_def in normalized_actions:
            canonical_name = action_def.name
            if not canonical_name:
                raise ValueError("Action names must be non-empty strings")

            names = [canonical_name, *(action_def.aliases or ())]
            for name in names:
                if not name:
                    continue
                key = name if case_sensitive else name.lower()
                if key in self._handlers:
                    raise ValueError(
                        f"Duplicate action alias '{name}' for tool '{tool_name}'"
                    )
                self._handlers[key] = action_def.handler
                self._canonical[key] = canonical_name
                self._summaries[canonical_name] = action_def.summary

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def allowed_actions(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._canonical.values())))

    def dispatch(self, action: str | None, **kwargs) -> dict:
        if not action:
            raise ActionRouterError(
                f"{self._tool_name} requires an action",
                allowed_actions=self.allowed_actions(),
            )

        key = action if self._case_sensitive else action.lower()
        handler = self._handlers.get(key)
        if handler is None:
            raise ActionRouterError(
                f"Unsupported action '{action}' for {self._tool_name}",
                allowed_actions=self.allowed_actions(),
            )

        return handler(**kwargs)

    def describe(self) -> Dict[str, str | None]:
        """Return summaries for the canonical actions."""

        return dict(self._summaries)
