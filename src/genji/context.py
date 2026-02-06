"""Execution context management for template rendering."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .backends.base import GenerationRequest

# Thread-local context variable for the current render context
_render_context: ContextVar[RenderContext | None] = ContextVar(
    "genji_render_context", default=None
)


@dataclass
class CollectedPrompt:
    """A prompt collected during the collection phase."""

    placeholder: str
    prompt: str
    source_id: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stop: list[str] | None = None

    def to_request(self) -> GenerationRequest:
        """Convert to a GenerationRequest."""
        return GenerationRequest(
            prompt=self.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop,
        )


@dataclass
class RenderContext:
    """Context for a single render operation.

    This is thread-safe through the use of contextvars.
    Each render() call gets its own isolated context.
    """

    # Counter for generating unique placeholder IDs
    counter: int = 0

    # List of prompts collected during the first pass
    prompts: list[CollectedPrompt] = field(default_factory=list)

    # Mapping from placeholder to generated content
    generated: dict[str, str] = field(default_factory=dict)

    # Template variables
    variables: dict[str, Any] = field(default_factory=dict)

    def collect_prompt(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        source_id: int | None = None,
    ) -> str:
        """Collect a prompt for generation and return a placeholder.

        Args:
            prompt: The prompt text.
            max_tokens: Optional max tokens override.
            temperature: Optional temperature override.
            stop: Optional stop sequences.
            source_id: ID identifying the template source position.

        Returns:
            A unique placeholder string.
        """
        placeholder = f"__GENJI_GEN_{self.counter}__"
        self.counter += 1

        collected = CollectedPrompt(
            placeholder=placeholder,
            prompt=prompt,
            source_id=source_id,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        self.prompts.append(collected)

        return placeholder

    def set_generated(self, placeholder: str, content: str) -> None:
        """Store generated content for a placeholder.

        Args:
            placeholder: The placeholder string.
            content: The generated content.
        """
        self.generated[placeholder] = content

    def get_generated(self, placeholder: str) -> str:
        """Get generated content for a placeholder.

        Args:
            placeholder: The placeholder string.

        Returns:
            The generated content.

        Raises:
            KeyError: If placeholder not found.
        """
        return self.generated[placeholder]


def get_current_context() -> RenderContext:
    """Get the current render context.

    Returns:
        The current RenderContext.

    Raises:
        RuntimeError: If no context is active.
    """
    ctx = _render_context.get()
    if ctx is None:
        raise RuntimeError("No active render context")
    return ctx


def set_current_context(ctx: RenderContext | None) -> None:
    """Set the current render context.

    Args:
        ctx: The context to set, or None to clear.
    """
    _render_context.set(ctx)
