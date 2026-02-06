"""Template parsing and Jinja2 extension for gen() function."""

from __future__ import annotations

from typing import Any

import jinja2
from jinja2.ext import Extension

from .context import get_current_context
from .exceptions import TemplateParseError


class GenjiExtension(Extension):
    """Jinja2 extension that adds the gen() function for LLM generation."""

    tags = set()  # We don't add any custom tags

    def __init__(self, environment: jinja2.Environment) -> None:
        """Initialize the extension.

        Args:
            environment: The Jinja2 environment.
        """
        super().__init__(environment)

        # Add gen() as a global function
        environment.globals["gen"] = self._gen_function

    def _gen_function(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | None = None,
        __source_id: int | None = None,
        **kwargs: Any,
    ) -> str:
        """The gen() function that collects prompts during rendering.

        This function is called during template rendering. It collects
        the prompt and returns a placeholder that will be replaced with
        generated content later.

        Args:
            prompt: The prompt text (may contain format placeholders like {variable}).
            max_tokens: Optional max tokens override.
            temperature: Optional temperature override.
            stop: Optional stop sequences.
            __source_id: Internal ID identifying the template position
                (injected by template parser).
            **kwargs: Additional arguments (ignored, for compatibility).

        Returns:
            A placeholder string that will be replaced with generated content.
        """
        try:
            ctx = get_current_context()
        except RuntimeError as e:
            raise TemplateParseError(
                "gen() called outside of render context. "
                "This should not happen - please report this bug."
            ) from e

        # Interpolate variables in the prompt using the context variables
        try:
            interpolated_prompt = prompt.format(**ctx.variables)
        except (KeyError, IndexError, ValueError):
            # If interpolation fails, use the prompt as-is
            interpolated_prompt = prompt

        placeholder = ctx.collect_prompt(
            prompt=interpolated_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            source_id=__source_id,
        )

        return placeholder


def create_environment() -> jinja2.Environment:
    """Create a Jinja2 environment with the GenjiExtension.

    Returns:
        A configured Jinja2 environment.
    """
    env = jinja2.Environment(
        extensions=[GenjiExtension],
        autoescape=False,  # We handle escaping through filters
        undefined=jinja2.StrictUndefined,  # Fail on undefined variables
    )

    return env


def parse_template(source: str) -> jinja2.Template:
    """Parse a template string into a Jinja2 template.

    Args:
        source: The template source string.

    Returns:
        A compiled Jinja2 template.

    Raises:
        TemplateParseError: If the template is invalid.
    """
    env = create_environment()

    try:
        return env.from_string(source)
    except jinja2.TemplateSyntaxError as e:
        raise TemplateParseError(
            f"Invalid template syntax at line {e.lineno}: {e.message}"
        ) from e
    except jinja2.TemplateError as e:
        raise TemplateParseError(f"Template error: {e}") from e
