"""Tests for parser and Jinja2 extension."""

from __future__ import annotations

import pytest
from genji.context import RenderContext, set_current_context
from genji.exceptions import TemplateParseError
from genji.parser import create_environment, parse_template


class TestParser:
    """Tests for template parsing and gen() function."""

    def test_parse_error_handling(self) -> None:
        """Test parsing invalid template syntax."""
        with pytest.raises(TemplateParseError, match="Invalid template syntax"):
            parse_template("Hello {{ name")

    def test_gen_prompt_collection(self) -> None:
        """Test that gen() collects prompts with parameters."""
        env = create_environment()
        template = env.from_string('{{ gen("test", max_tokens=100) }}')

        ctx = RenderContext()
        set_current_context(ctx)

        try:
            template.render()
            assert len(ctx.prompts) == 1
            assert ctx.prompts[0].prompt == "test"
            assert ctx.prompts[0].max_tokens == 100
        finally:
            set_current_context(None)

    def test_gen_in_loop(self) -> None:
        """Test gen() inside a loop with variable interpolation."""
        env = create_environment()
        template = env.from_string(
            '{% for item in items %}{{ gen("prompt for " + item) }}{% endfor %}'
        )

        ctx = RenderContext()
        set_current_context(ctx)

        try:
            template.render(items=["a", "b"])
            assert len(ctx.prompts) == 2
            assert ctx.prompts[0].prompt == "prompt for a"
            assert ctx.prompts[1].prompt == "prompt for b"
        finally:
            set_current_context(None)
