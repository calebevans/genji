"""Tests for Template class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from genji import Template
from genji.backends.mock import MockBackend
from genji.exceptions import TemplateRenderError


class TestTemplateBasics:
    """Basic template tests."""

    def test_gen_with_variable_interpolation(self, mock_backend: MockBackend) -> None:
        """Test gen() with variable interpolation in prompt."""
        template = Template('{{ gen("greeting for {name}") }}', backend=mock_backend)
        result = template.render(name="Alice")
        assert result == "Generated: greeting for Alice"


class TestTemplateWithFilters:
    """Tests for templates with filters."""

    def test_json_filter_produces_valid_json(self) -> None:
        """Test gen() with JSON filter produces valid, parseable JSON."""
        backend = MockBackend(default_response="Hello, World!")
        template = Template(
            '{"message": {{ gen("greeting") | json }}}',
            backend=backend,
        )
        result = template.render()
        data = json.loads(result)  # Should not raise
        assert data["message"] == "Hello, World!"

    def test_html_filter_escapes_dangerous_content(self) -> None:
        """Test gen() with HTML filter prevents XSS."""
        backend = MockBackend(default_response="<script>alert('xss')</script>")
        template = Template('<div>{{ gen("content") | html }}</div>', backend=backend)
        result = template.render()
        assert "&lt;script&gt;" in result
        assert "<script>" not in result


class TestTemplateControlFlow:
    """Tests for templates with Jinja2 control flow."""

    def test_gen_in_for_loop(self) -> None:
        """Test gen() inside for loop produces valid JSON."""
        backend = MockBackend(response_fn=lambda p: f"Generated {p}")
        template = Template(
            """
{
  "items": [
    {% for item in items %}
    {{ gen("content for " + item) | json }}{% if not loop.last %},{% endif %}
    {% endfor %}
  ]
}
            """.strip(),
            backend=backend,
        )
        result = template.render(items=["a", "b"])
        data = json.loads(result)
        assert len(data["items"]) == 2
        assert "Generated content for a" in data["items"][0]


class TestTemplateFromFile:
    """Tests for loading templates from files."""

    def test_load_from_file(self, mock_backend: MockBackend) -> None:
        """Test loading template from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".genji", delete=False) as f:
            f.write('{{ gen("test") }}')
            f.flush()
            temp_path = f.name

        try:
            template = Template.from_file(temp_path, backend=mock_backend)
            result = template.render()
            assert result == "Generated: test"
        finally:
            Path(temp_path).unlink()


class TestTemplateRenderJson:
    """Tests for render_json() method."""

    def test_render_json_parses_output(self) -> None:
        """Test render_json() parses JSON output to dict."""
        backend = MockBackend(default_response="test value")
        template = Template('{"key": {{ gen("prompt") | json }}}', backend=backend)
        data = template.render_json()
        assert isinstance(data, dict)
        assert data["key"] == "test value"

    def test_render_json_error_on_invalid(self) -> None:
        """Test render_json() raises error for invalid JSON."""
        backend = MockBackend(default_response="value")
        template = Template('This is not JSON: {{ gen("prompt") }}', backend=backend)
        with pytest.raises(TemplateRenderError, match="not valid JSON"):
            template.render_json()


class TestTemplateErrorHandling:
    """Tests for error handling."""

    def test_backend_error_propagation(self) -> None:
        """Test backend errors are caught and wrapped."""

        class ErrorBackend:
            def generate(self, request):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

            def generate_batch(self, requests):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

        template = Template('{{ gen("test") }}', backend=ErrorBackend())  # type: ignore[arg-type]
        with pytest.raises(TemplateRenderError, match="Backend failed"):
            template.render()


class TestTemplateIntegration:
    """Integration tests matching the success criteria."""

    def test_success_criteria_example(self) -> None:
        """Test the exact example from the success criteria."""
        backend = MockBackend(
            response_fn=lambda p: {
                "a friendly greeting for Alice": "Hello Alice! How are you today?",
                "a warm farewell for Alice": "Goodbye Alice! Take care!",
            }.get(p, f"Response for: {p}")
        )

        template = Template(
            """
{
  "greeting": {{ gen("a friendly greeting for {name}") | json }},
  "farewell": {{ gen("a warm farewell for {name}") | json }}
}
            """.strip(),
            backend=backend,
        )

        result = template.render(name="Alice")
        data = json.loads(result)  # Should not raise - valid JSON guaranteed

        assert "greeting" in data
        assert "farewell" in data
        assert "Alice" in data["greeting"]
        assert "Alice" in data["farewell"]
