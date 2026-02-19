"""Tests for async API (arender, arender_json, afrom_file, etc.)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from genji import Template
from genji.backends.base import GenerationRequest
from genji.backends.mock import MockBackend
from genji.exceptions import TemplateRenderError

# ---------------------------------------------------------------------------
# Async MockBackend tests
# ---------------------------------------------------------------------------


class TestAsyncMockBackend:
    """Tests for MockBackend async methods."""

    async def test_agenerate_basic(self) -> None:
        backend = MockBackend()
        resp = await backend.agenerate(GenerationRequest(prompt="test"))
        assert resp.text == "[MOCK: test]"

    async def test_agenerate_with_response_fn(self) -> None:
        backend = MockBackend(response_fn=lambda p: f"Reply: {p}")
        resp = await backend.agenerate(GenerationRequest(prompt="hello"))
        assert resp.text == "Reply: hello"

    async def test_agenerate_batch(self) -> None:
        backend = MockBackend(response_fn=lambda p: f"Response: {p}")
        requests = [
            GenerationRequest(prompt="first"),
            GenerationRequest(prompt="second"),
        ]
        responses = await backend.agenerate_batch(requests)

        assert len(responses) == 2
        assert responses[0].text == "Response: first"
        assert responses[1].text == "Response: second"

    async def test_agenerate_tracks_calls(self) -> None:
        backend = MockBackend()
        await backend.agenerate(GenerationRequest(prompt="a"))
        await backend.agenerate(GenerationRequest(prompt="b"))
        assert backend.call_count == 2
        assert len(backend.all_requests) == 2


# ---------------------------------------------------------------------------
# Async Template basics
# ---------------------------------------------------------------------------


class TestAsyncTemplateBasics:
    """Basic async template rendering tests."""

    async def test_arender_with_variable_interpolation(
        self, mock_backend: MockBackend
    ) -> None:
        template = Template('{{ gen("greeting for {name}") }}', backend=mock_backend)
        result = await template.arender(name="Alice")
        assert result == "Generated: greeting for Alice"

    async def test_arender_without_gen_calls(self) -> None:
        backend = MockBackend()
        template = Template("Hello {{ name }}", backend=backend)
        result = await template.arender(name="World")
        assert result == "Hello World"


# ---------------------------------------------------------------------------
# Async Template with filters
# ---------------------------------------------------------------------------


class TestAsyncTemplateWithFilters:
    """Tests for async templates with filters."""

    async def test_json_filter_produces_valid_json(self) -> None:
        backend = MockBackend(default_response="Hello, World!")
        template = Template(
            '{"message": {{ gen("greeting") | json }}}',
            backend=backend,
        )
        result = await template.arender()
        data = json.loads(result)
        assert data["message"] == "Hello, World!"

    async def test_html_filter_escapes_dangerous_content(self) -> None:
        backend = MockBackend(default_response="<script>alert('xss')</script>")
        template = Template('<div>{{ gen("content") | html }}</div>', backend=backend)
        result = await template.arender()
        assert "&lt;script&gt;" in result
        assert "<script>" not in result


# ---------------------------------------------------------------------------
# Async Template control flow
# ---------------------------------------------------------------------------


class TestAsyncTemplateControlFlow:
    """Tests for async templates with Jinja2 control flow."""

    async def test_gen_in_for_loop(self) -> None:
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
        result = await template.arender(items=["a", "b"])
        data = json.loads(result)
        assert len(data["items"]) == 2
        assert "Generated content for a" in data["items"][0]


# ---------------------------------------------------------------------------
# Async Template from file
# ---------------------------------------------------------------------------


class TestAsyncTemplateFromFile:
    """Tests for loading templates from files asynchronously."""

    async def test_afrom_file(self, mock_backend: MockBackend) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".genji", delete=False) as f:
            f.write('{{ gen("test") }}')
            f.flush()
            temp_path = f.name

        try:
            template = await Template.afrom_file(temp_path, backend=mock_backend)
            result = await template.arender()
            assert result == "Generated: test"
        finally:
            Path(temp_path).unlink()

    async def test_afrom_file_auto_detects_json_filter(self) -> None:
        backend = MockBackend(default_response="value")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json.genji", delete=False
        ) as f:
            f.write('{"key": {{ gen("prompt") }}}')
            f.flush()
            temp_path = f.name

        try:
            template = await Template.afrom_file(temp_path, backend=backend)
            result = await template.arender()
            data = json.loads(result)
            assert data["key"] == "value"
        finally:
            Path(temp_path).unlink()

    async def test_afrom_file_not_found(self, mock_backend: MockBackend) -> None:
        with pytest.raises(FileNotFoundError):
            await Template.afrom_file(
                "/nonexistent/template.genji", backend=mock_backend
            )


# ---------------------------------------------------------------------------
# Async render_json
# ---------------------------------------------------------------------------


class TestAsyncTemplateRenderJson:
    """Tests for arender_json() method."""

    async def test_arender_json_parses_output(self) -> None:
        backend = MockBackend(default_response="test value")
        template = Template('{"key": {{ gen("prompt") | json }}}', backend=backend)
        data = await template.arender_json()
        assert isinstance(data, dict)
        assert data["key"] == "test value"

    async def test_arender_json_error_on_invalid(self) -> None:
        backend = MockBackend(default_response="value")
        template = Template('This is not JSON: {{ gen("prompt") }}', backend=backend)
        with pytest.raises(TemplateRenderError, match="not valid JSON"):
            await template.arender_json()


# ---------------------------------------------------------------------------
# Async error handling
# ---------------------------------------------------------------------------


class TestAsyncTemplateErrorHandling:
    """Tests for async error handling."""

    async def test_backend_error_propagation(self) -> None:
        class ErrorBackend:
            def generate(self, request):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

            def generate_batch(self, requests):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

        template = Template('{{ gen("test") }}', backend=ErrorBackend())  # type: ignore[arg-type]
        with pytest.raises(TemplateRenderError, match="Backend failed"):
            await template.arender()

    async def test_async_backend_error_propagation(self) -> None:
        class AsyncErrorBackend:
            def generate(self, request):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

            def generate_batch(self, requests):  # type: ignore[no-untyped-def]
                raise RuntimeError("Backend failed")

            async def agenerate_batch(self, requests):  # type: ignore[no-untyped-def]
                raise RuntimeError("Async backend failed")

        template = Template('{{ gen("test") }}', backend=AsyncErrorBackend())  # type: ignore[arg-type]
        with pytest.raises(TemplateRenderError, match="Async backend failed"):
            await template.arender()


# ---------------------------------------------------------------------------
# Async with sync-only backend (asyncio.to_thread fallback)
# ---------------------------------------------------------------------------


class TestAsyncWithSyncBackend:
    """Tests that arender works with a backend that has no async methods."""

    async def test_arender_falls_back_to_thread(self) -> None:
        class SyncOnlyBackend:
            """Backend that only implements the sync protocol."""

            def generate(self, request: GenerationRequest):  # type: ignore[no-untyped-def]
                return MockBackend(response_fn=lambda p: f"sync: {p}").generate(request)

            def generate_batch(self, requests):  # type: ignore[no-untyped-def]
                return [self.generate(r) for r in requests]

        template = Template('{{ gen("hello") }}', backend=SyncOnlyBackend())  # type: ignore[arg-type]
        result = await template.arender()
        assert result == "sync: hello"


# ---------------------------------------------------------------------------
# Async integration
# ---------------------------------------------------------------------------


class TestAsyncTemplateIntegration:
    """Async integration tests matching the success criteria."""

    async def test_success_criteria_example(self) -> None:
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

        result = await template.arender(name="Alice")
        data = json.loads(result)

        assert "greeting" in data
        assert "farewell" in data
        assert "Alice" in data["greeting"]
        assert "Alice" in data["farewell"]

    async def test_arender_json_integration(self) -> None:
        backend = MockBackend(
            response_fn=lambda p: {
                "a friendly greeting for Bob": "Hi Bob!",
                "a warm farewell for Bob": "Bye Bob!",
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

        data = await template.arender_json(name="Bob")
        assert data["greeting"] == "Hi Bob!"
        assert data["farewell"] == "Bye Bob!"
