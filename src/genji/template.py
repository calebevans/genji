"""Main Template class for Genji."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import jinja2

from .backends.base import GenjiBackend
from .context import RenderContext, set_current_context
from .exceptions import TemplateRenderError
from .filters import FILTERS
from .parser import create_environment


class Template:
    """A Genji template that combines Jinja2 syntax with LLM generation.

    The template owns the structure/syntax while the LLM only generates content
    that gets safely interpolated into the template.

    Both synchronous (``render``, ``render_json``) and asynchronous
    (``arender``, ``arender_json``) rendering are supported.
    """

    def __init__(
        self, source: str, backend: GenjiBackend, default_filter: str | None = None
    ) -> None:
        """Initialize a template.

        Args:
            source: The template source string.
            backend: The LLM backend to use for generation.
            default_filter: Default filter to apply to all gen() calls
                (e.g., "json", "html"). Can be overridden per-prompt by
                explicitly using | filter in template. Use "raw" in template
                to skip the default filter for a specific gen().
        """
        self._source = source
        self._backend = backend
        self._default_filter = default_filter
        self._filter_chains, modified_source = self._extract_and_inject_filters(source)
        self._env = self._create_environment()
        self._template = self._env.from_string(modified_source)

    @classmethod
    def from_file(
        cls, path: str | Path, backend: GenjiBackend, default_filter: str | None = None
    ) -> Template:
        """Load a template from a file.

        Args:
            path: Path to the template file.
            backend: The LLM backend to use for generation.
            default_filter: Default filter to apply to all gen() calls.
                Auto-detected from file extension
                (.json.genji -> "json", .html.genji -> "html").

        Returns:
            A Template instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            TemplateRenderError: If the file can't be read.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as e:
            raise TemplateRenderError(
                f"Failed to read template file {path}: {e}"
            ) from e

        default_filter = cls._detect_filter(path, default_filter)
        return cls(source, backend, default_filter)

    @classmethod
    async def afrom_file(
        cls, path: str | Path, backend: GenjiBackend, default_filter: str | None = None
    ) -> Template:
        """Load a template from a file asynchronously.

        Args:
            path: Path to the template file.
            backend: The LLM backend to use for generation.
            default_filter: Default filter to apply to all gen() calls.
                Auto-detected from file extension
                (.json.genji -> "json", .html.genji -> "html").

        Returns:
            A Template instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            TemplateRenderError: If the file can't be read.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        try:
            source = await asyncio.to_thread(path.read_text, "utf-8")
        except Exception as e:
            raise TemplateRenderError(
                f"Failed to read template file {path}: {e}"
            ) from e

        default_filter = cls._detect_filter(path, default_filter)
        return cls(source, backend, default_filter)

    @staticmethod
    def _detect_filter(path: Path, default_filter: str | None) -> str | None:
        """Auto-detect the default filter from the file extension.

        Args:
            path: Path to the template file.
            default_filter: Explicit default filter (returned as-is when set).

        Returns:
            The resolved default filter name, or None.
        """
        if default_filter is not None:
            return default_filter

        filename = path.name.lower()
        if ".json.genji" in filename:
            return "json"
        if ".html.genji" in filename:
            return "html"
        if ".xml.genji" in filename:
            return "xml"
        if ".yaml.genji" in filename or ".yml.genji" in filename:
            return "yaml"
        return None

    def _extract_and_inject_filters(
        self, source: str
    ) -> tuple[dict[int, list[str]], str]:
        """Extract filter chains and inject source IDs into gen() calls.

        This parses the template to find patterns like:
        {{ gen(...) | filter1 | filter2 }}

        And transforms them to:
        {{ gen(..., __source_id=0) }}

        Args:
            source: The template source string.

        Returns:
            Tuple of (filter chains dict, modified source with injected IDs).
        """
        filter_chains: dict[int, list[str]] = {}
        modified_source = source
        gen_index = 0
        offset = 0  # Track offset for string replacements

        # Pattern to match {{ gen(...) with optional filters }}
        # We need to match parentheses carefully to handle nested parens in gen()
        pattern = r"\{\{\s*gen\(([^)]+(?:\([^)]*\)[^)]*)*)\)\s*(\|[^}]+)?\s*\}\}"

        for match in re.finditer(pattern, source):
            # Extract filter chain
            filters_part = match.group(2)
            if filters_part:
                # Extract filter names from | filter1 | filter2 format
                filter_names = [f.strip() for f in filters_part.split("|") if f.strip()]
                # Handle filters with arguments like truncate(10)
                cleaned_filters = []
                for f in filter_names:
                    # Extract just the filter name (before any parentheses)
                    filter_name = f.split("(")[0].strip()
                    if filter_name:
                        cleaned_filters.append(filter_name)
                filter_chains[gen_index] = cleaned_filters
            else:
                filter_chains[gen_index] = []

            # Inject source ID into the gen() call
            gen_args = match.group(1)
            replacement = f"{{{{ gen({gen_args}, __source_id={gen_index}) }}}}"

            # Replace in modified source (accounting for previous replacements)
            start = match.start() + offset
            end = match.end() + offset
            modified_source = (
                modified_source[:start] + replacement + modified_source[end:]
            )
            offset += len(replacement) - (match.end() - match.start())

            gen_index += 1

        return filter_chains, modified_source

    def _create_environment(self) -> jinja2.Environment:
        """Create a Jinja2 environment with custom filters.

        Returns:
            Configured Jinja2 environment.
        """
        env = create_environment()

        # Register all filters as identity functions during template compilation
        # The real filters will be applied after generation to the generated content
        for filter_name in FILTERS:
            env.filters[filter_name] = lambda x: x  # Identity function

        return env

    def _interpolate(self, first_pass: str, render_ctx: RenderContext) -> str:
        """Replace placeholders with filtered generated content.

        Args:
            first_pass: The template output from the collection phase.
            render_ctx: The render context containing prompts and generated content.

        Returns:
            The final rendered string with all placeholders replaced.
        """
        result = first_pass
        for prompt in render_ctx.prompts:
            generated = render_ctx.get_generated(prompt.placeholder)

            filter_chain = self._filter_chains.get(prompt.source_id or 0, [])

            if not filter_chain and self._default_filter:
                filter_chain = [self._default_filter]
            elif filter_chain == ["raw"]:
                filter_chain = []

            filtered_content = generated
            for filter_name in filter_chain:
                filter_fn = FILTERS.get(filter_name)
                if filter_fn:
                    try:
                        filtered_content = filter_fn(filtered_content)
                    except Exception as e:
                        raise TemplateRenderError(
                            f"Filter '{filter_name}' failed: {e}"
                        ) from e

            result = result.replace(prompt.placeholder, filtered_content)

        return result

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def render(self, **context: Any) -> str:
        """Render the template with the given context.

        This uses a three-phase rendering process:
        1. Collection phase: Execute Jinja2 logic, collect gen() calls
        2. Generation phase: Batch all prompts to LLM backend
        3. Interpolation phase: Replace placeholders with generated content

        Args:
            **context: Template variables.

        Returns:
            The rendered template string.

        Raises:
            TemplateRenderError: If rendering fails.
        """
        render_ctx = RenderContext(variables=context)

        try:
            set_current_context(render_ctx)

            # Phase 1: Collection
            try:
                first_pass = self._template.render(**context)
            except jinja2.TemplateError as e:
                raise TemplateRenderError(f"Template rendering failed: {e}") from e

            if not render_ctx.prompts:
                return first_pass

            # Phase 2: Generation
            requests = [p.to_request() for p in render_ctx.prompts]

            try:
                responses = self._backend.generate_batch(requests)
            except Exception as e:
                raise TemplateRenderError(f"LLM backend generation failed: {e}") from e

            for prompt, response in zip(render_ctx.prompts, responses):
                render_ctx.set_generated(prompt.placeholder, response.text)

            # Phase 3: Interpolation
            return self._interpolate(first_pass, render_ctx)

        finally:
            set_current_context(None)

    def render_json(self, **context: Any) -> dict[str, Any]:
        """Render the template and parse as JSON.

        Args:
            **context: Template variables.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            TemplateRenderError: If rendering fails or result is not valid JSON.
        """
        result = self.render(**context)

        try:
            return json.loads(result)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise TemplateRenderError(
                f"Template output is not valid JSON: {e}\nOutput was:\n{result}"
            ) from e

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------

    async def arender(self, **context: Any) -> str:
        """Render the template with the given context asynchronously.

        Uses the same three-phase process as ``render`` but awaits the
        generation phase.  If the backend exposes ``agenerate_batch`` it is
        called natively; otherwise the synchronous ``generate_batch`` is
        offloaded to a thread via ``asyncio.to_thread``.

        Args:
            **context: Template variables.

        Returns:
            The rendered template string.

        Raises:
            TemplateRenderError: If rendering fails.
        """
        render_ctx = RenderContext(variables=context)

        try:
            set_current_context(render_ctx)

            # Phase 1: Collection (sync -- Jinja2 template execution)
            try:
                first_pass = self._template.render(**context)
            except jinja2.TemplateError as e:
                raise TemplateRenderError(f"Template rendering failed: {e}") from e

            if not render_ctx.prompts:
                return first_pass

            # Phase 2: Generation (async)
            requests = [p.to_request() for p in render_ctx.prompts]

            try:
                if hasattr(self._backend, "agenerate_batch"):
                    responses = await self._backend.agenerate_batch(requests)
                else:
                    responses = await asyncio.to_thread(
                        self._backend.generate_batch, requests
                    )
            except Exception as e:
                raise TemplateRenderError(f"LLM backend generation failed: {e}") from e

            for prompt, response in zip(render_ctx.prompts, responses):
                render_ctx.set_generated(prompt.placeholder, response.text)

            # Phase 3: Interpolation
            return self._interpolate(first_pass, render_ctx)

        finally:
            set_current_context(None)

    async def arender_json(self, **context: Any) -> dict[str, Any]:
        """Render the template and parse as JSON asynchronously.

        Args:
            **context: Template variables.

        Returns:
            Parsed JSON as a Python dict.

        Raises:
            TemplateRenderError: If rendering fails or result is not valid JSON.
        """
        result = await self.arender(**context)

        try:
            return json.loads(result)  # type: ignore[no-any-return]
        except json.JSONDecodeError as e:
            raise TemplateRenderError(
                f"Template output is not valid JSON: {e}\nOutput was:\n{result}"
            ) from e
