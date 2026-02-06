"""Genji - Jinja2-based templating for LLM-generated structured output.

Genji combines Jinja2-style templating with LLM content generation, where the
template owns the structure/syntax (JSON brackets, HTML tags, YAML indentation)
while the LLM only generates content that gets safely interpolated into the template.

Example:
    >>> from genji import Template, LLMBackend
    >>> backend = LLMBackend(model="gpt-4o-mini")
    >>> template = Template('''
    ... {
    ...   "greeting": {{ gen("a friendly greeting for {name}") | json }}
    ... }
    ... ''', backend=backend)
    >>> result = template.render(name="Alice")
"""

from __future__ import annotations

from .backends.litellm import LLMBackend
from .backends.mock import MockBackend
from .exceptions import (
    BackendError,
    FilterError,
    GenjiError,
    TemplateParseError,
    TemplateRenderError,
)
from .template import Template

__version__ = "0.1.0"

__all__ = [
    # Main API
    "Template",
    # Backends
    "LLMBackend",
    "MockBackend",
    # Exceptions
    "GenjiError",
    "TemplateParseError",
    "TemplateRenderError",
    "BackendError",
    "FilterError",
    # Version
    "__version__",
]
