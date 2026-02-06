"""Exception classes for Genji."""

from __future__ import annotations


class GenjiError(Exception):
    """Base exception for all genji errors."""


class TemplateParseError(GenjiError):
    """Template syntax is invalid."""


class TemplateRenderError(GenjiError):
    """Error during template rendering."""


class BackendError(GenjiError):
    """LLM backend failed."""


class FilterError(GenjiError):
    """Filter application failed."""
