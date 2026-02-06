"""Format-specific escaping filters for Genji templates."""

from __future__ import annotations

import html
import json as json_module
import re
from collections.abc import Callable
from typing import Any

from .exceptions import FilterError


def json_filter(value: Any) -> str:
    """Escape value for JSON string (includes surrounding quotes).

    This filter outputs a complete JSON string value including the quotes,
    so it can be safely placed in a JSON context.

    Args:
        value: The value to escape.

    Returns:
        A JSON-escaped string with surrounding quotes.

    Example:
        >>> json_filter("Hello, World!")
        '"Hello, World!"'
        >>> json_filter('Line 1\\nLine 2')
        '"Line 1\\nLine 2"'
    """
    try:
        return json_module.dumps(str(value), ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise FilterError(f"Failed to JSON-encode value: {e}") from e


def html_filter(value: Any) -> str:
    """HTML entity escaping.

    Args:
        value: The value to escape.

    Returns:
        HTML-escaped string.
    """
    return html.escape(str(value), quote=True)


def xml_filter(value: Any) -> str:
    """XML escaping.

    Args:
        value: The value to escape.

    Returns:
        XML-escaped string.
    """
    # XML requires escaping of: & < > " '
    s = str(value)
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


def yaml_filter(value: Any) -> str:
    """YAML string escaping.

    Handles multiline strings and special characters properly.
    Returns a quoted string if necessary for YAML.

    Args:
        value: The value to escape.

    Returns:
        YAML-safe string representation.
    """
    s = str(value)

    # Check if we need quoting
    needs_quoting = any(
        [
            s.startswith((" ", "\t")),
            s.endswith((" ", "\t")),
            "\n" in s,
            ":" in s,
            "#" in s,
            "-" in s and s.startswith("-"),
            s.lower() in ("true", "false", "null", "yes", "no", "on", "off"),
            re.match(r"^[0-9]", s) is not None,
        ]
    )

    if needs_quoting or not s:
        # Use double quotes and escape necessary characters
        s = s.replace("\\", "\\\\")
        s = s.replace('"', '\\"')
        s = s.replace("\n", "\\n")
        s = s.replace("\r", "\\r")
        s = s.replace("\t", "\\t")
        return f'"{s}"'

    return s


def raw_filter(value: Any) -> str:
    """No escaping (pass-through).

    WARNING: This filter provides no escaping and can break output format.
    Use with caution and only when you're certain the content is safe.

    Args:
        value: The value to pass through.

    Returns:
        The string value unchanged.
    """
    return str(value)


def strip_filter(value: Any) -> str:
    """Strip leading and trailing whitespace.

    Args:
        value: The value to strip.

    Returns:
        String with whitespace stripped.
    """
    return str(value).strip()


def lower_filter(value: Any) -> str:
    """Convert to lowercase.

    Args:
        value: The value to convert.

    Returns:
        Lowercase string.
    """
    return str(value).lower()


def upper_filter(value: Any) -> str:
    """Convert to uppercase.

    Args:
        value: The value to convert.

    Returns:
        Uppercase string.
    """
    return str(value).upper()


def truncate_filter(value: Any, length: int = 255, suffix: str = "...") -> str:
    """Truncate string to specified length.

    Args:
        value: The value to truncate.
        length: Maximum length (including suffix).
        suffix: Suffix to append if truncated (default "...").

    Returns:
        Truncated string.

    Raises:
        FilterError: If length is less than suffix length.
    """
    s = str(value)

    if length < len(suffix):
        raise FilterError(
            f"Truncate length ({length}) must be >= suffix length ({len(suffix)})"
        )

    if len(s) <= length:
        return s

    return s[: length - len(suffix)] + suffix


# Registry of all filters
FILTERS: dict[str, Callable[[Any], str]] = {
    "json": json_filter,
    "html": html_filter,
    "xml": xml_filter,
    "yaml": yaml_filter,
    "raw": raw_filter,
    "strip": strip_filter,
    "lower": lower_filter,
    "upper": upper_filter,
    "truncate": truncate_filter,
}
