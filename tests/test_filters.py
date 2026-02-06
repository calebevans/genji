"""Tests for filter functions."""

from __future__ import annotations

import pytest
from genji.exceptions import FilterError
from genji.filters import (
    html_filter,
    json_filter,
    lower_filter,
    raw_filter,
    strip_filter,
    truncate_filter,
    upper_filter,
    xml_filter,
    yaml_filter,
)


class TestJsonFilter:
    """Tests for JSON filter."""

    def test_json_escaping(self) -> None:
        """Test JSON string escaping covers quotes, newlines, and backslashes."""
        assert json_filter("Hello, World!") == '"Hello, World!"'
        assert json_filter('Say "Hello"') == '"Say \\"Hello\\""'
        assert json_filter("Line 1\nLine 2") == '"Line 1\\nLine 2"'
        assert json_filter("C:\\path") == '"C:\\\\path"'


class TestHtmlFilter:
    """Tests for HTML filter."""

    def test_html_escaping(self) -> None:
        """Test HTML entity escaping."""
        assert html_filter("<script>alert('xss')</script>") == (
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"
        )
        assert html_filter("A & B") == "A &amp; B"


class TestXmlFilter:
    """Tests for XML filter."""

    def test_xml_escaping(self) -> None:
        """Test XML entity escaping."""
        assert xml_filter("&<>\"'") == "&amp;&lt;&gt;&quot;&apos;"


class TestYamlFilter:
    """Tests for YAML filter."""

    def test_yaml_quoting(self) -> None:
        """Test YAML quoting for special characters and plain strings."""
        assert yaml_filter("hello") == "hello"
        assert yaml_filter("key: value") == '"key: value"'
        assert yaml_filter("true") == '"true"'


class TestUtilityFilters:
    """Tests for utility filters."""

    def test_raw_filter(self) -> None:
        """Test raw filter passes through unchanged."""
        assert raw_filter("<dangerous>") == "<dangerous>"

    def test_strip_filter(self) -> None:
        """Test whitespace stripping."""
        assert strip_filter("  hello  ") == "hello"

    def test_case_filters(self) -> None:
        """Test case conversion filters."""
        assert lower_filter("HELLO") == "hello"
        assert upper_filter("hello") == "HELLO"

    def test_truncate_filter(self) -> None:
        """Test truncation."""
        assert truncate_filter("This is a long string", 10) == "This is..."

        # Test error handling
        with pytest.raises(FilterError, match="must be >= suffix length"):
            truncate_filter("text", 2, suffix="...")
