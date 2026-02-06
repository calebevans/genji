"""Shared test fixtures for Genji tests."""

from __future__ import annotations

import pytest
from genji import MockBackend


@pytest.fixture
def mock_backend() -> MockBackend:
    """Create a mock backend that returns predictable responses."""
    return MockBackend(response_fn=lambda prompt: f"Generated: {prompt}")


@pytest.fixture
def simple_mock_backend() -> MockBackend:
    """Create a mock backend that returns a simple fixed response."""
    return MockBackend(default_response="Test Response")
