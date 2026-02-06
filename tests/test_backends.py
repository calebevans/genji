"""Tests for LLM backends."""

from __future__ import annotations

from genji.backends.base import GenerationRequest
from genji.backends.mock import MockBackend


class TestMockBackend:
    """Tests for MockBackend."""

    def test_basic_generation(self) -> None:
        """Test basic generation with different response modes."""
        # Default echo
        backend1 = MockBackend()
        assert (
            backend1.generate(GenerationRequest(prompt="test")).text == "[MOCK: test]"
        )

        # Custom function
        backend2 = MockBackend(response_fn=lambda p: f"Reply: {p}")
        assert (
            backend2.generate(GenerationRequest(prompt="hello")).text == "Reply: hello"
        )

    def test_batch_generation(self) -> None:
        """Test batch generation."""
        backend = MockBackend(response_fn=lambda p: f"Response: {p}")
        requests = [
            GenerationRequest(prompt="first"),
            GenerationRequest(prompt="second"),
        ]
        responses = backend.generate_batch(requests)

        assert len(responses) == 2
        assert responses[0].text == "Response: first"
        assert responses[1].text == "Response: second"
