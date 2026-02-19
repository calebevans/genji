"""Base protocol and types for LLM backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass
class GenerationRequest:
    """Request for LLM text generation."""

    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    stop: Sequence[str] | None = None


@dataclass
class GenerationResponse:
    """Response from LLM text generation."""

    text: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None


class GenjiBackend(Protocol):
    """Protocol for synchronous LLM backends."""

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single completion.

        Args:
            request: The generation request.

        Returns:
            The generation response.
        """
        ...

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        """Generate multiple completions (implement for efficiency).

        Args:
            requests: Sequence of generation requests.

        Returns:
            Sequence of generation responses in the same order.
        """
        ...


class AsyncGenjiBackend(Protocol):
    """Protocol for asynchronous LLM backends."""

    async def agenerate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single completion asynchronously.

        Args:
            request: The generation request.

        Returns:
            The generation response.
        """
        ...

    async def agenerate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        """Generate multiple completions asynchronously.

        Args:
            requests: Sequence of generation requests.

        Returns:
            Sequence of generation responses in the same order.
        """
        ...
