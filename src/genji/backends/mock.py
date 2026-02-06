"""Mock backend for testing."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .base import GenerationRequest, GenerationResponse


class MockBackend:
    """Mock LLM backend for testing.

    Returns deterministic responses based on the provided response function
    or a simple echo of the prompt.
    """

    def __init__(
        self,
        response_fn: Callable[[str], str] | None = None,
        default_response: str | None = None,
    ) -> None:
        """Initialize the mock backend.

        Args:
            response_fn: Optional function that takes a prompt and returns a response.
                If None, uses default_response or echoes the prompt.
            default_response: Default response to return if response_fn is None.
                If both are None, echoes "[MOCK: {prompt}]".
        """
        self._response_fn = response_fn
        self._default_response = default_response
        self.call_count = 0
        self.last_request: GenerationRequest | None = None
        self.all_requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single mock completion.

        Args:
            request: The generation request.

        Returns:
            The mock generation response.
        """
        self.call_count += 1
        self.last_request = request
        self.all_requests.append(request)

        if self._response_fn:
            text = self._response_fn(request.prompt)
        elif self._default_response is not None:
            text = self._default_response
        else:
            text = f"[MOCK: {request.prompt}]"

        return GenerationResponse(
            text=text,
            finish_reason="stop",
            usage={
                "prompt_tokens": len(request.prompt),
                "completion_tokens": len(text),
            },
        )

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        """Generate multiple mock completions.

        Args:
            requests: Sequence of generation requests.

        Returns:
            Sequence of mock generation responses.
        """
        return [self.generate(req) for req in requests]
