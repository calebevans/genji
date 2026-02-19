"""LiteLLM backend implementation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import litellm
except ImportError as e:
    raise ImportError(
        "LiteLLM is required but not installed. Install it with: pip install litellm"
    ) from e

from ..exceptions import BackendError
from .base import GenerationRequest, GenerationResponse


class LLMBackend:
    """LLM backend using LiteLLM for multi-provider support.

    Supports both synchronous and asynchronous generation. Use ``generate``/
    ``generate_batch`` for sync code and ``agenerate``/``agenerate_batch``
    for async code.

    Supports 100+ LLM providers through LiteLLM's unified API including:
    - OpenAI: model="gpt-4o"
    - Anthropic: model="claude-3-5-sonnet-20241022"
    - Local Ollama: model="ollama/llama3", base_url="http://localhost:11434"
    - Azure: model="azure/deployment-name"
    - And many more
    """

    SYSTEM_INSTRUCTION = (
        "You are a content generator for structured output. "
        "Return ONLY the literal content requested. "
        "Do not provide multiple options, alternatives, explanations, "
        "or meta-commentary. "
        "If the request uses singular form (e.g., 'a title', 'a name'), "
        "return exactly one item. "
        "If the request specifies a quantity or length, match it exactly. "
        "Do not add formatting, numbering, or markdown unless explicitly "
        "requested in the prompt. "
        "Be direct and literal."
    )

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        add_system_prompt: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the LiteLLM backend.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet").
                Falls back to GENJI_MODEL env var. Required if env var not set.
            api_key: API key for the provider.
                Falls back to GENJI_API_KEY env var or provider-specific env vars.
            base_url: Base URL for local LLMs (Ollama, vLLM, etc.).
                Falls back to GENJI_BASE_URL env var.
            temperature: Temperature for generation. None uses provider default.
            max_tokens: Max tokens for generation. None means no limit.
            add_system_prompt: Whether to add system instruction for concise responses.
                Defaults to True. Set to False if you want full control over prompts.
            **kwargs: Additional arguments to pass to litellm.completion().

        Raises:
            ValueError: If model is not provided and GENJI_MODEL is not set.
        """
        self.model = model or os.getenv("GENJI_MODEL")
        if not self.model:
            raise ValueError(
                "Model name is required. Either pass model= parameter or "
                "set GENJI_MODEL environment variable."
            )
        self.api_key = api_key or os.getenv("GENJI_API_KEY")
        self.base_url = base_url or os.getenv("GENJI_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.add_system_prompt = add_system_prompt
        self.kwargs = kwargs

    def _build_litellm_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        """Build the keyword arguments dict for a litellm completion call."""
        messages: list[dict[str, str]] = []
        if self.add_system_prompt:
            messages.append({"role": "system", "content": self.SYSTEM_INSTRUCTION})
        messages.append({"role": "user", "content": request.prompt})

        litellm_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **self.kwargs,
        }

        temperature_value = request.temperature or self.temperature
        if temperature_value is not None:
            litellm_kwargs["temperature"] = temperature_value

        max_tokens_value = request.max_tokens or self.max_tokens
        if max_tokens_value is not None:
            litellm_kwargs["max_tokens"] = max_tokens_value

        if self.api_key:
            litellm_kwargs["api_key"] = self.api_key

        if self.base_url:
            litellm_kwargs["api_base"] = self.base_url

        if request.stop:
            litellm_kwargs["stop"] = request.stop

        return litellm_kwargs

    @staticmethod
    def _parse_response(response: Any) -> GenerationResponse:
        """Extract a GenerationResponse from a litellm response object."""
        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason

        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return GenerationResponse(
            text=text,
            finish_reason=finish_reason,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single completion.

        Args:
            request: The generation request.

        Returns:
            The generation response.

        Raises:
            BackendError: If generation fails.
        """
        litellm_kwargs = self._build_litellm_kwargs(request)
        try:
            response = litellm.completion(**litellm_kwargs)
            return self._parse_response(response)
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"LiteLLM generation failed: {e}") from e

    def generate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        """Generate multiple completions in parallel using threads.

        Args:
            requests: Sequence of generation requests.

        Returns:
            Sequence of generation responses in the same order.

        Raises:
            BackendError: If any generation fails.
        """
        if not requests:
            return []

        if len(requests) == 1:
            return [self.generate(requests[0])]

        responses: list[GenerationResponse | None] = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=min(len(requests), 10)) as executor:
            future_to_index = {
                executor.submit(self.generate, req): i for i, req in enumerate(requests)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    responses[index] = future.result()
                except Exception as e:
                    raise BackendError(
                        f"Batch generation failed for request {index}: {e}"
                    ) from e

        return [r for r in responses if r is not None]

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------

    async def agenerate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a single completion asynchronously.

        Args:
            request: The generation request.

        Returns:
            The generation response.

        Raises:
            BackendError: If generation fails.
        """
        litellm_kwargs = self._build_litellm_kwargs(request)
        try:
            response = await litellm.acompletion(**litellm_kwargs)
            return self._parse_response(response)
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(f"LiteLLM generation failed: {e}") from e

    async def agenerate_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> Sequence[GenerationResponse]:
        """Generate multiple completions concurrently with asyncio.gather.

        Args:
            requests: Sequence of generation requests.

        Returns:
            Sequence of generation responses in the same order.

        Raises:
            BackendError: If any generation fails.
        """
        if not requests:
            return []

        if len(requests) == 1:
            return [await self.agenerate(requests[0])]

        return list(await asyncio.gather(*(self.agenerate(r) for r in requests)))
