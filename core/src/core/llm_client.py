# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import json
import logging
from types import TracebackType
from typing import Any, Type

import httpx
import instructor
import litellm
from datarobot_genai.core.utils.token_tracking import TokenUsageTracker
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from opentelemetry import trace

from core.config import Config

log = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


class InstructorClientWrapper:
    """
    Wrapper that intercepts instructor client calls and tracks tokens.
    """

    def __init__(
        self,
        client: instructor.Instructor,
        tracker: TokenUsageTracker | None = None,
        api_base: str | None = None,
    ):
        self._client = client
        self._tracker = tracker
        self._api_base = api_base

    @property
    def chat(self) -> ChatWrapper:
        """Return wrapped chat interface."""
        return ChatWrapper(self._client.chat, self._tracker, self._api_base)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)


class ChatWrapper:
    """Wrapper for chat interface."""

    def __init__(
        self, chat: Any, tracker: TokenUsageTracker | None, api_base: str | None = None
    ):
        self._chat = chat
        self._tracker = tracker
        self._api_base = api_base

    @property
    def completions(self) -> CompletionsWrapper:
        """Return wrapped completions interface."""
        return CompletionsWrapper(self._chat.completions, self._tracker, self._api_base)


class CompletionsWrapper:
    """Wrapper for completions interface with token tracking and verbose error handling."""

    def __init__(
        self,
        completions: Any,
        tracker: TokenUsageTracker | None,
        api_base: str | None = None,
    ):
        self._completions = completions
        self._tracker = tracker
        self._api_base = api_base

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept create calls to track tokens and inject api_base."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Inject api_base if configured and not already provided
        if self._api_base and "api_base" not in kwargs:
            kwargs["api_base"] = self._api_base

        # Set timeout and max_retries if not provided (matching original AsyncOpenAI config)
        if "timeout" not in kwargs:
            kwargs["timeout"] = 900
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = 2

        log.debug(
            f"LLM API call starting - model: {model}, api_base: {kwargs.get('api_base', 'default')}, "
            f"messages_count: {len(messages)}, timeout: {kwargs.get('timeout')}s"
        )

        with _tracer.start_as_current_span(f"gen_ai.chat {model}") as span:
            span.set_attribute("gen_ai.prompt", json.dumps(messages, default=str))
            span.set_attribute("gen_ai.request.model", model)
            span.set_attribute("gen_ai.system", "datarobot")

            try:
                result = await self._completions.create(*args, **kwargs)

                # Track tokens if tracker is available
                if self._tracker:
                    self._tracker.track_call(messages, result, model)

                try:
                    completion_text = (
                        result.model_dump_json()
                        if hasattr(result, "model_dump_json")
                        else json.dumps(result, default=str)
                    )
                    span.set_attribute("gen_ai.completion", completion_text)
                except Exception:
                    log.warning(
                        "Failed to serialize LLM completion for tracing", exc_info=True
                    )

                log.debug(f"LLM API call completed successfully - model: {model}")
                return result

            except Exception as e:
                self._log_llm_error(e, model, kwargs)
                raise

    def _log_llm_error(self, e: Exception, model: str, kwargs: dict[str, Any]) -> None:
        """Log detailed error information for LLM API failures."""
        api_base = kwargs.get("api_base", "default")
        timeout = kwargs.get("timeout", "unknown")
        error_type = type(e).__name__
        error_message = str(e)

        # Extract status code if available
        status_code = None
        if hasattr(e, "status_code"):
            status_code = e.status_code
        elif hasattr(e, "response") and hasattr(e.response, "status_code"):
            status_code = e.response.status_code

        # Map exception types to error descriptions
        error_descriptions: dict[type, tuple[str, str]] = {
            APITimeoutError: (
                "TIMEOUT",
                f"Request timed out after {timeout}s. Check if the deployment is responsive.",
            ),
            AuthenticationError: (
                "AUTH ERROR (401)",
                "Unauthorized access. Check API token and permissions.",
            ),
            NotFoundError: (
                "NOT FOUND (404)",
                "Resource not found. The deployment ID may be incorrect or deleted.",
            ),
            RateLimitError: (
                "RATE LIMIT (429)",
                "Rate limit exceeded. Too many requests, please slow down.",
            ),
            InternalServerError: (
                "SERVER ERROR (500)",
                "Internal server error. The LLM deployment may be misconfigured or down. "
                "Often caused by: incorrect model name, deployment issues, or unavailable target model.",
            ),
            BadRequestError: (
                "BAD REQUEST (400)",
                "Invalid request. Check request parameters and model name.",
            ),
            APIConnectionError: (
                "CONNECTION ERROR",
                "Failed to connect. Check network connectivity and firewall rules.",
            ),
            APIStatusError: (
                f"STATUS ERROR ({status_code or 'unknown'})",
                "HTTP error from server.",
            ),
            APIError: (
                "ERROR",
                "API error occurred.",
            ),
        }

        # Find matching error type
        label, description = "ERROR", "Unexpected error occurred."
        for exc_type, (exc_label, exc_desc) in error_descriptions.items():
            if isinstance(e, exc_type):
                label, description = exc_label, exc_desc
                break
        else:
            # Also check for asyncio.TimeoutError and httpx.ConnectError
            if isinstance(e, asyncio.TimeoutError):
                label = "TIMEOUT"
                description = f"Request timed out after {timeout}s. Check if the deployment is responsive."
            elif isinstance(e, httpx.ConnectError):
                label = "CONNECTION ERROR"
                description = (
                    "Failed to connect. Check network connectivity and firewall rules."
                )

        log.error(
            f"LLM API {label}: {description} "
            f"Endpoint: {api_base}, Model: {model}. "
            f"Error: {error_type}: {error_message}"
        )


class AsyncLLMClient:
    """
    Async LLM client with token tracking using LiteLLM.

    Usage:
        from datarobot_genai.core.utils.token_tracking import (
            TokenUsageTracker, HeuristicTokenCountingStrategy
        )

        tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
        async with AsyncLLMClient(token_tracker=tracker) as client:
            result = await client.chat.completions.create(...)

        usage_info = TokenUsageInfo(**tracker.to_dict())

        # To use API response strategy:
        from datarobot_genai.core.utils.token_tracking import ApiResponseCountingStrategy

        tracker = TokenUsageTracker(strategy=ApiResponseCountingStrategy())
        async with AsyncLLMClient(token_tracker=tracker) as client:
            result = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        token_tracker: TokenUsageTracker | None = None,
    ):
        """
        Initialize AsyncLLMClient.

        Args:
            token_tracker: Optional token usage tracker
        """
        self.token_tracker = token_tracker
        self._config = Config()

        # Configure api_base for deployment-specific URLs if needed
        self._api_base = None
        if self._config.llm_deployment_id:
            self._api_base = (
                f"{self._config.datarobot_endpoint.rstrip('/')}/deployments/"
                f"{self._config.llm_deployment_id}/chat/completions"
            )

        # Create instructor client with LiteLLM
        self._instructor_client = instructor.from_litellm(
            litellm.acompletion,
            mode=instructor.Mode.MD_JSON,
        )

    async def __aenter__(self) -> InstructorClientWrapper:
        """Return wrapper that tracks tokens."""
        return InstructorClientWrapper(
            self._instructor_client, self.token_tracker, self._api_base
        )

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
