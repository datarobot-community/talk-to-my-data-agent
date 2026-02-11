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

import logging
from types import TracebackType
from typing import Any, Type

import instructor
import litellm

from core.config import Config
from core.token_tracking import TokenUsageTracker

log = logging.getLogger(__name__)


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
    """Wrapper for completions interface with token tracking."""

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

        # Call underlying implementation
        result = await self._completions.create(*args, **kwargs)

        # Track tokens if tracker is available
        if self._tracker:
            self._tracker.track_call(messages, result, model)

        return result


class AsyncLLMClient:
    """
    Async LLM client with token tracking using LiteLLM.

    Usage:
        from core.token_tracking import TokenUsageTracker, TiktokenCountingStrategy

        tracker = TokenUsageTracker(strategy=TiktokenCountingStrategy())
        async with AsyncLLMClient(token_tracker=tracker) as client:
            result = await client.chat.completions.create(...)

        usage_info = TokenUsageInfo(**tracker.to_dict())

        # To use API response strategy:
        from core.token_tracking import ApiResponseCountingStrategy

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
