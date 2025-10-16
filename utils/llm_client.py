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

from types import TracebackType
from typing import Any, Type

import instructor
from openai import AsyncOpenAI

from utils.token_tracking import TokenUsageTracker


class TokenTrackingProxy:
    """
    Proxy that intercepts instructor client calls and tracks tokens.
    """

    def __init__(
        self,
        client: instructor.AsyncInstructor,
        tracker: TokenUsageTracker | None = None,
    ):
        self._client = client
        self._tracker = tracker

    @property
    def chat(self) -> ChatProxy:
        """Return wrapped chat interface."""
        return ChatProxy(self._client.chat, self._tracker)

    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to underlying client."""
        return getattr(self._client, name)


class ChatProxy:
    """Proxy for chat interface."""

    def __init__(self, chat: Any, tracker: TokenUsageTracker | None):
        self._chat = chat
        self._tracker = tracker

    @property
    def completions(self) -> CompletionsProxy:
        """Return wrapped completions interface."""
        return CompletionsProxy(self._chat.completions, self._tracker)


class CompletionsProxy:
    """Proxy for completions interface with token tracking."""

    def __init__(self, completions: Any, tracker: TokenUsageTracker | None):
        self._completions = completions
        self._tracker = tracker

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        """Intercept create calls to track tokens."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "unknown")

        # Call underlying implementation
        result = await self._completions.create(*args, **kwargs)

        # Track tokens if tracker is available
        if self._tracker:
            self._tracker.track_call(messages, result, model)

        return result


class AsyncLLMClient:
    """
    Async LLM client with token tracking.

    Usage:
        from utils.token_tracking import TokenUsageTracker, TiktokenCountingStrategy

        tracker = TokenUsageTracker(strategy=TiktokenCountingStrategy())
        async with AsyncLLMClient(token_tracker=tracker) as client:
            result = await client.chat.completions.create(...)

        usage_info = TokenUsageInfo(**tracker.to_dict())

        # To use API response strategy:
        from utils.token_tracking import ApiResponseCountingStrategy

        tracker = TokenUsageTracker(strategy=ApiResponseCountingStrategy())
        async with AsyncLLMClient(token_tracker=tracker) as client:
            result = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        token_tracker: TokenUsageTracker | None = None,
        dr_client: Any | None = None,
        deployment_base_url: str | None = None,
    ):
        """
        Initialize AsyncLLMClient.

        Args:
            token_tracker: Optional token usage tracker
            dr_client: Optional DataRobot client (will be initialized if not provided)
            deployment_base_url: Optional deployment URL (will be initialized if not provided)
        """
        self.token_tracker = token_tracker
        self._dr_client = dr_client
        self._deployment_base_url = deployment_base_url
        self._openai_client: AsyncOpenAI | None = None
        self._instructor_client: instructor.AsyncInstructor | None = None

    async def __aenter__(self) -> TokenTrackingProxy:
        """Initialize clients on context entry."""
        # Import here to avoid circular imports
        from utils.api import initialize_deployment

        # Initialize deployment if not provided
        if self._dr_client is None or self._deployment_base_url is None:
            dr_client, deployment_base_url = initialize_deployment()
            self._dr_client = dr_client
            self._deployment_base_url = deployment_base_url

        # Create OpenAI client
        self._openai_client = AsyncOpenAI(
            api_key=self._dr_client.token,
            base_url=self._deployment_base_url,
            timeout=90,
            max_retries=2,
        )

        # Create instructor client
        self._instructor_client = instructor.from_openai(
            self._openai_client, mode=instructor.Mode.MD_JSON
        )

        # Return proxy that tracks tokens
        return TokenTrackingProxy(self._instructor_client, self.token_tracker)

    async def __aexit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up clients on context exit."""
        if self._openai_client:
            await self._openai_client.close()
