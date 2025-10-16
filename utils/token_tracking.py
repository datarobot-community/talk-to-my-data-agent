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

from typing import Any, Protocol, runtime_checkable

import tiktoken
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from utils.constants import (
    ALTERNATIVE_LLM_BIG,
    DEFAULT_TIKTOKEN_ENCODING,
    TIKTOKEN_ENCODING_MAP,
)
from utils.logging_helper import get_logger

logger = get_logger()


def count_tokens_tiktoken(text: str, model: str = ALTERNATIVE_LLM_BIG) -> int:
    """Count tokens in text using tiktoken encoding for the specified model.

    Args:
        text: Text to count tokens for
        model: Model name for encoding selection

    Returns:
        Number of tokens in the text
    """
    try:
        encoding_name = TIKTOKEN_ENCODING_MAP.get(model, DEFAULT_TIKTOKEN_ENCODING)
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(
            f"Error counting tokens with tiktoken: {e}. Using fallback estimation."
        )
        # Fallback: roughly 4 characters per token
        return len(text) // 4


def estimate_csv_rows_for_token_limit(
    df: Any,  # pandas.DataFrame
    max_tokens: int,
    initial_rows: int,
    model: str = ALTERNATIVE_LLM_BIG,
) -> tuple[str, int]:
    """
    Estimate the optimal number of rows for CSV data to fit within token limit.

    Args:
        df: pandas DataFrame to convert to CSV
        max_tokens: Maximum allowed tokens for the CSV data
        initial_rows: Initial number of rows to try
        model: Model name for token counting

    Returns:
        tuple: (csv_string, final_token_count)
    """
    df_csv = df.head(initial_rows).to_csv(index=False, quoting=1)
    csv_token_count = count_tokens_tiktoken(df_csv, model)

    if csv_token_count <= max_tokens:
        return df_csv, csv_token_count

    logger.warning(
        f"CSV data has {csv_token_count} tokens, exceeds limit of {max_tokens}. Reducing rows."
    )

    ratio = max_tokens / csv_token_count
    estimated_rows = int(initial_rows * ratio * 0.9)
    estimated_rows = max(100, estimated_rows)

    df_csv = df.head(estimated_rows).to_csv(index=False, quoting=1)
    final_token_count = count_tokens_tiktoken(df_csv, model)

    if final_token_count > max_tokens:
        estimated_rows = int(estimated_rows * 0.8)
        df_csv = df.head(estimated_rows).to_csv(index=False, quoting=1)
        final_token_count = count_tokens_tiktoken(df_csv, model)

    logger.info(
        f"Reduced CSV to {estimated_rows} rows ({final_token_count} tokens) to fit within context window."
    )
    return df_csv, final_token_count


@runtime_checkable
class TokenCountingStrategy(Protocol):
    """Protocol for token counting strategies."""

    def count_tokens(
        self,
        messages: list[ChatCompletionMessageParam],
        response: Any,
        model: str,
    ) -> tuple[int, int]:
        """
        Count prompt and completion tokens.

        Args:
            messages: Input messages sent to LLM
            response: Response from LLM
            model: Model name

        Returns:
            Tuple of (prompt_tokens, completion_tokens)
        """
        ...


class TiktokenCountingStrategy:
    """Token counting using tiktoken library."""

    def __init__(self) -> None:
        self._encodings: dict[str, tiktoken.Encoding] = {}

    def _get_encoding(self, model: str) -> tiktoken.Encoding:
        """Get or cache encoding for model."""
        encoding_name = TIKTOKEN_ENCODING_MAP.get(model, DEFAULT_TIKTOKEN_ENCODING)
        if encoding_name not in self._encodings:
            try:
                self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
                self._encodings[encoding_name] = tiktoken.get_encoding(
                    DEFAULT_TIKTOKEN_ENCODING
                )
        return self._encodings[encoding_name]

    def _count_text(self, text: str, model: str) -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = self._get_encoding(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(
                f"Error counting tokens with tiktoken: {e}. Using fallback estimation."
            )
            # Fallback: roughly 4 characters per token
            return len(text) // 4

    def _count_messages(
        self, messages: list[ChatCompletionMessageParam], model: str
    ) -> int:
        """Count tokens in messages."""
        total_tokens = 0
        for msg in messages:
            # Extract content based on message type
            role: str = ""
            content: str = ""

            if hasattr(msg, "get"):  # Dict-like
                role = str(msg.get("role", ""))
                content = str(msg.get("content", ""))
            else:
                # TypedDict attributes
                role = str(getattr(msg, "role", ""))
                content = str(getattr(msg, "content", ""))

            total_tokens += self._count_text(role, model)
            total_tokens += self._count_text(content, model)
            total_tokens += 4  # Message structure overhead

        return total_tokens

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from various response formats."""
        if hasattr(response, "content") and response.content:
            return str(response.content)
        if hasattr(response, "model_dump_json"):
            return str(response.model_dump_json())
        if hasattr(response, "__dict__"):
            return str(response.__dict__)
        return str(response)

    def count_tokens(
        self,
        messages: list[ChatCompletionMessageParam],
        response: Any,
        model: str,
    ) -> tuple[int, int]:
        """Count tokens using tiktoken."""
        prompt_tokens = self._count_messages(messages, model)
        response_text = self._extract_response_text(response)
        completion_tokens = self._count_text(response_text, model)

        return prompt_tokens, completion_tokens


class ApiResponseCountingStrategy:
    """Token counting from OpenAI API response (when available/correct)."""

    def __init__(self, fallback_strategy: TokenCountingStrategy | None = None) -> None:
        """
        Initialize with optional fallback strategy.

        Args:
            fallback_strategy: Strategy to use if API response doesn't have usage data
        """
        if fallback_strategy is None:
            fallback_strategy = TiktokenCountingStrategy()
        self.fallback_strategy: TokenCountingStrategy = fallback_strategy

    def count_tokens(
        self,
        messages: list[ChatCompletionMessageParam],
        response: Any,
        model: str,
    ) -> tuple[int, int]:
        """Extract token counts from API response."""
        # Try to get usage from response
        usage = self._extract_usage(response)

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

            if prompt_tokens > 0 and completion_tokens > 0:
                logger.debug(
                    f"Using API response token counts: {prompt_tokens} prompt, "
                    f"{completion_tokens} completion"
                )
                return prompt_tokens, completion_tokens

        # Fallback to alternative strategy
        logger.debug("API usage data not available, using fallback strategy")
        return self.fallback_strategy.count_tokens(messages, response, model)

    @staticmethod
    def _extract_usage(response: Any) -> Any | None:
        """Extract usage data from various response formats."""
        # Try instructor response format
        if hasattr(response, "_raw_response"):
            raw = response._raw_response
            if hasattr(raw, "usage"):
                return raw.usage

        # Try direct usage attribute
        if hasattr(response, "usage"):
            return response.usage

        # Try dict format
        if isinstance(response, dict) and "usage" in response:
            return response["usage"]

        return None


class TokenUsageTracker:
    """
    Accumulates token usage across multiple LLM calls.

    Compatible with existing TokenUsageInfo schema.
    """

    def __init__(self, strategy: TokenCountingStrategy):
        """
        Initialize tracker with counting strategy.

        Args:
            strategy: Token counting strategy (required).

        Example:
            tracker = TokenUsageTracker(strategy=TiktokenCountingStrategy())
            # or
            tracker = TokenUsageTracker(strategy=ApiResponseCountingStrategy())
        """
        self.strategy = strategy
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.model = ""

    def track_call(
        self,
        messages: list[ChatCompletionMessageParam],
        response: Any,
        model: str,
    ) -> None:
        """
        Track token usage from an LLM call.

        Args:
            messages: Input messages
            response: LLM response
            model: Model name
        """

        prompt_tokens, completion_tokens = self.strategy.count_tokens(
            messages, response, model
        )

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
        if model:
            self.model = model

        logger.info(
            f"Token tracker: +{prompt_tokens} prompt, +{completion_tokens} completion "
            f"(total calls: {self.call_count}, total tokens: {self.total_tokens})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TokenUsageInfo."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "model": self.model,
        }
