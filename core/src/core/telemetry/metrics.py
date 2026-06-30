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
"""Custom OTel metrics for the chat domain."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Generator

from opentelemetry import metrics

from core.telemetry.otel import otel

_METER_NAME = "chat"


@functools.cache
def _chat_requests_counter() -> metrics.Counter:
    return otel.get_meter(_METER_NAME).create_counter(
        "chat.requests",
        unit="{request}",
        description="Total chat completion requests",
    )


@functools.cache
def _chat_active_counter() -> metrics.UpDownCounter:
    return otel.get_meter(_METER_NAME).create_up_down_counter(
        "chat.active",
        unit="{request}",
        description="Currently in-flight chat requests",
    )


@functools.cache
def _chat_duration_histogram() -> metrics.Histogram:
    return otel.get_meter(_METER_NAME).create_histogram(
        "chat.response.duration",
        unit="s",
        description="End-to-end chat response duration",
    )


@contextmanager
def track_chat_request(model: str) -> Generator[None, None, None]:
    """Context manager that records chat request count, concurrency, and duration."""
    attrs = {"model": model}
    _chat_requests_counter().add(1, attrs)
    _chat_active_counter().add(1, attrs)
    start = time.monotonic()
    try:
        yield
    finally:
        duration = time.monotonic() - start
        _chat_duration_histogram().record(duration, attrs)
        _chat_active_counter().add(-1, attrs)
