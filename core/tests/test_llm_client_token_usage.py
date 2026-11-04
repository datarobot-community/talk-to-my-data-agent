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
"""Tests for the gen_ai.client.token.usage metric."""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import Histogram, InMemoryMetricReader, Metric

from core import llm_client

METRIC_NAME = "gen_ai.client.token.usage"
MODEL = "datarobot/bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"


@pytest.fixture
def reader() -> Iterator[InMemoryMetricReader]:
    """Route the metric through a real SDK reader so the exported data type is observable."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    llm_client._token_usage_histogram.cache_clear()
    with patch.object(
        llm_client.metrics, "get_meter", return_value=provider.get_meter("gen_ai")
    ):
        yield reader
    llm_client._token_usage_histogram.cache_clear()


async def _complete(usage: SimpleNamespace | None) -> None:
    completions = AsyncMock()
    completions.create.return_value = SimpleNamespace(usage=usage)
    wrapper = llm_client.CompletionsWrapper(completions, tracker=None)
    await wrapper.create(model=MODEL, messages=[{"role": "user", "content": "hi"}])


def _exported_metric(reader: InMemoryMetricReader) -> Metric | None:
    data = reader.get_metrics_data()
    metrics = [
        metric
        for resource_metrics in (data.resource_metrics if data else [])
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
        if metric.name == METRIC_NAME
    ]
    return metrics[0] if metrics else None


@pytest.mark.asyncio
async def test_token_usage_is_exported_as_a_histogram(
    reader: InMemoryMetricReader,
) -> None:
    await _complete(SimpleNamespace(prompt_tokens=1130, completion_tokens=35))

    metric = _exported_metric(reader)

    assert metric is not None
    assert isinstance(metric.data, Histogram)


@pytest.mark.asyncio
async def test_token_usage_records_input_and_output_tokens_separately(
    reader: InMemoryMetricReader,
) -> None:
    await _complete(SimpleNamespace(prompt_tokens=1130, completion_tokens=35))

    metric = _exported_metric(reader)

    assert metric is not None
    assert isinstance(metric.data, Histogram)
    sums_by_type = {
        (point.attributes or {})["gen_ai.token.type"]: point.sum
        for point in metric.data.data_points
    }
    assert sums_by_type == {"input": 1130, "output": 35}


@pytest.mark.asyncio
async def test_token_usage_records_nothing_without_usage_on_the_response(
    reader: InMemoryMetricReader,
) -> None:
    await _complete(None)

    assert _exported_metric(reader) is None


@pytest.mark.asyncio
async def test_token_usage_skips_a_count_the_response_does_not_carry(
    reader: InMemoryMetricReader,
) -> None:
    await _complete(SimpleNamespace(prompt_tokens=None, completion_tokens=35))

    metric = _exported_metric(reader)

    assert metric is not None
    assert [
        (p.attributes or {})["gen_ai.token.type"] for p in metric.data.data_points
    ] == ["output"]
