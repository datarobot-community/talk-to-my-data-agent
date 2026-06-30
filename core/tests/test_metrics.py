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
"""Tests for chat metrics instrumentation."""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from core.telemetry.metrics import track_chat_request


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """Clear functools.cache between tests so mocks take effect."""
    from core.telemetry import metrics as metrics_mod

    metrics_mod._chat_requests_counter.cache_clear()
    metrics_mod._chat_active_counter.cache_clear()
    metrics_mod._chat_duration_histogram.cache_clear()
    yield
    metrics_mod._chat_requests_counter.cache_clear()
    metrics_mod._chat_active_counter.cache_clear()
    metrics_mod._chat_duration_histogram.cache_clear()


@pytest.fixture
def mock_meter() -> Iterator[dict[str, MagicMock]]:
    counter = MagicMock()
    up_down_counter = MagicMock()
    histogram = MagicMock()

    meter = MagicMock()
    meter.create_counter.return_value = counter
    meter.create_up_down_counter.return_value = up_down_counter
    meter.create_histogram.return_value = histogram

    with patch("core.telemetry.metrics.otel") as mock_otel:
        mock_otel.get_meter.return_value = meter
        yield {
            "counter": counter,
            "up_down_counter": up_down_counter,
            "histogram": histogram,
        }


def test_track_chat_request_increments_counter(mock_meter: dict[str, MagicMock]) -> None:
    with track_chat_request(model="test-model"):
        pass

    mock_meter["counter"].add.assert_called_once_with(1, {"model": "test-model"})


def test_track_chat_request_tracks_active(mock_meter: dict[str, MagicMock]) -> None:
    with track_chat_request(model="test-model"):
        mock_meter["up_down_counter"].add.assert_called_with(1, {"model": "test-model"})

    calls = mock_meter["up_down_counter"].add.call_args_list
    assert len(calls) == 2
    assert calls[0].args == (1, {"model": "test-model"})
    assert calls[1].args == (-1, {"model": "test-model"})


def test_track_chat_request_records_duration(mock_meter: dict[str, MagicMock]) -> None:
    with track_chat_request(model="test-model"):
        pass

    mock_meter["histogram"].record.assert_called_once()
    args = mock_meter["histogram"].record.call_args
    duration = args[0][0]
    assert isinstance(duration, float)
    assert duration >= 0
    assert args[0][1] == {"model": "test-model"}


def test_track_chat_request_records_on_error(mock_meter: dict[str, MagicMock]) -> None:
    with pytest.raises(ValueError, match="boom"):
        with track_chat_request(model="err-model"):
            raise ValueError("boom")

    mock_meter["counter"].add.assert_called_once()
    mock_meter["histogram"].record.assert_called_once()
    calls = mock_meter["up_down_counter"].add.call_args_list
    assert calls[-1].args == (-1, {"model": "err-model"})
