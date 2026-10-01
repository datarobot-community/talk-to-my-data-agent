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

"""Tests for the Data Registry listing (AECO-44).

`Dataset.iterate`'s `limit` is a page size, not a cap, so the previous
implementation walked the whole AI Catalog on every call — and the frontend made
two such calls at once. These tests pin the behaviour that replaced it: one
lazy walk, capped, abandoned as soon as both listings are full.
"""

from datetime import datetime
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

from core.api import list_registry_datasets
from core.constants import REGISTRY_DATASET_SIZE_CUTOFF

SMALL = int(REGISTRY_DATASET_SIZE_CUTOFF // 2)
LARGE = int(REGISTRY_DATASET_SIZE_CUTOFF * 2)


def _dataset(
    dataset_id: str,
    *,
    size: int | None = SMALL,
    is_snapshot: bool = True,
    is_data_engine_eligible: bool = True,
) -> MagicMock:
    ds = MagicMock()
    ds.id = dataset_id
    ds.name = f"dataset-{dataset_id}"
    ds.created_at = datetime(2026, 1, 2)
    ds.size = size
    ds.is_snapshot = is_snapshot
    ds.is_data_engine_eligible = is_data_engine_eligible
    return ds


def test_splits_local_and_remote_from_one_walk() -> None:
    datasets = [
        _dataset("small", size=SMALL, is_data_engine_eligible=False),
        _dataset("large", size=LARGE),
        _dataset("not-a-snapshot", is_snapshot=False),
    ]

    with patch("core.api.Dataset.iterate", return_value=iter(datasets)) as iterate:
        result = list_registry_datasets(limit=10)

    assert [d.id for d in result.local] == ["small"]
    assert [d.id for d in result.remote] == ["large"]
    # One walk, not one per listing.
    assert iterate.call_count == 1


def test_small_engine_eligible_dataset_appears_in_both_listings() -> None:
    """A small snapshot that is also engine-eligible can be downloaded *or*
    wrangled, and showed up in both listings before they were fetched together.
    """
    with patch("core.api.Dataset.iterate", return_value=iter([_dataset("both")])):
        result = list_registry_datasets(limit=10)

    assert [d.id for d in result.local] == ["both"]
    assert [d.id for d in result.remote] == ["both"]


def test_limit_caps_each_listing() -> None:
    datasets = [_dataset(str(i)) for i in range(50)]

    with patch("core.api.Dataset.iterate", return_value=iter(datasets)):
        result = list_registry_datasets(limit=5)

    assert len(result.local) == 5
    assert len(result.remote) == 5


def test_stops_walking_once_both_listings_are_full() -> None:
    """The regression that made this slow: the catalog was walked to the end
    however few datasets the caller asked for. Each page is one AI Catalog
    search, and the platform serialises those per user.
    """
    consumed = 0

    def counting_iterator() -> Iterator[Any]:
        nonlocal consumed
        for i in range(10_000):
            consumed += 1
            yield _dataset(str(i))

    with patch("core.api.Dataset.iterate", return_value=counting_iterator()):
        result = list_registry_datasets(limit=3)

    assert len(result.local) == 3
    assert len(result.remote) == 3
    # One extra item is pulled before the loop sees both listings are full.
    assert consumed <= 4


def test_formats_size_and_created_date() -> None:
    ds = _dataset("formatted", size=5 * 1024 * 1024)

    with patch("core.api.Dataset.iterate", return_value=iter([ds])):
        result = list_registry_datasets(limit=10)

    assert result.local[0].size == "5.0 MB"
    assert result.local[0].created == "2026-01-02"


def test_missing_size_is_not_downloadable() -> None:
    ds = _dataset("unsized", size=None)

    with patch("core.api.Dataset.iterate", return_value=iter([ds])):
        result = list_registry_datasets(limit=10)

    assert result.local == []
    assert [d.id for d in result.remote] == ["unsized"]
    assert result.remote[0].size == "N/A"
