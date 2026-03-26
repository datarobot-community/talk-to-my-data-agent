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

from pathlib import Path

import pandas as pd
import pytest

from core.analyst_db import (
    ANALYST_DATABASE_VERSION,
    AnalystDB,
    InternalDataSourceType,
)
from core.schema import (
    AnalystDataset,
)


async def get_analyst_db(
    db_version: int | None = ANALYST_DATABASE_VERSION,
    user_id: str = "test_user_123",
) -> AnalystDB:
    """Get a test database instance."""
    analyst_db = await AnalystDB.create(
        user_id=user_id,
        db_path=Path("."),
        dataset_db_name="datasets",
        chat_db_name="chats",
        db_version=db_version,
    )
    return analyst_db


@pytest.mark.asyncio
async def test_drop_tables(url_diabetes: str, dataset_loaded: AnalystDataset) -> None:
    assert dataset_loaded.columns is not None

    df = pd.read_csv(url_diabetes)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    # Replace NaN with None
    df = df.where(pd.notnull(df), None)  # type:ignore[call-overload]

    # Create dataset dictionary
    dataset = AnalystDataset(
        name="new_dataset_name",
        data=df,
    )
    new_db = await get_analyst_db(ANALYST_DATABASE_VERSION + 1)
    await new_db.register_dataset(dataset, data_source=InternalDataSourceType.FILE)

    assert len(await new_db.list_analyst_datasets()) == 1


@pytest.mark.asyncio
async def test_data_source_type_saving() -> None:
    """Test that the DataSourceType is correctly saved to the database."""
    # Create a small test dataframe
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

    # Create test datasets for different data sources
    file_dataset = AnalystDataset(name="file_dataset", data=df)
    database_dataset = AnalystDataset(name="database_dataset", data=df)
    registry_dataset = AnalystDataset(name="registry_dataset", data=df)

    # Create a new database instance
    db = await get_analyst_db(ANALYST_DATABASE_VERSION + 2)

    await db.delete_all_tables()

    # Register datasets with different data sources
    await db.register_dataset(file_dataset, data_source=InternalDataSourceType.FILE)
    await db.register_dataset(
        database_dataset, data_source=InternalDataSourceType.DATABASE
    )
    await db.register_dataset(
        registry_dataset, data_source=InternalDataSourceType.REGISTRY
    )

    # Get all datasets
    all_datasets = await db.dataset_handler.list_datasets()

    # Check that we have 3 datasets
    assert len(all_datasets) == 3

    # Create a map of dataset names to their data sources
    dataset_sources = {dataset.name: dataset.data_source for dataset in all_datasets}

    # Verify each dataset has the correct data source
    assert dataset_sources["file_dataset"] == InternalDataSourceType.FILE
    assert dataset_sources["database_dataset"] == InternalDataSourceType.DATABASE
    assert dataset_sources["registry_dataset"] == InternalDataSourceType.REGISTRY

    # Test filtering by data source
    file_datasets = await db.dataset_handler.list_datasets(
        data_source=InternalDataSourceType.FILE
    )
    database_datasets = await db.dataset_handler.list_datasets(
        data_source=InternalDataSourceType.DATABASE
    )
    registry_datasets = await db.dataset_handler.list_datasets(
        data_source=InternalDataSourceType.REGISTRY
    )

    # Verify filtering works correctly
    assert len(file_datasets) == 1
    assert len(database_datasets) == 1
    assert len(registry_datasets) == 1

    assert file_datasets[0].name == "file_dataset"
    assert database_datasets[0].name == "database_dataset"
    assert registry_datasets[0].name == "registry_dataset"

    # Test the list_analyst_datasets method with data source filtering
    file_names = await db.list_analyst_datasets(data_source=InternalDataSourceType.FILE)
    db_names = await db.list_analyst_datasets(
        data_source=InternalDataSourceType.DATABASE
    )
    registry_names = await db.list_analyst_datasets(
        data_source=InternalDataSourceType.REGISTRY
    )

    # Verify filtering works correctly at the higher level API
    assert len(file_names) == 1
    assert len(db_names) == 1
    assert len(registry_names) == 1

    assert "file_dataset" in file_names
    assert "database_dataset" in db_names
    assert "registry_dataset" in registry_names
