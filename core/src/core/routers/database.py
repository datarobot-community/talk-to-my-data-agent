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

"""Router for database endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends

from core.analyst_db import AnalystDB, InternalDataSourceType
from core.data_connections.database.database_implementations import (
    get_external_database,
)
from core.deps import get_initialized_db
from core.schema import AnalystDataset, LoadDatabaseRequest

router = APIRouter(prefix="/database", tags=["database"])


async def process_and_update(
    dataset_names: list[str],
    analyst_db: AnalystDB,
    datasource_type: str | InternalDataSourceType,
) -> None:
    """Process datasets and update state."""
    from core.api import process_data_and_update_state

    async for _ in process_data_and_update_state(
        dataset_names, analyst_db, datasource_type
    ):
        pass


async def get_and_process_tables(
    table_names: list[str], analyst_db: AnalystDB, sample_size: int
) -> None:
    """Get data from tables and process them."""
    dataset_names = await get_external_database().get_data(
        *table_names, analyst_db=analyst_db, sample_size=sample_size
    )
    await process_and_update(dataset_names, analyst_db, InternalDataSourceType.DATABASE)


@router.get("/tables")
async def get_database_tables() -> list[str]:
    """Get list of all available database tables."""
    return await get_external_database().get_tables()


@router.post("/select")
async def load_from_database(
    data: LoadDatabaseRequest,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    sample_size: int = 1_000,
) -> list[str]:
    """Load data from selected database tables."""
    await asyncio.gather(
        *[
            analyst_db.register_dataset(
                AnalystDataset(name=table), InternalDataSourceType.DATABASE
            )
            for table in data.table_names
        ]
    )

    if data.table_names:
        background_tasks.add_task(
            get_and_process_tables, data.table_names, analyst_db, sample_size
        )

    return data.table_names
