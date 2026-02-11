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

"""Router for external data store endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from core.analyst_db import AnalystDB, ExternalDataStoreNameDataSourceType
from core.api import sync_data_sources_and_datasets
from core.data_connections.database.database_implementations import (
    get_external_database,
)
from core.data_connections.database.database_interface import NoDatabaseOperator
from core.data_connections.datarobot.datarobot_dataset_handler import (
    DatasetSparkRecipe,
    DataSourceRecipe,
)
from core.datarobot_client import use_user_token
from core.deps import get_initialized_db
from core.logging_helper import get_logger
from core.schema import (
    EmptyResponse,
    ExternalDataSourcesSelection,
    ExternalDataStore,
    SupportedDataSourceTypes,
)

logger = get_logger()


async def process_and_update(
    dataset_names: list[str],
    analyst_db: AnalystDB,
    datasource_type: str | ExternalDataStoreNameDataSourceType,
) -> None:
    """Process datasets and update state."""
    from core.api import process_data_and_update_state

    async for _ in process_data_and_update_state(
        dataset_names, analyst_db, datasource_type
    ):
        pass


async def update_data_sources_for_data_store(
    request: Request,
    selected_datasource_ids: ExternalDataSourcesSelection,
    external_data_store_id: str,
    analyst_db: AnalystDB,
) -> None:
    """Update data sources for a data store."""
    with use_user_token(request):
        logger.debug("Loading recipe for datasources %s.")
        recipe = await DataSourceRecipe.load_or_create(
            analyst_db, external_data_store_id
        )
        logger.debug("Updating recipe with data sources for %s", external_data_store_id)
        await recipe.select_data_sources(selected_datasource_ids.selected_data_sources)


external_data_stores_router = APIRouter(
    prefix="/external-data-stores", tags=["external-data-stores"]
)


# Sync as this is a long blocking request
@external_data_stores_router.get("/available/")
def get_available_external_data_stores(
    request: Request, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> list[ExternalDataStore]:
    """List all available datastores.

    An available datastore (a) has datasources configured and (b) has a supported driver.

    Args:
        request (Request): HTTP request.

    Returns:
        list[ExternalDataStore]: The given page of datastores.
    """
    with use_user_token(request):
        return asyncio.run(
            DataSourceRecipe.list_available_datastores(analyst_db.user_id)
        )


@external_data_stores_router.put(
    "/{external_data_store_id}/external-data-sources/",
)
async def register_external_data_sources(
    request: Request,
    selected_datasource_ids: ExternalDataSourcesSelection,
    external_data_store_id: str,
    background_tasks: BackgroundTasks,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> EmptyResponse:
    """
    Add data sources for a datastore, registering them as datasets in the app.

    Args:
        request (Request): The request
        selected_datasource_ids (ExternalDataSourcesSelection): Select data sources for a datasource
        external_data_store_id (str): Select the data store
        background_tasks (BackgroundTasks): Background tasks
        analyst_db (AnalystDB): The database

    Returns:
        EmptyResponse: Empty response
    """
    logger.debug(
        "PUT /external-data-stores/%s/external-data-sources/",
        external_data_store_id,
    )
    with use_user_token(request):
        name = await DataSourceRecipe.get_canonical_name_for_datastore_id(
            external_data_store_id
        )
        if not name:
            raise HTTPException(
                404, f"DataStore {external_data_store_id} does not exist."
            )

        background_tasks.add_task(
            update_data_sources_for_data_store,
            request,
            selected_datasource_ids,
            external_data_store_id,
            analyst_db,
        )

        logger.debug(
            "Registering data sources for data store %s (%s).",
            name,
            external_data_store_id,
        )
        _, tasks = await sync_data_sources_and_datasets(
            request=request,
            canonical_name=name,
            analyst_db=analyst_db,
            data_store_id=external_data_store_id,
            selected_datasource_ids=selected_datasource_ids,
        )

        logger.debug(
            "Adding background tasks for data store %s.", external_data_store_id
        )
        for f, arg, kwargs in tasks:
            background_tasks.add_task(f, *arg, **kwargs)
        background_tasks.add_task(
            process_and_update,
            [d.path for d in selected_datasource_ids.selected_data_sources],
            analyst_db,
            ExternalDataStoreNameDataSourceType.from_name(name),
        )
    return EmptyResponse()


supported_types_router = APIRouter(tags=["data-sources"])


@supported_types_router.get("/supported-data-source-types")
async def get_supported_datasource_types(
    request: Request,
) -> SupportedDataSourceTypes:
    """Get list of supported data source types."""
    from core.analyst_db import InternalDataSourceType

    types: list[InternalDataSourceType] = [
        InternalDataSourceType.FILE,
        InternalDataSourceType.REGISTRY,
    ]
    if not isinstance(get_external_database(), NoDatabaseOperator):
        types.append(InternalDataSourceType.DATABASE)
    if DatasetSparkRecipe.should_use_spark_recipe():
        types.append(InternalDataSourceType.REMOTE_REGISTRY)
    return SupportedDataSourceTypes(supported_types=[t.value for t in types])
