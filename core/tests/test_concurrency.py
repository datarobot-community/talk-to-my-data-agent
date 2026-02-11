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

import asyncio
import logging
import os
import time
from typing import Final
from uuid import uuid4

import httpx
import pytest

from core.analyst_db import ExternalDataStoreNameDataSourceType
from core.schema import (
    ExternalDataSourcesSelection,
    ExternalDataStore,
)

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG


def client(endpoint: str, headers: dict[str, str]) -> httpx.AsyncClient:
    endpoint = endpoint.strip("/") + "/"
    return httpx.AsyncClient(
        base_url=endpoint, timeout=httpx.Timeout(900), headers=headers
    )


async def upload_datasources(
    client: httpx.AsyncClient, datastore_id: str, tables: list[str]
) -> tuple[httpx.AsyncClient, ExternalDataStore]:
    ds_response = await client.get("external-data-stores/available/")
    logger.debug("DS Response %s", ds_response)
    while ds_response.is_server_error:
        await asyncio.sleep(10)
        ds_response = await client.get("external-data-stores/available/")

    logging.debug("Data Stores %s", ds_response.json())

    available_data_stores = [ExternalDataStore(**ds) for ds in ds_response.json()]

    selected_data_store = next(
        (ds for ds in available_data_stores if ds.id == datastore_id), None
    )

    logging.debug("Selected data store %s", selected_data_store)

    assert selected_data_store, "Data Store should exist."

    available_paths = {ds.path for ds in selected_data_store.defined_data_sources}

    for path in tables:
        assert path in available_paths

    datasources = [
        ds for ds in selected_data_store.defined_data_sources if ds.path in tables
    ]

    dictionary_response = await client.get("dictionaries")
    dictionary_names = {d["name"]: d["in_progress"] for d in dictionary_response.json()}
    if all(ds.path in dictionary_names for ds in datasources):
        return client, selected_data_store

    await client.put(
        f"external-data-stores/{datastore_id}/external-data-sources/",
        json=ExternalDataSourcesSelection(
            selected_data_sources=datasources
        ).model_dump(),
    )

    start_query = time.time()
    while True:
        dictionary_response = await client.get("dictionaries")
        dictionary_names = {
            d["name"]: d["in_progress"] for d in dictionary_response.json()
        }
        if all(dictionary_names.get(ds.path) == False for ds in datasources):  # noqa:E712
            break
        elif (time.time() - start_query) > 900:
            raise RuntimeError("Took more than fifteen minutes to upload datasets.")
        else:
            await asyncio.sleep(10)

    return client, selected_data_store


async def upload_dataset_for_endpoint(
    endpoint: str, datastore_id: str, tables: list[str]
) -> tuple[httpx.AsyncClient, ExternalDataStore]:
    user_header = {"X-USER-EMAIL": f"{uuid4()}@example.com"}

    c = client(endpoint, headers=user_header)

    return await upload_datasources(c, datastore_id, tables)


async def run_query_for_period(
    client: httpx.AsyncClient,
    datastore: ExternalDataStore,
    wait_timeout: int = 120,
) -> None:
    c = client

    chat_response = await c.post(
        "/chats/messages",
        json={
            "message": "How many rows are there?",
            "data_source": ExternalDataStoreNameDataSourceType.from_name(
                datastore.canonical_name
            ).name,
            "enable_business_insights": False,
            "enable_chart_generation": False,
        },
    )
    chat_id = chat_response.json()["id"]

    complete = False
    response_start = time.time()
    time_elapsed = 0.0
    while not complete and time_elapsed < wait_timeout:
        await asyncio.sleep(1)
        chat_get = await c.get(f"/chats/{chat_id}")
        messages = chat_get.json()["messages"]
        complete = len(messages) >= 2 and (
            not messages[-1]["in_progress"] or messages[-1].get("error")
        )
        if complete:
            assert not messages[-1].get("error")
        time_elapsed = time.time() - response_start
    if time_elapsed >= wait_timeout:
        logger.info(f"{time_elapsed:.02f}")
    else:
        logger.info(f"{time_elapsed:.02f}")

    await c.delete(f"/chats/{chat_id}")


async def run_queries_for_period(
    endpoint: str,
    concurrency: int,
    timeout: int,
    datastore_id: str,
    tables: list[str],
) -> None:
    datasources = await asyncio.gather(
        *[
            upload_dataset_for_endpoint(endpoint, datastore_id, tables)
            for _ in range(concurrency)
        ]
    )
    await asyncio.gather(
        *[run_query_for_period(client, ds, timeout) for client, ds in datasources]
    )


REMOTE_TEST_DATA_SOURCE_ID: Final[str] = "REMOTE_TEST_DATA_SOURCE_ID"
REMOTE_TEST_DATA_SOURCE_TABLES: Final[str] = "REMOTE_TEST_DATA_SOURCE_TABLES"
REMOTE_TEST_DATA_SOURCE_QUERY: Final[str] = "REMOTE_TEST_DATA_SOURCE_QUERY"
REMOTE_TEST_DATA_SOURCE_CONCURRENCY: Final[str] = (
    "REMOTE_TEST_DATA_SOURCE_QUERY_CONCURRENCY"
)


@pytest.mark.remote_datasource_concurrency
@pytest.mark.asyncio
async def test_concurrency_remote_redshift() -> None:
    """
    A simple integration/concurrency request. This requires more setup to run.

    1. Set up a Redshift (or Postgres) data source for your user in the DR platform. Capture its ID.
    2. Run the application locally.
    3. Run this test as follows. (Adjusting for your data source)

        export REMOTE_TEST_DATASOURCE_ID="68dfdf14cadb94a915ff8dd0"
        export REMOTE_TEST_DATA_SOURCE_TABLES="dev.public.users"
        export REMOTE_TEST_DATA_SOURCE_QUERY="What percentage of users like Las Vegas"
        export REMOTE_TEST_DATA_SOURCE_CONCURRENCY=5
        . ./set_env.sh && pytest -m remote_datasource_concurrency tests
    """
    datasource = os.environ.get("REMOTE_TEST_DATASOURCE_ID", "")
    tables_csv = os.environ.get("REMOTE_TEST_DATA_SOURCE_TABLES", "")
    query = os.environ.get("REMOTE_TEST_DATA_SOURCE_QUERY", "How many rows are there?")
    concurrency = int(os.environ.get("REMOTE_TEST_DATA_SOURCE_CONCURRENCY", 5))
    endpoint = "http://localhost:8080/api/v1"

    if not all([datasource, tables_csv, query, concurrency]):
        logger.warning("Skipping as prerequisites are not fulfilled")
        return None

    tables = [t.strip() for t in tables_csv.split(",")]

    await run_queries_for_period(
        endpoint=endpoint,
        concurrency=concurrency,
        timeout=300,
        datastore_id=datasource,
        tables=tables,
    )
