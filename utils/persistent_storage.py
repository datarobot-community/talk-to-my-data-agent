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

import json
import logging
import os
import time
from typing import Any, AsyncGenerator, List, Optional, Union, cast
from urllib.parse import urljoin

import aiofiles
import datarobot as dr
import httpx
from typing_extensions import ParamSpec, TypeVar

from utils.data_analyst_telemetry import telemetry
from utils.datarobot_client import (
    FILE_API_CONNECT_TIMEOUT,
    FILE_API_READ_TIMEOUT,
    File,
    KeyValue,
    KeyValueEntityType,
)

Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)

# httpx logs requests at INFO, which gets quite noisy.
logging.getLogger("httpx").setLevel(logging.WARNING)


class AsyncDataRobotClient:
    def __init__(self, token: str, endpoint: str):
        normalized_endpoint = (endpoint or "").rstrip("/") + "/"
        self._client = httpx.AsyncClient(
            base_url=normalized_endpoint, headers={"Authorization": f"Bearer {token}"}
        )

    async def unpaginate(
        self, initial_url: str, initial_params: dict[str, Any] | None = None
    ) -> AsyncGenerator[Any, None]:
        r = await self._client.get(initial_url, params=initial_params)
        resp_data = r.json()
        for row in resp_data["data"]:
            yield row
        while resp_data.get("next") is not None:
            next_url = resp_data["next"]
            r = await self._client.get(next_url, params=initial_params)
            resp_data = r.json()
            for row in resp_data["data"]:
                yield row

    async def find_key_value(
        self, entity_id: str, entity_type: KeyValueEntityType, name: str
    ) -> KeyValue | None:
        it = self.unpaginate(
            KeyValue._path,
            {"entityId": entity_id, "entityType": entity_type.value, "name": name},
        )
        response = await anext(it, None)
        if response:
            return KeyValue.from_server_data(response)
        return None

    async def iter_key_values(
        self, entity_id: str, entity_type: KeyValueEntityType
    ) -> AsyncGenerator[KeyValue, None]:
        async for x in self.unpaginate(
            KeyValue._path, {"entityId": entity_id, "entityType": entity_type.value}
        ):
            yield KeyValue.from_server_data(x)

    async def create_key_value(
        self,
        entity_id: str,
        entity_type: KeyValueEntityType,
        name: str,
        category: dr.enums.KeyValueCategory,
        value_type: dr.enums.KeyValueType,
        value: Optional[Union[str, float, bool]] = None,
        description: Optional[str] = None,
    ) -> None:
        value_keys = {
            dr.enums.KeyValueType.NUMERIC: "numericValue",
            dr.enums.KeyValueType.BOOLEAN: "booleanValue",
        }
        value_key = value_keys.get(value_type, "value")

        data: dict[str, str | float | bool | None] = {
            "entityId": entity_id,
            "entityType": entity_type.value,
            "name": name,
            "category": category.value,
            "valueType": value_type.value,
            value_key: value,
            "description": description,
        }

        json: dict[str, str | float | bool] = {
            k: v for k, v in data.items() if v is not None
        }

        await self._client.post(KeyValue._path, json=json)

    async def update_key_value(
        self,
        id: str,
        entity_id: str,
        entity_type: KeyValueEntityType,
        name: str,
        category: dr.enums.KeyValueCategory,
        value_type: dr.enums.KeyValueType,
        value: Optional[Union[str, float, bool]] = None,
        description: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        value_keys = {
            dr.enums.KeyValueType.NUMERIC: "numericValue",
            dr.enums.KeyValueType.BOOLEAN: "booleanValue",
        }
        value_key = value_keys.get(value_type, "value")

        payload: dict[str, str | float | bool | None] = {
            "entityId": entity_id,
            "entityType": entity_type.value,
            "name": name,
            "category": category.value,
            "valueType": value_type.value,
            value_key: value,
        }
        if description is not None:
            payload["description"] = description
        if comment is not None:
            payload["comment"] = comment

        json: dict[str, str | float | bool] = {
            k: v for k, v in payload.items() if v is not None
        }

        await self._client.patch(
            KeyValue._key_value_path(id),
            json=json,
        )

    async def delete_key_value(self, id: str) -> None:
        await self._client.delete(KeyValue._key_value_path(id))

    async def download_file(self, catalog_id: str, local_path: str) -> None:
        async with self._client.stream(
            "GET",
            urljoin(File._path, f"{catalog_id}/file/"),
            timeout=httpx.Timeout(
                FILE_API_READ_TIMEOUT,
                connect=FILE_API_CONNECT_TIMEOUT,
            ),
        ) as response:
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk in response.aiter_bytes(1024):
                    await f.write(chunk)

    async def upload_file(self, local_path: str) -> dict[str, Any]:
        with open(local_path, "rb") as read_file:
            r = await self._client.post(
                urljoin(File._path, "fromFile/"),
                files={"file": read_file},
                data={"useArchiveContents": "true"},
                timeout=httpx.Timeout(
                    FILE_API_READ_TIMEOUT,
                    connect=FILE_API_CONNECT_TIMEOUT,
                ),
            )
        return cast(dict[str, Any], r.json())

    async def delete_file(self, catalog_id: str) -> None:
        await self._client.delete(urljoin(File._path, f"{catalog_id}/"))


async_client = AsyncDataRobotClient(
    endpoint=os.environ["DATAROBOT_ENDPOINT"], token=os.environ["DATAROBOT_API_TOKEN"]
)


class PersistentStorage:
    def __init__(self, user_id: Optional[str]):
        self.app_id: str = os.environ.get("APPLICATION_ID")  # type: ignore[assignment]
        if not self.app_id:
            raise ValueError("APPLICATION_ID env variable is not set.")
        self.name_prefix = f"{user_id}_"

    @telemetry.meter_and_trace
    async def files(self) -> List[str]:
        relevant_files = []

        async for v in async_client.iter_key_values(
            self.app_id, KeyValueEntityType.CUSTOM_APPLICATION
        ):
            if (
                v.value_type == dr.enums.KeyValueType.JSON
                and v.category == dr.enums.KeyValueCategory.ARTIFACT
                and v.name.startswith(self.name_prefix)
            ):
                relevant_files.append(v.name.removeprefix(self.name_prefix))

        return relevant_files

    async def fetch_from_storage(self, file_name: str, local_path: str) -> None:
        logger.info(f"Fetching file {file_name} from storage")
        storage_link = await async_client.find_key_value(
            self.app_id,
            KeyValueEntityType.CUSTOM_APPLICATION,
            f"{self.name_prefix}{file_name}",
        )
        if not storage_link:
            return
        data = json.loads(storage_link.value)
        await async_client.download_file(data["catalogId"], local_path)

    @telemetry.meter_and_trace
    async def save_to_storage(self, file_name: str, local_path: str) -> None:
        logger.info(f"Storing file {file_name} to persistent storage")
        storing_label = f"{self.name_prefix}{file_name}"
        timestamp = time.time_ns()

        catalog_info = await async_client.upload_file(local_path)
        file_data = {**catalog_info, "timestamp": timestamp}

        storage_link = await async_client.find_key_value(
            self.app_id, KeyValueEntityType.CUSTOM_APPLICATION, storing_label
        )
        data = json.loads(storage_link.value) if storage_link else {}
        if not data:
            # there is no previous version of this file
            await async_client.create_key_value(
                entity_id=self.app_id,
                entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
                name=storing_label,
                category=dr.enums.KeyValueCategory.ARTIFACT,
                value_type=dr.enums.KeyValueType.JSON,
                value=json.dumps(file_data),
            )
            return

        if timestamp > data["timestamp"]:
            assert storage_link is not None
            # uploaded file is newer version and storage link needs to be updated and old file to be removed
            await async_client.update_key_value(
                id=storage_link.id,
                entity_id=storage_link.entity_id,
                entity_type=KeyValueEntityType(storage_link.entity_type.value),
                name=storage_link.name,
                category=storage_link.category,
                value_type=storage_link.value_type,
                value=storage_link.value,
                description=storage_link.description,
                comment=None,
            )
            await async_client.delete_file(data["catalogId"])
        else:
            # there is a newer file in storage and we drop just uploaded one
            await async_client.delete_file(catalog_info["catalogId"])

    @telemetry.meter_and_trace
    async def delete_file(self, file_name: str) -> None:
        storage_link = await async_client.find_key_value(
            self.app_id,
            KeyValueEntityType.CUSTOM_APPLICATION,
            f"{self.name_prefix}{file_name}",
        )
        if not storage_link:
            return
        data = json.loads(storage_link.value)
        await async_client.delete_file(data["catalogId"])
        await async_client.delete_key_value(storage_link.id)
