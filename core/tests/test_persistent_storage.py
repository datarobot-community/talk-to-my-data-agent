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

import json
import os
import tempfile
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch

import datarobot as dr
import httpx
import pytest

from core.datarobot_client import KeyValue, KeyValueEntityType
from core.persistent_storage import AsyncDataRobotClient, PersistentStorage


@pytest.fixture()
def httpx_async_client() -> Generator[AsyncMock, None, None]:
    with patch("httpx.AsyncClient", autospec=httpx.AsyncClient) as async_client:
        async_client.return_value = async_client
        yield async_client


@pytest.mark.asyncio
async def test_iter_key_value(httpx_async_client: AsyncMock) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    def r(v: Any) -> Any:
        x = Mock()
        x.json.return_value = v
        return x

    async def get_key_values(path: str, *, params: dict[str, Any]) -> Any:
        assert params == {"entityId": "e", "entityType": "customApplication"}
        if path == "keyValues/":
            return r({"data": [1, 2], "next": "keyValues/next/"})
        elif path == "keyValues/next/":
            return r(
                {
                    "data": [3],
                }
            )
        else:
            assert False, f"Unexpected path {path}."

    httpx_async_client.get = get_key_values
    KeyValue.from_server_data = lambda x: x  # type:ignore[method-assign,assignment,misc]

    key_values = []
    async for key_value in client.iter_key_values(
        "e", KeyValueEntityType.CUSTOM_APPLICATION
    ):
        key_values.append(key_value)

    assert key_values == [1, 2, 3]


@pytest.mark.asyncio
async def test_find_key_value(httpx_async_client: AsyncMock) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    def r(v: Any) -> Any:
        x = Mock()
        x.json.return_value = v
        return x

    async def get_key_values(path: str, *, params: dict[str, Any]) -> Any:
        assert params == {
            "entityId": "e",
            "entityType": "customApplication",
            "name": "n",
        }
        if path == "keyValues/":
            return r(
                {
                    "data": [1],
                }
            )
        else:
            assert False, f"Unexpected path {path}."

    httpx_async_client.get = get_key_values
    KeyValue.from_server_data = lambda x: x  # type:ignore[method-assign,assignment,misc]

    key_value = await client.find_key_value(
        "e", KeyValueEntityType.CUSTOM_APPLICATION, "n"
    )

    assert key_value == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value_type,value,data_key,value_type_str",
    [
        (dr.enums.KeyValueType.NUMERIC, 3.14, "numericValue", "numeric"),
        (dr.enums.KeyValueType.BOOLEAN, True, "booleanValue", "boolean"),
        (dr.enums.KeyValueType.JSON, '{"x": 1}', "value", "json"),
    ],
)
async def test_create_key_value_param(
    httpx_async_client: AsyncMock,
    value_type: dr.enums.KeyValueType,
    value: str | float | bool,
    data_key: str,
    value_type_str: str,
) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    await client.create_key_value(
        entity_id="id",
        entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
        name="test-name",
        category=dr.enums.KeyValueCategory.ARTIFACT,
        value_type=value_type,
        value=value,
        description="desc",
    )

    httpx_async_client.post.assert_called_once()

    called_args, called_kwargs = httpx_async_client.post.await_args

    assert called_args == ("keyValues/",)
    assert called_kwargs == {
        "json": {
            data_key: value,
            "valueType": value_type_str,
            "category": "artifact",
            "description": "desc",
            "entityId": "id",
            "entityType": "customApplication",
            "name": "test-name",
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value_type,value,data_key,value_type_str,description,comment",
    [
        (dr.enums.KeyValueType.NUMERIC, 3.14, "numericValue", "numeric", None, None),
        (dr.enums.KeyValueType.BOOLEAN, True, "booleanValue", "boolean", "desc", None),
        (dr.enums.KeyValueType.JSON, '{"x": 1}', "value", "json", None, "comment"),
    ],
)
async def test_update_key_value_param(
    httpx_async_client: AsyncMock,
    value_type: dr.enums.KeyValueType,
    value: str | float | bool,
    data_key: str,
    value_type_str: str,
    description: str | None,
    comment: str | None,
) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    await client.update_key_value(
        id="1",
        entity_id="id",
        entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
        name="test-name",
        category=dr.enums.KeyValueCategory.ARTIFACT,
        value_type=value_type,
        value=value,
        description=description,
        comment=comment,
    )

    httpx_async_client.patch.assert_called_once()

    called_args, called_kwargs = httpx_async_client.patch.await_args

    expected_call = {
        data_key: value,
        "valueType": value_type_str,
        "category": "artifact",
        "entityId": "id",
        "entityType": "customApplication",
        "name": "test-name",
    }

    if description is not None:
        expected_call["description"] = description
    if comment is not None:
        expected_call["comment"] = comment

    assert called_args == ("keyValues/1/",)
    assert called_kwargs == {"json": expected_call}


@pytest.mark.asyncio
async def test_delete_key_value(httpx_async_client: AsyncMock) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    await client.delete_key_value("1")

    httpx_async_client.delete.assert_called_once()

    called_args, called_kwargs = httpx_async_client.delete.await_args

    assert called_args == ("keyValues/1/",)
    assert called_kwargs == {}


@pytest.mark.asyncio
async def test_download_file_writes_contents(httpx_async_client: AsyncMock) -> None:
    chunks = [b"hello", b" ", b"world"]

    response = AsyncMock()

    async def aiter_bytes(chunk_size: int = 1024) -> AsyncGenerator[bytes, None]:
        for c in chunks:
            yield c

    response.aiter_bytes = aiter_bytes

    cm = AsyncMock()
    cm.__aenter__.return_value = response
    httpx_async_client.stream.return_value = cm

    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    _, path = tempfile.mkstemp()

    try:
        await client.download_file("catalog123", path)

        with open(path, "rb") as f:
            data = f.read()

        assert data == b"".join(chunks)
    finally:
        os.remove(path)


@pytest.mark.asyncio
async def test_upload_file_posts_given_path(httpx_async_client: AsyncMock) -> None:
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(b"upload-content")

        response = AsyncMock()
        expected_json = {"catalogId": "catalog-1", "size": 123}
        response.json = Mock(return_value=expected_json)
        httpx_async_client.post.return_value = response

        client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

        result = await client.upload_file(path)

        httpx_async_client.post.assert_called_once()

        called_args, called_kwargs = httpx_async_client.post.await_args

        assert called_args == ("files/fromFile/",)

        assert "files" in called_kwargs
        files_dict = called_kwargs["files"]
        assert "file" in files_dict
        uploaded_file = files_dict["file"]

        assert hasattr(uploaded_file, "name")
        assert uploaded_file.name == path

        assert "timeout" in called_kwargs

        assert result == expected_json
    finally:
        os.remove(path)


@pytest.mark.asyncio
async def test_delete_file(httpx_async_client: AsyncMock) -> None:
    client = AsyncDataRobotClient(token="token", endpoint="https://dr.example")

    await client.delete_file("1")

    httpx_async_client.delete.assert_called_once()

    called_args, called_kwargs = httpx_async_client.delete.await_args

    assert called_args == ("files/1/",)
    assert called_kwargs == {}


@pytest.fixture
def dr_async_client(monkeypatch: Any) -> Generator[AsyncMock, None, None]:
    monkeypatch.setenv("APPLICATION_ID", "app")

    async_client = AsyncMock(autospec=AsyncDataRobotClient)
    with patch("core.persistent_storage.AsyncDataRobotClient") as client_class:
        client_class.return_value = async_client
        yield async_client


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_files(dr_async_client: AsyncMock) -> None:
    async def kv(
        value_type: dr.enums.KeyValueType,
        category: dr.enums.KeyValueCategory,
        name: str,
    ) -> Any:
        x = Mock()
        x.value_type = value_type
        x.category = category
        x.name = name
        return x

    async def iter_kvs(
        entity_id: str, entity_type: dr.enums.KeyValueEntityType
    ) -> AsyncGenerator[Any, None]:
        for x in [
            (
                dr.enums.KeyValueType.STRING,
                dr.enums.KeyValueCategory.ARTIFACT,
                "user_abc",
            ),
            (dr.enums.KeyValueType.JSON, dr.enums.KeyValueCategory.METRIC, "user_def"),
            (
                dr.enums.KeyValueType.JSON,
                dr.enums.KeyValueCategory.ARTIFACT,
                "user_ghi",
            ),
        ]:
            yield await kv(*x)

    dr_async_client.iter_key_values = iter_kvs

    storage = PersistentStorage("user")

    assert (await storage.files()) == ["ghi"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_fetch_from_storage_present(dr_async_client: AsyncMock) -> None:
    path = "p"
    storage_link = Mock()
    storage_link.value = json.dumps({"catalogId": "catalog-1"})

    dr_async_client.find_key_value.return_value = storage_link

    storage = PersistentStorage("user")

    await storage.fetch_from_storage("name", path)

    dr_async_client.download_file.assert_called_once_with("catalog-1", path)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_fetch_from_storage_missing(dr_async_client: AsyncMock) -> None:
    path = "p"

    dr_async_client.find_key_value.return_value = None

    storage = PersistentStorage("user")

    await storage.fetch_from_storage("name", path)

    dr_async_client.download_file.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_save_to_storage_creates_key_value_when_missing(
    dr_async_client: AsyncMock,
) -> None:
    with patch("time.time_ns", new_callable=Mock(return_value=lambda: 5)):
        path = "path"
        dr_async_client.upload_file.return_value = {"catalogId": "catalog-1"}
        dr_async_client.find_key_value.return_value = None
        dr_async_client.create_key_value = AsyncMock()

        storage = PersistentStorage("user")

        await storage.save_to_storage("name", path)

        dr_async_client.upload_file.assert_called_once_with(path)

        dr_async_client.create_key_value.assert_called_once()

        _, called_kwargs = dr_async_client.create_key_value.await_args
        assert called_kwargs == {
            "category": dr.enums.KeyValueCategory.ARTIFACT,
            "entity_id": "app",
            "entity_type": KeyValueEntityType.CUSTOM_APPLICATION,
            "name": "user_name",
            "value": '{"catalogId": "catalog-1", "timestamp": 5}',
            "value_type": dr.enums.KeyValueType.JSON,
        }


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_save_to_storage_updates_key_value_when_present(
    dr_async_client: AsyncMock,
) -> None:
    with patch("time.time_ns", new_callable=Mock(return_value=lambda: 5)):
        path = "path"
        kv = Mock(
            value='{"timestamp":4,"catalogId":"c1"}',
            entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
        )
        dr_async_client.upload_file.return_value = {"catalogId": "catalog-1"}
        dr_async_client.find_key_value.return_value = kv

        storage = PersistentStorage("user")

        await storage.save_to_storage("name", path)

        dr_async_client.upload_file.assert_called_once_with(path)

        dr_async_client.update_key_value.assert_called_once()
        dr_async_client.delete_file.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_save_to_storage_skips_update_key_value_when_outdated(
    dr_async_client: AsyncMock,
) -> None:
    with patch("time.time_ns", new_callable=Mock(return_value=lambda: 5)):
        path = "path"
        kv = Mock(
            value='{"timestamp":6,"catalogId":"c1"}',
            entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
        )
        dr_async_client.upload_file.return_value = {"catalogId": "catalog-1"}
        dr_async_client.find_key_value.return_value = kv
        dr_async_client.create_key_value = AsyncMock()

        storage = PersistentStorage("user")

        await storage.save_to_storage("name", path)

        dr_async_client.upload_file.assert_called_once_with(path)

        dr_async_client.update_key_value.assert_not_called()
        dr_async_client.delete_file.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_delete_file_no_storage_link(dr_async_client: AsyncMock) -> None:
    dr_async_client.find_key_value.return_value = None

    await PersistentStorage("user").delete_file("p")

    dr_async_client.delete_file.assert_not_called()
    dr_async_client.delete_key_value.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_env_endpoint_and_token")
async def test_delete_file_storage_link(dr_async_client: AsyncMock) -> None:
    dr_async_client.find_key_value.return_value = Mock(value='{"catalogId":"c"}', id=1)

    await PersistentStorage("user").delete_file("p")

    dr_async_client.delete_file.assert_called_once_with("c")
    dr_async_client.delete_key_value.assert_called_once_with(1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_to_storage_already_existing() -> None:
    # This test ensures that when a storage link already exists for the given
    # file name, calling save_to_storage will upload the new file, update the
    # existing key-value (with the new catalog info and timestamp) and delete
    # the previously stored catalog file.
    with patch("time.time_ns", new_callable=Mock(return_value=lambda: 10)):
        path = "path"
        # existing stored key-value pointing to an older catalog
        kv = Mock(
            id="kv-id",
            entity_id="app",
            entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
            name="name",
            category=dr.enums.KeyValueCategory.ARTIFACT,
            value_type=dr.enums.KeyValueType.JSON,
            description="prev-desc",
            value=json.dumps({"timestamp": 1, "catalogId": "old-catalog"}),
        )

        # patch the async client methods that PersistentStorage will call
        dr_async_client = AsyncMock(autospec=AsyncDataRobotClient)
        with patch("core.persistent_storage.AsyncDataRobotClient") as client_class:
            client_class.return_value = dr_async_client
            dr_async_client.upload_file.return_value = {"catalogId": "catalog-2"}
            dr_async_client.find_key_value.return_value = kv
            dr_async_client.update_key_value = AsyncMock()
            dr_async_client.delete_file = AsyncMock()

            storage = PersistentStorage("user")

            await storage.save_to_storage("name", path)

            # ensure upload happened
            dr_async_client.upload_file.assert_called_once_with(path)

            # ensure update_key_value was awaited and inspect its await args
            dr_async_client.update_key_value.assert_awaited_once()
            aa = dr_async_client.update_key_value.await_args
            assert aa is not None
            called_args, called_kwargs = aa

            assert called_kwargs["id"] == "kv-id"
            assert called_kwargs["entity_id"] == "app"
            # entity_type should be the KeyValueEntityType matching the stored one
            assert called_kwargs["entity_type"] == KeyValueEntityType.CUSTOM_APPLICATION
            assert called_kwargs["category"] == dr.enums.KeyValueCategory.ARTIFACT
            assert called_kwargs["value_type"] == dr.enums.KeyValueType.JSON
            # value should be JSON with the new catalogId and the patched timestamp=10
            assert json.loads(called_kwargs["value"]) == {
                "catalogId": "catalog-2",
                "timestamp": 10,
            }
            assert called_kwargs["description"] == "prev-desc"
            assert called_kwargs["comment"] is None

            # ensure old catalog file was deleted
            dr_async_client.delete_file.assert_called_once_with("old-catalog")
