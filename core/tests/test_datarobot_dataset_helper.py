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
"""
Pytests for core/datarobot_dataset_helper.
"""

import datetime
from dataclasses import dataclass
from typing import Any, Generator
from unittest.mock import AsyncMock, Mock, patch

import datarobot
import datarobot as dr
import pytest
from datarobot.enums import DataWranglingDialect, RecipeInputType, RecipeType
from datarobot.models.data_store import DataStore
from datarobot.models.dataset import Dataset
from datarobot.models.recipe import Recipe, RecipeDatasetInput
from datarobot.models.use_cases.use_case import UseCase
from requests import HTTPError

from core.analyst_db import UserRecipe
from core.api_exceptions import ApplicationUsageException
from core.data_connections.datarobot.datarobot_dataset_handler import (
    DatasetSparkRecipe,
    DataSourceRecipe,
    load_or_create_spark_recipe,
)
from core.data_connections.datarobot.helpers import (
    RecipeError,
    find_underlying_client_message,
    handle_datarobot_error,
)
from core.schema import ExternalDataSource, ExternalDataStore
from datarobot.enums import DataWranglingDialect, RecipeInputType, RecipeType
from datarobot.models.data_store import DataStore
from datarobot.models.dataset import Dataset
from datarobot.models.recipe import Recipe, RecipeDatasetInput, RecipePreview
from datarobot.models.use_cases.use_case import UseCase
from requests import HTTPError


def test_handle_datarobot_error() -> None:
    with pytest.raises(RecipeError):
        with handle_datarobot_error("..."):
            raise datarobot.errors.ClientError("", 500)
    with pytest.raises(InterruptedError):
        with handle_datarobot_error("..."):
            raise InterruptedError()
    with pytest.raises(RecipeError):
        with handle_datarobot_error("..."):
            raise RuntimeError()


@dataclass
class Mocks:
    analyst_db: Mock
    use_case: Mock
    use_case_list: Mock
    recipe: Mock
    recipe_get: Mock
    recipe_from_dataset: Mock
    recipe_set_inputs: Mock
    recipe_set_recipe_metadata: Mock
    dataset: Mock
    dataset_get: Mock
    dataset_list: Mock
    dataset_iterate: Mock
    get_client: Mock
    dr_client: Mock

    @property
    def spark_recipe(self) -> DatasetSparkRecipe:
        return DatasetSparkRecipe(recipe=self.recipe, analyst_db=self.analyst_db)


@pytest.fixture(scope="function")
def mocks() -> Generator[Mocks, None, None]:
    with (
        patch("core.analyst_db.AnalystDB", new_callable=AsyncMock) as analyst_db,
        patch("datarobot.models.use_cases.use_case.UseCase") as use_case,
        patch.object(UseCase, "list") as use_case_list,
        patch("datarobot.models.recipe.Recipe") as recipe,
        patch.object(Recipe, "get") as recipe_get,
        patch.object(Recipe, "from_dataset") as recipe_from_dataset,
        patch.object(Recipe, "set_inputs") as recipe_set_inputs,
        patch.object(Recipe, "set_recipe_metadata") as recipe_set_recipe_metadata,
        patch("datarobot.models.dataset.Dataset") as dataset,
        patch.object(Dataset, "list") as dataset_list,
        patch.object(Dataset, "iterate") as dataset_iterate,
        patch.object(Dataset, "get") as dataset_get,
        patch("datarobot.client.get_client") as get_client,
        patch("datarobot.client.Client") as dr_client,
    ):
        use_case_list.return_value = [use_case]
        recipe_get.return_value = recipe
        recipe_from_dataset.return_value = recipe
        dataset_list.return_value = [dataset]
        dataset_iterate.return_value = iter([dataset])
        dataset_get.return_value = dataset

        analyst_db.get_user_recipe.return_value = UserRecipe(
            user_id="user", recipe_id="recipe", datastore_id=None
        )
        analyst_db.user_id = "user"
        analyst_db.list_analyst_dataset_metadata.return_value = []
        recipe.id = "recipe"
        use_case.id = "use_case"
        use_case.name = "TalkToMyData Data Wrangling user"
        use_case.created_at = 5
        dataset.name = "dataset"

        if hasattr(load_or_create_spark_recipe, "__cache__"):
            load_or_create_spark_recipe.__cache__ = {}

        get_client.return_value = dr_client

        yield Mocks(
            analyst_db=analyst_db,
            use_case=use_case,
            use_case_list=use_case_list,
            recipe=recipe,
            recipe_get=recipe_get,
            recipe_from_dataset=recipe_from_dataset,
            recipe_set_inputs=recipe_set_inputs,
            dataset=dataset,
            dataset_get=dataset_get,
            dataset_list=dataset_list,
            dataset_iterate=dataset_iterate,
            dr_client=dr_client,
            get_client=get_client,
            recipe_set_recipe_metadata=recipe_set_recipe_metadata,
        )


@pytest.mark.asyncio
async def test_load_or_create_recipe_exists_happy_path(mocks: Mocks) -> None:
    await load_or_create_spark_recipe(mocks.analyst_db)

    mocks.use_case_list.assert_not_called()
    mocks.recipe_get.assert_called_once_with("recipe")
    mocks.dataset_list.assert_not_called()
    mocks.analyst_db.get_user_recipe.assert_called_once()


@pytest.mark.asyncio
async def test_load_or_create_recipe_recipe_get_404(mocks: Mocks) -> None:
    def raise_404(*args, **kwargs):  # type:ignore[no-untyped-def]
        raise datarobot.errors.ClientError(exc_message="404", status_code=404)

    mocks.recipe_get.side_effect = raise_404

    await load_or_create_spark_recipe(mocks.analyst_db, ["dataset"])

    mocks.use_case_list.assert_called_once()
    mocks.recipe_get.assert_called_once_with("recipe")
    mocks.recipe_from_dataset.assert_called_once_with(
        use_case=mocks.use_case,
        dataset=mocks.dataset,
        dialect=DataWranglingDialect.SPARK,
        recipe_type=RecipeType.SQL,
    )
    mocks.analyst_db.get_user_recipe.assert_called_once()
    mocks.analyst_db.set_user_recipe.assert_called_once_with(
        UserRecipe(user_id="user", recipe_id="recipe", datastore_id=None)
    )


@pytest.mark.asyncio
async def test_load_or_create_recipe_no_persisted_recipe_but_database_entry(
    mocks: Mocks,
) -> None:
    mocks.analyst_db.get_user_recipe.return_value = UserRecipe(
        user_id="user", recipe_id="", datastore_id=None
    )

    await load_or_create_spark_recipe(mocks.analyst_db, ["dataset"])

    mocks.use_case_list.assert_called_once()
    mocks.recipe_get.assert_not_called()
    mocks.recipe_from_dataset.assert_called_once_with(
        use_case=mocks.use_case,
        dataset=mocks.dataset,
        dialect=DataWranglingDialect.SPARK,
        recipe_type=RecipeType.SQL,
    )
    mocks.analyst_db.get_user_recipe.assert_called_once()
    mocks.analyst_db.set_user_recipe.assert_called_once_with(
        UserRecipe(user_id="user", recipe_id="recipe", datastore_id=None)
    )


@pytest.mark.asyncio
async def test_load_or_create_recipe_no_persisted_recipe_no_database_entry(
    mocks: Mocks,
) -> None:
    mocks.analyst_db.get_user_recipe.return_value = None

    await load_or_create_spark_recipe(mocks.analyst_db, ["dataset"])

    mocks.use_case_list.assert_called_once_with()
    mocks.recipe_get.assert_not_called()
    mocks.recipe_from_dataset.assert_called_once_with(
        use_case=mocks.use_case,
        dataset=mocks.dataset,
        dialect=DataWranglingDialect.SPARK,
        recipe_type=RecipeType.SQL,
    )
    mocks.analyst_db.get_user_recipe.assert_called_once()
    mocks.analyst_db.set_user_recipe.assert_called_once_with(
        UserRecipe(user_id="user", recipe_id="recipe", datastore_id=None)
    )


@pytest.mark.asyncio
async def test_load_or_create_recipe_no_persisted_recipe_no_database_entry_no_datasets(
    mocks: Mocks,
) -> None:
    mocks.analyst_db.get_user_recipe.return_value = None
    mocks.dataset_iterate.return_value = iter([])

    def check(e: ApplicationUsageException) -> bool:
        return e.detail["code"] == "RECIPE_NOT_INITIALIZED"  # type:ignore[index]

    with pytest.raises(ApplicationUsageException, check=check):
        await load_or_create_spark_recipe(mocks.analyst_db, [])

    mocks.use_case_list.assert_not_called()
    mocks.recipe_get.assert_not_called()
    mocks.recipe_from_dataset.assert_not_called()
    mocks.analyst_db.get_user_recipe.assert_called()
    mocks.analyst_db.set_user_recipe.assert_not_called()


@pytest.mark.asyncio
@patch("datarobot.models.use_cases.use_case.UseCase")
async def test_load_or_create_recipe_uses_later_created_at(
    use_case_2: Mock,
    mocks: Mocks,
) -> None:
    mocks.analyst_db.get_user_recipe.return_value = None
    mocks.use_case_list.return_value = [mocks.use_case, use_case_2]

    use_case_2.id = "use_case2"
    use_case_2.created_at = 10
    use_case_2.name = mocks.use_case.name

    await load_or_create_spark_recipe(mocks.analyst_db, ["dataset"])

    mocks.use_case_list.assert_called_once_with()
    mocks.recipe_get.assert_not_called()
    mocks.dataset_get.assert_called_once()
    mocks.recipe_from_dataset.assert_called_once_with(
        use_case=use_case_2,
        dataset=mocks.dataset,
        dialect=DataWranglingDialect.SPARK,
        recipe_type=RecipeType.SQL,
    )
    mocks.analyst_db.get_user_recipe.assert_called_once()
    mocks.analyst_db.set_user_recipe.assert_called_once_with(
        UserRecipe(user_id="user", recipe_id="recipe", datastore_id=None)
    )


@pytest.mark.asyncio
async def test_add_datasets_no_existing_inputs(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe
    mocks.recipe.inputs = []
    mocks.recipe.id = "recipe"

    await spark_recipe.add_datasets(["d1", "d2"])

    # Need to unpack this rather than assert called with to be able to sort the inputs, as order doesn't matter and isn't guaranteed.
    call_args_list = mocks.recipe_set_inputs.call_args_list
    assert len(call_args_list) == 1
    call = call_args_list[0][0]
    assert call[0] == "recipe"
    input_1, input_2 = sorted(call[1], key=lambda x: x.dataset_id)
    assert (input_1.input_type, input_1.dataset_id, input_1.dataset_version_id) == (
        RecipeInputType.DATASET,
        "d1",
        None,
    )
    assert (input_2.input_type, input_2.dataset_id, input_2.dataset_version_id) == (
        RecipeInputType.DATASET,
        "d2",
        None,
    )


@pytest.mark.asyncio
async def test_add_datasets_inputs_redundant(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe
    mocks.recipe.inputs = [
        RecipeDatasetInput(RecipeInputType.DATASET, "d1"),
        RecipeDatasetInput(RecipeInputType.DATASET, "d2"),
        RecipeDatasetInput(RecipeInputType.DATASET, "d3"),
    ]
    mocks.recipe.id = "recipe"

    await spark_recipe.add_datasets(["d1", "d2"])

    mocks.recipe_set_inputs.assert_not_called()


@pytest.mark.asyncio
async def test_list_dataset_names(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe
    mocks.recipe.inputs = [
        RecipeDatasetInput(RecipeInputType.DATASET, "d1", alias="a1"),
        RecipeDatasetInput(RecipeInputType.DATASET, "d2", alias="a2"),
        RecipeDatasetInput(RecipeInputType.DATASET, "d3", alias="a3"),
    ]

    names = await spark_recipe.list_dataset_names()

    assert set(names) == {"a1", "a2", "a3"}


@pytest.mark.asyncio
async def test_set_query(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe

    await spark_recipe.set_query("SELECT * FROM `org`")

    mocks.recipe_set_recipe_metadata.assert_called_once_with(
        "recipe", {"sql": "SELECT * FROM `org`"}
    )
    mocks.dr_client.patch.assert_called_once_with(
        "recipes/recipe/settings", {"sparkInstanceSize": "large"}
    )


@pytest.mark.asyncio
async def test_clear_query(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe

    await spark_recipe.clear_query()

    mocks.recipe_set_recipe_metadata.assert_called_once_with("recipe", {})


@pytest.mark.asyncio
async def test_preview_dataset(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe

    mocks.recipe.get_preview.return_value = RecipePreview(
        columns=["id", "double", "string", "other"],
        total_count=2,
        count=2,
        stored_count=2,
        byte_size=10,
        estimated_size_exceeds_limit=False,
        result_schema=[
            {"name": "id", "dataType": "INT_TYPE"},
            {"name": "double", "dataType": "DOUBLE_TYPE"},
            {"name": "string", "dataType": "STRING_TYPE"},
            {"name": "other", "dataType": "STRING_TYPE"},
        ],
        data=[[1, 1.0, "1", "A"], [2, 2.0, "2", "B"]],
        next=None,
    )

    data = await spark_recipe.preview_dataset(mocks.dataset)

    assert data.response.to_dict() == [
        {"id": 1, "double": 1.0, "string": "1", "other": "A"},
        {"id": 2, "double": 2.0, "string": "2", "other": "B"},
    ]
    assert data.original_types == {
        "id": "INT_TYPE",
        "double": "DOUBLE_TYPE",
        "string": "STRING_TYPE",
        "other": "STRING_TYPE",
    }

    mocks.recipe_set_recipe_metadata.assert_called_once_with(
        "recipe", {"sql": "SELECT * FROM `dataset` LIMIT 1000"}
    )


@pytest.mark.asyncio
async def test_preview_dataset__snake_case_data_type(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe

    mocks.recipe.get_preview.return_value = RecipePreview(
        columns=["id", "double", "string", "other"],
        total_count=2,
        count=2,
        stored_count=2,
        byte_size=10,
        estimated_size_exceeds_limit=False,
        result_schema=[
            {"name": "id", "data_type": "INT_TYPE"},
            {"name": "double", "dataType": "DOUBLE_TYPE"},
            {"name": "string", "data_type": "STRING_TYPE"},
            {"name": "other", "data_type": "STRING_TYPE"},
        ],
        data=[[1, 1.0, "1", "A"], [2, 2.0, "2", "B"]],
        next=None,
    )

    data = await spark_recipe.preview_dataset(mocks.dataset)

    assert data.response.to_dict() == [
        {"id": 1, "double": 1.0, "string": "1", "other": "A"},
        {"id": 2, "double": 2.0, "string": "2", "other": "B"},
    ]
    assert data.original_types == {
        "id": "INT_TYPE",
        "double": "DOUBLE_TYPE",
        "string": "STRING_TYPE",
        "other": "STRING_TYPE",
    }

    mocks.recipe_set_recipe_metadata.assert_called_once_with(
        "recipe", {"sql": "SELECT * FROM `dataset` LIMIT 1000"}
    )


@pytest.mark.asyncio
async def test_preview_dataset_retryable_error(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe

    call_count = 0

    def _retrieve_preview(*args: Any) -> RecipePreview:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise datarobot.errors.ClientError(
                "test", status_code=404, json={"message": "Preview is not ready yet"}
            )
        else:
            return RecipePreview(
                columns=["id", "double", "string", "other"],
                total_count=2,
                count=2,
                stored_count=2,
                byte_size=10,
                estimated_size_exceeds_limit=False,
                result_schema=[
                    {"name": "id", "dataType": "INT_TYPE"},
                    {"name": "double", "dataType": "DOUBLE_TYPE"},
                    {"name": "string", "dataType": "STRING_TYPE"},
                    {"name": "other", "dataType": "STRING_TYPE"},
                ],
                data=[[1, "INVALID", "1", "A"], [2, "2.0", "2", "B"]],
                next=None,
            )

    mocks.recipe.get_preview.side_effect = _retrieve_preview

    data = await spark_recipe.preview_dataset(mocks.dataset)

    assert data.response.to_dict() == [
        {"id": 1, "double": None, "string": "1", "other": "A"},
        {"id": 2, "double": 2.0, "string": "2", "other": "B"},
    ]
    assert data.original_types == {
        "id": "INT_TYPE",
        "double": "DOUBLE_TYPE",
        "string": "STRING_TYPE",
        "other": "STRING_TYPE",
    }
    assert call_count == 2

    mocks.recipe_set_recipe_metadata.assert_called_once_with(
        "recipe", {"sql": "SELECT * FROM `dataset` LIMIT 1000"}
    )


@pytest.mark.parametrize(
    "schema,rows,expected",
    [
        (
            [{"name": "flag", "dataType": "BOOL"}],
            [["t"], ["f"], ["1"], ["0"], ["yes"], ["unknown"]],
            [{"flag": bool(t)} for t in [1, 0, 1, 0, 1, 0]],
        ),
        (
            [{"name": "val", "dataType": "REAL"}],
            [["1.5"], ["INVALID"], ["2"]],
            [{"val": 1.5}, {"val": None}, {"val": 2.0}],
        ),
        (
            [{"name": "val", "dataType": "NUMERIC"}],
            [["3.14"], ["2"], ["bad"]],
            [{"val": 3.14}, {"val": 2.0}, {"val": None}],
        ),
        (
            [{"name": "n", "dataType": "SMALLINT"}],
            [["1"], ["-2"], ["bad"]],
            [{"n": 1}, {"n": -2}, {"n": None}],
        ),
        (
            [{"name": "a", "dataType": "INT"}, {"name": "b", "dataType": "INT"}],
            [("1", "2"), (11.0, 12)],
            [{"a": 1, "b": 2}, {"a": 11, "b": 12}],
        ),
        (
            [{"name": "val", "dataType": "DATETIME"}],
            [
                (datetime.datetime(2025, 1, 1),),
                ("2025-01-02 00:00:00",),
                ("2025-01-03",),
                ("2025/1/4",),
                ("2025-01-05T00:00:00.000",),
            ],
            [
                {"val": datetime.datetime(2025, 1, 1)},
                {"val": datetime.datetime(2025, 1, 2)},
                {"val": datetime.datetime(2025, 1, 3)},
                {"val": datetime.datetime(2025, 1, 4)},
                {"val": datetime.datetime(2025, 1, 5)},
            ],
        ),
    ],
)
def test_convert_preview_to_dataframe(
    schema: list[dict[str, str]], rows: list[list[str]], expected: list[dict[str, Any]]
) -> None:
    data = DatasetSparkRecipe.convert_preview_to_dataframe(schema, rows)
    assert data.to_dict() == expected


def test_find_underlying_client_message_from_client_error_direct() -> None:
    err = dr.errors.ClientError(
        "err", status_code=400, json={"message": "client error msg"}
    )
    assert find_underlying_client_message(err) == "client error msg"


def test_find_underlying_client_message_from_client_error_in_cause() -> None:
    inner = dr.errors.ClientError(
        "inner", status_code=500, json={"message": "inner msg"}
    )
    outer = Exception("outer")
    # attach as __cause__ to simulate exception chaining
    outer.__cause__ = inner
    assert find_underlying_client_message(outer) == "inner msg"


def test_find_underlying_client_message_from_http_error_response_json() -> None:
    resp = Mock()
    resp.json.return_value = {"message": "http error msg"}
    http_err = HTTPError()
    http_err.response = resp
    assert find_underlying_client_message(http_err) == "http error msg"


def test_find_underlying_client_message_none_when_not_present() -> None:
    assert find_underlying_client_message(Exception("no useful info")) is None


def test_find_underlying_client_message_asyncprocess_arg_not_string() -> None:
    # args[0] is not a string -> should return None
    exc = dr.errors.AsyncProcessUnsuccessfulError(123)
    assert find_underlying_client_message(exc) is None


def test_find_underlying_client_message_asyncprocess_arg_string_no_token() -> None:
    # args[0] is a string but does not contain the marker -> return original string
    text = "Something went wrong without job data"
    exc = dr.errors.AsyncProcessUnsuccessfulError(text)
    assert find_underlying_client_message(exc) == text


def test_find_underlying_client_message_asyncprocess_token_with_invalid_json() -> None:
    # Contains the token but what follows is not valid JSON -> return the raw trailing portion
    payload = " not-a-json"
    full = "Error happened " + "Job Data:" + payload
    exc = dr.errors.AsyncProcessUnsuccessfulError(full)
    assert find_underlying_client_message(exc) == payload


def test_find_underlying_client_message_asyncprocess_token_with_json_no_message() -> (
    None
):
    # Contains the token and valid JSON, but JSON does not contain `message` -> return the JSON text
    payload = '{"other":"value"}'
    full = "Something bad " + "Job Data:" + payload
    exc = dr.errors.AsyncProcessUnsuccessfulError(full)
    assert find_underlying_client_message(exc) == payload


def test_find_underlying_client_message_asyncprocess_token_with_json_with_message() -> (
    None
):
    # Contains the token and valid JSON with `message` -> return the JSON message field
    payload = '{"message":"inner job error"}'
    full = "Ignored prefix " + "Job Data:" + payload
    exc = dr.errors.AsyncProcessUnsuccessfulError(full)
    assert find_underlying_client_message(exc) == "inner job error"


@pytest.mark.asyncio
async def test_list_available_datastores_fetches_and_returns_datastore(
    mocks: Mocks,
) -> None:
    user_id = "fresh_user"
    # Ensure cache miss for this user
    DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE.pop(user_id, None)

    def r(v: dict[str, Any]) -> Any:
        m = Mock()
        m.json.return_value = v
        m.status_code = 200
        return m

    def get(path: str, *args: Any, **kwargs: Any) -> Any:
        match path:
            case "externalDataStores/":
                return r(
                    {
                        "data": [
                            {
                                "id": "ds1",
                                "canonicalName": "canonicalName",
                                "driverClassType": "postgres",
                                "creator": "me",
                                "updated": "2025-03-04T00:00:00Z",
                                "type": "jdbc",
                                "params": {},
                                "role": "as",
                            },
                            {
                                "id": "ds2",
                                "canonicalName": "canonicalName",
                                "driverClassType": "redshift",
                                "creator": "me",
                                "updated": "2025-03-04T00:00:00Z",
                                "type": "jdbc",
                                "params": {},
                                "role": "as",
                            },
                            {
                                "id": "ds3",
                                "canonicalName": "canonicalName",
                                "driverClassType": "unsupported",
                                "creator": "me",
                                "updated": "2025-03-04T00:00:00Z",
                                "type": "jdbc",
                                "params": {},
                                "role": "as",
                            },
                        ]
                    }
                )
            case (
                "credentials/associations/dataconnection:ds1/?orderBy=-isDefault"
                | "credentials/associations/dataconnection:ds2/?orderBy=-isDefault"
            ):
                return r(
                    {
                        "data": [
                            {
                                "name": "c1",
                                "credentialId": "c1",
                                "credentialType": "password",
                                "creationDate": "2018-07-01T00:00:00Z",
                                "isDefault": True,
                                "description": "",
                            }
                        ]
                    }
                )
        raise RuntimeError(f"Path {path} not expected")

    def post(path: str, payload: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        match path:
            case "externalDataStores/ds1/schemas/":
                return r(
                    {
                        "catalog": "pg",
                        "schemas": ["pgs1", "pgs2"],
                        "catalogs": ["pg", "pg"],
                    }
                )
            case "externalDataStores/ds2/schemas/":
                return r(
                    {
                        "catalog": "rs",
                        "schemas": ["rs1"],
                    }
                )
            case "externalDataStores/ds1/tables/":
                if payload.get("schema") == "pgs1":
                    return r(
                        {
                            "tables": [
                                {"catalog": "pg", "schema": "pgs1", "name": "pgs1a"},
                                {"catalog": "pg", "schema": "pgs1", "name": "pgs1b"},
                            ]
                        }
                    )
                elif payload.get("schema") == "pgs2":
                    return r(
                        {
                            "tables": [
                                {"catalog": "pg", "schema": "pgs2", "name": "pgs2a"}
                            ]
                        }
                    )
            case "externalDataStores/ds2/tables/":
                return r({"tables": [{"name": "rs1a"}]})

        raise RuntimeError(f"Path {path} not expected")

    mocks.dr_client.get = get
    mocks.dr_client.post = post

    results = await DataSourceRecipe.list_available_datastores(user_id)

    assert DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE[user_id][1] is results

    results.sort(key=lambda x: x.id)
    for rs in results:
        rs.defined_data_sources.sort(key=lambda x: x.path)

    assert results == [
        ExternalDataStore(
            id="ds1",
            canonical_name="canonicalName",
            driver_class_type="postgres",
            defined_data_sources=[
                ExternalDataSource(
                    data_store_id="ds1",
                    database_catalog="pg",
                    database_schema="pgs1",
                    database_table="pgs1a",
                ),
                ExternalDataSource(
                    data_store_id="ds1",
                    database_catalog="pg",
                    database_schema="pgs1",
                    database_table="pgs1b",
                ),
                ExternalDataSource(
                    data_store_id="ds1",
                    database_catalog="pg",
                    database_schema="pgs2",
                    database_table="pgs2a",
                ),
            ],
        ),
        ExternalDataStore(
            id="ds2",
            canonical_name="canonicalName",
            driver_class_type="redshift",
            defined_data_sources=[
                ExternalDataSource(
                    data_store_id="ds2",
                    database_catalog=None,
                    database_schema=None,
                    database_table="rs1a",
                )
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_get_id_for_data_store_canonical_name_multiple_matches() -> None:
    store_a = Mock()
    store_a.canonical_name = "canonicalname"
    store_a.id = "ds1"

    store_b = Mock()
    store_b.canonical_name = "othername"
    store_b.id = "ds2"

    DataSourceRecipe.DATA_STORE_ID_TO_NAME_CACHE.clear()
    DataSourceRecipe.DATA_STORE_NAME_TO_ID_CACHE.clear()

    with patch.object(DataStore, "list", return_value=[store_b, store_a]):
        # Should match regardless of case/whitespace
        result = await DataSourceRecipe.get_id_for_data_store_canonical_name(
            " canonicalname "
        )
        assert result == "ds1"


@pytest.mark.asyncio
async def test_get_id_for_data_store_canonical_name_multiple_matches_not_found() -> (
    None
):
    store_a = Mock()
    store_a.canonical_name = "CanonicalName"
    store_a.id = "ds1"

    store_b = Mock()
    store_b.canonical_name = "OtherName"
    store_b.id = "ds2"

    DataSourceRecipe.DATA_STORE_ID_TO_NAME_CACHE.clear()
    DataSourceRecipe.DATA_STORE_NAME_TO_ID_CACHE.clear()

    with patch.object(DataStore, "list", return_value=[store_b, store_a]):
        # Should match regardless of case/whitespace
        result = await DataSourceRecipe.get_id_for_data_store_canonical_name(
            "NoSuchName"
        )
        assert result is None


@pytest.mark.asyncio
async def test_get_canonical_name_for_datastore_id_found() -> None:
    store = Mock()
    store.canonical_name = "FoundName"
    with patch.object(DataStore, "get", return_value=store):
        assert (
            await DataSourceRecipe.get_canonical_name_for_datastore_id("ds1")
            == "FoundName"
        )


@pytest.mark.asyncio
async def test_get_canonical_name_for_datastore_id_not_found() -> None:
    err = dr.errors.ClientError("not found", status_code=404, json={})
    with patch.object(DataStore, "get", side_effect=err):
        assert (
            await DataSourceRecipe.get_canonical_name_for_datastore_id("ds_missing")
            is None
        )


def test_query_friendly_name_dataset_spark_recipe_with_csv(mocks: Mocks) -> None:
    spark_recipe = mocks.spark_recipe
    dataset_name = "Bakery Dataset.csv"

    result = spark_recipe.query_friendly_name(dataset_name)

    assert result == "`Bakery Dataset.csv`"


@pytest.mark.parametrize(
    "driver_class_type,table_path,expected",
    [
        ("postgres", "catalog.schema.table", '"catalog"."schema"."table"'),
        ("redshift", "catalog.schema.table", '"catalog"."schema"."table"'),
    ],
)
def test_query_friendly_name_data_source_recipe_with_path(
    mocks: Mocks, driver_class_type: str, table_path: str, expected: str
) -> None:
    data_store = ExternalDataStore(
        id="ds1",
        canonical_name="TestStore",
        driver_class_type=driver_class_type,
        defined_data_sources=[],
    )
    data_source_recipe = DataSourceRecipe(
        analyst_db=mocks.analyst_db,
        recipe=mocks.recipe,
        data_store=data_store,
    )

    result = data_source_recipe.query_friendly_name(table_path)

    assert result == expected


@pytest.mark.asyncio
async def test_load_or_create_datastore_happy_path(mocks: Mocks) -> None:
    key = (mocks.analyst_db.user_id, "ds1")
    DataSourceRecipe.INSTANCE_CACHE.pop(key, None)

    resp = Mock()
    resp.status_code = 200
    resp.json.return_value = {
        "driverClassType": "postgres",
        "id": "ds1",
        "canonicalName": "canonicalName",
        "type": "jdbc",
    }

    mocks.dr_client.get = lambda path: resp

    dr_store = Mock()
    dr_store.id = "ds1"
    dr_store.canonical_name = "canonicalName"
    with patch.object(DataStore, "from_server_data", return_value=dr_store):
        mocks.analyst_db.get_user_recipe.return_value = UserRecipe(
            user_id=mocks.analyst_db.user_id, recipe_id="r1", datastore_id="ds1"
        )

        mocks.analyst_db.list_sources_for_data_store.return_value = []
        mocks.analyst_db.list_analyst_dataset_metadata.return_value = []

        result = await DataSourceRecipe.load_or_create(mocks.analyst_db, "ds1")

    assert DataSourceRecipe.INSTANCE_CACHE[key] is result

    mocks.analyst_db.register_data_store.assert_awaited()
    assert result.data_store.id == "ds1"
    assert result.data_store.canonical_name == "canonicalName"
