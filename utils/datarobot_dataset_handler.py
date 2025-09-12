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
Functionality for performing Spark SQL queries with DataRobot Recipes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from functools import cache, lru_cache
from typing import Any, Callable, Generator, Iterable, ParamSpec, TypeVar, cast

import datarobot as dr
import polars as pl
from datarobot.enums import DataWranglingDialect, RecipeInputType, RecipeType
from datarobot.errors import AsyncTimeoutError, ClientError
from datarobot.models.dataset import Dataset
from datarobot.models.recipe import Recipe, RecipeDatasetInput
from datarobot.models.use_cases.use_case import UseCase
from datarobot.utils.pagination import unpaginate
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from packaging.version import Version
from tenacity import (
    after_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.analyst_db import AnalystDB, DataSourceType, UserRecipe
from utils.code_execution import InvalidGeneratedCode
from utils.credentials import NoDatabaseCredentials
from utils.database_helpers import _DEFAULT_DB_QUERY_TIMEOUT, DatabaseOperator
from utils.logging_helper import get_logger
from utils.prompts import SYSTEM_PROMPT_DATAROBOT
from utils.schema import DataFrameWrapper

logger = get_logger("DatasetHandler")


class SparkRecipeError(RuntimeError):
    """
    Exception class for initializing/using Spark Recipe
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def find_underlying_client_message(exc: BaseException) -> str | None:
    stack: list[BaseException] = [exc]
    while stack:
        exc = stack.pop()
        if isinstance(exc, ClientError) and "message" in exc.json:
            return cast(str, exc.json["message"])
        stack.extend(
            [
                e
                for e in (
                    [exc.__cause__]
                    if exc.__cause__ is exc.__context__
                    else [exc.__cause__, exc.__context__]
                )
                if e is not None
            ]
        )
    return None


def retryable_recipe_preview_exception(exc: BaseException) -> bool:
    """A predicate on whether an exception raised in Recipe.preview is retryable

    Args:
        exc (BaseException): The exception.

    Returns:
        bool: True iff it is safe to retry previewing
    """
    return isinstance(exc, AsyncTimeoutError) or (
        isinstance(exc, ClientError)
        and (
            exc.json.get("status") == "ABORTED"
            or (
                exc.status_code == 404
                and exc.json.get("message") == "Preview is not ready yet"
            )
            or exc.status_code // 100 == 5
        )
    )


@contextmanager
def handle_datarobot_error(
    resource: str,
    exception_type: type[Exception] | None = SparkRecipeError,
    not_found_severity: int = logging.INFO,
    other_severity: int = logging.ERROR,
) -> Generator[None, None, None]:
    """
    A context manager that wraps and logs errors from a DataRobot call.

    Expected usage:
        with handle_data_robot_error(f"UseCase({use_case_id})"):
            use_case = UseCase.get(use_case_id)
    """
    try:
        yield
    except ClientError as e:
        if e.status_code == 404:
            message = f"{resource} not found (404.)"
            logger.log(not_found_severity, message, exc_info=True)
        else:
            message = f"Exception in retrieving {resource} ({e.status_code})."
            logger.log(other_severity, message, exc_info=True)
        if exception_type:
            raise exception_type(message) from e
        else:
            raise
    except InterruptedError:
        raise
    except BaseException as e:
        message = f"Unexpected exception in retrieving {resource}."
        logger.log(other_severity, message, exc_info=True)
        if exception_type:
            raise exception_type(message) from e
        else:
            raise


P = ParamSpec("P")
T = TypeVar("T")


def default_retry(func: Callable[P, T]) -> Callable[P, T]:
    return retry(
        wait=wait_random_exponential(),
        stop=stop_after_attempt(3),
        reraise=True,
        after=after_log(logger, logging.DEBUG),
    )(func)


@default_retry
async def load_or_create_spark_recipe(
    analyst_db: AnalystDB, initial_dataset_ids: list[str] = []
) -> SparkRecipe:
    """
    Load the recipe created for the user, if it has been persisted and is valid.
    Otherwise, create a recipe for the user and persist it.
    """
    key = analyst_db.user_id

    if result := load_or_create_spark_recipe.__dict__.setdefault("__cache__", {}).get(
        key
    ):
        return cast(SparkRecipe, result)

    # user id is a uuid, not anything identifiable.
    logger.debug("Looking up / creating recipe for user.", extra={"user": key})

    user_recipe = await analyst_db.get_user_recipe()

    recipe_exists = bool(user_recipe and user_recipe.recipe_id)

    recipe: Recipe

    if recipe_exists:
        logger.debug(
            "Recipe saved for user, checking that it exists.",
            extra={"user": key, "recipe": user_recipe and user_recipe.recipe_id},
        )
        with handle_datarobot_error(f"Recipe({user_recipe.recipe_id})"):  # type:ignore[union-attr]
            try:
                recipe = Recipe.get(user_recipe.recipe_id)  # type:ignore[union-attr]
            except ClientError as e:
                if e.status_code // 100 == 4:
                    logger.debug(
                        "Saved recipe is no longer valid.",
                        extra={
                            "user": key,
                            "recipe": user_recipe and user_recipe.recipe_id,
                        },
                        exc_info=True,
                    )
                    recipe_exists = False
                else:
                    raise

    if not recipe_exists:
        logger.debug("Creating recipe.", extra={"user": key})
        recipe = await create_new_recipe(analyst_db, initial_dataset_ids)

    spark_recipe = SparkRecipe(recipe=recipe, analyst_db=analyst_db)

    load_or_create_spark_recipe.__cache__[key] = spark_recipe  # type:ignore[attr-defined]

    return spark_recipe


async def create_new_recipe(
    analyst_db: AnalystDB, initial_dataset_ids: list[str]
) -> Recipe:
    if not initial_dataset_ids:
        raise RuntimeError("Cannot create a recipe from no datasets")

    dataset_id = initial_dataset_ids[0]

    with handle_datarobot_error(f"Dataset.get({dataset_id})"):
        dataset = Dataset.get(dataset_id)

    use_case_name = f"TalkToMyData Data Wrangling {analyst_db.user_id}"

    with handle_datarobot_error(f"UseCase.list({use_case_name})"):
        use_cases = UseCase.list()

    use_cases = [u for u in use_cases if u.name == use_case_name]

    if use_cases:
        use_cases.sort(key=lambda u: u.created_at, reverse=True)
        use_case = use_cases[0]
    else:
        logger.debug(
            "Use case for recipe not created, creating.",
            extra={"user": analyst_db.user_id, "use_case_name": use_case_name},
        )
        with handle_datarobot_error(f"UseCase.create({use_case_name})"):
            use_case = UseCase.create(
                use_case_name, "Recipe container for Talk To My Data user."
            )

    with handle_datarobot_error("Recipe.from_dataset(...)"):
        recipe = Recipe.from_dataset(
            use_case=use_case,
            dataset=dataset,
            inputs=[],
            dialect=DataWranglingDialect.SPARK,
            recipe_type=RecipeType.SQL,
        )

    with handle_datarobot_error("Recipe.set_inputs(...)"):
        Recipe.set_inputs(
            recipe.id,
            [
                RecipeDatasetInput(input_type=RecipeInputType.DATASET, dataset_id=ds)
                for ds in initial_dataset_ids
            ],
        )

    logger.debug(
        "Persisting created recipe for user.",
        extra={"user": analyst_db.user_id, "recipe": recipe.id},
    )
    await analyst_db.set_user_recipe(
        UserRecipe(user_id=analyst_db.user_id, recipe_id=recipe.id)
    )

    return recipe


try:
    try:
        raise RuntimeError("inner")
    except BaseException as e:
        raise RuntimeError("outer") from e
except BaseException as e:
    exc = e


class SparkRecipe:
    """
    A SparkRecipe exposes methods for initializing and performing Spark SQL queries
    with DataRobot recipes.
    """

    def __init__(self, analyst_db: AnalystDB, recipe: Recipe) -> None:
        self._analyst_db = analyst_db
        self._recipe = recipe
        self._lock = asyncio.Lock()

    def as_database_operator(self) -> DataRobotOperator:
        return DataRobotOperator(NoDatabaseCredentials(), self)

    @staticmethod
    @cache
    @default_retry
    def should_use_spark_recipe() -> bool:
        """Check if the API version is compatible with our usage."""
        version_response: str = dr.client.get_client().get("version/").content.decode()
        version = json.loads(version_response)["versionString"]
        logger.debug(
            "Checked if version is recent enough to have spark instance size configuration.",
            extra={"version": version, "expected_version": "2.38"},
        )

        return Version(version) >= Version("2.38")

    @default_retry
    async def refresh(self) -> bool:
        """Refreshes the underlying recipe, recreating if necessary. This ensures that the app's datasets and the recipe's input are in sync.

        Returns:
            (bool) True if any dataset in the app were deleted.
        """
        logger.debug("Refreshing recipe %s.", self._recipe.id)
        async with self._lock:
            expected_datasets = await self._analyst_db.list_analyst_dataset_metadata(
                DataSourceType.REMOTE_REGISTRY
            )

            if not expected_datasets:
                return False

            expected_dataset_ids = {
                ds.dataset_id for ds in expected_datasets if ds.dataset_id
            }

            deleted_datasets = False

            for d in expected_datasets:
                d_id = d.dataset_id
                if d_id is None:
                    continue

                with handle_datarobot_error(f"Dataset.get({d_id})"):
                    try:
                        Dataset.get(d_id)
                    except ClientError as e:
                        if e.status_code not in [404, 410]:
                            raise
                        expected_dataset_ids.remove(d_id)
                        deleted_datasets = True
                        await self._analyst_db.delete_table(d.name)

            recipe_missing = False

            try:
                self._recipe = Recipe.get(self._recipe.id)
            except ClientError as e:
                # 410 gone is raised if any inputs are deleted - the recipe is then in an inconsistent state and must be recreated.
                if e.status_code // 100 == 4:
                    raise
                recipe_missing = True

            recipe_inputs: list[RecipeDatasetInput] = [
                i for i in self._recipe.inputs if isinstance(i, RecipeDatasetInput)
            ]
            recipe_dataset_ids: set[str] = {i.dataset_id for i in recipe_inputs}

            if recipe_missing:
                self._recipe = await create_new_recipe(
                    analyst_db=self._analyst_db,
                    initial_dataset_ids=list(expected_dataset_ids),
                )
            # In order to ensure we keep the recipe's inputs in line with the user's selected datasets, we're here recreating the recipe
            # when a dataset is deselected. (You cannot remove the dataset the recipe was created from as an input, so simplest to recreate.)
            # This is not usually strictly necessary, but should prevent weird errors from cropping up, e.g. if the user adds a datset,
            # removes it, and then adds a different dataset with identical name.
            elif recipe_dataset_ids - expected_dataset_ids:
                # Just a basic effort on the delete, they're already siloed in their own use case, so this is more hygeine.
                try:
                    dr.client.get_client().delete(f"recipes/{self._recipe.id}")
                except ClientError:
                    logger.warning(
                        "Failed to delete %s", self._recipe.id, exc_info=True
                    )
                self._recipe = await create_new_recipe(
                    analyst_db=self._analyst_db,
                    initial_dataset_ids=list(expected_dataset_ids),
                )
            elif recipe_dataset_ids != expected_dataset_ids:
                self._set_inputs(expected_dataset_ids)

            return deleted_datasets

    def _set_inputs(self, expected_dataset_ids: Iterable[str]) -> None:
        with handle_datarobot_error(f"Recipe.set_inputs({self._recipe.id})"):
            self._recipe = Recipe.set_inputs(
                self._recipe.id,
                [
                    RecipeDatasetInput(
                        RecipeInputType.DATASET,
                        dataset_id=dataset_id,
                    )
                    for dataset_id in expected_dataset_ids
                ],
            )

    @default_retry
    def add_datasets(self, dataset_ids: list[str]) -> None:
        """This updates the recipe to add additional datasets in the use case and persists those datasets
        to the database. Note, this uses the default wrangling policy of using the latest dataset version.

        Args:
            dataset_ids (list[str]): The ids of the datasets to add.
        """
        logger.debug(
            "Adding datasets to recipe's input.", extra={"recipe_id": self._recipe.id}
        )
        recipe_inputs: list[RecipeDatasetInput] = [
            i for i in self._recipe.inputs if isinstance(i, RecipeDatasetInput)
        ]
        recipe_dataset_ids: set[str] = {i.dataset_id for i in recipe_inputs}

        additional_datasets = set(dataset_ids)

        if additional_datasets <= recipe_dataset_ids:
            return

        self._set_inputs(additional_datasets | recipe_dataset_ids)

    @default_retry
    def list_dataset_names(self) -> list[str]:
        """Return the names of datasets that are inputs to this recipe.

        Returns:
            list[str]: _description_
        """
        recipe_inputs = [
            input
            for input in self._recipe.inputs
            if isinstance(input, RecipeDatasetInput)
        ]
        return [i.alias for i in recipe_inputs if i.alias]

    def _set_large_spark_instance_size(self) -> None:
        """
        Update the recipe to use a large data size
        """
        # A new feature that hasn't been ported to SDK yet.
        logger.debug(
            "Setting recipe to large spark instance size.",
            extra={"recipe_id": self._recipe.id},
        )
        with handle_datarobot_error(f"PATCH recipes/{self._recipe.id}/settings"):
            dr.client.get_client().patch(
                f"recipes/{self._recipe.id}/settings", {"sparkInstanceSize": "large"}
            )

    @default_retry
    def set_query(self, query: str) -> None:
        """
        Update the recipe to use the given SQL query
        """
        self._set_large_spark_instance_size()
        logger.debug(
            "Setting recipe query.", extra={"recipe_id": self._recipe.id, "sql": query}
        )
        with handle_datarobot_error(f"Recipe.set_recipe_metadata({self._recipe.id})"):
            Recipe.set_recipe_metadata(self._recipe.id, {"sql": query})

    @default_retry
    def clear_query(self) -> None:
        """
        Update the recipe to not use any query.
        """
        with handle_datarobot_error(f"Recipe.set_recipe_metadata({self._recipe.id})"):
            Recipe.set_recipe_metadata(self._recipe.id, {})

    @retry(
        wait=wait_random_exponential(multiplier=2, max=300),
        retry=retry_if_exception(retryable_recipe_preview_exception),
        stop=stop_after_attempt(6),
        reraise=True,
        after=after_log(logger, logging.DEBUG),
    )
    def _retrieve_preview(self, timeout_seconds: int = 300) -> DataFrameWrapper:
        logger.debug(
            "Retrieving preview of recipe.", extra={"recipe_id": self._recipe.id}
        )
        preview = self._recipe.retrieve_preview(timeout_seconds)
        schema = preview["resultSchema"]

        # Unfortunately the Python SDK currently doesn't have a nice API for Previews
        all_rows: list[Any] = preview["data"]

        if preview.get("next"):
            logger.debug(
                "Fetching additional pages of preview.",
                extra={"recipe_id": self._recipe.id},
            )
            for row in unpaginate(
                preview["next"],
                initial_params=None,
                client=dr.client.get_client(),
            ):
                all_rows.append(row)

        return SparkRecipe.convert_preview_to_dataframe(schema, all_rows)

    def retrieve_preview(self, timeout_seconds: int = 900) -> DataFrameWrapper:
        """
        Retrieve a preview of the set SQL query.

        Args:
            timeout_seconds: How long to wait for results.
        """
        with handle_datarobot_error(f"Recipe.retrieve_preview({self._recipe.id})"):
            return self._retrieve_preview(timeout_seconds)

    def preview_dataset(
        self, dataset: Dataset, preview_limit: int = 1000
    ) -> DataFrameWrapper:
        """Preview the first `preview_limit` rows of a dataset (behind the hood queries and previews data.)

        Args:
            dataset (Dataset): The dataset to add
            preview_limit (int, optional): The maximum number of rows to return. Defaults to 1000.

        Returns:
            DataFrameWrapper: The first preview_limit rows.
        """
        self.set_query(f"SELECT * FROM `{dataset.name}` LIMIT {preview_limit}")
        return self.retrieve_preview()

    @staticmethod
    def convert_preview_to_dataframe(
        schema: list[dict[str, Any]], all_rows: list[Any]
    ) -> DataFrameWrapper:
        polars_schema = {
            col["name"]: SparkRecipe.map_datarobot_type_to_polars_type(col["dataType"])
            for col in schema
        }

        # Getting polars to ignore / set null invalid values is a bit of a chore.
        df = pl.DataFrame(
            all_rows, schema={k: str for k in polars_schema}, orient="row", strict=False
        )

        df = df.with_columns(
            [pl.col(col).cast(polars_schema[col], strict=False) for col in df.columns]
        )

        return DataFrameWrapper(df)

    @staticmethod
    def map_datarobot_type_to_polars_type(datarobot_type: str) -> type[pl.DataType]:
        """
        Return the matching Polars DataType for the given identifier of a DataRobot result type.
        """
        mapping = {
            "STRING_TYPE": pl.String,
            "INT_TYPE": pl.Int64,
            "DOUBLE_TYPE": pl.Float64,
        }
        return mapping.get(
            datarobot_type, pl.String
        )  # Default to String if type not found


class DataRobotOperator(DatabaseOperator[NoDatabaseCredentials]):
    """A wrapper around DataRobot's DataWrangling

    Args:
        DatabaseOperator (_type_): _description_
    """

    def __init__(
        self,
        credentials: NoDatabaseCredentials,
        recipe: SparkRecipe,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self.default_timeout = default_timeout
        self.recipe = recipe

    def _run_sql(self, sql_query: str, timeout: int) -> DataFrameWrapper:
        self.recipe.set_query(sql_query)

        return self.recipe.retrieve_preview(timeout_seconds=timeout)

    def execute_query(
        self,
        query: str,
        timeout: int | None = None,
        table_names: list[str] = [],
        **kwargs: Any,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a SQL query using DataRobot's Data Wrangling platform"""

        timeout = timeout if timeout is not None else self.default_timeout

        try:
            df = self._run_sql(query, timeout)
            return df.to_dict()  # TODO: this is silly as cast gets undone.

        except Exception as e:
            message = find_underlying_client_message(e)

            raise InvalidGeneratedCode(
                f"Query execution failed: {message}"
                if message
                else "Query execution failed.",
                code=query,
                exception=e,
                traceback_str=str(e.__traceback__),
            )

    @contextmanager
    def create_connection(self) -> Generator[None]:
        yield None

    @lru_cache(8)
    async def get_data(self, *args, **kwargs) -> Any:  # type:ignore[no-untyped-def]
        raise NotImplementedError()

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of available datasets from DataRobot"""
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            return self.recipe.list_dataset_names()

        except Exception as e:
            logger.error(f"Failed to fetch DataRobot dataset info: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system", content=SYSTEM_PROMPT_DATAROBOT
        )
