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

import abc
import asyncio
import datetime
import json
import logging
import threading
import uuid
from contextlib import contextmanager
from functools import cache, lru_cache
from typing import Any, Callable, Generator, Iterable, ParamSpec, TypeVar, cast

import datarobot as dr
import polars as pl
from datarobot.enums import (
    DataStoreListTypes,
    DataWranglingDataSourceTypes,
    DataWranglingDialect,
    RecipeInputType,
    RecipeType,
)
from datarobot.errors import AsyncTimeoutError, ClientError
from datarobot.models.credential import Credential
from datarobot.models.data_source import DataSource
from datarobot.models.data_store import DataStore
from datarobot.models.dataset import Dataset
from datarobot.models.recipe import (
    DataSourceInput,
    JDBCTableDataSourceInput,
    Recipe,
    RecipeDatasetInput,
)
from datarobot.models.use_cases.use_case import UseCase
from datarobot.utils.pagination import unpaginate
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from packaging.version import Version
from requests import HTTPError
from tenacity import (
    after_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from utils.analyst_db import (
    AnalystDB,
    ExternalDataStoreNameDataSourceType,
    InternalDataSourceType,
    UserRecipe,
)
from utils.api_exceptions import ApplicationUsageException, UsageExceptionType
from utils.code_execution import InvalidGeneratedCode
from utils.credentials import NoDatabaseCredentials
from utils.data_analyst_telemetry import telemetry
from utils.database_helpers import _DEFAULT_DB_QUERY_TIMEOUT, DatabaseOperator
from utils.logging_helper import get_logger
from utils.prompts import (
    PostgresPromptFactory,
    QueryPromptFactory,
    RedshiftPromptFactory,
    SparkPromptFactory,
)
from utils.schema import (
    DataFrameWrapper,
    ExternalDataSource,
    ExternalDataStore,
)

logger = get_logger(__name__)


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
        if isinstance(exc, HTTPError) and "message" in exc.response.json():
            return cast(str, exc.response.json()["message"])
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
        retry=retry_if_exception(
            lambda ex: not isinstance(ex, ApplicationUsageException)
        ),
        after=after_log(logger, logging.DEBUG),
    )(func)


@default_retry
@telemetry.trace
async def load_or_create_spark_recipe(
    analyst_db: AnalystDB, initial_dataset_ids: list[str] = []
) -> DatasetSparkRecipe:
    """
    Load the recipe created for the user, if it has been persisted and is valid.
    Otherwise, create a recipe for the user and persist it.
    """
    key = analyst_db.user_id

    if result := load_or_create_spark_recipe.__dict__.setdefault("__cache__", {}).get(
        key
    ):
        return cast(DatasetSparkRecipe, result)

    # Time inside the cache.
    with telemetry.time(
        f"{load_or_create_spark_recipe.__module__}.{load_or_create_spark_recipe.__name__}"
    ):
        # user id is a uuid, not anything identifiable.
        logger.debug("Looking up / creating recipe for user.", extra={"user": key})

        user_recipe = await analyst_db.get_user_recipe()

        recipe_exists = bool(user_recipe and user_recipe.recipe_id)

        maybe_recipe: Recipe | None = None

        if recipe_exists:
            logger.debug(
                "Recipe saved for user, checking that it exists.",
                extra={"user": key, "recipe": user_recipe and user_recipe.recipe_id},
            )
            maybe_recipe = lookup_recipe(key, user_recipe)  # type:ignore[arg-type]

        if maybe_recipe:
            recipe = maybe_recipe
        else:
            logger.debug("Creating recipe.", extra={"user": key})
            current_datasets = await analyst_db.list_analyst_dataset_metadata(
                data_source=InternalDataSourceType.REMOTE_REGISTRY
            )
            current_dataset_ids = {
                d.external_id for d in current_datasets if d.external_id
            }
            recipe = await create_new_recipe(
                analyst_db, list(current_dataset_ids | set(initial_dataset_ids))
            )

        spark_recipe = DatasetSparkRecipe(recipe=recipe, analyst_db=analyst_db)

    load_or_create_spark_recipe.__cache__[key] = spark_recipe  # type:ignore[attr-defined]

    return spark_recipe


@telemetry.trace
def lookup_recipe(user_id: str, user_recipe: UserRecipe) -> Recipe | None:
    recipe = None
    with handle_datarobot_error(f"Recipe({user_recipe.recipe_id})"):
        try:
            recipe = Recipe.get(user_recipe.recipe_id)
        except ClientError as e:
            if e.status_code // 100 == 4:
                logger.debug(
                    "Saved recipe is no longer valid.",
                    extra={
                        "user": user_id,
                        "recipe": user_recipe.recipe_id,
                    },
                    exc_info=True,
                )
            else:
                raise
    return recipe


@telemetry.trace
async def create_new_recipe(
    analyst_db: AnalystDB, initial_dataset_ids: list[str]
) -> Recipe:
    if not initial_dataset_ids:
        raise ApplicationUsageException(
            UsageExceptionType.RECIPE_NOT_INITIALIZED,
            "Cannot create a recipe from no datasets",
        )

    dataset_id = initial_dataset_ids[0]

    with handle_datarobot_error(f"Dataset.get({dataset_id})"):
        dataset = Dataset.get(dataset_id)

    use_case = get_or_create_wrangling_use_case(analyst_db)

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
        UserRecipe(user_id=analyst_db.user_id, recipe_id=recipe.id, datastore_id=None)
    )

    return recipe


@telemetry.trace
def get_or_create_wrangling_use_case(analyst_db: AnalystDB) -> UseCase:
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

    return use_case


class BaseRecipe(abc.ABC):
    MAX_ROWS = 1000

    def __init__(self, analyst_db: AnalystDB, recipe: Recipe | None) -> None:
        self._analyst_db = analyst_db
        self._recipe = recipe
        self._lock = asyncio.Lock()

    @classmethod
    @cache
    def should_use_spark_recipe(cls) -> bool:
        """Check if the API version is compatible with our usage."""
        return True

    @abc.abstractmethod
    async def refresh(self) -> bool:
        """Refreshes the underlying recipe, recreating if necessary. This ensures that the app's datasets and the recipe's input are in sync.

        Returns:
            (bool) True if any dataset in the app were deleted.
        """
        pass

    @abc.abstractmethod
    async def _configure(self) -> None:
        """Additional initialization before preview"""
        pass

    @default_retry
    @telemetry.trace
    async def list_dataset_names(self) -> list[str]:
        """Return the names of datasets that are inputs to this recipe.

        Returns:
            list[str]: _description_
        """
        await self._ensure_recipe_initialized()
        assert self._recipe is not None
        recipe_inputs = [input for input in self._recipe.inputs]
        return [i.alias for i in recipe_inputs if i.alias]

    async def _ensure_recipe_initialized(self) -> None:
        await self.refresh()
        if not self._recipe:
            raise ApplicationUsageException(UsageExceptionType.RECIPE_NOT_INITIALIZED)

    @default_retry
    @telemetry.trace
    async def set_query(self, query: str) -> None:
        """
        Update the recipe to use the given SQL query
        """
        logger.debug(
            "Setting recipe query: checking recipe initialized.",
            extra={"recipe_id": self._recipe and self._recipe.id, "query": query},
        )
        await self._ensure_recipe_initialized()
        assert self._recipe is not None

        await self._configure()
        logger.debug(
            "Setting recipe query.", extra={"recipe_id": self._recipe.id, "sql": query}
        )
        with handle_datarobot_error(f"Recipe.set_recipe_metadata({self._recipe.id})"):
            Recipe.set_recipe_metadata(self._recipe.id, {"sql": query})

    @default_retry
    @telemetry.trace
    async def clear_query(self) -> None:
        """
        Update the recipe to not use any query.
        """
        await self._ensure_recipe_initialized()
        assert self._recipe is not None
        with handle_datarobot_error(f"Recipe.set_recipe_metadata({self._recipe.id})"):
            Recipe.set_recipe_metadata(self._recipe.id, {})

    @retry(
        wait=wait_random_exponential(multiplier=2, max=300),
        retry=retry_if_exception(retryable_recipe_preview_exception),
        stop=stop_after_attempt(6),
        reraise=True,
        after=after_log(logger, logging.DEBUG),
    )
    @telemetry.trace
    async def _retrieve_preview(self, timeout_seconds: int = 300) -> DataFrameWrapper:
        assert self._recipe is not None
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
                preview["next"], initial_params=None, client=dr.client.get_client()
            ):
                all_rows.append(row)
                # Normally queries should be limited in how much data they return, this serves as a backstop.
                if len(all_rows) >= BaseRecipe.MAX_ROWS:
                    break

        return DatasetSparkRecipe.convert_preview_to_dataframe(schema, all_rows)

    @telemetry.meter_and_trace
    async def retrieve_preview(self, timeout_seconds: int = 900) -> DataFrameWrapper:
        """
        Retrieve a preview of the set SQL query.

        Args:
            timeout_seconds: How long to wait for results.
        """
        with handle_datarobot_error(
            f"Recipe.retrieve_preview({self._recipe and self._recipe.id})"
        ):
            return await self._retrieve_preview(timeout_seconds)

    @property
    @abc.abstractmethod
    def prompt(self) -> QueryPromptFactory:
        pass

    @property
    def catalog(self) -> str | None:
        return None

    @property
    def schema(self) -> str | None:
        return None

    def as_database_operator(self) -> DataRobotOperator:
        return DataRobotOperator(NoDatabaseCredentials(), self)

    @staticmethod
    def convert_preview_to_dataframe(
        schema: list[dict[str, Any]], all_rows: list[Any]
    ) -> DataFrameWrapper:
        polars_schema = {
            col["name"]: DatasetSparkRecipe.map_datarobot_type_to_polars_type(
                col["dataType"]
            )
            for col in schema
        }

        # Getting polars to ignore / set null invalid values is a bit of a chore.
        df = pl.DataFrame(
            all_rows, schema={k: str for k in polars_schema}, orient="row", strict=False
        )

        df = DatasetSparkRecipe.unstrict_cast_dataframe_to_schema(df, polars_schema)

        return DataFrameWrapper(df)

    @staticmethod
    def unstrict_cast_dataframe_to_schema(
        dataframe: pl.DataFrame, schema: dict[str, type[pl.DataType]]
    ) -> pl.DataFrame:
        """
        Cast data frame to schema. This will be lax in that failed casts are set to null.
        If none of a column can be cast, the column will retain its original type.

        Args:
            dataframe (pl.DataFrame): The dataframe.
            schema (dict[str, type[pl.DataType]]): The column types.

        Returns:
            pl.DataFrame: The cast types
        """
        cast_expr = [
            pl.col(col).str.to_time(strict=False)
            if schema[col] == pl.Time
            else pl.col(col).str.to_date(strict=False)
            if schema[col] == pl.Date
            else pl.col(col).str.to_datetime(strict=False, ambiguous="earliest")
            if schema[col] == pl.Datetime
            else pl.when(
                pl.col(col).str.to_lowercase().is_in(["true", "t", "1", "yes", "y"])
            )
            .then(True)
            .otherwise(False)
            .alias(col)
            if schema[col] == pl.Boolean
            else pl.col(col).cast(schema[col], strict=False)
            for col in dataframe.columns
        ]

        df = dataframe.with_columns(cast_expr)

        all_null_cols = [
            col for col, count in df.count().to_dict().items() if count.sum() == 0
        ]

        logger.debug(
            "Found null colls, replacing them with original.",
            extra={"all_null_cols": all_null_cols},
        )

        for col in all_null_cols:
            df._replace(col, dataframe[col])

        return df

    @staticmethod
    def map_datarobot_type_to_polars_type(datarobot_type: str) -> type[pl.DataType]:
        """
        Return the matching Polars DataType for the given identifier of a DataRobot result type.
        """
        spark_mapping = {
            "STRING_TYPE": pl.String,
            "INT_TYPE": pl.Int64,
            "DOUBLE_TYPE": pl.Float64,
        }

        if polars_type := spark_mapping.get(datarobot_type):
            return polars_type

        # For JDBC data connections, DataRobot gives the native type name of the source driver.
        # There's enough consistency among SQL variants that we should be able to get away without having to have
        # a map per connector.
        jdbc_prefix_to_type: dict[str, type[pl.DataType]] = {
            "REAL": pl.Float64,
            "NUMERIC": pl.Float64,
            "DECIMAL": pl.Float64,
            "FLOAT": pl.Float64,
            "DOUBLE": pl.Float64,
            "INT128": pl.Int128,
            "INT": pl.Int64,  # To cover INT2/4/8...
            "SMALLINT": pl.Int64,
            "BIGINT": pl.Int128,
            "BIT": pl.Boolean,
            "BOOL": pl.Boolean,  # also covers "Boolean"
            "BYTE": pl.Binary,  # also covers BYTEA/BYTEARRAY
            "VARCHAR": pl.String,
            "NVARCHAR": pl.String,
            "CHAR": pl.String,
            "NCHAR": pl.String,
            "TEXT": pl.String,
            "NTEXT": pl.String,
            "BPCHAR": pl.String,
            "DATE": pl.Datetime,
            "TIME": pl.Datetime,
        }

        matches = []
        for jdbc_prefix in jdbc_prefix_to_type:
            if datarobot_type.upper().startswith(jdbc_prefix):
                matches.append(jdbc_prefix)

        if matches:
            best_prefix = max(matches, key=len)
            return jdbc_prefix_to_type[best_prefix]

        logger.warning(
            "No match found for DataRobot type '%s' defined.", datarobot_type
        )

        return pl.String


class DataSourceRecipe(BaseRecipe):
    """
    A recipe defined over data sources.
    """

    # In order to add support for a new driver, there's two steps
    # 1. Create a corresponding `PromptFactory` in `prompts.py`.
    SUPPORTED_DRIVER_CLASS_TYPES: list[str] = ["postgres", "redshift"]
    DRIVER_CLASS_TYPE_TO_DIALECT: dict[str, DataWranglingDialect] = {
        "postgres": DataWranglingDialect.POSTGRES,
        "redshift": DataWranglingDialect.POSTGRES,
    }
    PROMPTS: dict[str, QueryPromptFactory] = {
        "postgres": PostgresPromptFactory(),
        "redshift": RedshiftPromptFactory(),
    }
    INSTANCE_CACHE: dict[tuple[str, str], DataSourceRecipe] = {}
    EXTERNAL_DATA_STORE_CACHE: dict[
        str, tuple[datetime.datetime, list[ExternalDataStore]]
    ] = {}
    EXTERNAL_DATA_STORE_CACHE_MAX_SIZE = 1024
    PER_USER_LOCKS: dict[str, threading.Lock] = {}
    PER_USER_LOCKS_LOCK: threading.Lock = threading.Lock()
    EXTERNAL_DATA_STORE_REFRESH_INTERVAL: datetime.timedelta = datetime.timedelta(
        minutes=30
    )

    def __init__(
        self,
        analyst_db: AnalystDB,
        recipe: Recipe | None,
        data_store: ExternalDataStore,
    ) -> None:
        super().__init__(analyst_db, recipe)
        self._data_store = data_store

    @default_retry
    @staticmethod
    @telemetry.trace
    async def list_available_datastores(user_id: str) -> list[ExternalDataStore]:
        # We have a pretty basic caching strategy here to avoid driving unnecessary load as this is an expensive lookup.
        # Each replica will fetch for each user all the datastores that user has access to every 30 minutes.
        # Does mean that our load scales with numbers of replicas and users. Replicas could be avoided by using
        # a distributed caching layer (or using our DB as a cache), but that's overkill for the vast majority of our users.
        # Also throwing on a lock in case a user has multiple sessions open (or refreshes).
        # We have to use threading locking here rather than asyncio's lock as despite this being declared `async`, it is used in a sync context.
        # We could make this lookup less expensive by fanning out the lookouts across datasources/schema, but that is significantly
        # annoying to do without an async client and in practice the slowness just means that the external data sources won't appear for a while.
        with DataSourceRecipe.PER_USER_LOCKS_LOCK:
            maybe_lock = DataSourceRecipe.PER_USER_LOCKS.get(user_id)
            if maybe_lock:
                lock = maybe_lock
            else:
                lock = DataSourceRecipe.PER_USER_LOCKS.setdefault(
                    user_id, threading.Lock()
                )

        # Short timeout here so that we don't have a buildup of threads  all waiting on a lock
        if not lock.acquire(blocking=True, timeout=1):
            raise RuntimeError(
                "Timed out waiting for another thread to list datastores."
            )

        try:
            if cached := DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE.get(user_id):
                cache_time, cache_value = cached
                if (
                    datetime.datetime.now(datetime.timezone.utc) - cache_time
                    <= DataSourceRecipe.EXTERNAL_DATA_STORE_REFRESH_INTERVAL
                ):
                    return cache_value

            with telemetry.time(
                f"{DataSourceRecipe.list_available_datastores.__module__}.{DataSourceRecipe.list_available_datastores.__qualname__}"
            ):
                data_stores = DataSourceRecipe._fetch_data_stores()

                external_data_stores = []

                for data_store in data_stores:
                    driver: str | None = data_store.get("driverClassType")

                    data_store_obj = DataStore.from_server_data(data_store)

                    if (
                        driver not in DataSourceRecipe.SUPPORTED_DRIVER_CLASS_TYPES
                        or not data_store_obj.id
                        or not data_store_obj.canonical_name
                    ):
                        continue

                    external_data_store = ExternalDataStore(
                        id=data_store_obj.id,
                        canonical_name=data_store_obj.canonical_name,
                        driver_class_type=driver,
                        defined_data_sources=[],
                    )

                    cred = DataSourceRecipe._fetch_default_cred(data_store_obj, user_id)

                    if not cred:
                        continue

                    external_data_sources = DataSourceRecipe._fetch_tables(
                        user_id, data_store_obj, cred
                    )

                    if not external_data_sources:
                        continue

                    external_data_store.defined_data_sources = external_data_sources
                    external_data_stores.append(external_data_store)

                refresh_time = datetime.datetime.now(datetime.timezone.utc)

                with DataSourceRecipe.PER_USER_LOCKS_LOCK:
                    DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE[user_id] = (
                        refresh_time,
                        external_data_stores,
                    )
                    # Since we're synchronizing this, we should never get more than one over our size
                    if (
                        len(DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE)
                        > DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE_MAX_SIZE
                    ):
                        oldest_key = min(
                            DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE,
                            key=lambda uid: DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE[
                                uid
                            ][0],
                        )
                        del DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE[oldest_key]

            return external_data_stores
        finally:
            lock.release()

    @staticmethod
    @telemetry.trace
    def _fetch_data_stores() -> list[dict[str, Any]]:
        logger.debug("Listing user-accessible data stores.")
        with handle_datarobot_error("DataStore.list()"):
            # TODO: the field `driverClassType` isn't exposed by the SDK (yet).
            # Should be soon and this can just become DataStore.list
            data_stores: list[dict[str, Any]] = (
                dr.client.get_client()
                .get(
                    DataStore._path,
                    params=dict(type=str(DataStoreListTypes.ALL)),
                )
                .json()["data"]
            )

        return data_stores

    @staticmethod
    @telemetry.trace
    def _fetch_default_cred(
        data_store_obj: DataStore, user_id: str
    ) -> Credential | None:
        logger.debug(
            "Finding credential for datastore.",
            extra={"data_store_id": data_store_obj.id, "user_id": user_id},
        )

        with handle_datarobot_error(
            f"GET /credentials/associations/dataconnection:{data_store_obj.id}"
        ):
            response = dr.client.get_client().get(
                f"credentials/associations/dataconnection:{data_store_obj.id}/?orderBy=-isDefault"
            )
            if response.status_code in [404, 410]:
                logger.warning(
                    "Encountered error in fetching credentials %s (%d)",
                    response.content,
                    response.status_code,
                    extra={
                        "data_store_id": data_store_obj.id,
                        "user_id": user_id,
                    },
                )
                return None
            elif response.status_code == 200:
                pass
            else:
                response.raise_for_status()

        credentials = response.json()["data"]

        default_cred = next((c for c in credentials if c.get("isDefault")), None)
        return Credential.from_server_data(default_cred) if default_cred else None

    @staticmethod
    @telemetry.trace
    def _fetch_tables(
        user_id: str, data_store_obj: DataStore, cred: Credential
    ) -> list[ExternalDataSource] | None:
        logger.debug(
            "Listing schemas for datastore.",
            extra={"user_id": user_id, "data_store_id": data_store_obj.id},
        )
        with handle_datarobot_error(
            f"POST /externalDataStores/{data_store_obj.id}/schemas/"
        ):
            response = dr.client.get_client().post(
                f"externalDataStores/{data_store_obj.id}/schemas/",
                {"credentialId": cred.credential_id},
            )
            if response.status_code in [404, 410]:
                logger.warning(
                    "Encountered error in fetching schema %s (%d)",
                    response.content,
                    response.status_code,
                    extra={
                        "data_store_id": data_store_obj.id,
                        "user_id": user_id,
                    },
                )
                return None
            elif response.status_code == 200:
                pass
            else:
                response.raise_for_status()

        schemas = response.json()

        catalog: str | None = schemas.get("catalog")
        schemata: list[str] | None = schemas.get("schemas")
        schemata_catalogs: list[str] | None = schemas.get("catalogs")

        tables_payloads = []
        if schemata:
            for i, schema in enumerate(schemata):
                payload = {"schema": schema, "credentialId": cred.credential_id}
                if schemata_catalogs and i < len(schemata_catalogs):
                    payload["catalog"] = schemata_catalogs[i]
                elif catalog:
                    payload["catalog"] = catalog
                tables_payloads.append(payload)
        else:
            payload = {"credentialId": cred.credential_id}
            if catalog:
                payload["catalog"] = catalog
            tables_payloads.append(payload)

        external_data_sources: dict[str, ExternalDataSource] = {}

        logger.debug(
            "Listing tables for datastore schemas.",
            extra={
                "user_id": user_id,
                "data_store_id": data_store_obj.id,
                "catalog": catalog,
                "schemas": schemas,
            },
        )

        for payload in tables_payloads:
            with handle_datarobot_error(
                f"POST /externalDataStores/{data_store_obj}/tables/"
            ):
                response = dr.client.get_client().post(
                    f"externalDataStores/{data_store_obj.id}/tables/", payload
                )
                if response.status_code in [404, 410]:
                    logger.warning(
                        "Encountered error in fetching schema %s (%d)",
                        response.content,
                        response.status_code,
                        extra={
                            "data_store_id": data_store_obj.id,
                            "user_id": user_id,
                        },
                    )
                    continue
                elif response.status_code == 200:
                    pass
                else:
                    response.raise_for_status()

            tables = response.json()["tables"]

            for table in tables:
                if "name" not in table:
                    logger.warning(
                        "Expected field name not found in table, skipping.",
                        extra={
                            "data_store_id": data_store_obj.id,
                            "user_id": user_id,
                            "table": table,
                        },
                    )
                    continue
                external_data_source = ExternalDataSource(
                    data_store_id=data_store_obj.id,
                    database_catalog=table.get("catalog"),
                    database_schema=table.get("schema"),
                    database_table=table["name"],
                )

                external_data_sources[external_data_source.path] = external_data_source
        return list(external_data_sources.values())

    @staticmethod
    @default_retry
    @lru_cache(128)
    @telemetry.trace
    def get_id_for_data_store_canonical_name(data_store_name: str) -> str | None:
        logger.debug(
            "Searching for datastore with name",
            extra={"data_store_canonical_name": data_store_name},
        )
        with handle_datarobot_error(f"DataStore.list({data_store_name})"):
            stores = DataStore.list(typ=DataStoreListTypes.ALL, name=data_store_name)

        for store in stores:
            if (
                store.canonical_name
                and store.canonical_name.lower().strip()
                == data_store_name.lower().strip()
            ):
                return store.id
        return None

    @staticmethod
    @lru_cache(128)
    @default_retry
    @telemetry.trace
    def get_canonical_name_for_datastore_id(data_store_id: str) -> str | None:
        """Return the canonical name of a given data store.

        Args:
            data_store_id (str): The data store in question.

        Returns:
            str | None: The canonical name, or None if there's no such datastore.
        """
        logger.debug(
            "Searching for datastore with name",
            extra={"data_store_id": data_store_id},
        )
        with handle_datarobot_error(f"DataStore.get({data_store_id})"):
            try:
                store = DataStore.get(data_store_id)
            except ClientError as e:
                if e.status_code in [404, 410]:
                    store = None
                else:
                    raise

        return store.canonical_name if store else None

    @staticmethod
    @default_retry
    @telemetry.meter_and_trace
    async def load_or_create(
        analyst_db: AnalystDB, data_store_id: str
    ) -> DataSourceRecipe:
        """Either load persisted recipe/datastore or load from datarobot and persist.

        Args:
            analyst_db (AnalystDB): The database for the user.
            data_store_id (str): The datastore to load.

        Raises:
            ApplicationUsageException: If the data store does not exist in Data Robot.

        Returns:
            DataSourceSparkRecipe: A recipe handler for the datastore.
        """
        if instance := DataSourceRecipe.INSTANCE_CACHE.get(
            (analyst_db.user_id, data_store_id)
        ):
            return instance

        logger.debug(
            "Loading / Creating Data Store Recipe.",
            extra={"user_id": analyst_db.user_id, "data_store_id": data_store_id},
        )
        with handle_datarobot_error(f"DataStore.get({data_store_id})"):
            data_store_response = dr.client.get_client().get(
                f"{DataStore._path}{data_store_id}/"
            )

        if data_store_response.status_code in [400, 404, 410]:
            logger.warning(
                "Data store not found",
                extra={"user_id": analyst_db.user_id, "data_store_id": data_store_id},
            )
            # Deregister to make sure we don't have any lingering data stores
            await analyst_db.delete_data_store(data_store_id)
            raise ApplicationUsageException(UsageExceptionType.NOT_FOUND)

        driver_class_type = data_store_response.json().get("driverClassType")

        if driver_class_type not in DataSourceRecipe.SUPPORTED_DRIVER_CLASS_TYPES:
            raise ApplicationUsageException(
                UsageExceptionType.DRIVER_CLASS_NOT_SUPPORTED
            )

        data_store_obj = DataStore.from_server_data(data_store_response.json())

        if not data_store_obj.id or not data_store_obj.canonical_name:
            logger.warning(
                "Data store ID/Name missing.", extra=data_store_response.json()
            )
            raise ApplicationUsageException(UsageExceptionType.DATA_SOURCE_INVALID)

        data_store = ExternalDataStore(
            id=data_store_obj.id,
            canonical_name=data_store_obj.canonical_name,
            driver_class_type=driver_class_type,
            defined_data_sources=[],
        )

        logger.debug(
            "Persisting retrieved data store.",
            extra={"user_id": analyst_db.user_id, "data_store_id": data_store_id},
        )
        await analyst_db.register_data_store(data_store)

        logger.debug(
            "Looking up stored recipe id.",
            extra={"user_id": analyst_db.user_id, "data_store_id": data_store_id},
        )

        recipe = await DataSourceRecipe._refresh_recipe(
            analyst_db, data_store_obj, data_store
        )

        return DataSourceRecipe.INSTANCE_CACHE.setdefault(
            (analyst_db.user_id, data_store_id),
            DataSourceRecipe(
                analyst_db=analyst_db, recipe=recipe, data_store=data_store
            ),
        )

    @staticmethod
    @telemetry.trace
    async def _refresh_recipe(
        analyst_db: AnalystDB, data_store_dr: DataStore, data_store: ExternalDataStore
    ) -> Recipe | None:
        logger.debug(
            "Retrieving stored recipe.",
            extra={
                "user_id": analyst_db.user_id,
                "data_store_id": data_store.id,
            },
        )
        user_recipe = await analyst_db.get_user_recipe(data_store_id=data_store.id)

        recipe = None
        if user_recipe:
            logger.debug(
                "Retrieved stored recipe.",
                extra={
                    "user_id": analyst_db.user_id,
                    "data_store_id": data_store.id,
                    "recipe_id": user_recipe.recipe_id,
                },
            )
            recipe = lookup_recipe(analyst_db.user_id, user_recipe)

        logger.debug(
            "Retrieving selected data sources.",
            extra={
                "user_id": analyst_db.user_id,
                "data_store_id": data_store.id,
            },
        )
        selected_data_sources = await analyst_db.list_sources_for_data_store(data_store)

        logger.debug("Adding registered datasets")
        registered_datasets = await analyst_db.list_analyst_dataset_metadata(
            ExternalDataStoreNameDataSourceType.from_name(data_store.canonical_name)
        )
        data_sources_by_path = {d.path: d for d in selected_data_sources}
        for ds in registered_datasets:
            data_sources_by_path[ds.name] = ExternalDataSource.from_path(
                path=ds.name, data_store_id=data_store.id
            )

        selected_data_sources = list(data_sources_by_path.values())

        data_store.defined_data_sources = selected_data_sources

        if selected_data_sources and not recipe:
            logger.debug(
                "Recipe missing / not created. (Re)creating.",
                extra={"user_id": analyst_db.user_id, "data_store_id": data_store.id},
            )
            use_case = get_or_create_wrangling_use_case(analyst_db)
            logger.debug(
                "Fetched / created use case for recipe. Creating recipe",
                extra={
                    "user_id": analyst_db.user_id,
                    "data_store_id": data_store.id,
                    "use_case_id": use_case.id,
                },
            )
            with handle_datarobot_error("Recipe.from_data_store()"):
                uuid4 = str(uuid.uuid4())
                recipe = Recipe.from_data_store(
                    data_store=data_store_dr,
                    use_case=use_case,
                    dialect=DataSourceRecipe.DRIVER_CLASS_TYPE_TO_DIALECT[
                        data_store.driver_class_type
                    ],
                    data_source_inputs=[
                        DataSourceInput(
                            canonical_name=f"{uuid4}-{d.path}",
                            table=d.database_table,
                            schema=d.database_schema,
                            catalog=d.database_catalog,
                            sampling=None,
                        )
                        for d in selected_data_sources
                        if d.database_table
                    ],
                    recipe_type="sql",  # type:ignore[arg-type]
                    data_source_type=DataWranglingDataSourceTypes.JDBC
                    if data_store_dr.type and data_store_dr.type.lower() == "jdbc"
                    else DataWranglingDataSourceTypes.DR_DATABASE_V1,
                )

            logger.debug(
                "Persisting recipe id.",
                extra={
                    "user_id": analyst_db.user_id,
                    "data_store_id": data_store.id,
                    "recipe_id": recipe.id,
                },
            )
            await analyst_db.set_user_recipe(
                UserRecipe(
                    user_id=analyst_db.user_id,
                    recipe_id=recipe.id,
                    datastore_id=data_store.id,
                )
            )

        return recipe

    @default_retry
    @telemetry.trace
    async def refresh(self) -> bool:
        async with self._lock:
            logger.debug(
                "Refreshing datastore",
                extra={
                    "datastore_id": self._data_store.id,
                    "recipe_id": self._recipe.id if self._recipe else "NA",
                },
            )
            with handle_datarobot_error(f"DataStore.get({self._data_store.id})"):
                try:
                    data_store = DataStore.get(self._data_store.id)
                except ClientError as e:
                    if e.status_code in [404, 410]:
                        logger.warning(
                            f"Data store {self._data_store.id} is missing! Recipe invalid."
                        )
                        await self._analyst_db.delete_data_store(self._data_store.id)
                        DataSourceRecipe.EXTERNAL_DATA_STORE_CACHE.pop(
                            self._analyst_db.user_id, None
                        )
                        raise ApplicationUsageException(
                            UsageExceptionType.DATASETS_INVALID
                        )
                    else:
                        raise

            logger.debug(
                "Refreshing datastore recipe",
                extra={
                    "datastore_id": self._data_store.id,
                    "recipe_id": self._recipe.id if self._recipe else "NA",
                },
            )

            self._recipe = await DataSourceRecipe._refresh_recipe(
                self._analyst_db, data_store, self._data_store
            )
            return False

    @default_retry
    @telemetry.meter_and_trace
    async def select_data_sources(self, data_sources: list[ExternalDataSource]) -> None:
        """Update recipe with given data sources.

        Args:
            data_source_paths (list[str]): The path in form

        Raises:
            ApplicationUsageException: If any data sources are not found.
        """
        data_store_ids = {d.data_store_id for d in data_sources}
        if len(data_store_ids) > 1 or (
            data_sources and self._data_store.id not in data_store_ids
        ):
            raise ApplicationUsageException(UsageExceptionType.DATA_SOURCE_INVALID)

        logger.debug(
            "Clearing recipe.",
            extra={
                "data_store_id": self._data_store.id,
                "recipe_id": self._recipe.id if self._recipe else "NA",
                "user_id": self._analyst_db.user_id,
            },
        )

        # Reusing a recipe for data sources is convoluted and expensive as it's nontrivial to check that recipe's data source
        # inputs are "the same" locations as the incomoing data source (you'd have to do a lookup on all the inputs or such).
        # So we just delete and recreate a fresh recipe whenver the selected inputs are resubmitted. Other replicas will find out
        # that the recipe is deleted when they call their refresh. There's a mild race condition here (other replicas check after
        # this replica deletes the recipe). We've structured the operations here (first clearing then deleting) to (a) limit the risk
        # of another replica beating us to the punch and reregistering a stale recipe and (b) leaving the application in a more obviously
        # broken state (all datasets are cleared), that at least is easy to spot and address (just retry selection).
        self._data_store.defined_data_sources = data_sources

        await self._analyst_db.clear_data_store_sources(self._data_store)

        data_sources_to_clear = set()

        if self._recipe:
            data_sources_to_clear = {
                i.data_source_id
                for i in self._recipe.inputs
                if isinstance(i, JDBCTableDataSourceInput)
            }
            try:
                dr.client.get_client().delete(f"{Recipe._path}{self._recipe.id}/")
            except ClientError as e:
                if e.status_code // 100 == 4:
                    logger.warning(
                        "Failed to delete recipe.",
                        exc_info=True,
                        extra={
                            "data_store_id": self._data_store.id,
                            "recipe_id": self._recipe.id if self._recipe else "NA",
                            "user_id": self._analyst_db.user_id,
                        },
                    )
                else:
                    raise
            self._recipe = None

        logger.debug(
            "Persisting selected datasources",
            extra={
                "data_store_id": self._data_store.id,
                "user_id": self._analyst_db.user_id,
                "datasource_count": len(self._data_store.defined_data_sources),
            },
        )

        await self._analyst_db.register_data_store(self._data_store)

        logger.debug(
            "Best effort clearing garbage datasources",
            extra={
                "data_store_id": self._data_store.id,
                "user_id": self._analyst_db.user_id,
            },
        )

        for ds in data_sources_to_clear:
            try:
                dr.client.get_client().delete(
                    f"{DataSource._path}{ds}/"
                ).raise_for_status()
            except HTTPError as e:
                if (
                    e.response
                    and hasattr(e.response, "status_code")
                    and e.response.status_code // 100 == 4
                ):
                    logger.warning(
                        "Failed to delete recipe.",
                        exc_info=True,
                        extra={
                            "data_store_id": self._data_store.id,
                            "recipe_id": self._recipe.id if self._recipe else "NA",
                            "user_id": self._analyst_db.user_id,
                        },
                    )
            else:
                raise

        await self.refresh()

    async def _configure(self) -> None:
        return

    @property
    def catalog(self) -> str | None:
        for ds in self._data_store.defined_data_sources:
            if ds.database_catalog:
                return ds.database_catalog

        return None

    @property
    def schema(self) -> str | None:
        for ds in self._data_store.defined_data_sources:
            if ds.database_schema:
                return ds.database_schema

        return None

    @property
    def prompt(self) -> QueryPromptFactory:
        return DataSourceRecipe.PROMPTS[self._data_store.driver_class_type]

    @property
    def data_store(self) -> ExternalDataStore:
        return self._data_store

    async def preview_datasource(
        self, dataset: ExternalDataSource, preview_limit: int = 1000
    ) -> DataFrameWrapper:
        """Preview the first `preview_limit` rows of a datasource (behind the hood queries and previews data.)

        Args:
            dataset (Dataset): The dataset to add
            preview_limit (int, optional): The maximum number of rows to return. Defaults to 1000.

        Returns:
            DataFrameWrapper: The first preview_limit rows.
        """
        await self._ensure_recipe_initialized()

        dataset_identifier = self.prompt.adapt_table_path(dataset.path)
        await self.set_query(
            f"SELECT * FROM {dataset_identifier} LIMIT {preview_limit}"
        )
        return await self.retrieve_preview()


class DatasetSparkRecipe(BaseRecipe):
    """
    A SparkRecipe exposes methods for initializing and performing Spark SQL queries
    with DataRobot recipes.
    """

    def __init__(self, analyst_db: AnalystDB, recipe: Recipe | None) -> None:
        super().__init__(analyst_db, recipe)

    @classmethod
    @cache
    @default_retry
    def should_use_spark_recipe(cls) -> bool:
        """Check if the API version is compatible with our usage."""
        version_response: str = dr.client.get_client().get("version/").content.decode()
        version = json.loads(version_response)["versionString"]
        logger.debug(
            "Checked if version is recent enough to have spark instance size configuration.",
            extra={"version": version, "expected_version": "2.38"},
        )

        return Version(version) >= Version("2.38")

    @default_retry
    @telemetry.trace
    async def refresh(self) -> bool:
        """Refreshes the underlying recipe, recreating if necessary. This ensures that the app's datasets and the recipe's input are in sync.

        Returns:
            (bool) True if any dataset in the app were deleted.
        """
        logger.debug(
            "Refreshing recipe %s.", self._recipe.id if self._recipe else "N/A"
        )
        async with self._lock:
            expected_datasets = await self._analyst_db.list_analyst_dataset_metadata(
                InternalDataSourceType.REMOTE_REGISTRY
            )

            if not expected_datasets:
                return False

            expected_dataset_ids = {
                ds.external_id for ds in expected_datasets if ds.external_id
            }

            deleted_datasets = False

            for d in expected_datasets:
                d_id = d.external_id
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

            recipe_missing = self._recipe is None

            if self._recipe:
                try:
                    self._recipe = Recipe.get(self._recipe.id)
                except ClientError as e:
                    # 410 gone is raised if any inputs are deleted - the recipe is then in an inconsistent state and must be recreated.
                    if e.status_code // 100 == 4:
                        raise
                    recipe_missing = True

            recipe_inputs: list[RecipeDatasetInput] = (
                [i for i in self._recipe.inputs if isinstance(i, RecipeDatasetInput)]
                if self._recipe
                else []
            )
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
                if self._recipe:
                    try:
                        dr.client.get_client().delete(f"recipes/{self._recipe.id}")
                    except ClientError:
                        logger.warning(
                            "Failed to delete %s",
                            self._recipe.id,
                            exc_info=True,
                        )
                self._recipe = await create_new_recipe(
                    analyst_db=self._analyst_db,
                    initial_dataset_ids=list(expected_dataset_ids),
                )
            elif recipe_dataset_ids != expected_dataset_ids:
                await self._set_inputs(expected_dataset_ids)

            return deleted_datasets

    async def _set_inputs(self, expected_dataset_ids: Iterable[str]) -> None:
        if not expected_dataset_ids:
            logger.debug("Skipping _set_inputs as no inputs were provided.")
            return

        await self._ensure_recipe_initialized()
        assert self._recipe is not None

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
    @telemetry.trace
    async def add_datasets(self, dataset_ids: list[str]) -> None:
        """This updates the recipe to add additional datasets in the use case and persists those datasets
        to the database. Note, this uses the default wrangling policy of using the latest dataset version.

        Args:
            dataset_ids (list[str]): The ids of the datasets to add.
        """
        await self._ensure_recipe_initialized()
        assert self._recipe is not None

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

        await self._set_inputs(additional_datasets | recipe_dataset_ids)

    async def _set_large_spark_instance_size(self) -> None:
        """
        Update the recipe to use a large data size
        """
        await self._ensure_recipe_initialized()
        assert self._recipe is not None

        # A new feature that hasn't been ported to SDK yet.
        logger.debug(
            "Setting recipe to large spark instance size.",
            extra={"recipe_id": self._recipe.id},
        )
        with handle_datarobot_error(f"PATCH recipes/{self._recipe.id}/settings"):
            dr.client.get_client().patch(
                f"recipes/{self._recipe.id}/settings", {"sparkInstanceSize": "large"}
            )

    async def _configure(self) -> None:
        await self._set_large_spark_instance_size()

    @property
    def prompt(self) -> QueryPromptFactory:
        return SparkPromptFactory()

    async def preview_dataset(
        self, dataset: Dataset, preview_limit: int = 1000
    ) -> DataFrameWrapper:
        """Preview the first `preview_limit` rows of a dataset (behind the hood queries and previews data.)

        Args:
            dataset (Dataset): The dataset to add
            preview_limit (int, optional): The maximum number of rows to return. Defaults to 1000.

        Returns:
            DataFrameWrapper: The first preview_limit rows.
        """
        await self.set_query(f"SELECT * FROM `{dataset.name}` LIMIT {preview_limit}")
        return await self.retrieve_preview()


class DataRobotOperator(DatabaseOperator[NoDatabaseCredentials]):
    """A wrapper around DataRobot's DataWrangling

    Args:
        DatabaseOperator (_type_): _description_
    """

    def __init__(
        self,
        credentials: NoDatabaseCredentials,
        recipe: BaseRecipe,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self.default_timeout = default_timeout
        self.recipe = recipe

    async def _run_sql(self, sql_query: str, timeout: int) -> DataFrameWrapper:
        logger.debug("Running SQL on Recipe.", extra={"sql": sql_query})
        await self.recipe.set_query(sql_query)
        logger.debug("Set query SQL, awaiting results.", extra={"sql": sql_query})
        return await self.recipe.retrieve_preview(timeout_seconds=timeout)

    async def execute_query(
        self,
        query: str,
        timeout: int | None = None,
        table_names: list[str] = [],
        **kwargs: Any,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a SQL query using DataRobot's Data Wrangling platform"""

        timeout = timeout if timeout is not None else self.default_timeout

        try:
            df = await self._run_sql(query, timeout)
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

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of available datasets from DataRobot"""
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            return await self.recipe.list_dataset_names()

        except Exception as e:
            logger.error(f"Failed to fetch DataRobot dataset info: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=self.recipe.prompt.prompt_for_datasource(
                catalog=self.recipe.catalog, schema=self.recipe.schema
            ),
        )

    def query_friendly_name(self, dataset_name: str) -> str:
        return self.recipe.prompt.adapt_table_path(dataset_name)
