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

import ast
import asyncio
import copy
import inspect
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from types import ModuleType
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    TypeVar,
    cast,
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import psutil
import scipy
import sklearn
import statsmodels as sm
from datarobot.models.dataset import Dataset
from fastapi import Request
from joblib import Memory
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from plotly.subplots import make_subplots
from pydantic import BaseModel, ValidationError

from core.api_exceptions import ApplicationUsageException, UsageExceptionType
from core.chat_dataset_helper import extract_and_store_datasets
from core.constants import (
    ALTERNATIVE_LLM_BIG,
    ALTERNATIVE_LLM_SMALL,
    DICTIONARY_BATCH_SIZE,
    DICTIONARY_PARALLEL_BATCH_SIZE,
    DICTIONARY_TIMEOUT,
    DISK_CACHE_LIMIT_BYTES,
    MAX_CSV_TOKENS,
    MAX_REGISTRY_DATASET_SIZE,
    REGISTRY_DATASET_SIZE_CUTOFF,
    VALUE_ERROR_MESSAGE,
)
from core.data_connections.database.database_implementations import (
    get_external_database,
)
from core.data_connections.datarobot.datarobot_dataset_handler import (
    BaseRecipe,
    DatasetSparkRecipe,
    DataSourceRecipe,
    load_or_create_spark_recipe,
)
from core.datarobot_client import use_user_token
from core.llm_client import AsyncLLMClient
from core.token_tracking import (
    TiktokenCountingStrategy,
    TokenUsageTracker,
    count_messages_tokens,
    estimate_csv_rows_for_token_limit,
)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from core import prompts, tools
from core.analyst_db import (
    AnalystDB,
    DatasetMetadata,
    DataSourceType,
    ExternalDataStoreNameDataSourceType,
    InternalDataSourceType,
    get_data_source_type,
)
from core.code_execution import (
    InvalidGeneratedCode,
    MaxReflectionAttempts,
    execute_python,
    reflect_code_generation_errors,
)
from core.data_cleansing_helpers import (
    add_summary_statistics,
    process_column,
)
from core.data_connections.database.database_interface import DatabaseOperator
from core.logging_helper import get_logger, log_api_call
from core.schema import (
    AnalysisError,
    AnalystChatMessage,
    AnalystDataset,
    BusinessAnalysisGeneration,
    ChartGenerationExecutionResult,
    ChatRequest,
    CleansedDataset,
    CodeGeneration,
    Component,
    DatabaseAnalysisCodeGeneration,
    DataDictionary,
    DataDictionaryColumn,
    DataRegistryDataset,
    DictionaryGeneration,
    DownloadedRegistryDataset,
    EnhancedQuestionGeneration,
    ExternalDataSource,
    ExternalDataSourcesSelection,
    GetBusinessAnalysisMetadata,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunAnalysisResultMetadata,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
    RunDatabaseAnalysisResult,
    RunDatabaseAnalysisResultMetadata,
    TokenUsageInfo,
    Tool,
    UsageInfoComponent,
)

from .telemetry import otel

logger = get_logger()
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.http_client").setLevel(logging.WARNING)


def log_memory() -> None:
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Memory usage: {memory:.2f} MB")


_memory = Memory(tempfile.gettempdir(), verbose=0)
_memory.clear(warn=False)  # clear cache on startup

T = TypeVar("T")


def cache(f: T) -> T:
    """Cache function and coroutine results to disk using joblib."""
    cached_f = _memory.cache(f)

    if asyncio.iscoroutinefunction(f):

        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = await cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"
                )
            return result

        return cast(T, awrapper)
    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            in_cache = cached_f.check_call_in_cache(*args, **kwargs)
            result = cached_f(*args, **kwargs)
            if not in_cache:
                _memory.reduce_size(DISK_CACHE_LIMIT_BYTES)
            else:
                logger.info(
                    f"Using previously cached result for function `{f.__name__}`"  # type: ignore[attr-defined]
                )
            return result

        return cast(T, wrapper)


# This can be large as we are not storing the actual datasets in memory, just metadata
@otel.meter_and_trace
def list_registry_datasets(
    remote: bool = False, limit: int = 100
) -> list[DataRegistryDataset]:
    """Fetch datasets from Data Registry with specified limit

    Args:
        filter_downloadable (bool, optional): Include only downloadable datasets. Defaults to False.
        limit (int, optional): _description_. Defaults to 100.

    Returns:
        list[DataRegistryDataset]: _description_
    """

    datasets = list(Dataset.iterate(limit=limit, filter_failed=True))

    return [
        DataRegistryDataset(
            id=ds.id,
            name=ds.name,
            created=(_year_month_day(ds.created_at) if ds.created_at else "N/A"),
            size=(f"{ds.size / (1024 * 1024):.1f} MB" if ds.size else "N/A"),
        )
        for ds in datasets
        if (remote and ds.is_data_engine_eligible and ds.is_snapshot)
        or (
            not remote
            and ds.size
            and ds.size <= REGISTRY_DATASET_SIZE_CUTOFF
            and ds.is_snapshot
        )
    ]


def _year_month_day(date: datetime | str) -> str:
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    return date.strftime("%Y-%m-%d")


@otel.trace
async def register_remote_registry_datasets(
    request: Request, dataset_ids: list[str], analyst_db: AnalystDB
) -> tuple[
    list[DownloadedRegistryDataset],
    list[tuple[Callable[..., Any], list[Any], dict[str, Any]]],
]:
    """Load selected datasets into the application, downloading the entire datasets.

    Args:
        dataset_ids (list[str]): The list of dataset IDs to load.
        analyst_db (AnalystDB): The database to register into

    Returns:
        tuple[list[AnalystDataset], list[tuple[Callable, list, dict]]: A tuple of
            1. a dictionary of dataset names and data and
            2. a list of callbacks + arguments to that callback to be run in the background
               to pull datasets.

    Raises:
        ValueError: If the loading cannot be performed. This can be either (a) the small datasets exceed
                    our size threshold, or (b) a remote dataset is invalid (e.g. it is not snapshotted)."""
    if not DatasetSparkRecipe.should_use_spark_recipe():
        logger.warning(
            "Attempted to register remote datasets in an unsupported feature (should be unreachable through UI)."
        )
        raise ApplicationUsageException(
            UsageExceptionType.FEATURE_NOT_SUPPORTED,
            "Cannot use remote datasets with an unsupported DataRobot API version.",
        )
    datasets = [Dataset.get(d_id) for d_id in dataset_ids]

    # Dynamic datasets cannot be used with data wrangling.
    invalid_remote_datasets = [ds for ds in datasets if not ds.is_data_engine_eligible]

    if invalid_remote_datasets:
        raise ApplicationUsageException(
            UsageExceptionType.DATASETS_INVALID,
            f"Cannot register remote, dynamic datasets: {[ds.name for ds in invalid_remote_datasets]}.",
        )

    existing_dataset_names = await find_existing_dataset_names(analyst_db, datasets)

    if existing_dataset_names:
        raise ApplicationUsageException(
            UsageExceptionType.DATASET_ALREADY_USED,
            f"Cannot register already registered datasets: {existing_dataset_names}.",
        )

    background_tasks: list[tuple[Callable[..., Any], list[Any], dict[str, Any]]] = []

    downloaded_datasets = []

    if dataset_ids:
        with use_user_token(request):
            recipe = await load_or_create_spark_recipe(analyst_db, dataset_ids)

            await recipe.refresh()  # Clear out any removed datasets.

        await recipe.add_datasets([ds.id for ds in datasets])

        for ds in datasets:
            await analyst_db.register_dataset(
                AnalystDataset(name=ds.name),
                InternalDataSourceType.REMOTE_REGISTRY,
                file_size=0,
                external_id=ds.id,
                clobber=False,
            )

        background_tasks.append(
            (register_remote_datasets, [request, recipe, analyst_db, datasets], {})
        )

        for ds in datasets:
            downloaded_datasets.append(DownloadedRegistryDataset(name=ds.name))

    return downloaded_datasets, background_tasks


async def find_existing_dataset_names(
    analyst_db: AnalystDB, datasets: list[Dataset]
) -> list[str]:
    dataset_names = {ds.name for ds in datasets}
    existing_names = set(await analyst_db.list_analyst_datasets())
    return list(dataset_names & existing_names)


@otel.trace
async def register_remote_datasets(
    request: Request,
    recipe: DatasetSparkRecipe,
    analyst_db: AnalystDB,
    datasets: list[Dataset],
) -> None:
    for dataset in datasets:
        with use_user_token(request):
            preview = await recipe.preview_dataset(dataset)
        analyst_dataset = AnalystDataset(name=dataset.name, data=preview.response)

        await analyst_db.register_dataset(
            analyst_dataset,
            InternalDataSourceType.REMOTE_REGISTRY,
            file_size=0,
            external_id=dataset.id,
            original_column_types=preview.original_types,
            clobber=True,
        )


@otel.trace
async def sync_data_sources_and_datasets(
    request: Request,
    canonical_name: str,
    analyst_db: AnalystDB,
    data_store_id: str,
    selected_datasource_ids: ExternalDataSourcesSelection,
) -> tuple[
    list[DownloadedRegistryDataset],
    list[tuple[Callable[..., Any], list[Any], dict[str, Any]]],
]:
    """
    Register any data sets for data sources *not already present*.

    Args:
        analyst_db (str): The database.
        data_source_id (str): The data source in question.

    Returns:
        tuple[list[AnalystDataset], list[tuple[Callable, list, dict]]: A tuple of
            1. a dictionary of dataset names and data and
            2. a list of callbacks + arguments to that callback to be run in the background
               to pull datasets.
    """
    logger.debug(
        "Syncing data sources and detaset.",
        extra={"data_store_id": data_store_id, "canonical_name": canonical_name},
    )
    datasets = await analyst_db.list_analyst_dataset_metadata(
        data_source=ExternalDataStoreNameDataSourceType.from_name(canonical_name)
    )

    already_registered_paths = {ds.name for ds in datasets}

    new_datasources = [
        ds
        for ds in selected_datasource_ids.selected_data_sources
        if ds.path not in already_registered_paths
    ]

    downloaded = []
    background_tasks: list[tuple[Callable[..., Any], list[Any], dict[str, Any]]] = []

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Initially registering data sources.",
            extra={
                "data_store_id": data_store_id,
                "canonical_name": canonical_name,
                "paths": [ds.path for ds in new_datasources],
            },
        )

    for ds in new_datasources:
        await analyst_db.register_dataset(
            AnalystDataset(name=ds.path),
            ExternalDataStoreNameDataSourceType.from_name(name=canonical_name),
            file_size=0,
            external_id=None,
            clobber=False,
        )
        downloaded.append(DownloadedRegistryDataset(name=ds.path))

    background_tasks.append(
        (register_datasource, [request, analyst_db, data_store_id, new_datasources], {})
    )

    return downloaded, background_tasks


@otel.meter_and_trace
async def register_datasource(
    request: Request,
    analyst_db: AnalystDB,
    data_store_id: str,
    datasources: list[ExternalDataSource],
) -> None:
    with use_user_token(request):
        recipe = await DataSourceRecipe.load_or_create(analyst_db, data_store_id)
        for ds in datasources:
            preview = await recipe.preview_datasource(ds)
            analyst_dataset = AnalystDataset(name=ds.path, data=preview.response)

            await analyst_db.register_dataset(
                analyst_dataset,
                ExternalDataStoreNameDataSourceType.from_name(
                    recipe.data_store.canonical_name
                ),
                file_size=0,
                external_id=None,
                clobber=True,
                original_column_types=preview.original_types,
            )


@otel.meter_and_trace
async def load_registry_datasets(
    dataset_ids: list[str],
    analyst_db: AnalystDB,
) -> list[DownloadedRegistryDataset]:
    """Load selected datasets into the application, downloading the entire datasets.

    Args:
        dataset_ids (list[str]): The list of dataset IDs to load.
        analyst_db (AnalystDB): The database to register into

    Returns:
        list[DownloadedRegistryDataset]: A list of dictionary of dataset names and data.

    Raises:
        ApplicationUsageException: If the loading cannot be performed. This can be either (a) the small datasets exceed
                                   our size threshold, or (b) a remote dataset is invalid (e.g. it is not snapshotted)
    """

    downloaded_datasets = []
    datasets = [Dataset.get(id_) for id_ in dataset_ids]

    if (
        sum([ds.size for ds in datasets if ds.size is not None])
        > MAX_REGISTRY_DATASET_SIZE
    ):
        raise ApplicationUsageException(
            UsageExceptionType.DATASETS_TOO_LARGE,
            f"The requested Data Registry datasets must total <= {int(MAX_REGISTRY_DATASET_SIZE)} bytes",
        )

    existing_datasets = await find_existing_dataset_names(analyst_db, datasets)

    if existing_datasets:
        raise ApplicationUsageException(
            UsageExceptionType.DATASET_ALREADY_USED,
            f"Some requested datasets are already present {existing_datasets}.",
        )

    for dataset in datasets:
        try:
            df = dataset.get_as_dataframe()
            result_dataset = AnalystDataset(name=dataset.name, data=df)
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            downloaded_datasets.append(
                DownloadedRegistryDataset(name=dataset.name, error=str(e))
            )
            continue

        await analyst_db.register_dataset(
            result_dataset, InternalDataSourceType.REGISTRY, dataset.size or 0
        )
        downloaded_datasets.append(DownloadedRegistryDataset(name=result_dataset.name))

    return downloaded_datasets


async def _get_dictionary_batch(
    columns: list[str],
    df: pl.DataFrame,
    batch_size: int = 5,
    token_tracker: TokenUsageTracker | None = None,
) -> list[DataDictionaryColumn]:
    """Process a batch of columns to get their descriptions"""

    # Get sample data and stats for just these columns
    # Convert timestamps to ISO format strings for JSON serialization
    try:
        logger.debug(f"Processing batch of {len(columns)} columns")
        sample_data = {}
        logger.debug("Converting datetime columns to ISO format")
        num_samples = 10
        for col in columns:
            if df[col].dtype.is_temporal():
                # Convert timestamps to ISO format strings
                sample_data[col] = (
                    df.select(
                        pl.col(col)
                        .cast(pl.Datetime)
                        .map_elements(
                            lambda x: x.isoformat() if x is not None else None
                        )
                    )
                    .head(num_samples)
                    .to_dict()
                )
            else:
                # For non-datetime columns, just take the samples as is
                sample_data[col] = df.select(pl.col(col)).head(num_samples).to_dict()

        # Handle numeric summary
        numeric_summary = {}
        logger.debug("Calculating numeric summaries")
        for col in columns:
            if df[col].dtype.is_numeric():
                desc = df[col].describe()
                numeric_summary[col] = desc.to_dict()

        # Get categories for non-numeric columns
        categories = []
        logger.debug("Getting categories for non-numeric columns")
        for column in columns:
            if not df[column].dtype.is_numeric():
                try:
                    value_counts = (
                        df[column].sample(n=1000, seed=42).value_counts().head(10)
                    )
                    # Convert any timestamp values to strings
                    if df[column].dtype.is_temporal():
                        value_counts[column] = value_counts[column].cast(pl.String)
                    categories.append({column: value_counts[column].to_list()})
                except Exception:
                    continue

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=prompts.SYSTEM_PROMPT_GET_DICTIONARY
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Data:\n{sample_data}\n"
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Statistical Summary:\n{numeric_summary}\n"
            ),
        ]

        if categories:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user", content=f"Categorical Values:\n{categories}\n"
                )
            )
        logger.debug(
            f"total_characters: {len(''.join([str(msg) for msg in messages]))}"
        )
        # Get descriptions from OpenAI
        async with AsyncLLMClient(token_tracker=token_tracker) as client:
            with otel.time(
                f"{_get_dictionary_batch.__module__}.{_get_dictionary_batch.__qualname__}.llm_call"
            ):
                completion: DictionaryGeneration = await client.chat.completions.create(
                    response_model=DictionaryGeneration,
                    model=ALTERNATIVE_LLM_SMALL,
                    messages=messages,
                    timeout=900,
                )

        # Convert to dictionary format
        descriptions = completion.to_dict()

        # Only return descriptions for requested columns
        return [
            DataDictionaryColumn(
                column=col,
                description=descriptions.get(col, "No description available"),
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]

    except ValueError as e:
        logger.error(f"Invalid dictionary response: {str(e)}")
        return [
            DataDictionaryColumn(
                column=col,
                description="No valid description available",
                data_type=str(df[col].dtype),
            )
            for col in columns
        ]


@log_api_call
@otel.meter_and_trace
async def get_dictionary(dataset: AnalystDataset) -> DataDictionary:
    """Process a single dataset with parallel column batch processing"""

    try:
        logger.info(f"Processing dataset {dataset.name} init")
        # Convert JSON to DataFrame
        df_full = dataset.to_df()
        df = df_full.sample(n=min(10000, len(df_full)), seed=42)

        # Add debug logging
        logger.info(f"Processing dataset {dataset.name} with shape {df.shape}")

        # Handle empty dataset
        if df.is_empty():
            logger.warning(f"Dataset {dataset.name} is empty")
            return DataDictionary(
                name=dataset.name,
                column_descriptions=[],
            )

        # Split columns into batches
        column_batches = [
            list(df.columns[i : i + DICTIONARY_BATCH_SIZE])
            for i in range(0, len(df.columns), DICTIONARY_BATCH_SIZE)
        ]
        logger.info(
            f"Created {len(column_batches)} batches for {len(df.columns)} columns"
        )

        # Create a semaphore to limit concurrent tasks to 2
        sem = asyncio.Semaphore(DICTIONARY_PARALLEL_BATCH_SIZE)

        async def throttled_get_dictionary_batch(
            batch: list[str],
        ) -> list[DataDictionaryColumn]:
            try:
                async with sem:
                    return await asyncio.wait_for(
                        _get_dictionary_batch(batch, df, DICTIONARY_BATCH_SIZE),
                        timeout=DICTIONARY_TIMEOUT,
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout processing batch: {batch}")
                return [
                    DataDictionaryColumn(
                        column=col,
                        description="No Description Available",
                        data_type=str(df[col].dtype),
                    )
                    for col in batch
                ]
            except Exception as e:
                logger.error(f"Error processing batch {batch}: {str(e)}")
                return [
                    DataDictionaryColumn(
                        column=col,
                        description="No Description Available",
                        data_type=str(df[col].dtype),
                    )
                    for col in batch
                ]

        tasks = [throttled_get_dictionary_batch(batch) for batch in column_batches]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out any exceptions and flatten results
        dictionary: list[DataDictionaryColumn] = []
        for result in results:
            if isinstance(result, BaseException):
                logger.error(f"Task failed with error: {str(result)}")
                continue
            dictionary.extend(result)

        logger.info(
            f"Created dictionary with {len(dictionary)} entries for dataset {dataset.name}"
        )

        return DataDictionary(
            name=dataset.name,
            column_descriptions=dictionary,
        )

    except Exception:
        return DataDictionary(
            name=dataset.name,
            column_descriptions=[
                DataDictionaryColumn(
                    column=c,
                    data_type=str(dataset.to_df()[c].dtype),
                    description="No Description Available",
                )
                for c in dataset.columns
            ],
        )


@otel.trace
def find_imports(module: ModuleType) -> list[str]:
    """
    Get top-level third-party imports from a Python module.

    Args:
        module: Python module object to analyze

    Returns:
        list of third-party package names

    Example:
        >>> import my_module
        >>> imports = find_third_party_imports(my_module)
        >>> print(imports)  # ['pandas', 'numpy', 'requests']
    """
    try:
        # Get the source code of the module
        source = inspect.getsource(module)
        tree = ast.parse(source)

        stdlib_modules = set(sys.stdlib_module_names)
        third_party = set()

        # Only look at top-level imports
        for node in tree.body:
            if isinstance(node, ast.Import):
                for name in node.names:
                    module_name = name.name.split(".")[0]
                    if module_name not in stdlib_modules:
                        third_party.add(module_name)

            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                module_name = node.module.split(".")[0]
                if module_name not in stdlib_modules:
                    third_party.add(module_name)

        return sorted(third_party)
    except Exception:
        return []


@otel.trace
def get_tools() -> list[Tool]:
    try:
        # find all functions defined in the tools module
        tool_functions = [func for func in dir(tools) if callable(getattr(tools, func))]

        # find the function signatures and doc strings
        tools_list = []
        for func_name in tool_functions:
            func = getattr(tools, func_name)
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func)
            tools_list.append(
                Tool(
                    name=func_name,
                    signature=str(signature),
                    docstring=docstring,
                    function=func,
                )
            )
        return tools_list
    except Exception:
        return []


@otel.trace
async def _generate_run_charts_python_code(
    request: RunChartsRequest,
    validation_error: InvalidGeneratedCode | None = None,
    token_tracker: TokenUsageTracker | None = None,
) -> str:
    df = request.dataset.to_df().to_pandas()
    question = request.question
    dataframe_metadata = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "statistics": df.describe(include="all").to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=prompts.SYSTEM_PROMPT_PLOTLY_CHART,
        ),
        ChatCompletionUserMessageParam(role="user", content=f"Question: {question}"),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Metadata:\n{dataframe_metadata}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data top 25 rows:\n{df.head(25).to_string()}"
        ),
    ]
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    # Get response based on model mode
    async with AsyncLLMClient(token_tracker=token_tracker) as client:
        with otel.time(
            f"{_generate_run_charts_python_code.__module__}.{_generate_run_charts_python_code.__qualname__}.llm_call"
        ):
            response: CodeGeneration = await client.chat.completions.create(
                response_model=CodeGeneration,
                model=ALTERNATIVE_LLM_BIG,
                temperature=0,
                messages=messages,
                timeout=900,
            )
    return response.code


@otel.trace
async def _generate_run_analysis_python_code(
    request: RunAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
    attempt: int = 0,
    token_tracker: TokenUsageTracker | None = None,
) -> str:
    """
    Generate Python analysis code based on JSON data and question.

    Parameters:
    - request: RunAnalysisRequest containing data and question
    - validation_errors: Past validation errors to include in prompt

    Returns:
    - Generated code
    """
    # Convert dictionary data structure to list of columns for all datasets
    logger.info("Starting code gen")

    all_columns = []
    all_descriptions = []
    all_data_types = []

    dictionaries = [
        await analyst_db.get_data_dictionary(name) for name in request.dataset_names
    ]
    for dictionary in dictionaries:
        if dictionary is None:
            continue
        for entry in dictionary.column_descriptions:
            all_columns.append(f"{dictionary.name}.{entry.column}")
            all_descriptions.append(entry.description)
            all_data_types.append(entry.data_type)

    # Create dictionary format for prompt
    dictionary_data = {
        "columns": all_columns,
        "descriptions": all_descriptions,
        "data_types": all_data_types,
    }

    # Get sample data and shape info for all datasets
    all_samples = []
    all_shapes = []

    logger.debug(f"datasets: {request.dataset_names}")
    for dataset_name in request.dataset_names:
        try:
            dataset = (await analyst_db.get_cleansed_dataset(dataset_name)).to_df()
        except Exception:
            dataset = (await analyst_db.get_dataset(dataset_name)).to_df()
        all_shapes.append(
            f"{dataset_name}: {dataset.shape[0]} rows x {dataset.shape[1]} columns"
        )
        # Limit sample to 10 rows
        sample_df = dataset.head(10)
        all_samples.append(f"{dataset_name}:\n{sample_df}")

    shape_info = "\n".join(all_shapes)
    sample_data = "\n\n".join(all_samples)
    logger.debug("Assembling messages")
    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_PYTHON_ANALYST
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Business Question: {request.question}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Shapes:\n{shape_info}"
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Sample Data:\n{sample_data}"
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Data Dictionary:\n{json.dumps(dictionary_data)}",
        ),
    ]

    tools_list = get_tools()
    if len(tools_list) > 0:
        messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content="If it helps the analysis, you can optionally use following functions:\n"
                + "\n".join([str(t) for t in tools_list]),
            )
        )

    logger.debug(f"total_characters: {len(''.join([str(msg) for msg in messages]))}")
    # Add error context if available
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )
        if attempt > 2:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Convert the dataframe to pandas!",
                )
            )
    logger.info("Running Code Gen")
    logger.debug(messages)
    async with AsyncLLMClient(token_tracker=token_tracker) as client:
        with otel.time(
            f"{_generate_run_analysis_python_code.__module__}.{_generate_run_analysis_python_code.__qualname__}.llm_call"
        ):
            completion: CodeGeneration = await client.chat.completions.create(
                response_model=CodeGeneration,
                model=ALTERNATIVE_LLM_BIG,
                temperature=0.1,
                messages=messages,
                max_retries=10,
                timeout=900,
            )
    logger.info("Code Gen complete")
    return completion.code


@otel.meter_and_trace
async def cleanse_dataframe(dataset: AnalystDataset) -> CleansedDataset:
    """Clean and standardize multiple pandas DataFrames in parallel.

    Args:
        datasets: List of AnalystDataset objects to clean
    Returns:
        List of CleansedDataset objects containing cleaned data and reports
    Raises:
        ValueError: If a dataset is empty
    """

    if dataset.to_df().is_empty():
        raise ValueError(f"Dataset {dataset.name} is empty")

    df = dataset.to_df()
    sample_df = df.sample(min(500, len(df)))

    results = []
    for col in df.columns:
        results.append(process_column(df, col, sample_df))

    # Create new DataFrame from processed columns
    new_columns = {}
    reports = []

    for new_name, series, report in results:
        new_columns[new_name] = series
        reports.append(report)

    cleaned_df = pl.DataFrame(new_columns)
    add_summary_statistics(cleaned_df, reports)

    return CleansedDataset(
        dataset=AnalystDataset(
            name=dataset.name,
            data=cleaned_df,
        ),
        cleaning_report=reports,
    )


@log_api_call
@otel.meter_and_trace
async def summarize_conversation(
    messages: list[ChatCompletionMessageParam],
    token_tracker: TokenUsageTracker | None = None,
) -> str:
    """Summarize a conversation history, when getting close to model's context window limit.

    Args:
        messages: list of message dictionaries with 'role' and 'content' fields
        token_tracker: Optional token usage tracker

    Returns:
        str: Summary of the conversation

    Raises:
        Exception: If summarization fails (network, LLM, parsing errors)
    """

    class ConversationSummary(BaseModel):
        summary: str

    try:
        messages_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in messages]
        )

        token_count = count_messages_tokens(messages, ALTERNATIVE_LLM_SMALL)
        logger.info(
            f"Summarizing conversation: {token_count} tokens, {len(messages)} messages"
        )

        prompt_messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                content=prompts.SYSTEM_PROMPT_SUMMARIZE_CONVERSATION,
                role="system",
            ),
            ChatCompletionUserMessageParam(
                content=f"Conversation History:\n{messages_str}",
                role="user",
            ),
        ]

        async with AsyncLLMClient(token_tracker=token_tracker) as client:
            with otel.time(
                f"{summarize_conversation.__module__}.{summarize_conversation.__qualname__}.llm_call"
            ):
                completion: ConversationSummary = await client.chat.completions.create(
                    response_model=ConversationSummary,
                    model=ALTERNATIVE_LLM_SMALL,
                    messages=prompt_messages,
                    timeout=900,
                )

        logger.info(f"Summary created: {len(completion.summary)} characters")
        return completion.summary

    except Exception as e:
        logger.error(f"Failed to summarize conversation: {e}")
        raise


@log_api_call
@otel.meter_and_trace
async def rephrase_message(
    messages: ChatRequest, token_tracker: TokenUsageTracker | None = None
) -> str:
    """Process chat messages history and return a new question

    Args:
        messages: list of message dictionaries with 'role' and 'content' fields
        token_tracker: Optional token usage tracker

    Returns:
        Dict[str, str]: Dictionary containing response content
    """
    # Debug logging
    token_count = count_messages_tokens(messages.messages, ALTERNATIVE_LLM_BIG)

    logger.info(
        f"DEBUG rephrase_message: {token_count} tokens, {len(messages.messages)} messages"
    )

    # Build prompt: system message + actual conversation history
    prompt_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content=prompts.SYSTEM_PROMPT_REPHRASE_MESSAGE,
            role="system",
        ),
    ]

    # Add the actual conversation messages (already includes summary if present)
    prompt_messages.extend(messages.messages)

    async with AsyncLLMClient(token_tracker=token_tracker) as client:
        with otel.time(
            f"{rephrase_message.__module__}.{rephrase_message.__qualname__}.llm_call"
        ):
            completion: EnhancedQuestionGeneration = (
                await client.chat.completions.create(
                    response_model=EnhancedQuestionGeneration,
                    model=ALTERNATIVE_LLM_BIG,
                    messages=prompt_messages,
                    timeout=900,
                )
            )

    return completion.enhanced_user_message


@reflect_code_generation_errors(max_attempts=7)
@otel.trace
async def _run_charts(
    request: RunChartsRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
    token_tracker: TokenUsageTracker | None = None,
) -> RunChartsResult:
    """Generate and validate chart code with retry logic"""
    # Create messages for OpenAI
    start_time = datetime.now()

    if not request.dataset:
        raise ValueError(VALUE_ERROR_MESSAGE)

    df = request.dataset.to_df().to_pandas()
    if exception_history is None:
        exception_history = []

    code = await _generate_run_charts_python_code(
        request, next(iter(exception_history[::-1]), None), token_tracker
    )
    try:
        result = execute_python(
            modules={
                "pd": pd,
                "np": np,
                "go": go,
                "pl": pl,
                "scipy": scipy,
            },
            functions={
                "make_subplots": make_subplots,
            },
            expected_function="create_charts",
            code=code,
            input_data=df,
            output_type=ChartGenerationExecutionResult,
            allowed_modules={
                "pandas",
                "numpy",
                "plotly",
                "scipy",
                "datetime",
                "polars",
            },
        )
    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=code, exception=e)

    duration = datetime.now() - start_time

    return RunChartsResult(
        status="success",
        code=code,
        fig1_json=result.fig1.to_json(),
        fig2_json=result.fig2.to_json(),
        metadata=RunAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history) + 1,
        ),
    )


@log_api_call
@otel.meter_and_trace
async def run_charts(
    request: RunChartsRequest, token_tracker: TokenUsageTracker | None = None
) -> RunChartsResult:
    """Execute analysis workflow on datasets."""
    try:
        chart_result = await _run_charts(request, token_tracker=token_tracker)
        return chart_result
    except ValidationError as e:
        logger.error(f"Failed to parse LLM response for chart generation: {e}")
        user_friendly_error = ValueError(
            "Unable to generate chart for your data. "
            "This could be due to data quality issues, incompatible data types, or complex visualization requirements. "
            "Try rephrasing your question."
        )
        return RunChartsResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(user_friendly_error),
            ),
        )
    except ValueError as e:
        logger.error(f"ValueError during chart generation: {e}")
        return RunChartsResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(e),
            ),
        )
    except MaxReflectionAttempts as e:
        return RunChartsResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )


@log_api_call
@otel.meter_and_trace
async def get_business_analysis(
    request: GetBusinessAnalysisRequest,
    token_tracker: TokenUsageTracker | None = None,
) -> GetBusinessAnalysisResult:
    """
    Generate business analysis based on data and question.

    Parameters:
    - request: BusinessAnalysisRequest containing data and question

    Returns:
    - Dictionary containing analysis components
    """
    try:
        # Convert JSON data to DataFrame for analysis
        start = datetime.now()

        df = request.dataset.to_df().to_pandas()

        initial_rows = 750

        df_csv, _ = estimate_csv_rows_for_token_limit(
            df, MAX_CSV_TOKENS, initial_rows, ALTERNATIVE_LLM_BIG
        )

        # Create messages for OpenAI
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system", content=prompts.SYSTEM_PROMPT_BUSINESS_ANALYSIS
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Business Question: {request.question}",
            ),
            ChatCompletionUserMessageParam(
                role="user", content=f"Analyzed Data:\n{df_csv}"
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=f"Data Dictionary:\n{request.dictionary.model_dump_json()}",
            ),
        ]
        async with AsyncLLMClient(token_tracker=token_tracker) as client:
            with otel.time(
                f"{get_business_analysis.__module__}.{get_business_analysis.__qualname__}.llm_call"
            ):
                completion: BusinessAnalysisGeneration = (
                    await client.chat.completions.create(
                        response_model=BusinessAnalysisGeneration,
                        model=ALTERNATIVE_LLM_BIG,
                        temperature=0.1,
                        messages=messages,
                        timeout=900,
                    )
                )
        duration = (datetime.now() - start).total_seconds()
        # Ensure all response fields are present
        metadata = GetBusinessAnalysisMetadata(
            duration=duration,
            question=request.question,
            rows_analyzed=len(df),
            columns_analyzed=len(df.columns),
        )
        return GetBusinessAnalysisResult(
            status="success",
            **completion.model_dump(),
            metadata=metadata,
        )

    except ValidationError as e:
        logger.error(f"Failed to parse LLM response for business analysis: {e}")
        user_friendly_error = ValueError(
            "Unable to generate business insights. "
            "This could be due to data quality issues or complex data structure. "
            "Try rephrasing your question."
        )
        return GetBusinessAnalysisResult(
            status="error",
            metadata=GetBusinessAnalysisMetadata(
                exception=AnalysisError.from_value_error(user_friendly_error)
            ),
            additional_insights="",
            follow_up_questions=[],
            bottom_line="",
        )
    except ValueError as e:
        logger.error(f"ValueError during business analysis generation: {e}")
        return GetBusinessAnalysisResult(
            status="error",
            metadata=GetBusinessAnalysisMetadata(
                exception=AnalysisError.from_value_error(e)
            ),
            additional_insights="",
            follow_up_questions=[],
            bottom_line="",
        )
    except Exception as e:
        msg = type(e).__name__ + f": {str(e)}"
        logger.error(f"Error in get_business_analysis: {msg}")
        return GetBusinessAnalysisResult(
            status="error",
            metadata=GetBusinessAnalysisMetadata(exception_str=msg),
            additional_insights="",
            follow_up_questions=[],
            bottom_line="",
        )


@reflect_code_generation_errors(max_attempts=7)
@otel.trace
async def _run_analysis(
    request: RunAnalysisRequest,
    analysis_context: RunCompleteAnalysisRequestContext,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunAnalysisResult:
    start_time = datetime.now()

    analyst_db = analysis_context.analyst_db
    token_tracker = analysis_context.token_tracker

    if not request.dataset_names:
        raise ValueError(VALUE_ERROR_MESSAGE)

    if exception_history is None:
        exception_history = []
    logger.info(f"Running analysis (attempt {len(exception_history)})")

    if analysis_context.assistant_message_id and analysis_context.assistant_message:
        analysis_context.assistant_message.step_value = "GENERATING_QUERY"
        analysis_context.assistant_message.step_reattempt = len(exception_history)
        analysis_context.stage_message_update()

    code = await _generate_run_analysis_python_code(
        request,
        analyst_db,
        next(iter(exception_history[::-1]), None),
        attempt=len(exception_history),
        token_tracker=token_tracker,
    )
    logger.info("Code generated, preparing execution")

    if analysis_context.assistant_message_id and analysis_context.assistant_message:
        analysis_context.assistant_message.step_value = "RUNNING_QUERY"
        analysis_context.assistant_message.step_reattempt = len(exception_history)
        analysis_context.stage_message_update()

    dataframes: dict[str, pl.DataFrame] = {}

    for dataset_name in request.dataset_names:
        try:
            dataset = (
                await analyst_db.get_cleansed_dataset(dataset_name, max_rows=None)
            ).to_df()
        except Exception:
            dataset = (
                await analyst_db.get_dataset(dataset_name, max_rows=None)
            ).to_df()
        dataframes[dataset_name] = dataset
    functions = {}
    tool_functions = get_tools()
    for tool in tool_functions:
        functions[tool.name] = tool.function
    try:
        logger.info("Executing")
        result = execute_python(
            modules={
                "pd": pd,
                "np": np,
                "sm": sm,
                "pl": pl,
                "scipy": scipy,
                "sklearn": sklearn,
            },
            functions=functions,
            expected_function="analyze_data",
            code=code,
            input_data=dataframes,
            output_type=AnalystDataset,
            allowed_modules={
                "pandas",
                "numpy",
                "scipy",
                "sklearn",
                "statsmodels",
                "datetime",
                "polars",
                *find_imports(tools),
            },
        )
    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=code, exception=e)
    logger.info("Execution done")
    duration = datetime.now() - start_time
    return RunAnalysisResult(
        status="success",
        code=code,
        dataset=result,
        metadata=RunAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history) + 1,
            datasets_analyzed=len(dataframes),
            total_rows_analyzed=sum(
                len(df) for df in dataframes.values() if not df.is_empty()
            ),
            total_columns_analyzed=sum(
                len(df.columns) for df in dataframes.values() if not df.is_empty()
            ),
        ),
    )


@log_api_call
async def run_analysis(
    request: RunAnalysisRequest,
    analysis_context: RunCompleteAnalysisRequestContext,
) -> RunAnalysisResult:
    """Execute analysis workflow on datasets."""
    logger.debug("Entering run_analysis")
    log_memory()
    try:
        return await _run_analysis(request, analysis_context=analysis_context)
    except MaxReflectionAttempts as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )
    except ValidationError as e:
        logger.error(f"Failed to parse LLM response for analysis: {e}")
        user_friendly_error = ValueError(
            "Unable to complete the analysis. "
            "This could be due to data quality issues, complex dataset structure, or the question being too complex. "
            "Try simplifying your question or verifying your data quality."
        )
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(user_friendly_error),
            ),
        )
    except ValueError as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(e),
            ),
        )


async def _generate_database_analysis_code(
    database: DatabaseOperator[Any],
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
    token_tracker: TokenUsageTracker | None = None,
) -> str:
    """
    Generate Snowflake SQL analysis code based on data samples and question.

    Parameters:
    - request: DatabaseAnalysisRequest containing data samples and question

    Returns:
    - Dictionary containing generated code and description
    """

    # Convert dictionary data structure to list of columns for all tables
    dictionaries = [
        (
            await analyst_db.get_data_dictionary(name),
            await analyst_db.get_dataset_metadata(name),
        )
        for name in request.dataset_names
    ]

    for dictionary, metadata in dictionaries:
        if dictionary:
            if metadata and metadata.original_column_types:
                for column in dictionary.column_descriptions:
                    if original_type := metadata.original_column_types.get(
                        column.column
                    ):
                        column.data_type = original_type
            dictionary.name = database.query_friendly_name(dictionary.name)
    all_tables_info = [
        d.model_dump(mode="json") for d, m in dictionaries if d is not None
    ]

    # Get sample data for all tables
    all_samples = []

    for table in request.dataset_names:
        df = (await analyst_db.get_dataset(table)).to_df().to_pandas()

        friendly_name = database.query_friendly_name(table)

        sample_str = f"Table: {friendly_name}\n{df.head(10).to_string()}"
        all_samples.append(sample_str)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        database.get_system_prompt(),
        ChatCompletionUserMessageParam(
            content=f"Business Question: {request.question}",
            role="user",
        ),
        ChatCompletionUserMessageParam(
            content=f"Sample Data:\n{chr(10).join(all_samples)}", role="user"
        ),
        ChatCompletionUserMessageParam(
            content=f"Data Dictionary:\n{json.dumps(all_tables_info)}", role="user"
        ),
    ]
    if validation_error:
        msg = type(validation_error).__name__ + f": {str(validation_error)}"
        messages.extend(
            [
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Previous attempt failed with error: {msg}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"Failed code: {validation_error.code}",
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content="Please generate new code that avoids this error.",
                ),
            ]
        )

    # Get response from OpenAI
    async with AsyncLLMClient(token_tracker=token_tracker) as client:
        with otel.time(
            f"{_generate_database_analysis_code.__module__}.{_generate_database_analysis_code.__qualname__}.llm_call"
        ):
            try:
                completion = await client.chat.completions.create(
                    response_model=DatabaseAnalysisCodeGeneration,
                    model=ALTERNATIVE_LLM_BIG,
                    temperature=0.1,
                    messages=messages,
                    timeout=900,
                )
            except ValidationError as e:
                logger.error(f"LLM returned invalid response: {e}")
                raise ValueError(
                    "Unable to analyze your data. "
                    "This could be due to data quality issues, complex dataset structure, or the question being too complex. "
                    "Try simplifying your question or checking your data."
                ) from e

    return str(completion.code)


@reflect_code_generation_errors(max_attempts=7)
@otel.trace
async def _run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    analysis_context: RunCompleteAnalysisRequestContext,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunDatabaseAnalysisResult:
    start_time = datetime.now()
    if not request.dataset_names:
        raise ValueError(VALUE_ERROR_MESSAGE)

    if exception_history is None:
        exception_history = []

    assert analysis_context.database, "Database has been assigned."
    database = analysis_context.database

    if analysis_context.assistant_message and analysis_context.assistant_message_id:
        analysis_context.assistant_message.step_value = "GENERATING_QUERY"
        analysis_context.assistant_message.step_reattempt = (
            len(exception_history) if exception_history else 0
        )
        analysis_context.stage_message_update()

    sql_code = await _generate_database_analysis_code(
        database,
        request,
        analysis_context.analyst_db,
        next(iter(exception_history[::-1]), None),
        analysis_context.token_tracker,
    )
    try:
        if analysis_context.assistant_message and analysis_context.assistant_message_id:
            analysis_context.assistant_message.step_value = "RUNNING_QUERY"
            analysis_context.assistant_message.step_reattempt = (
                len(exception_history) if exception_history else 0
            )
            analysis_context.stage_message_update()

        results = await database.execute_query(query=sql_code)
        results = cast(list[dict[str, Any]], results)
        duration = datetime.now() - start_time

    except InvalidGeneratedCode:
        raise
    except Exception as e:
        raise InvalidGeneratedCode(code=sql_code, exception=e)
    return RunDatabaseAnalysisResult(
        status="success",
        code=sql_code,
        dataset=AnalystDataset(
            data=results,
        ),
        metadata=RunDatabaseAnalysisResultMetadata(
            duration=duration.total_seconds(),
            attempts=len(exception_history),
            datasets_analyzed=len(request.dataset_names),
            # total_columns_analyzed=sum(len(ds.columns) for ds in request.datasets),
        ),
    )


@log_api_call
@otel.meter_and_trace
async def run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    analysis_context: RunCompleteAnalysisRequestContext,
) -> RunDatabaseAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_database_analysis(
            request, analysis_context=analysis_context, exception_history=[]
        )
    except MaxReflectionAttempts as e:
        return RunDatabaseAnalysisResult(
            status="error",
            metadata=RunDatabaseAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )
    except ValueError as e:
        return RunDatabaseAnalysisResult(
            status="error",
            metadata=RunDatabaseAnalysisResultMetadata(
                duration=0,
                attempts=1,
                exception=AnalysisError.from_value_error(e),
            ),
        )


# Type definitions
@dataclass
class AnalysisGenerationError:
    message: str
    original_error: BaseException | None = None


async def execute_business_analysis_and_charts(
    analysis_result: RunAnalysisResult | RunDatabaseAnalysisResult,
    enhanced_message: str,
    token_tracker: TokenUsageTracker | None = None,
    enable_chart_generation: bool = True,
    enable_business_insights: bool = True,
) -> tuple[
    RunChartsResult | BaseException | None,
    GetBusinessAnalysisResult | BaseException | None,
]:
    analysis_result.dataset = cast(AnalystDataset, analysis_result.dataset)
    # Prepare both requests
    chart_request = RunChartsRequest(
        dataset=analysis_result.dataset,
        question=enhanced_message,
    )

    business_request = GetBusinessAnalysisRequest(
        dataset=analysis_result.dataset,
        dictionary=DataDictionary.from_analyst_df(analysis_result.dataset.to_df()),
        question=enhanced_message,
    )

    if enable_chart_generation and enable_business_insights:
        # Run both analyses concurrently
        result = await asyncio.gather(
            run_charts(chart_request, token_tracker),
            get_business_analysis(business_request, token_tracker),
            return_exceptions=True,
        )

        return (result[0], result[1])
    elif enable_chart_generation:
        charts_result = await run_charts(chart_request, token_tracker)
        return charts_result, None
    else:
        business_result = await get_business_analysis(business_request, token_tracker)
        return None, business_result


@dataclass
class RunCompleteAnalysisRequestContext:
    chat_request: ChatRequest
    request: Request | None
    data_source: DataSourceType
    dataset_metadata: list[DatasetMetadata]
    analyst_db: AnalystDB
    chat_id: str
    user_message_id: str
    enable_chart_generation: bool
    enable_business_insights: bool
    token_tracker: TokenUsageTracker

    user_message: AnalystChatMessage | None = None
    assistant_message_id: str | None = None
    assistant_message: AnalystChatMessage | None = None
    database: DatabaseOperator[Any] | None = None
    recipe: BaseRecipe | None = None

    message_update_task: asyncio.Task[Any] | None = None

    def stage_message_update(
        self, target: Literal["assistant", "user"] = "assistant"
    ) -> None:
        # In order to separate out the work of persisting messages from the work of generating answers
        # (this matters as our "DB" is very slow) we run updates in a background task. In order to not
        # have updates conflict with one another, we run them separately.
        target_message_id = (
            self.assistant_message_id if target == "assistant" else self.user_message_id
        )
        # Without these shallow copies the message might change before we save it. Realistically,
        # this only happens in unit tests and doesn't have a negative impact. But it's cheap enough
        # to do and helps with unit tests.
        target_message = copy.copy(
            self.assistant_message if target == "assistant" else self.user_message
        )
        if target_message:
            target_message.step = copy.copy(target_message.step)

        if not target_message or not target_message_id:
            return

        previous_task = self.message_update_task

        async def update_task() -> None:
            if previous_task:
                await previous_task

            await self.analyst_db.update_chat_message(
                message_id=target_message_id, message=target_message
            )

        self.message_update_task = asyncio.create_task(update_task())

    async def await_message_update(self) -> None:
        if self.message_update_task:
            await self.message_update_task
            self.message_update_task = None


@otel.meter_and_trace
async def run_complete_analysis(
    chat_request: ChatRequest,
    data_source: DataSourceType,
    dataset_metadata: list[DatasetMetadata],
    analyst_db: AnalystDB,
    chat_id: str,
    message_id: str,
    request: Request | None,
    enable_chart_generation: bool = True,
    enable_business_insights: bool = True,
) -> AsyncGenerator[Component | AnalysisGenerationError, None]:
    analysis_context = RunCompleteAnalysisRequestContext(
        chat_request=chat_request,
        request=request,
        data_source=data_source,
        analyst_db=analyst_db,
        dataset_metadata=dataset_metadata,
        chat_id=chat_id,
        user_message_id=message_id,
        enable_business_insights=enable_business_insights,
        enable_chart_generation=enable_chart_generation,
        token_tracker=TokenUsageTracker(TiktokenCountingStrategy()),
    )

    analysis_context.user_message = await analyst_db.get_chat_message(
        message_id=analysis_context.user_message_id
    )
    if (
        analysis_context.user_message is None
        or analysis_context.user_message.role != "user"
    ):
        logger.error(
            "Attempted to respond to a message %s not found (perhaps deleted),"
        )
        yield AnalysisGenerationError("Message not found")

        return

    uses_spark = data_source == InternalDataSourceType.REMOTE_REGISTRY or bool(
        dataset_metadata and all(m.external_id is not None for m in dataset_metadata)
    )

    if uses_spark:
        uses_spark_consistently = (
            data_source == InternalDataSourceType.REMOTE_REGISTRY
        ) == all(m.external_id is not None for m in dataset_metadata)

        if not uses_spark_consistently:
            logger.error(
                "Using SPARK but non-spark datasets somehow selected. Removing those."
            )
            analysis_context.dataset_metadata = dataset_metadata = [
                m for m in dataset_metadata if m.external_id is not None
            ]

        if not DatasetSparkRecipe.should_use_spark_recipe():
            logger.fatal(
                "Should be unreachable. Ended up with remote datasets while remote datasets is disallowed."
            )

            analysis_context.user_message.error = "The application encountered an Internal Error. Contact app builder to review logs."
            analysis_context.user_message.in_progress = False
            analysis_context.stage_message_update(target="user")

            await analysis_context.await_message_update()
            yield AnalysisGenerationError(analysis_context.user_message.error)

            return

    if dataset_metadata:
        if any(m.external_id is not None for m in dataset_metadata) and any(
            m.external_id is None for m in dataset_metadata
        ):
            logger.error(
                "Should be unreachable. Ended up with a mix of remote and local datasets. Making consistent with specified datasource"
            )
            if data_source == InternalDataSourceType.REMOTE_REGISTRY:
                analysis_context.dataset_metadata = dataset_metadata = [
                    m for m in dataset_metadata if m.external_id is not None
                ]
            else:
                analysis_context.dataset_metadata = dataset_metadata = [
                    m for m in dataset_metadata if m.external_id is None
                ]

    datasets_names = [ds.name for ds in dataset_metadata]

    try:
        # Get enhanced message
        logger.info("Getting rephrased question...")
        enhanced_message = await rephrase_message(
            chat_request, analysis_context.token_tracker
        )
        logger.info("Getting rephrased question done")

        yield enhanced_message

    except Exception as e:
        logger.error(f"Error rephrasing message: {e}", exc_info=True)
        analysis_context.user_message.error = (
            f"Failed to process your question: {str(e)}"
        )
        analysis_context.user_message.in_progress = False
        analysis_context.stage_message_update(target="user")

        await analysis_context.await_message_update()

        yield AnalysisGenerationError(analysis_context.user_message.error)

        return

    should_test_connection = isinstance(
        analysis_context.data_source, ExternalDataStoreNameDataSourceType
    ) or analysis_context.data_source in [
        InternalDataSourceType.REMOTE_REGISTRY,
        InternalDataSourceType.DATABASE,
    ]

    analysis_context.assistant_message = AnalystChatMessage(
        role="assistant",
        content="",
        components=[],
        in_progress=True,
    )
    analysis_context.assistant_message.step_value = "ANALYZING_QUESTION"

    analysis_context.assistant_message_id = await analyst_db.add_chat_message(
        chat_id=chat_id, message=analysis_context.assistant_message
    )

    analysis_context.assistant_message.content = enhanced_message
    analysis_context.assistant_message.components.append(
        EnhancedQuestionGeneration(enhanced_user_message=enhanced_message)
    )
    analysis_context.assistant_message.step_value = (
        "TESTING_CONNECTION" if should_test_connection else "GENERATING_QUERY"
    )

    analysis_context.stage_message_update()

    analysis_context.user_message.in_progress = False

    analysis_context.stage_message_update(target="user")

    logger.debug("Initializing database.", extra={"data_source": data_source})

    if data_source == InternalDataSourceType.DATABASE:
        analysis_context.database = get_external_database()
    elif data_source == InternalDataSourceType.REMOTE_REGISTRY:
        if analysis_context.request:
            with use_user_token(analysis_context.request):
                logger.debug("Initializing SPARK recipe with a user token.")
                analysis_context.recipe = await load_or_create_spark_recipe(
                    analyst_db=analyst_db
                )
                refresh = await analysis_context.recipe.refresh()
        else:
            logger.debug("Initializing SPARK recipe with app token.")
            analysis_context.recipe = await load_or_create_spark_recipe(
                analyst_db=analyst_db
            )
            refresh = await analysis_context.recipe.refresh()

        if refresh:
            analysis_context.assistant_message.in_progress = False
            analysis_context.assistant_message.error = "A remote dataset was deleted and can no longer be used for analysis. Please refresh."
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(analysis_context.assistant_message.error)

            await analysis_context.await_message_update()

            return

        analysis_context.database = analysis_context.recipe.as_database_operator()
    elif isinstance(data_source, ExternalDataStoreNameDataSourceType):
        data_store_id = await DataSourceRecipe.get_id_for_data_store_canonical_name(
            data_source.friendly_name
        )
        if not data_store_id:
            analysis_context.assistant_message.in_progress = False
            analysis_context.assistant_message.error = "A remote dataset was deleted and can no longer be used for analysis. Please refresh."
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(analysis_context.assistant_message.error)

            await analysis_context.await_message_update()

            return

        if request:
            with use_user_token(request):
                logger.debug("Initializing Data Source recipe with a user token.")
                analysis_context.recipe = await DataSourceRecipe.load_or_create(
                    analyst_db, data_store_id
                )
                result = await analysis_context.recipe.refresh()
        else:
            logger.debug("Initializing Data Source recipe with app token.")
            analysis_context.recipe = await DataSourceRecipe.load_or_create(
                analyst_db, data_store_id
            )
            result = await analysis_context.recipe.refresh()
        if result:
            analysis_context.assistant_message.in_progress = False
            analysis_context.assistant_message.error = "A remote dataset was deleted and can no longer be used for analysis. Please refresh."
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(analysis_context.assistant_message.error)

            await analysis_context.await_message_update()
            return

        analysis_context.database = analysis_context.recipe.as_database_operator()

    if analysis_context.database:
        logger.debug("Warming up database.")
        try:
            await analysis_context.database.warmup()
        except BaseException:
            logger.warning("Failed to warmup database", exc_info=True)

        analysis_context.assistant_message.step_value = "GENERATING_QUERY"

        analysis_context.stage_message_update()

    # Run main analysis
    logger.info("Start main analysis")
    try:
        logger.info("Getting analysis result...")
        log_memory()

        analysis_result: RunAnalysisResult | RunDatabaseAnalysisResult

        if analysis_context.database:
            logger.info("Running database analysis")
            analysis_result = await run_database_analysis(
                RunDatabaseAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analysis_context=analysis_context,
            )

        else:
            logging.info("Running local analysis")
            analysis_result = await run_analysis(
                RunAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analysis_context=analysis_context,
            )

        log_memory()
        logger.info("Getting analysis result done")

        if isinstance(analysis_result, BaseException):
            error_message = f"Error running initial analysis. Try rephrasing: {str(analysis_result)}"
            analysis_context.assistant_message.in_progress = False
            analysis_context.assistant_message.error = error_message
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(error_message)

            await analysis_context.await_message_update()
            return

        yield analysis_result

        analysis_context.assistant_message.components.append(analysis_result)
        analysis_context.assistant_message.step_value = "ANALYZING_RESULTS"
        analysis_context.assistant_message = await extract_and_store_datasets(
            analyst_db, analysis_context.assistant_message
        )

        analysis_context.stage_message_update()

    except Exception as e:
        error_message = f"Error running initial analysis. Try rephrasing: {str(e)}"
        analysis_context.assistant_message.in_progress = False
        analysis_context.assistant_message.error = error_message
        analysis_context.stage_message_update()

        yield AnalysisGenerationError(error_message)

        await analysis_context.await_message_update()

        return

    # Only proceed with additional analysis if we have valid initial results
    if not (
        analysis_result
        and analysis_result.dataset
        and (enable_chart_generation or enable_business_insights)
    ):
        analysis_context.assistant_message.in_progress = False
        analysis_context.stage_message_update()
        await analysis_context.await_message_update()
        return

    # Run concurrent analyses
    try:
        charts_result, business_result = await execute_business_analysis_and_charts(
            analysis_result,
            enhanced_message,
            analysis_context.token_tracker,
            enable_business_insights=enable_business_insights,
            enable_chart_generation=enable_chart_generation,
        )

        # Handle chart results
        if isinstance(charts_result, BaseException):
            error_message = "Error generating charts"
            analysis_context.assistant_message.error = error_message
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(error_message)

        elif charts_result is not None:
            analysis_context.assistant_message.components.append(charts_result)
            analysis_context.stage_message_update()

            yield charts_result

        # Handle business analysis results
        if isinstance(business_result, BaseException):
            error_message = "Error generating business insights"
            analysis_context.assistant_message.error = error_message
            analysis_context.stage_message_update()

            yield AnalysisGenerationError(error_message)

        elif business_result is not None:
            analysis_context.assistant_message.components.append(business_result)
            analysis_context.stage_message_update()

            yield business_result

        analysis_context.assistant_message.in_progress = False
        analysis_context.stage_message_update()

    except Exception as e:
        error_message = f"Error setting up additional analysis: {str(e)}"
        analysis_context.assistant_message.in_progress = False
        analysis_context.assistant_message.error = error_message
        analysis_context.stage_message_update()

        yield AnalysisGenerationError(error_message)

    finally:
        # Generate token usage component
        if analysis_context.token_tracker.call_count > 0:
            final_usage_component = UsageInfoComponent(
                usage=TokenUsageInfo(**analysis_context.token_tracker.to_dict())
            )

            analysis_context.assistant_message.components.append(final_usage_component)

            analysis_context.stage_message_update()

            yield final_usage_component

        await analysis_context.await_message_update()


async def process_data_and_update_state(
    new_dataset_names: list[str],
    analyst_db: AnalystDB,
    data_source: str | DataSourceType,
) -> AsyncGenerator[str, None]:
    """Process datasets and yield progress updates asynchronously."""
    # Start processing and yield initial message
    logger.info(
        "Starting data processing",
        extra={"new_dataset_names": new_dataset_names, "data_source": data_source},
    )
    log_memory()
    yield "Starting data processing"

    # Handle data cleansing based on the source
    # Convert string data_source to DataSourceType if needed
    data_source_type = (
        data_source
        if isinstance(data_source, InternalDataSourceType)
        or isinstance(data_source, ExternalDataStoreNameDataSourceType)
        else get_data_source_type(data_source)
    )
    if data_source_type != InternalDataSourceType.DATABASE and not isinstance(
        data_source_type, ExternalDataStoreNameDataSourceType
    ):
        try:
            logger.info("Cleansing datasets")
            yield "Cleansing datasets"
            for analysis_dataset_name in new_dataset_names:
                metadata = await analyst_db.get_dataset_metadata(analysis_dataset_name)
                if metadata.data_source == InternalDataSourceType.REMOTE_REGISTRY:
                    # Skip remote datasets.
                    continue

                analysis_dataset = await analyst_db.get_dataset(
                    analysis_dataset_name, max_rows=None
                )

                cleansed_dataset = await cleanse_dataframe(analysis_dataset)
                await analyst_db.register_dataset(
                    cleansed_dataset, data_source=InternalDataSourceType.GENERATED
                )
                yield f"Cleansed dataset: {analysis_dataset_name}"
                del cleansed_dataset
                del analysis_dataset
                log_memory()

            logger.info("Cleansing datasets complete")
            yield "Cleansing datasets complete"
            log_memory()
        except Exception:
            logger.error("Data processing failed", exc_info=True)
            yield "Data processing failed"
            raise
    else:
        pass

    # Generate data dictionaries
    logger.info("Data processing successful, generating dictionaries")
    yield "Data processing successful, generating dictionaries"
    log_memory()
    try:
        for analysis_dataset_name in new_dataset_names:
            try:
                existing_dictionary = await analyst_db.get_data_dictionary(
                    analysis_dataset_name
                )
                logger.info(
                    f"Found existing dictionary for dataset: {analysis_dataset_name}"
                )
                if existing_dictionary is not None:
                    continue

            except Exception:
                pass
            logger.info(f"Creating dictionary for dataset: {analysis_dataset_name}")
            analysis_dataset = await analyst_db.get_dataset(analysis_dataset_name)
            new_dictionary = await get_dictionary(analysis_dataset)
            logger.info(new_dictionary.to_application_df())
            del analysis_dataset
            await analyst_db.register_data_dictionary(new_dictionary, clobber=True)
            logger.info(f"Registered dictionary for dataset: {analysis_dataset_name}")
            yield f"Registered data dictionary: {analysis_dataset_name}"
            log_memory()
            continue
    except Exception:
        logger.error("Failed to generate data dictionaries", exc_info=True)
        yield "Failed to generate data dictionaries"
        raise
    log_memory()
    # Final completion message
    yield "Processing complete"
