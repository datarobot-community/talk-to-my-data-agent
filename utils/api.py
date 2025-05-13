# Copyright 2024 DataRobot, Inc.
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
import inspect
import json
import logging
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from types import ModuleType
from typing import (
    Any,
    AsyncGenerator,
    TypeVar,
    cast,
)
from urllib.parse import urljoin

import datarobot as dr
import instructor
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import psutil
import scipy
import sklearn
import statsmodels as sm
from joblib import Memory
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from plotly.subplots import make_subplots
from pydantic import ValidationError

sys.path.append("..")
from utils import prompts, tools
from utils.analyst_db import AnalystDB, DataSourceType
from utils.code_execution import (
    InvalidGeneratedCode,
    MaxReflectionAttempts,
    execute_python,
    reflect_code_generation_errors,
)
from utils.data_cleansing_helpers import (
    add_summary_statistics,
    process_column,
)
from utils.database_helpers import Database
from utils.logging_helper import get_logger, log_api_call
from utils.resources import LLMDeployment
from utils.schema import (
    AiCatalogDataset,
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
    DictionaryGeneration,
    EnhancedQuestionGeneration,
    GetBusinessAnalysisMetadata,
    GetBusinessAnalysisRequest,
    GetBusinessAnalysisResult,
    QuestionListGeneration,
    RunAnalysisRequest,
    RunAnalysisResult,
    RunAnalysisResultMetadata,
    RunChartsRequest,
    RunChartsResult,
    RunDatabaseAnalysisRequest,
    RunDatabaseAnalysisResult,
    RunDatabaseAnalysisResultMetadata,
    Tool,
    ValidatedQuestion,
)

logger = get_logger()
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("openai.http_client").setLevel(logging.WARNING)


def log_memory() -> None:
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Memory usage: {memory:.2f} MB")


try:
    dr_client = dr.Client()
    chat_agent_deployment_id = LLMDeployment().id
    deployment_chat_base_url = urljoin(
        dr_client.endpoint, f"/deployments/{chat_agent_deployment_id}/"
    )

    class AsyncLLMClient:
        async def __aenter__(self) -> instructor.AsyncInstructor:
            dr_client = dr.Client()
            chat_agent_deployment_id = LLMDeployment().id
            deployment_chat_base_url = (
                dr_client.endpoint + f"/deployments/{chat_agent_deployment_id}/"
            )
            self.openai_client = AsyncOpenAI(
                api_key=dr_client.token,
                base_url=deployment_chat_base_url,
                timeout=90,
                max_retries=2,
            )
            self.client = instructor.from_openai(
                self.openai_client, mode=instructor.Mode.MD_JSON
            )
            return self.client

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type:ignore
            await self.openai_client.close()  # Properly close the client

except ValidationError as e:
    raise ValueError(
        "Unable to load Deployment ID."
        "If running locally, verify you have selected the correct "
        "stack and that it is active using `pulumi stack output`. "
        "If running in DataRobot, verify your runtime parameters have been set correctly."
    ) from e

ALTERNATIVE_LLM_BIG = "datarobot-deployed-llm"
ALTERNATIVE_LLM_SMALL = "datarobot-deployed-llm"
DICTIONARY_BATCH_SIZE = 10
MAX_AI_CATALOG_DATASET_SIZE = 400e6  # aligns to 400MB set in streamlit config.toml
DISK_CACHE_LIMIT_BYTES = 512e6

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
                    f"Using previously cached result for function `{f.__name__}`"  # type: ignore
                )
            return result

        return cast(T, wrapper)


# This can be large as we are not storing the actual datasets in memory, just metadata
def list_catalog_datasets(limit: int = 100) -> list[AiCatalogDataset]:
    """
    Fetch datasets from AI Catalog with specified limit

    Args:
        limit: int
        Datasets to retrieve. Max value: 100
    """

    url = f"datasets?limit={limit}"

    # Get all datasets and manually limit the results
    datasets = dr.client.get_client().get(url).json()["data"]

    return [
        AiCatalogDataset(
            id=ds["datasetId"],
            name=ds["name"],
            created=(
                ds["creationDate"][:10] if "creationDate" in ds else "N/A"  # %Y-%m-%d
            ),
            size=(
                f"{ds['datasetSize'] / (1024 * 1024):.1f} MB"
                if "datasetSize" in ds
                else "N/A"
            ),
        )
        for ds in datasets
    ]


async def download_catalog_datasets(
    dataset_ids: list[str], analyst_db: AnalystDB
) -> list[str]:
    """Load selected datasets as pandas DataFrames

    Args:
        *args: list of dataset IDs to download

    Returns:
        list[AnalystDataset]: Dictionary of dataset names and data
    """
    datasets = [dr.Dataset.get(id_) for id_ in dataset_ids]
    if (
        sum([ds.size for ds in datasets if ds.size is not None])
        > MAX_AI_CATALOG_DATASET_SIZE
    ):
        raise ValueError(
            f"The requested AI Catalog datasets must total <= {int(MAX_AI_CATALOG_DATASET_SIZE)} bytes"
        )

    result_datasets: list[AnalystDataset] = []
    for dataset in datasets:
        try:
            df_records = cast(
                list[dict[str, Any]],
                dataset.get_as_dataframe().to_dict(orient="records"),
            )
            result_datasets.append(AnalystDataset(name=dataset.name, data=df_records))
            logger.info(f"Successfully downloaded {dataset.name}")
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset.name}: {str(e)}")
            continue
    names = []
    for result_dataset in result_datasets:
        await analyst_db.register_dataset(
            result_dataset, DataSourceType.CATALOG, dataset.size or 0
        )
        names.append(result_dataset.name)
    return names


async def _get_dictionary_batch(
    columns: list[str], df: pl.DataFrame, batch_size: int = 5
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
        async with AsyncLLMClient() as client:
            completion: DictionaryGeneration = await client.chat.completions.create(
                response_model=DictionaryGeneration,
                model=ALTERNATIVE_LLM_SMALL,
                messages=messages,
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

        tasks = [
            _get_dictionary_batch(batch, df, DICTIONARY_BATCH_SIZE)
            for batch in column_batches
        ]

        results = await asyncio.gather(*tasks)
        dictionary = sum(results, [])

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


def _validate_question_feasibility(
    question: str, available_columns: list[str]
) -> ValidatedQuestion | None:
    """Validate if a question can be answered with available data

    Checks if common data elements mentioned in the question exist in columns
    """
    # Convert question and columns to lowercase for matching
    question_lower = question.lower()
    columns_lower = [col.lower() for col in available_columns]

    # Extract potential column references from question
    words = set(re.findall(r"\b\w+\b", question_lower))

    # Find matches and missing terms
    found_columns = [col for col in columns_lower if any(word in col for word in words)]

    is_valid = len(found_columns) > 0
    if is_valid:
        return ValidatedQuestion(
            question=question,
        )
    return None


@log_api_call
async def suggest_questions(
    datasets: list[AnalystDataset], max_columns: int = 40
) -> list[ValidatedQuestion]:
    """Generate and validate suggested analysis questions

    Args:
        dictionary: DataFrame containing data dictionary
        max_columns: Maximum number of columns to include in prompt

    Returns:
        Dict containing:
            - questions: list of validated question objects
            - metadata: Dictionary of processing information
    """
    # Validate input
    dictionary = sum(
        [
            DataDictionary.from_analyst_df(
                ds.to_df(),
                column_descriptions=f"Column from dataset {ds.name}",
            ).column_descriptions
            for ds in datasets
        ],
        [],
    )

    if len(dictionary) < 1:
        raise ValueError("Dictionary DataFrame cannot be empty")

    # Limit columns for OpenAI prompt
    total_columns = len(dictionary)
    if total_columns > max_columns:
        # Take first and last 20 columns
        half_max = max_columns // 2
        first_half = dictionary[:half_max]
        last_half = dictionary[-half_max:]

        # Remove any duplicates
        dictionary = first_half + last_half

        # deduplicate
        dictionary = list({item.column: item for item in dictionary}.values())

    # Convert dictionary to format expected by OpenAI
    dict_data = {
        "columns": [d.column for d in dictionary],
        "descriptions": [d.description for d in dictionary],
        "data_types": [d.data_type for d in dictionary],
    }

    # Create OpenAI messages
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system", content=prompts.SYSTEM_PROMPT_SUGGEST_A_QUESTION
        ),
        ChatCompletionUserMessageParam(
            role="user", content=f"Data Dictionary:\n{json.dumps(dict_data)}"
        ),
    ]
    async with AsyncLLMClient() as client:
        completion: QuestionListGeneration = await client.chat.completions.create(
            response_model=QuestionListGeneration,
            model=ALTERNATIVE_LLM_SMALL,
            messages=messages,
        )

    available_columns = dict_data["columns"]
    validated_questions: list[ValidatedQuestion] = []

    for question in completion.questions:
        validated_question = _validate_question_feasibility(question, available_columns)
        if validated_question is not None:
            validated_questions.append(validated_question)

    return validated_questions


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


async def _generate_run_charts_python_code(
    request: RunChartsRequest,
    validation_error: InvalidGeneratedCode | None = None,
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
    async with AsyncLLMClient() as client:
        response: CodeGeneration = await client.chat.completions.create(
            response_model=CodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0,
            messages=messages,
        )
    return response.code


async def _generate_run_analysis_python_code(
    request: RunAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
    attempt: int = 0,
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
    async with AsyncLLMClient() as client:
        completion: CodeGeneration = await client.chat.completions.create(
            response_model=CodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
            max_retries=10,
        )
    logger.info("Code Gen complete")
    return completion.code


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
    sample_df = df.sample(min(100, len(df)))

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
async def rephrase_message(messages: ChatRequest) -> str:
    """Process chat messages history and return a new question

    Args:
        messages: list of message dictionaries with 'role' and 'content' fields

    Returns:
        Dict[str, str]: Dictionary containing response content
    """
    # Convert messages to string format for prompt
    messages_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages.messages]
    )

    prompt_messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            content=prompts.SYSTEM_PROMPT_REPHRASE_MESSAGE,
            role="system",
        ),
        ChatCompletionUserMessageParam(
            content=f"Message History:\n{messages_str}",
            role="user",
        ),
    ]
    async with AsyncLLMClient() as client:
        completion: EnhancedQuestionGeneration = await client.chat.completions.create(
            response_model=EnhancedQuestionGeneration,
            model=ALTERNATIVE_LLM_BIG,
            messages=prompt_messages,
        )

    return completion.enhanced_user_message


@reflect_code_generation_errors(max_attempts=7)
async def _run_charts(
    request: RunChartsRequest,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunChartsResult:
    """Generate and validate chart code with retry logic"""
    # Create messages for OpenAI
    start_time = datetime.now()

    if not request.dataset:
        raise ValueError("Input data cannot be empty")

    df = request.dataset.to_df().to_pandas()
    if exception_history is None:
        exception_history = []

    code = await _generate_run_charts_python_code(
        request, next(iter(exception_history), None)
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
async def run_charts(request: RunChartsRequest) -> RunChartsResult:
    """Execute analysis workflow on datasets."""
    try:
        chart_result = await _run_charts(request)
        return chart_result
    except ValidationError:
        return RunChartsResult(
            status="error", metadata=RunAnalysisResultMetadata(duration=0, attempts=1)
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
async def get_business_analysis(
    request: GetBusinessAnalysisRequest,
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

        # Get first 1000 rows as CSV with quoted values for context
        df_csv = df.head(750).to_csv(index=False, quoting=1)

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
        async with AsyncLLMClient() as client:
            completion: BusinessAnalysisGeneration = (
                await client.chat.completions.create(
                    response_model=BusinessAnalysisGeneration,
                    model=ALTERNATIVE_LLM_BIG,
                    temperature=0.1,
                    messages=messages,
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
async def _run_analysis(
    request: RunAnalysisRequest,
    analyst_db: AnalystDB,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunAnalysisResult:
    start_time = datetime.now()

    if not request.dataset_names:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []
    logger.info(f"Running analysis (attempt {len(exception_history)})")
    code = await _generate_run_analysis_python_code(
        request,
        analyst_db,
        next(iter(exception_history), None),
        attempt=len(exception_history),
    )
    logger.info("Code generated, preparing execution")
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
    analyst_db: AnalystDB,
) -> RunAnalysisResult:
    """Execute analysis workflow on datasets."""
    logger.debug("Entering run_analysis")
    log_memory()
    try:
        return await _run_analysis(request, analyst_db=analyst_db)
    except MaxReflectionAttempts as e:
        return RunAnalysisResult(
            status="error",
            metadata=RunAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
            ),
        )


async def _generate_database_analysis_code(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    validation_error: InvalidGeneratedCode | None = None,
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
        await analyst_db.get_data_dictionary(name) for name in request.dataset_names
    ]
    all_tables_info = [d.model_dump(mode="json") for d in dictionaries if d is not None]

    # Get sample data for all tables
    all_samples = []

    for table in request.dataset_names:
        df = (await analyst_db.get_dataset(table)).to_df().to_pandas()

        sample_str = f"Table: {table}\n{df.head(10).to_string()}"
        all_samples.append(sample_str)

    # Create messages for OpenAI
    messages: list[ChatCompletionMessageParam] = [
        Database.get_system_prompt(),
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
    async with AsyncLLMClient() as client:
        completion = await client.chat.completions.create(
            response_model=DatabaseAnalysisCodeGeneration,
            model=ALTERNATIVE_LLM_BIG,
            temperature=0.1,
            messages=messages,
        )

    return completion.code


@reflect_code_generation_errors(max_attempts=7)
async def _run_database_analysis(
    request: RunDatabaseAnalysisRequest,
    analyst_db: AnalystDB,
    exception_history: list[InvalidGeneratedCode] | None = None,
) -> RunDatabaseAnalysisResult:
    start_time = datetime.now()
    if not request.dataset_names:
        raise ValueError("Input data cannot be empty")

    if exception_history is None:
        exception_history = []

    sql_code = await _generate_database_analysis_code(
        request, analyst_db, next(iter(exception_history), None)
    )
    try:
        results = Database.execute_query(query=sql_code)
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
async def run_database_analysis(
    request: RunDatabaseAnalysisRequest, analyst_db: AnalystDB
) -> RunDatabaseAnalysisResult:
    """Execute analysis workflow on datasets."""
    try:
        return await _run_database_analysis(request, analyst_db)
    except MaxReflectionAttempts as e:
        return RunDatabaseAnalysisResult(
            status="error",
            metadata=RunDatabaseAnalysisResultMetadata(
                duration=e.duration,
                attempts=len(e.exception_history) if e.exception_history else 0,
                exception=AnalysisError.from_max_reflection_exception(e),
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
            run_charts(chart_request),
            get_business_analysis(business_request),
            return_exceptions=True,
        )

        return (result[0], result[1])
    elif enable_chart_generation:
        charts_result = await run_charts(chart_request)
        return charts_result, None
    else:
        business_result = await get_business_analysis(business_request)
        return None, business_result


async def run_complete_analysis(
    chat_request: ChatRequest,
    data_source: DataSourceType,
    datasets_names: list[str],
    analyst_db: AnalystDB,
    chat_id: str,
    enable_chart_generation: bool = True,
    enable_business_insights: bool = True,
) -> AsyncGenerator[Component | AnalysisGenerationError, None]:
    # Get enhanced message
    try:
        logger.info("Getting rephrased question...")
        enhanced_message = await rephrase_message(chat_request)
        logger.info("Getting rephrased question done")
        yield enhanced_message
    except ValidationError:
        yield AnalysisGenerationError("LLM Error, please retry")
        return
    assistant_message = AnalystChatMessage(
        role="assistant",
        content=enhanced_message,
        components=[EnhancedQuestionGeneration(enhanced_user_message=enhanced_message)],
    )
    messages = await analyst_db.get_chat_messages(chat_id=chat_id)
    if messages:
        user_message = messages[-1]
        user_message.in_progress = False
    await analyst_db.update_chat(
        chat_message=user_message,
        chat_id=chat_id,
        mode="overwrite",
    )
    await analyst_db.update_chat(
        chat_message=assistant_message,
        chat_id=chat_id,
        mode="append",
    )
    # Run main analysis
    logger.info("Start main analysis")
    try:
        is_database = data_source == DataSourceType.DATABASE
        logger.info("Getting analysis result...")
        log_memory()

        if is_database:
            analysis_result: (
                RunAnalysisResult | RunDatabaseAnalysisResult
            ) = await run_database_analysis(
                RunDatabaseAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analyst_db,
            )
        else:
            analysis_result = await run_analysis(
                RunAnalysisRequest(
                    dataset_names=datasets_names,
                    question=enhanced_message,
                ),
                analyst_db,
            )

        log_memory()
        logger.info("Getting analysis result done")

        if isinstance(analysis_result, BaseException):
            error_message = f"Error running initial analysis. Try rephrasing: {str(analysis_result)}"

            yield AnalysisGenerationError(error_message)

            await analyst_db.update_chat(
                chat_message=assistant_message,
                chat_id=chat_id,
                mode="overwrite",
            )
            return

        yield analysis_result
        assistant_message.components.append(analysis_result)
        await analyst_db.update_chat(
            chat_message=assistant_message,
            chat_id=chat_id,
            mode="overwrite",
        )

    except Exception as e:
        error_message = f"Error running initial analysis. Try rephrasing: {str(e)}"

        yield AnalysisGenerationError(error_message)

        await analyst_db.update_chat(
            chat_message=assistant_message,
            chat_id=chat_id,
            mode="overwrite",
        )
        return

    # Only proceed with additional analysis if we have valid initial results
    if not (
        analysis_result
        and analysis_result.dataset
        and (enable_chart_generation or enable_business_insights)
    ):
        return

    # Run concurrent analyses
    try:
        charts_result, business_result = await execute_business_analysis_and_charts(
            analysis_result,
            enhanced_message,
            enable_business_insights=enable_business_insights,
            enable_chart_generation=enable_chart_generation,
        )

        # Handle chart results
        if isinstance(charts_result, BaseException):
            error_message = "Error generating charts"

            yield AnalysisGenerationError(error_message)

            await analyst_db.update_chat(
                chat_message=assistant_message,
                chat_id=chat_id,
                mode="overwrite",
            )

        elif charts_result is not None:
            yield charts_result
            assistant_message.components.append(charts_result)
            await analyst_db.update_chat(
                chat_message=assistant_message,
                chat_id=chat_id,
                mode="overwrite",
            )

        # Handle business analysis results
        if isinstance(business_result, BaseException):
            error_message = "Error generating business insights"

            yield AnalysisGenerationError("Error generating business insights")

            await analyst_db.update_chat(
                chat_message=assistant_message,
                chat_id=chat_id,
                mode="overwrite",
            )
        elif business_result is not None:
            yield business_result
            assistant_message.components.append(business_result)
            assistant_message.in_progress = False

            await analyst_db.update_chat(
                chat_message=assistant_message,
                chat_id=chat_id,
                mode="overwrite",
            )

    except Exception as e:
        error_message = f"Error setting up additional analysis: {str(e)}"

        yield AnalysisGenerationError(error_message)

        assistant_message.in_progress = False
        await analyst_db.update_chat(
            chat_message=assistant_message,
            chat_id=chat_id,
            mode="overwrite",
        )


async def process_data_and_update_state(
    new_dataset_names: list[str],
    analyst_db: AnalystDB,
    data_source: str | DataSourceType,
) -> AsyncGenerator[str, None]:
    """Process datasets and yield progress updates asynchronously."""
    # Start processing and yield initial message
    logger.info("Starting data processing")
    log_memory()
    yield "Starting data processing"

    # Handle data cleansing based on the source
    # Convert string data_source to DataSourceType if needed
    data_source_type = (
        data_source
        if isinstance(data_source, DataSourceType)
        else DataSourceType(data_source)
    )
    if data_source_type != DataSourceType.DATABASE:
        try:
            logger.info("Cleansing datasets")
            yield "Cleansing datasets"
            for analysis_dataset_name in new_dataset_names:
                analysis_dataset = await analyst_db.get_dataset(
                    analysis_dataset_name, max_rows=None
                )
                cleansed_dataset = await cleanse_dataframe(analysis_dataset)
                await analyst_db.register_dataset(
                    cleansed_dataset, data_source=DataSourceType.GENERATED
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
            await analyst_db.register_data_dictionary(new_dictionary)
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
