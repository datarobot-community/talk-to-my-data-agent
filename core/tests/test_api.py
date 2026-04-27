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


from datetime import datetime
from typing import Any, Generator, cast, no_type_check
from unittest.mock import AsyncMock, MagicMock, patch

import plotly.graph_objects as go
import pytest
import pytest_asyncio
from datarobot_genai.core.utils.token_tracking import (
    HeuristicTokenCountingStrategy,
    TokenUsageTracker,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from polars import DataFrame

from core.analyst_db import (
    AnalystDB,
    DatasetHandler,
    DatasetMetadata,
    DatasetType,
    InternalDataSourceType,
)
from core.api import RunCompleteAnalysisRequestContext, run_complete_analysis
from core.llm_client import (
    AsyncLLMClient,
    ChatWrapper,
    CompletionsWrapper,
    InstructorClientWrapper,
)
from core.schema import (
    AnalystChatMessage,
    AnalystDataset,
    BusinessAnalysisGeneration,
    ChartGenerationExecutionResult,
    ChatRequest,
    CleansedDataset,
    CodeGeneration,
    DatabaseAnalysisCodeGeneration,
    DataDictionary,
    DataFrameWrapper,
    DatasetCleansedResponse,
    DictionaryGeneration,
    EnhancedQuestionGeneration,
    GetBusinessAnalysisResult,
    RunAnalysisResult,
    RunChartsResult,
)


async def get_cleansed_dataset_response(
    name: str,
    analyst_db: AnalystDB,
    skip: int = 0,
    limit: int = 10000,
) -> DatasetCleansedResponse:
    ds_display = await analyst_db.get_dataset(name)
    response = DatasetCleansedResponse(
        dataset_name=name,
        cleaning_report=None,
        dataset=None,
    )

    try:
        ds_display_cleansed = await analyst_db.get_cleansed_dataset(name)
        response.cleaning_report = ds_display_cleansed.generate_cleaning_report()
    except ValueError:
        pass

    df_display = ds_display.to_df()
    if skip > 0 or limit > 0:
        df_display = df_display.slice(skip, limit)

    response.dataset = AnalystDataset(
        name=name,
        data=df_display.to_dicts(),
    )
    return response


@pytest_asyncio.fixture(scope="module")
async def dataset_cleansed(
    dataset_loaded: AnalystDataset, analyst_db: AnalystDB
) -> CleansedDataset:
    from core.api import (
        cleanse_dataframe,
    )

    result = await cleanse_dataframe(dataset_loaded)
    await analyst_db.register_dataset(result, data_source=InternalDataSourceType.FILE)
    return result


def test_dataset_is_cleansed(dataset_cleansed: CleansedDataset) -> None:
    assert dataset_cleansed.cleaning_report is not None


@pytest_asyncio.fixture(scope="module")
async def cleansed_dataset_from_api(
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
    dataset_cleansed: CleansedDataset,
) -> DatasetCleansedResponse:
    cleansed_dataset = await get_cleansed_dataset_response(
        name=dataset_loaded.name, skip=0, limit=10000, analyst_db=analyst_db
    )
    return cleansed_dataset


@pytest_asyncio.fixture(scope="module")
async def cleansed_dataset_with_pagination(
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
    dataset_cleansed: CleansedDataset,
) -> tuple[DatasetCleansedResponse, DatasetCleansedResponse, DatasetCleansedResponse]:
    dataset1 = await get_cleansed_dataset_response(
        name=dataset_loaded.name, skip=0, limit=2, analyst_db=analyst_db
    )

    dataset2 = await get_cleansed_dataset_response(
        name=dataset_loaded.name, skip=2, limit=2, analyst_db=analyst_db
    )

    dataset3 = await get_cleansed_dataset_response(
        name=dataset_loaded.name, skip=10000, limit=2, analyst_db=analyst_db
    )

    return dataset1, dataset2, dataset3


def test_get_cleansed_dataset_by_name_api(
    dataset_cleansed: CleansedDataset,
    cleansed_dataset_from_api: DatasetCleansedResponse,
) -> None:
    # Verify we can retrieve the cleansed dataset by name
    assert cleansed_dataset_from_api is not None

    # Verify the dataset name matches
    assert cleansed_dataset_from_api.dataset_name == dataset_cleansed.name

    # Verify the cleaning report exists
    if dataset_cleansed.cleaning_report:
        assert cleansed_dataset_from_api.cleaning_report is not None

    # Verify the dataset structure
    dataset_api = cleansed_dataset_from_api.dataset

    # Verify the columns match
    assert dataset_api is not None
    assert dataset_api.columns == dataset_cleansed.to_df().columns

    # Verify the data records match
    assert len(dataset_api.to_df().to_dicts()) == len(
        dataset_cleansed.to_df().to_dicts()
    )


def test_get_cleansed_dataset_with_pagination(
    dataset_cleansed: CleansedDataset,
    cleansed_dataset_with_pagination: tuple[
        DatasetCleansedResponse, DatasetCleansedResponse, DatasetCleansedResponse
    ],
) -> None:
    dataset1, dataset2, dataset3 = cleansed_dataset_with_pagination

    # Convert the original dataset to a DataFrame
    original_df = dataset_cleansed.to_df()

    # Test the first paginated dataset (skip=0, limit=2)
    dataset_api1 = dataset1.dataset  # Access the `dataset` attribute
    assert dataset_api1 is not None
    assert len(dataset_api1.to_df().to_dicts()) == min(2, len(original_df.rows()))

    # Test the second paginated dataset (skip=2, limit=2)
    dataset_api2 = dataset2.dataset  # Access the `dataset` attribute
    expected_rows = original_df.rows(named=True)[2:4]  # Rows 2 and 3
    assert dataset_api2 is not None
    assert len(dataset_api2.to_df().to_dicts()) == len(expected_rows)

    # Third dataset should be empty (skip > dataset size)
    dataset_api3 = dataset3.dataset  # Access the `dataset` attribute
    assert dataset_api3 is not None
    assert len(dataset_api3.to_df().to_dicts()) == 0


@pytest_asyncio.fixture(scope="module")
async def data_dictionary(
    dataset_loaded: AnalystDataset,
    analyst_db: AnalystDB,
) -> DataDictionary:
    from core.api import (
        get_dictionary,
    )

    dictionary_result = await get_dictionary(dataset_loaded)
    await analyst_db.register_data_dictionary(dictionary_result)

    return dictionary_result


@pytest.fixture
def question() -> str:
    return "What are some interesting insights about the medication?"


@pytest.fixture
def mock_analyst_db() -> AsyncMock:
    analyst_db = AsyncMock(spec=AnalystDB)
    dataset_handler = AsyncMock(spec=DatasetHandler)
    analyst_db.dataset_handler = dataset_handler
    return analyst_db


@pytest.fixture(scope="function")
def analysis_context(
    question: str, mock_analyst_db: AsyncMock, dataset_cleansed: CleansedDataset
) -> RunCompleteAnalysisRequestContext:
    return RunCompleteAnalysisRequestContext(
        chat_request=ChatRequest(
            messages=[
                cast(
                    ChatCompletionMessageParam,
                    ChatCompletionUserMessageParam(
                        content=question, name="user", role="user"
                    ),
                )
            ]
        ),
        request=None,
        data_source=InternalDataSourceType.FILE,
        dataset_metadata=[
            DatasetMetadata(
                name=dataset_cleansed.name,
                dataset_type=DatasetType.CLEANSED,
                original_name=dataset_cleansed.name,
                created_at=datetime.now(),
                original_column_types=None,
                external_id=None,
                row_count=0,
                columns=dataset_cleansed.dataset.columns,
                data_source=InternalDataSourceType.FILE,
            )
        ],
        analyst_db=mock_analyst_db,
        chat_id="chatid",
        user_message_id="message",
        enable_business_insights=True,
        enable_chart_generation=True,
        token_tracker=TokenUsageTracker(strategy=HeuristicTokenCountingStrategy()),
    )


@pytest.fixture
def mock_execute_python() -> Generator[MagicMock, None, None]:
    with patch("core.api.execute_python") as execute:
        yield execute


@pytest.fixture
def mock_llm() -> Generator[AsyncMock, None, None]:
    async def create(response_model: type, **kwargs: Any) -> Any:
        if response_model == EnhancedQuestionGeneration:
            return EnhancedQuestionGeneration(enhanced_user_message="enhanced message")
        if response_model == BusinessAnalysisGeneration:
            return BusinessAnalysisGeneration(
                bottom_line="bottom line",
                additional_insights="additional insights",
                follow_up_questions=[],
            )
        if response_model == DatabaseAnalysisCodeGeneration:
            return DatabaseAnalysisCodeGeneration(code="SELECT 1", description="code")
        if response_model == DictionaryGeneration:
            return DictionaryGeneration(columns=[], descriptions=[])
        if response_model == CodeGeneration:
            return CodeGeneration(code="python", description="code")

    with (
        patch(
            "core.llm_client.InstructorClientWrapper",
            spec=InstructorClientWrapper,
            new_callable=AsyncMock,
        ) as instructor_wrapper,
        patch(
            "core.llm_client.ChatWrapper", spec=ChatWrapper, new_callable=AsyncMock
        ) as chat_wrapper,
        patch(
            "core.llm_client.CompletionsWrapper",
            spec=CompletionsWrapper,
            new_callable=AsyncMock,
        ) as completions_wrapper,
    ):

        @no_type_check
        async def yield_instructor_wrapper(*args, **kwargs):
            return instructor_wrapper

        with (
            patch(
                "core.api.AsyncLLMClient", new_callable=MagicMock, spec=AsyncLLMClient
            ) as llm_client,
        ):
            llm_client.return_value = llm_client
            instructor_wrapper.return_value = instructor_wrapper
            llm_client.__aenter__ = yield_instructor_wrapper
            instructor_wrapper.chat = chat_wrapper
            chat_wrapper.completions = completions_wrapper
            completions_wrapper.create.side_effect = create

            yield completions_wrapper


@no_type_check
@pytest.mark.asyncio
async def test_run_complete_analysis_request(
    analysis_context: RunCompleteAnalysisRequestContext,
    dataset_cleansed: CleansedDataset,
    mock_llm: AsyncMock,
    mock_execute_python: MagicMock,
) -> None:
    analysis_context.analyst_db.get_chat_message.return_value = AnalystChatMessage(
        role="user", content="Question", components=[]
    )
    analysis_context.analyst_db.get_data_dictionary.return_value = DataDictionary(
        name="dataset", column_descriptions=[]
    )
    analysis_context.analyst_db.get_cleansed_dataset.return_value = dataset_cleansed
    analysis_context.analyst_db.get_dataset.return_value = dataset_cleansed.dataset
    analysis_context.analyst_db.add_chat_message.return_value = "added_message_id"

    figure = MagicMock(spec=go.Figure)

    def execute_python(output_type: type, **kwargs) -> Any:
        if output_type == AnalystDataset:
            return AnalystDataset(
                name="return", data=DataFrameWrapper(df=DataFrame({"a": [1, 2]}))
            )
        if output_type == ChartGenerationExecutionResult:
            return ChartGenerationExecutionResult(fig1=figure, fig2=figure)

    mock_execute_python.side_effect = execute_python

    messages = []
    async for message in run_complete_analysis(
        chat_request=analysis_context.chat_request,
        data_source=analysis_context.data_source,
        dataset_metadata=analysis_context.dataset_metadata,
        analyst_db=analysis_context.analyst_db,
        chat_id=analysis_context.chat_id,
        message_id=analysis_context.user_message_id,
        request=analysis_context.request,
        enable_business_insights=analysis_context.enable_business_insights,
        enable_chart_generation=analysis_context.enable_chart_generation,
    ):
        messages.append(message)

    assert (
        len(messages) == 4
    )  # rephrased message, run analysis result, charts, business analysis
    assert messages[0] == "enhanced message"
    assert isinstance(messages[1], RunAnalysisResult)
    assert isinstance(messages[2], RunChartsResult)
    assert isinstance(messages[3], GetBusinessAnalysisResult)

    message_update_calls = (
        analysis_context.analyst_db.update_chat_message.call_args_list
    )
    messages = [call.kwargs["message"] for call in message_update_calls]
    steps = [m.step_value for m in messages]
    assert steps == [
        "GENERATING_QUERY",  # Initial creation
        None,  # Marking user message closed
        "GENERATING_QUERY",  # Generate python
        "RUNNING_QUERY",  # Run python
        "ANALYZING_RESULTS",  # Start running charts/business
        "ANALYZING_RESULTS",  # Run charts/business
        "ANALYZING_RESULTS",  # Run charts/business
        "ANALYZING_RESULTS",  # Mark as success
    ]
    in_progress = [m.in_progress for m in messages]
    assert in_progress == [True, False] + [True] * 5 + [False]
    error = [m.error for m in messages]
    assert error == [None] * 8


@pytest.fixture
def run_analysis_result_canned() -> RunAnalysisResult:
    with open("tests/models/run_analysis_result.json") as f:
        return RunAnalysisResult.model_validate_json(f.read())


@pytest.fixture
def run_charts_result_canned() -> RunChartsResult:
    with open("tests/models/run_charts_result.json") as f:
        return RunChartsResult.model_validate_json(f.read())


@pytest.fixture
def run_business_result_canned() -> GetBusinessAnalysisResult:
    with open("tests/models/run_business_result.json") as f:
        return GetBusinessAnalysisResult.model_validate_json(f.read())


# TODO: add tests of reflection in run_analysis once test_api refactored/cleaned up


# --- Tests for _friendly_llm_error -----------------------------------------


def _mk_response(status: int = 400, body: str = "") -> "Any":
    import httpx

    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://example.invalid/v1/chat"),
        content=body.encode() if body else b"",
    )


def _mk_request() -> "Any":
    import httpx

    return httpx.Request("POST", "https://example.invalid/v1/chat")


def test_friendly_llm_error_timeout() -> None:
    from core.api import _friendly_llm_error

    assert "did not respond in time" in _friendly_llm_error(TimeoutError())


def test_friendly_llm_error_openai_api_timeout() -> None:
    from openai import APITimeoutError

    from core.api import _friendly_llm_error

    assert "did not respond in time" in _friendly_llm_error(
        APITimeoutError(_mk_request())
    )


def test_friendly_llm_error_authentication() -> None:
    from openai import AuthenticationError

    from core.api import _friendly_llm_error

    exc = AuthenticationError("bad token", response=_mk_response(401), body=None)
    assert "credentials are invalid or expired" in _friendly_llm_error(exc)


def test_friendly_llm_error_rate_limit() -> None:
    from openai import RateLimitError

    from core.api import _friendly_llm_error

    exc = RateLimitError("slow down", response=_mk_response(429), body=None)
    assert "rate-limiting" in _friendly_llm_error(exc)


def test_friendly_llm_error_not_found_enriched() -> None:
    from openai import NotFoundError

    from core.api import _friendly_llm_error

    exc = NotFoundError(
        'DatarobotException - {"detail":"Deployment X not found"}',
        response=_mk_response(404),
        body=None,
    )
    result = _friendly_llm_error(exc)
    assert "model or deployment was not found" in result
    assert "Detail: Deployment X not found" in result


def test_friendly_llm_error_reporter_case() -> None:
    from openai import APIConnectionError

    from core.api import _friendly_llm_error

    exc = APIConnectionError(
        message=(
            'DatarobotException - {"detail":"Model '
            'azure/gpt-5-1-2025-11-13- not found in catalog"}'
        ),
        request=_mk_request(),
    )
    result = _friendly_llm_error(exc)
    assert "Could not reach the LLM service" in result
    assert "model 'azure/gpt-5-1-2025-11-13-' not found in catalog" in result


def test_friendly_llm_error_internal_server() -> None:
    from openai import InternalServerError

    from core.api import _friendly_llm_error

    exc = InternalServerError("kaboom", response=_mk_response(500), body=None)
    assert "internal error" in _friendly_llm_error(exc)


def test_friendly_llm_error_bad_request_no_detail() -> None:
    from openai import BadRequestError

    from core.api import _friendly_llm_error

    exc = BadRequestError("invalid request", response=_mk_response(400), body=None)
    result = _friendly_llm_error(exc)
    assert "rejected as invalid" in result
    assert "Detail:" not in result


def test_friendly_llm_error_status_error_includes_code() -> None:
    from openai import APIStatusError

    from core.api import _friendly_llm_error

    exc = APIStatusError("teapot", response=_mk_response(418), body=None)
    result = _friendly_llm_error(exc)
    assert "418" in result


def test_friendly_llm_error_unwraps_instructor_retry() -> None:
    from instructor.core.exceptions import InstructorRetryException
    from instructor.core.retry import FailedAttempt
    from openai import AuthenticationError

    from core.api import _friendly_llm_error

    inner = AuthenticationError("bad token", response=_mk_response(401), body=None)
    retry = InstructorRetryException(
        "retry exhausted",
        n_attempts=2,
        total_usage=0,
        failed_attempts=[
            FailedAttempt(attempt_number=1, exception=inner, completion=None)
        ],
    )
    result = _friendly_llm_error(retry)
    assert "credentials are invalid or expired" in result
    assert "<failed_attempts>" not in result
    assert "<" not in result


def test_friendly_llm_error_unwraps_via_cause() -> None:
    from instructor.core.exceptions import InstructorRetryException
    from openai import APITimeoutError

    from core.api import _friendly_llm_error

    cause = APITimeoutError(_mk_request())
    retry = InstructorRetryException(
        "retry exhausted",
        n_attempts=1,
        total_usage=0,
        failed_attempts=[],
    )
    retry.__cause__ = cause
    assert "did not respond in time" in _friendly_llm_error(retry)


def test_friendly_llm_error_unknown_type_passes_through_message() -> None:
    from core.api import _friendly_llm_error

    result = _friendly_llm_error(RuntimeError("boom"))
    assert "boom" in result
    assert len(result) <= 200


def test_friendly_llm_error_empty_message_falls_back_to_generic() -> None:
    from core.api import _friendly_llm_error

    result = _friendly_llm_error(RuntimeError(""))
    assert "unexpected error" in result
    assert len(result) <= 200


def test_friendly_llm_error_strips_xml_and_clamps_length() -> None:
    from core.api import _friendly_llm_error

    noisy = RuntimeError("<x>" * 2000)
    result = _friendly_llm_error(noisy)
    assert len(result) <= 200
    assert "<" not in result and ">" not in result


def test_chat_message_error_uses_friendly_helper() -> None:
    from instructor.core.exceptions import InstructorRetryException
    from instructor.core.retry import FailedAttempt
    from openai import APIConnectionError

    from core.api import _friendly_llm_error

    inner = APIConnectionError(
        message=(
            'DatarobotException - {"detail":"Model '
            'azure/gpt-5-1-2025-11-13- not found in catalog"}'
        ),
        request=_mk_request(),
    )
    retry = InstructorRetryException(
        "retry exhausted",
        n_attempts=2,
        total_usage=0,
        failed_attempts=[
            FailedAttempt(attempt_number=1, exception=inner, completion=None),
        ],
    )

    prefix = "Failed to process your question: "
    rendered = f"{prefix}{_friendly_llm_error(retry)}"

    assert rendered.startswith(prefix)
    assert "Could not reach the LLM service" in rendered
    assert "model 'azure/gpt-5-1-2025-11-13-' not found in catalog" in rendered
    assert "<failed_attempts>" not in rendered
    assert "<" not in rendered and ">" not in rendered
    assert len(rendered) <= len(prefix) + 200
