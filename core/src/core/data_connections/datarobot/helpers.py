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
from contextlib import contextmanager
from typing import Callable, Generator, ParamSpec, TypeVar, cast

from datarobot.errors import (
    AsyncProcessUnsuccessfulError,
    AsyncTimeoutError,
    ClientError,
)
from requests import HTTPError
from tenacity import (
    after_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from core.api_exceptions import ApplicationUsageException

logger = logging.getLogger()

ASYNC_PROCESS_PRE_JOB_TOKEN = "Job Data:"


def find_underlying_client_message(exc: BaseException) -> str | None:
    stack: list[BaseException] = [exc]
    while stack:
        exc = stack.pop()
        if isinstance(exc, ClientError) and "message" in exc.json:
            return cast(str, exc.json["message"])
        if isinstance(exc, HTTPError) and "message" in exc.response.json():
            return cast(str, exc.response.json()["message"])
        if (
            isinstance(exc, AsyncProcessUnsuccessfulError)
            and exc.args
            and isinstance(exc.args[0], str)
        ):
            message: str = exc.args[0]
            if ASYNC_PROCESS_PRE_JOB_TOKEN in message:
                index = message.find(ASYNC_PROCESS_PRE_JOB_TOKEN)
                if index:
                    json_portion = message[index + len(ASYNC_PROCESS_PRE_JOB_TOKEN) :]
                    try:
                        message = json.loads(json_portion)["message"]
                    except Exception:
                        message = json_portion
            return message
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


class RecipeError(RuntimeError):
    """
    Exception class for initializing/using Spark Recipe
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


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
    exception_type: type[Exception] | None = RecipeError,
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
