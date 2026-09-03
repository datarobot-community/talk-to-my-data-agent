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

import requests
from datarobot.errors import (
    AppPlatformError,
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

from core.api_exceptions import ApplicationUsageException, UsageExceptionType

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


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a DataRobot ClientError/ServerError or a requests
    HTTPError. ClientError (4xx) and ServerError (5xx) share the AppPlatformError base
    and both carry ``status_code``, so one check covers both."""
    if isinstance(exc, AppPlatformError):
        return getattr(exc, "status_code", None)
    if isinstance(exc, HTTPError) and exc.response is not None:
        return exc.response.status_code
    return None


def _walk_causes(exc: BaseException) -> Generator[BaseException, None, None]:
    """Yield the exception and its explicit ``__cause__`` chain.

    Only ``__cause__`` (set by ``raise ... from e``, which the DataRobot handlers use)
    is followed — not the implicit ``__context__`` — so an unrelated exception that
    merely happened to be in flight when the real error was raised cannot flip the
    classification (e.g. a bad-SQL 400 carrying a stale ConnectionError in its context).
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


TIMEOUT_MESSAGE = (
    "The query exceeded the time limit — it is too expensive. Rewrite it to do less "
    "work: add filters to the WHERE clause, add a LIMIT, pre-aggregate, or avoid cross joins."
)
OUTAGE_MESSAGE = "The database is unavailable or refused the request."


def classify_db_failure(exc: BaseException | None) -> str | None:
    """Return a user-facing reason when a database failure is one that regenerating
    SQL cannot fix (a timeout, an outage, a rate limit, or an auth error), else None
    to keep it as ordinary retryable bad SQL.

    Walks the explicit __cause__ chain (the DataRobot handlers raise `... from e`, and
    the recipe path wraps in RecipeError) and unwraps requests.HTTPError, so a wrapped
    transport error is still classified. Only __cause__ is followed — not the implicit
    __context__ — so an unrelated error merely in flight cannot flip the result.

    The monolith flattens *connector-level* gRPC failures into HTTP status
    (DEADLINE_EXCEEDED->408, RESOURCE_EXHAUSTED->429, PERMISSION_DENIED->403, everything
    else including INTERNAL/UNAVAILABLE->400), so a mid-query connector outage arriving as
    400 is indistinguishable from bad SQL and stays retryable. A gateway/platform 5xx
    (502/503/504, raised as a datarobot ServerError) is a genuine outage and IS classified.
    """
    if exc is None:
        return None
    for e in _walk_causes(exc):
        status = _http_status(e)
        if isinstance(e, (requests.ReadTimeout, AsyncTimeoutError)) or status == 408:
            return TIMEOUT_MESSAGE
        if (
            isinstance(e, requests.ConnectionError)
            or status in (401, 403, 429)
            or (status is not None and status >= 500)
        ):
            return OUTAGE_MESSAGE
        # Generic timeout that isn't a connection failure.
        if isinstance(e, requests.Timeout):
            return TIMEOUT_MESSAGE
    return None


def _handle_403_client_error(client_error: ClientError) -> None:
    """
    Handle 403 ClientError, raising ApplicationUsageException for seat license restrictions.

    Args:
        client_error: The ClientError with status_code 403

    Raises:
        ApplicationUsageException: If the error is due to seat license restrictions
        ClientError: Re-raises the original error if not seat license related
    """
    error_message = (
        client_error.json.get("message", str(client_error))
        if client_error.json
        else str(client_error)
    )
    if "seat license" in error_message.lower():
        # Chain the 403 so classify_db_failure's __cause__ walk sees it and fails fast
        # (outage) instead of regenerating SQL 7x against an entitlement error (APP-6770).
        raise ApplicationUsageException(
            UsageExceptionType.USER_ACCESS_DENIED,
            "Feature unavailable due to seat license restrictions. Please contact your DataRobot administrator.",
        ) from client_error
    else:
        raise client_error


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
    except ValueError as e:
        # ClientError is often wrapped in ValueError: ('message', ClientError(...))
        client_error = None
        if e.args and len(e.args) >= 2:
            for arg in e.args:
                if isinstance(arg, ClientError):
                    client_error = arg
                    break

        if client_error:
            # Handle the wrapped ClientError
            if client_error.status_code == 404:
                message = f"{resource} not found (404.)"
                logger.log(not_found_severity, message, exc_info=True)
            elif client_error.status_code == 403:
                _handle_403_client_error(client_error)
            else:
                message = (
                    f"Exception in retrieving {resource} ({client_error.status_code})."
                )
                logger.log(other_severity, message, exc_info=True)
            if exception_type:
                raise exception_type(message) from client_error
            else:
                raise client_error
        else:
            # No ClientError found - treat as unexpected exception
            message = f"Unexpected exception in retrieving {resource}."
            logger.log(other_severity, message, exc_info=True)
            if exception_type:
                raise exception_type(message) from e
            else:
                raise
    except ClientError as e:
        if e.status_code == 404:
            message = f"{resource} not found (404.)"
            logger.log(not_found_severity, message, exc_info=True)
        elif e.status_code == 403:
            _handle_403_client_error(e)
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
