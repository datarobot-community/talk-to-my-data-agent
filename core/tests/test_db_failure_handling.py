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

"""Unit tests for APP-6770 database-failure classification and handling.

A failure that regenerating SQL cannot fix (timeout / outage / rate-limit / auth) is
classified once, at the choke point, into a DatabaseFailure that escapes the reflection
loop and fails fast; everything else stays retryable bad SQL.
"""

from unittest.mock import MagicMock

import pytest
import requests
from datarobot.errors import AsyncTimeoutError, ClientError, ServerError

from core.code_execution import DatabaseFailure, InvalidGeneratedCode
from core.data_connections.datarobot.helpers import (
    OUTAGE_MESSAGE,
    TIMEOUT_MESSAGE,
    RecipeError,
    _handle_403_client_error,
    classify_db_failure,
)
from core.schema import AnalysisError


def _client_error(status: int) -> ClientError:
    return ClientError("boom", status)


def _server_error(status: int) -> ServerError:
    return ServerError("boom", status)


def _http_error(status: int) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status
    return requests.HTTPError("boom", response=response)


def _wrap_in_recipe_error(exc: BaseException) -> RecipeError:
    """Mimic handle_datarobot_error: RecipeError with the original on __cause__."""
    err = RecipeError("recipe failed")
    err.__cause__ = exc
    return err


class TestClassifyDbFailure:
    @pytest.mark.parametrize(
        "exc",
        [
            requests.ReadTimeout("read timed out"),
            AsyncTimeoutError("async timed out"),
            _client_error(408),
            requests.Timeout("generic timeout"),
        ],
    )
    def test_timeout_bucket(self, exc: BaseException) -> None:
        assert classify_db_failure(exc) == TIMEOUT_MESSAGE

    @pytest.mark.parametrize(
        "exc",
        [
            requests.ConnectionError("refused"),
            requests.ConnectTimeout("could not connect"),
            _client_error(429),
            _client_error(403),
            _client_error(401),
            _server_error(500),  # gateway/platform 5xx is a real outage
            _server_error(502),
            _server_error(503),
        ],
    )
    def test_outage_bucket(self, exc: BaseException) -> None:
        assert classify_db_failure(exc) == OUTAGE_MESSAGE

    @pytest.mark.parametrize(
        "exc",
        [
            _client_error(400),  # bad SQL — the common case
            _client_error(404),  # not-ready / not-found stays retryable
            RuntimeError("something odd"),
            None,
        ],
    )
    def test_retryable_bucket_returns_none(self, exc: BaseException | None) -> None:
        assert classify_db_failure(exc) is None

    def test_walks_recipe_error_cause_chain(self) -> None:
        # The DataRobot recipe path wraps the real error in RecipeError.__cause__.
        assert classify_db_failure(_wrap_in_recipe_error(AsyncTimeoutError())) == (
            TIMEOUT_MESSAGE
        )
        assert classify_db_failure(
            _wrap_in_recipe_error(requests.ConnectionError())
        ) == OUTAGE_MESSAGE
        assert classify_db_failure(_wrap_in_recipe_error(_server_error(503))) == (
            OUTAGE_MESSAGE
        )

    def test_classifies_chained_http_error(self) -> None:
        assert classify_db_failure(_http_error(408)) == TIMEOUT_MESSAGE
        assert classify_db_failure(_http_error(429)) == OUTAGE_MESSAGE
        assert classify_db_failure(_http_error(503)) == OUTAGE_MESSAGE
        assert classify_db_failure(_http_error(400)) is None

    def test_ignores_implicit_context_chain(self) -> None:
        # Bad SQL (400) raised while handling an unrelated ConnectionError sets
        # __context__ but not __cause__. We must NOT fail-fast a self-correctable
        # query just because an outage was incidentally in flight.
        try:
            try:
                raise requests.ConnectionError("stale")
            except requests.ConnectionError:
                raise _client_error(400)
        except ClientError as e:
            assert classify_db_failure(e) is None

    def test_seat_license_403_is_classified_as_outage(self) -> None:
        # A seat-license 403 raises ApplicationUsageException chained (`from`) to the
        # 403, so the cause walk sees it and fails fast instead of amplifying 7x.
        client_error = ClientError(
            "denied", 403, json={"message": "seat license required"}
        )
        try:
            _handle_403_client_error(client_error)
        except Exception as exc:  # noqa: BLE001 - asserting the chained classification
            assert exc.__cause__ is client_error
            assert classify_db_failure(exc) == OUTAGE_MESSAGE
        else:
            raise AssertionError("expected _handle_403_client_error to raise")

    def test_follows_explicit_cause_chain(self) -> None:
        # An explicit `raise ... from outage` IS followed.
        try:
            try:
                raise requests.ConnectionError("down")
            except requests.ConnectionError as src:
                raise _client_error(400) from src
        except ClientError as e:
            assert classify_db_failure(e) == OUTAGE_MESSAGE


class TestDatabaseFailure:
    def test_is_not_invalid_generated_code(self) -> None:
        # NOT a subclass, so it escapes the retry loop instead of amplifying.
        assert not issubclass(DatabaseFailure, InvalidGeneratedCode)
        err = DatabaseFailure("down", code="SELECT 1", exception=RuntimeError("x"))
        assert err.code == "SELECT 1"
        assert isinstance(err.exception, RuntimeError)
        assert str(err) == "down"


class TestAnalysisErrorFromDatabaseFailure:
    def test_carries_generated_sql_and_reason(self) -> None:
        err = DatabaseFailure(
            OUTAGE_MESSAGE, code="SELECT 1", exception=requests.ConnectionError()
        )
        analysis_error = AnalysisError.from_database_failure(err)
        assert analysis_error.exception_history is not None
        entry = analysis_error.exception_history[0]
        assert entry.code == "SELECT 1"
        # The user-facing reason survives even though a bare ConnectionError()
        # stringifies to "" — the message comes off DatabaseFailure, not the cause.
        assert entry.exception_str == OUTAGE_MESSAGE
        assert entry.stderr == OUTAGE_MESSAGE
