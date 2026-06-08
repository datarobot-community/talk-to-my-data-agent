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

from unittest.mock import MagicMock, patch

import pytest

from core.data_connections.database.database_implementations import (
    JdbcPreviewOperator,
    get_database_operator,
)
from core.data_connections.database.database_interface import NoDatabaseOperator

_JDBC_PREVIEW = "core.data_connections.database.database_implementations.JdbcPreview"


def make_config(
    connection_type: str = "datarobot_jdbc",
) -> MagicMock:
    config = MagicMock()
    config.database_connection_type = connection_type
    return config


# ---------------------------------------------------------------------------
# JDBC factory branch
# ---------------------------------------------------------------------------


class TestGetDatabaseOperatorJdbc:
    def test_valid_config_returns_jdbc_operator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JDBC_URI", "jdbc:postgresql://localhost:5432/mydb")
        config = make_config()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            operator = get_database_operator(config)
        assert isinstance(operator, JdbcPreviewOperator)
        mock_jdbc.preview.assert_not_called()

    @pytest.mark.parametrize(
        "jdbc_uri,quoted_table",
        [
            ("jdbc:postgresql://localhost:5432/db", '"users"'),
            ("jdbc:mysql://localhost:3306/db", "`users`"),
            ("jdbc:sqlserver://localhost:1433;databaseName=db", "[users]"),
        ],
    )
    def test_supported_jdbc_uris_use_expected_dialect(
        self, monkeypatch: pytest.MonkeyPatch, jdbc_uri: str, quoted_table: str
    ) -> None:
        monkeypatch.setenv("JDBC_URI", jdbc_uri)
        config = make_config()
        with patch(_JDBC_PREVIEW):
            operator = get_database_operator(config)
        assert isinstance(operator, JdbcPreviewOperator)
        assert operator.query_friendly_name("users") == quoted_table

    def test_missing_uri_falls_back_to_no_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("JDBC_URI", raising=False)
        monkeypatch.delenv("MLOPS_RUNTIME_PARAM_JDBC_URI", raising=False)
        config = make_config()
        operator = get_database_operator(config)
        assert isinstance(operator, NoDatabaseOperator)

    def test_invalid_uri_prefix_falls_back_to_no_database(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JDBC_URI", "jdbc:oracle://localhost:1521/db")
        config = make_config()
        operator = get_database_operator(config)
        assert isinstance(operator, NoDatabaseOperator)
