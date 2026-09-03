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

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest
from pydantic import ValidationError

from core.code_execution import DatabaseFailure
from core.credentials import JDBCCredentials
from core.data_connections.database.database_implementations import JdbcPreviewOperator
from core.data_connections.datarobot.helpers import OUTAGE_MESSAGE

_JDBC_PREVIEW = "core.data_connections.database.database_implementations.JdbcPreview"


def make_credentials(
    uri: str = "jdbc:postgresql://localhost:5432/mydb",
) -> JDBCCredentials:
    return JDBCCredentials.model_validate(
        {"JDBC_URI": uri, "JDBC_CONNECTION_PARAMETERS": None}
    )


def make_operator(
    uri: str = "jdbc:postgresql://localhost:5432/mydb",
) -> JdbcPreviewOperator:
    return JdbcPreviewOperator(credentials=make_credentials(uri))


def make_preview_result(
    columns: list[str],
    records: list[list],  # type: ignore[type-arg]
) -> MagicMock:
    result = MagicMock()
    schema_entries = []
    for col in columns:
        entry = MagicMock()
        entry.name = col  # MagicMock(name=...) sets display name, not .name attribute
        entry.data_type = "VARCHAR"
        schema_entries.append(entry)
    result.result_schema = schema_entries
    result.columns = columns  # always populated by the SDK, even for empty results
    result.records = records
    return result


# ---------------------------------------------------------------------------
# JDBCCredentials
# ---------------------------------------------------------------------------


class TestJDBCCredentials:
    def test_valid_postgresql_uri(self) -> None:
        creds = make_credentials("jdbc:postgresql://localhost:5432/db")
        assert creds.jdbc_uri == "jdbc:postgresql://localhost:5432/db"

    def test_valid_mysql_uri(self) -> None:
        creds = make_credentials("jdbc:mysql://localhost:3306/db")
        assert creds.jdbc_uri == "jdbc:mysql://localhost:3306/db"

    def test_valid_sqlserver_uri(self) -> None:
        creds = make_credentials("jdbc:sqlserver://localhost:1433;databaseName=db")
        assert creds.jdbc_uri.startswith("jdbc:sqlserver://")

    def test_valid_snowflake_uri(self) -> None:
        creds = make_credentials("jdbc:snowflake://account.snowflakecomputing.com/")
        assert creds.jdbc_uri.startswith("jdbc:snowflake://")

    def test_valid_sap_uri(self) -> None:
        creds = make_credentials("jdbc:sap://host:443")
        assert creds.jdbc_uri.startswith("jdbc:sap://")

    def test_valid_bigquery_uri(self) -> None:
        creds = make_credentials("jdbc:bigquery://https://www.googleapis.com/bigquery/v2:443")
        assert creds.jdbc_uri.startswith("jdbc:bigquery://")

    def test_valid_databricks_uri(self) -> None:
        creds = make_credentials("jdbc:databricks://adb-1234.4.azuredatabricks.net:443")
        assert creds.jdbc_uri.startswith("jdbc:databricks://")

    def test_valid_redshift_uri(self) -> None:
        creds = make_credentials("jdbc:redshift://cluster.us-east-1.redshift.amazonaws.com:5439/mydb")
        assert creds.jdbc_uri.startswith("jdbc:redshift://")

    def test_valid_redshift_iam_uri(self) -> None:
        creds = make_credentials("jdbc:redshift:iam://cluster.us-east-1.redshift.amazonaws.com:5439/mydb")
        assert creds.jdbc_uri.startswith("jdbc:redshift:iam://")

    def test_invalid_uri_prefix_raises(self) -> None:
        with pytest.raises(ValidationError):
            make_credentials("jdbc:oracle://localhost:1521/db")

    def test_missing_uri_raises(self) -> None:
        with pytest.raises(ValidationError):
            JDBCCredentials.model_validate({"JDBC_URI": None})

    def test_repr_masks_uri(self) -> None:
        creds = make_credentials()
        assert "jdbc:postgresql" not in repr(creds)
        assert "***" in repr(creds)

    def test_connection_parameters_optional(self) -> None:
        creds = make_credentials()
        assert creds.jdbc_connection_parameters is None

    @pytest.mark.parametrize("blank_value", ["", "   "])
    def test_connection_parameters_blank_string_treated_as_none(
        self, blank_value: str
    ) -> None:
        creds = JDBCCredentials.model_validate(
            {
                "JDBC_URI": "jdbc:postgresql://localhost:5432/db",
                "JDBC_CONNECTION_PARAMETERS": blank_value,
            }
        )
        assert creds.jdbc_connection_parameters is None


# ---------------------------------------------------------------------------
# validate_connection
# ---------------------------------------------------------------------------


class TestValidateConnection:
    def test_calls_preview_with_select_1(self) -> None:
        operator = make_operator()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            operator.validate_connection()
            mock_jdbc.preview.assert_called_once_with(
                jdbc_url="jdbc:postgresql://localhost:5432/mydb",
                sql="SELECT 1",
                max_rows=1,
                parameters={},
            )

    def test_sdk_error_raises_value_error(self) -> None:
        operator = make_operator()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.side_effect = RuntimeError("conn refused")
            with pytest.raises(ValueError, match="JDBC connection validation failed"):
                operator.validate_connection()


# ---------------------------------------------------------------------------
# get_tables — dialect SQL selection
# ---------------------------------------------------------------------------


class TestGetTables:
    @pytest.mark.asyncio
    async def test_postgresql_uses_public_schema_filter(self) -> None:
        operator = make_operator("jdbc:postgresql://localhost:5432/mydb")
        result = make_preview_result(["table_name"], [["users"], ["orders"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "table_schema = 'public'" in sql
            assert tables == ["users", "orders"]

    @pytest.mark.asyncio
    async def test_mysql_uses_database_function(self) -> None:
        operator = make_operator("jdbc:mysql://localhost:3306/mydb")
        result = make_preview_result(["table_name"], [["products"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "DATABASE()" in sql
            assert tables == ["products"]

    @pytest.mark.asyncio
    async def test_sqlserver_uses_information_schema(self) -> None:
        operator = make_operator("jdbc:sqlserver://localhost:1433;databaseName=mydb")
        result = make_preview_result(["TABLE_NAME"], [["customers"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "INFORMATION_SCHEMA" in sql
            assert "DATABASE()" not in sql
            assert tables == ["customers"]

    @pytest.mark.asyncio
    async def test_snowflake_uses_current_schema(self) -> None:
        operator = make_operator("jdbc:snowflake://account.snowflakecomputing.com/")
        result = make_preview_result(["TABLE_NAME"], [["employees"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "CURRENT_SCHEMA()" in sql
            assert "VIEW" in sql
            assert tables == ["employees"]

    @pytest.mark.asyncio
    async def test_sap_uses_sys_tables(self) -> None:
        operator = make_operator("jdbc:sap://host:443")
        result = make_preview_result(["TABLE_NAME"], [["orders"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "SYS.TABLES" in sql
            assert tables == ["orders"]

    @pytest.mark.asyncio
    async def test_bigquery_uses_information_schema(self) -> None:
        operator = make_operator("jdbc:bigquery://https://www.googleapis.com/bigquery/v2:443")
        result = make_preview_result(["TABLE_NAME"], [["sales"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "INFORMATION_SCHEMA.TABLES" in sql
            assert tables == ["sales"]

    @pytest.mark.asyncio
    async def test_databricks_uses_information_schema(self) -> None:
        operator = make_operator("jdbc:databricks://adb-1234.4.azuredatabricks.net:443")
        result = make_preview_result(["table_name"], [["orders"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "information_schema.tables" in sql
            assert "current_schema()" in sql
            assert "'MANAGED'" in sql
            assert "'EXTERNAL'" in sql
            assert "'BASE TABLE'" not in sql
            assert tables == ["orders"]

    @pytest.mark.asyncio
    async def test_redshift_uses_information_schema(self) -> None:
        operator = make_operator("jdbc:redshift://cluster.us-east-1.redshift.amazonaws.com:5439/mydb")
        result = make_preview_result(["table_name"], [["sales"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            tables = await operator.get_tables()
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert "information_schema.tables" in sql
            assert "current_schema()" in sql
            assert "'BASE TABLE'" in sql
            assert "'VIEW'" in sql
            assert tables == ["sales"]

    @pytest.mark.asyncio
    async def test_raises_database_failure_when_unreachable(self) -> None:
        operator = make_operator()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.side_effect = RuntimeError("timeout")
            with pytest.raises(DatabaseFailure) as excinfo:
                await operator.get_tables()
        assert str(excinfo.value) == OUTAGE_MESSAGE

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_tables(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A reachable database with no tables: the preview succeeds with zero
        # records, so get_tables returns [] via the happy path — not the failure
        # handler, which would log and raise.
        operator = make_operator()
        result = MagicMock()
        result.records = []
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            with caplog.at_level(logging.ERROR):
                tables = await operator.get_tables()
        assert tables == []
        assert "failed to fetch tables" not in caplog.text


# ---------------------------------------------------------------------------
# execute_query
# ---------------------------------------------------------------------------


class TestExecuteQuery:
    @pytest.mark.asyncio
    async def test_uses_full_row_cap(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id", "name"], [[1, "alice"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            await operator.execute_query("SELECT * FROM users")
            assert mock_jdbc.preview.call_args.kwargs["max_rows"] == 10_000

    @pytest.mark.asyncio
    async def test_returns_dataframe(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id", "name"], [[1, "alice"], [2, "bob"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            frame = await operator.execute_query("SELECT * FROM users")
            assert isinstance(frame, pl.DataFrame)
            assert frame.to_dicts() == [
                {"id": 1, "name": "alice"},
                {"id": 2, "name": "bob"},
            ]

    @pytest.mark.asyncio
    async def test_empty_result_preserves_column_types(self) -> None:
        # An empty result must still carry its columns AND their source types
        # (APP-6778): a 0x0 frame would break table registration, charts, and the
        # data dictionary, while a String-only frame would mistype numeric columns
        # (and a Null dtype would round-trip as INTEGER through DuckDB).
        operator = make_operator()
        result = make_preview_result(["id", "name"], [])
        result.result_schema[0].data_type = "INTEGER"
        result.result_schema[1].data_type = "VARCHAR"
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            frame = await operator.execute_query("SELECT * FROM users WHERE 1=0")
            assert isinstance(frame, pl.DataFrame)
            assert frame.columns == ["id", "name"]
            assert frame.height == 0
            assert frame.schema == {"id": pl.Int64, "name": pl.String}

    @pytest.mark.asyncio
    async def test_empty_result_without_schema_falls_back_to_string(self) -> None:
        # `result_schema` is optional in the SDK; an empty preview may omit it.
        # Names must still come through via `result.columns`, defaulting to String
        # rather than raising.
        operator = make_operator()
        result = make_preview_result(["id", "name"], [])
        result.result_schema = None
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            frame = await operator.execute_query("SELECT * FROM users WHERE 1=0")
            assert frame.columns == ["id", "name"]
            assert frame.height == 0
            assert frame.schema == {"id": pl.String, "name": pl.String}

    @pytest.mark.asyncio
    async def test_sdk_error_raises_invalid_generated_code(self) -> None:
        from core.code_execution import InvalidGeneratedCode

        operator = make_operator()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.side_effect = RuntimeError("bad sql")
            with pytest.raises(InvalidGeneratedCode):
                await operator.execute_query("SELECT * FROM nonexistent")


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------


def make_analyst_db() -> MagicMock:
    analyst_db = MagicMock()
    analyst_db.register_dataset = AsyncMock()
    return analyst_db


class TestGetData:
    @pytest.mark.asyncio
    async def test_respects_sample_size_cap(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id"], [[1]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            await operator.get_data(
                "users", analyst_db=make_analyst_db(), sample_size=500
            )
            assert mock_jdbc.preview.call_args.kwargs["max_rows"] == 500

    @pytest.mark.asyncio
    async def test_sample_size_capped_at_10000(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id"], [[1]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            await operator.get_data(
                "users", analyst_db=make_analyst_db(), sample_size=99_999
            )
            assert mock_jdbc.preview.call_args.kwargs["max_rows"] == 10_000

    @pytest.mark.asyncio
    async def test_returns_table_names(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id"], [[1]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            names = await operator.get_data(
                "users", "orders", analyst_db=make_analyst_db()
            )
            assert names == ["users", "orders"]

    @pytest.mark.asyncio
    async def test_registers_dataset(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id"], [[1]])
        analyst_db = make_analyst_db()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            await operator.get_data("users", analyst_db=analyst_db)
            analyst_db.register_dataset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_per_table_error_continues(self) -> None:
        operator = make_operator()
        good_result = make_preview_result(["id"], [[1]])
        call_count = 0

        def preview_side_effect(**kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("table not found")
            return good_result

        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.side_effect = preview_side_effect
            names = await operator.get_data(
                "bad_table", "orders", analyst_db=make_analyst_db()
            )
            assert names == ["orders"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "uri,expected_quote",
        [
            ("jdbc:postgresql://localhost:5432/db", '"users"'),
            ("jdbc:mysql://localhost:3306/db", "`users`"),
            ("jdbc:sqlserver://localhost:1433;databaseName=db", "[users]"),
            ("jdbc:snowflake://account.snowflakecomputing.com/", '"users"'),
            ("jdbc:sap://host:443", '"users"'),
            ("jdbc:bigquery://https://www.googleapis.com/bigquery/v2:443", "`users`"),
            ("jdbc:databricks://adb-1234.4.azuredatabricks.net:443", "`users`"),
            ("jdbc:redshift://cluster.us-east-1.redshift.amazonaws.com:5439/mydb", '"users"'),
        ],
    )
    async def test_get_data_quotes_table_per_dialect(
        self, uri: str, expected_quote: str
    ) -> None:
        operator = make_operator(uri)
        result = make_preview_result(["id"], [[1]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            await operator.get_data("users", analyst_db=make_analyst_db())
            sql = mock_jdbc.preview.call_args.kwargs["sql"]
            assert expected_quote in sql
