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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from core.credentials import JDBCCredentials
from core.data_connections.database.database_implementations import JdbcPreviewOperator

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
    async def test_returns_empty_list_on_error(self) -> None:
        operator = make_operator()
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.side_effect = RuntimeError("timeout")
            tables = await operator.get_tables()
            assert tables == []


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
    async def test_returns_list_of_dicts(self) -> None:
        operator = make_operator()
        result = make_preview_result(["id", "name"], [[1, "alice"], [2, "bob"]])
        with patch(_JDBC_PREVIEW) as mock_jdbc:
            mock_jdbc.preview.return_value = result
            rows = await operator.execute_query("SELECT * FROM users")
            assert rows == [{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}]

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
