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

import asyncio
import functools
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, cast

from datarobot.models.jdbc_data_preview import JdbcPreview, JdbcResultSchemaEntry
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from pydantic import ValidationError

from core.analyst_db import AnalystDB, InternalDataSourceType
from core.code_execution import InvalidGeneratedCode
from core.config import Config
from core.credentials import (
    JDBCCredentials,
    NoDatabaseCredentials,
)
from core.data_connections.database.database_interface import (
    _DEFAULT_DB_QUERY_TIMEOUT,
    DatabaseOperator,
    NoDatabaseOperator,
)
from core.data_connections.datarobot.datarobot_dataset_handler import (
    BaseRecipe,
)
from core.prompts import (
    SYSTEM_PROMPT_BIGQUERY,
    SYSTEM_PROMPT_MYSQL,
    SYSTEM_PROMPT_POSTGRES,
    SYSTEM_PROMPT_SAP_DATASPHERE,
    SYSTEM_PROMPT_SNOWFLAKE,
    SYSTEM_PROMPT_SQLSERVER,
)
from core.schema import AnalystDataset

logger = logging.getLogger(__name__)


_JDBC_TABLE_DISCOVERY_SQL: dict[str, str] = {
    "postgresql": """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
    """,
    "mysql": """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_type = 'BASE TABLE'
    """,
    "sqlserver": """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
    """,
    "snowflake": """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = CURRENT_SCHEMA()
          AND TABLE_TYPE IN ('BASE TABLE', 'VIEW')
        ORDER BY TABLE_TYPE, TABLE_NAME
    """,
    "sap": """
        SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA
        UNION ALL
        SELECT VIEW_NAME FROM SYS.VIEWS WHERE SCHEMA_NAME = CURRENT_SCHEMA
        ORDER BY TABLE_NAME
    """,
    "bigquery": """
        SELECT table_name
        FROM INFORMATION_SCHEMA.TABLES
        WHERE table_type IN ('BASE TABLE', 'VIEW')
        ORDER BY table_name
    """,
}

_JDBC_DIALECT_NAMES: dict[str, str] = {
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "sqlserver": "SQL Server",
    "snowflake": "Snowflake",
    "sap": "SAP Datasphere",
    "bigquery": "BigQuery",
}


class JdbcPreviewOperator(DatabaseOperator[JDBCCredentials]):
    def __init__(
        self,
        credentials: JDBCCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials
        self.default_timeout = default_timeout

    @property
    def _dialect_key(self) -> str:
        uri = self._credentials.jdbc_uri
        if uri.startswith("jdbc:postgresql://"):
            return "postgresql"
        elif uri.startswith("jdbc:mysql://"):
            return "mysql"
        elif uri.startswith("jdbc:sqlserver://"):
            return "sqlserver"
        elif uri.startswith("jdbc:snowflake://"):
            return "snowflake"
        elif uri.startswith("jdbc:sap://"):
            return "sap"
        elif uri.startswith("jdbc:bigquery://"):
            return "bigquery"
        raise ValueError(f"Unsupported JDBC URI scheme: {uri.split(':')[1]!r}")

    def _dialect_name(self) -> str:
        return _JDBC_DIALECT_NAMES[self._dialect_key]

    def _quote_identifier(self, name: str) -> str:
        """Quote a table identifier using dialect-appropriate syntax."""
        match self._dialect_key:
            case "postgresql":
                return f'"{name}"'
            case "mysql":
                return f"`{name}`"
            case "sqlserver":
                return f"[{name}]"
            case "snowflake" | "sap":
                return f'"{name}"'
            case "bigquery":
                return f"`{name}`"
            case _:
                raise ValueError(f"Unsupported dialect: {self._dialect_key!r}")

    @staticmethod
    def _require_result_schema(result: Any) -> list[JdbcResultSchemaEntry]:
        result_schema = result.result_schema
        if not result_schema:
            raise ValueError("JDBC preview response did not include a result schema")
        return cast(list[JdbcResultSchemaEntry], result_schema)

    def validate_connection(self) -> None:
        """Run SELECT 1 to verify connectivity at startup."""
        try:
            JdbcPreview.preview(
                jdbc_url=self._credentials.jdbc_uri,
                sql="SELECT 1",
                max_rows=1,
                parameters=self._credentials.jdbc_connection_parameters or {},
            )
        except Exception as e:
            raise ValueError(f"JDBC connection validation failed: {e}") from e

    @asynccontextmanager
    async def create_connection(self) -> AsyncGenerator[None, None]:
        # JdbcPreview is stateless — no persistent connection object needed
        yield None

    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: JdbcPreview.preview(
                    jdbc_url=self._credentials.jdbc_uri,
                    sql=query,
                    max_rows=10_000,  # API ceiling — ad-hoc SQL gets the full budget
                    parameters=self._credentials.jdbc_connection_parameters or {},
                ),
            )
            result_schema = self._require_result_schema(result)
            columns = [entry.name for entry in result_schema]
            return [dict(zip(columns, row)) for row in result.records]
        except Exception as e:
            raise InvalidGeneratedCode(
                f"JDBC query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    async def get_tables(self, timeout: int | None = None) -> list[str]:
        sql = _JDBC_TABLE_DISCOVERY_SQL[self._dialect_key]
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: JdbcPreview.preview(
                    jdbc_url=self._credentials.jdbc_uri,
                    sql=sql,
                    max_rows=10_000,
                    parameters=self._credentials.jdbc_connection_parameters or {},
                ),
            )
            result_schema = self._require_result_schema(result)
            col = result_schema[0].name
            tables = [
                row[col] if isinstance(row, dict) else row[0] for row in result.records
            ]
            logger.info(f"JDBC ({self._dialect_name()}): found {len(tables)} tables")
            return tables
        except Exception:
            logger.error("JDBC: failed to fetch tables", exc_info=True)
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        loop = asyncio.get_running_loop()
        names = []
        for table in table_names:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda t=table: JdbcPreview.preview(  # type: ignore[misc]
                        jdbc_url=self._credentials.jdbc_uri,
                        sql=f"SELECT * FROM {self._quote_identifier(t)}",
                        max_rows=min(sample_size, 10_000),
                        parameters=self._credentials.jdbc_connection_parameters or {},
                    ),
                )
                result_schema = self._require_result_schema(result)
                schema = [
                    {"name": entry.name, "dataType": entry.data_type}
                    for entry in result_schema
                ]
                column_types = {entry.name: entry.data_type for entry in result_schema}
                df = BaseRecipe.convert_preview_to_dataframe(schema, result.records)
                logger.info(
                    f"JDBC: loaded table {table}: {len(df.df)} rows, {len(df.df.columns)} columns"
                )
                dataset = AnalystDataset(name=table, data=df)
                await analyst_db.register_dataset(
                    dataset,
                    InternalDataSourceType.DATABASE,
                    original_column_types=column_types,
                    clobber=True,
                )
                names.append(table)
            except Exception:
                logger.error(f"JDBC: error loading table {table}", exc_info=True)
                continue
        return names

    def query_friendly_name(self, dataset_name: str) -> str:
        return self._quote_identifier(dataset_name)

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        prompt = {
            "postgresql": SYSTEM_PROMPT_POSTGRES,
            "mysql": SYSTEM_PROMPT_MYSQL,
            "sqlserver": SYSTEM_PROMPT_SQLSERVER,
            "snowflake": SYSTEM_PROMPT_SNOWFLAKE,
            "sap": SYSTEM_PROMPT_SAP_DATASPHERE,
            "bigquery": SYSTEM_PROMPT_BIGQUERY,
        }[self._dialect_key]
        return ChatCompletionSystemMessageParam(role="system", content=prompt)


def get_database_operator(config: Config) -> DatabaseOperator[Any]:
    logger.info("Loading database %s", config.database_connection_type)
    if config.database_connection_type in (
        "snowflake",
        "sap",
        "bigquery",
        "datarobot_jdbc",
    ):
        try:
            jdbc_credentials = JDBCCredentials()
        except ValidationError as exc:
            raise ValueError(
                f"DATABASE_CONNECTION_TYPE is '{config.database_connection_type}' but JDBC_URI "
                "is missing or invalid. Set JDBC_URI to a valid JDBC connection string "
                "(e.g. jdbc:snowflake://..., jdbc:sap://..., or jdbc:bigquery://...)."
            ) from exc
        return cast(
            DatabaseOperator[Any],
            JdbcPreviewOperator(
                credentials=jdbc_credentials,
                default_timeout=_DEFAULT_DB_QUERY_TIMEOUT,
            ),
        )
    else:
        return NoDatabaseOperator(NoDatabaseCredentials())


def get_external_database() -> DatabaseOperator[Any]:
    try:
        return get_database_operator(Config())
    except ValueError:
        logger.error(
            "Database misconfigured — JDBC_URI missing or invalid. Falling back to no database.",
            exc_info=True,
        )
        return NoDatabaseOperator(NoDatabaseCredentials())
