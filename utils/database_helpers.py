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

import functools
import json
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Generic, TypeVar, cast

import pandas as pd
import polars as pl
import snowflake.connector
from google.cloud import bigquery
from hdbcli import dbapi
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from pydantic import ValidationError

from utils.analyst_db import AnalystDB, DataSourceType
from utils.code_execution import InvalidGeneratedCode
from utils.credentials import (
    GoogleCredentials,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
)
from utils.logging_helper import get_logger
from utils.prompts import (
    SYSTEM_PROMPT_BIGQUERY,
    SYSTEM_PROMPT_SAP_DATASPHERE,
    SYSTEM_PROMPT_SNOWFLAKE,
)
from utils.schema import (
    AnalystDataset,
    AppInfra,
)

logger = get_logger("DatabaseHelper")

T = TypeVar("T")
_DEFAULT_DB_QUERY_TIMEOUT = 300


@dataclass
class SnowflakeCredentialArgs:
    credentials: SnowflakeCredentials


@dataclass
class BigQueryCredentialArgs:
    credentials: GoogleCredentials


@dataclass
class SAPDatasphereCredentialArgs:
    credentials: SAPDatasphereCredentials


@dataclass
class NoDatabaseCredentialArgs:
    credentials: NoDatabaseCredentials


class DatabaseOperator(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, credentials: T, default_timeout: int): ...

    @abstractmethod
    @contextmanager
    def create_connection(self) -> Any: ...

    @abstractmethod
    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...

    @abstractmethod
    def get_tables(self, timeout: int | None = None) -> list[str]:
        return []

    @abstractmethod
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        return []

    @abstractmethod
    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content="")


class NoDatabaseOperator(DatabaseOperator[NoDatabaseCredentialArgs]):
    def __init__(
        self,
        credentials: NoDatabaseCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials

    @contextmanager
    def create_connection(self) -> Generator[None]:
        yield None

    def execute_query(
        self,
        query: str,
        timeout: int | None = 300,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        return []

    def get_tables(self, timeout: int | None = 300) -> list[str]:
        return []

    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = 300,
    ) -> list[str]:
        return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content="")


class SnowflakeOperator(DatabaseOperator[SnowflakeCredentialArgs]):
    def __init__(
        self,
        credentials: SnowflakeCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[snowflake.connector.SnowflakeConnection]:
        """Create a connection to Snowflake using environment variables"""
        if not self._credentials.is_configured():
            raise ValueError("Snowflake credentials not properly configured")

        connect_params: dict[str, Any] = {
            "user": self._credentials.user,
            "account": self._credentials.account,
            "warehouse": self._credentials.warehouse,
            "database": self._credentials.database,
            "schema": self._credentials.db_schema,
            "role": self._credentials.role,
        }

        # Try key file authentication first if configured
        project_root = Path(__file__).resolve().parent.parent
        if private_key := self._credentials.get_private_key(project_root=project_root):
            connect_params["private_key"] = private_key
        elif self._credentials.password:
            connect_params["password"] = self._credentials.password
        else:
            raise ValueError(
                "Neither private key nor password authentication configured"
            )

        connection = snowflake.connector.connect(**connect_params)
        yield connection
        connection.close()

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a Snowflake query with timeout and metadata capture

        Args:
            conn: Snowflake connection
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Tuple of (results, metadata)
        """
        timeout = timeout if timeout is not None else self.default_timeout
        conn: snowflake.connector.SnowflakeConnection
        try:
            with self.create_connection() as conn:
                with conn.cursor(snowflake.connector.DictCursor) as cursor:
                    cursor = conn.cursor(snowflake.connector.DictCursor)
                    # Set query timeout at cursor level
                    cursor.execute(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                    )

                    try:
                        # Execute query
                        cursor.execute(query)

                        # Get results
                        results = cursor.fetchall()

                        return results

                    except snowflake.connector.errors.ProgrammingError as e:
                        # Handle Snowflake-specific errors
                        raise InvalidGeneratedCode(
                            f"Snowflake error: {str(e.msg)}",
                            code=query,
                            exception=None,
                            traceback_str="",
                        )

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from Snowflake schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: snowflake.connector.SnowflakeConnection
        try:
            with self.create_connection() as conn:
                with conn.cursor() as cursor:
                    # Log current session info
                    logger.info("Checking current session settings...")
                    cursor.execute(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                    )

                    cursor.execute(
                        "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE(), CURRENT_WAREHOUSE()",
                    )
                    current_settings = cursor.fetchone()
                    logger.info(
                        f"Current settings - Database: {current_settings[0]}, Schema: {current_settings[1]}, Role: {current_settings[2]}, Warehouse: {current_settings[3]}"  # type: ignore[index]
                    )

                    # Check if schema exists
                    cursor.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {self._credentials.database}.INFORMATION_SCHEMA.SCHEMATA
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                    """,
                    )
                    schema_exists = cursor.fetchone()[0]  # type: ignore[index]
                    logger.info(f"Schema exists check: {schema_exists > 0}")

                    # Get all objects (tables and views)
                    cursor.execute(
                        f"""
                        SELECT table_name, table_type
                        FROM {self._credentials.database}.information_schema.tables
                        WHERE table_schema = '{self._credentials.db_schema}'
                        AND table_type IN ('BASE TABLE', 'VIEW')
                        ORDER BY table_type, table_name
                    """
                    )
                    results = cursor.fetchall()
                    tables = [row[0] for row in results]

                    # Log detailed results
                    logger.info(f"Total objects found: {len(results)}")
                    for table_name, table_type in results:
                        logger.info(f"Found {table_type}: {table_name}")

                    # Check schema privileges
                    cursor.execute(
                        f"""
                        SHOW GRANTS ON SCHEMA {self._credentials.database}.{self._credentials.db_schema}
                    """
                    )
                    privileges = cursor.fetchall()
                    logger.info("Schema privileges:")
                    for priv in privileges:
                        logger.info(f"Privilege: {priv}")

                    return tables

        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from Snowflake as pandas DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table

        Returns:
        - Dictionary of table names to list of records
        """

        timeout = timeout if timeout is not None else self.default_timeout

        conn: snowflake.connector.SnowflakeConnection

        dataframes = []
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()

                for table in table_names:
                    try:
                        qualified_table = f'{self._credentials.database}.{self._credentials.db_schema}."{table}"'
                        logger.info(f"Fetching data from table: {qualified_table}")
                        cursor.execute(
                            f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {timeout}"
                        )
                        cursor.execute(
                            f"""
                            SELECT * FROM {qualified_table}
                            SAMPLE ({sample_size} ROWS)
                        """
                        )

                        columns = [desc[0] for desc in cursor.description]
                        data = cursor.fetchall()
                        pandas_df = pd.DataFrame(data=data, columns=columns, dtype=str)
                        # If you want to use Polars later, do it after validation/registration
                        dataframes.append(AnalystDataset(name=table, data=pandas_df))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue
                names = []
                for dataframe in dataframes:
                    reg_result = await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    if not reg_result["success"]:
                        logger.error(
                            f"Failed to register dataset {dataframe.name}: {reg_result['msg']}"
                        )
                        continue
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching Snowflake data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SNOWFLAKE.format(
                warehouse=self._credentials.warehouse,
                database=self._credentials.database,
                schema=self._credentials.db_schema,
            ),
        )


class BigQueryOperator(DatabaseOperator[BigQueryCredentialArgs]):
    def __init__(
        self,
        credentials: GoogleCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials
        self._credentials.db_schema = self._credentials.db_schema
        self._database = credentials.service_account_key["project_id"]
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[bigquery.Client]:
        from google.oauth2 import service_account

        google_credentials = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
            GoogleCredentials().service_account_key,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=google_credentials,
        )

        yield client

        client.close()  # type: ignore[no-untyped-call]

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        conn: bigquery.Client
        timeout = timeout if timeout is not None else self.default_timeout
        try:
            with self.create_connection() as conn:
                results = conn.query(query, timeout=timeout)

                sql_result: pd.DataFrame = results.to_dataframe()

                sql_result_as_dicts = cast(
                    list[dict[str, Any]], sql_result.to_dict(orient="records")
                )
                return sql_result_as_dicts

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from BigQuery schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: bigquery.Client

        try:
            with self.create_connection() as conn:
                tables = [
                    i.table_id
                    for i in conn.list_tables(
                        str(self._credentials.db_schema), timeout=timeout
                    )
                ]

                # Log detailed results
                logger.info(f"Total objects found: {len(tables)}")
                logger.info(f"Found tables: {', '.join(tables)}")

                return tables

        except Exception as e:
            logger.error(f"Failed to fetch tables: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        timeout = timeout if timeout is not None else self.default_timeout

        dataframes = []

        conn: bigquery.Client

        try:
            with self.create_connection() as conn:
                for table in table_names:
                    try:
                        qualified_table = (
                            f"{self._database}.{self._credentials.db_schema}.{table}"
                        )
                        logger.info(f"Fetching data from table: {qualified_table}")

                        df: pd.DataFrame = conn.query(
                            f"""
                            SELECT * FROM `{qualified_table}`
                            LIMIT {sample_size}
                        """,
                            timeout=timeout,
                        ).to_dataframe()

                        # Convert date/datetime columns to string format
                        for col in df.columns:
                            if pd.api.types.is_datetime64_any_dtype(
                                df[col]
                            ) or isinstance(df[col].dtype, pd.DatetimeTZDtype):
                                df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
                            elif df[col].dtype == "object":
                                try:
                                    pd.to_datetime(df[col], errors="raise")
                                    df[col] = pd.to_datetime(df[col]).dt.strftime(
                                        "%Y-%m-%d"
                                    )
                                except (ValueError, TypeError):
                                    continue
                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )
                        data = cast(list[dict[str, Any]], df.to_dict("records"))
                        dataframes.append(AnalystDataset(name=table, data=data))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue

                names = []
                for dataframe in dataframes:
                    reg_result = await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    if not reg_result["success"]:
                        logger.error(
                            f"Failed to register dataset {dataframe.name}: {reg_result['msg']}"
                        )
                        continue
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_BIGQUERY.format(
                project=self._database,
                dataset=self._credentials.db_schema,
            ),
        )


class SAPDatasphereOperator(DatabaseOperator[SAPDatasphereCredentialArgs]):
    def __init__(
        self,
        credentials: SAPDatasphereCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        if not credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")
        self._credentials = credentials
        self.default_timeout = default_timeout

    @contextmanager
    def create_connection(self) -> Generator[dbapi.Connection]:
        """Create a connection to SAP Data Sphere"""
        if not self._credentials.is_configured():
            raise ValueError("SAP Data Sphere credentials not properly configured")

        connect_params: dict[str, Any] = {
            "address": self._credentials.host,
            "port": self._credentials.port,
            "user": self._credentials.user,
            "password": self._credentials.password,
        }

        try:
            # Connect to SAP Data Sphere
            connection = dbapi.connect(**connect_params)
            yield connection
        except Exception:
            raise
        finally:
            connection.close()

    def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        """Execute a SAP Data Sphere query with timeout

        Args:
            query: SQL query to execute
            timeout: Query timeout in seconds

        Returns:
            Query results
        """
        timeout = timeout if timeout is not None else self.default_timeout
        conn: dbapi.Connection
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Execute query
                    cursor.execute(query)

                    # Get results
                    results = cursor.fetchall()

                    return [
                        dict(zip(row.column_names, row.column_values))
                        for row in results
                    ]

                except Exception as e:
                    # Handle SAP Data Sphere specific errors
                    raise InvalidGeneratedCode(
                        f"SAP Data Sphere error: {str(e)}",
                        code=query,
                        exception=None,
                        traceback_str="",
                    )
                finally:
                    cursor.close()

        except Exception as e:
            raise InvalidGeneratedCode(
                f"Query execution failed: {str(e)}",
                code=query,
                exception=e,
                traceback_str=traceback.format_exc(),
            )

    def get_tables(self, timeout: int | None = None) -> list[str]:
        """Fetch list of tables from SAP Data Sphere schema"""
        timeout = timeout if timeout is not None else self.default_timeout

        conn: dbapi.Connection
        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()
                try:
                    # Get all tables and views in the schema
                    cursor.execute(
                        f"""
                        SELECT TABLE_NAME 
                        FROM SYS.TABLES 
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY TABLE_NAME
                        """
                    )
                    tables = [row[0] for row in cursor.fetchall()]

                    # Get all views
                    cursor.execute(
                        f"""
                        SELECT VIEW_NAME 
                        FROM SYS.VIEWS 
                        WHERE SCHEMA_NAME = '{self._credentials.db_schema}'
                        ORDER BY VIEW_NAME
                        """
                    )
                    views = [row[0] for row in cursor.fetchall()]

                    all_objects = tables + views

                    # Log detailed results
                    logger.info(
                        f"Total objects found in schema {self._credentials.db_schema}: {len(all_objects)}"
                    )
                    logger.info(f"Tables: {len(tables)}, Views: {len(views)}")

                    return all_objects

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to fetch tables from SAP Data Sphere: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    @functools.lru_cache(maxsize=8)
    async def get_data(
        self,
        *table_names: str,
        analyst_db: AnalystDB,
        sample_size: int = 5000,
        timeout: int | None = None,
    ) -> list[str]:
        """Load selected tables from SAP Data Sphere as DataFrames

        Args:
        - table_names: List of table names to fetch
        - sample_size: Number of rows to sample from each table
        - timeout: Query timeout in seconds

        Returns:
        - List of registered dataset names
        """
        timeout = timeout if timeout is not None else self.default_timeout
        dataframes = []

        try:
            with self.create_connection() as conn:
                cursor = conn.cursor()

                for table in table_names:
                    try:
                        qualified_table = f'"{self._credentials.db_schema}"."{table}"'
                        logger.info(f"Fetching data from table: {qualified_table}")

                        # Execute query to get data with limit
                        cursor.execute(
                            f"""
                            SELECT * FROM {qualified_table}
                            LIMIT {sample_size}
                            """
                        )

                        # Get column names
                        columns = [desc[0] for desc in cursor.description]
                        data = cursor.fetchall()

                        # Convert to pandas DataFrame
                        pandas_df = pd.DataFrame(data=data, columns=columns, dtype=str)

                        # Convert to polars DataFrame
                        df = pl.DataFrame(
                            data=pandas_df, schema={col: pl.String for col in columns}
                        )

                        logger.info(
                            f"Successfully loaded table {table}: {len(df)} rows, {len(df.columns)} columns"
                        )
                        dataframes.append(AnalystDataset(name=table, data=pandas_df))

                    except Exception as e:
                        logger.error(f"Error loading table {table}: {str(e)}")
                        logger.error(f"Error type: {type(e)}")
                        logger.error(f"Error details: {str(e)}")
                        continue

                # Register datasets
                names = []
                for dataframe in dataframes:
                    reg_result = await analyst_db.register_dataset(
                        dataframe, DataSourceType.DATABASE
                    )
                    if not reg_result["success"]:
                        logger.error(
                            f"Failed to register dataset {dataframe.name}: {reg_result['msg']}"
                        )
                        continue
                    names.append(dataframe.name)
                return names

        except Exception as e:
            logger.error(f"Error fetching SAP Data Sphere data: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {str(e)}")
            return []

    def get_system_prompt(self) -> ChatCompletionSystemMessageParam:
        return ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT_SAP_DATASPHERE.format(
                schema=self._credentials.db_schema,
            ),
        )


def get_database_operator(app_infra: AppInfra) -> DatabaseOperator[Any]:
    if app_infra.database == "bigquery":
        credentials: (
            GoogleCredentials
            | SnowflakeCredentials
            | SAPDatasphereCredentials
            | NoDatabaseCredentials
        )
        try:
            credentials = GoogleCredentials()
            if credentials.service_account_key and credentials.db_schema:
                return BigQueryOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "BigQuery credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "snowflake":
        try:
            credentials = SnowflakeCredentials()
            if credentials.is_configured():
                return SnowflakeOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "Snowflake credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    elif app_infra.database == "sap":
        try:
            credentials = SAPDatasphereCredentials()
            if credentials.is_configured():
                return SAPDatasphereOperator(credentials)
        except (ValidationError, ValueError):
            logger.warning(
                "SAP credentials not properly configured, falling back to no database"
            )
        return NoDatabaseOperator(NoDatabaseCredentials())
    else:
        return NoDatabaseOperator(NoDatabaseCredentials())


def load_app_infra() -> AppInfra:
    try:
        with open("app_infra.json", "r") as infra_selection:
            app_infra = AppInfra(**json.load(infra_selection))
        return app_infra
    except (FileNotFoundError, ValidationError):
        try:
            with open("frontend/app_infra.json", "r") as infra_selection:
                app_infra = AppInfra(**json.load(infra_selection))
            return app_infra
        except (FileNotFoundError, ValidationError):
            try:
                with open(
                    "frontend_react/deploy/app_infra.json", "r"
                ) as infra_selection:
                    app_infra = AppInfra(**json.load(infra_selection))
                return app_infra
            except (FileNotFoundError, ValidationError) as e:
                raise ValueError(
                    "Failed to read app_infra.json.\n"
                    "If running locally, verify you have selected the correct "
                    "stack and that it is active using `pulumi stack output`.\n"
                    f"Ensure file is created by running `pulumi up`: {str(e)}"
                ) from e


def get_external_database() -> DatabaseOperator[Any]:
    return get_database_operator(load_app_infra())
