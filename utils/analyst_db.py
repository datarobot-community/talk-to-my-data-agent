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
import asyncio
import hashlib
import json
import uuid
from abc import ABC
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, cast

import duckdb
import polars as pl

from utils.logging_helper import get_logger
from utils.persistent_storage import PersistentStorage
from utils.schema import (
    AnalystChatMessage,
    AnalystDataset,
    ChatJSONEncoder,
    CleansedColumnReport,
    CleansedDataset,
    DataDictionary,
)

logger = get_logger("ApplicationDB")

# increment this number if the database schema has changed to prevent conflicts with existing deployments
# this will force reinitialisation - all tables will be dropped
ANALYST_DATABASE_VERSION = 5


class DatasetType(Enum):
    STANDARD = "standard"
    CLEANSED = "cleansed"
    DICTIONARY = "dictionary"


class DataSourceType(Enum):
    FILE = "file"
    DATABASE = "database"
    REGISTRY = "catalog"
    GENERATED = "generated"


@dataclass
class DatasetMetadata:
    name: str
    dataset_type: DatasetType
    original_name: (
        str  # For cleansed/dictionary datasets, links to their original dataset
    )
    created_at: datetime
    columns: list[str]
    row_count: int
    data_source: DataSourceType
    file_size: int = 0  # Size of the file in bytes


class BaseDuckDBHandler(ABC):
    """Abstract base class defining the common async DuckDB interface."""

    def __init__(
        self,
        *,
        user_id: str | None = None,
        db_path: Path | None = None,
        name: str | None = None,
        db_version: (
            int | None
        ) = 1,  # should be updated after updating db tables structure
        use_persistent_storage: bool = False,
    ) -> None:
        """Initialize database path and create tables."""
        self.db_version = db_version
        self.user_id = user_id
        self.db_path = self.get_db_path(user_id=user_id, db_path=db_path, name=name)
        self._storage = PersistentStorage(user_id) if use_persistent_storage else None

    async def _create_db_version_table(
        self,
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        # create db_versuion table
        await self.execute_query(
            conn,
            """
            CREATE TABLE IF NOT EXISTS db_version (
                version INTEGER PRIMARY KEY
            )
            """,
        )
        # insert new version
        await self.execute_query(
            conn, "INSERT OR IGNORE INTO db_version VALUES (?)", [self.db_version]
        )

    async def _initialize_database(self) -> None:
        """Initialize database tables and extensions."""
        if self._storage and not self.db_path.exists():
            self._storage.fetch_from_storage(
                self.db_path.name, str(self.db_path.absolute())
            )
        async with self._get_connection() as conn:
            # check if db_version table exist
            db_version_result = await self.execute_query(
                conn,
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = 'db_version');",
            )
            old_db_version_table = await asyncio.get_running_loop().run_in_executor(
                None, db_version_result.fetchone
            )
            if old_db_version_table and old_db_version_table[0]:
                # get db version
                db_version_result = await self.execute_query(
                    conn, "SELECT version FROM db_version"
                )
                db_version_row = await asyncio.get_running_loop().run_in_executor(
                    None, db_version_result.fetchone
                )
                if db_version_row:
                    db_version = db_version_row[0]
                    if db_version != self.db_version:
                        # drop all tables
                        tables_result = await self.execute_query(
                            conn,
                            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main';",
                        )
                        table_rows = await asyncio.get_running_loop().run_in_executor(
                            None, tables_result.fetchall
                        )
                        for (table_name,) in table_rows:
                            await self.execute_query(
                                conn, f'DROP TABLE IF EXISTS "{table_name}";'
                            )

                        await self._create_db_version_table(conn)
            else:
                await self._create_db_version_table(conn)

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[duckdb.DuckDBPyConnection, Any]:
        """Async context manager for database connections."""
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, duckdb.connect, self.db_path)
        async with self._save_to_storage():
            try:
                yield conn
            finally:
                await loop.run_in_executor(None, conn.close)

    @asynccontextmanager
    async def _save_to_storage(self) -> AsyncGenerator[None, None]:
        # if we have storage, check DB hashsum before and after connection
        # and save if file changed
        if self._storage:
            before = hashlib.sha256()
            with self.db_path.open("rb") as db_file:
                while chunk := db_file.read(8192):
                    before.update(chunk)
            yield
            after = hashlib.sha256()
            with self.db_path.open("rb") as db_file:
                while chunk := db_file.read(8192):
                    after.update(chunk)
            if before.digest() != after.digest():
                self._storage.save_to_storage(
                    self.db_path.name, str(self.db_path.absolute())
                )
        else:
            yield

    async def execute_query(
        self,
        conn: duckdb.DuckDBPyConnection,
        query: str,
        params: list[Any] | None = None,
    ) -> duckdb.DuckDBPyConnection:
        """Execute a query asynchronously."""
        loop = asyncio.get_running_loop()
        if params:
            return await loop.run_in_executor(None, lambda: conn.execute(query, params))
        return await loop.run_in_executor(None, lambda: conn.execute(query))

    @staticmethod
    def get_db_path(
        db_path: Path | None = None, user_id: str | None = None, name: str | None = None
    ) -> Path:
        """Return the database path for a given user."""
        path = Path(db_path or ".")
        name = f"{name or 'app'}_db{'_' + user_id if user_id else ''}.db"
        return path / name


class DatasetHandler(BaseDuckDBHandler):
    async def _initialize_database(self) -> None:
        """Initialize database tables and metadata tracking."""
        await super()._initialize_database()
        async with self._get_connection() as conn:
            # Create metadata table
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    table_name VARCHAR PRIMARY KEY,
                    dataset_type VARCHAR,
                    original_name VARCHAR,
                    created_at TIMESTAMP,
                    columns JSON,
                    row_count INTEGER,
                    data_source VARCHAR,
                    file_size INTEGER DEFAULT 0
                )
                """,
            )
            # Create cleansing reports table
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS cleansing_reports (
                    dataset_name VARCHAR,
                    report JSON,
                    PRIMARY KEY (dataset_name)
                )
                """,
            )

    async def register_dataframe(
        self,
        df: pl.DataFrame,
        name: str,
        dataset_type: DatasetType,
        data_source: DataSourceType,
        original_name: str | None = None,
        file_size: int = 0,
    ) -> None:
        """
        Register a Polars DataFrame with explicit dataset type tracking.

        Args:
            df: The dataframe to register
            name: Name for the table
            dataset_type: Type of dataset (STANDARD, CLEANSED, or DICTIONARY)
            original_name: For CLEANSED/DICTIONARY types, the name of the original dataset
            data_source: The source of the data (DataSourceType.FILE, DataSourceType.DATABASE, or DataSourceType.REGISTRY)
            file_size: Size of the source file in bytes (for FILE data sources)
        """
        logger.info(f"Registering dataframe {name} as {dataset_type.value}")

        if await self.table_exists(name):
            raise ValueError(f"Table '{name}' already exists in the database")

        # For cleansed/dictionary datasets, verify original exists
        if dataset_type in (DatasetType.CLEANSED, DatasetType.DICTIONARY):
            if not original_name:
                raise ValueError(
                    f"original_name required for {dataset_type.value} datasets"
                )
            if not await self.table_exists(original_name):
                raise ValueError(f"Original dataset '{original_name}' not found")

        async with self._get_connection() as conn:
            # Create the table
            arrow_table = df.to_arrow()

            def create_table() -> None:
                conn.register("temp_view", arrow_table)
                conn.execute(f"CREATE TABLE '{name}' AS SELECT * FROM temp_view")
                conn.unregister("temp_view")

            await asyncio.get_running_loop().run_in_executor(None, create_table)

            # Store metadata
            metadata = DatasetMetadata(
                name=name,
                dataset_type=dataset_type,
                original_name=original_name or name,
                created_at=datetime.now(timezone.utc),
                columns=list(df.columns),
                row_count=len(df),
                data_source=data_source,
                file_size=file_size,
            )

            await self.execute_query(
                conn,
                """
                INSERT INTO dataset_metadata
                (table_name, dataset_type, original_name, created_at, columns, row_count, data_source, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    metadata.name,
                    metadata.dataset_type.value,
                    metadata.original_name,
                    metadata.created_at,
                    json.dumps(metadata.columns),
                    metadata.row_count,
                    metadata.data_source.value,
                    metadata.file_size,
                ],
            )

    async def list_datasets(
        self,
        dataset_type: DatasetType | None = None,
        data_source: DataSourceType | None = None,
    ) -> list[DatasetMetadata]:
        """
        List all datasets, optionally filtered by dataset type and/or data source.

        Args:
            dataset_type: Optional filter by dataset type (STANDARD, CLEANSED, DICTIONARY)
            data_source: Optional filter by data source (FILE, DATABASE, REGISTRY)

        Returns:
            List of DatasetMetadata for matching datasets
        """
        async with self._get_connection() as conn:
            query = """
                SELECT
                    table_name, dataset_type, original_name,
                    created_at, columns, row_count, data_source, file_size
                FROM dataset_metadata
            """
            params = []
            where_clauses = []

            if dataset_type:
                where_clauses.append("dataset_type = ?")
                params.append(dataset_type.value)

            if data_source:
                where_clauses.append("data_source = ?")
                params.append(data_source.value)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            result = await self.execute_query(conn, query, params)
            rows = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchall
            )

            return [
                DatasetMetadata(
                    name=row[0],
                    dataset_type=DatasetType(row[1]),
                    original_name=row[2],
                    created_at=row[3],
                    columns=json.loads(row[4]),
                    row_count=row[5],
                    data_source=DataSourceType(row[6]),
                    file_size=row[7],
                )
                for row in rows
            ]

    async def get_dataset_type(self, name: str) -> DatasetType:
        """Get the type of a dataset."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                "SELECT dataset_type FROM dataset_metadata WHERE table_name = ?",
                [name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            if not row:
                raise ValueError(f"Dataset '{name}' not found")
            return DatasetType(row[0])

    async def table_exists(self, name: str) -> bool:
        """Check if a table exists in the database."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT COUNT(*)
                FROM dataset_metadata
                WHERE table_name = ?
                """,
                [name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            return bool(row and row[0])

    async def get_related_datasets(self, name: str) -> dict[str, list[str]]:
        """
        Get all related datasets (cleansed versions and data dictionaries)
        for a given standard dataset.
        """
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT table_name, dataset_type
                FROM dataset_metadata
                WHERE original_name = ?
                """,
                [name],
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchall
            )

            related: dict[str, list[str]] = {"cleansed": [], "dictionary": []}

            for table_name, dtype in rows:
                if dtype == DatasetType.CLEANSED.value:
                    related["cleansed"].append(table_name)
                elif dtype == DatasetType.DICTIONARY.value:
                    related["dictionary"].append(table_name)

            return related

    async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
        """Get metadata for a dataset by name"""
        try:
            if not await self.table_exists(name):
                raise ValueError(f"Dataset '{name}' not found")

            async with self._get_connection() as conn:
                result = await self.execute_query(
                    conn,
                    """
                    SELECT
                        table_name, dataset_type, original_name,
                        created_at, columns, row_count, data_source, file_size
                    FROM dataset_metadata
                    WHERE table_name = ?
                    """,
                    [name],
                )
                row = await asyncio.get_running_loop().run_in_executor(
                    None, result.fetchone
                )

                if not row:
                    raise ValueError(f"Metadata for dataset '{name}' not found")

                # Format the metadata as a dictionary
                metadata = DatasetMetadata(
                    name=row[0],
                    dataset_type=row[1],
                    original_name=row[2],
                    created_at=row[3].isoformat() if row[3] else datetime.min,
                    columns=json.loads(row[4]),
                    row_count=row[5],
                    data_source=DataSourceType(row[6]),
                    file_size=row[7],
                )

                return metadata

        except Exception as e:
            # Catch all other exceptions and provide a clear error message
            logger.error(f"Error getting metadata for dataset {name}: {e}")
            raise ValueError(
                f"Failed to retrieve metadata for dataset '{name}': {str(e)}"
            )

    async def get_dataframe(
        self,
        name: str,
        expected_type: DatasetType | None = None,
        max_rows: int | None = None,
    ) -> pl.DataFrame:
        """
        Retrieve a registered table as a Polars DataFrame.

        Args:
            name: Name of the dataset to retrieve
            expected_type: Optional type validation - will raise error if dataset is not of expected type

        Returns:
            Polars DataFrame containing the dataset

        Raises:
            ValueError: If dataset doesn't exist or is of wrong type
        """
        logger.info(f"Retrieving dataframe {name}")

        # First verify the dataset exists and check its type
        if not await self.table_exists(name):
            raise ValueError(f"Dataset '{name}' not found")

        if expected_type:
            actual_type = await self.get_dataset_type(name)
            if actual_type != expected_type:
                raise ValueError(
                    f"Dataset '{name}' is of type {actual_type.value}, "
                    f"expected {expected_type.value}"
                )

        # Retrieve the data
        async with self._get_connection() as conn:
            try:
                result = await self.execute_query(
                    conn,
                    f'SELECT * FROM "{name}"'
                    + (f" LIMIT {max_rows}" if max_rows is not None else ""),
                )
                arrow_table = await asyncio.get_running_loop().run_in_executor(
                    None, result.arrow
                )
                return cast(pl.DataFrame, pl.from_arrow(arrow_table))
            except duckdb.CatalogException as e:
                raise ValueError(f"Error retrieving dataset '{name}': {str(e)}") from e

    async def store_cleansing_report(
        self, dataset_name: str, reports: list[CleansedColumnReport]
    ) -> None:
        """Store cleansing reports in the metadata table asynchronously."""
        async with self._get_connection() as conn:
            report_json = json.dumps([report.model_dump() for report in reports])
            await self.execute_query(
                conn,
                """
                INSERT OR REPLACE INTO cleansing_reports (dataset_name, report)
                VALUES (?, ?)
                """,
                [dataset_name, report_json],
            )

    async def get_cleansing_report(
        self, dataset_name: str
    ) -> Optional[list[CleansedColumnReport]]:
        """Retrieve cleansing reports asynchronously."""
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                "SELECT report FROM cleansing_reports WHERE dataset_name = ?",
                [dataset_name],
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, result.fetchone
            )
            if row:
                reports_data = json.loads(row[0])
                return [CleansedColumnReport(**report) for report in reports_data]
            return []

    async def delete_dataset(self, name: str) -> None:
        """
        Delete a specific dataset and its metadata.
        Will not delete related datasets (cleansed/dictionary versions).

        Args:
            name: Name of the dataset to delete

        Raises:
            ValueError: If dataset doesn't exist
        """
        if not await self.table_exists(name):
            raise ValueError(f"Dataset '{name}' not found")

        async with self._get_connection() as conn:
            # Delete the actual table
            await self.execute_query(conn, f'DROP TABLE IF EXISTS "{name}"')

            # Delete metadata
            await self.execute_query(
                conn, "DELETE FROM dataset_metadata WHERE table_name = ?", [name]
            )

            # Delete any cleansing reports
            await self.execute_query(
                conn, "DELETE FROM cleansing_reports WHERE dataset_name = ?", [name]
            )

        logger.info(f"Deleted dataset {name}")

    async def delete_related_datasets(self, name: str) -> None:
        """
        Delete all related datasets (cleansed and dictionary) for a given standard dataset.
        Does not delete the original dataset itself.
        """
        related = await self.get_related_datasets(name)

        for dataset_list in related.values():
            for dataset_name in dataset_list:
                await self.delete_dataset(dataset_name)

        logger.info(f"Deleted all related datasets for {name}")

    async def delete_dataset_with_related(self, name: str) -> None:
        """
        Delete a dataset and all its related datasets (cleansed and dictionary versions).
        """
        # Delete related datasets first
        await self.delete_related_datasets(name)
        # Then delete the main dataset
        await self.delete_dataset(name)

    async def delete_all_datasets(
        self, dataset_type: DatasetType | None = None
    ) -> None:
        """
        Delete all datasets of a specific type, or all datasets if type is None.
        """
        datasets = await self.list_datasets(dataset_type)

        for dataset in datasets:
            await self.delete_dataset(dataset.name)

        type_str = f" of type {dataset_type.value}" if dataset_type else ""
        logger.info(f"Deleted all datasets{type_str}")

    async def delete_empty_datasets(self) -> None:
        """
        Delete all datasets that have 0 rows.
        """
        datasets = await self.list_datasets()

        for dataset in datasets:
            if dataset.row_count == 0:
                await self.delete_dataset(dataset.name)

        logger.info("Deleted all empty datasets")


class ChatHandler(BaseDuckDBHandler):
    """Async handler for chat-related operations."""

    async def _initialize_database(self) -> None:
        """Initialize chat-related tables."""
        await super()._initialize_database()
        async with self._get_connection() as conn:
            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id VARCHAR PRIMARY KEY,
                    user_id VARCHAR NOT NULL,
                    chat_name VARCHAR NOT NULL,
                    data_source VARCHAR DEFAULT 'catalog',
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
                """,
            )

            await self.execute_query(
                conn,
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id VARCHAR PRIMARY KEY,
                    chat_id VARCHAR NOT NULL,
                    message JSON NOT NULL,
                    created_at TIMESTAMP,
                )
                """,
            )

    async def create_chat(
        self, chat_name: str, data_source: str | None = DataSourceType.FILE.value
    ) -> str:
        """
        Create a new chat with the given name and no messages.

        Args:
            chat_name: The name of the chat to create
            data_source: The data source type for this chat (default: registry)

        Returns:
            The ID of the newly created chat
        """
        logger.info(f"Creating new chat '{chat_name}' for user {self.user_id}")

        # Generate a new chat ID
        chat_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc)

        # Create an empty chat
        async with self._get_connection() as conn:
            await self.execute_query(
                conn,
                """
                INSERT INTO chat_history
                    (id, user_id, chat_name, data_source, created_at, updated_at)
                VALUES
                    (?, ?, ?, ?, ?, ?)
                """,
                [
                    chat_id,
                    self.user_id,
                    chat_name,
                    data_source,
                    current_time,
                    current_time,
                ],
            )

        return chat_id

    async def get_chat_messages(
        self, chat_name: str | None = None, chat_id: str | None = None
    ) -> list[AnalystChatMessage]:
        """
        Retrieve a specific chat conversation by name or ID.

        Args:
            chat_name: The name of the chat to retrieve (used if chat_id is not provided)
            chat_id: The ID of the chat to retrieve (takes precedence over chat_name)

        Returns:
            List of chat messages or empty list if not found
        """
        if chat_id:
            logger.info(f"Retrieving chat with ID {chat_id}")
        elif chat_name:
            logger.info(f"Retrieving chat {chat_name} for user {self.user_id}")
        else:
            logger.warning(
                "Neither chat_name nor chat_id provided, returning empty list"
            )
            return []

        async with self._get_connection() as conn:
            # First, get the chat ID if only name was provided
            if not chat_id:
                id_result = await self.execute_query(
                    conn,
                    """
                    SELECT id
                    FROM chat_history
                    WHERE user_id = ? AND chat_name = ?
                    """,
                    [self.user_id, chat_name],
                )
                id_row = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: id_result.fetchone()
                )

                if not id_row:
                    return []

                chat_id = id_row[0]

            # Now retrieve messages for this chat ID
            result = await self.execute_query(
                conn,
                """
                SELECT id, message, chat_id, created_at
                FROM chat_messages
                WHERE chat_id = ?
                ORDER BY created_at
                """,
                [chat_id],
            )

            rows = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchall()
            )

            if rows:
                messages = []
                for row in rows:
                    message = AnalystChatMessage.model_validate(json.loads(row[1]))
                    # Ensure the message has the correct id and chat_id
                    message.id = row[0]
                    message.chat_id = row[2]
                    messages.append(message)
                return messages
            return []

    async def get_chat_names(self) -> list[str]:
        """Get all chat names for the user."""
        logger.info(f"Retrieving chat names for user {self.user_id}")

        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT chat_name
                FROM chat_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                [self.user_id],
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchall()
            )
            return [row[0] for row in rows]

    async def get_chat_list(self) -> list[dict[str, Any]]:
        """
        Get a list of all chats for the user with their IDs and metadata.

        Returns:
            List of dictionaries containing chat information (id, name, data_source, created_at, updated_at)
        """
        logger.info(f"Retrieving chat list for user {self.user_id}")

        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn,
                """
                SELECT id, chat_name, data_source, created_at, updated_at
                FROM chat_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                [self.user_id],
            )
            rows = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchall()
            )

            return [
                {
                    "id": row[0],
                    "name": row[1],
                    "data_source": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                }
                for row in rows
            ]

    async def rename_chat(self, chat_id: str, new_name: str) -> None:
        """
        Rename a chat history entry by its ID.

        Args:
            chat_id: The ID of the chat to rename
            new_name: The new name for the chat
        """
        logger.info(f"Renaming chat with ID {chat_id} to '{new_name}'")

        async with self._get_connection() as conn:
            # Check if the chat exists
            result = await self.execute_query(
                conn, "SELECT 1 FROM chat_history WHERE id = ?", [chat_id]
            )
            exists = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone() is not None
            )

            if not exists:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return

            # Update the chat name
            await self.execute_query(
                conn,
                """
                UPDATE chat_history
                SET chat_name = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                [new_name, datetime.now(timezone.utc), chat_id],
            )

    async def update_chat_data_source(self, chat_id: str, data_source: str) -> None:
        """
        Update the data source for a specific chat.

        Args:
            chat_id: The ID of the chat to update
            data_source: The new data source value
        """
        logger.info(f"Updating data source for chat {chat_id} to '{data_source}'")

        async with self._get_connection() as conn:
            # Check if the chat exists
            result = await self.execute_query(
                conn, "SELECT 1 FROM chat_history WHERE id = ?", [chat_id]
            )
            exists = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone() is not None
            )

            if not exists:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return

            # Update the data source
            await self.execute_query(
                conn,
                """
                UPDATE chat_history
                SET data_source = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                [data_source, datetime.now(timezone.utc), chat_id],
            )

    async def add_chat_message(
        self,
        chat_id: str,
        message: AnalystChatMessage,
    ) -> str:
        """
        Add a new message to a chat.

        Args:
            chat_id: The ID of the chat to update
            message: The message to add

        Returns:
            The ID of the newly added message
        """
        if not chat_id:
            logger.warning("No chat_id provided for add_chat_message operation")
            return ""

        logger.info(f"Adding message to chat with ID {chat_id}")

        # Ensure message has the chat_id and a unique ID
        if not message.id:
            message.id = str(uuid.uuid4())
        message.chat_id = chat_id

        # Check if this chat exists
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn, "SELECT 1 FROM chat_history WHERE id = ?", [chat_id]
            )
            exists = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone() is not None
            )

            if not exists:
                logger.error(f"Chat with ID {chat_id} does not exist")
                return ""

            # Insert the new message
            message_json = json.dumps(message.model_dump(), cls=ChatJSONEncoder)
            await self.execute_query(
                conn,
                """
                INSERT INTO chat_messages
                    (id, chat_id, message, created_at)
                VALUES
                    (?, ?, ?, ?)
                """,
                [
                    message.id,
                    chat_id,
                    message_json,
                    message.created_at,
                ],
            )

            # Update the chat's updated_at timestamp
            await self.execute_query(
                conn,
                """
                UPDATE chat_history SET
                    updated_at = ?
                WHERE id = ?
                """,
                [datetime.now(timezone.utc), chat_id],
            )

            return message.id

    async def delete_chat_message(
        self,
        message_id: str,
    ) -> bool:
        """
        Delete a specific chat message by its ID.

        Args:
            message_id: ID of the message to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        if not message_id:
            logger.warning("No message_id provided for delete_chat_message operation")
            return False

        logger.info(f"Deleting chat message with ID {message_id}")

        async with self._get_connection() as conn:
            # Check if the message exists
            result = await self.execute_query(
                conn, "SELECT chat_id FROM chat_messages WHERE id = ?", [message_id]
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone()
            )

            if not row:
                logger.warning(f"Chat message with ID {message_id} not found")
                return False

            chat_id = row[0]

            # Delete the message
            await self.execute_query(
                conn,
                "DELETE FROM chat_messages WHERE id = ?",
                [message_id],
            )

            # Update the chat's updated_at timestamp
            await self.execute_query(
                conn,
                """
                UPDATE chat_history SET
                    updated_at = ?
                WHERE id = ?
                """,
                [datetime.now(timezone.utc), chat_id],
            )

            return True

    async def get_chat_message(
        self,
        message_id: str,
    ) -> AnalystChatMessage | None:
        """
        Get a specific chat message by its ID.

        Args:
            message_id: ID of the message to retrieve

        Returns:
            The message if found, None otherwise
        """
        if not message_id:
            logger.warning("No message_id provided for get_chat_message operation")
            return None

        logger.info(f"Getting chat message with ID {message_id}")

        async with self._get_connection() as conn:
            # Retrieve the message
            result = await self.execute_query(
                conn,
                """
                SELECT id, message, chat_id, created_at
                FROM chat_messages
                WHERE id = ?
                """,
                [message_id],
            )

            row = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone()
            )

            if not row:
                logger.warning(f"Chat message with ID {message_id} not found")
                return None

            message = AnalystChatMessage.model_validate(json.loads(row[1]))
            # Ensure the message has the correct id and chat_id
            message.id = row[0]
            message.chat_id = row[2]
            return message

    async def update_chat_message(
        self,
        message_id: str,
        message: AnalystChatMessage,
    ) -> bool:
        """
        Update an existing chat message.

        Args:
            message_id: The ID of the message to update
            message: The updated message content

        Returns:
            True if update was successful, False otherwise
        """
        if not message_id:
            logger.warning("No message_id provided for update_chat_message operation")
            return False

        logger.info(f"Updating chat message with ID {message_id}")

        # Preserve the message ID in the updated message
        message.id = message_id

        async with self._get_connection() as conn:
            # Check if the message exists
            result = await self.execute_query(
                conn, "SELECT chat_id FROM chat_messages WHERE id = ?", [message_id]
            )
            row = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone()
            )

            if not row:
                logger.warning(f"Chat message with ID {message_id} does not exist")
                return False

            chat_id = row[0]
            # Ensure chat_id is preserved
            message.chat_id = chat_id

            # Update the message
            message_json = json.dumps(message.model_dump(), cls=ChatJSONEncoder)
            await self.execute_query(
                conn,
                """
                UPDATE chat_messages
                SET message = ?,
                    created_at = ?
                WHERE id = ?
                """,
                [
                    message_json,
                    message.created_at,
                    message_id,
                ],
            )

            # Update the chat's updated_at timestamp
            await self.execute_query(
                conn,
                """
                UPDATE chat_history SET
                    updated_at = ?
                WHERE id = ?
                """,
                [datetime.now(timezone.utc), chat_id],
            )

            return True

    async def update_chat(
        self,
        chat_id: str,
        chat_name: str | None = None,
        messages: list[AnalystChatMessage] | None = None,
        data_source: str | None = None,
    ) -> None:
        """
        Update a specific chat conversation by ID, selectively updating chat_name and/or data_source.
        If messages are provided, they will replace all existing messages for this chat.

        Args:
            chat_id: The ID of the chat to update (required)
            chat_name: Optional new name for the chat
            messages: Optional new list of messages for the chat (will replace all existing messages)
            data_source: Optional new data source for the chat
        """
        if not chat_id:
            logger.warning("No chat_id provided for update operation")
            return

        if not chat_name and messages is None and data_source is None:
            logger.warning(
                "Neither chat_name, messages, nor data_source provided for update operation"
            )
            return

        logger.info(f"Updating chat with ID {chat_id}")

        # Check if this chat exists
        async with self._get_connection() as conn:
            result = await self.execute_query(
                conn, "SELECT 1 FROM chat_history WHERE id = ?", [chat_id]
            )
            exists = await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchone() is not None
            )

            if not exists:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return

            # Build the update query for chat_history
            current_time = datetime.now(timezone.utc)
            update_parts = ["updated_at = ?"]
            params: List[Any] = [current_time]

            if chat_name:
                update_parts.append("chat_name = ?")
                params.append(chat_name)

            if data_source is not None:
                update_parts.append("data_source = ?")
                params.append(data_source)

            # Add chat_id to params
            params.append(chat_id)

            # Execute the update for chat_history
            query = f"""
                UPDATE chat_history SET
                    {", ".join(update_parts)}
                WHERE id = ?
            """
            await self.execute_query(conn, query, params)

            # If messages are provided, replace all existing messages
            if messages is not None:
                # First, delete all existing messages
                await self.execute_query(
                    conn,
                    "DELETE FROM chat_messages WHERE chat_id = ?",
                    [chat_id],
                )

                # Then insert new messages
                for message in messages:
                    # Ensure message has the chat_id and a unique ID if not already set
                    if not message.id:
                        message.id = str(uuid.uuid4())
                    message.chat_id = chat_id

                    message_json = json.dumps(message.model_dump(), cls=ChatJSONEncoder)
                    await self.execute_query(
                        conn,
                        """
                        INSERT INTO chat_messages
                            (id, chat_id, message, created_at)
                        VALUES
                            (?, ?, ?, ?)
                        """,
                        [
                            message.id,
                            chat_id,
                            message_json,
                            message.created_at,
                        ],
                    )

    async def delete_chat(
        self, chat_name: str | None = None, chat_id: str | None = None
    ) -> None:
        """
        Delete a specific chat conversation by name or ID.

        Args:
            chat_name: The name of the chat to delete (used if chat_id is not provided)
            chat_id: The ID of the chat to delete (takes precedence over chat_name)
        """
        if chat_id:
            logger.info(f"Deleting chat with ID {chat_id}")

            async with self._get_connection() as conn:
                # First delete all associated messages
                await self.execute_query(
                    conn, "DELETE FROM chat_messages WHERE chat_id = ?", [chat_id]
                )

                # Then delete the chat history record
                await self.execute_query(
                    conn, "DELETE FROM chat_history WHERE id = ?", [chat_id]
                )
        elif chat_name:
            logger.info(f"Deleting chat {chat_name} for user {self.user_id}")

            async with self._get_connection() as conn:
                # First, get the chat ID
                result = await self.execute_query(
                    conn,
                    """
                    SELECT id
                    FROM chat_history
                    WHERE user_id = ? AND chat_name = ?
                    """,
                    [self.user_id, chat_name],
                )
                row = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: result.fetchone()
                )

                if row:
                    chat_id = row[0]
                    # Delete all associated messages
                    await self.execute_query(
                        conn, "DELETE FROM chat_messages WHERE chat_id = ?", [chat_id]
                    )

                    # Then delete the chat history record
                    await self.execute_query(
                        conn, "DELETE FROM chat_history WHERE id = ?", [chat_id]
                    )
        else:
            logger.warning(
                "Neither chat_name nor chat_id provided for delete operation"
            )

    async def delete_all_chats(self) -> None:
        """Delete all chats for the user."""
        logger.info(f"Deleting all chats for user {self.user_id}")

        async with self._get_connection() as conn:
            # Get all chat IDs for this user
            result = await self.execute_query(
                conn, "SELECT id FROM chat_history WHERE user_id = ?", [self.user_id]
            )
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: result.fetchall()
            )

            # First delete all chat messages for this user's chats
            chat_ids_result = await self.execute_query(
                conn, "SELECT id FROM chat_history WHERE user_id = ?", [self.user_id]
            )
            chat_ids = await asyncio.get_running_loop().run_in_executor(
                None, lambda: chat_ids_result.fetchall()
            )

            for (chat_id,) in chat_ids:
                await self.execute_query(
                    conn, "DELETE FROM chat_messages WHERE chat_id = ?", [chat_id]
                )

            # Then delete the chat history records
            await self.execute_query(
                conn, "DELETE FROM chat_history WHERE user_id = ?", [self.user_id]
            )


class AnalystDB:
    dataset_handler: DatasetHandler
    chat_handler: ChatHandler
    user_id: str
    db_path: Path
    dataset_db_name: str
    chat_db_name: str
    db_version: int

    @classmethod
    async def create(
        cls,
        user_id: str,
        db_path: Path,
        dataset_db_name: str = "dataset",
        chat_db_name: str = "chat",
        db_version: int | None = ANALYST_DATABASE_VERSION,
        use_persistent_storage: bool = False,
    ) -> "AnalystDB":
        self = cls.__new__(cls)
        self.dataset_handler = DatasetHandler(
            user_id=user_id,
            db_path=db_path,
            name=dataset_db_name,
            db_version=db_version,
            use_persistent_storage=use_persistent_storage,
        )
        self.chat_handler = ChatHandler(
            user_id=user_id,
            db_path=db_path,
            name=chat_db_name,
            db_version=db_version,
            use_persistent_storage=use_persistent_storage,
        )
        self.user_id = user_id
        self.db_path = db_path
        self.dataset_db_name = dataset_db_name
        self.chat_db_name = chat_db_name
        await self.initialize()
        return self

    async def initialize(self) -> None:
        """Initialize both database handlers."""
        await self.dataset_handler._initialize_database()
        await self.chat_handler._initialize_database()

    # Dataset operations
    async def register_dataset(
        self,
        df: AnalystDataset | CleansedDataset,
        data_source: DataSourceType,
        file_size: int = 0,
    ) -> None:
        if isinstance(df, CleansedDataset):
            is_cleansed = True
            await self.dataset_handler.store_cleansing_report(
                df.name, df.cleaning_report
            )
        else:
            is_cleansed = False
        try:
            await self.dataset_handler.register_dataframe(
                df.to_df(),
                f"{df.name}_cleansed" if is_cleansed else df.name,
                dataset_type=(
                    DatasetType.CLEANSED if is_cleansed else DatasetType.STANDARD
                ),
                data_source=data_source,
                original_name=df.name,
                file_size=file_size,
            )
        except Exception as e:
            logger.warning(f"Error registering dataset: {e}")

    async def get_dataset(
        self, name: str, max_rows: int | None = 10000
    ) -> AnalystDataset:
        data = AnalystDataset(
            data=await self.dataset_handler.get_dataframe(
                name, expected_type=DatasetType.STANDARD, max_rows=max_rows
            ),
            name=name,
        )
        return data

    async def get_dataset_metadata(self, name: str) -> DatasetMetadata:
        data = await self.dataset_handler.get_dataset_metadata(name)
        return data

    async def get_cleansed_dataset(
        self, name: str, max_rows: int | None = 10000
    ) -> CleansedDataset:
        data = AnalystDataset(
            name=name,
            data=await self.dataset_handler.get_dataframe(
                f"{name}_cleansed",
                expected_type=DatasetType.CLEANSED,
                max_rows=max_rows,
            ),
        )
        cleansing_report = await self.dataset_handler.get_cleansing_report(name)
        return CleansedDataset(dataset=data, cleaning_report=cleansing_report)

    async def register_data_dictionary(self, data_dictionary: DataDictionary) -> None:
        try:
            return await self.dataset_handler.register_dataframe(
                data_dictionary.to_application_df(),
                name=f"{data_dictionary.name}_dict",
                dataset_type=DatasetType.DICTIONARY,
                data_source=DataSourceType.GENERATED,
                original_name=data_dictionary.name,
            )
        except Exception as e:
            logger.warning(
                f"Failed to register data dictionary {data_dictionary.name}: {e}"
            )

    async def get_data_dictionary(self, name: str) -> DataDictionary | None:
        try:
            df = await self.dataset_handler.get_dataframe(
                f"{name}_dict", expected_type=DatasetType.DICTIONARY
            )
            return DataDictionary.from_application_df(df, name=name)
        except Exception as e:
            logger.error(f"Failed to get data dictionary {name}: {e}")
            return None

    async def get_cleansing_report(
        self, dataset_name: str
    ) -> list[CleansedColumnReport] | None:
        return await self.dataset_handler.get_cleansing_report(dataset_name)

    async def list_analyst_datasets(
        self, data_source: DataSourceType | None = None
    ) -> list[str]:
        """
        List all standard datasets, optionally filtered by data source.

        Args:
            data_source: Optional filter by data source (FILE, DATABASE, REGISTRY)

        Returns:
            List of dataset names
        """
        datasets = await self.dataset_handler.list_datasets(
            dataset_type=DatasetType.STANDARD, data_source=data_source
        )
        return [dataset.name for dataset in datasets]

    async def delete_table(self, table_name: str) -> None:
        logger.info(f"Deleting table: {table_name} and related datasets")
        await self.dataset_handler.delete_dataset_with_related(table_name)

    async def delete_dictionary(self, dataset_name: str) -> None:
        logger.info(f"Deleting dictionary for: {dataset_name}")
        await self.dataset_handler.delete_dataset(f"{dataset_name}_dict")

    async def delete_all_tables(self) -> None:
        await self.dataset_handler.delete_all_datasets()

    # Chat operations
    async def create_chat(
        self,
        chat_name: str | None,
        data_source: str | None = DataSourceType.FILE.value,
    ) -> str:
        """
        Create a new chat with the given name and no messages.

        Args:
            chat_name: The name of the chat to create

        Returns:
            The ID of the newly created chat
        """
        now = datetime.now(timezone.utc).isoformat()
        chat_name = chat_name if chat_name else f"chat_{now}"
        return await self.chat_handler.create_chat(
            chat_name=chat_name, data_source=data_source
        )

    async def get_chat_names(
        self,
    ) -> list[str]:
        return await self.chat_handler.get_chat_names()

    async def add_chat_message(
        self,
        chat_id: str,
        message: AnalystChatMessage,
    ) -> str:
        """
        Add a new message to a chat.

        Args:
            chat_id: The ID of the chat to update
            message: The message to add

        Returns:
            The ID of the newly added message
        """
        return await self.chat_handler.add_chat_message(
            chat_id=chat_id, message=message
        )

    async def update_chat_message(
        self,
        message_id: str,
        message: AnalystChatMessage,
    ) -> bool:
        """
        Update an existing chat message directly by ID.

        Args:
            message_id: ID of the message to update
            message: New message content

        Returns:
            True if update was successful, False otherwise
        """
        return await self.chat_handler.update_chat_message(
            message_id=message_id, message=message
        )

    async def delete_chat_message(
        self,
        message_id: str,
    ) -> bool:
        """
        Delete a specific chat message by its ID.

        Args:
            message_id: ID of the message to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        return await self.chat_handler.delete_chat_message(message_id=message_id)

    async def get_chat_message(
        self,
        message_id: str,
    ) -> AnalystChatMessage | None:
        """
        Get a specific chat message by its ID.

        Args:
            message_id: ID of the message to retrieve

        Returns:
            The message if found, None otherwise
        """
        return await self.chat_handler.get_chat_message(message_id=message_id)

    async def get_chat_list(self) -> list[dict[str, Any]]:
        """
        Get a list of all chats for the current user with IDs, names and timestamps.

        Returns:
            List of dictionaries containing chat information
        """
        return await self.chat_handler.get_chat_list()

    async def rename_chat(self, chat_id: str, new_name: str) -> None:
        """
        Rename a chat by its ID.

        Args:
            chat_id: The ID of the chat to rename
            new_name: The new name for the chat
        """
        return await self.chat_handler.rename_chat(chat_id=chat_id, new_name=new_name)

    async def get_chat_messages(
        self, name: str | None = None, chat_id: str | None = None
    ) -> list[AnalystChatMessage]:
        """
        Get a chat by name or ID.

        Args:
            name: The name of the chat (used if chat_id not provided)
            chat_id: The ID of the chat (takes precedence over name)

        Returns:
            List of chat messages or None if not found
        """
        chat_history = await self.chat_handler.get_chat_messages(
            chat_name=name, chat_id=chat_id
        )
        return chat_history

    async def delete_all_chats(self) -> None:
        await self.chat_handler.delete_all_chats()

    async def delete_chat(
        self, name: str | None = None, chat_id: str | None = None
    ) -> None:
        """
        Delete a chat by name or ID.

        Args:
            name: The name of the chat to delete (used if chat_id not provided)
            chat_id: The ID of the chat to delete (takes precedence over name)
        """
        return await self.chat_handler.delete_chat(chat_name=name, chat_id=chat_id)

    async def update_chat_data_source(self, chat_id: str, data_source: str) -> None:
        """
        Update the data source for a specific chat.

        Args:
            chat_id: The ID of the chat to update
            data_source: The new data source setting
        """
        return await self.chat_handler.update_chat_data_source(chat_id, data_source)
