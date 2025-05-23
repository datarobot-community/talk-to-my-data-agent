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
import json
from pathlib import Path
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional, cast

import polars as pl
import pandas as pd
from sqlalchemy import select, update, delete, text, quoted_name
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import selectinload

from utils.logging_helper import get_logger
from utils.schema import (
    AnalystChatMessage,
    AnalystDataset,
    ChatJSONEncoder,
    CleansedColumnReport,
    CleansedDataset,
    DataDictionary,
    DatasetType,
    DataSourceType,
)
from utils.models import DatasetMetadata, CleansingReport, ChatHistory, ChatMessage
from utils.database_config import get_async_session

logger = get_logger("ApplicationDB")

# increment this number if the database schema has changed to prevent conflicts with existing deployments
# this will force reinitialisation - all tables will be dropped
ANALYST_DATABASE_VERSION = 5


@dataclass
class DatasetMetadataInfo:
    name: str
    dataset_type: DatasetType
    original_name: str  # For cleansed/dictionary datasets, links to their original dataset
    created_at: datetime
    columns: list[str]
    row_count: int
    data_source: DataSourceType
    file_size: int = 0  # Size of the file in bytes


class DatasetHandler:
    def __init__(self, user_id: str):
        self.user_id = user_id

    async def register_dataframe(
        self,
        df: pl.DataFrame,
        name: str,
        dataset_type: DatasetType,
        data_source: DataSourceType,
        original_name: str | None = None,
        file_size: int = 0,
    ) -> None:
        """Register a Polars DataFrame with explicit dataset type tracking."""
        logger.info(f"Registering dataframe {name} as {dataset_type.value}")

        async for session in get_async_session():
            try:
                # Check if dataset already exists
                result = await session.execute(
                    select(DatasetMetadata).where(DatasetMetadata.table_name == name)
                )
                if result.scalar_one_or_none():
                    raise ValueError(f"Table '{name}' already exists in the database")

                # For cleansed/dictionary datasets, verify original exists
                if dataset_type.value in (DatasetType.CLEANSED.value, DatasetType.DICTIONARY.value):
                    if not original_name:
                        raise ValueError(
                            f"original_name required for {dataset_type.value} datasets"
                        )
                    result = await session.execute(
                        select(DatasetMetadata).where(DatasetMetadata.table_name == original_name)
                    )
                    if not result.scalar_one_or_none():
                        raise ValueError(f"Original dataset '{original_name}' not found")

                # Create metadata with timezone-naive datetime
                metadata = DatasetMetadata(
                    table_name=name,
                    dataset_type=dataset_type.value,
                    original_name=original_name or name,
                    created_at=datetime.now(timezone.utc).replace(tzinfo=None),  # Convert to naive datetime
                    columns=list(df.columns),
                    row_count=len(df),
                    data_source=data_source.value,
                    file_size=file_size,
                )
                session.add(metadata)

                # Convert Polars DataFrame to pandas for SQLAlchemy
                pandas_df = df.to_pandas()

                # Get the engine from the session
                engine = session.get_bind()

                # Use to_sql within async context
                await session.run_sync(
                    lambda sync_conn: pandas_df.to_sql(
                        name,
                        engine,  # Use the engine directly
                        if_exists='replace',
                        index=False,
                        method='multi',
                        chunksize=1000
                    )
                )

                # Commit all changes at once
                await session.commit()

            except Exception as e:
                logger.error(f"Error registering dataframe {name}: {e}")
                # Rollback all changes if anything fails
                await session.rollback()
                raise

    async def list_datasets(
        self,
        dataset_type: DatasetType | None = None,
        data_source: DataSourceType | None = None,
    ) -> list[DatasetMetadataInfo]:
        """List all registered datasets of a specific type."""
        logger.info(f"Listing datasets with type={dataset_type.value if dataset_type else 'any'} and source={data_source.value if data_source else 'any'}")

        async for session in get_async_session():
            try:
                # Query datasets with optional filters
                stmt = select(DatasetMetadata)
                if dataset_type:
                    stmt = stmt.where(DatasetMetadata.dataset_type == dataset_type.value)
                if data_source:
                    stmt = stmt.where(DatasetMetadata.data_source == data_source.value)

                result = await session.execute(stmt)
                datasets = result.scalars().all()

                # Convert to DatasetMetadataInfo objects
                return [
                    DatasetMetadataInfo(
                        name=ds.table_name,
                        dataset_type=DatasetType(ds.dataset_type),
                        original_name=ds.original_name,
                        created_at=ds.created_at,
                        columns=ds.columns,
                        row_count=ds.row_count,
                        data_source=DataSourceType(ds.data_source),
                        file_size=ds.file_size,
                    )
                    for ds in datasets
                ]

            except Exception as e:
                logger.error(f"Error listing datasets: {e}")
                raise

    async def get_dataset_metadata(self, name: str) -> DatasetMetadataInfo:
        """Get metadata for a dataset by name."""
        async for session in get_async_session():
            result = await session.execute(
                select(DatasetMetadata).where(DatasetMetadata.table_name == name)
            )
            metadata = result.scalar_one_or_none()
            if not metadata:
                raise ValueError(f"Dataset '{name}' not found")
            return DatasetMetadataInfo(
                name=metadata.table_name,
                dataset_type=DatasetType(metadata.dataset_type),
                original_name=metadata.original_name,
                created_at=metadata.created_at,
                columns=metadata.columns,
                row_count=metadata.row_count,
                data_source=DataSourceType(metadata.data_source),
                file_size=metadata.file_size,
            )

    async def get_dataframe(
        self,
        name: str,
        expected_type: DatasetType | None = None,
        max_rows: int | None = None,
    ) -> pl.DataFrame:
        """Retrieve a registered table as a Polars DataFrame."""
        logger.info(f"Retrieving dataframe {name}")

        async for session in get_async_session():
            try:
                # First verify the dataset exists and check its type
                stmt = select(DatasetMetadata).where(DatasetMetadata.table_name == name)
                result = await session.execute(stmt)
                metadata = result.scalar_one_or_none()
                if not metadata:
                    raise ValueError(f"Dataset '{name}' not found")

                if expected_type and metadata.dataset_type != expected_type.value:
                    raise ValueError(
                        f"Dataset '{name}' is of type {metadata.dataset_type}, "
                        f"expected {expected_type.value}"
                    )

                # Get the data from the table using text() with proper table name quoting
                query = text(f'SELECT * FROM "{name}"')
                if max_rows is not None:
                    query = text(f'SELECT * FROM "{name}" LIMIT :max_rows').bindparams(max_rows=max_rows)

                # Execute the query and convert to Polars DataFrame
                result = await session.execute(query)
                rows = result.fetchall()

                if not rows:
                    # Create empty DataFrame with correct schema
                    return pl.DataFrame(schema={col: pl.String for col in metadata.columns})

                # Convert to Polars DataFrame
                # First convert to pandas for easier handling
                df = pd.DataFrame(rows, columns=metadata.columns)
                # Then convert to Polars
                return pl.from_pandas(df)

            except Exception as e:
                logger.error(f"Error retrieving dataframe {name}: {e}")
                raise

    async def store_cleansing_report(
        self, dataset_name: str, reports: list[CleansedColumnReport]
    ) -> None:
        """Store cleansing reports in the metadata table."""
        async for session in get_async_session():
            report_json = [report.model_dump() for report in reports]
            cleansing_report = CleansingReport(
                dataset_name=dataset_name,
                report=report_json
            )
            session.merge(cleansing_report)
            await session.commit()

    async def get_cleansing_report(
        self, dataset_name: str
    ) -> Optional[list[CleansedColumnReport]]:
        """Retrieve cleansing reports."""
        async for session in get_async_session():
            result = await session.execute(
                select(CleansingReport).where(CleansingReport.dataset_name == dataset_name)
            )
            report = result.scalar_one_or_none()
            if report:
                return [CleansedColumnReport(**r) for r in report.report]
            return []

    async def delete_dataset(self, name: str) -> None:
        """Delete a specific dataset and its metadata."""
        async for session in get_async_session():
            await session.execute(
                delete(DatasetMetadata).where(DatasetMetadata.table_name == name)
            )
            await session.execute(
                delete(CleansingReport).where(CleansingReport.dataset_name == name)
            )
            await session.commit()

    async def delete_dataset_with_related(self, name: str) -> None:
        """Delete a dataset and all its related datasets."""
        async for session in get_async_session():
            # Get related datasets
            result = await session.execute(
                select(DatasetMetadata).where(DatasetMetadata.original_name == name)
            )
            related = result.scalars().all()

            # Delete related datasets
            for dataset in related:
                await session.execute(
                    delete(DatasetMetadata).where(DatasetMetadata.table_name == dataset.table_name)
                )
                await session.execute(
                    delete(CleansingReport).where(CleansingReport.dataset_name == dataset.table_name)
                )

            # Delete the main dataset
            await session.execute(
                delete(DatasetMetadata).where(DatasetMetadata.table_name == name)
            )
            await session.execute(
                delete(CleansingReport).where(CleansingReport.dataset_name == name)
            )
            await session.commit()

    async def delete_all_datasets(self) -> None:
        """Delete all datasets."""
        async for session in get_async_session():
            await session.execute(delete(CleansingReport))
            await session.execute(delete(DatasetMetadata))
            await session.commit()


class ChatHandler:
    def __init__(self, user_id: str):
        self.user_id = user_id

    async def create_chat(
        self, chat_name: str, data_source: str | None = DataSourceType.FILE.value
    ) -> str:
        """Create a new chat with the given name and no messages."""
        logger.info(f"Creating new chat '{chat_name}' for user {self.user_id}")

        chat_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc)

        async for session in get_async_session():
            chat = ChatHistory(
                id=chat_id,
                user_id=self.user_id,
                chat_name=chat_name,
                data_source=data_source,
                created_at=current_time,
                updated_at=current_time,
            )
            session.add(chat)
            await session.commit()
            return chat_id

    async def get_chat_messages(
        self, chat_name: str | None = None, chat_id: str | None = None
    ) -> list[AnalystChatMessage]:
        """Retrieve a specific chat conversation by name or ID."""
        if chat_id:
            logger.info(f"Retrieving chat with ID {chat_id}")
        elif chat_name:
            logger.info(f"Retrieving chat {chat_name} for user {self.user_id}")
        else:
            logger.warning("Neither chat_name nor chat_id provided, returning empty list")
            return []

        async for session in get_async_session():
            if not chat_id:
                result = await session.execute(
                    select(ChatHistory).where(
                        ChatHistory.user_id == self.user_id,
                        ChatHistory.chat_name == chat_name,
                    )
                )
                chat = result.scalar_one_or_none()
                if not chat:
                    return []
                chat_id = chat.id

            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.chat_id == chat_id)
                .order_by(ChatMessage.created_at)
            )
            messages = result.scalars().all()
            return [
                AnalystChatMessage(
                    id=m.id,
                    chat_id=m.chat_id,
                    role=m.role,
                    content=m.content,
                    components=m.components,
                    in_progress=m.in_progress,
                    created_at=m.created_at,
                )
                for m in messages
            ]

    async def get_chat_names(self) -> list[str]:
        """Get all chat names for the user."""
        logger.info(f"Retrieving chat names for user {self.user_id}")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatHistory.chat_name)
                .where(ChatHistory.user_id == self.user_id)
                .order_by(ChatHistory.created_at.desc())
            )
            return [row[0] for row in result.all()]

    async def get_chat_list(self) -> list[dict[str, Any]]:
        """Get a list of all chats for the user with their IDs and metadata."""
        logger.info(f"Retrieving chat list for user {self.user_id}")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatHistory)
                .where(ChatHistory.user_id == self.user_id)
                .order_by(ChatHistory.created_at.desc())
            )
            chats = result.scalars().all()
            return [
                {
                    "id": chat.id,
                    "name": chat.chat_name,
                    "data_source": chat.data_source,
                    "created_at": chat.created_at,
                    "updated_at": chat.updated_at,
                }
                for chat in chats
            ]

    async def rename_chat(self, chat_id: str, new_name: str) -> None:
        """Rename a chat history entry by its ID."""
        logger.info(f"Renaming chat with ID {chat_id} to '{new_name}'")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.id == chat_id)
            )
            chat = result.scalar_one_or_none()
            if not chat:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return

            chat.chat_name = new_name
            chat.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def update_chat_data_source(self, chat_id: str, data_source: str) -> None:
        """Update the data source for a specific chat."""
        logger.info(f"Updating data source for chat {chat_id} to '{data_source}'")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.id == chat_id)
            )
            chat = result.scalar_one_or_none()
            if not chat:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return

            chat.data_source = data_source
            chat.updated_at = datetime.now(timezone.utc)
            await session.commit()

    async def add_chat_message(
        self,
        chat_id: str,
        message: AnalystChatMessage,
    ) -> str:
        """Add a new message to a chat."""
        if not chat_id:
            logger.warning("No chat_id provided for add_chat_message operation")
            return ""

        logger.info(f"Adding message to chat with ID {chat_id}")

        if not message.id:
            message.id = str(uuid.uuid4())
        message.chat_id = chat_id

        async for session in get_async_session():
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.id == chat_id)
            )
            chat = result.scalar_one_or_none()
            if not chat:
                logger.warning(f"Chat with ID {chat_id} does not exist")
                return ""

            chat_message = ChatMessage(
                id=message.id,
                chat_id=chat_id,
                role=message.role,
                content=message.content,
                components=[c.model_dump() for c in message.components],
                in_progress=message.in_progress,
                created_at=message.created_at,
            )
            session.add(chat_message)

            chat.updated_at = datetime.now(timezone.utc)
            await session.commit()
            return message.id

    async def delete_chat_message(
        self,
        message_id: str,
    ) -> bool:
        """Delete a specific chat message by its ID."""
        if not message_id:
            logger.warning("No message_id provided for delete_chat_message operation")
            return False

        logger.info(f"Deleting chat message with ID {message_id}")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatMessage).where(ChatMessage.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                logger.warning(f"Chat message with ID {message_id} not found")
                return False

            chat_id = message.chat_id
            await session.delete(message)

            # Update the chat's updated_at timestamp
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.id == chat_id)
            )
            chat = result.scalar_one_or_none()
            if chat:
                chat.updated_at = datetime.now(timezone.utc)

            await session.commit()
            return True

    async def get_chat_message(
        self,
        message_id: str,
    ) -> AnalystChatMessage | None:
        """Get a specific chat message by its ID."""
        if not message_id:
            logger.warning("No message_id provided for get_chat_message operation")
            return None

        logger.info(f"Getting chat message with ID {message_id}")

        async for session in get_async_session():
            result = await session.execute(
                select(ChatMessage).where(ChatMessage.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                logger.warning(f"Chat message with ID {message_id} not found")
                return None

            return AnalystChatMessage(
                id=message.id,
                chat_id=message.chat_id,
                role=message.role,
                content=message.content,
                components=message.components,
                in_progress=message.in_progress,
                created_at=message.created_at,
            )

    async def update_chat_message(
        self,
        message_id: str,
        message: AnalystChatMessage,
    ) -> bool:
        """Update an existing chat message."""
        if not message_id:
            logger.warning("No message_id provided for update_chat_message operation")
            return False

        logger.info(f"Updating chat message with ID {message_id}")

        message.id = message_id

        async for session in get_async_session():
            result = await session.execute(
                select(ChatMessage).where(ChatMessage.id == message_id)
            )
            existing_message = result.scalar_one_or_none()
            if not existing_message:
                logger.warning(f"Chat message with ID {message_id} does not exist")
                return False

            chat_id = existing_message.chat_id
            message.chat_id = chat_id

            existing_message.content = message.content
            existing_message.components = [c.model_dump() for c in message.components]
            existing_message.in_progress = message.in_progress
            existing_message.created_at = message.created_at

            # Update the chat's updated_at timestamp
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.id == chat_id)
            )
            chat = result.scalar_one_or_none()
            if chat:
                chat.updated_at = datetime.now(timezone.utc)

            await session.commit()
            return True

    async def delete_chat(
        self, chat_name: str | None = None, chat_id: str | None = None
    ) -> None:
        """Delete a specific chat conversation by name or ID."""
        if chat_id:
            logger.info(f"Deleting chat with ID {chat_id}")

            async for session in get_async_session():
                await session.execute(
                    delete(ChatMessage).where(ChatMessage.chat_id == chat_id)
                )
                await session.execute(
                    delete(ChatHistory).where(ChatHistory.id == chat_id)
                )
                await session.commit()
        elif chat_name:
            logger.info(f"Deleting chat {chat_name} for user {self.user_id}")

            async for session in get_async_session():
                result = await session.execute(
                    select(ChatHistory).where(
                        ChatHistory.user_id == self.user_id,
                        ChatHistory.chat_name == chat_name,
                    )
                )
                chat = result.scalar_one_or_none()
                if chat:
                    await session.execute(
                        delete(ChatMessage).where(ChatMessage.chat_id == chat.id)
                    )
                    await session.execute(
                        delete(ChatHistory).where(ChatHistory.id == chat.id)
                    )
                    await session.commit()
        else:
            logger.warning("Neither chat_name nor chat_id provided for delete operation")

    async def delete_all_chats(self) -> None:
        """Delete all chats for the user."""
        logger.info(f"Deleting all chats for user {self.user_id}")

        async for session in get_async_session():
            # Get all chat IDs for this user
            result = await session.execute(
                select(ChatHistory).where(ChatHistory.user_id == self.user_id)
            )
            chats = result.scalars().all()

            # Delete all messages for these chats
            for chat in chats:
                await session.execute(
                    delete(ChatMessage).where(ChatMessage.chat_id == chat.id)
                )

            # Delete all chat history records
            await session.execute(
                delete(ChatHistory).where(ChatHistory.user_id == self.user_id)
            )
            await session.commit()


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
    ) -> "AnalystDB":
        self = cls.__new__(cls)
        self.dataset_handler = DatasetHandler(user_id)
        self.chat_handler = ChatHandler(user_id)
        self.user_id = user_id
        self.db_path = db_path
        self.dataset_db_name = dataset_db_name
        self.chat_db_name = chat_db_name
        self.db_version = db_version
        return self

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
            raise

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

    async def get_dataset_metadata(self, name: str) -> DatasetMetadataInfo:
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

    async def get_dataframe(self, name: str) -> pl.DataFrame:
        """Get a dataframe by name."""
        logger.info(f"Getting dataframe {name} for user {self.user_id}")

        async for session in get_async_session():
            result = await session.execute(
                select(DatasetMetadata).where(DatasetMetadata.table_name == name)
            )
            dataset = result.scalar_one_or_none()
            if not dataset:
                logger.warning(f"Dataset {name} not found")
                return pl.DataFrame()

            # For now, return an empty DataFrame with the correct columns
            # TODO: Implement actual data retrieval from PostgreSQL
            return pl.DataFrame(columns=dataset.columns)
