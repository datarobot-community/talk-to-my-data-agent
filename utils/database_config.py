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

import os
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from utils.logging_helper import get_logger

logger = get_logger("DatabaseConfig")

class DatabaseConfig(BaseModel):
    """Database configuration model."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="analyst_db")
    user: str = Field(default="postgres")
    password: str = Field(default="postgres")
    schema: str = Field(default="public")

    @property
    def sync_url(self) -> str:
        """Get the synchronous database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        """Get the asynchronous database URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

def get_db_config() -> DatabaseConfig:
    """Get database configuration from environment variables."""
    return DatabaseConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "analyst_db"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        schema=os.getenv("POSTGRES_SCHEMA", "public"),
    )

# Create database engines
db_config = get_db_config()
sync_engine = create_engine(db_config.sync_url)
async_engine = create_async_engine(db_config.async_url)

# Create session factories
SyncSession = sessionmaker(bind=sync_engine)
AsyncSession = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

def get_sync_session():
    """Get a synchronous database session."""
    session = SyncSession()
    try:
        yield session
    finally:
        session.close()

async def get_async_session():
    """Get an asynchronous database session."""
    async with AsyncSession() as session:
        try:
            yield session
        finally:
            await session.close()