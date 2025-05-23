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
from typing import AsyncGenerator
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.models import Base

# Load environment variables
load_dotenv()

# Get database connection details from environment variables
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "analyst_db")

# Create the database URL
DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create the async engine with better connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,  # Increased pool size
    max_overflow=20,  # Increased max overflow
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,  # Enable connection health checks
    isolation_level="READ COMMITTED",  # Set transaction isolation level
)

# Create the async session factory with better concurrency settings
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session with proper connection management."""
    async with async_session_factory() as session:
        try:
            # Start a new transaction
            async with session.begin():
                yield session
        except Exception as e:
            # Rollback on error
            await session.rollback()
            raise
        finally:
            # Ensure session is closed
            await session.close()

# async def init_db():
#     """Initialize the database."""
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)

# Create sync engine for Alembic
SYNC_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
sync_engine = create_engine(SYNC_DATABASE_URL)