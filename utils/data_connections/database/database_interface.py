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

from __future__ import annotations

import functools
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generic, TypeVar

from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)

from utils.analyst_db import AnalystDB
from utils.credentials import (
    GoogleCredentialsBQ,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
)
from utils.logging_helper import get_logger

logger = get_logger("DatabaseHelper")

T = TypeVar("T")
_DEFAULT_DB_QUERY_TIMEOUT = 300


@dataclass
class SnowflakeCredentialArgs:
    credentials: SnowflakeCredentials


@dataclass
class BigQueryCredentialArgs:
    credentials: GoogleCredentialsBQ


@dataclass
class SAPDatasphereCredentialArgs:
    credentials: SAPDatasphereCredentials


@dataclass
class NoDatabaseCredentialArgs:
    credentials: NoDatabaseCredentials


WARMUP_RECHECK_INTERVAL_SECONDS = 30 * 60


class DatabaseOperator(ABC, Generic[T]):
    @abstractmethod
    def __init__(self, credentials: T, default_timeout: int): ...

    @abstractmethod
    @asynccontextmanager
    def create_connection(self) -> AsyncGenerator[Any, None]: ...

    async def warmup(self, timeout: int | None = 300) -> None:
        query = self.warmup_query()
        if not query:
            return

        now = time.time()
        if not hasattr(self, "_warmup_time"):
            self._warmup_time = now
            await self.execute_query(query, timeout=timeout)
        elif now - self._warmup_time > WARMUP_RECHECK_INTERVAL_SECONDS:
            self._warmup_time = now
            await self.execute_query(query, timeout=timeout)

    def warmup_query(self) -> str | None:
        # "SELECT 1" would almost certainly work here, but not putting it to avoid causing problems with non-sql datasources that might be integrated.
        return None

    @abstractmethod
    async def execute_query(
        self, query: str, timeout: int | None = None
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]: ...

    @abstractmethod
    async def get_tables(self, timeout: int | None = None) -> list[str]:
        return []

    @functools.lru_cache(maxsize=8)
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

    def query_friendly_name(self, dataset_name: str) -> str:
        """Return a query-friendly version of the dataset name (e.g. quoted table name if that's required)."""
        return dataset_name


class NoDatabaseOperator(DatabaseOperator[NoDatabaseCredentialArgs]):
    def __init__(
        self,
        credentials: NoDatabaseCredentials,
        default_timeout: int = _DEFAULT_DB_QUERY_TIMEOUT,
    ):
        self._credentials = credentials

    @asynccontextmanager
    async def create_connection(self) -> AsyncGenerator[None, None]:
        yield None

    async def execute_query(
        self,
        query: str,
        timeout: int | None = 300,
    ) -> list[tuple[Any, ...]] | list[dict[str, Any]]:
        return []

    async def get_tables(self, timeout: int | None = 300) -> list[str]:
        return []

    @functools.lru_cache(8)
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
