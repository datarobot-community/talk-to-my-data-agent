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

from dataclasses import dataclass
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.analyst_db import BaseDuckDBHandler
from core.persistent_fs.duckdb_extension import DuckDBPyConnectionWrapper


class StubDBHandler(BaseDuckDBHandler):
    def __init__(self) -> None:
        super().__init__(use_persistent_storage=True)
        self._initialized = False

    async def _initialize_child(self) -> None:
        self._initialized = True
        return

    async def blank_write(self) -> None:
        async with self._write_connection():
            pass

    async def blank_read(self) -> None:
        async with self._read_connection():
            pass


@dataclass
class Mocks:
    conn: Mock
    dr_conn: Mock
    preload_file: Mock


@pytest.fixture
def mocks() -> Generator[Mocks, None, None]:
    dr_conn = MagicMock()
    with (
        patch("duckdb.DuckDBPyConnection") as conn,
        patch("duckdb.connect") as connect,
        patch("core.analyst_db.connect_dr_fs", return_value=dr_conn),
        patch("core.analyst_db.preload_file") as mock_preload_file,
    ):
        connect.return_value = conn
        yield Mocks(conn=conn, dr_conn=dr_conn, preload_file=mock_preload_file)


@pytest.mark.asyncio
async def test_write_conn_saves_to_storage(mocks: Mocks) -> None:
    test_db = StubDBHandler()
    async with test_db._write_connection() as conn:
        assert conn is mocks.dr_conn

    mocks.dr_conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_read_conn_not_saves_to_storage(mocks: Mocks) -> None:
    test_db = StubDBHandler()

    async with test_db._read_connection():
        pass

    mocks.dr_conn.close.assert_not_called()
    mocks.conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_database_fetches_when_db_missing(mocks: Mocks) -> None:
    test_db = StubDBHandler()

    await test_db._initialize_database()

    execute_calls = mocks.dr_conn.execute.call_args_list

    create_db_version_calls = [
        c[0][0]
        for c in execute_calls
        if c[0][0].strip().lower().startswith("create table if not exists db_version")
    ]

    assert len(create_db_version_calls) == 1

    mocks.preload_file.assert_called_once_with(str(test_db.db_path.absolute()))
    assert test_db._initialized is True
