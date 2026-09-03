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
from unittest.mock import Mock, patch

import pytest

from core.analyst_db import AnalystDB, BaseDuckDBHandler
from core.persistent_storage import PersistentStorage


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
    storage: Mock


@pytest.fixture
def mocks() -> Generator[Mocks, None, None]:
    with (
        patch("duckdb.DuckDBPyConnection") as conn,
        patch("duckdb.connect") as connect,
        patch(
            "core.analyst_db.PersistentStorage", autospec=PersistentStorage
        ) as storage,
    ):
        connect.return_value = conn
        storage.return_value = storage
        yield Mocks(conn=conn, storage=storage)


@pytest.mark.asyncio
async def test_write_conn_saves_to_storage(mocks: Mocks) -> None:
    test_db = StubDBHandler()
    async with test_db._write_connection() as conn:
        assert conn is mocks.conn

    mocks.storage.save_to_storage.assert_called_once_with(
        "app_db.db", str(Path("./app_db.db").absolute())
    )
    mocks.conn.close.assert_called()


@pytest.mark.asyncio
async def test_read_conn_not_saves_to_storage(mocks: Mocks) -> None:
    test_db = StubDBHandler()

    async with test_db._read_connection():
        pass

    mocks.storage.save_to_storage.assert_not_called()
    mocks.conn.close.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_database_fetches_when_db_missing(mocks: Mocks) -> None:
    test_db = StubDBHandler()

    await test_db._initialize_database()

    execute_calls = mocks.conn.execute.call_args_list

    create_db_version_calls = [
        c[0][0]
        for c in execute_calls
        if c[0][0].strip().lower().startswith("create table if not exists db_version")
    ]

    assert len(create_db_version_calls) == 1

    mocks.storage.fetch_from_storage.assert_awaited_once_with(
        test_db.db_path.name, str(test_db.db_path.absolute())
    )
    assert test_db._initialized is True


# --- Persisted dictionary-generation error tracking -------------------------


@pytest.mark.asyncio
async def test_dictionary_error_upsert_and_clear(tmp_path: Path) -> None:
    from core.analyst_db import AnalystDB

    db = await AnalystDB.create(user_id="err_upsert", db_path=tmp_path)

    assert await db.get_dictionary_error("ds") is None

    await db.mark_dictionary_failed("ds", "boom")
    assert await db.get_dictionary_error("ds") == "boom"

    await db.mark_dictionary_failed("ds", "new")
    assert await db.get_dictionary_error("ds") == "new"

    await db.clear_dictionary_error("ds")
    assert await db.get_dictionary_error("ds") is None


@pytest.mark.asyncio
async def test_dictionary_error_survives_restart(tmp_path: Path) -> None:
    from core.analyst_db import AnalystDB

    db1 = await AnalystDB.create(user_id="err_restart", db_path=tmp_path)
    await db1.mark_dictionary_failed("ds", "invalid model")

    db2 = await AnalystDB.create(user_id="err_restart", db_path=tmp_path)
    assert await db2.get_dictionary_error("ds") == "invalid model"


@pytest.mark.asyncio
async def test_dictionary_error_cleared_on_dataset_delete(tmp_path: Path) -> None:
    import polars as pl

    from core.analyst_db import AnalystDB, InternalDataSourceType
    from core.schema import AnalystDataset

    db = await AnalystDB.create(user_id="err_delete", db_path=tmp_path)

    dataset = AnalystDataset(name="ds", data=pl.DataFrame({"a": [1, 2]}))
    await db.register_dataset(dataset, data_source=InternalDataSourceType.FILE)
    await db.mark_dictionary_failed("ds", "boom")

    await db.delete_table("ds")
    assert await db.get_dictionary_error("ds") is None


# --- APP-6841: SQL injection via dataset name -> DuckDB table identifier ------


async def _catalog_table_names(db: AnalystDB) -> set[str]:
    """Every physical table in the dataset DB's DuckDB catalog.

    Not `list_datasets`/`table_exists`, which read only the `dataset_metadata` table
    and would miss an injected table that has no metadata row.
    """
    handler = db.dataset_handler
    async with handler._read_connection() as conn:
        result = await handler.execute_query(
            conn, "SELECT table_name FROM information_schema.tables"
        )
        rows = result.fetchall()
    return {str(row[0]) for row in rows}


async def _register_standard(
    db: AnalystDB, name: str, rows: dict[str, list[int]]
) -> None:
    """Register a non-empty STANDARD file dataset via the low-level handler, bypassing
    `register_dataset`'s exception-swallowing so a rejected name surfaces in tests."""
    import polars as pl

    from core.analyst_db import DatasetType, InternalDataSourceType

    await db.dataset_handler.register_dataframe(
        pl.DataFrame(rows),
        name=name,
        dataset_type=DatasetType.STANDARD,
        data_source=InternalDataSourceType.FILE,
    )


@pytest.mark.asyncio
async def test_single_quote_payload_does_not_inject_on_create(tmp_path: Path) -> None:
    """A filename payload that breaks out of the old single-quoted CREATE is now just
    one ordinary table; no extra table is created."""
    db = await AnalystDB.create(user_id="inj_create", db_path=tmp_path)

    payload = "a' AS SELECT 1; CREATE TABLE pwned AS SELECT 1; --"
    await _register_standard(db, payload, {"a": [1, 2]})

    tables = await _catalog_table_names(db)
    assert "pwned" not in tables
    assert payload in tables  # created byte-for-byte, not broken up

    df = await db.dataset_handler.get_dataframe(payload)
    assert df["a"].to_list() == [1, 2]


@pytest.mark.asyncio
async def test_double_quote_payload_does_not_inject_on_read_or_delete(
    tmp_path: Path,
) -> None:
    """A payload containing a double quote exercises the ``"`` -> ``""`` escaping and the
    already-double-quoted read/delete sinks. Create, read, and delete stay contained."""
    db = await AnalystDB.create(user_id="inj_readdelete", db_path=tmp_path)

    payload = 'x"; CREATE TABLE pwned AS SELECT 1; --'
    await _register_standard(db, payload, {"a": [1, 2, 3]})

    # Read path (get_dataframe) — no breakout.
    df = await db.dataset_handler.get_dataframe(payload)
    assert df["a"].to_list() == [1, 2, 3]
    assert "pwned" not in await _catalog_table_names(db)

    # Delete path (delete_dataset) — drops exactly the payload table, nothing else.
    await db.dataset_handler.delete_dataset(payload)
    tables = await _catalog_table_names(db)
    assert payload not in tables
    assert "pwned" not in tables


# --- APP-6778: empty query results (0 rows / 0 columns) ----------------------


@pytest.mark.asyncio
async def test_register_empty_dataframe_with_columns_creates_real_table(
    tmp_path: Path,
) -> None:
    """A 0-row frame that carries its columns must create a real, empty, typed
    table — not a phantom `dataset_metadata` row with no physical table (which
    later 404s on read, APP-6778)."""
    import polars as pl

    from core.analyst_db import DatasetType, InternalDataSourceType

    db = await AnalystDB.create(user_id="empty_cols", db_path=tmp_path)

    empty = pl.DataFrame(schema={"a": pl.Int64, "b": pl.String})
    await db.dataset_handler.register_dataframe(
        empty,
        name="empty_ds",
        dataset_type=DatasetType.STANDARD,
        data_source=InternalDataSourceType.FILE,
    )

    # A physical table exists (not merely a metadata row) and reads back empty.
    assert "empty_ds" in await _catalog_table_names(db)
    assert await db.dataset_handler.table_exists("empty_ds")

    got = await db.dataset_handler.get_dataframe("empty_ds")
    assert got.columns == ["a", "b"]
    assert got.height == 0
    # dtypes survive the round-trip — not collapsed to INTEGER (the Null-dtype bug).
    assert got.schema["a"] == pl.Int64
    assert got.schema["b"] == pl.String

    metadata = await db.dataset_handler.get_dataset_metadata("empty_ds")
    assert metadata.row_count == 0
    assert metadata.columns == ["a", "b"]


@pytest.mark.asyncio
async def test_register_zero_by_zero_dataframe_raises(tmp_path: Path) -> None:
    """A 0-column frame cannot be stored (DuckDB rejects a 0-column CREATE), so
    refuse it outright instead of writing a phantom metadata-without-table row."""
    import polars as pl

    from core.analyst_db import DatasetType, InternalDataSourceType

    db = await AnalystDB.create(user_id="zero_by_zero", db_path=tmp_path)

    with pytest.raises(ValueError, match="no columns"):
        await db.dataset_handler.register_dataframe(
            pl.DataFrame(),
            name="empty00",
            dataset_type=DatasetType.STANDARD,
            data_source=InternalDataSourceType.FILE,
        )

    # Neither a phantom metadata row nor a physical table was left behind.
    assert not await db.dataset_handler.table_exists("empty00")
    assert "empty00" not in await _catalog_table_names(db)


def test_validate_table_name_accepts_normal_names() -> None:
    from core.analyst_db import validate_table_name

    for good in [
        "my report (2024)",
        "O'Brien",
        "kunden_übersicht",
        "catalog.schema.table",
        'weird"quote',
        "semi;colon",
    ]:
        validate_table_name(good)  # must not raise


def test_validate_table_name_rejects_malformed() -> None:
    from core.analyst_db import validate_table_name

    for bad in ["", "   ", "x\x00y", "line\nbreak", "tab\tchar", "c1\x85ctrl"]:
        with pytest.raises(ValueError):
            validate_table_name(bad)


def test_validate_table_name_rejects_reserved_case_insensitive() -> None:
    from core.analyst_db import validate_table_name

    for reserved in [
        "dataset_metadata",
        "DATASET_METADATA",
        "cleansing_reports",
        "dictionary_errors",
        "db_version",
        "temp_view",
        "Temp_View",
    ]:
        with pytest.raises(ValueError):
            validate_table_name(reserved)


@pytest.mark.asyncio
async def test_reserved_name_rejected_and_metadata_intact(tmp_path: Path) -> None:
    """Uploading a file named after a control table must not clobber it."""
    db = await AnalystDB.create(user_id="reserved", db_path=tmp_path)
    assert "dataset_metadata" in await _catalog_table_names(db)

    with pytest.raises(ValueError):
        await _register_standard(db, "dataset_metadata", {"a": [1]})

    # Control table survived and the DB is still usable.
    assert "dataset_metadata" in await _catalog_table_names(db)
    await _register_standard(db, "ok_ds", {"a": [1]})
    assert (await db.dataset_handler.get_dataframe("ok_ds")).shape[0] == 1
