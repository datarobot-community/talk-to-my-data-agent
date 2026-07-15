# Copyright 2026 DataRobot, Inc.
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

import argparse
import csv
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from core.analyst_db import AnalystDB
from core.schema import AnalystChatMessage, UserRoleType
from core.scripts import dump_chat_history

USER_ID = "test_user_123"
OTHER_USER_ID = "test_user_456"


def _args(
    tmp_path: Path,
    *,
    has_feedback: bool = False,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    output: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        db_path=tmp_path,
        output=output,
        has_feedback=has_feedback,
        start_time=start_time,
        end_time=end_time,
    )


async def _create_db(tmp_path: Path, user_id: str = USER_ID) -> AnalystDB:
    # Names must match app_backend/app/middleware.py's get_database() so tests
    # exercise the same on-disk/storage naming as the real app.
    return await AnalystDB.create(
        user_id=user_id,
        db_path=tmp_path,
        dataset_db_name="datasets.db",
        chat_db_name="chat.db",
        user_recipe_db_name="recipe.db",
        data_source_db_name="datasources.db",
        user_metadata_db_name="user_metadata.db",
    )


async def _add_message(
    db: AnalystDB,
    chat_id: str,
    *,
    role: UserRoleType,
    content: str,
    created_at: datetime,
    components: list[str] | None = None,
) -> str:
    message = AnalystChatMessage(
        role=role,
        content=content,
        components=components or [],
        created_at=created_at,
    )
    return await db.add_chat_message(chat_id, message)


def _csv_rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(csv_text)))


def _local_chat_db_path(tmp_path: Path, user_id: str) -> Path:
    return tmp_path / f"chat.db_db_{user_id}.db"


@pytest.mark.asyncio
async def test_collect_rows_exports_question_response_and_components(
    tmp_path: Path,
) -> None:
    db = await _create_db(tmp_path)
    chat_id = await db.create_chat("Test Chat")
    await _add_message(
        db,
        chat_id,
        role="user",
        content="What changed?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Sales increased.",
        components=["chart", "summary"],
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )

    rows = dump_chat_history.collect_rows(
        chat_db_path=_local_chat_db_path(tmp_path, USER_ID),
        has_feedback=False,
        start_time=None,
        end_time=None,
    )

    assert len(rows) == 1
    assert rows[0].user == USER_ID
    assert rows[0].question == "What changed?"
    assert rows[0].response == "Sales increased."
    assert json.loads(rows[0].response_components) == ["chart", "summary"]
    assert rows[0].time == "2026-01-01T10:01:00"
    assert rows[0].feedback == ""
    assert rows[0].feedback_text == ""


@pytest.mark.asyncio
async def test_collect_rows_filters_to_feedback_rows(tmp_path: Path) -> None:
    db = await _create_db(tmp_path)
    chat_id = await db.create_chat("Test Chat")
    await _add_message(
        db,
        chat_id,
        role="user",
        content="Question 1?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    positive_message_id = await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Good answer.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )
    await db.update_message_feedback(
        positive_message_id,
        user_rating=1.0,
        user_feedback="Helpful",
    )
    await _add_message(
        db,
        chat_id,
        role="user",
        content="Question 2?",
        created_at=datetime.fromisoformat("2026-01-01T10:02:00"),
    )
    negative_message_id = await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Bad answer.",
        created_at=datetime.fromisoformat("2026-01-01T10:03:00"),
    )
    await db.update_message_feedback(
        negative_message_id,
        user_rating=-1.0,
        user_feedback="Wrong",
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="No feedback.",
        created_at=datetime.fromisoformat("2026-01-01T10:04:00"),
    )

    rows = dump_chat_history.collect_rows(
        chat_db_path=_local_chat_db_path(tmp_path, USER_ID),
        has_feedback=True,
        start_time=None,
        end_time=None,
    )

    assert [row.response for row in rows] == ["Good answer.", "Bad answer."]
    assert [row.feedback for row in rows] == ["1", "-1"]
    assert [row.feedback_text for row in rows] == ["Helpful", "Wrong"]


@pytest.mark.asyncio
async def test_collect_rows_filters_by_response_time(tmp_path: Path) -> None:
    db = await _create_db(tmp_path)
    chat_id = await db.create_chat("Test Chat")
    await _add_message(
        db,
        chat_id,
        role="user",
        content="Question?",
        created_at=datetime.fromisoformat("2026-01-01T09:59:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Too early.",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Included start.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Included end.",
        created_at=datetime.fromisoformat("2026-01-01T10:02:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Too late.",
        created_at=datetime.fromisoformat("2026-01-01T10:03:00"),
    )

    rows = dump_chat_history.collect_rows(
        chat_db_path=_local_chat_db_path(tmp_path, USER_ID),
        has_feedback=False,
        start_time=datetime.fromisoformat("2026-01-01T10:01:00"),
        end_time=datetime.fromisoformat("2026-01-01T10:02:00"),
    )

    assert [row.response for row in rows] == ["Included start.", "Included end."]


def test_parse_args_rejects_invalid_timestamp() -> None:
    with pytest.raises(SystemExit):
        dump_chat_history.parse_args(["--start-time", "not-a-timestamp"])


def test_parse_args_rejects_start_after_end() -> None:
    with pytest.raises(SystemExit):
        dump_chat_history.parse_args(
            [
                "--start-time",
                "2026-01-01T10:01:00",
                "--end-time",
                "2026-01-01T10:00:00",
            ]
        )


def test_parse_args_rejects_user_id() -> None:
    with pytest.raises(SystemExit):
        dump_chat_history.parse_args(["--user-id", "someone"])


@pytest.mark.asyncio
async def test_run_writes_csv_to_file(tmp_path: Path) -> None:
    db = await _create_db(tmp_path)
    chat_id = await db.create_chat("Test Chat")
    await _add_message(
        db,
        chat_id,
        role="user",
        content="Question?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Answer.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )
    output = tmp_path / "chat_history.csv"

    with patch.dict("os.environ", {"APPLICATION_ID": ""}):
        rc = await dump_chat_history.run(_args(tmp_path, output=output))

    assert rc == 0
    rows = _csv_rows(output.read_text(encoding="utf-8"))
    assert rows[0]["question"] == "Question?"
    assert rows[0]["response"] == "Answer."


@pytest.mark.asyncio
async def test_run_supports_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = await _create_db(tmp_path)
    chat_id = await db.create_chat("Test Chat")
    await _add_message(
        db,
        chat_id,
        role="user",
        content="Question?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db,
        chat_id,
        role="assistant",
        content="Answer.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )

    with patch.dict("os.environ", {"APPLICATION_ID": ""}):
        rc = await dump_chat_history.run(_args(tmp_path))

    assert rc == 0
    rows = _csv_rows(capsys.readouterr().out)
    assert rows[0]["question"] == "Question?"
    assert rows[0]["response"] == "Answer."


@pytest.mark.asyncio
async def test_run_dumps_every_local_user_writes_all_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_a = await _create_db(tmp_path, user_id=USER_ID)
    chat_id_a = await db_a.create_chat("Chat A")
    await _add_message(
        db_a,
        chat_id_a,
        role="user",
        content="Question A?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db_a,
        chat_id_a,
        role="assistant",
        content="Answer A.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )

    db_b = await _create_db(tmp_path, user_id=OTHER_USER_ID)
    chat_id_b = await db_b.create_chat("Chat B")
    await _add_message(
        db_b,
        chat_id_b,
        role="user",
        content="Question B?",
        created_at=datetime.fromisoformat("2026-01-01T11:00:00"),
    )
    await _add_message(
        db_b,
        chat_id_b,
        role="assistant",
        content="Answer B.",
        created_at=datetime.fromisoformat("2026-01-01T11:01:00"),
    )

    with patch.dict("os.environ", {"APPLICATION_ID": ""}):
        rc = await dump_chat_history.run(_args(tmp_path))

    assert rc == 0
    rows = _csv_rows(capsys.readouterr().out)
    assert {row["user"] for row in rows} == {USER_ID, OTHER_USER_ID}
    assert [row["response"] for row in rows] == ["Answer A.", "Answer B."]


@pytest.mark.asyncio
async def test_run_raises_when_no_local_databases_found(tmp_path: Path) -> None:
    with (
        patch.dict("os.environ", {"APPLICATION_ID": ""}),
        pytest.raises(FileNotFoundError),
    ):
        await dump_chat_history.run(_args(tmp_path))


@pytest.mark.asyncio
async def test_run_fetches_every_users_chat_db_from_persistent_storage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_a = await _create_db(tmp_path, user_id=USER_ID)
    chat_id_a = await db_a.create_chat("Chat A")
    await _add_message(
        db_a,
        chat_id_a,
        role="user",
        content="Remote question A?",
        created_at=datetime.fromisoformat("2026-01-01T10:00:00"),
    )
    await _add_message(
        db_a,
        chat_id_a,
        role="assistant",
        content="Remote answer A.",
        created_at=datetime.fromisoformat("2026-01-01T10:01:00"),
    )

    db_b = await _create_db(tmp_path, user_id=OTHER_USER_ID)
    chat_id_b = await db_b.create_chat("Chat B")
    await _add_message(
        db_b,
        chat_id_b,
        role="user",
        content="Remote question B?",
        created_at=datetime.fromisoformat("2026-01-01T11:00:00"),
    )
    await _add_message(
        db_b,
        chat_id_b,
        role="assistant",
        content="Remote answer B.",
        created_at=datetime.fromisoformat("2026-01-01T11:01:00"),
    )

    storage_key_a = f"{USER_ID}_chat.db_db_{USER_ID}.db"
    storage_key_b = f"{OTHER_USER_ID}_chat.db_db_{OTHER_USER_ID}.db"
    source_paths = {
        storage_key_a: _local_chat_db_path(tmp_path, USER_ID),
        storage_key_b: _local_chat_db_path(tmp_path, OTHER_USER_ID),
    }

    class StubStorage:
        def __init__(self) -> None:
            self.fetched: list[tuple[str, str]] = []

        async def files(self) -> list[str]:
            return [
                storage_key_a,
                storage_key_b,
                # Non-chat artifacts must be filtered out.
                f"{USER_ID}_datasets.db_db_{USER_ID}.db",
            ]

        async def fetch_from_storage(self, file_name: str, local_path: str) -> None:
            self.fetched.append((file_name, local_path))
            shutil.copyfile(source_paths[file_name], local_path)

    storage = StubStorage()
    with (
        patch.dict(
            "os.environ",
            {
                "APPLICATION_ID": "app-id",
                "DATAROBOT_ENDPOINT": "https://example.test",
                "DATAROBOT_API_TOKEN": "token",
            },
        ),
        patch.object(
            dump_chat_history,
            "PersistentStorage",
            return_value=storage,
        ) as storage_cls,
    ):
        rc = await dump_chat_history.run(_args(Path("/unused/local/path")))

    assert rc == 0
    storage_cls.assert_called_once_with(None, global_user=True)
    assert {name for name, _ in storage.fetched} == {storage_key_a, storage_key_b}
    rows = _csv_rows(capsys.readouterr().out)
    assert {row["user"] for row in rows} == {USER_ID, OTHER_USER_ID}
    assert [row["response"] for row in rows] == [
        "Remote answer A.",
        "Remote answer B.",
    ]
