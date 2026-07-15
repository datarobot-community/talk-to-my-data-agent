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

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import duckdb

from core.persistent_storage import PersistentStorage

logger = logging.getLogger(__name__)

# Must match the `chat_db_name` passed to AnalystDB.create() in app_backend/app/middleware.py.
CHAT_DB_NAME = "chat.db"
# Persistent storage keys are named "<user_id>_{CHAT_DB_NAME}_db_<user_id>.db".
_CHAT_STORAGE_MARKER = f"_{CHAT_DB_NAME}_db_"
FIELDNAMES = [
    "user",
    "question",
    "response",
    "response_components",
    "time",
    "feedback",
    "feedback_text",
]


@dataclass(frozen=True)
class ChatHistoryRow:
    user: str
    question: str
    response: str
    response_components: str
    time: str
    feedback: str
    feedback_text: str

    def to_csv_row(self) -> dict[str, str]:
        return {
            "user": self.user,
            "question": self.question,
            "response": self.response,
            "response_components": self.response_components,
            "time": self.time,
            "feedback": self.feedback,
            "feedback_text": self.feedback_text,
        }


def _parse_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid ISO-8601 timestamp: {value}") from e
    return _to_naive_utc(parsed)


def _to_naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _format_datetime(value: datetime) -> str:
    return _to_naive_utc(value).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dump chat history for every user to CSV."
    )
    parser.add_argument(
        "--db-path",
        default=".",
        type=Path,
        help=(
            "Directory containing local chat database files "
            "(default: current directory). Ignored when APPLICATION_ID is set, "
            "in which case every user's chat database is fetched from "
            "persistent storage."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="CSV output path. Writes to stdout when omitted.",
    )
    parser.add_argument(
        "--has-feedback",
        action="store_true",
        help="Only include assistant responses that have feedback.",
    )
    parser.add_argument(
        "--start-time",
        type=_parse_datetime,
        help="Only include responses at or after this ISO-8601 timestamp.",
    )
    parser.add_argument(
        "--end-time",
        type=_parse_datetime,
        help="Only include responses at or before this ISO-8601 timestamp.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    start_time = args.start_time
    end_time = args.end_time
    if start_time is not None and end_time is not None and start_time > end_time:
        parser.error("--start-time must be before or equal to --end-time")
    return args


def _require_persistent_storage_env() -> None:
    missing = [
        env_name
        for env_name in (
            "APPLICATION_ID",
            "DATAROBOT_ENDPOINT",
            "DATAROBOT_API_TOKEN",
        )
        if not os.environ.get(env_name)
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")


def _uses_persistent_storage() -> bool:
    return bool(os.environ.get("APPLICATION_ID"))


async def _fetch_chat_db_paths_from_storage(temp_dir: Path) -> list[Path]:
    _require_persistent_storage_env()
    storage = PersistentStorage(None, global_user=True)
    all_files = await storage.files()
    chat_files = sorted(f for f in all_files if _CHAT_STORAGE_MARKER in f)
    if not chat_files:
        raise FileNotFoundError(
            "No chat databases found in persistent storage for application "
            f"{os.environ.get('APPLICATION_ID')}."
        )
    logger.info("Found %d chat database(s) in persistent storage.", len(chat_files))
    local_paths = []
    for file_name in chat_files:
        local_path = temp_dir / file_name
        await storage.fetch_from_storage(file_name, str(local_path))
        local_paths.append(local_path)
    return local_paths


def _local_chat_db_paths(db_path: Path) -> list[Path]:
    pattern = f"{CHAT_DB_NAME}_db_*.db"
    local_paths = sorted(db_path.glob(pattern))
    if not local_paths:
        raise FileNotFoundError(
            f"No chat databases matching {pattern!r} found in {db_path}."
        )
    logger.info("Found %d local chat database(s).", len(local_paths))
    return local_paths


async def _source_chat_db_paths(args: argparse.Namespace, temp_dir: Path) -> list[Path]:
    if _uses_persistent_storage():
        return await _fetch_chat_db_paths_from_storage(temp_dir)
    return _local_chat_db_paths(args.db_path)


def _decode_message(message_json: object) -> dict[str, Any]:
    if isinstance(message_json, str):
        data = json.loads(message_json)
    else:
        data = json.loads(str(message_json))
    if not isinstance(data, dict):
        raise ValueError("Chat message JSON must decode to an object.")
    return data


def _feedback_value(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, Real):
        return str(value)
    rating = float(value)
    if rating.is_integer():
        return str(int(rating))
    return str(rating)


def _has_feedback(user_rating: object, user_feedback: object) -> bool:
    return user_rating is not None or user_feedback not in (None, "")


def _should_include(
    created_at: datetime,
    start_time: datetime | None,
    end_time: datetime | None,
) -> bool:
    comparable_created_at = _to_naive_utc(created_at)
    if start_time is not None and comparable_created_at < start_time:
        return False
    if end_time is not None and comparable_created_at > end_time:
        return False
    return True


def collect_rows(
    *,
    chat_db_path: Path,
    has_feedback: bool,
    start_time: datetime | None,
    end_time: datetime | None,
) -> list[ChatHistoryRow]:
    rows: list[ChatHistoryRow] = []
    last_question_by_chat_id: dict[str, str] = {}

    with duckdb.connect(str(chat_db_path), read_only=True) as conn:
        query_rows = conn.execute(
            """
            SELECT
                h.user_id,
                m.chat_id,
                m.message,
                m.created_at,
                mf.user_rating,
                mf.user_feedback
            FROM chat_messages m
            INNER JOIN chat_history h
                ON h.id = m.chat_id
            LEFT JOIN message_feedback mf
                ON mf.message_id = m.id
            ORDER BY m.chat_id, m.created_at, m.id
            """
        ).fetchall()

    for row in query_rows:
        row_user_id, chat_id, message_json, created_at, user_rating, user_feedback = row
        message = _decode_message(message_json)
        role = message.get("role")
        content = str(message.get("content") or "")

        if role == "user":
            last_question_by_chat_id[str(chat_id)] = content
            continue

        if role != "assistant" or not isinstance(created_at, datetime):
            continue

        if not _should_include(created_at, start_time, end_time):
            continue

        if has_feedback and not _has_feedback(user_rating, user_feedback):
            continue

        components = message.get("components", [])
        rows.append(
            ChatHistoryRow(
                user=str(row_user_id),
                question=last_question_by_chat_id.get(str(chat_id), ""),
                response=content,
                response_components=json.dumps(components, ensure_ascii=False),
                time=_format_datetime(created_at),
                feedback=_feedback_value(user_rating),
                feedback_text=str(user_feedback) if user_feedback is not None else "",
            )
        )

    return rows


def write_csv(rows: list[ChatHistoryRow], output: Path | None) -> None:
    output_context = (
        output.open("w", encoding="utf-8", newline="")
        if output is not None
        else nullcontext(sys.stdout)
    )
    with output_context as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


async def run(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        chat_db_paths = await _source_chat_db_paths(args, Path(temp_dir))
        rows: list[ChatHistoryRow] = []
        for chat_db_path in chat_db_paths:
            rows.extend(
                collect_rows(
                    chat_db_path=chat_db_path,
                    has_feedback=args.has_feedback,
                    start_time=args.start_time,
                    end_time=args.end_time,
                )
            )
    rows.sort(key=lambda row: row.time)
    output: Path | None = args.output
    write_csv(rows, output)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
