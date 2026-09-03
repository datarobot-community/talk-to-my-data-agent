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

"""Unit tests for the APP-6770 best-effort read-only SQL guard."""

import pytest

from core.code_execution import InvalidGeneratedCode
from core.data_connections.read_only import validate_read_only

ALLOWED = [
    "SELECT * FROM users",
    "select id from t",
    "WITH t AS (SELECT 1) SELECT * FROM t",
    "SELECT 1;",
    "  SELECT 1  ",
    "(SELECT id FROM a) UNION (SELECT id FROM b)",
    "-- a comment\nSELECT 1",
    "/* block */ SELECT 1",
    "SELECT * FROM t WHERE status = 'DELETED'",  # keyword only inside a literal
    "SELECT * FROM t WHERE note = 'DROP me'",
    "SELECT(1)",  # no space after SELECT — first token still reads as SELECT
]

REJECTED = [
    "DROP TABLE users",
    "DELETE FROM users",
    "INSERT INTO t VALUES (1)",
    "UPDATE t SET x = 1",
    "TRUNCATE t",
    "ALTER TABLE t ADD c int",
    "CREATE TABLE t (id int)",
    "EXPLAIN ANALYZE DELETE FROM t",  # EXPLAIN ANALYZE runs the DML on Postgres
    "SELECT 1; DROP TABLE t",  # smuggled second statement
    "SELECT '/*' AS a; DROP TABLE t; SELECT '*/'",  # ; hidden by a fake comment span
    "",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_read_only(sql: str) -> None:
    validate_read_only(sql)  # must not raise


@pytest.mark.parametrize("sql", REJECTED)
def test_rejects_writes_and_multistatement(sql: str) -> None:
    with pytest.raises(InvalidGeneratedCode) as exc_info:
        validate_read_only(sql)
    # The message must survive to the terminal UI / logs, which read str(exc.exception) —
    # so exception must be set and non-empty (not None → would render as "None").
    assert exc_info.value.exception is not None
    assert str(exc_info.value.exception)
    assert exc_info.value.code == sql


def test_semicolon_in_literal_is_rejected_failsafe() -> None:
    # Documented limitation (APP-6770): a ';' inside a string literal trips the
    # multi-statement check. This fails SAFE — it rejects a valid query rather than
    # letting a write through. The real control is a read-only DB login.
    with pytest.raises(InvalidGeneratedCode):
        validate_read_only("SELECT * FROM t WHERE note = 'a;b'")
