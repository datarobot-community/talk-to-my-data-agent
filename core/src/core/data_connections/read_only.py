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

"""Best-effort read-only guard for LLM-generated SQL (APP-6770).

This is defense-in-depth, NOT a security boundary. It blocks obvious top-level writes
and DDL before an LLM-generated statement reaches the customer database, using a simple
first-token allowlist. It deliberately does not try to parse SQL, so it does not catch
every possible write — e.g. a data-modifying CTE (``WITH x AS (DELETE ...)``) or a write
performed by a function inside a SELECT will pass.

The real control is a read-only database login: the privilege level is whatever account
is embedded in ``JDBC_URI`` (see ``core/credentials.py``); there is no code toggle for it.
"""

import re

from core.code_execution import InvalidGeneratedCode

# Line (``-- ...``) and block (``/* ... */``) comments, stripped before inspection.
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

# The only statement forms the analysis LLM uses to return data. EXPLAIN/SHOW/DESCRIBE are
# deliberately excluded — ``EXPLAIN ANALYZE <DML>`` actually executes the DML on Postgres.
_ALLOWED_FIRST_TOKENS = frozenset({"SELECT", "WITH"})


def validate_read_only(sql: str) -> None:
    """Raise ``InvalidGeneratedCode`` if ``sql`` is not an obviously read-only query.

    Raised as ``InvalidGeneratedCode`` (retryable) so the reflection loop re-prompts the
    LLM to rewrite it; the rejected statement never reaches the database. ``exception=`` is
    set so the message still surfaces in the terminal error UI and logs (which read
    ``str(exc.exception)``), not just in the re-prompt.
    """
    # Strip comments, then a single trailing ``;`` so a normal ``SELECT ...;`` is accepted.
    stripped = _COMMENT_RE.sub(" ", sql).strip().removesuffix(";").strip()
    # Ignore a leading ``(`` so ``(SELECT ...) UNION (SELECT ...)`` is accepted, then take
    # the leading keyword — a word match, so ``SELECT(1)`` (no space) reads as ``SELECT``.
    keyword = re.match(r"[a-zA-Z]+", stripped.lstrip("( \t\r\n"))
    first_token = keyword.group(0).upper() if keyword else ""

    if first_token not in _ALLOWED_FIRST_TOKENS:
        msg = (
            f"Only read-only SELECT/WITH queries are permitted; a statement starting with "
            f"'{first_token or '(empty)'}' is not allowed."
        )
        raise InvalidGeneratedCode(msg, code=sql, exception=ValueError(msg))

    # Reject multiple statements. Check the RAW sql (minus one trailing ``;``), NOT the
    # comment-stripped text: a ``;`` smuggled inside a fake ``/* ... */`` span in a string
    # literal would otherwise be removed by the naive comment strip and slip a second
    # statement past this check. Fails safe — a ``;`` in a legit literal/comment rejects.
    if ";" in sql.strip().removesuffix(";"):
        msg = (
            "Only a single read-only statement is permitted; "
            "multiple statements are not allowed."
        )
        raise InvalidGeneratedCode(msg, code=sql, exception=ValueError(msg))
