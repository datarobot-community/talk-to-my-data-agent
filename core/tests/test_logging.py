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
import inspect
import json
import logging

from core.telemetry.logging import (
    _LITELLM_LOGGER_NAMES,
    JsonFormatter,
    get_logger,
    init_logging,
)


def _make_record(msg: str) -> logging.LogRecord:
    return logging.getLogger("test.logger").makeRecord(
        "test.logger", logging.INFO, __file__, 1, msg, (), None
    )


def test_json_formatter_emits_levelname_not_level() -> None:
    # Regression test (APP-6631): the DataRobot OTel collector's severity_parser
    # only reads "levelname" (matching platform services' JSON logs), never
    # "level" - every JSON log line from this formatter was silently defaulted
    # to INFO otherwise.
    formatted = json.loads(JsonFormatter().format(_make_record("hello")))
    assert formatted["levelname"] == "INFO"
    assert "level" not in formatted


def test_init_logging_defaults_to_json() -> None:
    assert inspect.signature(init_logging).parameters["format_type"].default == "json"


def test_get_logger_defaults_to_json() -> None:
    # log_api_call calls get_logger() with no explicit format_type - its default
    # must match init_logging's so its logs aren't silently downgraded to plaintext.
    assert inspect.signature(get_logger).parameters["format_type"].default == "json"


def test_init_logging_detaches_litellm_handlers() -> None:
    # LiteLLM attaches its own unconfigured StreamHandler (defaults to stderr) to
    # these loggers at import time. Left in place, it duplicates every record
    # through a plain-text, unredacted path and can defeat the OTel collector's
    # recombine heuristic (litellm's own messages can start with "\n").
    for name in _LITELLM_LOGGER_NAMES:
        logging.getLogger(name).addHandler(logging.NullHandler())

    init_logging()

    for name in _LITELLM_LOGGER_NAMES:
        assert logging.getLogger(name).handlers == []
