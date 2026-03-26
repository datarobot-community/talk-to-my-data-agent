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

import json
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Coroutine, ParamSpec, TypeVar

logging.basicConfig(level=logging.INFO)


class ExFormatter(logging.Formatter):
    def_keys = [
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
    ]

    def format(self, record: logging.LogRecord) -> str:
        string = super().format(record)
        extra = {k: v for k, v in record.__dict__.items() if k not in self.def_keys}
        if len(extra) > 0:
            string += " - extra: " + str(extra)
        return string


def get_logger(name: str = "DataAnalystBackend") -> logging.Logger:
    formatter = ExFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    consoleHandle = logging.StreamHandler()
    consoleHandle.setLevel(logging.INFO)
    consoleHandle.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.propagate = False  # Prevent propagation to root logger
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.addHandler(consoleHandle)
    return logger


# Helper functions
def format_json(obj: Any) -> str:
    try:
        if hasattr(obj, "dict"):
            obj = obj.dict()
        if isinstance(obj, dict) and "messages" in obj:
            formatted_obj = obj.copy()
            for msg in formatted_obj["messages"]:
                if len(msg.get("content", "")) > 100:
                    msg["content"] = msg["content"][:100] + "..."
            return json.dumps(
                formatted_obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
            )
        return json.dumps(
            obj, indent=2, sort_keys=True, default=str, ensure_ascii=False
        )
    except Exception as e:
        return f"Error formatting JSON: {str(e)}\nOriginal object: {str(obj)}"


P = ParamSpec("P")
T = TypeVar("T")


def log_api_call(
    func: Callable[P, Coroutine[Any, Any, T]],
) -> Callable[P, Coroutine[Any, Any, T]]:
    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        logger = get_logger()
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        separator = f"\n{'=' * 80}\n"
        logger.info(
            f"{separator}API CALL START: {func.__name__} [{request_id}]{separator}"
        )
        try:
            result = await func(*args, **kwargs)

            logger.info(
                f"{separator}API CALL COMPLETE: {func.__name__} [{request_id}]{separator}"
            )
            return result
        except Exception as e:
            error_log = (
                f"ERROR IN API CALL [{request_id}]\n"
                "------------------------\n"
                f"Function: {func.__name__}\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {str(e)}\n\n"
                "Stack Trace:\n"
            )
            logger.error(error_log, exc_info=True)
            raise

    return wrapper
