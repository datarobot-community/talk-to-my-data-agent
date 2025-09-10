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
from enum import Enum

from fastapi import HTTPException


class UsageExceptionType(Enum):
    DATASET_ALREADY_USED = (400, "DATASET_USED")
    DATASETS_TOO_LARGE = (400, "DATASET_TOO_LARGE")
    DATASETS_INVALID = (400, "DATASET_INVALID")


class ApplicationUsageException(HTTPException):
    def __init__(
        self, exception_type: UsageExceptionType, message: str | None = None
    ) -> None:
        status_code, code = exception_type.value
        detail = {"code": code}
        if message:
            detail["message"] = message
        super().__init__(status_code, detail)
