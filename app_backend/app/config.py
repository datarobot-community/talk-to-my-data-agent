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
import os

from core.telemetry.enums import FormatType, LogLevel
from datarobot.core.config import DataRobotAppFrameworkBaseSettings
from pydantic import ValidationInfo, field_validator


class Config(DataRobotAppFrameworkBaseSettings):
    log_level: LogLevel = LogLevel.INFO
    log_format: FormatType = "readable"
    otel_entity_id: str = ""
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""
    otel_sdk_disabled: bool = False

    @field_validator("otel_exporter_otlp_headers", mode="before")
    @classmethod
    def _assemble_otel_headers(cls, v: object, info: ValidationInfo) -> object:
        if v:
            return v
        entity_id = (info.data or {}).get("otel_entity_id", "")
        api_token = (info.data or {}).get("datarobot_api_token", "") or os.environ.get(
            "DATAROBOT_API_TOKEN", ""
        )
        if entity_id and api_token:
            return f"x-datarobot-entity-id={entity_id},x-datarobot-api-key={api_token}"
        return v

    @field_validator("otel_sdk_disabled", mode="before")
    @classmethod
    def _coerce_empty_string(cls, v: object) -> object:
        return False if v == "" else v
