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

from __future__ import annotations

from typing import Any, Dict

from pydantic import AliasChoices, AliasPath, Field, field_validator
from pydantic_settings import BaseSettings


class DRCredentials(BaseSettings): ...


class NoDatabaseCredentials(DRCredentials):
    pass


_VALID_JDBC_PREFIXES = (
    "jdbc:postgresql://",
    "jdbc:mysql://",
    "jdbc:sqlserver://",
    "jdbc:snowflake://",
    "jdbc:sap://",
    "jdbc:bigquery://",
)


class JDBCCredentials(DRCredentials):
    jdbc_uri: str = Field(
        validation_alias=AliasChoices(
            "JDBC_URI",
            AliasPath("MLOPS_RUNTIME_PARAM_JDBC_URI", "payload", "apiToken"),
        )
    )
    jdbc_connection_parameters: Dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "JDBC_CONNECTION_PARAMETERS",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_JDBC_CONNECTION_PARAMETERS", "payload", "apiToken"
            ),
        ),
    )

    @field_validator("jdbc_uri")
    @classmethod
    def validate_jdbc_uri(cls, v: str) -> str:
        if not any(v.startswith(prefix) for prefix in _VALID_JDBC_PREFIXES):
            raise ValueError(f"jdbc_uri must start with one of {_VALID_JDBC_PREFIXES}")
        return v

    def __repr__(self) -> str:
        return "JDBCCredentials(jdbc_uri='***', jdbc_connection_parameters=***)"
