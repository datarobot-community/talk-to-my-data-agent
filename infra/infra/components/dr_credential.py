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

import json
import logging
import textwrap
from typing import Any

import pulumi_datarobot as datarobot
import pydantic
from core.credentials import (  # type: ignore[import-not-found]
    DRCredentials,
    JDBCCredentials,
    NoDatabaseCredentials,
)
from core.schema import (  # type: ignore[import-not-found]
    DatabaseConnectionType,
)
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME

logger = logging.getLogger("DataAnalystFrontend")


def get_credential_runtime_parameter_values(
    credentials: DRCredentials | None,
) -> list[datarobot.CustomModelRuntimeParameterValueArgs]:
    if credentials is None:
        return []

    credential_rtp_dicts: list[dict[str, Any]] = []

    rtps: list[dict[str, Any]] = []

    if isinstance(credentials, NoDatabaseCredentials):
        credential_rtp_dicts = []  # No credentials to add for NoDatabaseCredentials
    elif isinstance(credentials, JDBCCredentials):
        rtps = [
            {
                "key": "DATABASE_CONNECTION_TYPE",
                "type": "string",
                "value": "datarobot_jdbc",
            },
            {
                "key": "JDBC_URI",
                "type": "credential",
                "value": credentials.jdbc_uri,
            },
        ]
        if credentials.jdbc_connection_parameters:
            rtps.append(
                {
                    "key": "JDBC_CONNECTION_PARAMETERS",
                    "type": "credential",
                    "value": json.dumps(credentials.jdbc_connection_parameters),
                }
            )
        credential_rtp_dicts = rtps

    credential_runtime_parameter_values: list[
        datarobot.CustomModelRuntimeParameterValueArgs
    ] = []

    for rtp_dict in credential_rtp_dicts:
        dr_credential: (
            datarobot.ApiTokenCredential
            | datarobot.GoogleCloudCredential
            | datarobot.AwsCredential
            | datarobot.BasicCredential
        )
        if "credential" in rtp_dict["type"]:
            if rtp_dict["type"] == "credential":
                dr_credential = datarobot.ApiTokenCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} Credential [{PROJECT_NAME}]",
                    api_token=rtp_dict["value"],
                )
            elif rtp_dict["type"] == "basic_credential":
                dr_credential = datarobot.BasicCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} Credential [{PROJECT_NAME}]",
                    user=rtp_dict["value"]["user"],
                    password=rtp_dict["value"]["password"],
                )
            elif rtp_dict["type"] == "google_credential":
                dr_credential = datarobot.GoogleCloudCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} Credential [{PROJECT_NAME}]",
                    gcp_key=rtp_dict["value"]["gcpKey"],
                )
            else:
                raise ValueError(f"Unsupported credential type {rtp_dict['type']}")
            rtp = datarobot.CustomModelRuntimeParameterValueArgs(
                key=rtp_dict["key"],
                type=(
                    "credential"
                    if "credential" in rtp_dict["type"]
                    else rtp_dict["type"]
                ),
                value=dr_credential.id,
            )
        else:
            rtp = datarobot.CustomModelRuntimeParameterValueArgs(
                key=rtp_dict["key"],
                type=rtp_dict["type"],
                value=rtp_dict["value"],
            )
        credential_runtime_parameter_values.append(rtp)

    return credential_runtime_parameter_values


def get_database_credentials(
    database: DatabaseConnectionType,
    test_credentials: bool = True,
) -> JDBCCredentials | NoDatabaseCredentials:
    credentials: JDBCCredentials | NoDatabaseCredentials

    try:
        if database == "no_database":
            return NoDatabaseCredentials()

        if database in ("snowflake", "bigquery", "sap", "datarobot_jdbc"):
            credentials = JDBCCredentials()  # type: ignore[call-arg]
            if test_credentials:
                from core.data_connections.database.database_implementations import (  # type: ignore[import-not-found]
                    JdbcPreviewOperator,
                )

                try:
                    JdbcPreviewOperator(credentials).validate_connection()
                except Exception as e:
                    raise ValueError(f"Failed to connect via JDBC ({database}).") from e
            return credentials

    except pydantic.ValidationError as exc:
        msg = "Validation errors in database credentials. Using no database configuration.\n"
        logger.exception(msg)
        raise ValueError(
            textwrap.dedent(
                f"""
                There was an error validating the database credentials.

                Please validate your credentials or check {__file__} for details.
                """
            )
        ) from exc

    raise ValueError(
        textwrap.dedent(
            f"""
            The supplied database of {database} did not correspond to a supported database.
            """
        )
    )
