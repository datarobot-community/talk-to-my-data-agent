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
    GoogleCredentialsBQ,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
)
from core.schema import (  # type: ignore[import-not-found]
    DatabaseConnectionType,
)
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME

from .. import project_dir

logger = logging.getLogger("DataAnalystFrontend")


def get_credential_runtime_parameter_values(
    credentials: DRCredentials | None,
) -> list[datarobot.CustomModelRuntimeParameterValueArgs]:
    if credentials is None:
        return []

    credential_rtp_dicts: list[dict[str, Any]] = []

    rtps: list[dict[str, Any]] = []

    if isinstance(credentials, GoogleCredentialsBQ):
        rtps = [
            {
                "key": "DATABASE_CONNECTION_TYPE",
                "type": "string",
                "value": "bigquery",
            },
            {
                "key": "GOOGLE_SERVICE_ACCOUNT_BQ",
                "type": "google_credential",
                "value": {"gcpKey": json.dumps(credentials.service_account_key)},
            },
        ]
        if credentials.region:
            rtps.append(
                {
                    "key": "GOOGLE_REGION_BQ",
                    "type": "string",
                    "value": credentials.region,
                }
            )
        if credentials.db_schema:
            rtps.append(
                {
                    "key": "GOOGLE_DB_SCHEMA_BQ",
                    "type": "string",
                    "value": credentials.db_schema,
                }
            )
        credential_rtp_dicts = rtps
    elif isinstance(credentials, NoDatabaseCredentials):
        credential_rtp_dicts = []  # No credentials to add for NoDatabaseCredentials
    elif isinstance(credentials, SnowflakeCredentials):
        rtps = [
            {
                "key": "DATABASE_CONNECTION_TYPE",
                "type": "string",
                "value": "snowflake",
            }
        ]
        if credentials.user and credentials.password:
            rtps.append(
                {
                    "key": "db_credential",
                    "type": "basic_credential",
                    "value": {
                        "user": credentials.user,
                        "password": credentials.password,
                    },
                }
            )
        elif credentials.user:
            rtps.append(
                {
                    "key": "SNOWFLAKE_USER",
                    "type": "string",
                    "value": credentials.user,
                }
            )
        rtps.extend(
            [
                {
                    "key": "SNOWFLAKE_ACCOUNT",
                    "type": "string",
                    "value": credentials.account,
                },
                {
                    "key": "SNOWFLAKE_WAREHOUSE",
                    "type": "string",
                    "value": credentials.warehouse,
                },
                {
                    "key": "SNOWFLAKE_DATABASE",
                    "type": "string",
                    "value": credentials.database,
                },
                {
                    "key": "SNOWFLAKE_SCHEMA",
                    "type": "string",
                    "value": credentials.db_schema,
                },
                {
                    "key": "SNOWFLAKE_ROLE",
                    "type": "string",
                    "value": credentials.role,
                },
            ]
        )
        if credentials.snowflake_key_path:
            rtps.append(
                {
                    "key": "SNOWFLAKE_KEY_PATH",
                    "type": "string",
                    "value": credentials.snowflake_key_path,
                }
            )
        credential_rtp_dicts = rtps
    elif isinstance(credentials, SAPDatasphereCredentials):
        rtps = [
            {
                "key": "DATABASE_CONNECTION_TYPE",
                "type": "string",
                "value": "sap",
            },
            {
                "key": "db_credential",
                "type": "basic_credential",
                "value": {
                    "user": credentials.user,
                    "password": credentials.password,
                },
            },
            {
                "key": "SAP_DATASPHERE_HOST",
                "type": "string",
                "value": credentials.host,
            },
            {
                "key": "SAP_DATASPHERE_PORT",
                "type": "string",
                "value": str(credentials.port),
            },
            {
                "key": "SAP_DATASPHERE_SCHEMA",
                "type": "string",
                "value": credentials.db_schema,
            },
        ]
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
) -> (
    SnowflakeCredentials
    | GoogleCredentialsBQ
    | SAPDatasphereCredentials
    | NoDatabaseCredentials
):
    credentials: (
        SnowflakeCredentials
        | GoogleCredentialsBQ
        | SAPDatasphereCredentials
        | NoDatabaseCredentials
    )

    try:
        if database == "no_database":
            return NoDatabaseCredentials()

        if database == "snowflake":
            credentials = SnowflakeCredentials()  # type: ignore[call-arg]
            if not credentials.is_configured():
                logger.error("Snowflake credentials not fully configured")
                raise ValueError(
                    textwrap.dedent(
                        f"""
                        Your Snowflake credentials and environment variables were not configured properly.

                        Please validate your environment variables or check {__file__} for details.
                        """
                    )
                )

            if test_credentials:
                import snowflake.connector

                connect_params: dict[str, Any] = {
                    "user": credentials.user,
                    "account": credentials.account,
                    "warehouse": credentials.warehouse,
                    "database": credentials.database,
                    "schema": credentials.db_schema,
                    "role": credentials.role,
                }

                if private_key := credentials.get_private_key(
                    project_root=project_dir.parent
                ):
                    connect_params["private_key"] = private_key
                elif credentials.password:
                    connect_params["password"] = credentials.password
                else:
                    logger.error(
                        "No valid authentication method configured for Snowflake"
                    )
                    raise ValueError(
                        textwrap.dedent(
                            f"""
                            No authentication method was configured for Snowflake.

                            Please validate your credentials or check {__file__} for details.
                            """
                        )
                    )

                try:
                    sf_con = snowflake.connector.connect(**connect_params)
                    sf_con.close()
                except Exception as e:
                    logger.exception("Failed to test Snowflake connection")
                    raise ValueError(
                        textwrap.dedent(
                            f"""
                            Unable to run a successful test of snowflake with the given credentials.

                            Please validate your credentials or check {__file__} for details.
                            """
                        )
                    ) from e

            return credentials

        elif database == "bigquery":
            credentials = GoogleCredentialsBQ()  # type: ignore[call-arg]
            if test_credentials:
                import google.cloud.bigquery
                from google.oauth2 import service_account

                google_credentials = (
                    service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                        credentials.service_account_key,
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                )
                bq_con = google.cloud.bigquery.Client(credentials=google_credentials)
                bq_con.close()  # type: ignore[no-untyped-call]
            return credentials
        elif database == "sap":
            credentials = SAPDatasphereCredentials()  # type: ignore[call-arg]
            if test_credentials:
                from hdbcli import dbapi  # type: ignore[import-untyped]

                connect_params = {
                    "address": credentials.host,
                    "port": credentials.port,
                    "user": credentials.user,
                    "password": credentials.password,
                }

                # Connect to SAP Data Sphere
                try:
                    connection = dbapi.connect(**connect_params)
                    connection.close()
                except Exception as e:
                    raise ValueError("Failed to connect to SAP Data Sphere.") from e
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
