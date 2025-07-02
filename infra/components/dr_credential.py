# Copyright 2024 DataRobot, Inc.
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

import pulumi
import pulumi_datarobot as datarobot
import pydantic
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.llms import LLMConfig, LLMs

from utils.credentials import (
    AWSBedrockCredentials,
    AzureOpenAICredentials,
    DRCredentials,
    GoogleCredentials,
    GoogleCredentialsBQ,
    NoDatabaseCredentials,
    SAPDatasphereCredentials,
    SnowflakeCredentials,
)
from utils.schema import (
    DatabaseConnectionType,
    RuntimeCredentialType,
)

from ..settings_main import PROJECT_ROOT

logger = logging.getLogger("DataAnalystFrontend")


def get_credential_runtime_parameter_values(
    credentials: DRCredentials | None,
    credential_type: RuntimeCredentialType = "llm",
) -> list[datarobot.CustomModelRuntimeParameterValueArgs]:
    if credentials is None:
        return []
    if isinstance(credentials, AzureOpenAICredentials):
        rtps: list[dict[str, Any]] = [
            {
                "key": "OPENAI_API_KEY",
                "type": "credential",
                "value": credentials.api_key,
                "description": "API Token credential for Azure OpenAI",
            },
            {
                "key": "OPENAI_API_BASE",
                "type": "string",
                "value": credentials.azure_endpoint,
                "description": "Azure OpenAI endpoint URL",
            },
            {
                "key": "OPENAI_API_DEPLOYMENT_ID",
                "type": "string",
                "value": credentials.azure_deployment,
                "description": "Azure OpenAI deployment name",
            },
            {
                "key": "OPENAI_API_VERSION",
                "type": "string",
                "value": credentials.api_version,
                "description": "Azure OpenAI API version",
            },
        ]
        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]
    elif isinstance(credentials, GoogleCredentials):
        rtps = [
            {
                "key": "GOOGLE_SERVICE_ACCOUNT",
                "type": "google_credential",
                "value": {"gcpKey": json.dumps(credentials.service_account_key)},
            }
        ]
        if credentials.region:
            rtps.append(
                {"key": "GOOGLE_REGION", "type": "string", "value": credentials.region}
            )
        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]
    elif isinstance(credentials, GoogleCredentialsBQ):
        rtps = [
            {
                "key": "GOOGLE_SERVICE_ACCOUNT_BQ",
                "type": "google_credential",
                "value": {"gcpKey": json.dumps(credentials.service_account_key)},
            }
        ]
        if credentials.region:
            rtps.append(
                {
                    "key": "GOOGLE_REGION_BQ",
                    "type": "string",
                    "value": credentials.region,
                }
            )
        if credential_type == "db":
            rtps.append(
                {
                    "key": "GOOGLE_DB_SCHEMA_BQ",
                    "type": "string",
                    "value": credentials.db_schema,
                }
            )
        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]
    elif isinstance(credentials, AWSBedrockCredentials):
        rtps = [
            {
                "key": "AWS_ACCOUNT",
                "type": "aws_credential",
                "value": {
                    "awsAccessKeyId": credentials.aws_access_key_id,
                    "awsSecretAccessKey": credentials.aws_secret_access_key,
                    "awsSessionToken": credentials.aws_session_token,
                },
            }
        ]
        if credentials.region_name:
            rtps.append(
                {
                    "key": "AWS_REGION",
                    "type": "string",
                    "value": credentials.region_name,
                }
            )

        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]
    elif isinstance(credentials, NoDatabaseCredentials):
        credential_rtp_dicts = []  # No credentials to add for NoDatabaseCredentials
    elif isinstance(credentials, SnowflakeCredentials):
        rtps = (
            [
                {
                    "key": "db_credential",
                    "type": "basic_credential",
                    "value": {
                        "user": credentials.user,
                        "password": credentials.password,
                    },
                }
            ]
            if credentials.user and credentials.password
            else [
                {
                    "key": "SNOWFLAKE_USER",
                    "type": "string",
                    "value": credentials.user,
                }
            ]
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
                {
                    "key": "SNOWFLAKE_KEY_PATH",
                    "type": "string",
                    "value": credentials.snowflake_key_path,
                },
            ]
        )
        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]
    elif isinstance(credentials, SAPDatasphereCredentials):
        rtps = [
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
                "value": credentials.port,
            },
            {
                "key": "SAP_DATASPHERE_SCHEMA",
                "type": "string",
                "value": credentials.db_schema,
            },
        ]
        credential_rtp_dicts = [rtp for rtp in rtps if rtp["value"] is not None]

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
            elif rtp_dict["type"] == "google_credential":
                dr_credential = datarobot.GoogleCloudCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} {credential_type} Credential [{PROJECT_NAME}]",
                    gcp_key=rtp_dict["value"].get("gcpKey"),
                )
            elif rtp_dict["type"] == "aws_credential":
                dr_credential = datarobot.AwsCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} Credential [{PROJECT_NAME}]",
                    aws_access_key_id=rtp_dict["value"]["awsAccessKeyId"],
                    aws_secret_access_key=rtp_dict["value"]["awsSecretAccessKey"],
                    aws_session_token=rtp_dict["value"].get("awsSessionToken"),
                )
            elif rtp_dict["type"] == "basic_credential":
                dr_credential = datarobot.BasicCredential(
                    resource_name=f"Generative Analyst {rtp_dict['key']} Credential [{PROJECT_NAME}]",
                    user=rtp_dict["value"]["user"],
                    password=rtp_dict["value"]["password"],
                )
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


# Initialize the LLM client based on the selected LLM and its credential type
def get_llm_credentials(
    llm: LLMConfig, test_credentials: bool = True
) -> DRCredentials | None:
    try:
        credentials: DRCredentials
        if llm == LLMs.DEPLOYED_LLM:
            return None
        if llm.credential_type == "azure":
            credentials = AzureOpenAICredentials()
            if test_credentials:
                try:
                    import openai

                    lookup = {
                        LLMs.AZURE_OPENAI_GPT_3_5_TURBO.name: "gpt-35-turbo",
                        LLMs.AZURE_OPENAI_GPT_3_5_TURBO_16K.name: "gpt-35-turbo-16k",
                        LLMs.AZURE_OPENAI_GPT_4.name: "gpt-4",
                        LLMs.AZURE_OPENAI_GPT_4_32K.name: "gpt-4-32k",
                        LLMs.AZURE_OPENAI_GPT_4_O.name: "gpt-4o",
                        LLMs.AZURE_OPENAI_GPT_4_TURBO.name: "gpt-4-turbo",
                        LLMs.AZURE_OPENAI_GPT_4_O_MINI.name: "gpt-4o-mini",
                    }
                    if (
                        credentials.azure_deployment is not None
                        and credentials.azure_deployment != lookup[llm.name]
                    ):
                        pulumi.warn(
                            textwrap.dedent(
                                f"""\
                                Environment variable OPENAI_API_DEPLOYMENT_ID doesn't match the LLM Blueprint specified in settings_generative.py.

                                LLM Blueprint specified in settings_generative.py: {llm.name}
                                Expected:\tOPENAI_API_DEPLOYMENT_ID="{lookup[llm.name]}"
                                Current:\tOPENAI_API_DEPLOYMENT_ID="{credentials.azure_deployment}"
                                """
                            )
                        )
                    openai_client = openai.AzureOpenAI(
                        azure_endpoint=credentials.azure_endpoint,
                        azure_deployment=credentials.azure_deployment
                        or lookup[llm.name],
                        api_key=credentials.api_key,
                        api_version=credentials.api_version or "2023-05-15",
                    )
                    openai_client.chat.completions.create(
                        model=llm.name,
                        messages=[{"role": "user", "content": "Hello"}],
                    )
                except Exception as e:
                    raise ValueError(
                        textwrap.dedent(
                            f"""\
                            Unable to run a successful test completion against deployment '{credentials.azure_deployment or lookup[llm.name]}'
                            on '{credentials.azure_endpoint}' with API version '{credentials.api_version or "2023-05-15"}'
                            with provided Azure OpenAI credentials. Please validate your credentials.
                            
                            Please validate your credentials or check {__file__} for details.
                            """
                        )
                    ) from e

        elif llm.credential_type == "aws":
            credentials = AWSBedrockCredentials()
            if test_credentials:
                lookup = {
                    LLMs.ANTHROPIC_CLAUDE_3_HAIKU.name: "anthropic.claude-3-haiku-20240307-v1:0",
                    LLMs.ANTHROPIC_CLAUDE_3_SONNET.name: "anthropic.claude-3-sonnet-20240229-v1:0",
                    LLMs.ANTHROPIC_CLAUDE_3_OPUS.name: "anthropic.claude-3-opus-20240229-v1:0",
                    LLMs.AMAZON_TITAN.name: "amazon.titan-text-express-v1",
                    LLMs.ANTHROPIC_CLAUDE_2.name: "anthropic.claude-v2:1",
                }
                if credentials.region_name is None:
                    pulumi.warn("AWS region not set. Using default 'us-west-1'.")
                try:
                    import boto3

                    if "anthropic" in lookup[llm.name]:
                        request_body = {
                            "anthropic_version": "bedrock-2023-05-31",
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 100,
                            "temperature": 0,
                        }
                    else:
                        request_body = {"inputText": "Hello"}

                    session = boto3.Session(
                        aws_access_key_id=credentials.aws_access_key_id,
                        aws_secret_access_key=credentials.aws_secret_access_key,
                        aws_session_token=credentials.aws_session_token,
                        region_name=credentials.region_name or "us-west-1",
                    )
                    bedrock_client = session.client("bedrock-runtime")
                    bedrock_client.invoke_model(
                        accept="application/json",
                        contentType="application/json",
                        modelId=lookup[llm.name],
                        body=json.dumps(request_body),
                    )

                except Exception as e:
                    raise ValueError(
                        textwrap.dedent(
                            f"""
                            Unable to run a successful test completion against model '{lookup[llm.name]}' in region '{credentials.region_name or "us-west-1"}' 
                            using request body '{request_body}' with provided AWS credentials.
                            
                            
                            Please validate your credentials or check {__file__} for details.
                            """
                        )
                    ) from e
        elif llm.credential_type == "google":
            credentials = GoogleCredentials()
            if test_credentials:
                lookup = {
                    LLMs.GOOGLE_1_5_PRO.name: "gemini-1.5-pro-002",
                    LLMs.GOOGLE_BISON.name: "chat-bison@002",
                    LLMs.GOOGLE_GEMINI_1_5_FLASH.name: "gemini-1.5-flash-002",
                }
                try:
                    import openai
                    from google.auth.transport.requests import Request
                    from google.oauth2 import service_account

                    google_credentials = (
                        service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                            credentials.service_account_key,
                            scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        )
                    )

                    auth_request = Request()  # type: ignore[no-untyped-call]
                    google_credentials.refresh(auth_request)

                    # OpenAI Client
                    base_url = f"https://{credentials.region}-aiplatform.googleapis.com/v1beta1/projects/{google_credentials.project_id}/locations/{credentials.region}/endpoints/openapi"

                    google_client = openai.OpenAI(
                        base_url=base_url,
                        api_key=google_credentials.token,
                    )
                    google_client.chat.completions.create(
                        model=f"google/{lookup[llm.name]}",
                        messages=[{"role": "user", "content": "Why is the sky blue?"}],
                    )

                except Exception as e:
                    raise ValueError(
                        textwrap.dedent(
                            f"""
                            Unable to run a successful test completion against model '{lookup[llm.name]}' in region '{credentials.region}'
                            using base url '{base_url}'
                            with provided Google Cloud credentials.

                            Please validate your credentials or check {__file__} for details.
                            """
                        )
                    ) from e

    except pydantic.ValidationError as exc:
        msg = "Validation errors, please check that .env is correct. Remember to run `source set_env.sh` (or set_env.bat/Set-Env.ps1 on windows):\n\n"
        for error in exc.errors():
            msg += f"- Field '{error['loc'][0]}': {error['msg']}" + "\n"
        raise TypeError("Could not Validate LLM Credentials" + "\n" + msg) from exc
    return credentials


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
            credentials = SnowflakeCredentials()
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
                    project_root=PROJECT_ROOT
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
            credentials = GoogleCredentialsBQ()
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
            credentials = SAPDatasphereCredentials()
            if test_credentials:
                from hdbcli import dbapi

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
