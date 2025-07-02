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

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import AliasChoices, AliasPath, Field
from pydantic_settings import BaseSettings


class DRCredentials(BaseSettings): ...


class AzureOpenAICredentials(DRCredentials):
    """LLM credentials auto-constructed using environment variables."""

    api_key: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_KEY",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_KEY", "payload", "apiToken"),
        ),
    )
    azure_endpoint: str = Field(
        validation_alias=AliasChoices(
            "OPENAI_API_BASE",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_BASE", "payload"),
        )
    )
    api_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_VERSION",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_VERSION", "payload"),
        ),
    )
    azure_deployment: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENAI_API_DEPLOYMENT_ID",
            AliasPath("MLOPS_RUNTIME_PARAM_OPENAI_API_DEPLOYMENT_ID", "payload"),
        ),
    )


class GoogleCredentials(DRCredentials):
    service_account_key: Dict[str, Any] = Field(
        validation_alias=AliasChoices(
            "GOOGLE_SERVICE_ACCOUNT",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_GOOGLE_SERVICE_ACCOUNT", "payload", "gcpKey"
            ),
        )
    )
    region: Optional[str] = Field(default="us-west1", validation_alias="GOOGLE_REGION")


class GoogleCredentialsBQ(DRCredentials):
    service_account_key: Dict[str, Any] = Field(
        validation_alias=AliasChoices(
            "GOOGLE_SERVICE_ACCOUNT_BQ",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_GOOGLE_SERVICE_ACCOUNT_BQ", "payload", "gcpKey"
            ),
        )
    )
    region: Optional[str] = Field(
        default="us-west1", validation_alias="GOOGLE_REGION_BQ"
    )
    db_schema: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "GOOGLE_DB_SCHEMA_BQ", AliasPath("MLOPS_RUNTIME_PARAM_GOOGLE_DB_SCHEMA_BQ")
        ),
    )


class AWSBedrockCredentials(DRCredentials):
    aws_access_key_id: str = Field(
        validation_alias=AliasChoices(
            "AWS_ACCESS_KEY_ID",
            AliasPath("MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsAccessKeyId"),
        )
    )
    aws_secret_access_key: str = Field(
        validation_alias=AliasChoices(
            "AWS_SECRET_ACCESS_KEY",
            AliasPath(
                "MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsSecretAccessKey"
            ),
        )
    )
    aws_session_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "AWS_SESSION_TOKEN",
            AliasPath("MLOPS_RUNTIME_PARAM_AWS_ACCOUNT", "payload", "awsSessionToken"),
        ),
    )
    region_name: Optional[str] = Field(default=None, validation_alias="AWS_REGION")


class SnowflakeCredentials(DRCredentials):
    """Snowflake Connection credentials auto-constructed using environment variables."""

    user: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "username"),
            "MLOPS_RUNTIME_PARAM_SNOWFLAKE_USER",
            "SNOWFLAKE_USER",
        ),
    )
    password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "password"),
            "SNOWFLAKE_PASSWORD",
        ),
    )
    account: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_ACCOUNT",
        ),
    )
    database: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_DATABASE"),
            "SNOWFLAKE_DATABASE",
        ),
    )
    warehouse: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_WAREHOUSE"),
            "SNOWFLAKE_WAREHOUSE",
        ),
    )
    db_schema: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_SCHEMA"),
            "SNOWFLAKE_SCHEMA",
        ),
    )
    role: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ROLE"),
            "SNOWFLAKE_ROLE",
        ),
    )
    snowflake_key_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_KEY_PATH"), "SNOWFLAKE_KEY_PATH"
        ),
    )

    def get_private_key(self, project_root: Path | None = None) -> bytes | None:
        """Get private key for Snowflake authentication if configured."""
        logger = logging.getLogger(__name__)

        key_path = self.snowflake_key_path
        if not key_path:
            return None

        try:
            if project_root:
                key_path = os.path.join(project_root, key_path)
            else:
                key_path = os.path.abspath(key_path)
            if not os.path.exists(key_path):
                logger.warning(f"Snowflake key file not found at {key_path}")
                return None

            # Read and process private key
            with open(key_path, "rb") as key_file:
                private_key_data = key_file.read()
                logger.info("Successfully read private key file")

            # Load and convert key
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            p_key = serialization.load_pem_private_key(
                private_key_data, password=None, backend=default_backend()
            )
            logger.info("Successfully loaded PEM key")

            private_key = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            logger.info("Successfully converted key to DER format")
            return private_key

        except Exception as e:
            logger.warning(f"Failed to process private key: {str(e)}")
            return None

    def is_configured(self) -> bool:
        """Check if Snowflake is properly configured with either key or password auth."""
        has_basic_config = all(
            [
                self.user,
                self.account,
                self.warehouse,
                self.database,
                self.db_schema,
                self.role,
            ]
        )
        if not has_basic_config:
            return False

        # Check if we have either key file or password authentication
        has_key_auth = self.snowflake_key_path is not None
        has_password_auth = self.password is not None

        return has_key_auth or has_password_auth


class SAPDatasphereCredentials(DRCredentials):
    """
    Credentials for connecting to SAP Data Sphere.
    """

    host: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_HOST"),
            "SAP_DATASPHERE_HOST",
        ),
    )
    port: int = Field(
        default=443,
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_PORT"),
            "SAP_DATASPHERE_PORT",
        ),
    )
    user: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "username"),
            "SAP_DATASPHERE_USER",
        ),
    )
    password: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "password"),
            "SAP_DATASPHERE_PASSWORD",
        ),
    )
    db_schema: str = Field(
        validation_alias=AliasChoices(
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_SCHEMA"),
            "SAP_DATASPHERE_SCHEMA",
        ),
    )

    def is_configured(self) -> bool:
        """
        Check if SAP Data Sphere credentials are properly configured.
        """
        return bool(self.host and self.port and self.user and self.password)


class NoDatabaseCredentials(DRCredentials):
    pass
