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

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import AliasChoices, AliasPath, Field
from pydantic_settings import BaseSettings


class DRCredentials(BaseSettings): ...


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


class SnowflakeCredentials(DRCredentials):
    """Snowflake Connection credentials auto-constructed using environment variables."""

    user: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "MLOPS_RUNTIME_PARAM_SNOWFLAKE_USER",
            "SNOWFLAKE_USER",
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "username"),
        ),
    )
    password: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "SNOWFLAKE_PASSWORD",
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "password"),
        ),
    )
    account: str = Field(
        validation_alias=AliasChoices(
            "SNOWFLAKE_ACCOUNT",
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ACCOUNT"),
        ),
    )
    database: str = Field(
        validation_alias=AliasChoices(
            "SNOWFLAKE_DATABASE",
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_DATABASE"),
        ),
    )
    warehouse: str = Field(
        validation_alias=AliasChoices(
            "SNOWFLAKE_WAREHOUSE",
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_WAREHOUSE"),
        ),
    )
    db_schema: str = Field(
        validation_alias=AliasChoices(
            "SNOWFLAKE_SCHEMA",
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_SCHEMA"),
        ),
    )
    role: str = Field(
        validation_alias=AliasChoices(
            "SNOWFLAKE_ROLE",
            AliasPath("MLOPS_RUNTIME_PARAM_SNOWFLAKE_ROLE"),
        )
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

        if project_root is None:
            project_root = Path(__file__).resolve().parents[1]

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
            "SAP_DATASPHERE_HOST",
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_HOST"),
        ),
    )
    port: int = Field(
        default=443,
        validation_alias=AliasChoices(
            "SAP_DATASPHERE_PORT",
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_PORT"),
        ),
    )
    user: str = Field(
        validation_alias=AliasChoices(
            "SAP_DATASPHERE_USER",
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "username"),
        ),
    )
    password: str = Field(
        validation_alias=AliasChoices(
            "SAP_DATASPHERE_PASSWORD",
            AliasPath("MLOPS_RUNTIME_PARAM_db_credential", "payload", "password"),
        ),
    )
    db_schema: str = Field(
        validation_alias=AliasChoices(
            "SAP_DATASPHERE_SCHEMA",
            AliasPath("MLOPS_RUNTIME_PARAM_SAP_DATASPHERE_SCHEMA"),
        ),
    )

    def is_configured(self) -> bool:
        """
        Check if SAP Data Sphere credentials are properly configured.
        """
        return bool(self.host and self.port and self.user and self.password)


class NoDatabaseCredentials(DRCredentials):
    pass
