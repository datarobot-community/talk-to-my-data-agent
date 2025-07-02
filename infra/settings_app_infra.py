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
import os
import re
import textwrap
from pathlib import Path
from typing import List, Sequence, Tuple

import pulumi_datarobot as datarobot
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME
from datarobot_pulumi_utils.schema.apps import ApplicationSourceArgs
from datarobot_pulumi_utils.schema.exec_envs import RuntimeEnvironments
from settings_database import DATABASE_CONNECTION_TYPE

from utils.credentials import SnowflakeCredentials

from .settings_main import PROJECT_ROOT

FRONTEND_PATHS = {
    "react": Path("app_backend"),
    "streamlit": Path("frontend"),
}


def get_frontend_path() -> Path:
    frontend_type = os.environ.get("FRONTEND_TYPE", "react")
    return FRONTEND_PATHS[frontend_type]


application_path = PROJECT_ROOT / get_frontend_path()

app_source_args = ApplicationSourceArgs(
    resource_name=f"Data Analyst App Source [{PROJECT_NAME}]",
    base_environment_id=RuntimeEnvironments.PYTHON_312_APPLICATION_BASE.value.id,
).model_dump(mode="json", exclude_none=True)

app_resource_name: str = f"Data Analyst Application [{PROJECT_NAME}]"


def _prep_metadata_yaml(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs
    ],
) -> None:
    from jinja2 import BaseLoader, Environment

    llm_runtime_parameter_specs = "\n".join(
        [
            textwrap.dedent(
                f"""\
            - fieldName: {param.key}
              type: {param.type}
        """
            )
            for param in runtime_parameter_values
        ]
    )
    with open(application_path / "metadata.yaml.jinja") as f:
        template = Environment(loader=BaseLoader()).from_string(f.read())
    (application_path / "metadata.yaml").write_text(
        template.render(
            additional_params=llm_runtime_parameter_specs,
        )
    )


def get_app_files(
    runtime_parameter_values: Sequence[
        datarobot.ApplicationSourceRuntimeParameterValueArgs
        | datarobot.CustomModelRuntimeParameterValueArgs,
    ],
) -> List[Tuple[str, str]]:
    _prep_metadata_yaml(runtime_parameter_values)
    # Get all files from application path
    source_files = [
        (f.as_posix(), f.relative_to(application_path).as_posix())
        for f in application_path.glob("**/*")
        if f.is_file() and ".yaml" not in f.name
    ]

    # Get all .py files from utils directory
    utils_files = [
        (str(PROJECT_ROOT / f"utils/{f.name}"), f"utils/{f.name}")
        for f in (PROJECT_ROOT / "utils").glob("*.py")
        if f.is_file()
    ]

    # Add the metadata.yaml file
    source_files.extend(utils_files)
    source_files.append(
        ((application_path / "metadata.yaml").as_posix(), "metadata.yaml")
    )

    if DATABASE_CONNECTION_TYPE == "snowflake":
        credentials = SnowflakeCredentials()
        if credentials.snowflake_key_path:
            # Add the snowflake connection file if it exists
            snowflake_file = PROJECT_ROOT / credentials.snowflake_key_path
            if snowflake_file.is_file():
                source_files.append(
                    (str(snowflake_file), credentials.snowflake_key_path)
                )

    EXCLUDE_PATTERNS = [re.compile(pattern) for pattern in [r".*\.pyc"]]
    source_files = [
        (file_path, file_name)
        for file_path, file_name in source_files
        if not any(
            exclude_pattern.match(file_name) for exclude_pattern in EXCLUDE_PATTERNS
        )
    ]

    return source_files
