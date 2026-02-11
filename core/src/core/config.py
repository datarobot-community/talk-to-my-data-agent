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


from datarobot.core.config import DataRobotAppFrameworkBaseSettings

from .schema import DatabaseConnectionType


class Config(DataRobotAppFrameworkBaseSettings):
    datarobot_endpoint: str
    datarobot_api_token: str

    llm_deployment_id: str | None = None
    use_datarobot_llm_gateway: bool = False
    llm_default_model: str = "custom-model"

    database_connection_type: DatabaseConnectionType = "no_database"
