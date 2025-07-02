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

import json
import logging
import os
import time
from typing import Callable, List, Optional, ParamSpec, TypeVar

import datarobot as dr

from utils.datarobot_client import File, KeyValue, KeyValueEntityType

Param = ParamSpec("Param")
ReturnType = TypeVar("ReturnType")

logger = logging.getLogger(__name__)


def _use_owner_creds(func: Callable[Param, ReturnType]) -> Callable[Param, ReturnType]:
    def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> ReturnType:
        with dr.Client(
            token=os.environ.get("DATAROBOT_API_TOKEN"),
            endpoint=os.environ.get("DATAROBOT_ENDPOINT"),
        ):
            return func(*args, **kwargs)

    return wrapper


class PersistentStorage:
    def __init__(self, user_id: Optional[str]):
        self.app_id: str = os.environ.get("APPLICATION_ID")  # type: ignore[assignment]
        if not self.app_id:
            raise ValueError("APPLICATION_ID env variable is not set.")
        self.name_prefix = f"{user_id}_"

    @_use_owner_creds
    def files(self) -> List[str]:
        stored_values = KeyValue.list(
            self.app_id, KeyValueEntityType.CUSTOM_APPLICATION
        )
        return [
            v.name.removeprefix(self.name_prefix)
            for v in stored_values
            if v.value_type == dr.KeyValueType.JSON
            and v.category == dr.KeyValueCategory.ARTIFACT
            and v.name.startswith(self.name_prefix)
        ]

    @_use_owner_creds
    def fetch_from_storage(self, file_name: str, local_path: str) -> None:
        logger.info(f"Fetching file {file_name} from storage")
        storage_link = KeyValue.find(
            self.app_id,
            KeyValueEntityType.CUSTOM_APPLICATION,
            f"{self.name_prefix}{file_name}",
        )
        if not storage_link:
            return
        data = json.loads(storage_link.value)
        File.file(data["catalogId"], local_path)

    @_use_owner_creds
    def save_to_storage(self, file_name: str, local_path: str) -> None:
        logger.info(f"Storing file {file_name} to persistent storage")
        storing_label = f"{self.name_prefix}{file_name}"
        timestamp = time.time_ns()

        catalog_info = File.from_file(local_path)
        file_data = {**catalog_info, "timestamp": timestamp}

        storage_link = KeyValue.find(
            self.app_id, KeyValueEntityType.CUSTOM_APPLICATION, storing_label
        )
        data = json.loads(storage_link.value) if storage_link else {}
        if not data:
            # there is no previous version of this file
            KeyValue.create(
                entity_id=self.app_id,
                entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
                name=storing_label,
                category=dr.KeyValueCategory.ARTIFACT,
                value_type=dr.KeyValueType.JSON,
                value=json.dumps(file_data),
            )
            return

        if timestamp > data["timestamp"]:
            # uploaded file is newer version and storage link needs to be updated and old file to be removed
            storage_link.update(value=json.dumps(file_data))  # type: ignore[union-attr]
            File.delete(data["catalogId"])
        else:
            # there is a newer file in storage and we drop just uploaded one
            File.delete(catalog_info["catalogId"])

    @_use_owner_creds
    def delete_file(self, file_name: str) -> None:
        storage_link = KeyValue.find(
            self.app_id,
            KeyValueEntityType.CUSTOM_APPLICATION,
            f"{self.name_prefix}{file_name}",
        )
        if not storage_link:
            return
        data = json.loads(storage_link.value)
        File.delete(data["catalogId"])
        storage_link.delete()
