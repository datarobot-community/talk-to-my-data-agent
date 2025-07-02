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
from enum import Enum
from typing import Any, Dict, List, Optional, Union, cast
from urllib.parse import urljoin

import datarobot as dr
import trafaret as t

FILE_API_CONNECT_TIMEOUT = os.environ.get("FILE_API_CONNECT_TIMEOUT", 180)
FILE_API_READ_TIMEOUT = os.environ.get("FILE_API_READ_TIMEOUT", 180)


class KeyValueEntityType(Enum):
    DEPLOYMENT = "deployment"
    MODEL_PACKAGE = "modelPackage"
    REGISTERED_MODEL = "registeredModel"
    CUSTOM_JOB = "customJob"
    CUSTOM_JOB_RUN = "customJobRun"

    CUSTOM_APPLICATION_SOURCE_VERSION = "customApplicationSourceVersion"
    CUSTOM_APPLICATION = "customApplication"


class KeyValue(dr.KeyValue):
    _converter = t.Dict(
        {
            t.Key("id"): dr._compat.String(),
            t.Key("created_at"): dr._compat.String(),
            t.Key("entity_id"): dr._compat.String(),
            t.Key("entity_type"): t.Enum(*[e.value for e in KeyValueEntityType]),
            t.Key("name"): dr._compat.String(),
            t.Key("value"): dr._compat.String(),
            t.Key("numeric_value"): t.Float(),
            t.Key("boolean_value", optional=True, default=False): t.Bool(),
            t.Key("value_type"): t.Enum(*[e.value for e in dr.enums.KeyValueType]),
            t.Key("description"): dr._compat.String(allow_blank=True),
            t.Key("creator_id"): dr._compat.String(),
            t.Key("creator_name"): dr._compat.String(),
            t.Key("category"): t.Enum(*[e.value for e in dr.enums.KeyValueCategory]),
            t.Key("artifact_size"): t.Int(),
            t.Key("original_file_name"): dr._compat.String(allow_blank=True),
            t.Key("is_editable"): t.Bool(),
            t.Key("is_dataset_missing"): t.Bool(),
            t.Key("error_message"): dr._compat.String(allow_blank=True),
        }
    ).ignore_extra("*")

    schema = _converter

    def __init__(
        self,
        id: str,
        created_at: str,
        entity_id: str,
        entity_type: KeyValueEntityType,
        name: str,
        value: str,
        numeric_value: float,
        boolean_value: bool,
        value_type: dr.enums.KeyValueType,
        description: str,
        creator_id: str,
        creator_name: str,
        category: dr.enums.KeyValueCategory,
        artifact_size: int,
        original_file_name: str,
        is_editable: bool,
        is_dataset_missing: bool,
        error_message: str,
    ) -> None:
        self.id = id
        self.created_at = created_at
        self.entity_id = entity_id
        self.entity_type = KeyValueEntityType(entity_type)  # type: ignore[assignment]
        self.name = name
        self.value = value
        self.numeric_value = numeric_value
        self.boolean_value = boolean_value
        self.value_type = dr.enums.KeyValueType(value_type)
        self.description = description
        self.creator_id = creator_id
        self.creator_name = creator_name
        self.category = dr.enums.KeyValueCategory(category)
        self.artifact_size = artifact_size
        self.original_file_name = original_file_name
        self.is_editable = is_editable
        self.is_dataset_missing = is_dataset_missing
        self.error_message = error_message

    @classmethod
    def create(
        cls,
        entity_id: str,
        entity_type: KeyValueEntityType,  # type: ignore[override]
        name: str,
        category: dr.enums.KeyValueCategory,
        value_type: dr.enums.KeyValueType,
        value: Optional[Union[str, float, bool]] = None,
        description: Optional[str] = None,
    ) -> "KeyValue":
        return super().create(
            entity_id,
            entity_type,  # type: ignore[arg-type]
            name,
            category,
            value_type,
            value,
            description,
        )  # type: ignore[return-value]

    @classmethod
    def list(cls, entity_id: str, entity_type: KeyValueEntityType) -> List["KeyValue"]:  # type: ignore[override]
        return super().list(entity_id, entity_type)  # type: ignore[arg-type, return-value]

    @classmethod
    def find(
        cls,
        entity_id: str,
        entity_type: KeyValueEntityType,  # type: ignore[override]
        name: str,
    ) -> Optional["KeyValue"]:
        return super().find(entity_id, entity_type, name)  # type: ignore[arg-type, return-value]


class File:
    _client = dr.client.staticproperty(dr.client.get_client)  # type: ignore[arg-type]
    _path = "files/"

    @classmethod
    def from_file(cls, file_path: str) -> Dict[str, Any]:
        with open(file_path, "rb") as read_file:
            response = cls._client.post(
                urljoin(cls._path, "fromFile/"),
                files={"file": read_file},
                data={"useArchiveContents": "true"},
                timeout=(FILE_API_CONNECT_TIMEOUT, FILE_API_READ_TIMEOUT),
            )
        return cast(Dict[str, Any], response.json())

    @classmethod
    def file(cls, catalog_id: str, save_path: str) -> None:
        response = cls._client.get(
            urljoin(cls._path, f"{catalog_id}/file/"),
            timeout=(FILE_API_CONNECT_TIMEOUT, FILE_API_READ_TIMEOUT),
        )
        with open(save_path, "wb") as write_file:
            write_file.write(response.content)

    @classmethod
    def delete(cls, catalog_id: str) -> None:
        cls._client.delete(
            urljoin(cls._path, f"{catalog_id}/"),
        )
