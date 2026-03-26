# Copyright 2026 DataRobot, Inc.
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
"""
Module stores Dataset Preview representation models.
"""

from typing import Any

from pydantic import BaseModel, Field


class DatasetPreviewColumn(BaseModel):
    name: str
    data_type: str | None = None
    nullable: bool | None = None
    description: str | None = None


class DatasetPreview(BaseModel):
    source_kind: str
    source_id: str | None = None
    title: str | None = None
    description: str | None = None
    rows_previewed: int
    rows_total: int | None = None
    is_complete: bool
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    columns: list[DatasetPreviewColumn] = Field(default_factory=list)
    column_markdown: str
    human_summary: str
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] | None = None
