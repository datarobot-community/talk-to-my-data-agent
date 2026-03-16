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
from __future__ import annotations

import logging
import uuid
from typing import (
    Any,
    Literal,
    cast,
)

import datarobot as dr
import pandas as pd
import requests
from pydantic import (
    BaseModel,
    Field,
)

from core.data_connections.database.datastore.preview import (
    DatasetPreview,
)
from core.data_connections.database.datastore.utils import (
    build_dataset_preview,
)

logger = logging.getLogger(__name__)


class DataRobotDatastoreConnection(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:5])
    type: Literal["datarobot_datastore"] = "datarobot_datastore"
    use_case_id: str
    datastore_id: str
    dataset_name: str
    credential_id: str | None = None

    def _verify_sql_internal(self, query: str, max_rows: int = 100) -> dict[str, Any]:
        """Internal method to verify SQL and return response from verifySQL endpoint.

        Args:
            query: The SQL query to verify.
            max_rows: Maximum number of rows for verification.

        Returns:
            dict: Response from verifySQL endpoint.
        """
        dr_client = dr.client.get_client()
        logger.info(f"Verifying SQL for datastore {self.datastore_id}")

        payload = {
            "query": query,
            "maxRows": max_rows,
        }
        if self.credential_id:
            payload["credentialId"] = self.credential_id

        res = dr_client.post(
            f"externalDataStores/{self.datastore_id}/verifySQL/", json=payload
        )

        self.raise_for_error(res)
        return cast(dict[str, Any], res.json())

    def run_preview(self, query: str, sample_size: int = 10) -> DatasetPreview:
        """Verify SQL query and return preview metadata using verifySQL endpoint.

        Args:
            query: The SQL query to verify.
            sample_size: Maximum number of sample rows to return.

        Returns:
            DatasetPreview: Structured preview describing schema + sampled rows.
        """

        user_query = query
        limited_query = f"SELECT * FROM ({query}) LIMIT {sample_size}"
        res = self._verify_sql_internal(limited_query, max_rows=sample_size)

        rows: list[dict[str, Any]] = []
        column_types: dict[str, str | None] = {}

        if "records" in res:
            columns = res.get("columns", [])
            df = pd.DataFrame(data=res["records"], columns=columns)
            rows = cast(list[dict[str, Any]], df.to_dict("records"))
            column_types = {str(col): None for col in columns}

        rows_total = len(rows) if len(rows) < sample_size else None
        notes = [f"Preview limited to {sample_size} row(s)."]
        if rows_total is not None:
            notes.append(
                "Query returned fewer rows than the applied limit; this is the full result."
            )
        else:
            notes.append(
                "Result may be truncated; rerun without preview to retrieve all rows."
            )

        return build_dataset_preview(
            source_kind="datastore_query",
            source_id=self.datastore_id,
            title=self.dataset_name,
            description=f"Preview of SQL executed against datastore {self.datastore_id}",
            sample_rows=rows,
            rows_total=rows_total,
            sample_size=sample_size,
            column_types=column_types,
            notes=notes,
            extra={
                "applied_limit": sample_size,
                "datastore_id": self.datastore_id,
                "query": user_query,
            },
        )

    @staticmethod
    def raise_for_error(res: requests.Response) -> None:
        if res.status_code != 200:
            res.raise_for_status()
