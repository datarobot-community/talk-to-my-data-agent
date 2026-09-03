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
from unittest.mock import AsyncMock, MagicMock

import pytest
from core.code_execution import DatabaseFailure
from core.data_connections.datarobot.helpers import OUTAGE_MESSAGE
from fastapi.testclient import TestClient

import app.routers.database as database_router


def _patch_operator(monkeypatch: pytest.MonkeyPatch, get_tables: AsyncMock) -> None:
    operator = MagicMock()
    operator.get_tables = get_tables
    monkeypatch.setattr(database_router, "get_external_database", lambda: operator)


def test_get_database_tables_returns_503_on_outage(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_operator(monkeypatch, AsyncMock(side_effect=DatabaseFailure(OUTAGE_MESSAGE)))

    response = client.get("/api/v1/database/tables")

    assert response.status_code == 503
    assert response.json()["detail"] == OUTAGE_MESSAGE


def test_get_database_tables_returns_empty_list_when_no_tables(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_operator(monkeypatch, AsyncMock(return_value=[]))

    response = client.get("/api/v1/database/tables")

    assert response.status_code == 200
    assert response.json() == []
