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
from fastapi import Request

import app.middleware as middleware


@pytest.mark.asyncio
async def test_initialize_database_sets_provided_user_email_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.session = middleware.SessionState({"analyst_db": None})

    mock_analyst_db = AsyncMock()
    mock_analyst_db.get_user_email = AsyncMock(return_value=None)
    mock_analyst_db.set_user_email = AsyncMock()

    async def mock_get_database(user_id: str) -> AsyncMock:
        assert user_id == "user-123"
        return mock_analyst_db

    monkeypatch.setattr(middleware, "get_database", mock_get_database)

    await middleware._initialize_database(
        request, user_id="user-123", user_email="new_user@example.com"
    )

    assert request.state.session.analyst_db is mock_analyst_db
    mock_analyst_db.get_user_email.assert_awaited_once()
    mock_analyst_db.set_user_email.assert_awaited_once_with("new_user@example.com")
