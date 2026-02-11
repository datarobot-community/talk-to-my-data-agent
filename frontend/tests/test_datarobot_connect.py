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
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from app.datarobot_connect import DataRobotTokenManager, UserInfo


@pytest.fixture
def mock_dr_client() -> Generator[MagicMock, None, None]:
    """Mock DataRobot client for testing."""
    with patch("datarobot.Client") as mock_client:
        client_instance = MagicMock(name="dr_client_instance")
        client_instance.token = "test-token"
        client_instance.endpoint = "https://test-endpoint.datarobot.com"
        client_instance.get = MagicMock(
            return_value=MagicMock(
                ok=True,
                json=MagicMock(
                    return_value={"firstName": "Test", "lastName": "User", "uid": "123"}
                ),
            )
        )
        client_instance.__enter__.return_value = client_instance
        client_instance.__exit__.return_value = None

        mock_client.return_value = client_instance

        yield mock_client


@pytest.fixture
def mock_dr_endpoint() -> Generator[MagicMock, None, None]:
    """Mock DataRobot endpoint in env variables for testing."""
    with patch.dict(
        os.environ, {"DATAROBOT_ENDPOINT": "https://test-endpoint.datarobot.com"}
    ) as mock_environ:
        yield mock_environ


@pytest.mark.asyncio
async def test_display_info_with_user() -> None:
    """Test the display_info method with a user."""
    token_manager = DataRobotTokenManager()
    token_manager.user_info = UserInfo(first_name="John", last_name="Doe")

    mock_stc = MagicMock()

    await token_manager.display_info(mock_stc)

    mock_stc.write.assert_called_once_with("Hello John Doe")
    mock_stc.text_input.assert_not_called()


@pytest.mark.asyncio
async def test_display_info_without_last_name() -> None:
    """Test the display_info method without a last name."""
    token_manager = DataRobotTokenManager()
    token_manager.user_info = UserInfo(first_name="John", last_name=None)

    mock_stc = MagicMock()

    await token_manager.display_info(mock_stc)

    mock_stc.write.assert_called_once_with("Hello John")
    mock_stc.text_input.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_dr_endpoint")
@pytest.mark.parametrize(
    "entered_token", (False, True), ids=("header-token", "manual-token")
)
async def test_display_info_fetched_from_dr(
    mock_dr_client: MagicMock, mock_dr_endpoint: MagicMock, entered_token: bool
) -> None:
    token_manager = DataRobotTokenManager()
    if entered_token:
        token_manager.provided_user_token = "user-test-key"

    mock_stc = MagicMock()

    mock_context = MagicMock()
    mock_context.headers = {"x-datarobot-api-key": "header-test-key"}

    with patch("streamlit.context", mock_context):
        await token_manager.display_info(mock_stc)

    mock_stc.write.assert_called_once_with("Hello Test User")
    mock_stc.text_input.assert_not_called()

    expected_token = "user-test-key" if entered_token else "header-test-key"
    expected_endpoint = mock_dr_endpoint["DATAROBOT_ENDPOINT"]
    mock_dr_client.assert_called_once_with(
        token=expected_token, endpoint=expected_endpoint
    )
