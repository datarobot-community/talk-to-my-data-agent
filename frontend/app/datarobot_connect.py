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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import datarobot as dr
import streamlit as st
from core.logging_helper import get_logger
from datarobot.rest import RESTClientObject
from streamlit.delta_generator import DeltaGenerator

from app.helpers import state_init

logger = get_logger("DR Connect")


class DataRobotTokenManager:
    """Manages DataRobot API tokens in a Streamlit environment."""

    _API_URLS = {
        "account": "/api/v2/account/info/",
    }

    def __init__(self) -> None:
        logger.info("dr_connect_init")
        """Initialize the token manager."""
        self.provided_user_token: Optional[str] = None
        self.user_info = self._get_user_info()

    @contextmanager
    def use_user_token(self) -> Iterator[Optional[RESTClientObject]]:
        """Context manager to temporarily use the user's DataRobot token."""
        user_token = self._get_user_token()
        if not user_token:
            yield None
            return

        with dr.Client(token=user_token, endpoint=self._get_dr_endpoint()) as client:
            yield client

    @contextmanager
    def use_app_token(self) -> Iterator[RESTClientObject]:
        """Context manager to temporarily use the app's DataRobot token."""
        app_token = os.environ.get("DATAROBOT_API_TOKEN")
        with dr.Client(token=app_token, endpoint=self._get_dr_endpoint()) as client:
            yield client

    async def display_info(self, stc: DeltaGenerator) -> None:
        if self.user_info:
            stc.write(f"Hello {self.user_info.username}")
            return

        if self._get_user_token():
            self.user_info = self._get_user_info()
            if self.user_info:
                stc.write(f"Hello {self.user_info.username}")
                return

        stc.warning("Data Registry disabled. Please provide your API token")
        token = stc.text_input(
            "API Token",
            key="datarobot_api_token_provided",
            type="password",
            placeholder="DataRobot API Token",
            label_visibility="collapsed",
        )

        if token:
            logger.info(f"Setting Token {token}")
            self.provided_user_token = token
            self.user_info = self._get_user_info()
            await state_init()
            st.rerun()

    @staticmethod
    def _get_dr_endpoint() -> str:
        # It should be done during dr.Client creation automatically,
        # but we want to have an explicit error message.
        endpoint = os.environ.get("DATAROBOT_ENDPOINT")
        if not endpoint:
            raise ValueError("DATAROBOT_ENDPOINT env variable is not set.")
        return endpoint

    def _get_user_token(self) -> Optional[str]:
        return self.provided_user_token or st.context.headers.get("x-datarobot-api-key")

    def _get_user_info(self) -> Optional["UserInfo"]:
        """Set user information in session state."""
        with self.use_user_token() as client:
            if not client:
                return None

            response = client.get("account/info/")

            if not response.ok:
                logger.warning(
                    f"Cannot get user info from {response.url}",
                    extra={"code": response.status_code, "message": response.text},
                )
                return None

            return UserInfo.from_api(response.json())


@dataclass
class UserInfo:
    first_name: str
    last_name: str | None

    @property
    def username(self) -> str:
        return (self.first_name + " " + str(self.last_name or "")).strip()

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "UserInfo":
        return cls(
            first_name=data["firstName"],
            last_name=data.get("lastName"),
        )
