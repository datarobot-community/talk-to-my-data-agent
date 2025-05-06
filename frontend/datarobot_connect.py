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

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, Optional, cast

import datarobot as dr
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_javascript import st_javascript


@dataclass
class DataRobotCredentials:
    """Container for DataRobot credentials."""

    token: Optional[str] = None
    endpoint: Optional[str] = None


class DataRobotTokenManager:
    """Manages DataRobot API tokens in a Streamlit environment."""

    _API_URLS = {
        "account": "/api/v2/account/info/",
        "apikeys": "/api/v2/account/apiKeys/",
    }

    _JS_COMMAND_TEMPLATE = """fetch(
        "URL"
    ).then((response) => {
        if (response.ok) {
            return response.text();
        } else {
            return response.status + " : " + response.statusText;
        }
    }).then((data) => {
        return data;
    });"""

    def __init__(self) -> None:
        """Initialize the token manager and set up initial credentials."""
        self._original_creds = self._get_current_credentials()
        self._set_user_credentials()
        self._set_user_info()

    def _get_current_credentials(self) -> DataRobotCredentials:
        """Get the current DataRobot credentials from the client."""
        client = dr.Client()
        return DataRobotCredentials(token=client.token, endpoint=client.endpoint)

    def _get_contents_from_url(self, url: str) -> dict[str, Any]:
        """Fetch data from DataRobot API using JavaScript."""
        js_command = self._JS_COMMAND_TEMPLATE.replace("URL", url)
        result = st_javascript(js_command)
        data = {}
        try:
            data = json.loads(result)
            data = cast(dict[str, Any], data)
            data["info_ok"] = True
        except Exception:
            data = {"info_ok": False, "reply": str(result)}
        return data

    def _set_user_credentials(self) -> None:
        """Set user-specific DataRobot credentials."""
        if (
            "datarobot_api_token_provided" in st.session_state
            and st.session_state.datarobot_api_token_provided
        ):
            self._user_creds = DataRobotCredentials(
                token=st.session_state.datarobot_api_token_provided,
                endpoint=self._original_creds.endpoint,
            )
            return
        if not os.environ.get("DR_CUSTOM_APP_EXTERNAL_URL"):
            self._user_creds = self._original_creds
            return

        # Fetch API keys
        apikeys_data = self._get_contents_from_url(self._API_URLS["apikeys"])

        api_token = None
        if "data" in apikeys_data:
            # Find first non-expiring key
            keys = [
                key["key"] for key in apikeys_data["data"] if key["expireAt"] is None
            ]
            if len(keys) > 0:
                api_token = keys[0]

        self._user_creds = DataRobotCredentials(
            token=api_token, endpoint=os.environ.get("DATAROBOT_ENDPOINT")
        )

        # Store in session state for compatibility
        st.session_state["datarobot_api_token"] = api_token
        st.session_state["datarobot_api_endpoint"] = os.environ.get(
            "DATAROBOT_ENDPOINT"
        )

    def _set_user_info(self) -> None:
        """Set user information in session state."""
        if self._user_creds.token:
            with dr.Client(
                endpoint=self._user_creds.endpoint,
                token=self._user_creds.token,
            ) as client:
                reply = client.get("account/info/")
                user_info = reply.json()
                user_info["info_ok"] = reply.ok
        else:
            user_info = self._get_contents_from_url(self._API_URLS["account"])

        if "uid" in user_info:
            for k, v in user_info.items():
                st.session_state[f"datarobot_{k}"] = v

    def display_info(self, stc: DeltaGenerator) -> None:
        # stc.subheader("DataRobot Connect", divider="rainbow")
        first_name = st.session_state.get("datarobot_firstName", "") or ""
        last_name = st.session_state.get("datarobot_lastName", "") or ""
        username = (first_name + " " + last_name).strip()
        if len(username) > 0:
            stc.write(f"Hello {username}")

        if not self._user_creds.token:
            stc.text_input(
                "API Token",
                key="datarobot_api_token_provided",
                type="password",
                placeholder="DataRobot API Token",
                label_visibility="collapsed",
            )

    @contextmanager
    def use_user_token(self) -> Generator[None, None, None]:
        """Context manager to temporarily use the user's DataRobot token."""
        if not self._user_creds.token:
            yield
            return

        with dr.Client(
            token=self._user_creds.token, endpoint=self._user_creds.endpoint
        ):
            yield

    @contextmanager
    def use_app_token(self) -> Generator[None, None, None]:
        """Context manager to temporarily use the app's DataRobot token."""
        with dr.Client(
            token=self._original_creds.token, endpoint=self._original_creds.endpoint
        ):
            yield

    @property
    def has_valid_user_token(self) -> bool:
        """Check if there's a valid user token available."""
        return bool(self._user_creds.token and self._user_creds.endpoint)
