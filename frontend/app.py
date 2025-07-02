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

import asyncio
from typing import List

import streamlit as st
from app_settings import PAGE_ICON, apply_custom_css
from datarobot_connect import DataRobotTokenManager
from streamlit.navigation.page import StreamlitPage

pages: List[StreamlitPage] = [
    st.Page("01_connect_and_explore.py", title="Connect & Explore"),
    st.Page("02_chat_with_data.py", title="AI Data Analyst"),
]

st.set_page_config(
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_css()

if "datarobot_connect" not in st.session_state:
    datarobot_connect = DataRobotTokenManager()
    st.session_state.datarobot_connect = datarobot_connect


asyncio.run(
    st.session_state.datarobot_connect.display_info(
        st.sidebar.container(key="user_info")
    )
)

pg = st.navigation(pages)
pg.run()
