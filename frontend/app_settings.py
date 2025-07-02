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
import os
import sys

import streamlit as st
from streamlit_theme import st_theme

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils.schema import AppInfra

PAGE_ICON = "./datarobot_favicon.png"


def display_page_logo() -> None:
    theme = st_theme()
    # logo placeholder used for initial load
    logo = '<svg width="133" height="20" xmlns="http://www.w3.org/2000/svg" id="datarobot-logo"></svg>'
    if theme:
        if theme.get("base") == "light":
            logo = "./DataRobot_black.svg"
        else:
            logo = "./DataRobot_white.svg"
    with st.container(key="datarobot-logo"):
        st.image(logo, width=200)


def get_database_logo(app_infra: AppInfra) -> None:
    if app_infra.database == "snowflake":
        st.image("./Snowflake.svg", width=100)
    elif app_infra.database == "bigquery":
        st.image("./Google_Cloud.svg", width=100)
    elif app_infra.database == "sap":
        st.image("./sap.svg", width=100)
    return None


def get_database_loader_message(app_infra: AppInfra) -> str:
    if app_infra.database == "snowflake":
        return "Load Datasets from Snowflake"
    elif app_infra.database == "bigquery":
        return "Load Datasets from BigQuery"
    elif app_infra.database == "sap":
        return "Load Datasets from SAP"
    return "No database available"


def apply_custom_css() -> None:
    with open("./style.css") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
