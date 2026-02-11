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


import logging
import os
from typing import Any, cast

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.skipif(
    os.environ.get("FRONTEND_TYPE") == "react",
    reason="Skipping because FRONTEND_TYPE is 'react'",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def application(setup_pulumi_stack: Any) -> Any:
    """Uses the shared setup_pulumi_stack fixture and returns an AppTest instance"""
    yield AppTest.from_file("app/01_connect_and_explore.py", default_timeout=180)


@pytest.fixture
def app_post_database_load(application: AppTest, pulumi_up: Any) -> AppTest:
    logger.info(os.getcwd())
    at = application.run()
    expander = next(i for i in at.expander if i.label.lower() == "database")
    # Test assumes access to a database with LENDING_CLUB_PROFILE table
    expander.multiselect[0].select("LENDING_CLUB_PROFILE")
    expander.button[0].click().run(timeout=120)
    return at


@pytest.mark.integration
def test_database_queried(app_post_database_load: AppTest) -> None:
    success_message = "Processing complete"
    assert app_post_database_load.toast[-1].value == success_message


@pytest.mark.integration
def test_data_dictionary_generated(
    app_post_database_load: AppTest,
) -> None:
    names = cast(list[str], app_post_database_load.session_state.datasets_names)
    dictionaries = []

    for name in names:
        dictionaries.append(
            app_post_database_load.session_state.analyst_db.get_data_dictionary(name)
        )
    for dictionary in dictionaries:
        column_decriptions = dictionary.column_descriptions
        logger.info(column_decriptions)
        assert len(column_decriptions)
        assert "No description available" not in [
            c.description for c in column_decriptions
        ]
