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


# mypy: ignore-errors

import contextlib
import json
import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

# Set env vars before any imports that might need them
os.environ.setdefault("DATAROBOT_ENDPOINT", "https://dummy-endpoint.datarobot.com")
os.environ.setdefault("DATAROBOT_API_TOKEN", "dummy-api-token-12345")
os.environ.setdefault("DATAROBOT_API_BASE", "https://dummy-endpoint.datarobot.com")
os.environ.setdefault("APPLICATION_ID", "dummy-application-id")

import datarobot as dr
import pandas as pd
import pytest
import pytest_asyncio
from dotenv import dotenv_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://dummy-endpoint.datarobot.com")
    monkeypatch.setenv("DATAROBOT_API_TOKEN", "dummy-api-token-12345")
    monkeypatch.setenv("DATAROBOT_API_BASE", "https://dummy-endpoint.datarobot.com")
    monkeypatch.setenv("APPLICATION_ID", "dummy-application-id")


@pytest.fixture
def mock_env_endpoint_and_token(monkeypatch):
    monkeypatch.setenv("DATAROBOT_ENDPOINT", "https://app.datarobot.com/api/v2")
    monkeypatch.setenv("APPLICATION_ID", "app")


@pytest.fixture(scope="session")
def stack_name():
    short_uuid = str(uuid.uuid4())[:5]
    return f"test-stack-{short_uuid}"


@pytest.fixture(scope="session")
def session_env_vars(request, stack_name):
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    env_vars = dotenv_values(env_file)
    session_vars = {
        "PROJECT_NAME": stack_name,
    }
    env_vars.update(session_vars)
    os.environ.update(env_vars)
    return session_vars


@pytest.fixture(scope="session")
def subprocess_runner():
    def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(command, check=False, text=True, capture_output=True)
        cmd = " ".join(command)
        if proc.returncode:
            msg = f"'{cmd}' exited {proc.returncode}"
            logger.warning(msg)
            msg = f"'{cmd}' STDOUT:\n{proc.stdout}"
            logger.warning(msg)
            msg = f"'{cmd}' STDERR:\n{proc.stderr}"
            logger.warning(msg)
            logger.info(proc)
        return proc

    return run_command


@pytest.fixture
def dr_client(session_env_vars):
    return dr.Client()


@pytest.fixture
def llm_deployment_id():
    from core.resources import LLMDeployment

    return LLMDeployment().id


DATA_FILES = {
    "lending_club_profile": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Profile.csv",
    "lending_club_target": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Target.csv",
    "lending_club_transactions": "https://s3.amazonaws.com/datarobot_public_datasets/drx/Lending+Club+Transactions.csv",
    "diabetes": "https://s3.amazonaws.com/datarobot_public_datasets/10k_diabetes_20.csv",
    "mpg": "https://s3.us-east-1.amazonaws.com/datarobot_public_datasets/auto-mpg.csv",
}


@pytest.fixture(scope="module")
def url_lending_club_profile():
    return DATA_FILES["lending_club_profile"]


@pytest.fixture(scope="module")
def url_lending_club_target():
    return DATA_FILES["lending_club_target"]


@pytest.fixture(scope="module")
def url_lending_club_transactions():
    return DATA_FILES["lending_club_transactions"]


@pytest.fixture(scope="module")
def url_diabetes():
    return DATA_FILES["diabetes"]


@pytest.fixture(scope="module")
def url_mpg():
    return DATA_FILES["mpg"]


@pytest_asyncio.fixture(scope="module")
async def dataset_loaded(url_diabetes: str, analyst_db):
    from core.analyst_db import InternalDataSourceType
    from core.schema import AnalystDataset

    df = pd.read_csv(url_diabetes)
    # Replace non-JSON compliant values
    df = df.replace([float("inf"), -float("inf")], None)  # Replace infinity with None
    df = df.where(pd.notnull(df), None)  # Replace NaN with None

    # Create dataset dictionary
    dataset = AnalystDataset(
        name=os.path.splitext(os.path.basename(url_diabetes))[0],
        data=df,
    )
    await analyst_db.register_dataset(dataset, data_source=InternalDataSourceType.FILE)
    return dataset


@pytest_asyncio.fixture(scope="module")
async def analyst_db():
    from core.analyst_db import AnalystDB

    analyst_db = await AnalystDB.create(
        user_id="test_user_123",
        db_path=".",
        dataset_db_name="datasets",
        chat_db_name="chats",
    )
    return analyst_db


@contextlib.contextmanager
def cd(new_dir: Path) -> Any:
    """Changes the current working directory to the given path and restores the old directory on exit."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)


@pytest.fixture
def example_chat_file_content() -> dict:
    with open(
        os.path.join(os.path.dirname(__file__), "models", "example_chat.json"), "r"
    ) as f:
        return json.load(f)


@pytest_asyncio.fixture
async def mock_analyst_db_creation(monkeypatch):
    """Fixture to prevent real AnalystDB creation during tests."""
    from unittest.mock import AsyncMock

    from core.analyst_db import AnalystDB

    original_create = AnalystDB.create

    async def mock_create(user_id: str, **kwargs):
        """Return a mock AnalystDB instead of creating a real one."""
        mock_db = AsyncMock(spec=AnalystDB)
        mock_db.user_id = user_id
        return mock_db

    monkeypatch.setattr(AnalystDB, "create", staticmethod(mock_create))
    yield
    # Restore original
    monkeypatch.setattr(AnalystDB, "create", staticmethod(original_create))
