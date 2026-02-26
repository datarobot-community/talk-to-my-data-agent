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

from pathlib import Path
from unittest.mock import patch

from app import get_app_version


def test_version_from_file(tmp_path: Path) -> None:
    version_file = tmp_path / "VERSION"
    version_file.write_text("v11.5.2\n")
    with patch("app.ROOT_DIR", tmp_path):
        assert get_app_version() == "v11.5.2"


def test_version_empty_when_no_file(tmp_path: Path) -> None:
    with patch("app.ROOT_DIR", tmp_path):
        assert get_app_version() == ""


def test_version_strips_whitespace(tmp_path: Path) -> None:
    version_file = tmp_path / "VERSION"
    version_file.write_text("  v11.5.2-3-gabcdef0  \n")
    with patch("app.ROOT_DIR", tmp_path):
        assert get_app_version() == "v11.5.2-3-gabcdef0"
