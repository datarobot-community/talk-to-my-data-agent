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


import hashlib
from pathlib import Path

import pulumi
import pulumi_command as command  # type: ignore[import-not-found]
from datarobot_pulumi_utils.pulumi.stack import PROJECT_NAME

from . import project_dir

FRONTEND_SOURCE_GLOBS = [
    "src/**/*",
    "public/**/*",
    "package.json",
    "package-lock.json",
    "index.html",
    "tsconfig*.json",
    "vite.config.*",
    "tailwind.config.*",
    "postcss.config.*",
    "eslint.config.*",
    "components.json",
    ".prettierrc*",
    ".npmrc",
]


def _hash_frontend_sources(frontend_dir: Path) -> str:
    """Compute a SHA-256 hash of all relevant frontend source files."""
    h = hashlib.sha256()
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in FRONTEND_SOURCE_GLOBS:
        for p in frontend_dir.glob(pattern):
            if p.is_file() and p not in seen:
                seen.add(p)
                paths.append(p)
    for p in sorted(paths):
        h.update(str(p.relative_to(frontend_dir)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def build_frontend() -> command.local.Command:
    """
    Build the frontend application before deploying infrastructure.
    Split into two stages: install dependencies and build application.
    """
    frontend_dir = project_dir.parent / "app_frontend"
    source_hash = _hash_frontend_sources(frontend_dir)

    build_react_app = command.local.Command(
        f"Talk to My Data [{PROJECT_NAME}] Build Frontend",
        create=f"cd {frontend_dir} && npm install && npm run build",
        triggers=[source_hash],
        opts=pulumi.ResourceOptions(
            # This resource should be created first
            depends_on=[]
        ),
    )

    return build_react_app


app_frontend = build_frontend()

__all__ = ["app_frontend"]
