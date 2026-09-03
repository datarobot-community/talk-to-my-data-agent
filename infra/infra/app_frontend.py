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
import os
import time
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
    """Build the frontend application before deploying infrastructure."""
    frontend_dir = project_dir.parent / "app_frontend"
    static_assets_dir = project_dir.parent / "app_backend" / "static" / "assets"
    source_hash = _hash_frontend_sources(frontend_dir)

    # Build via the shell pulumi_command uses (cmd on Windows, POSIX sh elsewhere).
    # Windows: `cd /d` also switches drive; `if exist "dir\."` is the directory test
    # that avoids cmd mis-parsing a trailing backslash before the closing quote. The
    # trailing check fails "Build Frontend" loudly when the build produced no assets.
    if os.name == "nt":
        create_cmd = (
            f'cd /d "{frontend_dir}" && npm install && npm run build '
            f'&& if not exist "{static_assets_dir}\\." exit /b 1'
        )
    else:
        create_cmd = (
            f'cd "{frontend_dir}" && npm install && npm run build '
            f'&& test -d "{static_assets_dir}"'
        )

    build_react_app = command.local.Command(
        f"Talk to My Data [{PROJECT_NAME}] Build Frontend",
        create=create_cmd,
        # The build output (static_assets_dir) is gitignored, so a fresh checkout or
        # cleaned workspace has none. Pulumi records the trigger computed before the
        # build runs, so a plain presence flag would store "absent" and let the next
        # clean deploy match it and skip the build — shipping an app with no frontend.
        # A unique token for the absent case always differs, forcing a rebuild whenever
        # the output is missing; a present, unchanged build stays "present" and skips.
        triggers=[
            source_hash,
            "present" if static_assets_dir.is_dir() else f"missing-{time.time()}",
        ],
        opts=pulumi.ResourceOptions(
            # This resource should be created first
            depends_on=[]
        ),
    )

    return build_react_app


app_frontend = build_frontend()

__all__ = ["app_frontend"]
