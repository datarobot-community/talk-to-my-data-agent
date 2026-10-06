# Copyright 2026 DataRobot, Inc.
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
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging", "pyyaml"]
# ///
"""Fail if .datarobot/cli/versions.yaml has drifted from the uv.lock files.

dr-environment bakes the pulumi-datarobot and pulumi-command providers and the
base-image Python from versions.yaml. A provider version that differs from infra/uv.lock makes Pulumi download
the provider at deploy time, which fails in an air-gapped environment.

Run from the repository root: uv run .github/scripts/check_versions_lock.py
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version

VERSIONS_YAML = Path(".datarobot/cli/versions.yaml")
INFRA_LOCK = Path("infra/uv.lock")


def locked_version(lock_path: Path, package: str) -> str:
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    for entry in lock.get("package", []):
        if entry["name"] == package:
            return str(entry["version"])
    raise SystemExit(f"{package} not found in {lock_path}")


def check_provider(versions: dict, package: str, plugin: str) -> list[str]:
    entry = versions.get(package)
    if entry is None:
        return [f"{VERSIONS_YAML}: missing {package} entry"]

    errors = []
    locked = locked_version(INFRA_LOCK, package)
    pinned = str(entry.get("minimum-version"))
    if pinned != locked:
        errors.append(
            f"{VERSIONS_YAML}: {package} minimum-version is {pinned}, "
            f"but {INFRA_LOCK} locks {locked}"
        )
    for platform, command in (entry.get("install") or {}).items():
        installs = re.findall(rf"\bresource {plugin} v?(\S+)", command)
        if installs != [locked]:
            errors.append(
                f"{VERSIONS_YAML}: {package} {platform} install command "
                f"does not install v{locked}: {command!r}"
            )
    return errors


def check_python(versions: dict) -> list[str]:
    entry = versions.get("python")
    if entry is None:
        return [f"{VERSIONS_YAML}: missing python entry"]

    errors = []
    image_version = Version(str(entry.get("version")))
    minimum = Version(str(entry.get("minimum-version")))
    if minimum.release[:2] != image_version.release[:2]:
        errors.append(
            f"{VERSIONS_YAML}: python minimum-version {minimum} does not match "
            f"python version {image_version}"
        )

    lock_paths = subprocess.run(
        ["git", "ls-files", "*uv.lock"], check=True, capture_output=True, text=True
    ).stdout.split()
    for lock_path in map(Path, lock_paths):
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        requires = SpecifierSet(lock.get("requires-python", ""))
        for label, version in (
            ("version", image_version),
            ("minimum-version", minimum),
        ):
            if version not in requires:
                errors.append(
                    f"{VERSIONS_YAML}: python {label} {version} does not satisfy "
                    f"requires-python {requires} in {lock_path}"
                )
    return errors


def main() -> int:
    versions = yaml.safe_load(VERSIONS_YAML.read_text(encoding="utf-8"))
    errors = (
        check_provider(versions, "pulumi-datarobot", "datarobot")
        + check_provider(versions, "pulumi-command", "command")
        + check_python(versions)
    )
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print(f"{VERSIONS_YAML} is in sync with the uv.lock files")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
