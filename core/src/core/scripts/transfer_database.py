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

import argparse
import asyncio
import logging
import os
import tempfile
from pathlib import Path

import datarobot as dr

os.environ["DISABLE_TELEMETRY"] = "true"

from core.persistent_storage import PersistentStorage

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer files between DataRobot application IDs using PersistentStorage."
        )
    )
    parser.add_argument(
        "--source-app",
        required=True,
        dest="source_app_id",
        help="Source application ID or name",
    )
    parser.add_argument(
        "--target-app",
        required=True,
        dest="target_app_id",
        help="Target application ID or name",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Perform deletions and uploads",
    )
    parser.add_argument(
        "--continue-transfer-non-conflicting",
        action="store_true",
        help="Proceed with non-conflicting files if conflicts are found",
    )
    parser.set_defaults(dry_run=True)
    return parser


def _require_env() -> None:
    missing = [
        env_name
        for env_name in ("DATAROBOT_ENDPOINT", "DATAROBOT_API_TOKEN")
        if not os.environ.get(env_name)
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {missing_list}")


def _new_storage(app_id: str) -> PersistentStorage:
    original_app_id = os.environ.get("APPLICATION_ID")
    os.environ["APPLICATION_ID"] = app_id
    try:
        return PersistentStorage(user_id=None, global_user=True)
    finally:
        if original_app_id is None:
            os.environ.pop("APPLICATION_ID", None)
        else:
            os.environ["APPLICATION_ID"] = original_app_id


def _get_app_status(client: dr.rest.RESTClientObject, app_id: str) -> str:
    response = client.get(f"customApplications/{app_id}/", timeout=30)
    status = response.json().get("status", "").lower()
    return str(status)


def _iter_custom_applications(
    client: dr.rest.RESTClientObject,
) -> list[dict[str, object]]:
    apps: list[dict[str, object]] = []
    response = client.get("customApplications/", timeout=30)
    payload = response.json()
    apps.extend(payload.get("data", []))
    next_url = payload.get("next")
    while next_url:
        response = client.get(next_url, timeout=30)
        payload = response.json()
        apps.extend(payload.get("data", []))
        next_url = payload.get("next")
    return apps


def _resolve_app_id(client: dr.rest.RESTClientObject, app_id_or_name: str) -> str:
    try:
        client.get(f"customApplications/{app_id_or_name}/", timeout=30)
        return app_id_or_name
    except Exception:
        pass

    matches: list[str] = []
    for app in _iter_custom_applications(client):
        name = app.get("name")
        app_id = app.get("id")
        if isinstance(name, str) and isinstance(app_id, str) and name == app_id_or_name:
            matches.append(app_id)

    if not matches:
        raise ValueError(f"No application found with name: {app_id_or_name}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple applications found with name: {app_id_or_name}. "
            "Use an application ID instead."
        )
    return matches[0]


def _ensure_apps_stopped(client: dr.rest.RESTClientObject, app_ids: list[str]) -> None:
    allowed_statuses = {"stopped", "paused"}
    statuses = {app_id: _get_app_status(client, app_id) for app_id in app_ids}
    running = {
        app_id: status
        for app_id, status in statuses.items()
        if status not in allowed_statuses
    }
    if running:
        details = ", ".join(f"{app_id}={status}" for app_id, status in running.items())
        raise RuntimeError(
            "Applications must be stopped or paused before transfer. "
            f"Current statuses: {details}"
        )


def _log_key_values(client: dr.rest.RESTClientObject, app_id: str) -> None:
    with client:
        for kv in dr.KeyValue.list(app_id, dr.KeyValueEntityType.CUSTOM_APPLICATION):
            logger.debug("%s => %s", kv.name, kv.value)


async def _download_files(
    storage: PersistentStorage, files: list[str], temp_dir: str
) -> dict[str, str]:
    local_paths: dict[str, str] = {}
    for file_name in files:
        local_path = Path(temp_dir) / file_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        await storage.fetch_from_storage(file_name, str(local_path))
        local_paths[file_name] = str(local_path)
    return local_paths


async def _upload_files(
    storage: PersistentStorage, local_paths: dict[str, str]
) -> None:
    for file_name, local_path in local_paths.items():
        await storage.save_to_storage(file_name, local_path)


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(message)s"
    )
    _require_env()

    client = dr.Client(
        token=os.environ.get("DATAROBOT_API_TOKEN"),
        endpoint=os.environ.get("DATAROBOT_ENDPOINT"),
    )
    source_app_id = _resolve_app_id(client, args.source_app_id)
    target_app_id = _resolve_app_id(client, args.target_app_id)
    if source_app_id == target_app_id:
        raise ValueError("Source and target application IDs must be different.")

    _ensure_apps_stopped(client, [source_app_id, target_app_id])

    logger.debug("KeyValues for source application: %s", source_app_id)
    _log_key_values(client, source_app_id)
    logger.debug("KeyValues for target application: %s", target_app_id)
    _log_key_values(client, target_app_id)

    source_storage = _new_storage(source_app_id)

    logger.info("Listing files in source application: %s", source_app_id)
    source_files = await source_storage.files()
    logger.info("Source contains %d files", len(source_files))

    target_storage = _new_storage(target_app_id)

    target_files = await target_storage.files()
    conflicts = sorted(set(target_files) & set(source_files))
    if conflicts and not args.continue_transfer_non_conflicting:
        conflict_list = ", ".join(conflicts)
        raise RuntimeError(
            f"Refusing to overwrite existing target files. Conflicts: {conflict_list}"
        )

    upload_files = sorted(set(source_files) - set(conflicts))

    if args.dry_run:
        logger.info("Dry run: listing target files to delete")
        for file_name in conflicts:
            logger.warning("Would skip (conflict): %s", file_name)
        for file_name in upload_files:
            logger.info("Would add: %s", file_name)
        logger.info("Dry run complete")
        logger.info(
            "Review the changes above carefully before re-running with --no-dry-run."
        )
        return 0

    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info("Downloading source files")
        local_paths = await _download_files(source_storage, upload_files, temp_dir)

        logger.info("Uploading files to target")
        await _upload_files(target_storage, local_paths)

    logger.info("Transfer complete")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
