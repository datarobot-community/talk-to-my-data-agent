# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from typing import (
    Any,
    BinaryIO,
    Callable,
    List,
    Optional,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import datarobot as dr
from datarobot.enums import FilesOverwriteStrategy
from datarobot.enums import KeyValueEntityType as DRKeyValueEntityType
from datarobot.errors import ClientError
from datarobot.fs import DataRobotFile, DataRobotFileSystem
from datarobot.fs.file_system import FileInfo as BaseFileInfo
from datarobot.models.files import File, Files, FilesDetails
from fsspec.spec import AbstractFileSystem

from core.persistent_fs.kv_custom_app_implementattion import (
    KeyValue,
    KeyValueEntityType,
)

Path = str
NodeInfo = dict[str, str | int | float]
Metadata = dict[Path, NodeInfo]

CatalogId = str
LocalFileInfo = tuple[str, float]
LocalFilesMetadata = dict[CatalogId, LocalFileInfo]

WrapperParams = ParamSpec("WrapperParams")
WrapperReturnType = TypeVar("WrapperReturnType")

logger = logging.getLogger(__name__)

METADATA_STORAGE_NAME = "fs_metadata"
TIMESTAMP_STORAGE_NAME = "fs_timestamp"
CATALOG_STORAGE_NAME = "fs_catalog"

FILE_API_CONNECT_TIMEOUT = float(os.environ.get("FILE_API_CONNECT_TIMEOUT", 180))
FILE_API_READ_TIMEOUT = float(os.environ.get("FILE_API_READ_TIMEOUT", 180))


class FileInfo(BaseFileInfo):
    """
    Information about a file or directory in DataRobot File System.

    Attributes
    ----------
    name:
        The path of the file or directory. Does not include the protocol prefix.
    size:
        The size of the file in bytes. For directories, this is 0.
    type:
        The type of the item, either 'file' or 'directory'.
    format:
        The file format (e.g., 'csv', 'pdf') if the item is a file; None for directories.
    created_at:
        The file creation timestamp if the item is a file; None for directories.
    catalog_id:
        The catalog id of the file; None for legacy directories.
    modified_at:
        The modification timestamp of the file.
    """

    catalog_id: Optional[str]
    modified_at: Optional[float]


def _keep_metadata_in_sync(
    func: Callable[WrapperParams, WrapperReturnType],
) -> Callable[WrapperParams, WrapperReturnType]:
    def wrapper(
        *args: WrapperParams.args, **kwargs: WrapperParams.kwargs
    ) -> WrapperReturnType:
        fs_entity: "DRFileSystem" = cast("DRFileSystem", args[0])
        logger.debug(
            "Entering metadata sync wrapper.", extra={"stack": fs_entity._sync_stack}
        )
        fs_entity._sync_stack.append(func.__name__)
        if (
            len(fs_entity._sync_stack) == 1
            and not fs_entity._remote_metadata_was_updated()
        ):
            fs_entity._refresh_local_metadata()

        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.debug(
                "Exception caught by sync wrapper.",
                extra={"function": func.__name__, "stack": fs_entity._sync_stack},
            )
            fs_entity._sync_stack.pop()
            raise

        if len(fs_entity._sync_stack) == 1 and fs_entity._local_metadata_was_updated():
            fs_entity._update_stored_metadata()
        fs_entity._sync_stack.pop()
        logger.debug(
            "Exiting metadata sync wrapper.", extra={"stack": fs_entity._sync_stack}
        )
        return result

    return wrapper


class DRFileSystem(DataRobotFileSystem):
    """DRFileSystem is fsspec implementation to interact with the DataRobot filesystem."""

    _catalog_id: str = ""

    def __init__(
        self,
        dr_client: dr.rest.RESTClientObject | None = None,
        catalog_id: str | None = None,
        default_overwrite_strategy: FilesOverwriteStrategy = FilesOverwriteStrategy.REPLACE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._legacy_fs = LegacyDRFileSystem(dr_client, *args, **kwargs)
        self.client = dr_client or dr.Client(
            token=os.environ.get("DATAROBOT_API_TOKEN"),
            endpoint=os.environ.get("DATAROBOT_ENDPOINT"),
        )
        self.default_overwrite_strategy = default_overwrite_strategy
        self.app_id: str | None = None
        if catalog_id:
            self._catalog_id = catalog_id
        else:
            self.app_id = os.environ.get("APPLICATION_ID")
            if not self.app_id:
                raise ValueError("APPLICATION_ID env variable is not set.")
            self._catalog_id = ""
            self._initialize_catalog_id()

    def _initialize_catalog_id(self) -> None:
        """Create or get catalog id used for the storing files for this application."""
        with self.client:
            catalog_stored = dr.KeyValue.find(
                cast(str, self.app_id),
                DRKeyValueEntityType.CUSTOM_APPLICATION,
                CATALOG_STORAGE_NAME,
            )
            if catalog_stored:
                self._catalog_id = catalog_stored.value
                logger.info(
                    "Found existing catalog id=%s from key value storage",
                    self._catalog_id,
                )
                return

            self._catalog_id = self.create_catalog_item_dir()
            dr.KeyValue.create(
                entity_id=cast(str, self.app_id),
                entity_type=DRKeyValueEntityType.CUSTOM_APPLICATION,
                name=CATALOG_STORAGE_NAME,
                category=dr.KeyValueCategory.ARTIFACT,
                value_type=dr.KeyValueType.STRING,
                value=self._catalog_id,
            )
            logger.info(
                "Created new catalog id=%s and stored it in key value storage",
                self._catalog_id,
            )

    def current_version_id(self) -> str | None:
        """Return the catalog item's latest version id with a single request.

        The catalog version advances every time a file is uploaded, so this is a
        cheap change-detection signal: if the version is unchanged since a prior
        download, the persisted files have not changed and a re-download can be
        skipped. Returns None if the version cannot be determined.
        """
        with self.client:
            try:
                return FilesDetails.get(self._catalog_id).version_id
            except ClientError:
                logger.debug(
                    "Could not fetch catalog version id.",
                    extra={"catalog_id": self._catalog_id},
                )
                return None

    def download_file(self, rpath: str, lpath: str) -> None:
        """Download the latest version of a known file directly to ``lpath``.

        ``DRFileSystem.get()`` resolves a path through fsspec's generic
        machinery, which lists the parent directory and re-derives file info
        several times (each doubled by a legacy-filesystem probe), then signs the
        file and streams it from object storage. When the exact path is known and
        the latest version is wanted, the Files API ``downloads`` endpoint streams
        the bytes in a single request, skipping all of that.

        If the file is not found in the new filesystem, this falls back to the
        legacy filesystem so files that have not yet been migrated are still
        served.

        Raises
        ------
        FileNotFoundError
            If the file does not exist in either the new or legacy filesystem.
        """
        internal_path = self._strip_protocol(rpath)
        files = self._get_files_wrapper_for_folder_id(self._catalog_id)
        try:
            with self._try_convert_to_fsspec_exception():
                files.download(file_name=internal_path, file_path=lpath)
        except FileNotFoundError:
            logger.debug(
                "download_file - File %s not found in new fs, using legacy fs.",
                rpath,
                extra={"path": rpath},
            )
            self._legacy_fs.get_file(rpath, lpath)

    def _strip_catalog(self, path: str) -> str:
        """Remove catalog prefix from path."""
        return path.removeprefix(f"{self._catalog_id}/")

    def unstrip_protocol(self, name: str) -> str:
        """Prefix path with protocol and catalog id."""
        name = name.lstrip("/")
        if name.startswith(f"{self.protocol}://{self._catalog_id}/"):
            return f"{self.protocol}://{self._catalog_id}/{name.removeprefix(f'{self.protocol}://{self._catalog_id}/')}"
        elif name.startswith(f"{self.protocol}://"):
            return f"{self.protocol}://{self._catalog_id}/{name.removeprefix(f'{self.protocol}://')}"

        return f"{self.protocol}://{self._catalog_id}/{name}"

    def _split_path(self, path: str) -> Tuple[str, str]:
        """Split path into catalog id and path without protocol."""
        if path.strip() == "/" or path.strip() == f"{self.protocol}://":
            return self._catalog_id, ""

        path_without_protocol = self._strip_protocol(path)
        return self._catalog_id, path_without_protocol

    def _format_path_details_for_files(  # type: ignore[override]
        self, catalog_id: str, files: List[File], show_details: bool
    ) -> Union[List[str], List[FileInfo]]:
        """Format path details for a list of files. Files can represent both files and directories."""
        ret = super()._format_path_details_for_files(catalog_id, files, show_details)
        if show_details:
            for file in ret:
                file = cast(FileInfo, file)
                file["catalog_id"] = self._catalog_id
                file["name"] = self._strip_catalog(file["name"])
                if file["type"] != "directory" and file["created_at"] is not None:
                    file["modified_at"] = file["created_at"].timestamp()
                else:
                    file["modified_at"] = None
            return cast(List[FileInfo], ret)
        return [self._strip_catalog(file_path) for file_path in ret]  # type: ignore[arg-type]

    def _augment_path_details(self, detail: BaseFileInfo) -> FileInfo:
        """Add fields to path details that might be missing to harmonize between new and legacy systems."""
        detail = cast(FileInfo, detail)
        if "created_at" in detail and detail["created_at"] is not None:
            back_up_modified_at = detail["created_at"].timestamp()
        else:
            back_up_modified_at = None
        if detail["type"] == "directory":
            detail["catalog_id"] = detail.get("catalog_id", None)
            detail["format"] = detail.get("format", None)
            detail["modified_at"] = detail.get("modified_at", back_up_modified_at)
            detail["created_at"] = detail.get("created_at", None)
            detail["size"] = detail.get("size", 0)
            return detail

        detail["catalog_id"] = detail.get("catalog_id", self._catalog_id)
        detail["format"] = detail.get("format", "unknown")
        detail["modified_at"] = detail.get("modified_at", back_up_modified_at)
        detail["created_at"] = detail.get("created_at", None)
        detail["size"] = detail.get("size", 0)
        return detail

    def ls(  # type: ignore[override]
        self, path: str, detail: bool = True, **kwargs: Any
    ) -> Union[List[str], List[FileInfo]]:
        """List paths in the filesystem."""
        new_paths: Optional[List[BaseFileInfo]] = None
        legacy_paths: Optional[List[BaseFileInfo]] = None

        try:
            new_paths = super().ls(path, detail=True, **kwargs)
        except FileNotFoundError:
            logger.debug("ls - No paths found in new fs.", extra={"path": path})

        try:
            legacy_paths = cast(
                List[BaseFileInfo], self._legacy_fs.ls(path, detail=True, **kwargs)
            )
        except FileNotFoundError:
            logger.debug("ls - No paths found in legacy fs.", extra={"path": path})

        if new_paths is None and legacy_paths is None:
            raise FileNotFoundError(f"Path {path} not found.")

        new_paths = new_paths or []
        legacy_paths = legacy_paths or []
        all_paths = {p["name"]: p for p in new_paths}
        for legacy_path in legacy_paths:
            name_if_dir = f"{legacy_path['name'].rstrip('/')}/"
            if legacy_path["type"] == "directory" and name_if_dir not in all_paths:
                logger.debug(
                    "ls - Found directory %s in legacy fs with no corresponding entry in new fs.",
                    name_if_dir,
                    extra={"path": name_if_dir},
                )
                legacy_path["name"] = name_if_dir
                all_paths[name_if_dir] = legacy_path
            elif (
                legacy_path["type"] != "directory"
                and legacy_path["name"] not in all_paths
            ):
                logger.debug(
                    "ls - Found file %s in legacy fs with no corresponding entry in new fs.",
                    legacy_path["name"],
                    extra={"path": legacy_path["name"]},
                )
                all_paths[legacy_path["name"]] = legacy_path

        if not detail:
            return list(all_paths.keys())
        return list(map(self._augment_path_details, all_paths.values()))

    def info(self, path: str, **kwargs: Any) -> FileInfo:
        """Get info for file or directory at path."""
        path_without_protocol = self._strip_protocol(path)
        if path_without_protocol == self.root_marker:
            return {
                "name": self.root_marker,
                "type": "directory",
                "size": 0,
                "format": None,
                "modified_at": None,
                "created_at": None,
                "catalog_id": self._catalog_id,
            }
        return super().info(path, **kwargs)  # type: ignore[return-value]

    def _open(  # type: ignore[override]
        self, path: str, mode: str = "rb", **kwargs: Any
    ) -> Union[DataRobotFile, BinaryIO]:
        overwrite_strategy = kwargs.pop(
            "overwrite_strategy", self.default_overwrite_strategy
        )
        if mode == "rb":
            try:
                new_exists = self.info(path).get("catalog_id") == self._catalog_id
            except FileNotFoundError:
                new_exists = False
            if new_exists:
                return super()._open(
                    path, mode=mode, overwrite_strategy=overwrite_strategy, **kwargs
                )
            logger.debug(
                "open - File %s not found in new fs, using legacy fs.",
                path,
                extra={"path": path},
            )
            return self._legacy_fs._open(path, mode=mode, **kwargs)
        else:
            has_legacy_file = self._legacy_fs.exists(path) and self._legacy_fs.isfile(
                path
            )
            file_handle = super()._open(
                path, mode=mode, overwrite_strategy=overwrite_strategy, **kwargs
            )
            original_upload_chunk = file_handle._upload_chunk

            def upload_and_cleanup(final: bool = False) -> bool:
                file_was_uploaded = original_upload_chunk(final=final)
                if not has_legacy_file or not file_was_uploaded:
                    return file_was_uploaded

                logger.debug(
                    "open - Existing file %s found in legacy fs, removing it.",
                    path,
                    extra={"path": path},
                )
                self._legacy_fs.rm_file(path)
                return file_was_uploaded

            # monkey patch to only delete legacy file on new file upload completion.
            file_handle._upload_chunk = upload_and_cleanup  # type: ignore[method-assign]
            return file_handle

    def open(  # type: ignore[override]
        self, path: str, mode: str = "rb", **kwargs: Any
    ) -> Union[DataRobotFile, BinaryIO]:
        """Open file at path for reading or writing."""
        overwrite_strategy = kwargs.pop(
            "overwrite_strategy", self.default_overwrite_strategy
        )
        return super().open(
            path, mode=mode, overwrite_strategy=overwrite_strategy, **kwargs
        )

    def put(  # type: ignore[override]
        self, lpath: Union[str, List[str]], rpath: Union[str, List[str]], **kwargs: Any
    ) -> None:
        """Upload file from local path to remote path."""
        overwrite_strategy = kwargs.pop(
            "overwrite_strategy", self.default_overwrite_strategy
        )
        return super().put(
            lpath, rpath, overwrite_strategy=overwrite_strategy, **kwargs
        )

    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:  # type: ignore[override]
        """Copy file from source path to destination path."""
        try:
            new_exists = self.info(path1).get("catalog_id") == self._catalog_id
        except FileNotFoundError:
            new_exists = False
        legacy_exists = self._legacy_fs.exists(path1)
        overwrite_strategy = kwargs.pop(
            "overwrite_strategy", self.default_overwrite_strategy
        )

        if new_exists and self.isfile(path1):
            super().cp_file(
                path1, path2, overwrite_strategy=overwrite_strategy, **kwargs
            )
        elif legacy_exists and self._legacy_fs.isfile(path1):
            logger.debug(
                "cp_file - Moving legacy file %s from legacy fs to new fs before copy.",
                path1,
                extra={"path": path1},
            )
            info = self._legacy_fs.info(path1)
            source_path = os.path.basename(path1.rstrip("/"))
            Files(info["catalog_id"], "", "", [], datetime.now(), "").copy(
                source_path=source_path,
                target=path1,
                target_files=Files(self._catalog_id, "", "", [], datetime.now(), ""),
                overwrite=FilesOverwriteStrategy.SKIP,
            )
            self._legacy_fs.rm_file(path1)
            super().cp_file(
                path1, path2, overwrite_strategy=overwrite_strategy, **kwargs
            )
        elif (
            new_exists
            and legacy_exists
            and self.isdir(path1)
            and self._legacy_fs.isdir(path1)
        ):
            for file_name, file_info in self._legacy_fs.find(
                path1, withdirs=False, detail=True
            ).items():
                logger.debug(
                    "cp_file - Moving legacy file %s from legacy fs to new fs before copy.",
                    file_name,
                    extra={"path": file_name},
                )
                source_path = os.path.basename(file_name.rstrip("/"))
                Files(file_info["catalog_id"], "", "", [], datetime.now(), "").copy(
                    source_path=source_path,
                    target=file_name,
                    target_files=Files(
                        self._catalog_id, "", "", [], datetime.now(), ""
                    ),
                    overwrite=FilesOverwriteStrategy.SKIP,
                )
            self._legacy_fs.rm(path1, recursive=True)
            super().cp_file(
                path1, path2, overwrite_strategy=overwrite_strategy, **kwargs
            )
        elif new_exists and self.isdir(path1):
            super().cp_file(path1, path2, **kwargs)
        elif legacy_exists and self._legacy_fs.isdir(path1):
            for file_name, file_info in self._legacy_fs.find(
                path1, withdirs=False, detail=True
            ).items():
                logger.debug(
                    "cp_file - Moving legacy file %s from legacy fs to new fs before copy.",
                    file_name,
                    extra={"path": file_name},
                )
                source_path = os.path.basename(file_name.rstrip("/"))
                Files(file_info["catalog_id"], "", "", [], datetime.now(), "").copy(
                    source_path=source_path,
                    target=file_name,
                    target_files=Files(
                        self._catalog_id, "", "", [], datetime.now(), ""
                    ),
                    overwrite=FilesOverwriteStrategy.SKIP,
                )
                self._legacy_fs.rm_file(file_name)
            self._legacy_fs.rm(path1, recursive=True)
            super().cp_file(
                path1, path2, overwrite_strategy=overwrite_strategy, **kwargs
            )
        else:
            raise FileNotFoundError(f"File {path1} not found")

    def cp_directory(self, path1: str, path2: str, **kwargs: Any) -> None:  # type: ignore[override]
        """Copy directory from source path to destination path."""
        overwrite_strategy = kwargs.pop(
            "overwrite_strategy", self.default_overwrite_strategy
        )
        super().cp_directory(
            path1, path2, overwrite_strategy=overwrite_strategy, **kwargs
        )

    def rm_file(self, path: Union[str, List[str]], **kwargs: Any) -> None:
        """Remove file at path."""
        super().rm_file(path, **kwargs)
        # in new fs rm_file is always recursive, and also deletes directories
        # also supports deleting multiple paths at once, similar to rm (without globs)
        legacy_paths = [path] if isinstance(path, str) else path
        for legacy_path in legacy_paths:
            path_to_delete = legacy_path.rstrip("/")
            if self._legacy_fs.exists(path_to_delete):
                if self._legacy_fs.isdir(path_to_delete):
                    self._legacy_fs.rm(path_to_delete, recursive=True, **kwargs)
                else:
                    self._legacy_fs.rm_file(path_to_delete, **kwargs)

    def rm_directory(self, path: Union[str, List[str]], **kwargs: Any) -> None:
        """Remove directory at path."""
        super().rm_directory(path, **kwargs)
        legacy_paths = [path] if isinstance(path, str) else path
        for legacy_path in legacy_paths:
            if self._legacy_fs.exists(legacy_path):
                if not self._legacy_fs.isdir(legacy_path):
                    raise ValueError(f"{legacy_path} is not a directory")
                path_to_delete = legacy_path.rstrip("/")
                self._legacy_fs.rm(path_to_delete, recursive=True, **kwargs)

    def mv(
        self,
        path1: Union[str, List[str]],
        path2: Union[str, List[str]],
        recursive: bool = False,
        maxdepth: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Move file(s) or directory from source path to destination path."""
        # Override definition to simply things and handle new/legacy system
        self.copy(path1, path2, recursive=recursive, maxdepth=maxdepth, **kwargs)
        self.rm(path1, recursive=recursive, maxdepth=maxdepth, **kwargs)

    def list_unmigrated_files(self, path: str) -> List[FileInfo]:
        """
        List files that are not migrated to the new filesystem.
        If all files are migrated, return an empty list, and it is safe to upgrade to the new filesystem.
        """
        unmigrated_files = []
        for _, details in self.find(path, withdirs=False, detail=True).items():
            if details.get("catalog_id") != self._catalog_id:
                unmigrated_files.append(details)
        return unmigrated_files  # type: ignore[return-value]

    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:  # type: ignore[override]
        """Create empty directory at path. Only persists directory in legacy filesystem."""
        self._legacy_fs.mkdir(path, create_parents=create_parents, **kwargs)

    def makedirs(self, path: str, exist_ok: bool = False, **kwargs: Any) -> None:  # type: ignore[override]
        """Create directories at path. Only persists directories in legacy filesystem."""
        self._legacy_fs.makedirs(path, exist_ok=exist_ok, **kwargs)

    def rmdir(self, path: str, **kwargs: Any) -> None:  # type: ignore[override]
        """Delete empty directory at path. Only affects legacy filesystem."""
        self._legacy_fs.rmdir(path, **kwargs)

    def modified(self, path: str, **kwargs: Any) -> float:  # type: ignore[override]
        """Get the modification time of the file at path. Directory paths are not supported."""
        if self.exists(path):
            created_at = self.info(path)["created_at"]
            if created_at:
                return created_at.timestamp()
        return self._legacy_fs.modified(path)


class LegacyDRFileSystem(AbstractFileSystem):  # type: ignore[misc]
    """
    DRFileSystem is fsspec implementation for interact with Datarobot
    KeyValue and File storage for having persistent storage inside
    custom applications.
    """

    protocol = "dr"

    def __init__(
        self,
        dr_client: dr.rest.RESTClientObject | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.client = dr_client or dr.Client(
            token=os.environ.get("DATAROBOT_API_TOKEN"),
            endpoint=os.environ.get("DATAROBOT_ENDPOINT"),
        )
        self.app_id: str = os.environ.get("APPLICATION_ID")  # type: ignore[assignment]
        if not self.app_id:
            raise ValueError("APPLICATION_ID env variable is not set.")

        self._temp_dir = tempfile.mkdtemp()
        self._downloaded_files: LocalFilesMetadata = {}

        self._fs_metadata: Metadata = {}
        self._fs_metadata_timestamp: float = 0.0  # timestamp of when we have data

        self._fs_metadata_stored: KeyValue | None = None  # remotely stored metadata
        self._fs_metadata_timestamp_stored: KeyValue | None = (
            None  # remotely stored timestamp
        )

        self._sync_stack: list[
            str
        ] = []  # making sure that local metadata fetched for first and updated for last nested call

        logger.debug("Initialized DRFileSystem.", extra={"tmp_dir": self._temp_dir})

    def __del__(self) -> None:
        """Cleanup temporary directory on object destruction."""
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def _refresh_fs_metadata_timestamp_stored(self) -> None:
        with self.client:
            if self._fs_metadata_timestamp_stored:
                self._fs_metadata_timestamp_stored.refresh()
            else:
                self._fs_metadata_timestamp_stored = KeyValue.find(
                    self.app_id,
                    KeyValueEntityType.CUSTOM_APPLICATION,
                    TIMESTAMP_STORAGE_NAME,
                )

    def _refresh_fs_metadata_stored(self) -> None:
        with self.client:
            if self._fs_metadata_stored:
                self._fs_metadata_stored.refresh()
            else:
                self._fs_metadata_stored = KeyValue.find(
                    self.app_id,
                    KeyValueEntityType.CUSTOM_APPLICATION,
                    METADATA_STORAGE_NAME,
                )

    def _remote_metadata_was_updated(self) -> bool:
        self._refresh_fs_metadata_timestamp_stored()
        if not self._fs_metadata_timestamp_stored:
            return True
        return (
            self._fs_metadata_timestamp_stored.numeric_value
            <= self._fs_metadata_timestamp
        )

    def _local_metadata_was_updated(self) -> bool:
        if self._fs_metadata_timestamp == 0:
            return False
        if not self._fs_metadata_timestamp_stored:
            return True
        return (
            self._fs_metadata_timestamp
            > self._fs_metadata_timestamp_stored.numeric_value
        )

    def _update_stored_metadata(self) -> None:
        logger.debug("Updating metadata in persistent storage.")
        with self.client:
            if self._fs_metadata_timestamp_stored:
                self._fs_metadata_timestamp_stored.update(
                    value=self._fs_metadata_timestamp
                )
            else:
                self._fs_metadata_timestamp_stored = KeyValue.create(
                    entity_id=self.app_id,
                    entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
                    name=TIMESTAMP_STORAGE_NAME,
                    category=dr.KeyValueCategory.ARTIFACT,
                    value_type=dr.KeyValueType.NUMERIC,
                    value=self._fs_metadata_timestamp,
                )

            if self._fs_metadata_stored:
                self._fs_metadata_stored.update(value=json.dumps(self._fs_metadata))
            else:
                self._fs_metadata_stored = KeyValue.create(
                    entity_id=self.app_id,
                    entity_type=KeyValueEntityType.CUSTOM_APPLICATION,
                    name=METADATA_STORAGE_NAME,
                    category=dr.KeyValueCategory.ARTIFACT,
                    value_type=dr.KeyValueType.JSON,
                    value=json.dumps(self._fs_metadata),
                )

    def _refresh_local_metadata(self) -> None:
        logger.debug("Updating local metadata from persistent storage.")
        self._refresh_fs_metadata_timestamp_stored()
        if self._fs_metadata_timestamp_stored:
            self._fs_metadata_timestamp = (
                self._fs_metadata_timestamp_stored.numeric_value
            )

        self._refresh_fs_metadata_stored()
        if self._fs_metadata_stored:
            self._fs_metadata = json.loads(self._fs_metadata_stored.value)

    @_keep_metadata_in_sync
    def mkdir(self, path: str, create_parents: bool = True, **kwargs: Any) -> None:
        logger.debug(
            "Making directory.", extra={"path": path, "create_parents": create_parents}
        )
        path = self._strip_protocol(path)
        if self.exists(path):
            if create_parents:
                return
            else:
                raise FileExistsError()

        parent = self._parent(path)
        if parent and not self.exists(parent):
            if create_parents:
                self.makedir(parent, create_parents=True, **kwargs)
            else:
                raise FileNotFoundError()
        clean_path = path.rstrip("/")
        self._fs_metadata[clean_path] = {
            "type": "directory",
            "name": clean_path,
            "modified_at": time.time(),
        }
        self._fs_metadata_timestamp = time.time()

    @_keep_metadata_in_sync
    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        logger.debug("Making directories.", extra={"path": path, "exist_ok": exist_ok})
        path = self._strip_protocol(path)
        if self.exists(path) and not exist_ok:
            raise FileExistsError()

        self.mkdir(path, create_parents=True)

    @_keep_metadata_in_sync
    def rmdir(self, path: str) -> None:
        logger.debug("Removing directory.", extra={"path": path})
        path = self._strip_protocol(path)
        if not self.exists(path):
            raise FileNotFoundError()
        if not self.isdir(path):
            raise ValueError(f"{path} is not a directory")
        if self.ls(path, detail=False):
            raise ValueError(f"{path} is not empty")

        self._fs_metadata.pop(path, None)
        self._fs_metadata_timestamp = time.time()

    @_keep_metadata_in_sync
    def ls(
        self, path: str, detail: bool = True, **kwargs: Any
    ) -> list[str] | list[dict[str, Any]]:
        logger.debug(
            "Providing directory info.", extra={"path": path, "detail": detail}
        )
        path = self._strip_protocol(path)
        clean_path = path.rstrip("/")
        # empty clean_path is root
        if clean_path and clean_path not in self._fs_metadata:
            raise FileNotFoundError()
        if clean_path and self._fs_metadata[clean_path].get("type") != "directory":
            return []
        children = {
            p
            for p in self._fs_metadata.keys()
            if p.startswith(clean_path) and "/" not in p[len(clean_path) + 1 :]
        }
        children.discard(clean_path)
        ordered_children = sorted(children)
        if detail:
            return [self._fs_metadata[c] for c in ordered_children]
        return ordered_children

    @_keep_metadata_in_sync
    def modified(self, path: str) -> float:
        if not self.exists(path):
            raise FileNotFoundError()
        return cast(float, self.info(path).get("modified_at", 0.0))

    @_keep_metadata_in_sync
    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> BinaryIO:
        logger.debug("Opening file.", extra={"path": path, "mode": mode})
        path = self._strip_protocol(path)

        if mode not in ["rb", "wb"]:
            raise NotImplementedError("Only read and write modes are supported")

        if mode == "rb":
            if not self.exists(path):
                raise FileNotFoundError()
            if not self.isfile(path):
                raise ValueError(f"{path} is not a file")
            local_path = self._get_local_path(self.info(path))
            return cast(BinaryIO, open(local_path, mode))
        elif mode == "wb":
            parent = self._parent(path)
            if not self.exists(parent):
                raise FileNotFoundError(parent)
            if not self.isdir(parent):
                raise ValueError(f"{parent} is not a directory")
            local_path = os.path.join(self._temp_dir, str(uuid.uuid4()))
            return _FileIOWrapper(
                fs_entity=self, virtual_path=path, name=local_path, mode=mode
            )
        else:
            raise NotImplementedError()

    def _get_local_path(self, file_info: dict[str, Any]) -> str:
        catalog_id = file_info.get("catalog_id")
        if not catalog_id:
            raise ValueError(f"{file_info} is missing catalog_id")
        if file_info["catalog_id"] not in self._downloaded_files:
            # there is no local copy
            return self._download_file(file_info)
        local_path, local_copy_update_time = self._downloaded_files[
            file_info["catalog_id"]
        ]
        if file_info["modified_at"] > local_copy_update_time:
            # local copy is outdated
            return self._download_file(file_info)
        return local_path

    def _download_file(self, file_info: dict[str, Any]) -> str:
        catalog_id = file_info.get("catalog_id")
        if not catalog_id:
            raise ValueError(f"{file_info} is missing catalog_id")

        local_path = os.path.join(self._temp_dir, catalog_id)
        logger.debug(
            "Downloading file from catalog.",
            extra={"catalog_id": catalog_id, "local_path": local_path},
        )

        response = self.client.get(
            f"files/{catalog_id}/file/",
            timeout=(FILE_API_CONNECT_TIMEOUT, FILE_API_READ_TIMEOUT),
        )
        with open(local_path, "wb") as f:
            f.write(response.content)

        self._downloaded_files[catalog_id] = (
            local_path,
            file_info.get("modified_at", 0),
        )
        return local_path

    def _remove_catalog_item(self, catalog_id: str) -> None:
        logger.debug("Removing file from catalog.", extra={"catalog_id": catalog_id})
        self.client.delete(f"files/{catalog_id}/")

    @_keep_metadata_in_sync
    def _upload_to_catalog(self, virtual_path: str, local_path: str) -> None:
        logger.debug("Uploading file to catalog.", extra={"virtual_path": virtual_path})
        with open(local_path, "rb") as f:
            response = self.client.post(
                "files/fromFile/",
                files={"file": (virtual_path, f)},
                data={"useArchiveContents": "false"},
                timeout=(FILE_API_CONNECT_TIMEOUT, FILE_API_READ_TIMEOUT),
            )
        catalog_id = response.json()["catalogId"]
        modified_at = time.time()
        fs_info = {
            "catalog_id": catalog_id,
            "type": "file",
            "name": virtual_path,
            "modified_at": modified_at,
            "size": os.path.getsize(local_path),
        }
        self._downloaded_files[catalog_id] = (local_path, modified_at)
        existing_info = self._fs_metadata.get(virtual_path)
        if existing_info:
            catalog_id = cast(str, existing_info["catalog_id"])
            self._remove_catalog_item(catalog_id)
            local_path, _ = self._downloaded_files.pop(catalog_id, ("", 0.0))
            if local_path:
                os.remove(local_path)
        self._fs_metadata[virtual_path] = fs_info
        self._fs_metadata_timestamp = modified_at

    @_keep_metadata_in_sync
    def rm_file(self, path: str) -> None:
        logger.debug("Removing node.", extra={"path": path})
        if not self.exists(path):
            raise FileNotFoundError()
        if self.isdir(path):
            self.rmdir(path)
            return
        if self.isfile(path):
            clear_path = self._strip_protocol(path).rstrip("/")
            info = self._fs_metadata[clear_path]
            catalog_id = cast(str, info["catalog_id"])
            self._remove_catalog_item(catalog_id)
            local_path, _ = self._downloaded_files.pop(catalog_id, ("", 0.0))
            if local_path:
                os.remove(local_path)

            self._fs_metadata.pop(clear_path)
            self._fs_metadata_timestamp = time.time()
            return
        raise NotImplementedError(f"No remove logic for node: {path}")

    @_keep_metadata_in_sync
    def cp_file(self, path1: str, path2: str, **kwargs: Any) -> None:
        logger.debug("Copy file.", extra={"src_path": path1, "dst_path": path2})
        if not self.exists(path1):
            raise FileNotFoundError()
        if self.exists(path2):
            raise FileExistsError()
        if self.isdir(path1):
            self.mkdir(path2)
            return
        if self.isfile(path1):
            parent = self._parent(path2)
            if not self.exists(parent):
                raise FileNotFoundError(parent)
            if not self.isdir(parent):
                raise ValueError(f"{parent} is not a directory")

            local_path = self._get_local_path(self.info(path1))
            self._upload_to_catalog(self._strip_protocol(path2).rstrip("/"), local_path)
            return
        raise NotImplementedError(f"No copy logic for node: {path1}")


def calculate_checksum(path: str) -> bytes:
    adder = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(8192):
            adder.update(chunk)
    return adder.digest()


def all_env_variables_present() -> bool:
    # check if all env variables are present
    expected_envs = ["DATAROBOT_ENDPOINT", "DATAROBOT_API_TOKEN", "APPLICATION_ID"]
    return not any(not os.environ.get(env_name) for env_name in expected_envs)


class _FileIOWrapper(io.FileIO):
    def __init__(
        self, fs_entity: DRFileSystem, virtual_path: str, name: str, mode: str
    ) -> None:
        super().__init__(name, mode)
        self._fs_entity = fs_entity
        self._virtual_path = virtual_path

    def close(self) -> None:
        upload_file = False
        if not self.closed:
            self.seek(0, io.SEEK_END)
            size = self.tell()
            upload_file = size > 0
        super().close()
        if upload_file:
            self._fs_entity._upload_to_catalog(self._virtual_path, self.name)
        else:
            logger.debug("Wrapper was empty")
