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
import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pytest
from datarobot.fs import DataRobotFileSystem
from fsspec.implementations.local import LocalFileSystem

from core.persistent_fs.dr_file_system import (
    DRFileSystem,
    FileInfo,
    LegacyDRFileSystem,
)

run_fs_tests = pytest.mark.skipif(
    not (
        os.environ.get("DATAROBOT_API_TOKEN")
        and os.environ.get("DATAROBOT_ENDPOINT")
        and os.environ.get("APPLICATION_ID")
        and os.environ.get("RUN_FS_TESTS", "").strip().lower() == "true"
    ),
    reason="Requires DATAROBOT_API_TOKEN, DATAROBOT_ENDPOINT, APPLICATION_ID, and RUN_FS_TESTS=True to be set",
)


def _tmp_dir() -> str:
    return f"_tmp_{time.time()}"


@contextmanager
def create_legacy_temp_dir(
    fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
) -> Iterator[str]:
    name = _tmp_dir()
    legacy_fs.mkdir(name)
    yield name
    fs.rm(name, recursive=True)
    assert fs.exists(name) is False
    assert legacy_fs.exists(name) is False


@run_fs_tests
class TestDRFileSystem:
    @pytest.fixture(scope="session")
    def fs(self) -> DRFileSystem:
        return DRFileSystem()

    @pytest.fixture(scope="session")
    def dr_fs(self) -> DataRobotFileSystem:
        # Need this to create files in new system *without* deleting file in legacy system for test.
        return DataRobotFileSystem()

    @pytest.fixture(scope="session")
    def legacy_fs(self) -> LegacyDRFileSystem:
        return LegacyDRFileSystem()

    def test_ls_and_info(
        self,
        fs: DRFileSystem,
        legacy_fs: LegacyDRFileSystem,
        dr_fs: DataRobotFileSystem,
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # root works
            assert f"{temp_dir}/" in fs.ls("", detail=False)
            assert f"{temp_dir}/" in fs.ls("/", detail=False)
            assert fs.ls(f"{temp_dir}", detail=True) == []
            with pytest.raises(FileNotFoundError):
                fs.ls(f"{temp_dir}/non-existent/")

            legacy_fs.mkdir(f"{temp_dir}/subdir")
            subdirs = cast(list[FileInfo], fs.ls(f"{temp_dir}", detail=True))
            assert subdirs[0]["name"] == f"{temp_dir}/subdir/"
            assert subdirs[0]["type"] == "directory"
            assert (
                subdirs[0]["catalog_id"] is None
            )  # legacy dirs have catalog_id None for dirs

            # create files in new and old system simultaneously
            legacy_fs.pipe_file(f"{temp_dir}/test.txt", b"test")
            dr_fs.pipe_file(f"{fs._catalog_id}/{temp_dir}/test2.txt", b"test2")
            assert fs.ls(f"{temp_dir}", detail=False) == [
                f"{temp_dir}/subdir/",
                f"{temp_dir}/test.txt",
                f"{temp_dir}/test2.txt",
            ]
            legacy_file = fs.info(f"{temp_dir}/test.txt")
            assert legacy_file["catalog_id"] != fs._catalog_id
            assert legacy_file["size"] == 4
            assert legacy_file["name"] == f"{temp_dir}/test.txt"
            assert legacy_file["type"] == "file"
            assert legacy_file["format"] == "unknown"

            new_file = fs.info(f"{temp_dir}/test2.txt")
            assert new_file["catalog_id"] == fs._catalog_id
            assert new_file["size"] == 5
            assert new_file["name"] == f"{temp_dir}/test2.txt"
            assert new_file["type"] == "file"
            assert new_file["format"] == "txt"

            # Override the old file with new one
            dr_fs.pipe_file(f"{fs._catalog_id}/{temp_dir}/test.txt", b"test_overritten")
            assert set(fs.ls(f"{temp_dir}", detail=False)) == {
                f"{temp_dir}/subdir/",
                f"{temp_dir}/test.txt",
                f"{temp_dir}/test2.txt",
            }
            # Override dir in new system
            dr_fs.pipe_file(f"{fs._catalog_id}/{temp_dir}/subdir/test3.txt", b"test3")
            dr_fs.pipe_file(
                f"{fs._catalog_id}/{temp_dir}/new_sub_dir/test4.txt", b"test4"
            )
            assert set(fs.ls(f"{temp_dir}", detail=False)) == {
                f"{temp_dir}/subdir/",
                f"{temp_dir}/test.txt",
                f"{temp_dir}/test2.txt",
                f"{temp_dir}/new_sub_dir/",
            }

            override_file = fs.info(f"{temp_dir}/test.txt")
            assert override_file["catalog_id"] == fs._catalog_id
            assert override_file["size"] == 15
            assert override_file["name"] == f"{temp_dir}/test.txt"
            assert override_file["type"] == "file"
            assert override_file["format"] == "txt"

            override_dir = fs.info(f"{temp_dir}/subdir/")
            assert override_dir["catalog_id"] == fs._catalog_id
            assert override_dir["size"] == 0
            assert override_dir["name"] == f"{temp_dir}/subdir/"
            assert override_dir["type"] == "directory"
            assert override_dir["format"] is None

    def test_info_root(self, fs: DRFileSystem) -> None:
        assert fs.info("") == {
            "name": "",
            "type": "directory",
            "size": 0,
            "format": None,
            "modified_at": None,
            "created_at": None,
            "catalog_id": fs._catalog_id,
        }
        assert fs.info("/") == fs.info("")

    def test_modified(self, fs: DRFileSystem) -> None:
        # check modified on new system returns created_at
        name = _tmp_dir()
        fs.pipe_file(f"{name}/test.txt", b"test")
        created_at = fs.info(f"{name}/test.txt")["created_at"]
        assert created_at is not None
        assert fs.modified(f"{name}/test.txt") == created_at.timestamp()
        fs.rm(name, recursive=True)

    def test_created_at(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        # check old records have new created_at field
        file_name = f"test_{time.time()}.txt"
        legacy_fs.pipe_file(file_name, b"test")
        assert fs.info(file_name)["created_at"] is None
        legacy_fs.rm(file_name)

    def test_du(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/test.txt", b"test")
            legacy_fs.pipe_file(f"{temp_dir}/test.txt", b"sdkfhweuihskjfbweiuweh")
            legacy_fs.pipe_file(f"{temp_dir}/test2.txt", b"test2")
            assert fs.du(temp_dir, total=True) == 9

    def test_mkdir(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        tmp = _tmp_dir()
        fs.mkdir(tmp)
        assert fs.exists(tmp)
        assert legacy_fs.exists(tmp)  # uses fallback
        fs.rmdir(tmp)
        assert fs.exists(tmp) is False
        assert legacy_fs.exists(tmp) is False

    def test_makedirs(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        tmp = _tmp_dir()
        tmp_subdir = f"{tmp}/subdir"
        fs.makedirs(tmp_subdir)
        assert fs.exists(tmp_subdir)
        assert fs.exists(tmp)
        assert legacy_fs.exists(tmp_subdir)  # uses fallback
        legacy_fs.rm(tmp, recursive=True)
        assert fs.exists(tmp) is False
        assert legacy_fs.exists(tmp) is False

    def test_find_glob_tree(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # Mixed bag: 5 files + directories created across both new and legacy systems
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_new")
            legacy_fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_old")
            legacy_fs.mkdir(f"{temp_dir}/subdir")
            fs.pipe_file(f"{temp_dir}/subdir/file3.txt", b"file3")
            legacy_fs.pipe_file(f"{temp_dir}/subdir/file4.json", b"file4")

            assert set(fs.find(temp_dir)) == {
                f"{temp_dir}/file1.txt",
                f"{temp_dir}/file2.csv",
                f"{temp_dir}/subdir/file3.txt",
                f"{temp_dir}/subdir/file4.json",
            }
            # withdirs=True also returns directory entries (with trailing slash)
            assert set(fs.find(temp_dir, withdirs=True)) == {
                f"{temp_dir}/",
                f"{temp_dir}/file1.txt",
                f"{temp_dir}/file2.csv",
                f"{temp_dir}/subdir/",
                f"{temp_dir}/subdir/file3.txt",
                f"{temp_dir}/subdir/file4.json",
            }
            # --- glob ---
            # all .txt files recursively across both systems
            assert set(fs.glob(f"{temp_dir}/**/*.txt")) == {
                f"{temp_dir}/file1.txt",
                f"{temp_dir}/subdir/file3.txt",
            }
            # direct children of temp_dir: files and subdirs (dirs have trailing slash)
            assert set(fs.glob(f"{temp_dir}/*")) == {
                f"{temp_dir}/file1.txt",
                f"{temp_dir}/file2.csv",
                f"{temp_dir}/subdir/",
            }
            # --- tree ---
            expected_tree = (
                f"{temp_dir}/\n"
                "├── subdir/\n"
                "│   ├── file3.txt\n"
                "│   └── file4.json\n"
                "├── file1.txt\n"
                "└── file2.csv"
            )
            assert fs.tree(temp_dir, recursion_limit=5) == expected_tree

    def test_cp_file_file_to_file(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # legacy only, copy legacy to new, delete legacy
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            assert legacy_fs.exists(f"{temp_dir}/file1.txt")
            fs.cp_file(f"{temp_dir}/file1.txt", f"{temp_dir}/file3.txt")
            assert fs.cat(f"{temp_dir}/file3.txt") == b"file1"
            assert legacy_fs.exists(f"{temp_dir}/file1.txt") is False

            # both exists, just copy new to new, ignore old
            fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_new")
            legacy_fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_old")
            fs.cp_file(f"{temp_dir}/file2.csv", f"{temp_dir}/file4.csv")
            assert fs.cat(f"{temp_dir}/file4.csv") == b"file2_new"
            assert legacy_fs.exists(f"{temp_dir}/file4.csv") is False

            # only new exists
            fs.pipe_file(f"{temp_dir}/file5.txt", b"file5")
            fs.cp_file(f"{temp_dir}/file5.txt", f"{temp_dir}/file6.txt")
            assert fs.cat(f"{temp_dir}/file6.txt") == b"file5"
            assert legacy_fs.exists(f"{temp_dir}/file5.txt") is False

            # copy sanity check
            legacy_fs.pipe_file(f"{temp_dir}/file7.txt", b"file7")
            fs.copy(f"{temp_dir}/file7.txt", f"{temp_dir}/file8.txt")
            assert fs.cat(f"{temp_dir}/file8.txt") == b"file7"
            assert legacy_fs.exists(f"{temp_dir}/file7.txt") is False

    def test_cp_dir_to_dir(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # legacy only
            legacy_fs.mkdir(f"{temp_dir}/subdir")
            fs.pipe_file(f"{temp_dir}/subdir", b"subdir file")
            legacy_fs.pipe_file(f"{temp_dir}/subdir/file7.txt", b"file7")
            fs.copy(f"{temp_dir}/subdir/", f"{temp_dir}/subdir2/", recursive=True)
            assert fs.exists(f"{temp_dir}/subdir2/")
            assert fs.cat(f"{temp_dir}/subdir2/file7.txt") == b"file7"
            assert legacy_fs.exists(f"{temp_dir}/subdir/file7.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir2/") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir/") is False

            # new only
            fs.pipe_file(f"{temp_dir}/subdir3/file8.txt", b"file8")
            fs.pipe_file(f"{temp_dir}/subdir3/extra_sub/file9.txt", b"file9")
            fs.copy(f"{temp_dir}/subdir3/", f"{temp_dir}/subdir4/", recursive=True)
            assert fs.exists(f"{temp_dir}/subdir4/")
            assert fs.cat(f"{temp_dir}/subdir4/file8.txt") == b"file8"
            assert fs.cat(f"{temp_dir}/subdir4/extra_sub/file9.txt") == b"file9"
            assert legacy_fs.exists(f"{temp_dir}/subdir3/") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir4/") is False

            # both, exist same name, some overlap
            fs.pipe_file(f"{temp_dir}/subdir5/file10.txt", b"file10")
            fs.pipe_file(f"{temp_dir}/subdir5/extra_sub/file11.txt", b"file11_new")
            fs.pipe_file(f"{temp_dir}/subdir5/extra_sub/file12.txt", b"file12")
            fs.pipe_file(f"{temp_dir}/subdir6", b"file13")
            legacy_fs.mkdir(f"{temp_dir}/subdir5/extra_sub")
            legacy_fs.pipe_file(f"{temp_dir}/subdir5/file13.txt", b"file13")
            legacy_fs.pipe_file(
                f"{temp_dir}/subdir5/extra_sub/file11.txt", b"file11_old"
            )
            legacy_fs.pipe_file(f"{temp_dir}/subdir5/extra_sub/file14.txt", b"file14")
            legacy_fs.mkdir(f"{temp_dir}/subdir5/another_dir")
            fs.copy(f"{temp_dir}/subdir5/", f"{temp_dir}/subdir6/", recursive=True)
            assert fs.exists(f"{temp_dir}/subdir6/")
            assert fs.cat(f"{temp_dir}/subdir6/file10.txt") == b"file10"
            assert fs.cat(f"{temp_dir}/subdir6/extra_sub/file11.txt") == b"file11_new"
            assert fs.cat(f"{temp_dir}/subdir6/extra_sub/file12.txt") == b"file12"
            assert fs.cat(f"{temp_dir}/subdir6/file13.txt") == b"file13"
            assert fs.cat(f"{temp_dir}/subdir6/extra_sub/file14.txt") == b"file14"
            assert fs.exists(f"{temp_dir}/subdir6/another_dir/") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir5/") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir5/another_dir/") is False
            assert legacy_fs.exists(f"{temp_dir}/subdir6") is False

    def test_cp_file_to_dir(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # copy new only to new
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            fs.pipe_file(f"{temp_dir}/dir/file2.txt", b"file2")
            fs.cp_file(f"{temp_dir}/file1.txt", f"{temp_dir}/dir/")
            assert fs.cat(f"{temp_dir}/dir/file1.txt") == b"file1_new"

            # copy legacy only to new
            legacy_fs.mkdir(f"{temp_dir}/dir/")
            legacy_fs.pipe_file(f"{temp_dir}/file3.txt", b"file3")
            fs.cp_file(f"{temp_dir}/file3.txt", f"{temp_dir}/dir/")
            assert fs.cat(f"{temp_dir}/dir/file3.txt") == b"file3"
            assert legacy_fs.exists(f"{temp_dir}/file3.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir/file3.txt") is False

            # copy legacy to legacy
            legacy_fs.mkdir(f"{temp_dir}/dir2/")
            legacy_fs.pipe_file(f"{temp_dir}/file4.txt", b"file4")
            fs.cp_file(f"{temp_dir}/file4.txt", f"{temp_dir}/dir2/")
            assert fs.cat(f"{temp_dir}/dir2/file4.txt") == b"file4"
            assert legacy_fs.exists(f"{temp_dir}/file4.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir2/file4.txt") is False

            # copy new to legacy
            fs.pipe_file(f"{temp_dir}/file5.txt", b"file5_new")
            fs.cp_file(f"{temp_dir}/file5.txt", f"{temp_dir}/dir2/")
            assert fs.cat(f"{temp_dir}/dir2/file5.txt") == b"file5_new"
            assert legacy_fs.exists(f"{temp_dir}/file5.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir2/file5.txt") is False

    def test_cp_mv_overwrites(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")

            fs.cp_file(f"{temp_dir}/file1.txt", f"{temp_dir}/file2.txt")

            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new_2")
            fs.cp_file(f"{temp_dir}/file1.txt", f"{temp_dir}/file2.txt")

            assert fs.cat(f"{temp_dir}/file2.txt") == b"file1_new_2"
            assert legacy_fs.exists(f"{temp_dir}/file1.txt") is False
            assert fs.ls(temp_dir, detail=False) == [
                f"{temp_dir}/file1.txt",
                f"{temp_dir}/file2.txt",
            ]

            fs.mv(f"{temp_dir}/file1.txt", f"{temp_dir}/file3.txt")
            fs.pipe_file(f"{temp_dir}/file2.txt", b"file2_new_3")
            fs.mv(f"{temp_dir}/file2.txt", f"{temp_dir}/file3.txt")
            assert fs.cat(f"{temp_dir}/file3.txt") == b"file2_new_3"
            assert fs.ls(temp_dir, detail=False) == [f"{temp_dir}/file3.txt"]

    def test_mv(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # new only
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            fs.mv(f"{temp_dir}/file1.txt", f"{temp_dir}/file2.txt")
            assert fs.exists(f"{temp_dir}/file1.txt") is False
            assert fs.exists(f"{temp_dir}/file2.txt")
            assert legacy_fs.exists(f"{temp_dir}/file2.txt") is False

            fs.pipe_file(f"{temp_dir}/dir/file3.txt", b"file3_new")
            fs.mv(f"{temp_dir}/dir/", f"{temp_dir}/dir2/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir/") is False
            assert fs.exists(f"{temp_dir}/dir2/")
            assert fs.exists(f"{temp_dir}/dir2/file3.txt")
            assert legacy_fs.exists(f"{temp_dir}/dir2/") is False

            # legacy only
            legacy_fs.pipe_file(f"{temp_dir}/file4.txt", b"file4")
            fs.mv(f"{temp_dir}/file4.txt", f"{temp_dir}/file5.txt")
            assert fs.exists(f"{temp_dir}/file4.txt") is False
            assert fs.exists(f"{temp_dir}/file5.txt")
            assert legacy_fs.exists(f"{temp_dir}/file5.txt") is False

            legacy_fs.mkdir(f"{temp_dir}/dir3/")
            legacy_fs.pipe_file(f"{temp_dir}/dir3/file4.txt", b"file4")
            fs.mv(f"{temp_dir}/dir3/", f"{temp_dir}/dir4/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir3/") is False
            assert fs.exists(f"{temp_dir}/dir4/")
            assert fs.exists(f"{temp_dir}/dir4/file4.txt")
            assert legacy_fs.exists(f"{temp_dir}/dir4/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir4/file4.txt") is False

            # both
            fs.pipe_file(f"{temp_dir}/file5.txt", b"file5_new")
            legacy_fs.pipe_file(f"{temp_dir}/file5.txt", b"file5")
            fs.mv(f"{temp_dir}/file5.txt", f"{temp_dir}/file6.txt")
            assert fs.exists(f"{temp_dir}/file5.txt") is False
            assert fs.cat(f"{temp_dir}/file6.txt") == b"file5_new"
            assert legacy_fs.exists(f"{temp_dir}/file6.txt") is False

            fs.pipe_file(f"{temp_dir}/dir5/subdir/fileA.txt", b"fileA_new")
            legacy_fs.mkdir(f"{temp_dir}/dir5/subdir/")
            legacy_fs.pipe_file(f"{temp_dir}/dir5/subdir/fileA.txt", b"fileA")
            legacy_fs.pipe_file(f"{temp_dir}/dir5/subdir/fileB.txt", b"fileB")
            fs.mv(f"{temp_dir}/dir5/", f"{temp_dir}/dir6/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir5/") is False
            assert fs.exists(f"{temp_dir}/dir6/")
            assert fs.exists(f"{temp_dir}/dir6/subdir/")
            assert fs.cat(f"{temp_dir}/dir6/subdir/fileA.txt") == b"fileA_new"
            assert fs.cat(f"{temp_dir}/dir6/subdir/fileB.txt") == b"fileB"
            assert legacy_fs.exists(f"{temp_dir}/dir6/subdir/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir6/subdir/fileB.txt") is False

    def test_remove(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            # rm_file
            # new only
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            fs.rm_file(f"{temp_dir}/file1.txt")
            assert fs.exists(f"{temp_dir}/file1.txt") is False
            fs.pipe_file(f"{temp_dir}/dir/file1.txt", b"file1_new")
            fs.rm_file(f"{temp_dir}/dir/")
            assert fs.exists(f"{temp_dir}/dir/file1.txt") is False
            assert fs.exists(f"{temp_dir}/dir/") is False

            # legacy only
            legacy_fs.pipe_file(f"{temp_dir}/file2.txt", b"file2")
            legacy_fs.mkdir(f"{temp_dir}/dir2/")
            legacy_fs.pipe_file(f"{temp_dir}/dir2/file3.txt", b"file3")
            fs.rm_file(f"{temp_dir}/file2.txt")
            assert fs.exists(f"{temp_dir}/file2.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/file2.txt") is False
            fs.rm_file(f"{temp_dir}/dir2/")
            assert fs.exists(f"{temp_dir}/dir2/file3.txt") is False
            assert fs.exists(f"{temp_dir}/dir2/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir2/file3.txt") is False

            # both
            fs.pipe_file(f"{temp_dir}/file3.txt", b"file3_new")
            legacy_fs.pipe_file(f"{temp_dir}/file3.txt", b"file3")
            fs.rm_file(f"{temp_dir}/file3.txt")
            assert fs.exists(f"{temp_dir}/file3.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/file3.txt") is False

            fs.pipe_file(f"{temp_dir}/dir3/subdir/file3.txt", b"file3_new")
            legacy_fs.mkdir(f"{temp_dir}/dir3/subdir/")
            legacy_fs.pipe_file(f"{temp_dir}/dir3/subdir/file4.txt", b"file4")
            fs.rm_file(f"{temp_dir}/dir3/")
            assert fs.exists(f"{temp_dir}/dir3/subdir/file3.txt") is False
            assert fs.exists(f"{temp_dir}/dir3/subdir/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir3/subdir/file4.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir3/subdir/") is False

            # neither
            fs.rm_file(f"{temp_dir}/non-existent.txt")
            fs.rm_file(f"{temp_dir}/non-existent/")
            assert fs.exists(f"{temp_dir}/non-existent.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/non-existent/") is False

            # rm_directory syntax
            with pytest.raises(ValueError):
                fs.rm_directory(f"{temp_dir}/file3.txt")

            # rm
            # new only
            fs.pipe_file(f"{temp_dir}/file3.txt", b"file3_new")
            fs.rm(f"{temp_dir}/file*.txt")
            assert fs.exists(f"{temp_dir}/file3.txt") is False
            fs.pipe_file(f"{temp_dir}/dir4/subdir/fileA.txt", b"fileA")
            fs.pipe_file(f"{temp_dir}/dir4/fileB.txt", b"fileB")
            fs.rm(f"{temp_dir}/dir4/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir4/") is False
            assert fs.exists(f"{temp_dir}/dir4/subdir/") is False
            assert fs.exists(f"{temp_dir}/dir4/fileB.txt") is False

            # legacy only
            legacy_fs.pipe_file(f"{temp_dir}/file4.txt", b"file4")
            fs.rm(f"{temp_dir}/file4.txt")
            assert legacy_fs.exists(f"{temp_dir}/file4.txt") is False
            legacy_fs.mkdir(f"{temp_dir}/dir5/")
            legacy_fs.pipe_file(f"{temp_dir}/dir5/file5.txt", b"file5")
            fs.rm(f"{temp_dir}/dir5/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir5/") is False
            assert fs.exists(f"{temp_dir}/dir5/file5.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir5/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir5/file5.txt") is False

            # both
            fs.pipe_file(f"{temp_dir}/file5.txt", b"file5_new")
            legacy_fs.pipe_file(f"{temp_dir}/file5.txt", b"file5")
            fs.rm(f"{temp_dir}/file5.txt")
            assert fs.exists(f"{temp_dir}/file5.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/file5.txt") is False
            fs.pipe_file(f"{temp_dir}/dir6/subdir/fileA.txt", b"fileA")
            legacy_fs.mkdir(f"{temp_dir}/dir6/subdir/")
            legacy_fs.pipe_file(f"{temp_dir}/dir6/subdir/fileB.txt", b"fileB")
            fs.rm(f"{temp_dir}/dir6/", recursive=True)
            assert fs.exists(f"{temp_dir}/dir6/") is False
            assert fs.exists(f"{temp_dir}/dir6/subdir/") is False
            assert fs.exists(f"{temp_dir}/dir6/subdir/fileB.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/file5.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir6/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir6/subdir/") is False
            assert legacy_fs.exists(f"{temp_dir}/dir6/subdir/fileB.txt") is False

            # both with glob pattern
            fs.pipe_file(f"{temp_dir}/dir/file7.txt", b"file7_new")
            legacy_fs.mkdir(f"{temp_dir}/dir/")
            legacy_fs.pipe_file(f"{temp_dir}/dir/file7.txt", b"file7")
            fs.pipe_file(f"{temp_dir}/dir/file8.csv", b"file8")
            fs.rm(f"{temp_dir}/dir/*.txt")
            assert fs.exists(f"{temp_dir}/dir/file7.txt") is False
            assert fs.exists(f"{temp_dir}/dir/file8.csv")
            assert legacy_fs.exists(f"{temp_dir}/dir/file7.txt") is False
            assert legacy_fs.exists(f"{temp_dir}/dir/file8.csv") is False

            # neither
            fs.rm(f"{temp_dir}/non-existent.txt")
            fs.rm(f"{temp_dir}/non-existent/")

    def test_cat(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")  # new only
            fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_new")  # both
            legacy_fs.pipe_file(f"{temp_dir}/file2.csv", b"file2_old")  # both
            legacy_fs.pipe_file(f"{temp_dir}/file3.txt", b"file3")  # legacy only

            assert fs.cat(f"{temp_dir}/file1.txt") == b"file1"
            assert fs.cat_file(f"{temp_dir}/file1.txt", start=2, end=6) == b"le1"
            assert fs.cat_file(f"{temp_dir}/file2.csv") == b"file2_new"
            assert fs.cat(f"{temp_dir}/file3.txt") == b"file3"
            with pytest.raises(FileNotFoundError):
                fs.cat(f"{temp_dir}/file4.txt")

    def test_reads(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            legacy_fs.pipe_file(f"{temp_dir}/file2.txt", b"file2")

            # new only
            assert fs.open(f"{temp_dir}/file1.txt", mode="rb").read() == b"file1_new"
            assert fs.read_text(f"{temp_dir}/file1.txt") == "file1_new"
            assert fs.head(f"{temp_dir}/file1.txt", size=2) == b"fi"

            # legacy only
            assert fs.open(f"{temp_dir}/file2.txt", mode="rb").read() == b"file2"
            assert fs.read_text(f"{temp_dir}/file2.txt") == "file2"
            assert fs.head(f"{temp_dir}/file2.txt", size=2) == b"fi"

            # both
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            assert fs.open(f"{temp_dir}/file1.txt", mode="rb").read() == b"file1_new"
            assert fs.read_text(f"{temp_dir}/file1.txt") == "file1_new"
            assert fs.head(f"{temp_dir}/file1.txt", size=2) == b"fi"

            with pytest.raises(FileNotFoundError):
                fs.open(f"{temp_dir}/file3.txt")

    def test_writes(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            legacy_fs.pipe_file(f"{temp_dir}/file2.txt", b"file2")

            fs.touch(f"{temp_dir}/file3.txt")
            assert fs.exists(f"{temp_dir}/file3.txt")
            assert legacy_fs.exists(f"{temp_dir}/file3.txt") is False

            assert legacy_fs.exists(f"{temp_dir}/file2.txt")
            fs.pipe_file(f"{temp_dir}/file2.txt", b"file2_new")
            assert fs.cat(f"{temp_dir}/file2.txt") == b"file2_new"
            assert legacy_fs.exists(f"{temp_dir}/file2.txt") is False

            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(b"tmp_file_new")
                tmp_file.flush()
                fs.put(tmp_file.name, f"{temp_dir}/file4.txt")
                assert fs.cat(f"{temp_dir}/file4.txt") == b"tmp_file_new"
                assert legacy_fs.exists(f"{temp_dir}/file4.txt") is False

    def test_writes_overwrites_by_default(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new_2")
            assert fs.cat(f"{temp_dir}/file1.txt") == b"file1_new_2"
            assert legacy_fs.exists(f"{temp_dir}/file1.txt") is False
            assert fs.ls(temp_dir, detail=False) == [f"{temp_dir}/file1.txt"]

            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(b"tmp_file_new")
                tmp_file.flush()
                fs.put(tmp_file.name, f"{temp_dir}/file1.txt")
                assert fs.cat(f"{temp_dir}/file1.txt") == b"tmp_file_new"
                assert fs.ls(temp_dir, detail=False) == [f"{temp_dir}/file1.txt"]

            with fs.open(f"{temp_dir}/file1.txt", "wb") as file_overwrite_stream:
                file_overwrite_stream.write(b"tmp_file_new_2")
            assert fs.cat(f"{temp_dir}/file1.txt") == b"tmp_file_new_2"
            assert fs.ls(temp_dir, detail=False) == [f"{temp_dir}/file1.txt"]

    def test_write_open_error_before_close(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            # If file write does not complete, the legacy file should not be removed.
            with pytest.raises(ValueError):
                file_overwrite_stream = fs.open(f"{temp_dir}/file1.txt", "wb")
                file_overwrite_stream.write(b"file_new_error")
                raise ValueError("test error")
            assert legacy_fs.exists(f"{temp_dir}/file1.txt")
            assert fs.info(f"{temp_dir}/file1.txt")["catalog_id"] != fs._catalog_id

            with fs.open(f"{temp_dir}/file1.txt", "wb") as f:
                f.write(b"file_new")
            assert fs.cat(f"{temp_dir}/file1.txt") == b"file_new"
            assert legacy_fs.exists(f"{temp_dir}/file1.txt") is False

    def test_download(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        lfs = LocalFileSystem()
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
                fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
                legacy_fs.pipe_file(f"{temp_dir}/file2.txt", b"file2")

                # new
                fs.download(f"{temp_dir}/file1.txt", local_tmp_dir)
                assert lfs.exists(os.path.join(local_tmp_dir, "file1.txt"))
                assert (
                    lfs.read_text(os.path.join(local_tmp_dir, "file1.txt"))
                    == "file1_new"
                )

                # legacy
                fs.download(f"{temp_dir}/file2.txt", local_tmp_dir)
                assert lfs.exists(os.path.join(local_tmp_dir, "file2.txt"))
                assert (
                    lfs.read_text(os.path.join(local_tmp_dir, "file2.txt")) == "file2"
                )

                # both
                fs.get(f"{temp_dir}/file1.txt", local_tmp_dir)
                assert lfs.exists(os.path.join(local_tmp_dir, "file1.txt"))
                assert (
                    lfs.read_text(os.path.join(local_tmp_dir, "file1.txt"))
                    == "file1_new"
                )
                with pytest.raises(FileNotFoundError):
                    fs.download(f"{temp_dir}/file3.txt", local_tmp_dir)

    def test_mapper(self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")

            mapper = fs.get_mapper("")
            assert mapper[f"{temp_dir}/file1.txt"] == b"file1_new"
            mapper[f"{temp_dir}/file2.txt"] = b"file2.txt"
            assert mapper[f"{temp_dir}/file2.txt"] == b"file2.txt"
            assert f"{temp_dir}/file2.txt" in mapper

            with pytest.raises(KeyError):
                mapper["file3.txt"]

    def test_list_unmigrated_files(
        self, fs: DRFileSystem, legacy_fs: LegacyDRFileSystem
    ) -> None:
        with create_legacy_temp_dir(fs, legacy_fs) as temp_dir:
            fs.pipe_file(f"{temp_dir}/file1.txt", b"file1_new")
            fs.pipe_file(f"{temp_dir}/dir1/file2.txt", b"file2_new")
            legacy_fs.pipe_file(f"{temp_dir}/file1.txt", b"file1")
            legacy_fs.pipe_file(f"{temp_dir}/file3.txt", b"file3")
            legacy_fs.mkdir(f"{temp_dir}/dir1/dir2/", create_parents=True)
            legacy_fs.pipe_file(f"{temp_dir}/dir1/file2.txt", b"file2")
            legacy_fs.pipe_file(f"{temp_dir}/dir1/dir2/file3.txt", b"file3")
            assert [f["name"] for f in fs.list_unmigrated_files(temp_dir)] == [
                f"{temp_dir}/dir1/dir2/file3.txt",
                f"{temp_dir}/file3.txt",
            ]
            assert [
                f["name"] for f in fs.list_unmigrated_files(f"{temp_dir}/dir1/")
            ] == [f"{temp_dir}/dir1/dir2/file3.txt"]
