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
"""properdocs hooks: generate docs/README.md and inject CSS from .bin/."""

import re
from pathlib import Path

from mkdocs.structure.files import File

EXCLUDED_NAMES = {"README.md", "Taskfile.yaml", "Taskfile.yml"}


def _get_doc_files(docs_dir: Path) -> list[Path]:
    doc_files = []
    for item in sorted(docs_dir.iterdir()):
        if item.name in EXCLUDED_NAMES or item.name.startswith("."):
            continue
        if item.is_file() and item.suffix == ".md":
            doc_files.append(item)
        elif item.is_dir():
            readme = item / "README.md"
            if readme.exists():
                doc_files.append(readme)
            for subfile in sorted(item.iterdir()):
                if subfile.is_file() and subfile.suffix == ".md" and subfile != readme:
                    doc_files.append(subfile)
    return doc_files


def _extract_title(content: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _extract_description(content: str) -> str:
    """Return the first non-heading, non-empty paragraph line."""
    in_fence = False
    in_frontmatter = False
    lines = content.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == 0 and stripped == "---":
            in_frontmatter = True
            continue
        if in_frontmatter:
            if stripped in ("---", "..."):
                in_frontmatter = False
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not stripped or stripped.startswith("#"):
            continue
        return stripped
    return ""


def _build_readme(doc_files: list[Path], docs_dir: Path) -> str:
    toc_lines: list[str] = []
    docs_yaml: list[str] = []

    for doc_file in doc_files:
        content = doc_file.read_text(encoding="utf-8").strip()
        relative = doc_file.relative_to(docs_dir)
        fallback = relative.stem if relative.stem != "README" else relative.parent.name
        title = _extract_title(content, fallback)
        description = _extract_description(content)

        posix = relative.as_posix()
        toc_lines.append(f"- [{title}]({posix})")
        if description:
            toc_lines.append(f"  {description}")

        docs_yaml.append(f"  - {posix}")

    toc = "\n".join(toc_lines)
    docs_list = "\n".join(docs_yaml)

    return f"""\
---
source: docs/
docs:
{docs_list}
---

# Documentation

## Table of Contents

{toc}
"""


def on_files(files, config):
    """Inject static assets from .bin/ into the site without exposing them in docs_dir."""
    bin_dir = Path(__file__).parent
    for asset in (bin_dir / "stylesheets").glob("*.css"):
        files.append(
            File(
                path=f"stylesheets/{asset.name}",
                src_dir=str(bin_dir),
                dest_dir=config["site_dir"],
                use_directory_urls=config["use_directory_urls"],
            )
        )
    return files


def on_pre_build(config) -> None:
    """Generate docs/README.md before each build."""
    docs_dir = Path(config["docs_dir"])
    output_file = docs_dir / "README.md"

    doc_files = _get_doc_files(docs_dir)
    if not doc_files:
        return

    content = _build_readme(doc_files, docs_dir)

    # Skip write if content unchanged — prevents serve-mode rebuild loop
    if output_file.exists() and output_file.read_text(encoding="utf-8") == content:
        return

    output_file.write_text(content, encoding="utf-8")
    print(f"properdocs hook: generated README.md from {len(doc_files)} doc(s)")  # noqa: T201
