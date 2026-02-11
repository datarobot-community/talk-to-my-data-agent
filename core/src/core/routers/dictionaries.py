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

"""Router for data dictionary endpoints."""

from __future__ import annotations

import io
from typing import cast

from fastapi import APIRouter, Depends, HTTPException, Response

from core.analyst_db import AnalystDB
from core.deps import get_initialized_db
from core.schema import (
    DataDictionary,
    DataDictionaryResponse,
    DictionaryCellUpdate,
)

router = APIRouter(prefix="/dictionaries", tags=["dictionaries"])


@router.get("")
async def get_dictionaries(
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> list[DataDictionaryResponse]:
    """Get all data dictionaries."""
    # Get datasets from the database
    dataset_names = await analyst_db.list_analyst_datasets()

    # Fetch dictionaries for each dataset
    dictionaries = []
    for name in dataset_names:
        dictionary = await analyst_db.get_data_dictionary(name)
        if dictionary and len(dictionary.column_descriptions):
            dictionaries.append(cast(DataDictionaryResponse, dictionary))
        else:
            dictionaries.append(
                DataDictionaryResponse(
                    name=name, column_descriptions=[], in_progress=True
                )
            )

    return dictionaries if dictionaries else []


@router.delete("/{name}", status_code=200)
async def delete_dictionary(
    name: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> None:
    """Delete a data dictionary by name."""
    await analyst_db.delete_table(name)


@router.get("/{name}/download")
async def download_dictionary(
    name: str,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    bom: bool = False,
) -> Response:
    """
    Download a dictionary as a CSV file.

    Args:
        name: Name of the dataset whose dictionary to download
        bom: Whether to include UTF-8 BOM for Excel compatibility

    Returns:
        CSV file attachment
    """
    dictionary = await analyst_db.get_data_dictionary(name)

    if not dictionary:
        raise HTTPException(status_code=404, detail=f"Dictionary '{name}' not found")

    # Convert the dictionary to a DataFrame
    df = dictionary.to_application_df()

    # Convert to CSV
    csv_content = io.StringIO()
    df.write_csv(csv_content)

    # Create response with CSV attachment
    csv_text = csv_content.getvalue()
    if bom:
        csv_text = (
            "\ufeff" + csv_text
        )  # Prefix UTF-8 BOM for Excel compatibility with international characters
    response = Response(content=csv_text)
    response.headers["Content-Type"] = "text/csv; charset=utf-8"

    return response


@router.patch("/{name}/cells")
async def update_dictionary_cell(
    name: str,
    update: DictionaryCellUpdate,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> DataDictionary:
    """Update a specific cell in a data dictionary."""
    dictionary = await analyst_db.get_data_dictionary(name)

    if not dictionary:
        raise HTTPException(status_code=404, detail=f"Dictionary '{name}' not found")

    if update.rowIndex < 0 or update.rowIndex >= len(dictionary.column_descriptions):
        raise HTTPException(
            status_code=400, detail=f"Row index {update.rowIndex} is out of range"
        )

    column_description = dictionary.column_descriptions[update.rowIndex]
    if hasattr(column_description, update.field):
        setattr(column_description, update.field, update.value)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Field '{update.field}' not found in dictionary row",
        )

    await analyst_db.register_data_dictionary(dictionary, clobber=True)

    return dictionary
