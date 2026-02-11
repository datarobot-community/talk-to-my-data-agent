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

"""Router for dataset endpoints."""

from __future__ import annotations

import io
import json
import os
from typing import List

import polars as pl
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

from core.analyst_db import AnalystDB, DatasetMetadata, InternalDataSourceType
from core.api import (
    load_registry_datasets,
    log_memory,
    process_data_and_update_state,
    register_remote_registry_datasets,
)
from core.datarobot_client import use_user_token
from core.deps import get_initialized_db
from core.file_utils import detect_and_decode_csv, load_and_validate_csv
from core.logging_helper import get_logger
from core.schema import (
    AnalystDataset,
    DatasetCleansedResponse,
    FileUploadResponse,
)

logger = get_logger()

router = APIRouter(prefix="/datasets", tags=["datasets"])


async def process_and_update(
    dataset_names: List[str],
    analyst_db: AnalystDB,
    datasource_type: str | InternalDataSourceType,
) -> None:
    """Process datasets and update state."""
    async for _ in process_data_and_update_state(
        dataset_names, analyst_db, datasource_type
    ):
        pass


@router.post("/upload")
async def upload_files(
    request: Request,
    background_tasks: BackgroundTasks,
    data_source: str | InternalDataSourceType | None = None,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    files: List[UploadFile] | None = None,
    registry_ids: str | None = Form(None),
) -> list[FileUploadResponse]:
    """Upload CSV/Excel files or load datasets from the Data Registry."""
    normalized_data_source: InternalDataSourceType = (
        InternalDataSourceType.FILE
        if data_source is None
        else InternalDataSourceType(data_source)
        if isinstance(data_source, str)
        else data_source
    )
    dataset_names = []
    response: list[FileUploadResponse] = []
    if files:
        for file in files:
            try:
                file_size = file.size or 0
                contents = await file.read()
                if file.filename is None:
                    continue

                file_extension = os.path.splitext(file.filename)[1].lower()

                if file_extension == ".csv":
                    logger.info(f"Loading CSV: {file.filename}")
                    log_memory()

                    try:
                        decoded_content = detect_and_decode_csv(contents, file.filename)
                        df = load_and_validate_csv(decoded_content, file.filename)
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=str(e))
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to read CSV file '{file.filename}': {str(e)}",
                        )

                    log_memory()
                    dataset_name = os.path.splitext(file.filename)[0]
                    dataset = AnalystDataset(name=dataset_name, data=df)

                    # Register dataset with the database
                    await analyst_db.register_dataset(
                        dataset, InternalDataSourceType.FILE, file_size=file_size
                    )

                    # Add to processing queue
                    dataset_names.append(dataset.name)

                    file_response: FileUploadResponse = {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "size": len(contents),
                        "dataset_name": dataset_name,
                    }
                    response.append(file_response)

                elif file_extension in [".xlsx", ".xls"]:
                    base_name = os.path.splitext(file.filename)[0]

                    try:
                        excel_dataset = pl.read_excel(
                            io.BytesIO(contents), sheet_id=None
                        )
                    except Exception as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Unable to read Excel file '{file.filename}'. Error: {str(e)}",
                        )

                    if isinstance(excel_dataset, dict):
                        for sheet_name, data in excel_dataset.items():
                            dataset_name = f"{base_name}_{sheet_name}"
                            dataset = AnalystDataset(name=dataset_name, data=data)
                            await analyst_db.register_dataset(
                                dataset,
                                InternalDataSourceType.FILE,
                                file_size=file_size,
                            )
                            # Add to processing queue
                            dataset_names.append(dataset.name)

                            excel_sheet_response: FileUploadResponse = {
                                "filename": file.filename,
                                "content_type": file.content_type,
                                "size": len(contents),
                                "dataset_name": dataset_name,
                            }
                            response.append(excel_sheet_response)
                    elif isinstance(excel_dataset, pl.DataFrame):
                        dataset_name = base_name
                        dataset = AnalystDataset(name=dataset_name, data=excel_dataset)
                        await analyst_db.register_dataset(
                            dataset,
                            InternalDataSourceType.FILE,
                            file_size=file_size,
                        )
                        # Add to processing queue
                        dataset_names.append(dataset.name)

                        excel_file_response: FileUploadResponse = {
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "size": len(contents),
                            "dataset_name": dataset_name,
                        }
                        response.append(excel_file_response)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

            except Exception as e:
                error_response: FileUploadResponse = {
                    "filename": file.filename or "unknown_file",
                    "error": str(e),
                }
                response.append(error_response)

    # Process the data in the background (cleansing and dictionary generation)
    if dataset_names:
        background_tasks.add_task(
            process_and_update,
            dataset_names,
            analyst_db,
            InternalDataSourceType.FILE,
        )

    if registry_ids:
        id_list: list[str] = json.loads(registry_ids)
        if id_list:
            with use_user_token(request):
                if normalized_data_source == InternalDataSourceType.REMOTE_REGISTRY:
                    dataframes, tasks = await register_remote_registry_datasets(
                        request, id_list, analyst_db
                    )
                else:
                    dataframes = await load_registry_datasets(id_list, analyst_db)
                    tasks = []
                for func, args, kwargs in tasks:
                    background_tasks.add_task(func, *args, **kwargs)
                dataset_names = [
                    dataset.name for dataset in dataframes if not dataset.error
                ]
                background_tasks.add_task(
                    process_and_update,
                    dataset_names,
                    analyst_db,
                    InternalDataSourceType.REMOTE_REGISTRY
                    if normalized_data_source == InternalDataSourceType.REMOTE_REGISTRY
                    else InternalDataSourceType.REGISTRY,
                )
                for dts in dataframes:
                    dts_response: FileUploadResponse = {
                        "dataset_name": dts.name,
                        "error": dts.error,
                    }
                    response.append(dts_response)

    return response


@router.get("/{dataset_id}")
async def get_dataset_by_id(
    dataset_id: str,
    skip: int = 0,
    limit: int = 1000,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> DatasetCleansedResponse:
    """
    Get a dataset by its ID with pagination support.

    Args:
        dataset_id: The unique identifier of the dataset
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)

    Returns:
        A response containing the dataset (similar to cleansed dataset format)

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        from core.analyst_db import DatasetType

        df = await analyst_db.dataset_handler.get_dataframe(
            dataset_id,
            expected_type=DatasetType.ANALYST_RESULT_DATASET,
            max_rows=None,
        )

        if skip > 0 or limit > 0:
            df = df.slice(skip, limit)

        metadata = await analyst_db.dataset_handler.get_dataset_metadata(dataset_id)

        dataset = AnalystDataset(
            name=metadata.original_name,
            data=df,
        )

        # Return in the same format as cleansed dataset
        return DatasetCleansedResponse(
            dataset_name=metadata.original_name,
            cleaning_report=None,  # No cleaning report for analyst datasets
            dataset=dataset,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error retrieving dataset '{dataset_id}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dataset: {str(e)}"
        )


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    analyst_db: AnalystDB = Depends(get_initialized_db),
    bom: bool = False,
) -> Response:
    """
    Download a dataset by ID as a CSV file.

    Args:
        dataset_id: The unique identifier of the dataset to download
        bom: Whether to include UTF-8 BOM for Excel compatibility (default: False)

    Returns:
        CSV file attachment

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        from core.analyst_db import DatasetType

        df = await analyst_db.dataset_handler.get_dataframe(
            dataset_id,
            expected_type=DatasetType.ANALYST_RESULT_DATASET,
            max_rows=None,
        )

        csv_content = io.StringIO()
        df.write_csv(csv_content)

        csv_text = csv_content.getvalue()
        if bom:
            csv_text = "\ufeff" + csv_text

        response = Response(content=csv_text)
        response.headers["Content-Type"] = "text/csv; charset=utf-8"

        filename = f"analysis_result_dataset_{dataset_id[:8]}.csv"
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{dataset_id}' not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error downloading dataset '{dataset_id}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error downloading dataset: {str(e)}"
        )


@router.get("/{name}/metadata")
async def get_dataset_metadata(
    name: str, analyst_db: AnalystDB = Depends(get_initialized_db)
) -> DatasetMetadata:
    """
    Get metadata for a dataset by name from the database.

    Args:
        name: The name of the dataset

    Returns:
        A dictionary containing dataset metadata including type, creation date, columns, and row count

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        return await analyst_db.get_dataset_metadata(name)
    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Dataset metadata not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving dataset metadata: {str(e)}"
        )


@router.get("/{name}/cleansed")
async def get_cleansed_dataset(
    name: str,
    skip: int = 0,
    limit: int = 10000,
    search: str | None = None,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> DatasetCleansedResponse:
    """
    Get a cleansed dataset by name from the database with pagination and search support.

    Args:
        name: The name of the dataset
        skip: Number of records to skip (for pagination)
        limit: Maximum number of records to return (for pagination)
        search: Optional search term to filter columns by name (raw data view)

    Returns:
        A dictionary containing the cleaning report (if available) and the dataset as a list of records.

    Raises:
        HTTPException: If the dataset doesn't exist or cannot be retrieved
    """
    try:
        ds_display = await analyst_db.get_dataset(name)

        # Initialize the response structure
        response = DatasetCleansedResponse(
            dataset_name=name,
            cleaning_report=None,
            dataset=None,
        )

        # Attempt to fetch the cleaning report
        try:
            ds_display_cleansed = await analyst_db.get_cleansed_dataset(name)
            response.cleaning_report = ds_display_cleansed.generate_cleaning_report()

        except ValueError:
            # Cleaning report is not available
            response.cleaning_report = None

        # Convert the dataset to a DataFrame
        df_display = ds_display.to_df()

        # Apply search filter if provided - filter columns by name
        if search and search.strip():
            search_term = search.strip().lower()
            original_columns = list(df_display.columns)
            matching_columns = [
                col for col in original_columns if search_term in col.lower()
            ]
            if matching_columns:
                # Select only the matching columns
                try:
                    df_display = df_display.select(matching_columns)
                except Exception as e:
                    logger.error(f"Error selecting columns: {e}")
                    # Fallback to original dataframe if selection fails
                    pass
            else:
                # No matching columns, create empty dataframe
                df_display = df_display.select([]).limit(0)

        # Apply pagination (skip and limit)
        if skip > 0 or limit > 0:
            df_display = df_display.slice(skip, limit)

        # Create an instance of AnalystDataset
        dataset = AnalystDataset(
            name=name,
            columns=df_display.columns,
            data=df_display.to_dicts(),  # Convert rows to a list of dictionaries
        )

        # Add the dataset to the response
        response.dataset = dataset

        return response

    except ValueError as e:
        raise HTTPException(
            status_code=404, detail=f"Dataset '{name}' not found: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error retrieving cleansed dataset '{name}': {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving cleansed dataset: {str(e)}"
        )


@router.delete("", status_code=200)
async def delete_datasets(
    request: Request,
    analyst_db: AnalystDB = Depends(get_initialized_db),
) -> None:
    """Delete all datasets."""
    await analyst_db.delete_all_tables()
