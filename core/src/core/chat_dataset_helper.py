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

"""Helper functions for managing datasets in chat messages."""

import uuid

from core.analyst_db import AnalystDB, DatasetType, InternalDataSourceType
from core.logging_helper import get_logger
from core.schema import (
    AnalystChatMessage,
    RunAnalysisResult,
    RunDatabaseAnalysisResult,
)

logger = get_logger("ChatMessageDatasetHelper")


async def extract_and_store_datasets(
    analyst_db: AnalystDB,
    message: AnalystChatMessage,
) -> AnalystChatMessage:
    """
    Extract datasets from message components and store them separately,
    replacing them with dataset_id references.

    Args:
        analyst_db: The analyst database instance
        message: The message to process

    Returns:
        Modified message with dataset_id references instead of full datasets
    """
    modified_message = message.model_copy(deep=True)

    for component in modified_message.components:
        if isinstance(component, (RunAnalysisResult, RunDatabaseAnalysisResult)):
            if component.dataset is not None:
                # Generate unique dataset_id
                dataset_id = str(uuid.uuid4())

                # Store dataset in datasets table
                await analyst_db.dataset_handler.register_dataframe(
                    component.dataset.to_df(),
                    name=dataset_id,
                    dataset_type=DatasetType.ANALYST_RESULT_DATASET,
                    data_source=InternalDataSourceType.GENERATED,
                    external_id=dataset_id,
                    original_name=component.dataset.name,
                    clobber=True,
                )

                # Set dataset_id (dataset is automatically excluded from JSON by Pydantic)
                component.dataset_id = dataset_id

    return modified_message


async def cleanup_message_datasets(
    analyst_db: AnalystDB,
    message: AnalystChatMessage,
) -> None:
    """
    Clean up datasets associated with a message.

    Args:
        analyst_db: The analyst database instance
        message: The message whose datasets should be cleaned up
    """
    for component in message.components:
        if isinstance(component, (RunAnalysisResult, RunDatabaseAnalysisResult)):
            if component.dataset_id:
                try:
                    await analyst_db.dataset_handler.delete_dataset(
                        component.dataset_id
                    )
                    logger.info(f"Deleted dataset {component.dataset_id}")
                except Exception as e:
                    logger.warning(
                        f"Error deleting dataset {component.dataset_id}: {e}"
                    )
