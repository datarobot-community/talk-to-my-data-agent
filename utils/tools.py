# Copyright 2024 DataRobot, Inc.
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


# Add additional tools that can be used by the analysis code execution. Remember to include the necessary imports and provide a docstring and signature for each function.
# signature and docstring will be provided to the LLM in the prompt.
# Uncomment the examples below to get started.


# import datarobot as dr
# import pandas as pd
# from datarobot_predict.deployment import predict

# def calculate_summary(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate summary statistics for a DataFrame.

#     Args:
#         df (pd.DataFrame): Input DataFrame.

#     Returns:
#         pd.DataFrame: Summary statistics including count, mean, std, min, max, and percentiles.
#     """
#     description = df.describe(percentiles=[0.2, 0.4, 0.6, 0.8])
#     return description


# def filter_data(df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
#     """
#     Filter DataFrame based on a condition.
#     Args:
#         df (pd.DataFrame): Input DataFrame.
#         column (str): Column to apply the filter on.
#         value (float): Value to compare against in the filter.
#     Returns:
#         pd.DataFrame: Filtered DataFrame where the specified column's values are greater than the given value.
#     """
#     filtered_df = df[df[column] > value]
#     return filtered_df


# def call_datarobot_deployment(df: pd.DataFrame, deployment_id: str) -> pd.DataFrame:
#     """
#     Call a DataRobot deployment to get predictions.

#     Args:
#         df (pd.DataFrame): Input DataFrame with features for prediction.
#         deployment_id (str): ID of the DataRobot deployment to use for predictions.

#     Returns:
#         pd.DataFrame: DataFrame containing the predictions from DataRobot. The prediction column is named 'predictions'.
#     """
#     deployment = dr.Deployment.get(deployment_id)  # type: ignore[attr-defined]
#     prediction_response: pd.DataFrame = predict(
#         deployment=deployment, data_frame=df
#     ).dataframe

#     prediction_response.columns = [
#         c.replace("_PREDICTION", "")
#         for c in prediction_response.columns  # type: ignore[assignment]
#     ]

#     if deployment.model is not None:
#         target_column = deployment.model.get("target_name")
#         if target_column:
#             prediction_response["predictions"] = prediction_response[target_column]

#     return prediction_response
