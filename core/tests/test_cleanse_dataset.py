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


# mypy: disable-error-code="no-untyped-def"

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest

from core.data_cleansing_helpers import try_string_trim
from core.schema import AnalystDataset

log = logging.getLogger(__name__)

@pytest.fixture
def sample_datasets():
    # Basic dataset with various types of columns
    df1 = pd.DataFrame(
        {
            "simple numeric": ["1", "2", "3", "4 "],
            "quoted numeric": ["'1'", "'2.5'", "'3'", "4"],
            "currency": ["$1,234", "€2,345", "£3,456", "$4,567"],
            "percentage": ["15%", "22.5%", "-5%", "100%"],
            "magnitude": ["1.5M", "500K", "2.2B", "1000"],
            "dates_mdy": ["01/15/2023", "02/28/2023", "12/31/2023", "01/01/2024"],
            "dates_dmy": ["15/01/2023", "28/02/2023", "31/12/2023", "01/01/2024"],
            "mixed_column": ["$1M", "50%", "1000", "invalid"],
            "text_column": ["abc", "def", "ghi", "jkl"],
        }
    )

    # Dataset with edge cases
    df2 = pd.DataFrame(
        {
            "empty_strings": ["", " ", None, "1"],
            "all_invalid": ["abc", "def", "ghi", "jkl"],
            "mostly_invalid": ["1", "abc", "def", "ghi"],
            "mixed_nulls": ["1", None, np.nan, "2"],
            "special_chars": ["1#", "2@", "3$", "4%"],
        }
    )

    return [
        AnalystDataset(name="basic_data", data=df1),
        AnalystDataset(name="edge_cases", data=df2),
    ]


@pytest.mark.asyncio
async def test_empty_dataset():
    from core.api import cleanse_dataframe

    empty_df = pd.DataFrame()
    empty_dataset = AnalystDataset(name="empty", data=empty_df)

    with pytest.raises(ValueError, match="Dataset empty is empty"):
        await cleanse_dataframe(empty_dataset)


@pytest.mark.asyncio
async def test_simple_numeric_conversion(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_data = results[0].dataset.to_df()

    log.info(results[0].cleaning_report)

    # Check simple numeric conversion
    assert basic_data["simple numeric"].dtype.is_numeric()
    assert basic_data["simple numeric"].to_list() == [1.0, 2.0, 3.0, 4.0]

    # Check quoted numeric conversion
    assert basic_data["quoted numeric"].dtype.is_numeric()
    assert basic_data["quoted numeric"].to_list() == [1.0, 2.5, 3.0, 4.0]


@pytest.mark.asyncio
async def test_currency_conversion(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_data = results[0].dataset.to_df()

    assert basic_data["currency"].dtype.is_numeric()
    assert basic_data["currency"].to_list() == [1234.0, 2345.0, 3456.0, 4567.0]


@pytest.mark.asyncio
async def test_percentage_conversion(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_data = results[0].dataset.to_df()

    assert basic_data["percentage"].dtype.is_numeric()
    assert basic_data["percentage"].to_list() == [0.15, 0.225, -0.05, 1.0]


@pytest.mark.asyncio
async def test_magnitude_conversion(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_data = results[0].dataset.to_df()

    assert basic_data["magnitude"].dtype.is_numeric()
    expected = [1500000.0, 500000.0, 2200000000.0, 1000.0]
    assert all(
        abs(a - b) < 0.001 for a, b in zip(basic_data["magnitude"].to_list(), expected)
    )


@pytest.mark.asyncio
async def test_date_conversion(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_data = results[0].dataset.to_df()

    assert basic_data["dates_mdy"].dtype.is_temporal()
    assert basic_data["dates_dmy"].dtype.is_temporal()

    # Verify specific dates
    assert basic_data["dates_mdy"][0] == pd.Timestamp("2023-01-15")
    assert basic_data["dates_dmy"][0] == pd.Timestamp("2023-01-15")


@pytest.mark.asyncio
async def test_edge_cases(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    edge_data = results[1].dataset.to_df()

    # Check that invalid text columns remain as object type
    assert edge_data["all_invalid"].dtype == pl.String

    # Check mixed nulls handling
    assert edge_data["mixed_nulls"].dtype.is_numeric()
    assert edge_data["mixed_nulls"][0] == 1.0
    assert edge_data["mixed_nulls"].is_null()[1]
    assert edge_data["mixed_nulls"].is_null()[2]
    assert edge_data["mixed_nulls"][3] == 2.0


@pytest.mark.asyncio
async def test_column_report_generation(sample_datasets):
    from core.api import cleanse_dataframe

    results = [
        await cleanse_dataframe(sample_dataset) for sample_dataset in sample_datasets
    ]
    basic_report = results[0].cleaning_report

    # Find currency column report
    currency_report = next(r for r in basic_report if r.new_column_name == "currency")
    assert currency_report.original_dtype == "string"
    assert currency_report.new_dtype in ["Float64", "Int64"]
    assert currency_report.conversion_type is not None
    assert currency_report.conversion_type == "unit_conversion"
    assert "currency" in " ".join(currency_report.warnings)

    # Find date column report
    date_report = next(r for r in basic_report if r.new_column_name == "dates_mdy")
    assert date_report.new_dtype is not None
    assert "Datetime" in date_report.new_dtype
    assert date_report.conversion_type == "datetime"


@pytest.mark.asyncio
async def test_column_name_cleaning():
    from core.api import cleanse_dataframe

    # Test column name cleaning with spaces and special characters
    df = pd.DataFrame(
        {
            "  Column  Name  ": [1, 2, 3],
            "Special!@#$Characters": [4, 5, 6],
            "Multiple    Spaces": [7, 8, 9],
        }
    )
    dataset = AnalystDataset(name="test", data=df)

    result = await cleanse_dataframe(dataset)
    cleaned_cols = result.dataset.to_df().columns

    assert "Column Name" in cleaned_cols
    assert "Special!@#$Characters" in cleaned_cols
    assert "Multiple Spaces" in cleaned_cols


@pytest.mark.asyncio
async def test_10k_diabetes(dataset_loaded):
    from core.api import cleanse_dataframe

    result = await cleanse_dataframe(dataset_loaded)

    assert len(result.dataset.to_df()) > 0


@pytest.mark.asyncio
async def test_string_trim_conversion():
    from core.api import cleanse_dataframe

    # Create a dataset with whitespace in string values
    df = pd.DataFrame(
        {
            "whitespace_strings": ["  abc  ", "def   ", "   ghi", "jkl"],
            "clean_strings": ["abc", "def", "ghi", "jkl"],
            "mixed_content": ["  123  ", "   abc", "def   ", "ghi"],
        }
    )
    dataset = AnalystDataset(name="whitespace_test", data=df)

    result = await cleanse_dataframe(dataset)
    cleaned_data = result.dataset.to_df()

    # Test that whitespace was trimmed from string values
    assert cleaned_data["whitespace_strings"].to_list() == ["abc", "def", "ghi", "jkl"]

    # Test that already clean strings remain unchanged
    assert cleaned_data["clean_strings"].to_list() == ["abc", "def", "ghi", "jkl"]

    # Check the cleaning report to ensure string_trim was used
    whitespace_report = next(
        r for r in result.cleaning_report if r.new_column_name == "whitespace_strings"
    )
    assert whitespace_report.original_dtype == "string"
    assert (
        whitespace_report.new_dtype == "string"
        or whitespace_report.new_dtype == "String"
    )
    assert whitespace_report.conversion_type == "string_trim"
    assert any(
        "whitespace" in warning.lower() for warning in whitespace_report.warnings
    )


def test_try_string_trim_direct():
    # Series with whitespace that needs trimming
    series1 = pl.Series(["  abc  ", "def   ", "   ghi", "jkl"])
    original_nulls1 = series1.is_null()
    success1, result1, warnings1 = try_string_trim(series1, series1, original_nulls1, 1)

    assert success1 is True
    assert result1.to_list() == ["abc", "def", "ghi", "jkl"]
    assert len(warnings1) == 1
    assert "whitespace" in warnings1[0].lower()

    # Series without whitespace that needs trimming
    series2 = pl.Series(["abc", "def", "ghi", "jkl"])
    original_nulls2 = series2.is_null()
    success2, result2, warnings2 = try_string_trim(series2, series2, original_nulls2, 1)

    assert success2 is False
    assert result2.to_list() == ["abc", "def", "ghi", "jkl"]
    assert len(warnings2) == 0

    # Series with null values
    series3 = pl.Series(["  abc  ", None, "   ghi", "jkl"])
    original_nulls3 = series3.is_null()
    success3, result3, warnings3 = try_string_trim(series3, series3, original_nulls3, 1)

    assert success3 is True
    assert result3.is_null()[1]  # Null value preserved
    assert result3[0] == "abc"
    assert result3[2] == "ghi"
    assert result3[3] == "jkl"
    assert len(warnings3) == 1
