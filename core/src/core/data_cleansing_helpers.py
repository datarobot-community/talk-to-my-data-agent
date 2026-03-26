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

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from typing import Generator, cast

import polars as pl

from core.logging_helper import get_logger
from core.schema import CleansedColumnReport

logger = get_logger("DataCleansingHelper")


@contextmanager
def suppress_datetime_warnings() -> Generator[None, None]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Parsing dates in %d/%m/%Y format when dayfirst=False \(the default\) was specified.",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %m/%d/%Y format when dayfirst=True was specified.",
        )
        yield


def try_simple_numeric_conversion(
    series: pl.Series,
    sample_series: pl.Series,
    original_nulls: pl.Series,
    column_index: int,
) -> tuple[bool, pl.Series, list[str]]:
    simple_cleaned = sample_series.str.strip_chars().str.replace_all(
        r"['\s]+",
        "",
    )
    numeric_simple = simple_cleaned.cast(pl.Float64, strict=False)
    new_nulls = numeric_simple.is_null()
    simple_success_rate = 1 - (new_nulls.sum() - original_nulls.sum()) / len(
        sample_series
    )

    warnings = []

    # First column might be ids with alphanumerics
    threshold = 1.0 if column_index == 0 else 0.9

    if simple_success_rate >= threshold:
        warnings.append(
            f"Converted to numeric after removing spaces/quotes. Success rate: {simple_success_rate:.1%}"
        )
        series = series.str.strip_chars().str.replace_all(r"['\s]+", "")
        series = series.cast(pl.Float64)
        return True, series, warnings
    else:
        # If simple cleaning failed but had some success, add warning
        if simple_success_rate > 0.2:
            warnings.append(
                f"Simple numeric conversion partially successful ({simple_success_rate:.1%}) but below threshold"
            )
    return False, series, warnings


def _convert_units(
    series: pl.Series, patterns: dict[str, bool] | None = None
) -> tuple[pl.Series, dict[str, bool]]:
    """
    Convert a series containing various unit formats to numeric values.
    Returns the converted series and pattern detection info.

    Args:
        series: Input series to convert
        patterns: Optional pre-detected patterns. If None, patterns will be detected.
    """
    if patterns is None:
        # Only detect patterns if not provided
        patterns = {
            "has_currency": cast(float, series.str.contains(r"[$€£¥]").mean()) > 0.7,
            "has_commas": cast(float, series.str.contains(r",").mean()) > 0.7,
            "has_percent": cast(float, series.str.contains(r"%").mean()) > 0.7,
            "has_magnitude": cast(float, series.str.contains(r"(?i)[KMB]$").mean())
            > 0.7,
        }

    # Create a working copy
    result = series.clone()

    # Apply transformations based on detected patterns
    if patterns["has_currency"]:
        result = result.str.replace_all(r"[$€£¥]", "")

    if patterns["has_commas"]:
        result = result.str.replace_all(",", "")

    if patterns["has_magnitude"]:
        k_mask = result.str.contains(r"(?i)K$")
        m_mask = result.str.contains(r"(?i)M$")
        b_mask = result.str.contains(r"(?i)B$")

        result = result.str.replace(r"(?i)[KMB]$", "")
        numeric_result = result.cast(pl.Float64, strict=False)

        # Only apply magnitude multipliers to valid numbers
        valid_mask = numeric_result.is_not_null()
        temp_df = pl.DataFrame([numeric_result]).with_columns(
            result=pl.when(valid_mask & k_mask)
            .then(numeric_result * 1000)
            .when(valid_mask & m_mask)
            .then(numeric_result * 1000000)
            .when(valid_mask & b_mask)
            .then(numeric_result * 1000000000)
            .otherwise(numeric_result)
        )
        result = temp_df.get_column("result")

    if patterns["has_percent"]:
        result = result.str.replace("%", "")
        result = result.cast(pl.Float64, strict=False) / 100
    else:
        result = result.cast(pl.Float64, strict=False)

    return result, patterns


def try_unit_conversion(
    series: pl.Series,
    sample_series: pl.Series,
    original_nulls: pl.Series,
    column_index: int,
) -> tuple[bool, pl.Series, list[str]]:
    """
    Try to convert a series with units to numeric values, first testing on a sample.
    Returns success status, converted series, and warnings.
    """
    warnings: list[str] = []

    # Check if series potentially contains convertible numeric data
    if (
        not cast(
            float, sample_series.cast(pl.String).str.contains(r"[$€£¥%KMB,\d.]").mean()
        )
        > 0.5
    ):
        return False, series, warnings

    # Try conversion on sample first and detect patterns
    sample_result, patterns = _convert_units(sample_series)

    # Generate warning messages for detected patterns
    pattern_names = {
        "has_currency": "currency symbols",
        "has_commas": "thousand separators",
        "has_magnitude": "magnitude suffixes (K/M/B)",
        "has_percent": "percentages",
    }

    detected = [pattern_names[k] for k, v in patterns.items() if v]
    if detected:
        warnings.append(f"Detected patterns in data: {', '.join(detected)}")

    # Calculate conversion success rate
    new_nulls = sample_result.is_null()
    conversion_success_rate = 1 - (new_nulls.sum() - original_nulls.sum()) / len(
        sample_result
    )

    # First column might be ids with alphanumerics
    threshold = 1.0 if column_index == 0 else 0.9

    if conversion_success_rate >= threshold:
        # If sample conversion was successful, convert full dataset using detected patterns
        warnings.append(
            f"Converted to numeric with pattern handling. Success rate: {conversion_success_rate:.1%}"
        )
        result, _ = _convert_units(series, patterns)  # Reuse patterns from sample
        return True, result, warnings

    elif conversion_success_rate > 0.2:
        warnings.append(
            f"Complex numeric conversion partially successful ({conversion_success_rate:.1%}) but below threshold"
        )

    return False, series, warnings


def try_datetime_conversion(
    series: pl.Series,
    sample_series: pl.Series,
    original_nulls: pl.Series,
    column_index: int,
) -> tuple[bool, pl.Series, list[str]]:
    # try to convert to date

    warnings = []
    sample_series_pd = sample_series.to_pandas()

    with suppress_datetime_warnings():
        from pandas.core.tools.datetimes import (  # type: ignore[attr-defined]
            _guess_datetime_format_for_array,
        )

        format_1 = _guess_datetime_format_for_array(sample_series_pd.values)
        format_2 = _guess_datetime_format_for_array(
            sample_series_pd.values, dayfirst=True
        )
    candidate_1 = sample_series.str.to_datetime(format=format_1)
    candidate_2 = sample_series.str.to_datetime(format=format_2)

    candidate_1_success_rate = cast(float, candidate_1.is_not_null().mean())
    candidate_2_success_rate = cast(float, candidate_2.is_not_null().mean())

    if candidate_1_success_rate > 0.8 or candidate_2_success_rate > 0.8:
        success_rate = max(
            candidate_1_success_rate,
            candidate_2_success_rate,
        )
        warnings.append(f"Converted to datetime. Success rate: {success_rate:.1%}")
        if candidate_1_success_rate > candidate_2_success_rate:
            warnings.append("Used month-first date parsing")
            with suppress_datetime_warnings():
                ts_series = series.str.to_datetime(format=format_1, strict=False)
        else:
            warnings.append("Used day-first date parsing")
            with suppress_datetime_warnings():
                ts_series = series.str.to_datetime(format=format_2, strict=False)
        return True, ts_series, warnings
    return False, series, warnings


def try_string_trim(
    series: pl.Series,
    sample_series: pl.Series,
    original_nulls: pl.Series,
    column_index: int,
) -> tuple[bool, pl.Series, list[str]]:
    """
    Trim leading and trailing whitespace from string values.
    Only applies if whitespace trimming actually changes the data.
    """
    warnings = []

    sample_cleaned = sample_series.str.strip_chars()
    original_values = sample_series.cast(pl.String)
    has_whitespace = (original_values != sample_cleaned).any()

    if has_whitespace:
        cleaned_series = series.str.strip_chars()
        warnings.append("Removed leading and trailing whitespace from string values")
        return True, cleaned_series, warnings

    return False, series, warnings


def add_summary_statistics(
    df: pl.DataFrame, report: list[CleansedColumnReport]
) -> None:
    """Add summary statistics to the report."""
    for column_report in report:
        col = column_report.new_column_name
        if column_report.new_dtype:
            null_count = df[col].is_null().sum()
            total_count = len(df)
            if null_count > 0:
                column_report.warnings.append(
                    f"Contains {null_count} null values ({null_count / total_count:.1%} of data)"
                )

            if column_report.new_dtype == "float64":
                unique_count = df[col].n_unique()
                if unique_count == 1:
                    column_report.warnings.append("Contains only one unique value")
                elif unique_count == 2:
                    column_report.warnings.append(
                        "Contains only two unique values - consider boolean conversion"
                    )


def process_column(
    df: pl.DataFrame,
    column_name: str,
    sample_df: pl.DataFrame,
) -> tuple[str, pl.Series, CleansedColumnReport]:
    """Process a single column asynchronously."""
    cleaned_column_name = re.sub(r"\s+", " ", str(column_name).strip())
    original_nulls = sample_df[column_name].is_null()
    column_report = CleansedColumnReport(new_column_name=cleaned_column_name)

    if cleaned_column_name != column_name:
        column_report.original_column_name = column_name
        column_report.warnings.append(
            f"Column renamed from '{column_name}' to '{cleaned_column_name}'"
        )

    if df[column_name].dtype == pl.String:
        column_report.original_dtype = "string"

        column_index = list(df.columns).index(column_name)

        conversions = [
            ("simple_clean", try_simple_numeric_conversion),
            ("unit_conversion", try_unit_conversion),
            ("datetime", try_datetime_conversion),
            ("string_trim", try_string_trim),
        ]

        # Run conversion attempts in parallel using ThreadPoolExecutor
        for conversion_type, conversion_func in conversions:
            try:
                logger.debug(
                    f"Attempting {conversion_type} conversion for '{column_name}'"
                )
                success, cleaned_series, warnings = conversion_func(
                    df[column_name],
                    sample_df[column_name],
                    original_nulls,
                    column_index,
                )

                logger.debug(
                    f"{conversion_type} conversion for '{column_name}' {'succeeded' if success else 'failed'}"
                )
                column_report.warnings.extend(warnings)
                if success:
                    column_report.new_dtype = str(cleaned_series.dtype)
                    column_report.conversion_type = conversion_type
                    return cleaned_column_name, cleaned_series, column_report
            except Exception as e:
                column_report.errors.append(str(e))

    return cleaned_column_name, df[column_name], column_report
