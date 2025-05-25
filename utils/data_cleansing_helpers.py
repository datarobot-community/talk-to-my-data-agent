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
from __future__ import annotations

import re
import warnings
from contextlib import contextmanager
from typing import Generator, cast

import pandas as pd
from utils.logging_helper import get_logger
from utils.schema import CleansedColumnReport

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
    series: pd.Series,
    sample_series: pd.Series,
    original_nulls: pd.Series,
) -> tuple[bool, pd.Series, list[str]]:
    # 文字列の前後の空白を削除し、特殊文字（引用符や空白）を削除
    simple_cleaned = sample_series.astype(str).str.strip().str.replace(r"['\s]+", "", regex=True)

    # 数値に変換
    numeric_simple = pd.to_numeric(simple_cleaned, errors='coerce')

    # 新しいnull値の数をカウント
    new_nulls = numeric_simple.isna()
    original_nulls_count = original_nulls.sum()

    # 成功率の計算
    simple_success_rate = 1 - (new_nulls.sum() - original_nulls_count) / len(sample_series)

    warnings = []

    if simple_success_rate > 0.8:
        warnings.append(
            f"Converted to numeric after removing spaces/quotes. Success rate: {simple_success_rate:.1%}"
        )
        # 実際のシリーズにも同じ処理を適用
        clean_series = series.astype(str).str.strip().str.replace(r"['\s]+", "", regex=True)
        result_series = pd.to_numeric(clean_series, errors='coerce')
        return True, result_series, warnings
    else:
        # 変換が部分的に成功した場合は警告
        if simple_success_rate > 0.2:
            warnings.append(
                f"Simple numeric conversion partially successful ({simple_success_rate:.1%}) but below threshold"
            )
    return False, series, warnings


def _convert_units(
    series: pd.Series, patterns: dict[str, bool] | None = None
) -> tuple[pd.Series, dict[str, bool]]:
    """
    Convert a series containing various unit formats to numeric values.
    Returns the converted series and pattern detection info.

    Args:
        series: Input series to convert
        patterns: Optional pre-detected patterns. If None, patterns will be detected.
    """
    if patterns is None:
        # パターンを検出
        patterns = {
            "has_currency": cast(float, series.astype(str).str.contains(r"[$€£¥]", regex=True).mean()) > 0.7,
            "has_commas": cast(float, series.astype(str).str.contains(r",", regex=True).mean()) > 0.7,
            "has_percent": cast(float, series.astype(str).str.contains(r"%", regex=True).mean()) > 0.7,
            "has_magnitude": cast(float, series.astype(str).str.contains(r"(?i)[KMB]$", regex=True).mean()) > 0.7,
        }

    # コピーを作成
    result = series.copy()

    # 検出されたパターンに基づいて変換
    if patterns["has_currency"]:
        result = result.astype(str).str.replace(r"[$€£¥]", "", regex=True)

    if patterns["has_commas"]:
        result = result.astype(str).str.replace(",", "", regex=True)

    if patterns["has_magnitude"]:
        k_mask = result.astype(str).str.contains(r"(?i)K$", regex=True)
        m_mask = result.astype(str).str.contains(r"(?i)M$", regex=True)
        b_mask = result.astype(str).str.contains(r"(?i)B$", regex=True)

        # K/M/B を削除
        result = result.astype(str).str.replace(r"(?i)[KMB]$", "", regex=True)
        numeric_result = pd.to_numeric(result, errors='coerce')

        # 有効な数値にのみ乗数を適用
        valid_mask = ~numeric_result.isna()
        
        # 各乗数に基づいて調整
        if valid_mask.any() and k_mask.any():
            numeric_result.loc[valid_mask & k_mask] *= 1000
        if valid_mask.any() and m_mask.any():
            numeric_result.loc[valid_mask & m_mask] *= 1000000
        if valid_mask.any() and b_mask.any():
            numeric_result.loc[valid_mask & b_mask] *= 1000000000
            
        result = numeric_result

    if patterns["has_percent"]:
        result = result.astype(str).str.replace("%", "", regex=True)
        result = pd.to_numeric(result, errors='coerce') / 100
    else:
        result = pd.to_numeric(result, errors='coerce')

    return result, patterns


def try_unit_conversion(
    series: pd.Series,
    sample_series: pd.Series,
    original_nulls: pd.Series,
) -> tuple[bool, pd.Series, list[str]]:
    """
    Try to convert a series with units to numeric values, first testing on a sample.
    Returns success status, converted series, and warnings.
    """
    warnings: list[str] = []

    # 数値データを含む可能性があるかチェック
    if not cast(
        float, sample_series.astype(str).str.contains(r"[$€£¥%KMB,\d.]", regex=True).mean()
    ) > 0.5:
        return False, series, warnings

    # サンプルでまず変換を試みてパターンを検出
    sample_result, patterns = _convert_units(sample_series)

    # 検出されたパターンに基づいて警告メッセージを生成
    pattern_names = {
        "has_currency": "currency symbols",
        "has_commas": "thousand separators",
        "has_magnitude": "magnitude suffixes (K/M/B)",
        "has_percent": "percentages",
    }

    detected = [pattern_names[k] for k, v in patterns.items() if v]
    if detected:
        warnings.append(f"Detected patterns in data: {', '.join(detected)}")

    # 変換成功率の計算
    new_nulls = sample_result.isna()
    conversion_success_rate = 1 - (new_nulls.sum() - original_nulls.sum()) / len(sample_result)

    if conversion_success_rate > 0.8:
        # サンプル変換が成功した場合、検出されたパターンを使用して完全なデータセットを変換
        warnings.append(
            f"Converted to numeric with pattern handling. Success rate: {conversion_success_rate:.1%}"
        )
        result, _ = _convert_units(series, patterns)  # サンプルから得たパターンを再利用
        return True, result, warnings

    elif conversion_success_rate > 0.2:
        warnings.append(
            f"Complex numeric conversion partially successful ({conversion_success_rate:.1%}) but below threshold"
        )

    return False, series, warnings


def try_datetime_conversion(
    series: pd.Series,
    sample_series: pd.Series,
    original_nulls: pd.Series,
) -> tuple[bool, pd.Series, list[str]]:
    # 日付への変換を試みる
    warnings = []

    with suppress_datetime_warnings():
        from pandas.core.tools.datetimes import (  # type: ignore[attr-defined]
            _guess_datetime_format_for_array,
        )

        format_1 = _guess_datetime_format_for_array(sample_series.values)
        format_2 = _guess_datetime_format_for_array(
            sample_series.values, dayfirst=True
        )
        
    # 2つの形式で日付時刻への変換を試す
    with suppress_datetime_warnings():
        candidate_1 = pd.to_datetime(sample_series, format=format_1, errors='coerce')
        candidate_2 = pd.to_datetime(sample_series, format=format_2, errors='coerce', dayfirst=True)

    candidate_1_success_rate = cast(float, (~candidate_1.isna()).mean())
    candidate_2_success_rate = cast(float, (~candidate_2.isna()).mean())

    if candidate_1_success_rate > 0.8 or candidate_2_success_rate > 0.8:
        success_rate = max(
            candidate_1_success_rate,
            candidate_2_success_rate,
        )
        warnings.append(f"Converted to datetime. Success rate: {success_rate:.1%}")
        
        if candidate_1_success_rate >= candidate_2_success_rate:
            warnings.append("Used month-first date parsing")
            with suppress_datetime_warnings():
                ts_series = pd.to_datetime(series, format=format_1, errors='coerce')
        else:
            warnings.append("Used day-first date parsing")
            with suppress_datetime_warnings():
                ts_series = pd.to_datetime(series, format=format_2, errors='coerce', dayfirst=True)
        
        return True, ts_series, warnings
    
    return False, series, warnings


def add_summary_statistics(
    df: pd.DataFrame, report: list[CleansedColumnReport]
) -> None:
    """Add summary statistics to the report."""
    for column_report in report:
        col = column_report.new_column_name
        if column_report.new_dtype:
            null_count = df[col].isna().sum()
            total_count = len(df)
            if null_count > 0:
                column_report.warnings.append(
                    f"Contains {null_count} null values ({null_count / total_count:.1%} of data)"
                )

            if column_report.new_dtype == "float64":
                unique_count = df[col].nunique()
                if unique_count == 1:
                    column_report.warnings.append("Contains only one unique value")
                elif unique_count == 2:
                    column_report.warnings.append(
                        "Contains only two unique values - consider boolean conversion"
                    )


def process_column(
    df: pd.DataFrame,
    column_name: str,
    sample_df: pd.DataFrame,
) -> tuple[str, pd.Series, CleansedColumnReport]:
    """Process a single column asynchronously."""
    cleaned_column_name = re.sub(r"\s+", " ", str(column_name).strip())
    original_nulls = sample_df[column_name].isna()
    column_report = CleansedColumnReport(new_column_name=cleaned_column_name)

    if cleaned_column_name != column_name:
        column_report.original_column_name = column_name
        column_report.warnings.append(
            f"Column renamed from '{column_name}' to '{cleaned_column_name}'"
        )

    if pd.api.types.is_string_dtype(df[column_name]):
        column_report.original_dtype = "string"
        conversions = [
            ("simple_clean", try_simple_numeric_conversion),
            ("unit_conversion", try_unit_conversion),
            ("datetime", try_datetime_conversion),
        ]

        # それぞれの変換方法を試す
        for conversion_type, conversion_func in conversions:
            try:
                logger.debug(
                    f"Attempting {conversion_type} conversion for '{column_name}'"
                )
                success, cleaned_series, warnings = conversion_func(
                    df[column_name],
                    sample_df[column_name],
                    original_nulls,
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
