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

"""File handling utilities for CSV and Excel files."""

from __future__ import annotations

import io

import chardet
import polars as pl

from core.logging_helper import get_logger

logger = get_logger()


def detect_and_decode_csv(raw_bytes: bytes, filename: str) -> str:
    """
    Detect encoding and decode CSV content with BOM handling.
    Tries multiple strategies to ensure robust file handling.

    Raises ValueError with descriptive message on failure.
    """
    # Strip UTF-8 BOM if present
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        logger.info(f"Stripped UTF-8 BOM from '{filename}'")
        raw_bytes = raw_bytes[3:]

    if not raw_bytes:
        raise ValueError(f"File '{filename}' is empty or could not be read.")

    # Detect encoding
    detection_result = chardet.detect(raw_bytes)
    detected_encoding = detection_result.get("encoding")
    confidence = detection_result.get("confidence") or 0

    if detected_encoding:
        logger.info(
            f"Detected encoding for '{filename}': {detected_encoding} "
            f"(confidence: {confidence:.2f})"
        )

    # Try detected encoding first if confidence is reasonable
    if detected_encoding and confidence >= 0.7:
        try:
            return raw_bytes.decode(detected_encoding)
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(
                f"Failed to decode '{filename}' with detected encoding "
                f"{detected_encoding}: {e}"
            )

    # Try UTF-8 as fallback
    try:
        decoded = raw_bytes.decode("utf-8")
        if detected_encoding != "utf-8":
            logger.info(f"Successfully decoded '{filename}' using fallback: utf-8")
        return decoded
    except UnicodeDecodeError:
        pass

    raise ValueError(
        f"Unable to decode file '{filename}'. All encoding attempts failed."
    )


def detect_delimiter(content: str) -> str:
    """Detect CSV delimiter by checking first few lines. Defaults to comma on any ambiguity."""
    lines = content.split("\n")[:5]

    delimiters = {",": 0, ";": 0, "\t": 0, "|": 0}
    for line in lines:
        if not line.strip():
            continue
        for d in delimiters:
            delimiters[d] += line.count(d)

    best = max(delimiters, key=lambda k: delimiters[k])
    return best if delimiters[best] > 0 else ","


def load_and_validate_csv(decoded_content: str, filename: str) -> pl.DataFrame:
    """
    Parse CSV content and validate the resulting DataFrame.

    Raises ValueError with descriptive message on failure.
    """
    # Normalize line endings (old Mac CR → LF, Windows CRLF → LF)
    if "\r" in decoded_content:
        decoded_content = decoded_content.replace("\r\n", "\n").replace("\r", "\n")

    # Auto-detect delimiter
    separator = detect_delimiter(decoded_content)
    if separator != ",":
        logger.info(f"Detected delimiter '{separator}' for '{filename}'")

    try:
        df = pl.read_csv(
            io.StringIO(decoded_content),
            infer_schema_length=10000,
            low_memory=True,
            separator=separator,
        )
    except pl.exceptions.ComputeError as e:
        raise ValueError(f"Unable to parse CSV file '{filename}'. Error: {str(e)}")

    if df.height == 0:
        raise ValueError(f"CSV file '{filename}' contains only headers, no data rows.")

    # Check if ragged lines were truncated
    lines = [line for line in decoded_content.split("\n") if line.strip()]
    if len(lines) > 1:
        expected_rows = len(lines) - 1  # subtract header
        if df.height < expected_rows * 0.9:  # More than 10% loss
            logger.warning(
                f"CSV file '{filename}' had inconsistent row lengths. "
                f"Expected ~{expected_rows} rows, got {df.height}."
            )

    return df
