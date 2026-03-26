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


"""
Test for BusinessAnalysisGeneration JSON sanitization.

This test reproduces and verifies the fix for a production bug where LLMs
would return JSON with non-breaking spaces and other control characters,
causing JSON parsing to fail.

The issue was observed in production where data from databases (Snowflake, etc.)
contained Unicode whitespace that propagated through to LLM responses.
"""

import pytest
from pydantic import ValidationError

from core.schema import BusinessAnalysisGeneration


class TestBusinessAnalysisGenerationSanitization:
    """Test JSON sanitization in BusinessAnalysisGeneration model."""

    def test_parses_json_with_non_breaking_spaces(self) -> None:
        """Test that JSON with non-breaking spaces (U+00A0) is handled correctly.

        This reproduces the exact issue from production logs where the LLM
        returned JSON with \xa0 (non-breaking space) before the keys.
        """
        # This is the pattern from production logs
        json_with_nbsp = """{
 \xa0"bottom_line": "The analysis shows trends",
 \xa0"additional_insights": "Further insights here",
 \xa0"follow_up_questions": ["What drives the trends?"]
}"""

        result = BusinessAnalysisGeneration.model_validate_json(json_with_nbsp)

        assert result.bottom_line == "The analysis shows trends"
        assert result.additional_insights == "Further insights here"
        assert result.follow_up_questions == ["What drives the trends?"]

    def test_parses_json_with_various_control_characters(self) -> None:
        """Test handling of various Unicode control characters."""
        # Mix of problematic characters
        json_with_controls = """{
\x01"bottom_line": "Test\x00line",
\x1f"additional_insights": "More\x7finfo",
\x80"follow_up_questions": ["Q1"]
}"""

        result = BusinessAnalysisGeneration.model_validate_json(json_with_controls)

        # Control chars should be replaced with spaces
        assert result.bottom_line == "Test line"
        assert result.additional_insights == "More info"

    def test_preserves_valid_whitespace_in_strings(self) -> None:
        """Test that valid JSON whitespace (tab, newline, CR) is preserved."""
        json_with_valid_whitespace = """{
    "bottom_line": "Line 1\\nLine 2\\tTabbed",
    "additional_insights": "More\\r\\ninsights",
    "follow_up_questions": ["Question 1", "Question 2"]
}"""

        result = BusinessAnalysisGeneration.model_validate_json(
            json_with_valid_whitespace
        )

        # Valid escape sequences should work
        assert "Line 1\nLine 2\tTabbed" == result.bottom_line
        assert "More\r\ninsights" == result.additional_insights

    def test_normal_json_still_works(self) -> None:
        """Test that normal, clean JSON still parses correctly."""
        clean_json = """{
    "bottom_line": "Normal text here",
    "additional_insights": "Additional information",
    "follow_up_questions": ["Question 1", "Question 2", "Question 3"]
}"""

        result = BusinessAnalysisGeneration.model_validate_json(clean_json)

        assert result.bottom_line == "Normal text here"
        assert result.additional_insights == "Additional information"
        assert len(result.follow_up_questions) == 3

    def test_handles_bytes_input(self) -> None:
        """Test that bytes input is properly decoded."""
        json_bytes = b'{"bottom_line": "test", "additional_insights": "info", "follow_up_questions": []}'

        result = BusinessAnalysisGeneration.model_validate_json(json_bytes)

        assert result.bottom_line == "test"

    def test_invalid_json_still_fails_appropriately(self) -> None:
        """Test that actually invalid JSON (not just control chars) still raises errors."""
        invalid_json = '{"bottom_line": "test", "missing_fields": true}'

        with pytest.raises(ValidationError):
            BusinessAnalysisGeneration.model_validate_json(invalid_json)
