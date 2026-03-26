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


# mypy: disable-error-code="arg-type"
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from core.code_execution import InvalidGeneratedCode, execute_python


# Define a mock Pydantic model for testing purposes
class MockOutputModel(BaseModel):
    result: float | str | int | None


@pytest.fixture
def setup() -> dict[str, Any]:
    """Fixture to set up test environment."""
    return {
        "expected_function": "expected_function",
        "input_data": 5,
        "modules": {},
        "functions": {},
    }


def test_execute_python_with_valid_code(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return {{'result': x * 2}}
"""
    allowed_modules = {"math"}

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=setup["input_data"],
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == 10


def test_execute_python_with_illegal_imports(setup: Dict[str, Any]) -> None:
    code = """
import os
def expected_function(x):
    return x
"""
    allowed_modules = {"math"}

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules=allowed_modules,
        )

    assert "Illegal imports detected: {'os'}" in str(excinfo.value)


def test_execute_python_without_expected_function(setup: Dict[str, Any]) -> None:
    code = """
def some_other_function(x):
    return x
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules={"math"},
        )

    assert f"code didn't include required function {setup['expected_function']}" in str(
        excinfo.value
    )


def test_execute_python_with_error_in_function(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    raise ValueError("Test error")
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules={"math"},
        )

    assert (
        f"Function {setup['expected_function']} raised an error during execution: Test error"
        in str(excinfo.value)
    )


def test_execute_python_output_type_mismatch(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return [x]
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules={"math"},
        )

    assert "Expected MockOutputModel, got list" in str(excinfo.value)


def test_execute_python_with_syntax_error(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x)
    return x
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules={"math"},
        )

    assert "SyntaxError" in str(excinfo.value)


def test_execute_python_without_allowed_module(setup: Dict[str, Any]) -> None:
    code = """
import os
def expected_function(x):
    return x
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules=None,
        )
    assert "Illegal imports detected: {'os'}" in str(excinfo.value)


def test_execute_python_with_string_input(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return {{'result': x}}
"""
    allowed_modules = {"math"}
    input_data = "Test string"

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == "Test string"


def test_execute_python_with_none_input(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return {{'result': x}}
"""
    allowed_modules = {"math"}
    input_data = None

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result is None


def test_execute_python_with_multiple_parameters(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x, y, z=1):
    return {{'result': x + y + z}}
"""
    allowed_modules = {"math"}
    input_data = (2, 3)
    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=input_data,
            output_type=MockOutputModel,
            allowed_modules=allowed_modules,
        )

    assert (
        "TypeError: expected_function() missing 1 required positional argument: 'y'"
        in str(excinfo.value)
    )


def test_execute_python_with_pandas_dataframe(setup: Dict[str, Any]) -> None:
    import pandas as pd  # noqa: F401

    code = """
import pandas as pd
def expected_function(x):
    df = pd.DataFrame(x)
    return {'result': len(df)}
"""
    allowed_modules = {"pandas", "math"}
    input_data = [[1, 2], [3, 4]]

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == 2


def test_execute_python_with_numpy_array(setup: Dict[str, Any]) -> None:
    import numpy as np  # noqa: F401

    code = """
import numpy as np
def expected_function(x):
    arr = np.array(x)
    return {'result': arr.mean()}
"""
    allowed_modules = {"numpy", "math"}
    input_data = [1, 2, 3, 4]

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == 2.5


def test_execute_python_error_handling_complex(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return x / 0
"""

    with pytest.raises(InvalidGeneratedCode) as excinfo:
        execute_python(
            modules=setup["modules"],
            functions=setup["functions"],
            expected_function=setup["expected_function"],
            code=code,
            input_data=setup["input_data"],
            output_type=MockOutputModel,
            allowed_modules={"math"},
        )

    assert "division by zero" in str(excinfo.value)


def test_execute_python_with_kwargs(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x, y=0, z=1):
    return {{"result": x + y + z}}
"""
    allowed_modules = {"math"}
    input_data = 2

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == 3


def test_execute_python_output_model_from_dict(setup: Dict[str, Any]) -> None:
    code = f"""
def {setup["expected_function"]}(x):
    return {{'result': x}}
"""
    allowed_modules = {"math"}
    input_data = 5

    result = execute_python(
        modules=setup["modules"],
        functions=setup["functions"],
        expected_function=setup["expected_function"],
        code=code,
        input_data=input_data,
        output_type=MockOutputModel,
        allowed_modules=allowed_modules,
    )

    assert result.result == 5
