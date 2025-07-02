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
import ast
import functools
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from types import FunctionType, ModuleType
from typing import Any, Awaitable, Callable, ParamSpec, Type, TypeVar, cast

import pandas as pd
import polars as pl
from pydantic import BaseModel

from utils.logging_helper import get_logger

logger = get_logger("PythonExecutor")

U = TypeVar("U")
V = TypeVar("V", bound=BaseModel)


class InvalidGeneratedCode(Exception):
    """Raised when LLM generated code is found to be invalid."""

    def __init__(
        self,
        *args: Any,
        code: str | None = None,
        exception: Exception | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        traceback_str: str | None = None,
    ):
        super().__init__(*args)
        self.code = code
        self.exception = exception
        self.stdout = stdout
        self.stderr = stderr
        self.traceback_str = traceback_str

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.traceback_str:
            parts.append(f"\nTraceback:\n{self.traceback_str}")
        if self.stdout and self.stdout.strip():
            parts.append(f"\nStdout:\n{self.stdout}")
        if self.stderr and self.stderr.strip():
            parts.append(f"\nStderr:\n{self.stderr}")
        return "\n".join(parts)


class MaxReflectionAttempts(Exception):
    """Raised after final attempt to self-correct LLM code generation"""

    def __init__(
        self,
        *args: Any,
        exception_history: list[InvalidGeneratedCode] | None = None,
        duration: float | None = None,
    ):
        super().__init__(*args)
        self.exception_history = exception_history
        self.duration = duration


P = ParamSpec("P")  # For capturing all parameter types
R = TypeVar("R")  # For capturing the return type


def reflect_code_generation_errors(
    max_attempts: int,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Reflect LLM code generation errors for self-correction

    Exceptions raised by invalid code will be injected back into the
    decorated function via the `exception_history` keyword argument.

    `exception_history` contains a list of InvalidGeneratedCode
    exceptions.
    """

    def _outer_wrapper(
        f: Callable[P, Awaitable[R]],
    ) -> Callable[P, Awaitable[R]]:
        @functools.wraps(f)
        async def _inner_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            now = datetime.now()

            attempts = 1
            exception_history: list[InvalidGeneratedCode] = []
            kwargs["exception_history"] = exception_history
            while attempts <= max_attempts:
                try:
                    return await f(*args, **kwargs)
                except InvalidGeneratedCode as e:
                    msg = type(e.exception).__name__ + f": {str(e.exception)}"
                    logger.info(
                        f"LLM generated code raised {msg}\nGenerated code:\n{e.code}"
                    )
                    exception_history.append(e)
                attempts += 1

            msg = f"{f.__name__} failed to generate valid code after {max_attempts} attempts"
            logger.error(msg)
            time = datetime.now() - now
            raise MaxReflectionAttempts(
                msg, exception_history=exception_history, duration=time.seconds
            )

        return _inner_wrapper

    return _outer_wrapper


def validate_python_code(
    code: str,
    expected_function: str,
    allowed_modules: set[str],
) -> None:
    """
    Validate Python code for safety and correctness.

    Args:
        code: The Python code to validate
        expected_function: Name of the function that should be defined
        allowed_modules: Set of module names that are allowed to be imported

    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    tree = ast.parse(code)
    imports: list[str] = []

    # Check imports
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                imports.extend(n.name.split(".")[0] for n in node.names)
            elif node.module is not None:
                imports.append(node.module.split(".")[0])

    illegal_imports = set(imports) - allowed_modules
    if illegal_imports:
        raise ImportError(f"Illegal imports detected: {illegal_imports}")

    # Verify expected function exists
    has_function = any(
        isinstance(node, ast.FunctionDef) and node.name == expected_function
        for node in ast.walk(tree)
    )
    if not has_function:
        raise InvalidGeneratedCode(
            f"code didn't include required function {expected_function}",
            code=code,
        )


def execute_python(
    modules: dict[str, ModuleType],
    functions: dict[str, Callable[..., Any]],
    expected_function: str,
    code: str,
    input_data: pd.DataFrame
    | dict[str, pd.DataFrame]
    | pl.DataFrame
    | dict[str, pl.DataFrame],
    output_type: Type[V],
    allowed_modules: set[str] | None = None,
) -> V:
    """
    Executes Python code in a given namespace and checks if the expected function is defined.
    Raises InvalidGeneratedCode if the code is invalid or execution fails.
    """
    if allowed_modules is None:
        allowed_modules = set(modules.keys())

    allowed_modules = allowed_modules.union(
        {
            "bisect",
            "collections",
            "csv",
            "datetime",
            "dask",
            "difflilb",
            "fnmatch",
            "glob",
            "json",
            "math",
            "matplotlib",
            "numpy",
            "openpyxl",
            "pandas",
            "plotly",
            "polars",
            "pyspark",
            "random",
            "re",
            "scipy",
            "seaborn",
            "statistics",
            "string",
            "time",
            "xlsxwriter",
            "yaml",
        }
    )

    namespace = {**modules, **functions}

    try:
        validate_python_code(code, expected_function, allowed_modules)

        stdout = io.StringIO()
        stderr = io.StringIO()

        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exec(code, namespace)

                if not isinstance(namespace[expected_function], FunctionType):
                    raise InvalidGeneratedCode(
                        f"{expected_function} is not a valid function in the provided code.",
                        code=code,
                        stdout=stdout.getvalue(),
                        stderr=stderr.getvalue(),
                    )

                func = cast(Callable[[Any], Any], namespace[expected_function])
                try:
                    result = func(input_data)
                except Exception as e:
                    raise InvalidGeneratedCode(
                        f"Function {expected_function} raised an error during execution: {str(e)}",
                        code=code,
                        exception=e,
                        stdout=stdout.getvalue(),
                        stderr=stderr.getvalue(),
                        traceback_str=traceback.format_exc(),
                    )

                if not isinstance(result, dict) and not isinstance(result, output_type):
                    raise InvalidGeneratedCode(
                        f"Expected {output_type.__name__}, got {type(result).__name__}",
                        code=code,
                        stdout=stdout.getvalue(),
                        stderr=stderr.getvalue(),
                    )

                if isinstance(result, dict):
                    try:
                        return output_type(**result)
                    except Exception as e:
                        raise InvalidGeneratedCode(
                            "Failed to convert dictionary to Pydantic model",
                            code=code,
                            exception=e,
                        )
                return result

        except (SyntaxError, ValueError) as e:
            raise InvalidGeneratedCode(
                str(e),
                code=code,
                exception=e,
                stdout=stdout.getvalue(),
                stderr=stderr.getvalue(),
                traceback_str=traceback.format_exc(),
            )
    except Exception as e:
        if isinstance(e, InvalidGeneratedCode):
            raise
        raise InvalidGeneratedCode(
            f"Unexpected error during code execution: {str(e)}",
            code=code,
            exception=e,
            traceback_str=traceback.format_exc(),
        )
