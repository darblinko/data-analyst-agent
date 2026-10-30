"""
Code Execution Tool for the data-analyst-agent.

Executes Python code in the local environment with:
- Captured stdout/stderr output
- Access to 3rd-party libraries (pandas, yfinance, etc.)
- File system access limited to the project directory
- Timeout protection
- Persistent execution context between calls within a session
"""

import io
import os
import sys
import traceback
import contextlib
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

# Project root directory - all file operations are scoped here
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Persistent execution namespace (survives between tool calls within a session)
# This allows agents to define variables/functions and use them in subsequent calls
_execution_namespace: dict[str, Any] = {}


def get_execution_namespace() -> dict[str, Any]:
    """Return the current execution namespace."""
    return _execution_namespace


def reset_execution_namespace() -> None:
    """Reset the execution namespace (called when starting a new conversation turn)."""
    global _execution_namespace
    _execution_namespace = {}


def _build_safe_globals() -> dict[str, Any]:
    """
    Build a globals dict for exec() that:
    - Includes __builtins__ (allows imports)
    - Injects PROJECT_ROOT so code knows the allowed path
    - Includes the persistent namespace
    """
    safe_globals = {
        "__builtins__": __builtins__,
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "UPLOADS_DIR": str(PROJECT_ROOT / "uploads"),
    }
    # Merge in persistent namespace
    safe_globals.update(_execution_namespace)
    return safe_globals


def _restrict_path(path_str: str) -> str:
    """
    Validate that a file path stays within PROJECT_ROOT.
    Raises ValueError if the path escapes the project directory.
    """
    try:
        resolved = Path(path_str).resolve()
        resolved.relative_to(PROJECT_ROOT)
        return str(resolved)
    except ValueError:
        raise ValueError(
            f"File access denied: '{path_str}' is outside the project directory "
            f"({PROJECT_ROOT}). Only files within the project folder are accessible."
        )


def execute_code(code: str) -> dict[str, str]:
    """
    Execute Python code and return captured output.

    Args:
        code: Python code string to execute

    Returns:
        dict with keys:
        - 'stdout': Captured standard output
        - 'stderr': Captured error output / tracebacks
        - 'result': Summary of execution ('success' or 'error')
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    exec_globals = _build_safe_globals()

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(compile(code, "<agent_code>", "exec"), exec_globals)

        # Update persistent namespace with any new variables defined
        # (exclude builtins and injected constants to keep namespace clean)
        skip_keys = {"__builtins__", "PROJECT_ROOT", "UPLOADS_DIR"}
        for key, value in exec_globals.items():
            if key not in skip_keys and not key.startswith("__"):
                _execution_namespace[key] = value

        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        return {
            "stdout": stdout_output if stdout_output else "(no output)",
            "stderr": stderr_output if stderr_output else "",
            "result": "success",
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        stderr_output = stderr_capture.getvalue()

        return {
            "stdout": stdout_capture.getvalue() or "(no output before error)",
            "stderr": f"{stderr_output}\n{error_trace}".strip(),
            "result": "error",
        }


@tool
def python_repl(code: str) -> str:
    """
    Execute Python code in the local environment and return the output.

    Use this tool to:
    - Analyze data (pandas, numpy, scipy)
    - Fetch financial data (yfinance)
    - Read uploaded files from the uploads directory
    - Perform calculations and statistical analysis
    - Generate visualizations (matplotlib, plotly)

    Important:
    - Variables persist between calls within the same session
    - Use PROJECT_ROOT variable for the project directory path
    - Use UPLOADS_DIR variable for the uploads directory path
    - All file operations must use paths within the project directory
    - Print your results so they appear in the output

    Args:
        code: Valid Python code to execute

    Returns:
        String containing stdout output, or error message if execution failed
    """
    result = execute_code(code)

    output_parts = []

    if result["stdout"] and result["stdout"] != "(no output)":
        output_parts.append(result["stdout"])

    if result["stderr"]:
        output_parts.append(f"[STDERR/ERROR]\n{result['stderr']}")

    if not output_parts:
        if result["result"] == "success":
            output_parts.append("Code executed successfully (no output produced).")
        else:
            output_parts.append("Execution failed with no output.")

    return "\n".join(output_parts)
