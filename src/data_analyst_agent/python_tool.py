"""The run_python tool that the agent calls."""

from langchain_core.tools import BaseTool, tool

from data_analyst_agent.executor import ExecutionLimits, format_for_model, run_python
from data_analyst_agent.workspace import Workspace

TOOL_NAME = "run_python"


def make_python_tool(workspace: Workspace, limits: ExecutionLimits) -> BaseTool:
    """Build the tool for one session. The tool message artifact lists the saved files."""

    @tool(TOOL_NAME, response_format="content_and_artifact")
    def run_python_tool(code: str) -> tuple[str, dict]:
        """Run Python code in a new process and return what it printed.

        Variables and imports do not persist between calls. The working directory holds
        data/ with copies of the user's files and outputs/ for files to show the user.
        """
        result = run_python(
            code,
            data_files=workspace.data_files(),
            artifact_dir=workspace.artifact_dir,
            limits=limits,
        )
        artifact = {
            "files": list(result.artifacts),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_s": round(result.duration_s, 3),
        }
        return format_for_model(result, limits), artifact

    return run_python_tool
