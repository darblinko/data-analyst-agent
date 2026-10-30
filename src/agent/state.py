"""
LangGraph state definition for the data-analyst-agent.
Tracks messages, execution context, and uploaded file information.
"""

from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State for the data-analyst-agent graph.

    Fields:
    - messages: Conversation history (HumanMessage, AIMessage, ToolMessage)
    - uploaded_files: Dict mapping filename -> absolute path for uploaded files
    - execution_count: Number of code executions in current turn (for limiting loops)
    - last_code: The last code block executed (for display in UI)
    - last_result: The last execution result (for display in UI)
    """

    messages: Annotated[list, add_messages]
    uploaded_files: dict[str, str]  # {filename: file_path}
    execution_count: int
    last_code: str
    last_result: str
