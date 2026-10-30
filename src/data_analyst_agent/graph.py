"""The ReAct loop: the model calls the Python tool until it answers without a tool call."""

from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

AGENT_NODE = "agent"
TOOLS_NODE = "tools"


def build_graph(
    model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str
) -> CompiledStateGraph:
    model_with_tools = model.bind_tools(list(tools))
    system_message = SystemMessage(system_prompt)

    def call_model(state: MessagesState) -> dict:
        response = model_with_tools.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node(AGENT_NODE, call_model)
    graph.add_node(TOOLS_NODE, ToolNode(list(tools)))
    graph.add_edge(START, AGENT_NODE)
    graph.add_conditional_edges(AGENT_NODE, route_after_agent, [TOOLS_NODE, END])
    graph.add_edge(TOOLS_NODE, AGENT_NODE)
    return graph.compile()


def route_after_agent(state: MessagesState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return TOOLS_NODE
    return END


def recursion_limit(max_model_calls: int) -> int:
    """LangGraph step limit that allows `max_model_calls` model calls and no more.

    Measured with LangGraph 1.0: n model calls with n - 1 tool rounds need a limit of 2n.
    tests/test_graph.py checks both sides of the boundary.
    """
    return 2 * max_model_calls
