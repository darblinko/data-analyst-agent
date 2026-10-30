"""
LangGraph agent graph for data-analyst-agent.

Implements a ReAct-style agent loop:
  agent_node → (if tool call) → tool_node → agent_node → ...
               (if done)       → END

The agent has access to one tool: python_repl (code executor).
State persists across the full agent loop within a single user turn.
"""

import os
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.llm_manager import get_llm
from src.tools.code_executor import python_repl, reset_execution_namespace

# ─── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are data-analyst-agent, an expert data analyst and programmer with access to a Python code execution environment.

## Your Capabilities
You can execute Python code to:
- Analyze data files (CSV, Excel) using pandas
- Fetch and analyze stock market data using yfinance
- Perform statistical analysis and create visualizations
- Do technical analysis (RSI, MACD, Bollinger Bands, moving averages)
- Answer quantitative questions by writing and running code

## How to Use Your Code Tool
Use the `python_repl` tool to execute Python code. Important notes:
- **Variables persist** between tool calls in the same session - you can build on previous code
- **Print your results** explicitly - only printed output is returned
- `PROJECT_ROOT` and `UPLOADS_DIR` variables are pre-injected for file access
- All major data science libraries are available: pandas, numpy, scipy, matplotlib, plotly, yfinance
- If code fails, read the error and fix it in the next tool call

## Workflow
1. Understand the user's question
2. Write Python code to solve it - start simple, iterate if needed
3. Execute the code and examine results
4. If results are incomplete or wrong, write corrected code and run again
5. Interpret the results and give a clear, insightful final answer

## Output Format
- Explain what you're doing before each code execution
- After getting results, provide a clear interpretation in natural language
- For financial analysis, explain trends, signals, and what they mean
- For data analysis, explain the statistics and what they reveal about the data

## Creating Visualizations (IMPORTANT)
When creating charts or plots:
1. **Use Plotly** (preferred for interactive charts) rather than matplotlib
2. Save interactive plots as HTML files: `fig.write_html(f"{UPLOADS_DIR}/chart_name.html")`
3. For static images, use: `fig.write_image(f"{UPLOADS_DIR}/chart_name.png")` (requires kaleido)
4. **Always print the file path** after saving: `print(f"VISUALIZATION_SAVED: {UPLOADS_DIR}/chart_name.html")`
5. The UI will automatically detect and display saved visualizations

Example:
```python
import plotly.express as px
fig = px.scatter(df, x='time', y='value', title='My Chart')
chart_path = f"{UPLOADS_DIR}/my_chart.html"
fig.write_html(chart_path)
print(f"VISUALIZATION_SAVED: {chart_path}")
```

## File Access
- Uploaded files are in: UPLOADS_DIR (use this variable in your code)
- You may read/write files only within the project directory
"""

# ─── Agent node ──────────────────────────────────────────────────────────────


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Call the LLM with the current message history.
    The LLM decides whether to call python_repl or give a final answer.
    """
    configurable = config.get("configurable", {})
    provider = configurable.get("provider", "openai")
    model = configurable.get("model", None)

    llm = get_llm(provider=provider, model=model)
    tools = [python_repl]
    llm_with_tools = llm.bind_tools(tools)

    messages = state["messages"]

    # Prepend system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)

    # Track last code block for UI display
    last_code = state.get("last_code", "")
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc.get("name") == "python_repl":
                last_code = tc.get("args", {}).get("code", last_code)

    return {
        "messages": [response],
        "last_code": last_code,
        "execution_count": state.get("execution_count", 0),
    }


# ─── Tool node ────────────────────────────────────────────────────────────────

tool_node = ToolNode(tools=[python_repl])


# ─── Conditional routing ─────────────────────────────────────────────────────


def should_continue(state: AgentState) -> Literal["tools", END]:
    """
    Route to the tool node if the last AI message has tool calls,
    otherwise end the graph and return the final answer.
    """
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ─── Graph construction ───────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    """Build and compile the data-analyst-agent graph."""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # Set entry point
    builder.set_entry_point("agent")

    # Add conditional edge from agent
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )

    # Always go back to agent after tool execution
    builder.add_edge("tools", "agent")

    return builder.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    """Return the compiled agent graph (singleton)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(
    user_message: str,
    provider: str,
    model: str,
    conversation_history: list | None = None,
    uploaded_files: dict[str, str] | None = None,
    reset_namespace: bool = True,
):
    """
    Run the agent for a single user message and yield state updates.

    Args:
        user_message: The user's question or instruction
        provider: LLM provider ('openai' or 'anthropic')
        model: Model name
        conversation_history: Prior LangChain BaseMessage objects to pass as context
        uploaded_files: Dict of {filename: filepath} for uploaded files
        reset_namespace: Whether to reset the Python execution namespace

    Yields:
        LangGraph stream events as (mode, data) tuples where mode is
        "messages" (data = (chunk, metadata)) or "values" (data = state_dict)
    """
    from langchain_core.messages import HumanMessage

    if reset_namespace:
        reset_execution_namespace()

    graph = get_graph()

    initial_state: AgentState = {
        # Prepend conversation history so the agent sees prior context
        "messages": [*(conversation_history or []), HumanMessage(content=user_message)],
        "uploaded_files": uploaded_files or {},
        "execution_count": 0,
        "last_code": "",
        "last_result": "",
    }

    config = {
        "configurable": {
            "provider": provider,
            "model": model,
        }
    }

    # Stream both message chunks (for incremental UI) and state values (for history)
    for event in graph.stream(
        initial_state, config=config, stream_mode=["messages", "values"]
    ):
        yield event
