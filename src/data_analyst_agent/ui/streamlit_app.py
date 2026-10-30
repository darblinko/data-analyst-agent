"""
Streamlit UI for data-analyst-agent.

Features:
- Chat interface with multi-turn conversation support
- Conversation history persisted across turns via LangChain BaseMessage objects
- File upload widget for CSV/Excel data files
- Incremental token streaming (true character-by-character display)
- Syntax-highlighted code display as the agent writes code
- LLM provider and model selection in the sidebar
- Code execution results displayed inline with expanders
"""

import os
import sys
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
import streamlit.components.v1 as components
import re
import json
from datetime import datetime

from data_analyst_agent.agent.llm_manager import get_provider_model_options, get_default_model
from data_analyst_agent.agent.graph import run_agent
from data_analyst_agent.tools.code_executor import reset_execution_namespace

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="data-analyst-agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State Initialization ─────────────────────────────────────────────


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        # Display messages: [{role, content, type}]
        st.session_state.messages = []
    if "langgraph_messages" not in st.session_state:
        # LangChain BaseMessage objects for conversation history persistence
        st.session_state.langgraph_messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}  # {filename: filepath}
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
    if "model" not in st.session_state:
        st.session_state.model = get_default_model("openai")
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
    if "total_output_tokens" not in st.session_state:
        st.session_state.total_output_tokens = 0
    if "last_input_tokens" not in st.session_state:
        st.session_state.last_input_tokens = 0
    if "last_output_tokens" not in st.session_state:
        st.session_state.last_output_tokens = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────


def render_sidebar():
    """Render the sidebar with LLM settings and file upload."""
    with st.sidebar:
        st.title("🤖 data-analyst-agent")
        st.caption("AI Agent with Code Execution")

        st.divider()

        # LLM Provider Selection
        st.subheader("🤖 Model Settings")
        model_options = get_provider_model_options()
        provider_names = list(model_options.keys())

        selected_provider = st.selectbox(
            "Provider",
            options=provider_names,
            index=provider_names.index(st.session_state.provider),
            format_func=lambda p: p.capitalize(),
        )

        if selected_provider != st.session_state.provider:
            st.session_state.provider = selected_provider
            st.session_state.model = get_default_model(selected_provider)

        # Model Selection
        models_for_provider = model_options[selected_provider]
        model_names = list(models_for_provider.keys())
        model_labels = list(models_for_provider.values())

        current_model_idx = (
            model_names.index(st.session_state.model)
            if st.session_state.model in model_names
            else 0
        )

        selected_model_label = st.selectbox(
            "Model",
            options=model_labels,
            index=current_model_idx,
        )
        # Map label back to model name
        selected_model = model_names[model_labels.index(selected_model_label)]
        st.session_state.model = selected_model

        st.divider()

        # File Upload
        st.subheader("📁 Upload Data Files")
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload a data file for analysis. The agent can read it from the uploads directory.",
        )

        if uploaded_file is not None:
            # Save uploaded file to uploads directory
            uploads_dir = PROJECT_ROOT / "uploads"
            uploads_dir.mkdir(exist_ok=True)

            file_path = uploads_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_files[uploaded_file.name] = str(file_path)
            st.success(f"✅ {uploaded_file.name} uploaded")

        # Show uploaded files
        if st.session_state.uploaded_files:
            st.caption("**Uploaded files:**")
            for fname in list(st.session_state.uploaded_files.keys()):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(f"📄 {fname}")
                with col2:
                    if st.button("✕", key=f"del_{fname}", help=f"Remove {fname}"):
                        file_path = Path(st.session_state.uploaded_files[fname])
                        if file_path.exists():
                            file_path.unlink()
                        del st.session_state.uploaded_files[fname]
                        st.rerun()

        st.divider()

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.langgraph_messages = []
            st.session_state.total_input_tokens = 0
            st.session_state.total_output_tokens = 0
            st.session_state.last_input_tokens = 0
            st.session_state.last_output_tokens = 0
            reset_execution_namespace()
            st.rerun()

        # Status indicator
        st.divider()
        provider_display = st.session_state.provider.capitalize()
        st.caption(f"**Active:** {provider_display} / {st.session_state.model}")

        if st.session_state.uploaded_files:
            st.caption(f"**Files:** {len(st.session_state.uploaded_files)} uploaded")

        turn_count = len(
            [m for m in st.session_state.messages if m["role"] == "user"]
        )
        if turn_count > 0:
            st.caption(f"**Turns:** {turn_count}")
        
        # Token usage summary
        if st.session_state.total_input_tokens > 0 or st.session_state.total_output_tokens > 0:
            st.divider()
            st.caption("**Token Usage (Session):**")
            st.caption(f"Input: {st.session_state.total_input_tokens:,}")
            st.caption(f"Output: {st.session_state.total_output_tokens:,}")
            st.caption(f"Total: {st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}")


# ─── Chat Display ─────────────────────────────────────────────────────────────


def render_message(msg: dict):
    """Render a single chat message with appropriate formatting."""
    role = msg["role"]
    content = msg["content"]
    msg_type = msg.get("type", "text")

    with st.chat_message(role):
        if msg_type == "code":
            st.markdown("**🐍 Code executed:**")
            st.code(content, language="python")
        elif msg_type == "tool_result":
            with st.expander("📤 Execution Output", expanded=False):
                st.text(content)
        elif msg_type == "viz_html":
            # Render saved HTML visualization
            viz_path = content.replace("visualization:", "")
            viz_file = Path(viz_path)
            if viz_file.exists():
                st.markdown("**📊 Interactive Visualization:**")
                with open(viz_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                components.html(html_content, height=600, scrolling=True)
                with open(viz_file, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {viz_file.name}",
                        data=f,
                        file_name=viz_file.name,
                        mime="text/html",
                        key=f"download_html_{viz_file.name}_{id(msg)}"
                    )
        elif msg_type == "viz_image":
            # Render saved image visualization
            viz_path = content.replace("visualization:", "")
            viz_file = Path(viz_path)
            if viz_file.exists():
                st.markdown("**📊 Visualization:**")
                st.image(str(viz_file))
                with open(viz_file, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {viz_file.name}",
                        data=f,
                        file_name=viz_file.name,
                        mime=f"image/{viz_file.suffix[1:]}",
                        key=f"download_img_{viz_file.name}_{id(msg)}"
                    )
        else:
            st.markdown(content)


def render_chat_history():
    """Render all messages in the chat history."""
    for msg in st.session_state.messages:
        render_message(msg)


# ─── Incremental Streaming ────────────────────────────────────────────────────


def stream_agent_response(user_input: str):
    """
    Run the agent with true incremental streaming and conversation history.

    Event loop handles:
    - ("messages", (chunk, metadata)) → stream text tokens / capture tool msgs
    - ("values", state_dict) → detect pending tool code / capture final history

    After streaming, persists updated message list to session state.
    """
    uploaded_files = st.session_state.uploaded_files.copy()

    # Enhance input with file context if files are uploaded
    enhanced_input = user_input
    if uploaded_files:
        file_list = "\n".join(
            [f"- {fname}: `{fpath}`" for fname, fpath in uploaded_files.items()]
        )
        enhanced_input = (
            f"{user_input}\n\n"
            f"**Uploaded files available:**\n{file_list}\n"
            f"Access them using the UPLOADS_DIR variable or the exact paths above."
        )

    # Track what the agent wrote (for chat history replay)
    display_messages = []  # [{role, content, type}]
    final_state = None
    final_text = ""
    
    # Token tracking for this turn
    turn_input_tokens = 0
    turn_output_tokens = 0
    
    # Initialize LLM interaction log
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"llm_interactions_{datetime.now().strftime('%Y%m%d')}.log"

    with st.chat_message("assistant"):
        # -- Streaming state --
        text_placeholder = st.empty()  # current AI text segment
        full_text = ""                  # accumulated text for current segment
        pending_code = None            # Python code from latest tool call

        try:
            for event in run_agent(
                user_message=enhanced_input,
                provider=st.session_state.provider,
                model=st.session_state.model,
                conversation_history=st.session_state.langgraph_messages,
                uploaded_files=uploaded_files,
                reset_namespace=False,  # preserve namespace across turns
            ):
                mode, data = event

                # ── Message chunks: stream tokens, detect tool results ─────
                if mode == "messages":
                    chunk, metadata = data

                    # AI text / tool call chunks
                    if isinstance(chunk, (AIMessageChunk, AIMessage)):
                        # Extract token usage metadata
                        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                            usage = chunk.usage_metadata
                            if "input_tokens" in usage:
                                turn_input_tokens = usage["input_tokens"]
                            if "output_tokens" in usage:
                                turn_output_tokens = usage["output_tokens"]
                        
                        # Log the LLM interaction
                        if hasattr(chunk, "response_metadata") or hasattr(chunk, "usage_metadata"):
                            with open(log_file, "a", encoding="utf-8") as f:
                                log_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "provider": st.session_state.provider,
                                    "model": st.session_state.model,
                                    "message_type": type(chunk).__name__,
                                    "content": chunk.content if isinstance(chunk.content, str) else str(chunk.content),
                                    "usage_metadata": chunk.usage_metadata if hasattr(chunk, "usage_metadata") else None,
                                    "response_metadata": chunk.response_metadata if hasattr(chunk, "response_metadata") else None,
                                }
                                f.write(json.dumps(log_entry) + "\n")
                        
                        content = chunk.content
                        if isinstance(content, str) and content:
                            full_text += content
                            text_placeholder.markdown(full_text + "▌")
                        elif isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_chunk = item.get("text", "")
                                    if text_chunk:
                                        full_text += text_chunk
                                        text_placeholder.markdown(full_text + "▌")

                    # Tool execution result
                    elif isinstance(chunk, ToolMessage):
                        # Commit the AI's explanation text (remove cursor)
                        if full_text:
                            text_placeholder.markdown(full_text)
                            display_messages.append(
                                {"role": "assistant", "content": full_text, "type": "text"}
                            )
                            full_text = ""

                        # Show the code that was just run
                        if pending_code:
                            st.markdown("**🐍 Code executed:**")
                            st.code(pending_code, language="python")
                            display_messages.append(
                                {"role": "assistant", "content": pending_code, "type": "code"}
                            )
                            pending_code = None

                        # Check for saved visualizations in the output
                        tool_output = chunk.content
                        viz_match = re.search(r"VISUALIZATION_SAVED:\s*(.+?)(?:\n|$)", tool_output)
                        
                        if viz_match:
                            viz_path = viz_match.group(1).strip()
                            viz_file = Path(viz_path)
                            
                            if viz_file.exists():
                                if viz_file.suffix.lower() == ".html":
                                    # Display interactive Plotly chart
                                    st.markdown("**📊 Interactive Visualization:**")
                                    with open(viz_file, "r", encoding="utf-8") as f:
                                        html_content = f.read()
                                    components.html(html_content, height=600, scrolling=True)
                                    
                                    # Also provide download button
                                    with open(viz_file, "rb") as f:
                                        st.download_button(
                                            label=f"⬇️ Download {viz_file.name}",
                                            data=f,
                                            file_name=viz_file.name,
                                            mime="text/html"
                                        )
                                    display_messages.append(
                                        {"role": "assistant", "content": f"visualization:{viz_path}", "type": "viz_html"}
                                    )
                                    
                                elif viz_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".svg"]:
                                    # Display static image
                                    st.markdown("**📊 Visualization:**")
                                    st.image(str(viz_file))
                                    
                                    # Provide download button
                                    with open(viz_file, "rb") as f:
                                        st.download_button(
                                            label=f"⬇️ Download {viz_file.name}",
                                            data=f,
                                            file_name=viz_file.name,
                                            mime=f"image/{viz_file.suffix[1:]}"
                                        )
                                    display_messages.append(
                                        {"role": "assistant", "content": f"visualization:{viz_path}", "type": "viz_image"}
                                    )

                        # Show execution output with token counter
                        with st.expander("📤 Execution Output", expanded=False):
                            # Display token usage if available
                            if turn_input_tokens > 0 or turn_output_tokens > 0:
                                st.caption(
                                    f"**Tokens:** Input: {turn_input_tokens}, "
                                    f"Input (Agg): {st.session_state.total_input_tokens + turn_input_tokens}, "
                                    f"Output: {turn_output_tokens}, "
                                    f"Output (Agg): {st.session_state.total_output_tokens + turn_output_tokens}"
                                )
                                st.divider()
                            st.text(tool_output)
                        display_messages.append(
                            {"role": "assistant", "content": tool_output, "type": "tool_result"}
                        )

                        # New text placeholder for agent's interpretation
                        text_placeholder = st.empty()

                # ── State snapshots: detect pending code / capture history ─
                elif mode == "values":
                    final_state = data
                    msgs = data.get("messages", [])
                    if msgs:
                        last_msg = msgs[-1]
                        # AIMessage with tool_calls → extract the Python code
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                if tc.get("name") == "python_repl":
                                    pending_code = tc.get("args", {}).get("code", "")

            # Final text render (remove streaming cursor)
            if full_text:
                text_placeholder.markdown(full_text)
                display_messages.append(
                    {"role": "assistant", "content": full_text, "type": "text"}
                )
                final_text = full_text

        except Exception as e:
            err = traceback.format_exc()
            error_msg = f"❌ **Agent error:** {str(e)}\n\n```\n{err}\n```"
            text_placeholder.error(str(e))
            st.expander("Error details").markdown(f"```\n{err}\n```")
            display_messages.append(
                {"role": "assistant", "content": error_msg, "type": "text"}
            )

    # ── Persist conversation history ──────────────────────────────────────────
    if final_state and "messages" in final_state:
        all_messages = final_state["messages"]
        # New messages = everything after the prior history
        prior_len = len(st.session_state.langgraph_messages)
        new_messages = all_messages[prior_len:]
        st.session_state.langgraph_messages.extend(new_messages)
        
        # Log the full conversation state
        with open(log_file, "a", encoding="utf-8") as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event": "conversation_state",
                "turn_input_tokens": turn_input_tokens,
                "turn_output_tokens": turn_output_tokens,
                "total_messages": len(all_messages),
            }
            f.write(json.dumps(log_entry) + "\n")

    # ── Update token counters ─────────────────────────────────────────────────
    if turn_input_tokens > 0 or turn_output_tokens > 0:
        st.session_state.total_input_tokens += turn_input_tokens
        st.session_state.total_output_tokens += turn_output_tokens
        st.session_state.last_input_tokens = turn_input_tokens
        st.session_state.last_output_tokens = turn_output_tokens

    # ── Update display chat history ───────────────────────────────────────────
    for msg in display_messages:
        st.session_state.messages.append(msg)


# ─── Main App ─────────────────────────────────────────────────────────────────


def main():
    """Main Streamlit application."""
    init_session_state()
    render_sidebar()

    # Main chat area
    st.title("🤖 data-analyst-agent")
    st.caption("An AI agent that writes and executes Python code to answer your questions.")

    # Welcome message when conversation is empty
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                """
**Welcome to data-analyst-agent!** I'm an AI agent that can write and execute Python code to help you analyze data and answer questions.

**Try these tasks:**
- 📊 Upload a CSV or Excel file and ask: *"What are the key statistics in this dataset?"*
- 📈 Ask about stocks: *"Analyze AAPL stock over the past 6 months and give me a technical analysis"*
- 🧮 Any data question: *"Fetch TSLA and NVDA stock data, compare their performance, and calculate Sharpe ratios"*

**How it works:**
1. I write Python code to solve your problem
2. I execute it locally and see the results  
3. I interpret the results and give you a clear answer
4. If something doesn't work, I fix my code and try again
5. I **remember our conversation** — ask follow-up questions freely!

Upload a file in the sidebar or just type your question below!
"""
            )

    # Render existing chat history
    render_chat_history()

    # Chat input
    if prompt := st.chat_input(
        "Ask a question or upload a file and describe what you want to analyze...",
        disabled=st.session_state.is_running,
    ):
        # Add user message to display history
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "type": "text"}
        )

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run agent with streaming
        st.session_state.is_running = True
        try:
            stream_agent_response(prompt)
        finally:
            st.session_state.is_running = False


if __name__ == "__main__":
    main()
