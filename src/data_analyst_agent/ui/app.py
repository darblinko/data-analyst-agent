"""Streamlit chat page. Start it with ./run.sh or `uv run streamlit run app.py`."""

import os

import streamlit as st

from data_analyst_agent.demo_script import DEMO_QUESTIONS, SAMPLE_DATA_PATH
from data_analyst_agent.executor import ExecutionLimits
from data_analyst_agent.graph import build_graph
from data_analyst_agent.models import MissingApiKeyError, get_chat_model
from data_analyst_agent.prompts import build_system_prompt
from data_analyst_agent.python_tool import make_python_tool
from data_analyst_agent.settings import Settings
from data_analyst_agent.transcript import AssistantEntry, UserEntry
from data_analyst_agent.turn import run_turn
from data_analyst_agent.ui.render import render_entry, stream_reply
from data_analyst_agent.ui.sidebar import ModelChoice, render_sidebar, reset_conversation
from data_analyst_agent.workspace import Workspace

INTRO = (
    "Ask a question about your data. The agent writes Python, runs it in a separate "
    "process, reads the printed output and answers from it."
)


def main() -> None:
    st.set_page_config(page_title="data-analyst-agent", layout="wide")
    settings = _load_settings()
    _init_state(settings)
    workspace: Workspace = st.session_state.workspace
    choice = render_sidebar(settings, workspace)

    st.title("data-analyst-agent")
    st.caption(INTRO)
    chat_area = st.container()
    with chat_area:
        if not st.session_state.transcript:
            st.markdown(_empty_state_text(settings))
        for entry in st.session_state.transcript:
            render_entry(entry, workspace)

    question = _demo_question_buttons() if settings.demo_mode else None
    typed = st.chat_input("Ask a question about the data")
    question = question or typed
    if question:
        with chat_area:
            _answer(question, choice, settings, workspace)
        st.rerun()


def _load_settings() -> Settings:
    try:
        return Settings.from_env(os.environ)
    except ValueError as error:
        st.error(f"Invalid configuration: {error}")
        st.stop()


def _init_state(settings: Settings) -> None:
    if "workspace" in st.session_state:
        return
    workspace = Workspace.create()
    if settings.demo_mode:
        workspace.copy_in(SAMPLE_DATA_PATH)
    st.session_state.workspace = workspace
    st.session_state.processed_uploads = frozenset()
    reset_conversation()


def _empty_state_text(settings: Settings) -> str:
    if settings.demo_mode:
        return "Pick one of the demo questions below. Typed questions work if they match one."
    return "Upload a CSV or Excel file in the sidebar, or add the sample dataset, then ask."


def _demo_question_buttons() -> str | None:
    clicked = None
    for column, text in zip(st.columns(len(DEMO_QUESTIONS)), DEMO_QUESTIONS, strict=True):
        if column.button(text, use_container_width=True):
            clicked = text
    return clicked


def _answer(question: str, choice: ModelChoice, settings: Settings, workspace: Workspace) -> None:
    user_entry = UserEntry(question)
    render_entry(user_entry, workspace)
    with st.chat_message("assistant"):
        try:
            model = get_chat_model(choice.provider, choice.model)
        except MissingApiKeyError as error:
            st.session_state.transcript += (user_entry, AssistantEntry((), str(error)))
            return
        limits = ExecutionLimits(settings.code_timeout_s, settings.max_output_bytes)
        prompt = build_system_prompt(limits, workspace.data_file_names())
        graph = build_graph(model, [make_python_tool(workspace, limits)], prompt)
        events = run_turn(graph, st.session_state.history, question, settings.max_agent_iterations)
        outcome = stream_reply(events, workspace)
    # A failed turn is shown but kept out of the history the model sees next time. Its
    # completed model calls were billed, so their usage is counted either way.
    st.session_state.transcript += (user_entry, AssistantEntry(outcome.parts, outcome.error))
    st.session_state.session_usage += outcome.usage
    st.session_state.last_usage = outcome.usage
    if outcome.finished is not None:
        st.session_state.history += outcome.finished.new_messages
