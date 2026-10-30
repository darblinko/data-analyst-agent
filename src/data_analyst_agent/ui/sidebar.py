"""Sidebar: model choice, data files, conversation reset and token usage."""

from dataclasses import dataclass

import streamlit as st

from data_analyst_agent.demo_script import SAMPLE_DATA_PATH
from data_analyst_agent.models import DEMO_MODEL, DEMO_PROVIDER
from data_analyst_agent.settings import Settings
from data_analyst_agent.usage import TokenUsage
from data_analyst_agent.workspace import Workspace

PROVIDERS = ("openai", "anthropic")
PROVIDER_LABELS = {"openai": "OpenAI", "anthropic": "Anthropic"}
UPLOAD_TYPES = ["csv", "tsv", "txt", "json", "xlsx", "xls"]


@dataclass(frozen=True)
class ModelChoice:
    provider: str
    model: str


def render_sidebar(settings: Settings, workspace: Workspace) -> ModelChoice:
    with st.sidebar:
        st.title("data-analyst-agent")
        st.caption("Answers data questions by writing Python and running it.")
        choice = _model_section(settings)
        st.divider()
        _data_section(settings, workspace)
        st.divider()
        if st.button("New conversation", use_container_width=True):
            reset_conversation()
            st.rerun()
        _usage_section(settings)
        st.caption(
            f"Limits per code run: {settings.code_timeout_s:g} s, "
            f"{settings.max_output_bytes:,} bytes of output per stream. "
            f"Model calls per question: {settings.max_agent_iterations}."
        )
    return choice


def reset_conversation() -> None:
    st.session_state.history = ()
    st.session_state.transcript = ()
    st.session_state.session_usage = TokenUsage()
    st.session_state.last_usage = None


def _model_section(settings: Settings) -> ModelChoice:
    if settings.demo_mode:
        st.info(
            "Demo mode. Answers are replayed from a script, so no language model or API key "
            "is used. The code in each answer runs for real on the sample dataset."
        )
        return ModelChoice(DEMO_PROVIDER, DEMO_MODEL)
    provider = st.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(settings.default_provider),
        format_func=PROVIDER_LABELS.__getitem__,
    )
    model = st.selectbox("Model", settings.models_for(provider))
    return ModelChoice(provider, model)


def _data_section(settings: Settings, workspace: Workspace) -> None:
    st.subheader("Data files")
    if settings.demo_mode:
        st.caption(f"data/{SAMPLE_DATA_PATH.name} (sample dataset)")
        return
    uploads = st.file_uploader("Upload", type=UPLOAD_TYPES, accept_multiple_files=True)
    for upload in uploads or []:
        # The uploader returns its files on every rerun; add each one once.
        if upload.file_id not in st.session_state.processed_uploads:
            workspace.add_data(upload.name, upload.getvalue())
            st.session_state.processed_uploads |= {upload.file_id}
    if st.button("Add sample dataset", use_container_width=True):
        workspace.copy_in(SAMPLE_DATA_PATH)
    names = workspace.data_file_names()
    if not names:
        st.caption("No files yet.")
    for name in names:
        name_column, button_column = st.columns([4, 1])
        name_column.caption(f"data/{name}")
        if button_column.button("Remove", key=f"remove-{name}"):
            workspace.remove_data(name)
            st.rerun()


def _usage_section(settings: Settings) -> None:
    st.subheader("Token usage")
    if settings.demo_mode:
        st.caption("Not reported in demo mode, because no model is called.")
        return
    session: TokenUsage = st.session_state.session_usage
    last: TokenUsage | None = st.session_state.last_usage
    if last is not None and last.reported_calls == 0:
        st.caption("The provider did not report usage for the last question.")
    st.caption(
        f"Session: {session.input_tokens:,} in, {session.output_tokens:,} out, "
        f"{session.total_tokens:,} total"
    )
    if last is not None:
        st.caption(
            f"Last question: {last.total_tokens:,} tokens over {last.reported_calls} model calls"
        )
