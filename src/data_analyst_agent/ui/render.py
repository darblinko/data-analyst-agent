"""Draw chat entries, code runs and saved files."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import streamlit as st
import streamlit.components.v1 as components

from data_analyst_agent.transcript import (
    AssistantEntry,
    CodePart,
    Entry,
    Part,
    TextPart,
    apply_event,
)
from data_analyst_agent.turn import StepLimitReached, TurnEvent, TurnFinished
from data_analyst_agent.workspace import Workspace

HTML_HEIGHT_PX = 500
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg"}

logger = logging.getLogger(__name__)


def render_entry(entry: Entry, workspace: Workspace) -> None:
    if not isinstance(entry, AssistantEntry):
        with st.chat_message("user"):
            st.markdown(entry.text)
        return
    with st.chat_message("assistant"):
        for part in entry.parts:
            render_part(part, workspace)
        if entry.error:
            st.error(entry.error)


def render_part(part: Part, workspace: Workspace) -> None:
    if isinstance(part, TextPart):
        st.markdown(part.text)
        return
    st.code(part.code, language="python")
    if part.result is None:
        st.caption("Running...")
        return
    result = part.result
    succeeded = result.exit_code == 0 and not result.timed_out
    with st.expander(_result_label(part), expanded=not succeeded):
        st.code(result.output, language="text")
    for name in result.files:
        render_saved_file(name, part.call_id, workspace)


def render_saved_file(name: str, call_id: str, workspace: Workspace) -> None:
    path = workspace.artifact_path(name)
    if path is None:
        st.caption(f"{name} is no longer available.")
        return
    if path.suffix.lower() == ".html":
        # The HTML was written by generated code. It is shown in Streamlit's component iframe.
        components.html(path.read_text(encoding="utf-8"), height=HTML_HEIGHT_PX, scrolling=True)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        st.image(str(path))
    st.download_button(
        f"Download {name}", data=path.read_bytes(), file_name=name, key=f"download-{call_id}-{name}"
    )


@dataclass(frozen=True)
class ReplyOutcome:
    parts: tuple[Part, ...]
    finished: TurnFinished | None  # None when the turn failed
    error: str | None = None


def stream_reply(events: Iterable[TurnEvent], workspace: Workspace) -> ReplyOutcome:
    """Draw a reply while it streams. Each part gets a slot that is redrawn when it changes."""
    parts: tuple[Part, ...] = ()
    slots: list = []
    try:
        for event in events:
            if isinstance(event, TurnFinished):
                return ReplyOutcome(parts, event)
            updated = apply_event(parts, event)
            for index, part in enumerate(updated):
                if index == len(slots):
                    slots.append(st.empty())
                if index >= len(parts) or part != parts[index]:
                    with slots[index].container():
                        render_part(part, workspace)
            parts = updated
    except StepLimitReached as error:
        return ReplyOutcome(parts, None, str(error))
    except Exception as error:
        # Provider errors (authentication, rate limits, network) end the turn, not the app.
        logger.exception("Turn failed")
        return ReplyOutcome(parts, None, f"The request failed: {type(error).__name__}: {error}")
    return ReplyOutcome(parts, None, "The turn ended without a final answer.")


def _result_label(part: CodePart) -> str:
    result = part.result
    if result is None:
        return "Running"
    if result.timed_out:
        return "Output (timed out, process killed)"
    if result.exit_code != 0:
        return f"Output (failed, exit code {result.exit_code})"
    return "Output"
