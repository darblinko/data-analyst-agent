"""Smoke tests of the Streamlit page with Streamlit's in-process test runner."""

from dataclasses import dataclass

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest

from data_analyst_agent.demo_script import DEMO_QUESTIONS, DEMO_SCRIPT
from data_analyst_agent.executor import MAX_ARTIFACT_BYTES
from data_analyst_agent.replay import ReplayChatModel
from data_analyst_agent.ui.sidebar import add_new_uploads
from data_analyst_agent.workspace import Workspace

TIMEOUT_S = 60


@pytest.fixture
def app_path(repo_root):
    return str(repo_root / "app.py")


def all_markdown(app: AppTest) -> str:
    return "\n".join(element.value for element in app.markdown)


def all_captions(app: AppTest) -> str:
    return "\n".join(element.value for element in app.caption)


@pytest.fixture
def replay_with_usage(monkeypatch):
    """Live-model mode with the replay model standing in for the provider."""
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        "data_analyst_agent.ui.app.get_chat_model",
        lambda provider, model: ReplayChatModel(script=DEMO_SCRIPT, reported_usage=(10, 4)),
    )


def test_demo_mode_answers_a_scripted_question(app_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr("data_analyst_agent.models.DEMO_TOKEN_DELAY_S", 0)
    app = AppTest.from_file(app_path, default_timeout=TIMEOUT_S).run()
    assert not app.exception
    assert app.title[0].value == "data-analyst-agent"

    button = next(b for b in app.button if b.label == DEMO_QUESTIONS[0])
    app = button.click().run()

    assert not app.exception
    assert DEMO_SCRIPT[0].steps[-1].text in all_markdown(app)
    assert app.code[0].value == DEMO_SCRIPT[0].steps[0].code.rstrip()
    assert "rows: 8,772, columns: 6" in app.code[1].value


def test_missing_api_key_is_shown_as_an_error(app_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "0")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")  # an empty value also blocks a local .env key
    app = AppTest.from_file(app_path, default_timeout=TIMEOUT_S).run()

    app = app.chat_input[0].set_value("How many rows?").run()

    assert not app.exception
    assert "OPENAI_API_KEY is not set" in app.error[0].value


def _render_saved_files(root: str, names: list[str]) -> None:
    from pathlib import Path

    from data_analyst_agent.ui.render import render_saved_file
    from data_analyst_agent.workspace import Workspace

    for name in names:
        render_saved_file(name, "call-1", Workspace(Path(root)))


def test_saved_file_over_the_size_limit_is_not_loaded(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "small.txt").write_text("hello")
    with (artifact_dir / "huge.bin").open("wb") as handle:
        handle.truncate(MAX_ARTIFACT_BYTES + 1)
    app = AppTest.from_function(
        _render_saved_files, args=(str(tmp_path), ["small.txt", "huge.bin"])
    ).run()

    assert not app.exception
    assert [b.label for b in app.get("download_button")] == ["Download small.txt"]
    assert "huge.bin is larger than 25,000,000 bytes" in app.caption[0].value


def test_failed_turn_still_counts_the_tokens_of_completed_model_calls(
    app_path, monkeypatch, replay_with_usage
):
    monkeypatch.setenv("MAX_AGENT_ITERATIONS", "2")  # the growth question needs three calls
    app = AppTest.from_file(app_path, default_timeout=TIMEOUT_S).run()

    app = app.chat_input[0].set_value(DEMO_SCRIPT[2].question).run()

    assert not app.exception
    assert "Stopped after 2 model calls" in app.error[0].value
    assert "Session: 20 in, 8 out, 28 total" in all_captions(app)
    assert "Last question: 28 tokens over 2 model calls" in all_captions(app)


class FailsAfterToolRun(ReplayChatModel):
    """Replay model that raises like a provider error once a tool result is in the input."""

    def next_step(self, messages):
        if isinstance(messages[-1], ToolMessage):
            raise RuntimeError("rate limited")
        return super().next_step(messages)


def test_provider_error_after_a_tool_run_keeps_the_output_and_the_next_turn_works(
    app_path, monkeypatch, replay_with_usage
):
    monkeypatch.setattr(
        "data_analyst_agent.ui.app.get_chat_model",
        lambda provider, model: FailsAfterToolRun(script=DEMO_SCRIPT),
    )
    app = AppTest.from_file(app_path, default_timeout=TIMEOUT_S).run()
    app = next(b for b in app.button if b.label == "Add sample dataset").click().run()

    app = app.chat_input[0].set_value(DEMO_SCRIPT[0].question).run()

    assert not app.exception
    assert "The request failed: RuntimeError: rate limited" in app.error[0].value
    assert "rows: 8,772, columns: 6" in app.code[1].value
    assert app.session_state["history"] == ()

    monkeypatch.setattr(
        "data_analyst_agent.ui.app.get_chat_model",
        lambda provider, model: ReplayChatModel(script=DEMO_SCRIPT),
    )
    app = app.chat_input[0].set_value(DEMO_SCRIPT[0].question).run()

    assert not app.exception
    assert DEMO_SCRIPT[0].steps[-1].text in all_markdown(app)
    history = app.session_state["history"]
    assert [m.content for m in history if isinstance(m, HumanMessage)] == [DEMO_SCRIPT[0].question]


def test_demo_chart_is_rendered_with_a_download_button(app_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setattr("data_analyst_agent.models.DEMO_TOKEN_DELAY_S", 0)
    app = AppTest.from_file(app_path, default_timeout=TIMEOUT_S).run()

    app = next(b for b in app.button if b.label == DEMO_QUESTIONS[1]).click().run()

    assert not app.exception
    labels = {b.label for b in app.get("download_button")}
    assert labels == {"Download monthly_revenue_by_region.html"}
    assert len(app.get("iframe")) >= 1


@dataclass(frozen=True)
class FakeUpload:
    file_id: str
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


def test_uploads_are_added_once_under_a_safe_name(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.data_dir.mkdir()
    uploads = [FakeUpload("id-1", "sales.csv", b"a\n1\n"), FakeUpload("id-2", "../x.csv", b"b")]

    seen = add_new_uploads(workspace, uploads, frozenset())
    workspace.remove_data("sales.csv")
    seen_again = add_new_uploads(workspace, uploads, seen)

    assert seen == seen_again == frozenset({"id-1", "id-2"})
    assert workspace.data_file_names() == ["x.csv"]
    assert (workspace.data_dir / "x.csv").read_bytes() == b"b"
