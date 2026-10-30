"""Smoke tests of the Streamlit page with Streamlit's in-process test runner."""

import pytest
from streamlit.testing.v1 import AppTest

from data_analyst_agent.demo_script import DEMO_QUESTIONS, DEMO_SCRIPT
from data_analyst_agent.executor import MAX_ARTIFACT_BYTES

TIMEOUT_S = 60


@pytest.fixture
def app_path(repo_root):
    return str(repo_root / "app.py")


def all_markdown(app: AppTest) -> str:
    return "\n".join(element.value for element in app.markdown)


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
