import pytest

from data_analyst_agent.settings import Settings


def test_defaults_when_environment_is_empty():
    settings = Settings.from_env({})

    assert settings == Settings()
    assert settings.demo_mode is False
    assert settings.default_provider == "openai"
    assert settings.openai_model == "gpt-4o"
    assert settings.anthropic_model == "claude-sonnet-5-5"
    assert settings.anthropic_small_model == "claude-haiku-5-5"
    assert settings.code_timeout_s == 30.0
    assert settings.max_output_bytes == 20_000
    assert settings.max_agent_iterations == 10


def test_every_variable_is_read():
    environ = {
        "DEMO_MODE": "1",
        "LLM_PROVIDER": "Anthropic",
        "OPENAI_MODEL": "gpt-test",
        "ANTHROPIC_MODEL": "claude-main-test",
        "ANTHROPIC_SMALL_MODEL": "claude-small-test",
        "CODE_TIMEOUT_S": "2.5",
        "MAX_OUTPUT_BYTES": "512",
        "MAX_AGENT_ITERATIONS": "3",
    }

    settings = Settings.from_env(environ)

    assert settings == Settings(
        demo_mode=True,
        default_provider="anthropic",
        openai_model="gpt-test",
        anthropic_model="claude-main-test",
        anthropic_small_model="claude-small-test",
        code_timeout_s=2.5,
        max_output_bytes=512,
        max_agent_iterations=3,
    )


def test_env_example_lists_exactly_the_variables_settings_reads(repo_root):
    names = {
        line.split("=", 1)[0]
        for line in (repo_root / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    }
    read_by_settings = {
        "DEMO_MODE",
        "LLM_PROVIDER",
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_MODEL",
        "CODE_TIMEOUT_S",
        "MAX_OUTPUT_BYTES",
        "MAX_AGENT_ITERATIONS",
    }
    read_by_model_factory = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}

    assert names == read_by_settings | read_by_model_factory


@pytest.mark.parametrize(
    ("raw", "expected"), [("true", True), ("YES", True), ("0", False), ("", False)]
)
def test_demo_mode_accepts_common_boolean_spellings(raw, expected):
    assert Settings.from_env({"DEMO_MODE": raw}).demo_mode is expected


def test_blank_model_variable_falls_back_to_default():
    assert Settings.from_env({"ANTHROPIC_MODEL": "  "}).anthropic_model == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({"LLM_PROVIDER": "gemini"}, "LLM_PROVIDER"),
        ({"DEMO_MODE": "maybe"}, "DEMO_MODE"),
        ({"CODE_TIMEOUT_S": "ten"}, "CODE_TIMEOUT_S"),
        ({"CODE_TIMEOUT_S": "0"}, "CODE_TIMEOUT_S"),
        ({"MAX_OUTPUT_BYTES": "1.5"}, "MAX_OUTPUT_BYTES"),
        ({"MAX_AGENT_ITERATIONS": "-1"}, "MAX_AGENT_ITERATIONS"),
    ],
)
def test_invalid_values_name_the_variable(environ, message):
    with pytest.raises(ValueError, match=message):
        Settings.from_env(environ)


def test_models_for_lists_main_model_first_without_duplicates():
    settings = Settings(anthropic_model="claude-a", anthropic_small_model="claude-a")

    assert settings.models_for("anthropic") == ("claude-a",)
    assert Settings().models_for("anthropic") == ("claude-sonnet-5-5", "claude-haiku-5-5")
    assert Settings().models_for("openai") == ("gpt-4o",)
    with pytest.raises(ValueError):
        Settings().models_for("demo")
