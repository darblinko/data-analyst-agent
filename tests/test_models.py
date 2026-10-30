import pytest

from data_analyst_agent.models import (
    DEMO_PROVIDER,
    MissingApiKeyError,
    get_chat_model,
)
from data_analyst_agent.replay import ReplayChatModel
from data_analyst_agent.settings import Settings

RETIRED_PREFIXES = ("claude-3", "claude-2", "claude-instant")


def test_demo_provider_needs_no_key():
    model = get_chat_model(DEMO_PROVIDER, environ={})

    assert isinstance(model, ReplayChatModel)


@pytest.mark.parametrize(
    ("provider", "variable"), [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]
)
def test_missing_key_names_the_variable_and_the_demo_option(provider, variable):
    with pytest.raises(MissingApiKeyError, match=variable) as error:
        get_chat_model(provider, "some-model", environ={variable: "  "})

    assert "DEMO_MODE=1" in str(error.value)


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_chat_model("other", "m", environ={})


def test_openai_model_uses_the_configured_name():
    model = get_chat_model("openai", "gpt-test", environ={"OPENAI_API_KEY": "test-key"})

    assert model.model_name == "gpt-test"
    assert model.temperature == 0
    assert model.streaming is True


def test_anthropic_model_sends_no_temperature():
    model = get_chat_model(
        "anthropic", "claude-sonnet-5-5", environ={"ANTHROPIC_API_KEY": "test-key"}
    )

    assert model.model == "claude-sonnet-5-5"
    assert model.temperature is None
    assert model.max_tokens == 16_000


def test_default_anthropic_models_are_not_retired():
    for model in Settings().models_for("anthropic"):
        assert not model.startswith(RETIRED_PREFIXES)
