"""Chat model factory for the OpenAI, Anthropic and scripted demo providers."""

import os
from collections.abc import Mapping

from langchain_core.language_models.chat_models import BaseChatModel

from data_analyst_agent.demo_script import DEMO_SCRIPT
from data_analyst_agent.replay import ReplayChatModel

DEMO_PROVIDER = "demo"
DEMO_MODEL = "scripted-replay"
DEMO_TOKEN_DELAY_S = 0.03
ANTHROPIC_MAX_TOKENS = 16_000
API_KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class MissingApiKeyError(ValueError):
    """Raised when the selected provider has no API key in the environment."""


def get_chat_model(
    provider: str, model: str = DEMO_MODEL, environ: Mapping[str, str] | None = None
) -> BaseChatModel:
    """Return a streaming chat model. The demo provider needs no key and no network."""
    if provider == DEMO_PROVIDER:
        return ReplayChatModel(script=DEMO_SCRIPT, token_delay_s=DEMO_TOKEN_DELAY_S)
    if provider not in API_KEY_VARIABLES:
        raise ValueError(f"Unknown provider {provider!r}")
    api_key = _require_key(provider, os.environ if environ is None else environ)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=0, streaming=True)

    from langchain_anthropic import ChatAnthropic

    # No temperature: current Claude models reject non-default sampling parameters.
    return ChatAnthropic(
        model=model, api_key=api_key, max_tokens=ANTHROPIC_MAX_TOKENS, streaming=True
    )


def _require_key(provider: str, environ: Mapping[str, str]) -> str:
    variable = API_KEY_VARIABLES[provider]
    key = environ.get(variable, "").strip()
    if not key:
        raise MissingApiKeyError(
            f"{variable} is not set. Add it to .env, or start with DEMO_MODE=1 to run "
            "the scripted demo without a key."
        )
    return key
