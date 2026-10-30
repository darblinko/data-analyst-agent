"""
LLM Manager - Factory pattern for multi-provider LLM support.
Supports OpenAI (GPT-4o, GPT-4o-mini) and Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku).
"""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel


# Available models per provider
AVAILABLE_MODELS = {
    "openai": {
        "gpt-4o": "GPT-4o (Latest, Most Capable)",
        "gpt-4o-mini": "GPT-4o Mini (Fast, Efficient)",
        "gpt-4-turbo": "GPT-4 Turbo",
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet (Latest)",
        "claude-3-5-haiku-20241022": "Claude 3.5 Haiku (Fast)",
        "claude-3-opus-20240229": "Claude 3 Opus (Most Capable)",
    },
}

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
}


def get_llm(provider: str, model: str | None = None, **kwargs: Any) -> BaseChatModel:
    """
    Factory function to get an LLM instance for the given provider.

    Args:
        provider: "openai" or "anthropic"
        model: Model name (optional, uses default if not specified)
        **kwargs: Additional kwargs passed to the LLM constructor

    Returns:
        BaseChatModel instance ready for use with LangGraph
    """
    provider = provider.lower().strip()

    if provider not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unsupported provider: '{provider}'. Choose from: {list(AVAILABLE_MODELS.keys())}"
        )

    # Use default model if not specified
    selected_model = model or DEFAULT_MODELS[provider]

    if selected_model not in AVAILABLE_MODELS[provider]:
        raise ValueError(
            f"Unsupported model '{selected_model}' for provider '{provider}'. "
            f"Choose from: {list(AVAILABLE_MODELS[provider].keys())}"
        )

    if provider == "openai":
        return _get_openai_llm(selected_model, **kwargs)
    elif provider == "anthropic":
        return _get_anthropic_llm(selected_model, **kwargs)


def _get_openai_llm(model: str, **kwargs: Any) -> BaseChatModel:
    """Create OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Install langchain-openai: pip install langchain-openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=kwargs.pop("temperature", 0),
        streaming=True,
        **kwargs,
    )


def _get_anthropic_llm(model: str, **kwargs: Any) -> BaseChatModel:
    """Create Anthropic LLM instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("Install langchain-anthropic: pip install langchain-anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=kwargs.pop("temperature", 0),
        streaming=True,
        **kwargs,
    )


def get_provider_model_options() -> dict[str, dict[str, str]]:
    """Return available models per provider for UI display."""
    return AVAILABLE_MODELS


def get_default_model(provider: str) -> str:
    """Return the default model for a provider."""
    return DEFAULT_MODELS.get(provider.lower(), "")
