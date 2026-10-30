"""Runtime settings read from environment variables. See .env.example for the list."""

from collections.abc import Mapping
from dataclasses import dataclass

PROVIDERS = ("openai", "anthropic")
TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"", "0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    demo_mode: bool = False
    default_provider: str = "openai"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_small_model: str = "claude-haiku-5-5"
    code_timeout_s: float = 30.0
    max_output_bytes: int = 20_000
    max_agent_iterations: int = 10

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "Settings":
        defaults = cls()
        provider = environ.get("LLM_PROVIDER", defaults.default_provider).strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {', '.join(PROVIDERS)}, got {provider!r}"
            )
        return cls(
            demo_mode=_read_bool(environ, "DEMO_MODE", defaults.demo_mode),
            default_provider=provider,
            openai_model=_read_text(environ, "OPENAI_MODEL", defaults.openai_model),
            anthropic_model=_read_text(environ, "ANTHROPIC_MODEL", defaults.anthropic_model),
            anthropic_small_model=_read_text(
                environ, "ANTHROPIC_SMALL_MODEL", defaults.anthropic_small_model
            ),
            code_timeout_s=_read_number(environ, "CODE_TIMEOUT_S", defaults.code_timeout_s, float),
            max_output_bytes=_read_number(
                environ, "MAX_OUTPUT_BYTES", defaults.max_output_bytes, int
            ),
            max_agent_iterations=_read_number(
                environ, "MAX_AGENT_ITERATIONS", defaults.max_agent_iterations, int
            ),
        )

    def models_for(self, provider: str) -> tuple[str, ...]:
        """Model IDs offered in the UI for a provider, main model first, without duplicates."""
        if provider == "openai":
            return (self.openai_model,)
        if provider == "anthropic":
            return tuple(dict.fromkeys((self.anthropic_model, self.anthropic_small_model)))
        raise ValueError(f"Unknown provider {provider!r}")


def _read_text(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name, "").strip()
    return value or default


def _read_bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    if name not in environ:
        return default
    value = environ[name].strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of 1, 0, true, false, yes, no; got {environ[name]!r}")


def _read_number(
    environ: Mapping[str, str], name: str, default: float, kind: type[int] | type[float]
) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = kind(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a {kind.__name__}, got {raw!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0, got {raw!r}")
    return value
