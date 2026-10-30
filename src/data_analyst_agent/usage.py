"""Token usage summed over the model calls of a turn or a session."""

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reported_calls: int = 0  # model responses that carried usage metadata

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reported_calls=self.reported_calls + other.reported_calls,
        )


def sum_usage(messages: Iterable[BaseMessage]) -> TokenUsage:
    """Add up the usage of every model response. A turn with tool calls has several."""
    total = TokenUsage()
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            total += TokenUsage(
                input_tokens=message.usage_metadata.get("input_tokens", 0),
                output_tokens=message.usage_metadata.get("output_tokens", 0),
                reported_calls=1,
            )
    return total
