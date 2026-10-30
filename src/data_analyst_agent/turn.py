"""Run one user turn through the graph and report it as a sequence of UI events."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from data_analyst_agent.graph import AGENT_NODE, TOOLS_NODE, recursion_limit
from data_analyst_agent.usage import TokenUsage, sum_usage


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class CodeSubmitted:
    call_id: str
    code: str


@dataclass(frozen=True)
class CodeFinished:
    call_id: str
    output: str
    files: tuple[str, ...]
    exit_code: int | None
    timed_out: bool


@dataclass(frozen=True)
class TurnFinished:
    new_messages: tuple[BaseMessage, ...]
    usage: TokenUsage


TurnEvent = TextDelta | CodeSubmitted | CodeFinished | TurnFinished


class StepLimitReached(RuntimeError):
    """The model kept calling tools past the configured number of model calls."""


def run_turn(
    graph: CompiledStateGraph,
    history: Sequence[BaseMessage],
    question: str,
    max_model_calls: int,
) -> Iterator[TurnEvent]:
    """Stream one turn. The last event is TurnFinished with the messages to keep."""
    inputs = {"messages": [*history, HumanMessage(question)]}
    config = {"recursion_limit": recursion_limit(max_model_calls)}
    final_messages: Sequence[BaseMessage] = ()
    streamed_text = False
    try:
        for mode, payload in graph.stream(
            inputs, config, stream_mode=["messages", "updates", "values"]
        ):
            if mode == "messages":
                chunk, metadata = payload
                if (
                    isinstance(chunk, AIMessageChunk)
                    and metadata.get("langgraph_node") == AGENT_NODE
                ):
                    text = text_of(chunk.content)
                    if text:
                        streamed_text = True
                        yield TextDelta(text)
            elif mode == "updates":
                yield from _events_from_update(payload, streamed_text)
                streamed_text = False
            else:
                final_messages = payload["messages"]
    except GraphRecursionError as error:
        raise StepLimitReached(
            f"Stopped after {max_model_calls} model calls without a final answer."
        ) from error
    new_messages = tuple(final_messages[len(history) :])
    yield TurnFinished(new_messages=new_messages, usage=sum_usage(new_messages))


def text_of(content: str | list) -> str:
    """Visible text of a message. Anthropic content is a list of typed blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _events_from_update(update: dict, streamed_text: bool) -> Iterator[TurnEvent]:
    for node, node_update in update.items():
        for message in (node_update or {}).get("messages", []):
            if node == AGENT_NODE and isinstance(message, AIMessage):
                # Models that do not stream deliver their text here in one piece.
                if not streamed_text and text_of(message.content):
                    yield TextDelta(text_of(message.content))
                for call in message.tool_calls:
                    yield CodeSubmitted(call_id=call["id"], code=call["args"].get("code", ""))
            elif node == TOOLS_NODE and isinstance(message, ToolMessage):
                yield _code_finished(message)


def _code_finished(message: ToolMessage) -> CodeFinished:
    artifact = message.artifact if isinstance(message.artifact, dict) else {}
    return CodeFinished(
        call_id=message.tool_call_id,
        output=text_of(message.content),
        files=tuple(artifact.get("files", ())),
        exit_code=artifact.get("exit_code"),
        timed_out=bool(artifact.get("timed_out", False)),
    )
