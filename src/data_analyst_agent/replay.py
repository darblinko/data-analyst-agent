"""A chat model that replays a fixed script of text and tool calls.

It stands in for a real model in demo mode and in tests. It makes no network calls. The
tool calls it emits are executed for real by the graph, so the code output a user sees
in demo mode is produced by the executor, not by the script.
"""

import json
import re
import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from data_analyst_agent.python_tool import TOOL_NAME

_WORD_PATTERN = re.compile(r"\S+\s*|\s+")


@dataclass(frozen=True)
class ScriptStep:
    """One model response: text, plus code for the tool when `code` is set."""

    text: str
    code: str | None = None


@dataclass(frozen=True)
class ScriptedQuestion:
    question: str
    steps: tuple[ScriptStep, ...]


class ReplayChatModel(BaseChatModel):
    """Answers the scripted questions step by step; other input gets a help message."""

    script: tuple[ScriptedQuestion, ...]
    tool_name: str = TOOL_NAME
    token_delay_s: float = 0.0
    bound_tools: tuple[str, ...] = ()

    @property
    def _llm_type(self) -> str:
        return "scripted-replay"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ReplayChatModel":
        names = tuple(convert_to_openai_tool(t)["function"]["name"] for t in tools)
        return self.model_copy(update={"bound_tools": names})

    def next_step(self, messages: Sequence[BaseMessage]) -> ScriptStep:
        """The step that follows the last user question in `messages`."""
        question_index = _last_human_index(messages)
        if question_index is None:
            return ScriptStep(self._help_text())
        question = _normalize(_text_of(messages[question_index]))
        scripted = next((q for q in self.script if _normalize(q.question) == question), None)
        if scripted is None:
            return ScriptStep(self._help_text())
        answered = sum(isinstance(m, AIMessage) for m in messages[question_index + 1 :])
        if answered >= len(scripted.steps):
            return ScriptStep("The scripted answer to this question is complete.")
        step = scripted.steps[answered]
        if step.code is not None and self.tool_name not in self.bound_tools:
            raise ValueError(f"The script calls {self.tool_name!r}, which is not bound")
        return step

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        step = self.next_step(messages)
        tool_calls = []
        if step.code is not None:
            tool_calls.append(
                {"name": self.tool_name, "args": {"code": step.code}, "id": _new_call_id()}
            )
        message = AIMessage(content=step.text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # The base class reports each chunk to the callbacks, so this method does not.
        step = self.next_step(messages)
        for piece in _WORD_PATTERN.findall(step.text):
            if self.token_delay_s:
                time.sleep(self.token_delay_s)
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
        if step.code is not None:
            tool_call_chunk = {
                "name": self.tool_name,
                "args": json.dumps({"code": step.code}),
                "id": _new_call_id(),
                "index": 0,
                "type": "tool_call_chunk",
            }
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", tool_call_chunks=[tool_call_chunk])
            )

    def _help_text(self) -> str:
        listed = "\n".join(f"- {q.question}" for q in self.script)
        return (
            "This is the scripted demo. It replays prepared answers and does not call a "
            f"language model, so it can only answer these questions:\n{listed}"
        )


def _last_human_index(messages: Sequence[BaseMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return index
    return None


def _text_of(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in message.content
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".?!")


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"
