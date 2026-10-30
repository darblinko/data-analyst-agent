"""What the chat shows: user questions and assistant replies built from turn events."""

from dataclasses import dataclass, replace

from data_analyst_agent.turn import CodeFinished, CodeSubmitted, TextDelta, TurnEvent


@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class CodePart:
    call_id: str
    code: str
    result: CodeFinished | None = None  # None while the code is running


Part = TextPart | CodePart


@dataclass(frozen=True)
class UserEntry:
    text: str


@dataclass(frozen=True)
class AssistantEntry:
    parts: tuple[Part, ...]
    error: str | None = None


Entry = UserEntry | AssistantEntry


def apply_event(parts: tuple[Part, ...], event: TurnEvent) -> tuple[Part, ...]:
    """Return the reply parts after one event. Events other than the three below are ignored."""
    if isinstance(event, TextDelta):
        if parts and isinstance(parts[-1], TextPart):
            return (*parts[:-1], TextPart(parts[-1].text + event.text))
        return (*parts, TextPart(event.text))
    if isinstance(event, CodeSubmitted):
        return (*parts, CodePart(call_id=event.call_id, code=event.code))
    if isinstance(event, CodeFinished):
        return tuple(
            replace(part, result=event)
            if isinstance(part, CodePart) and part.call_id == event.call_id
            else part
            for part in parts
        )
    return parts
