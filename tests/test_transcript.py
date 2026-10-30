from data_analyst_agent.transcript import CodePart, TextPart, apply_event
from data_analyst_agent.turn import CodeFinished, CodeSubmitted, TextDelta, TurnFinished
from data_analyst_agent.usage import TokenUsage


def finished(call_id: str, output: str) -> CodeFinished:
    return CodeFinished(call_id=call_id, output=output, files=(), exit_code=0, timed_out=False)


def test_text_deltas_are_joined_until_code_starts():
    events = [
        TextDelta("I will "),
        TextDelta("count."),
        CodeSubmitted("c1", "print(1)"),
        finished("c1", "1"),
        TextDelta("There is "),
        TextDelta("1."),
        TurnFinished(new_messages=(), usage=TokenUsage()),
    ]

    parts = ()
    for event in events:
        parts = apply_event(parts, event)

    assert parts == (
        TextPart("I will count."),
        CodePart("c1", "print(1)", finished("c1", "1")),
        TextPart("There is 1."),
    )


def test_result_is_attached_to_the_matching_call_only():
    parts = (CodePart("c1", "a"), CodePart("c2", "b"))

    updated = apply_event(parts, finished("c2", "out"))

    assert updated[0].result is None
    assert updated[1].result == finished("c2", "out")
    assert parts[1].result is None
