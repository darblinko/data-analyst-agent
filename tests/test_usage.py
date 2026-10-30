from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from data_analyst_agent.usage import TokenUsage, sum_usage


def ai_message(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        "",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def test_every_model_call_in_a_turn_is_counted():
    messages = [
        HumanMessage("question"),
        ai_message(100, 20),
        ToolMessage("output", tool_call_id="call_1"),
        ai_message(150, 30),
    ]

    usage = sum_usage(messages)

    assert usage == TokenUsage(input_tokens=250, output_tokens=50, reported_calls=2)
    assert usage.total_tokens == 300


def test_messages_without_usage_are_skipped():
    assert sum_usage([AIMessage("no usage"), HumanMessage("q")]) == TokenUsage()


def test_session_total_adds_turns():
    session = TokenUsage() + TokenUsage(10, 2, 1) + TokenUsage(5, 1, 1)

    assert session == TokenUsage(input_tokens=15, output_tokens=3, reported_calls=2)
