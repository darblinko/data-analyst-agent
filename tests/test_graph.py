"""The full agent loop with the replay model, the real executor and the sample data."""

import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from data_analyst_agent.demo_script import DEMO_SCRIPT, SAMPLE_DATA_PATH
from data_analyst_agent.executor import ExecutionLimits
from data_analyst_agent.graph import AGENT_NODE, TOOLS_NODE, build_graph, route_after_agent
from data_analyst_agent.prompts import build_system_prompt
from data_analyst_agent.python_tool import make_python_tool
from data_analyst_agent.replay import ReplayChatModel, ScriptedQuestion, ScriptStep
from data_analyst_agent.turn import (
    CodeFinished,
    CodeSubmitted,
    ModelUsage,
    StepLimitReached,
    TextDelta,
    TurnFinished,
    run_turn,
    text_of,
)
from data_analyst_agent.usage import TokenUsage
from data_analyst_agent.workspace import Workspace

LIMITS = ExecutionLimits(timeout_s=60.0, max_output_bytes=20_000)
NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    workspace = Workspace(tmp_path)
    workspace.data_dir.mkdir()
    workspace.artifact_dir.mkdir()
    workspace.copy_in(SAMPLE_DATA_PATH)
    return workspace


def demo_graph(workspace: Workspace, script=DEMO_SCRIPT, **model_fields):
    model = ReplayChatModel(script=script, **model_fields)
    prompt = build_system_prompt(LIMITS, workspace.data_file_names())
    return build_graph(model, [make_python_tool(workspace, LIMITS)], prompt)


def collect(graph, question: str, history=(), max_model_calls: int = 10) -> list:
    return list(run_turn(graph, list(history), question, max_model_calls))


@pytest.mark.parametrize("scripted", DEMO_SCRIPT, ids=lambda q: q.question)
def test_demo_answers_quote_only_numbers_that_the_code_printed(workspace, scripted):
    events = collect(demo_graph(workspace), scripted.question)

    outputs = "\n".join(e.output for e in events if isinstance(e, CodeFinished))
    final_answer = scripted.steps[-1].text
    missing = [n for n in NUMBER.findall(final_answer) if n not in outputs]
    assert missing == []
    assert isinstance(events[-1], TurnFinished)
    assert text_of(events[-1].new_messages[-1].content) == final_answer


def test_demo_turn_event_sequence_and_saved_messages(workspace):
    overview = DEMO_SCRIPT[0]

    events = collect(demo_graph(workspace), overview.question)

    kinds = [type(e) for e in events if not isinstance(e, TextDelta)]
    assert kinds == [CodeSubmitted, CodeFinished, TurnFinished]
    submitted, finished, done = (e for e in events if not isinstance(e, TextDelta))
    assert submitted.code == overview.steps[0].code
    assert finished.call_id == submitted.call_id
    assert finished.exit_code == 0
    assert "rows: 8,772, columns: 6" in finished.output
    streamed = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert streamed == overview.steps[0].text + overview.steps[1].text
    assert [type(m) for m in done.new_messages] == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert done.usage == TokenUsage()


def test_chart_is_saved_to_the_workspace(workspace):
    events = collect(demo_graph(workspace), DEMO_SCRIPT[1].question)

    finished = next(e for e in events if isinstance(e, CodeFinished))
    assert finished.files == ("monthly_revenue_by_region.html",)
    chart = workspace.artifact_path("monthly_revenue_by_region.html")
    assert chart is not None
    assert "plotly" in chart.read_text()


def test_failed_code_is_reported_and_the_retry_succeeds(workspace):
    events = collect(demo_graph(workspace), DEMO_SCRIPT[2].question)

    first_try, retry = (e for e in events if isinstance(e, CodeFinished))
    assert first_try.exit_code == 1
    assert "KeyError: 'Revenue'" in first_try.output
    assert retry.exit_code == 0
    assert "+27.2%" in retry.output


def test_history_is_passed_in_and_not_returned_again(workspace):
    graph = demo_graph(workspace)
    first = collect(graph, DEMO_SCRIPT[0].question)[-1]

    second = collect(graph, DEMO_SCRIPT[1].question, history=first.new_messages)[-1]

    assert isinstance(second.new_messages[0], HumanMessage)
    assert second.new_messages[0].content == DEMO_SCRIPT[1].question
    assert len(second.new_messages) == 4


def test_step_limit_stops_a_turn_that_needs_more_model_calls(workspace):
    growth = DEMO_SCRIPT[2]  # three model calls
    graph = demo_graph(workspace)

    assert isinstance(collect(graph, growth.question, max_model_calls=3)[-1], TurnFinished)
    with pytest.raises(StepLimitReached, match="2 model calls"):
        collect(graph, growth.question, max_model_calls=2)


def test_usage_of_each_model_call_is_reported_before_a_step_limit_failure(workspace):
    graph = demo_graph(workspace, reported_usage=(10, 4))
    events = []

    with pytest.raises(StepLimitReached):
        for event in run_turn(graph, [], DEMO_SCRIPT[2].question, max_model_calls=2):
            events.append(event)

    reported = [e.usage for e in events if isinstance(e, ModelUsage)]
    assert reported == [TokenUsage(10, 4, 1), TokenUsage(10, 4, 1)]


def test_reported_usage_matches_the_turn_total(workspace):
    graph = demo_graph(workspace, reported_usage=(10, 4))

    events = collect(graph, DEMO_SCRIPT[2].question)

    reported = sum((e.usage for e in events if isinstance(e, ModelUsage)), TokenUsage())
    assert reported == events[-1].usage == TokenUsage(30, 12, 3)


def test_text_arrives_when_the_model_does_not_stream(workspace):
    script = (ScriptedQuestion("q", (ScriptStep("x", "print(2 + 2)"), ScriptStep("It is 4."))),)
    graph = demo_graph(workspace, script=script, disable_streaming=True)

    events = collect(graph, "q")

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["x", "It is 4."]
    assert next(e for e in events if isinstance(e, CodeFinished)).output == "4"


def test_route_after_agent():
    with_call = AIMessage("", tool_calls=[{"name": "run_python", "args": {}, "id": "c1"}])

    assert route_after_agent({"messages": [with_call]}) == TOOLS_NODE
    assert route_after_agent({"messages": [AIMessage("done")]}) != AGENT_NODE
    assert route_after_agent({"messages": [AIMessage("done")]}) == "__end__"


def test_text_of_reads_anthropic_content_blocks():
    blocks = [
        {"type": "thinking", "thinking": "hidden"},
        {"type": "text", "text": "Hello "},
        {"type": "tool_use", "id": "t1", "name": "run_python", "input": {}},
        {"type": "text", "text": "world"},
    ]

    assert text_of(blocks) == "Hello world"
    assert text_of("plain") == "plain"


def test_system_prompt_lists_files_and_limits():
    prompt = build_system_prompt(ExecutionLimits(timeout_s=12.5, max_output_bytes=4_096), ["a.csv"])

    assert "- data/a.csv" in prompt
    assert "12.5 seconds" in prompt
    assert "4,096 bytes" in prompt
    assert "No data files are loaded" in build_system_prompt(LIMITS, [])
