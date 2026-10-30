import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from data_analyst_agent.demo_script import DEMO_QUESTIONS, DEMO_SCRIPT
from data_analyst_agent.replay import ReplayChatModel, ScriptedQuestion, ScriptStep

SCRIPT = (
    ScriptedQuestion(
        question="How many rows?",
        steps=(ScriptStep("Counting rows.", "print(3)"), ScriptStep("There are 3 rows.")),
    ),
)


@tool("run_python")
def stand_in_tool(code: str) -> str:
    """Stand-in with the same name and arguments as the real tool."""
    return code


def bound_model(**fields) -> ReplayChatModel:
    return ReplayChatModel(script=SCRIPT, **fields).bind_tools([stand_in_tool])


def test_first_step_calls_the_tool_with_the_scripted_code():
    response = bound_model().invoke([HumanMessage("How many rows?")])

    assert response.content == "Counting rows."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "run_python"
    assert response.tool_calls[0]["args"] == {"code": "print(3)"}


def test_step_index_counts_model_responses_after_the_last_question():
    first = bound_model().invoke([HumanMessage("How many rows?")])
    tool_result = ToolMessage("3", tool_call_id=first.tool_calls[0]["id"])

    second = bound_model().invoke([HumanMessage("How many rows?"), first, tool_result])

    assert second.content == "There are 3 rows."
    assert second.tool_calls == []


def test_question_matching_ignores_case_spacing_and_end_punctuation():
    response = bound_model().invoke([HumanMessage("  how many   ROWS ")])

    assert response.content == "Counting rows."


def test_an_earlier_turn_does_not_shift_the_next_question():
    earlier = [HumanMessage("How many rows?"), AIMessage("a"), AIMessage("b")]

    response = bound_model().invoke([*earlier, HumanMessage("How many rows?")])

    assert response.content == "Counting rows."


def test_unknown_question_lists_the_scripted_questions():
    response = bound_model().invoke([HumanMessage("What is the weather?")])

    assert response.tool_calls == []
    assert "does not call a language model" in response.content
    assert "- How many rows?" in response.content


def test_tool_call_ids_are_new_on_every_call():
    ids = {
        bound_model().invoke([HumanMessage("How many rows?")]).tool_calls[0]["id"] for _ in range(5)
    }

    assert len(ids) == 5


def test_tool_step_requires_the_tool_to_be_bound():
    with pytest.raises(ValueError, match="not bound"):
        ReplayChatModel(script=SCRIPT).invoke([HumanMessage("How many rows?")])


def test_stream_yields_the_text_in_pieces_and_then_the_tool_call():
    chunks = list(bound_model().stream([HumanMessage("How many rows?")]))

    text_chunks = [c for c in chunks if c.content]
    assert len(text_chunks) == 2
    assert "".join(c.content for c in text_chunks) == "Counting rows."
    merged = chunks[0]
    for chunk in chunks[1:]:
        merged += chunk
    assert merged.tool_calls[0]["args"] == {"code": "print(3)"}
    assert json.loads(merged.tool_call_chunks[0]["args"]) == {"code": "print(3)"}


def test_demo_script_has_three_questions_that_each_end_with_an_answer():
    assert len(DEMO_SCRIPT) == 3
    assert tuple(q.question for q in DEMO_SCRIPT) == DEMO_QUESTIONS
    for question in DEMO_SCRIPT:
        assert question.steps[0].code is not None
        assert question.steps[-1].code is None
