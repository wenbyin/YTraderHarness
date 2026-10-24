"""ask_user_question 的 interrupt 载荷 + resume 值 -> 可落库问答。"""

from __future__ import annotations

from app.agent.tools.hitl_tool import parse_interaction

_PAYLOAD = {
    "type": "ask_user_question",
    "questions": [
        {
            "question": "Q1?",
            "header": "h1",
            "options": [
                {"label": "A", "description": "a"},
                {"label": "B", "description": "b"},
            ],
            "multiSelect": False,
        },
        {
            "question": "Q2?",
            "header": "h2",
            "options": [
                {"label": "C", "description": "c"},
                {"label": "D", "description": "d"},
            ],
            "multiSelect": True,
        },
    ],
}


def test_structured_answers_shape() -> None:
    questions, answers, annotations = parse_interaction(
        _PAYLOAD,
        {
            "answers": {"Q1?": "A", "Q2?": ["C", "D"]},
            "annotations": {"Q1?": {"notes": "n"}},
        },
    )

    assert [q["question"] for q in questions] == ["Q1?", "Q2?"]
    assert answers == {"Q1?": "A", "Q2?": "C, D"}
    assert annotations == {"Q1?": {"notes": "n"}}


def test_index_keys_are_normalized_to_question_text() -> None:
    _, answers, _ = parse_interaction(
        _PAYLOAD,
        {"0": "B", "1": "C"},
    )

    assert answers == {"Q1?": "B", "Q2?": "C"}


def test_non_ask_payload_returns_empty() -> None:
    assert parse_interaction({"type": "approval"}, {}) == ([], {}, {})
