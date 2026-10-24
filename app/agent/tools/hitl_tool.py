"""
HITL 工具：向用户主动提问。

谁用
----
规划阶段（planner / 计划模式下的主 agent）发现计划执行缺少信息或有逻辑
漏洞时调用：让用户从 1–4 个问题的选项里选，拿到答案后再继续规划。

中断发生在**工具函数体内**
------------------------
`interrupt()` 在工具里抛出 `GraphInterrupt`，LangGraph 把当前 Send 分支
挂起；run 进入 `waiting_input`，前端拿到 `hitl.requested` 的载荷。用户
回答后，dispatcher 用 `Command(resume={interrupt_id: data})` 恢复，同一
个工具调用从 `interrupt()` 处返回 `data`，继续执行。

⚠️ 执行器必须放行 GraphInterrupt
------------------------------
`ToolExecutor` 对工具异常有一层宽泛的 `except Exception`（把失败包成
ToolOutcome）。`GraphInterrupt` 是 Exception 的子类，如果不在那里显式
re-raise，中断会被吞掉，run 会以"工具失败"结束。见 tool_executor.py
的 `except GraphBubbleUp: raise`。

载荷 / 恢复值
-------------
发给前端（`interrupt()` 的参数）：

    {
      "type": "ask_user_question",
      "questions": [
        {"question": "...", "header": "...", "options": [
            {"label": "...", "description": "...", "preview": "..."}
        ], "multiSelect": false}
      ],
      "metadata": {"source": "planner"},
      "tool_call_id": "..."
    }

用户回答（resume 值）两种形状都认：

    {question_text: answer, ...}
    {"answers": {question_text: answer, ...},
     "annotations": {question_text: {"preview": "...", "notes": "..."}}}

`answers` 是**恢复时注入的**，不是模型参数 —— 模型的 JSON Schema 里
只有 `questions` 与 `metadata`。

返回值
------
    {
      "questions": [...],
      "answers": {question_text: answer},
      "annotations": {question_text: {"preview": ..., "notes": ...}}
    }

`multiSelect` 的答案若是列表，会拼成逗号分隔的字符串（契约里 answer
是 string）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

__all__ = [
    "ASK_USER_QUESTION_TOOL_KEY",
    "Question",
    "QuestionOption",
    "ask_user_question",
    "parse_interaction",
]


class QuestionOption(BaseModel):
    """一个选项。"""

    label: str = Field(
        min_length=1,
        description="选项的短标签，用户看到并选择的那一行。",
    )

    description: str = Field(
        min_length=1,
        description="这个选项意味着什么、有什么取舍。",
    )

    preview: str | None = Field(
        default=None,
        description="可选的预览内容（代码 / 配置 / 示例），帮助用户判断。",
    )


class Question(BaseModel):
    """一个问题。"""

    question: str = Field(
        min_length=1,
        description="要问用户的完整问题。",
    )

    header: str = Field(
        min_length=1,
        max_length=12,
        description="问题卡片上的短标题，最多 12 个字符。",
    )

    options: list[QuestionOption] = Field(
        min_length=2,
        max_length=4,
        description="2–4 个选项；不要给「其他」选项，用户可以用 Other 自由输入。",
    )

    multiSelect: bool = Field(
        default=False,
        description="是否允许多选。默认单选。",
    )


@tool
async def ask_user_question(
        questions: Annotated[
            list[Question],
            Field(min_length=1, max_length=4),
        ],
        metadata: dict[str, Any] | None = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> dict[str, Any]:
    """
    向用户提出 1–4 个带选项的问题，等待用户回答后继续。

    只用于**计划执行确实缺少信息**、或发现逻辑漏洞必须先澄清的场合；
    能从上下文推断的不要问。每个问题给 2–4 个具体选项，multiSelect
    按需要设置。

    工具会中断当前运行，直到用户回答；答案作为工具结果返回。
    """

    payload = {
        # 前端用 type 区分审批（approval）与提问（ask_user_question）。
        "type": "ask_user_question",
        "questions": [question.model_dump() for question in questions],
        "metadata": metadata or {},
        "tool_call_id": tool_call_id,
    }

    # ⚠️ 这里会挂起当前 Send 分支；恢复后返回值就是用户的答案。
    resumed = interrupt(payload)

    answers, annotations = _parse_resume(resumed, questions)

    return {
        "questions": [question.model_dump() for question in questions],
        "answers": answers,
        "annotations": annotations,
    }


def parse_interaction(
        payload: Mapping[str, Any],
        data: Any,
) -> tuple[
        list[dict[str, Any]],
        dict[str, str],
        dict[str, dict[str, str]],
]:
    """
    `ask_user_question` 的 interrupt 载荷 + resume 值 -> 可落库的问答。

    返回 `(questions, answers, annotations)`。questions 是模型当初给出的
    原始结构（已校验），answers/annotations 已按问题原文归一。

    用途：resume 路径把这次 HITL 问答写成 `kind=interaction` 的消息，
    让消息历史里能看到"问了什么、答了什么"。
    """

    raw_questions = payload.get("questions")

    if not isinstance(raw_questions, Sequence) or isinstance(
            raw_questions, (str, bytes)
    ):
        return [], {}, {}

    questions = [
        Question.model_validate(item)
        for item in raw_questions
        if isinstance(item, Mapping)
    ]

    answers, annotations = _parse_resume(data, questions)

    return (
        [question.model_dump() for question in questions],
        answers,
        annotations,
    )


def _parse_resume(
        raw: Any,
        questions: Sequence[Question],
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """
    用户回答 -> (answers, annotations)。

    兼容两种恢复值形状（见模块 docstring），并把 key 归一成**问题原文**：

        问题原文          -> 原样使用
        数字下标 "0"      -> 映射到 questions[0].question

    只保留本次实际问过的问题，避免前端多传的字段混进结果。
    """

    if not isinstance(raw, Mapping):
        return {}, {}

    if "answers" in raw:
        answers_raw = raw.get("answers")
        annotations_raw = raw.get("annotations")
    else:
        answers_raw = raw
        annotations_raw = None

    #: question 原文 / 下标 -> 规范化后的 key
    targets = _targets(questions)

    answers: dict[str, str] = {}

    if isinstance(answers_raw, Mapping):
        for key, value in answers_raw.items():
            target = targets.get(str(key))
            if target is not None:
                answers[target] = _answer_text(value)

    annotations: dict[str, dict[str, str]] = {}

    if isinstance(annotations_raw, Mapping):
        for key, value in annotations_raw.items():
            target = targets.get(str(key))
            if target is None or not isinstance(value, Mapping):
                continue
            annotations[target] = {
                name: str(item)
                for name, item in value.items()
                if name in {"preview", "notes"} and item is not None
            }

    return answers, annotations


def _targets(questions: Sequence[Question]) -> dict[str, str]:
    """问题原文与下标都映射到问题原文。"""

    result: dict[str, str] = {}

    for index, question in enumerate(questions):
        result[question.question] = question.question
        result[str(index)] = question.question

    return result


def _answer_text(value: Any) -> str:
    """答案 -> string（多选列表拼成逗号分隔）。"""

    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)

    return str(value)


#: tool_key 由扫描器推导（source_ref=agent，name=函数名）。
ASK_USER_QUESTION_TOOL_KEY = f"agent.{ask_user_question.name}"
