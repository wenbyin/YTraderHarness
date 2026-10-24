"""任务指令与任务决策提示的**唯一**渲染入口。

四个地方用同一段文本：

    TaskResolver   注入 system section（短头）与 required HumanMessage
    task_listener  领取任务时把同一段 HumanMessage 写进 messages（落库）
    team decision  main agent 决定「自己处理 / 开子代理」时的当前输入
    未来的 sub-agent 也可以复用

如果各拼一份，prompt 里看到的任务与 `agent_message` 里落库的任务迟早
会漂。这里只做数据 -> 文本 / HumanMessage，不查库。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from app.agent.context.skills import (
    capabilities_of,
    skill_key_of,
    task_payload_of,
)

__all__ = [
    "last_ai_text",
    "render_task_header",
    "render_task_instruction",
    "render_task_results",
    "task_instruction_message",
    "task_results_message",
    "team_decision_message",
]

#: system 短头里任务内容的截断长度；完整内容在 required HumanMessage。
_HEADER_DESCRIPTION_LIMIT = 200


def render_task_header(
        task: Mapping[str, Any],
        *,
        owner: str | None = None,
        position: int = 0,
        total: int = 0,
) -> str:
    """
    system 里的短任务头；完整数据在 HumanMessage。

    `position` / `total` 是当前任务在任务列表里的位置（从 1 开始）与总数。
    为 0 时不输出这一行（state 里没有完整任务列表的旧运行）。
    """

    task_id = str(task.get("id") or "").strip()
    skill_key = skill_key_of(task)
    capabilities = capabilities_of(task)
    description = _description(task)

    lines = [
        "你当前处于任务执行模式。",
        f"负责人：{_owner_label(owner)}（你）",
        f"任务 ID：{task_id or '（未命名）'}",
    ]

    progress = _progress(position, total)

    if progress:
        lines.append(progress)

    if skill_key:
        lines.append(f"技能：{skill_key}")

    if capabilities:
        lines.append("授予的能力：" + "、".join(capabilities))

    if description:
        lines.extend([
            "",
            "任务内容：",
            _short(description, _HEADER_DESCRIPTION_LIMIT),
        ])

    lines.append("完整任务说明见当前输入；不要回答用户输入。")

    return "\n".join(lines)


def render_task_instruction(
        task: Mapping[str, Any],
        *,
        owner: str | None = None,
        position: int = 0,
        total: int = 0,
) -> str:
    """把 A2A task 的 message payload 渲染成可执行的任务指令。"""

    task_id = str(task.get("id") or "").strip()
    skill_key = skill_key_of(task)
    capabilities = capabilities_of(task)
    payload = task_payload_of(task)

    description = str(payload.get("description") or "").strip()
    params = payload.get("params")

    lines: list[str] = [
        "你当前处于任务执行模式。",
        "请完成下面的任务，产出任务结果；**不要**回答用户输入。",
        "",
        f"负责人：{_owner_label(owner)}（你）",
        f"任务 ID：{task_id or '（未命名）'}",
    ]

    progress = _progress(position, total)

    if progress:
        lines.append(progress)

    if skill_key:
        lines.append(f"技能：{skill_key}")

    if capabilities:
        lines.append("授予的能力：" + "、".join(capabilities))

    if description:
        lines.extend(["", "任务说明：", description])

    if params not in (None, {}, []):

        try:
            rendered = json.dumps(
                params, ensure_ascii=False, indent=2,
            )
        except (TypeError, ValueError):
            rendered = str(params)

        lines.extend(["", "任务参数：", "```json", rendered, "```"])

    lines.extend([
        "",
        "输出要求：给出可被父图汇总的任务结果；不要重复用户问题，"
        "不要把任务说明当作对用户的答复。",
    ])

    return "\n".join(lines)


def render_task_results(
        tasks: Sequence[Mapping[str, Any]],
        plan: Mapping[str, Any] | None = None,
) -> str:
    """把完成的任务结果渲染成 main agent 汇总回合的正文。"""

    lines: list[str] = ["## 任务结果"]

    for task in tasks:

        task_id = str(task.get("id") or "").strip() or "（未命名任务）"
        skill_key = skill_key_of(task) or "未指定"
        status = _task_status(task)

        lines.append(
            f"### {task_id}（skill={skill_key}，status={status}）"
        )
        lines.append(_result_text(task) or "（该任务没有产出结果）")

    open_questions = (
        plan.get("open_questions")
        if isinstance(plan, Mapping)
        else None
    )

    if isinstance(open_questions, Sequence) and not isinstance(
        open_questions, (str, bytes)
    ):

        questions = [
            str(item).strip() for item in open_questions
            if str(item).strip()
        ]

        if questions:
            lines.append("## 计划中的待确认项")
            lines.extend(f"- {item}" for item in questions)

    return "\n\n".join(lines)


def task_results_message(
        tasks: Sequence[Mapping[str, Any]],
        plan: Mapping[str, Any] | None = None,
) -> HumanMessage:
    """汇总回合的 required HumanMessage。"""

    return HumanMessage(
        content=(
            "所有任务已完成。以下是各任务的结果：\n\n"
            + render_task_results(tasks, plan)
            + "\n\n请基于以上结果直接回答用户，按用户要求的结构输出；"
            "不要重新发布任务，不要声称结果稍后给出。"
            "若有任务失败或被取消，请说明失败项与原因。"
        ),
        additional_kwargs={"kind": "task_results"},
    )


def team_decision_message(
        *,
        plan: Mapping[str, Any],
        tasks: list[Mapping[str, Any]],
        pending_count: int,
        ready_count: int,
        can_create_team: bool,
) -> HumanMessage:
    """
    没有 current_task、但任务列表还有待办时，交给 main agent 的决策输入。

    它不是用户说的话，而是一条内部 HumanMessage。id 按
    `(plan_id, revision)` 稳定；`kind=team_decision` 让落库层把它当作
    内部消息过滤掉（见 message_recorder.new_assistant_messages）。
    """

    plan_id = str(plan.get("plan_id") or "").strip()
    revision = _revision(plan)

    lines: list[str] = [
        "当前没有正在执行的任务，但计划里还有待办任务。",
        f"任务列表：总计 {len(tasks)}，待处理 {pending_count}，"
        f"可并行 {ready_count}。",
        "",
    ]

    if ready_count >= 2 and can_create_team:

        lines.extend([
            "你可以选择：",
            "1. 自己处理：直接结束本轮，系统会按顺序 claim 下一个任务；",
            f"2. 并行处理：调用 create_team(worker_num={ready_count}) "
            "开启子代理。",
            "",
            "本轮只做选择，不要执行任务内容，也不要把计划当成对用户的答复。",
        ])

    elif ready_count >= 2:

        lines.extend([
            "存在可并行任务，但当前没有开启子代理的工具。",
            "请直接结束本轮，系统会按顺序 claim 下一个任务；"
            "本轮不要执行任务内容。",
        ])

    else:

        lines.extend([
            "当前没有可并行任务。请直接结束本轮，系统会按顺序 claim "
            "下一个任务；本轮不要执行任务内容。",
        ])

    return HumanMessage(
        content="\n".join(lines),
        id=f"team:{plan_id}:{revision}" if plan_id else "team:decision",
        additional_kwargs={
            "kind": "team_decision",
            "plan_id": plan_id,
            "revision": revision,
            "pending_count": pending_count,
            "ready_count": ready_count,
        },
    )


def task_instruction_message(
        task: Mapping[str, Any],
        *,
        owner: str | None = None,
        position: int = 0,
        total: int = 0,
) -> HumanMessage:
    """
    任务指令的 HumanMessage。

    id 稳定（同一 task 重跑不会在 messages 里堆多份）；`kind` 供
    agent_message 落库使用，与 `MessageKind.TASK_INSTRUCTION` 对齐。
    """

    task_id = str(task.get("id") or "").strip()
    skills = str(skill_key_of(task) or "")
    metadata = task.get("metadata")
    plan_id = ""

    if isinstance(metadata, Mapping):
        plan_id = str(metadata.get("plan_id") or "")

    return HumanMessage(
        content=render_task_instruction(
            task,
            owner=owner,
            position=position,
            total=total,
        ),
        id=(f"task:{task_id}:instruction" if task_id else None),
        additional_kwargs={
            "kind": "task_instruction",
            "task_id": task_id,
            "skill_key": skills,
            "plan_id": plan_id,
        },
    )


def last_ai_text(
        messages: Sequence[AnyMessage],
        *,
        limit: int | None = None,
) -> str:
    """
    最后一条 AIMessage 的正文。

    主图 listener 与子图 listener 共用：任务完成时 `summary` 取截断版
    （进 metadata 预览），`result_text` 取全文（进 A2A artifact）。
    `limit=None` 返回完整正文。

    兼容 content 为字符串与 content block 列表两种形状。
    """

    for message in reversed(list(messages or ())):

        if not isinstance(message, AIMessage):
            continue

        content = message.content

        if isinstance(content, str):
            text = content

        elif isinstance(content, (list, tuple)):
            text = "".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, Mapping)
            )

        else:
            text = str(content)

        text = text.strip()

        return text if limit is None else text[:limit]

    return ""


# ============================================================
# internals
# ============================================================


def _owner_label(owner: str | None) -> str:
    """
    claim owner -> 给人看的短名。

        run-1:main -> main
        空         -> main agent
    """

    text = str(owner or "").strip()

    if not text:
        return "main agent"

    return text.rsplit(":", 1)[-1] or text


def _progress(position: int, total: int) -> str:
    if position <= 0 or total <= 0:
        return ""

    return f"任务列表：总计 {total}，当前第 {position} 个"


#: 单个任务结果进汇总上下文的截断长度。
_RESULT_LIMIT = 4000


def _task_status(task: Mapping[str, Any]) -> str:
    status = task.get("status")

    if isinstance(status, Mapping):
        return str(status.get("state") or "UNKNOWN")

    return str(status or "UNKNOWN")


def _result_text(task: Mapping[str, Any]) -> str:
    """优先用 result_summary；没有就从 artifact 的 text part 取。"""

    metadata = task.get("metadata")

    if isinstance(metadata, Mapping):

        summary = metadata.get("result_summary")

        if isinstance(summary, str) and summary.strip():
            return _short(summary.strip(), _RESULT_LIMIT)

    artifacts = task.get("artifacts")

    if isinstance(artifacts, Sequence) and not isinstance(
        artifacts, (str, bytes)
    ):

        for artifact in artifacts:

            if not isinstance(artifact, Mapping):
                continue

            for part in artifact.get("parts") or ():

                if not isinstance(part, Mapping):
                    continue

                text = part.get("text")

                if isinstance(text, str) and text.strip():
                    return _short(text.strip(), _RESULT_LIMIT)

    return ""


def _description(task: Mapping[str, Any]) -> str:
    payload = task_payload_of(task)

    return str(payload.get("description") or "").strip()


def _short(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text

    return text[:limit] + "…"


def _revision(plan: Mapping[str, Any]) -> int:
    try:
        return int(plan.get("revision") or 0)
    except (TypeError, ValueError):
        return 0
