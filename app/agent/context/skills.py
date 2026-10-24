"""统一的 Skill 结构：渲染 prompt、携带 capabilities。

为什么单独一个结构
------------------
task / skill 相关的渲染会出现在多个阶段（TaskResolver、SkillLoader、
将来的 SubAgent）。如果每处自己从 `Task.metadata` / `SkillDetail` 里拼
字符串，prompt 的样子会漂成两份。

这里只放**数据 + 渲染**，不查库：
    SkillSnapshot       一次执行用的技能快照（instructions + capabilities）
    render_prompt()     把快照渲染成 system prompt
    skill_key/caps      从 A2A Task dict 里取调度字段（容错）

真正的 SkillService 查询在 `app/infra/agent/skill_exposure.py`。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

__all__ = [
    "TASK_EXECUTION_FALLBACK_PROMPT",
    "SkillCapability",
    "SkillSnapshot",
    "capabilities_of",
    "skill_key_of",
    "task_payload_of",
]


#: 没有 skill / skill 加载失败时的任务执行提示词。
#:
#: 它只负责“切换到任务执行模式”，不假装自己懂业务；真正的业务说明在
#: skill.instructions 里。
TASK_EXECUTION_FALLBACK_PROMPT = (
    "你正在执行一个由 planner 拆出的任务。\n"
    "根据下面的任务说明完成工作，输出任务结果；不要回答用户输入。\n"
    "任务所需工具由本次上下文提供；没有提供的工具不要假设它存在。"
)


@dataclass(frozen=True, slots=True)
class SkillCapability:
    """skill 绑定的一项能力。"""

    code: str
    name: str = ""


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    """一次任务执行用的 skill 快照。

    使用快照而不是每次现查：审批人看到的能力集与执行时加载的必须一致。
    """

    skill_key: str
    name: str
    description: str = ""
    instructions: str = ""
    capabilities: tuple[SkillCapability, ...] = ()
    updated_at: datetime | None = None

    def render_prompt(self) -> str:
        """渲染成 system prompt（instructions + 能力清单）。"""

        parts: list[str] = []

        title = self.name or self.skill_key
        parts.append(f"# 技能：{title}（{self.skill_key}）")

        if self.description:
            parts.append(self.description.strip())

        if self.instructions.strip():
            parts.append(self.instructions.strip())

        if self.capabilities:
            lines = [
                f"- {item.code}：{item.name}" if item.name
                else f"- {item.code}"
                for item in self.capabilities
            ]
            parts.append("## 本技能可用的能力\n" + "\n".join(lines))

        return "\n\n".join(part for part in parts if part.strip())

    def as_state(self) -> dict[str, Any]:
        """观测 / 快照用的 JSON 安全形状。"""

        return {
            "skill_key": self.skill_key,
            "name": self.name,
            "description": self.description,
            "capabilities": [item.code for item in self.capabilities],
            "updated_at": (
                self.updated_at.isoformat()
                if self.updated_at is not None
                else None
            ),
        }


# ============================================================
# 从 A2A Task dict 取调度字段
# ============================================================


def _metadata(task: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(task, Mapping):
        return {}

    metadata = task.get("metadata")

    return metadata if isinstance(metadata, Mapping) else {}


def skill_key_of(task: Mapping[str, Any] | None) -> str:
    """从 `current_task.metadata.skill_key` 取技能 key；没有就空串。"""

    return str(_metadata(task).get("skill_key") or "").strip()


def capabilities_of(task: Mapping[str, Any] | None) -> list[str]:
    """从 `current_task.metadata.capabilities` 取 planner 授予的 code。"""

    raw = _metadata(task).get("capabilities")

    if isinstance(raw, str):
        return [raw] if raw.strip() else []

    if isinstance(raw, (list, tuple)):
        return [
            str(item).strip()
            for item in raw
            if isinstance(item, str) and item.strip()
        ]

    return []


def task_payload_of(task: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """取 A2A Task.history[0].parts[0].data（planner 的 message）。"""

    if not isinstance(task, Mapping):
        return {}

    history = task.get("history")

    if not isinstance(history, (list, tuple)):
        return {}

    for message in history:

        if not isinstance(message, Mapping):
            continue

        parts = message.get("parts")

        if not isinstance(parts, (list, tuple)):
            continue

        for part in parts:

            if not isinstance(part, Mapping):
                continue

            data = part.get("data")

            if isinstance(data, Mapping):
                return data

    return {}
