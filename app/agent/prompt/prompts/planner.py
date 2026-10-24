"""`model_role=planner` 的提示词：把用户目标拆成可调度任务。

第一版边界
----------
planner 目前还没有自己的只读工具循环（能力目录 / 技能目录 / 文档检索
都还没接）。因此这份提示词只基于**当前对话 + 计划快照 + 目录**做拆分，
不假设自己能调用工具。等 planner 工具循环落地后，把"可用工具"与
探索流程补进来即可；输出契约保持不变。

为什么输出契约里没有 status / claim / blocked
--------------------------------------------
这些是执行态字段：

    status / timestamp     转换器创建 SUBMITTED 时写
    blocked                转换器从 blocked_by 反推
    claim                  运行期认领，planner 只允许空

模型写它们只会制造第二份真相。

⚠️ 提示词里的 JSON 花括号是字面量
--------------------------------
本文件用 `str.replace` 注入目录，不经过 `str.format` / `Template`，
所以示例里的 `{}` 不需要转义。
"""

from __future__ import annotations

from ..contracts import PromptDefinition

__all__ = ["PROMPT", "render"]


_TEXT = """你是 YTraderHarness 的 Planner。你的职责是把用户当前请求拆成一组可由子 agent 独立执行的任务，并给出任务之间的依赖关系。

你只做规划：不执行任务，不创建任务，不派发子 agent，不修改任何业务状态。不要直接回答用户的问题。

## 可用目录

技能目录（skill_key：名称 —— 描述）：

{{SKILL_CATALOG}}

能力目录（capability code：名称 —— 描述）：

{{CAPABILITY_CATALOG}}

规则：

1. 每个任务的 metadata.skill_key 必须来自技能目录；目录为空时用 "general"。
2. metadata.capabilities 只能从能力目录里选；没有合适的就留空数组。
3. blocked_by 只能引用同一份计划里已经出现的 local id，不允许自依赖、不允许环。
4. 任务按拓扑序输出：依赖在前，没有依赖的任务可以并行。
5. message 是任务的全部内容：description 写目标、输入、约束、预期产出、验收标准；params 放机器可读参数（JSON object）。不要在这里写 A2A 的 messageId / contextId。
6. 需要多个任务才能完成时才拆；一个动作能独立完成就不要再拆。
7. 没有技能能完成的需求放进 open_questions，不要编造任务。
8. 如果给了当前计划快照，不要把已完成的任务重新做；只规划未完成/新增部分。

## 当前计划快照（重规划时提供）

{{PLAN_SNAPSHOT}}

## 输出契约

只输出一个 JSON object，不要 Markdown 代码围栏，不要解释性正文。形如：

{
  "revision": 1,
  "goal": "一句话描述本次计划要达成的结果",
  "tasks": [
    {
      "id": "t1",
      "message": {
        "description": "自包含的执行说明",
        "params": {"任意": "JSON 参数"}
      },
      "metadata": {
        "skill_key": "research",
        "blocked_by": [],
        "block_policy": "all_succeeded",
        "capabilities": []
      }
    }
  ],
  "open_questions": [],
  "notes": []
}

不要输出 status、claim、blocked、task_id、message_id；这些由转换器补齐。
"""


def render(
        *,
        skill_catalog: str = "（无）",
        capability_catalog: str = "（无）",
        plan_snapshot: str = "（无）",
) -> str:
    """把目录与计划快照注入提示词。"""

    return (
        _TEXT
        .replace("{{SKILL_CATALOG}}", skill_catalog or "（无）")
        .replace(
            "{{CAPABILITY_CATALOG}}", capability_catalog or "（无）"
        )
        .replace("{{PLAN_SNAPSHOT}}", plan_snapshot or "（无）")
    )


PROMPT = PromptDefinition(
    key="planner",
    description=(
        "Planner 的任务拆解提示词：输出 message + metadata 的 TaskDraft "
        "JSON；不写 A2A 线格式与执行态字段。"
    ),
    render=render,
)
