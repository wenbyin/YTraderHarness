"""子代理 / worker 的上下文构建（Processor 流水线实现）。

它是什么
--------
`ContextBuilderType.SUB_AGENT` 的**具体实现**：与主 Agent 的
`agent_invoke` 走同一套 13 阶段流水线、同一批 Processor，只换
`builder_type`。跑不跑某个阶段仍由 Processor 的 `when` gate 与
`context_policy` 决定，而不是在这里硬砍阶段 —— 以后给子任务开
画像 / 记忆 / 知识时不需要改这个类。

为什么不另写一套流水线
--------------------
阶段顺序、gate 校验、绑定解析、快照格式全部是「上下文怎么长出来」
的公共契约。子任务与主任务的区别在**输入与环境**，不在流水线：

    输入      current_task 来自任务板 claim 的那一条
    state     子图自己的 SubAgentState（不是父图 ReactState）
    回写      summary / memory / task_tool_keys 写回子图 state

因此它与 `AgentInvokeProcessorContextBuilder` 的差别只有 `builder_type`
（继承自 `SubAgentContextBuilder`）与身份元数据。

⚠️ 在哪个图形里用它
-------------------
    app/agent/graph/subagent/nodes.py 的 llm_node
        builder = runtime.context.subagent_builder
        builder_type = catalog.SUBAGENT_CONTEXT_BUILDER_TYPE

`create_context_builder(ContextBuilderType.SUB_AGENT)` 在**启动期**解析
一次（见 graph/react/runtime.py），节点里不查表 —— 与主图同一条约定。
"""

from __future__ import annotations

from typing import ClassVar

from app.common.context import StageId

from .agent_invoke import ProcessorContextBuilder
from .base import SubAgentContextBuilder

__all__ = ["SubAgentProcessorContextBuilder"]


#: 声明的阶段：全部 13 个。参与是静态事实，跑不跑由 gate 决定。
_ALL_STAGES: frozenset[StageId] = frozenset(StageId)

#: 不可被运行时跳过的阶段（理由见类 docstring）。
_REQUIRED_STAGES: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    StageId.PROMPT_ASSEMBLER,
})


class SubAgentProcessorContextBuilder(
    ProcessorContextBuilder,
    SubAgentContextBuilder,
):
    """
    子代理的上下文构建：同一套 13 阶段，任务 scope 的 state。

    ⚠️ required 阶段只有两个
    ----------------------
    CONTEXT_RESOLVER（入口：策略与预算）与 PROMPT_ASSEMBLER（出口：
    ModelRequest）。TASK_RESOLVER / SKILL_LOADER 带 gate，声明成
    required 会让流水线在构造期报「required 阶段上的 Processor 必须
    when=None」—— 见 processor/pipeline.py 的双向校验。

    ⚠️ 子图节点必须守卫 current_task
    -------------------------------
    没有 current_task 时 `task_mode=False`，task / skill 两个阶段会
    **静默跳过**，worker 会带着默认策略跑 —— 这比报错更难查。守卫在
    `graph/subagent/nodes.py::llm_node`，不在这里（Builder 只负责构建，
    不解释协作前提）。
    """

    version = "0.1.0"
    display_name = "子代理上下文构建（Processor 流水线）"
    description = (
        "按主 Agent 同一套 13 阶段流水线构建子任务上下文："
        "任务 / 技能 / 历史 / 工具 / 组装 / 预算 / 压缩 / 校验。"
        "画像与记忆默认由 ContextPolicy 关掉。"
    )

    enabled_stages: ClassVar[frozenset[StageId]] = _ALL_STAGES
    required_stages: ClassVar[frozenset[StageId]] = _REQUIRED_STAGES
