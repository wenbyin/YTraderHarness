"""
`classifier` 的 Processor 版实现：路由 / 分类用的**窄**上下文。

它只声明三个阶段
----------------
    CONTEXT_RESOLVER    （required）解析策略与预算
    SUMMARY_LOADER               注入 state 里已生成的对话摘要（只读）
    HISTORY_SELECTOR             选出当前这一轮
    PROMPT_ASSEMBLER    （required）产出 ModelRequest

`{CONTEXT_RESOLVER, PROMPT_ASSEMBLER}` 是 `BaseContextBuilder` 的硬要求
（入口 + 出口，任何实现都省不掉，见 builder/base.py 的
`__init_subclass__`）。第三个是本实现**必须**额外加的，理由见下。

⚠️ 为什么 HISTORY_SELECTOR 不能省（这一条会静默出错）
----------------------------------------------------
`ContextBuildContext.add_required_message()` 全仓**只有 history 处理器
调用**（processors/history.py 的 `_ensure_current_message`）。而
`DefaultContextPolicyResolver` 给 CLASSIFIER 的基线是

    include_history=False, min_recent_turns=0, max_recent_turns=0

history 被 gate 掉 -> `add_required_message` 永不执行 -> 组装出来的
消息列表里只有 system 提示词，**没有用户这句话**。

而且它不报错：`prompt_assembler` 只在"一条消息都没有"时才抛，有一条
system 就算"非空"。于是分类模型会对着一个没有问题的提示词做分类 ——
它照样返回一个结构化的、格式完全合法的答案，只是这个答案与用户说的话
无关。这类错误在观测里看不出来（模型没报错、格式合法）。

因此本实现保留 HISTORY_SELECTOR，并由调用方把
`include_history=True, min_recent_turns=1, max_recent_turns=1` 覆盖进
`context_policy` —— 窗口宽度是 `clamp(未覆盖轮数, min, 上限)`，两者都
钉成 1 的语义正是"只留当前这一轮，最后一轮永不裁"（history.py 的
注释），落点精确。

可跳过的阶段（本实现**不**声明）
--------------------------------
    PROFILE_LOADER / SUMMARY_LOADER / MEMORY_RETRIEVER /
    KNOWLEDGE_RETRIEVER / CAPABILITY_LOADER     分类只需要用户这一句话，
                                                画像 / 记忆 / 资料 / 工具
                                                都只是噪声。CLASSIFIER 的
                                                策略基线也已把这几个关掉。
    TOKEN_BUDGETER     窄上下文不会被预算裁掉；多一步重算没有收益。
    CONTEXT_COMPACTOR  超预算时压历史窗口与摘要的重叠。窄上下文（一轮、
                       零重叠）没有可压的东西；而它所在阶段的 gate 也
                       已被基线关掉（allow_compaction=False）。
    CONTEXT_VALIDATOR  校验的是"发给主模型的上下文能不能用"。窄分类调用
                       的输入由我们自己拼，这一层是给主调用兜底的。

⚠️ 为什么 SUMMARY_LOADER 要声明，但不能让它生成摘要
------------------------------------------------
分类需要**历史概览**来理解当前输入的指代（"那再查一下昨天的"），
但不需要为了分类先花一次模型调用去重写摘要。SummaryProcessor 的
生成分支受 `policy.allow_compaction` 门控；调用方（agent_builder_node
的 `_graph_state`）把 `allow_compaction=False` 钉死，同时打开
`include_summary=True`。

于是这个阶段只做 `_inject`：把 state 里**已经存在**的摘要作为
「历史摘要」system 分节注入，绝不调模型。没有摘要时它是 no-op。
如果只声明阶段却不钉 `allow_compaction`，主 agent 的 context_policy
可能带着 True，分类阶段就会意外再压一次历史 —— 延迟与成本都翻倍。

⚠️ 副作用：本实现的 ModelRequest 不带上下文快照
---------------------------------------------
`ProcessorContextBuilder._delegate` 用**模块级** `_FINAL_STAGE`
（= CONTEXT_VALIDATOR）判断何时把快照写进 metadata，而本实现不跑那个
阶段，因此 `metadata["context_snapshot"]` 不会出现。这是可接受的 ——
分类调用不需要"这一轮模型看到了什么"的观测快照，省下算哈希的开销。
要它的话得覆写 `_delegate`，不是加一个阶段了事。

与 agent_invoke 实现的关系
--------------------------
共用 `ProcessorContextBuilder`（阶段 -> 流水线的桥），只是声明的阶段集
不同。所以"两个 builder 的行为差异"只有下面这十来行，不是两套驱动。
"""

from __future__ import annotations

from typing import ClassVar

from app.common.context import StageId

from .agent_invoke import ProcessorContextBuilder
from .base import ClassifierContextBuilder

__all__ = ["ClassifierProcessorContextBuilder"]


#: 本实现声明参与的阶段。见模块 docstring 逐条理由。
_CLASSIFIER_STAGES: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    # 只读注入 state 里已有的摘要；生成分支由 allow_compaction=False
    # 关掉（见模块 docstring）。
    StageId.SUMMARY_LOADER,
    StageId.HISTORY_SELECTOR,
    StageId.PROMPT_ASSEMBLER,
})

#: 不可被运行时跳过的阶段 —— 入口与出口。
_CLASSIFIER_REQUIRED_STAGES: frozenset[StageId] = frozenset({
    StageId.CONTEXT_RESOLVER,
    StageId.PROMPT_ASSEMBLER,
})


class ClassifierProcessorContextBuilder(
        ProcessorContextBuilder,
        ClassifierContextBuilder,
):
    """
    路由 / 分类调用用的 Builder（`classifier`）。

    零参构造、进程一份，由 `create_context_builder(CLASSIFIER)` 解析
    （见 context/registry.py：按包扫描 + 反射发现，本类放在
    `app/agent/context/builder/` 包内就会自动被找到）。

    builder_type 由 `ClassifierContextBuilder` 继承而定，本类**不能**
    也不该自己声明。
    """

    version = "1.0.0"
    display_name = "分类 / 路由上下文构建（窄输入）"
    description = (
        "当前这一轮用户输入 + 已有历史摘要（只读）的最小上下文："
        "策略解析 -> 摘要注入 -> 当前轮 -> 组装。"
        "不含画像 / 记忆 / 资料 / 工具，也不做预算裁剪与新摘要生成。"
    )

    enabled_stages: ClassVar[frozenset[StageId]] = _CLASSIFIER_STAGES
    required_stages: ClassVar[frozenset[StageId]] = (
        _CLASSIFIER_REQUIRED_STAGES
    )
