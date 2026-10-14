"""
阶段 5：MEMORY_RETRIEVER —— 注入**召回**出来的长期记忆（规格 §二十九 第 7 条）。

本处理器只管两层
-------------
    L0 原话      `agent_message` 的混合召回（片段）
    L1 原子记忆   `memory_atom` 的混合召回（完整陈述）

这两层是**贵**的那两层（embedder + 向量检索 + 融合），所以"这一轮要不要
召回"这个决策只对它们有意义 —— 判它的是 Agent Builder 那个语义识别节点
（分类提示词里写着四条触发、两条排除），结果经
`context_policy.memory_retrieve` 传到这里，与 `tool_keys` 走的是同一条路
（那个节点是 policy 的唯一写者，本处理器按声明式绑定读）。

⚠️ 另外两层在 `profile.py`，而且是**每轮现读**
--------------------------------------
        L3 画像 + L2 场景索引    profile.py 的 `MemoryProfileProcessor`

它们每轮都进上下文（画像一句话、场景清单十几行），而且**便宜**（纯 SQL：
一篇画像 + 一次场景列举），所以那边不做保活、每次构建现读 —— 见
`react/state.py` 里 `memory` 通道那段说明（那个通道存在的唯一理由就是
"L0/L1 贵"）。

拆开还有一个直接好处：L0/L1 的召回**需要查询串**（本处理器在拿不到它就
什么都不注入），而画像与场景索引与"这一轮问了什么"无关 —— 合成一个处理器
时，一句没有用户消息的轮次会把那两段也一起丢掉。

⚠️ 一次运行只召回一次
------------------
主循环每经过一次工具结果就重建一次上下文（`agent_node` 每轮都跑
ContextBuilder），所以"每次构建都召回一遍"等于每过一个工具往返就付一次
embedder + 向量检索。而 system 每次都是重建的 —— 只看 INITIAL 那一次注入
的话，模型在第二次迭代就看不到那两段了（那正是 sources.py 里 §四B 那条
红线担心的一类不一致）。

因此分两步走：

    INITIAL        真召回，注入两段，并把**渲染好的分节**交给图 state 保活
    CONTINUATION   不碰服务，直接从 state 读回来重新注入（`_replay`）

⚠️ 名字里有 memory 的那条 state 通道
-------------------------------
`state["memory"]` 装的就是上面那份保活数据（见 react/state.py 的字段
说明）。它**不是**"每轮都带的记忆"—— 只有本处理器召回的两层在里面，
而它们一个运行只召回一次。

回写走 `ModelRequest.metadata["context_state_updates"]`（本处理器写进
`context.metadata` 上的一个槽，由 `ProcessorContextBuilder._delegate` 汇总、
`agent_node` 按白名单落到图 state），链路见 agent_node 的 `_STATE_UPDATE_KEYS`。

⚠️ `optional=True` —— 降级分两层
----------------------------
规格 §二十八 把"Memory 不可用"列为 Optional：记忆是**旁路信息**，
拿不到不该让整轮对话失败。失败时流水线记 FAILED + 一条 warning 继续。

分工是两层，别混：

    层内失败          `retrieval_preview` 自己收进 `degraded`
                      （某一路召回挂了、其余照常 —— 见
                      memory_query_service_impl.py 的 `_recall_l0` /
                      `_recall_l1`）。这一层**不抛**，所以"L1 挂了但
                      L0/L3 还在"成立。
    连服务都取不到    本模块**不兜**，让它抛 -> 流水线按 optional 记
                      FAILED + 一条 warning，然后照常跑完这一轮。

第二种自己 catch 掉会丢掉 FAILED 与那条 warning，于是"记忆整段不可用"
在快照里变得看不见 —— 只剩一行日志，而快照才是排查时看的地方。

数据从哪来
---------
直接走服务：`get_service_factory().scope(MemoryQueryService, begin=False)`
-> `retrieval_preview(...)`。与 `POST /memory/retrieval` 那条路**同一条
实现**，所以原规格那句"预览里看到的就是 Agent 能拿到的"仍然成立。

⚠️ 不再有 `deps.memory_source` 分支
-------------------------------
`ContextDependencies.memory_source` 那个端口今天全仓没有实现，而且它的形状
装不下这里要的东西（没有会话维度、没有分层、没有"要不要召回"、没有分路
降级）。留着它比删掉更糟：哪天有人配上一个源，CONTINUATION 就会去再调一次
源，把上面那套保活直接架空。将来真要接第二个记忆后端，它应该是
`MemoryQueryService` 的另一个实现，而不是这里的一条 if。

⚠️ `get_service_factory()` 是**进程级取用**
----------------------------------------
依赖束（`ContextDependencies`）里没有"服务工厂"这一格，而 Processor 不能
自造服务。这里与 `agent_builder_node` 取 `CapabilityService` 是同一条手法
（那边也直接 `get_service_factory()`）。它与"Processor 不从全局摸东西"
那条纪律有一处刻意的不一致，记在这里而不是假装没有。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.common.context import StageId
from app.common.factories import get_service_factory
from app.services.memory.application import MemoryQueryService
from app.services.memory.domain.schemas.query import (
    MemoryPreviewLayers,
    MemoryPreviewLimits,
    MemoryRetrievalRequest,
    RetrievalTarget,
)

from ..base import ContextProcessor
from ..context import BuildPhase, ContextBuildContext
from ..messages import message_text
from ..spec import ContextProcessorSpec, StateBinding
from ._blocks import blocks_from, inject, mapping

__all__ = ["MemoryProcessor"]

#: L0 / L1 每层召回的条数。
#:
#: 比 `MemoryPreviewLimits` 的默认值（10）小：进主模型上下文的记忆段每一轮
#: 都占预算，而管理端预览可以看宽一点。两处口径不同是有意的 —— 要看全貌，
#: 去管理端把 L0/L1 勾开。
_LIMIT = 5


class MemoryProcessor(ContextProcessor):
    """把召回出来的 L0/L1 注入成两段 system 片段。"""

    spec = ContextProcessorSpec(
        name="memory",
        order=50,
        stage=StageId.MEMORY_RETRIEVER,
        when="context_policy.include_memory",
        inputs={
            # 召回指令：搜哪几层 + 以谁的身份搜。Agent Builder 写。
            #
            # ⚠️ required=False：builder 节点在"没有用户消息"那条早返回里
            # 根本不写 context_policy（见 agent_builder_node 的
            # `_memory_retrieve` 调用点），所以这一格**合法地不存在** ——
            # 缺了就当作"不召回"。
            "retrieve": StateBinding(
                "context_policy.memory_retrieve", required=False
            ),
            # INITIAL / CONTINUATION：决定这一轮要不要真去召回。
            # history 处理器也绑它（同一条字段，另一种用法：
            # "后续轮次不要重新从库里加载历史"）。
            "phase": StateBinding("run.phase"),
            # 身份。缺任一个就不召回（见 process 里那段说明）。
            "user_id": StateBinding("user_id", required=False),
            # 召回用的查询串 —— pipeline 从最近一条用户消息派生
            # （见 processor/pipeline.py 的 `_current_user_message`）。
            # **原话**，不做关键词化：检索实现吃的就是查询串本身。
            "query": StateBinding("current_user_message", required=False),
            # 上一次 INITIAL 存下来的分节（保活，见模块 docstring）。
            "persisted": StateBinding("memory", required=False),
        },
        optional=True,
        description=(
            "注入召回出来的两层记忆：相关原话（L0）/ 相关记忆（L1）。"
        ),
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        retrieve = mapping(inputs.get("retrieve"))

        agent_id = str(retrieve.get("agent_id") or "").strip()

        user_id = str(inputs.get("user_id") or "").strip()

        if not agent_id or not user_id:
            # 身份不全 = 不知道"以谁的身份搜"，也就没有记忆可注入。
            # 这不是错误：匿名 / 系统调用 / 存量运行（user_id 为 NULL 的
            # 那批）都会走到这里。
            #
            # `agent_id`（= duty_key）那一半还额外表达了一件事：
            # `_memory_retrieve` 在 duty_key 为空时干脆不给决策，所以这一格
            # 同时是"上游认为这次不该召回"。
            return context

        if str(inputs.get("phase") or "") == BuildPhase.INITIAL.value:
            return await self._recall(
                inputs, context,
                retrieve=retrieve, agent_id=agent_id, user_id=user_id,
            )

        return self._replay(inputs, context)

    # --------------------------------------------------------

    async def _recall(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
            *,
            retrieve: Mapping[str, Any],
            agent_id: str,
            user_id: str,
    ) -> ContextBuildContext:
        """INITIAL：真召回 -> 注入两段 -> 把分节交给图 state 保活。"""

        preview = await self._preview(
            inputs,
            retrieve=retrieve,
            agent_id=agent_id,
            user_id=user_id,
        )

        if preview is None:
            return context

        blocks = blocks_from(preview)

        if not blocks:
            return context

        inject(context, blocks)

        # 交给图 state 保活（白名单里有 memory，见 agent_node）。
        #
        # ⚠️ **只在 INITIAL 写**：CONTINUATION 再写就违反 state.py 那条
        # "节点不要每轮都重复写同一个值"（channel 会看起来一直在变）。
        #
        # 放进 `context.metadata` 而不是 `draft.metadata`：本处理器在阶段 5
        # 跑，而 draft 要到阶段 8（PROMPT_ASSEMBLER）才存在 —— `_delegate`
        # 在 `draft is None` 时直接 return，那时根本没有 draft 可写。
        updates = context.metadata.setdefault("context_state_updates", {})

        if isinstance(updates, dict):
            updates["memory"] = {
                "blocks": blocks,
                "degraded": _degraded(preview),
            }

        return context

    def _replay(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:
        """
        CONTINUATION：不碰服务，用上一次召回的结果重新注入。

        为什么不是"跳过注入"：system 每次都重建，跳过等于模型在第 2 次
        迭代就看不到刚刚还在的那两段 —— 上下文中途变形，而它可能正要
        引用其中一句。
        """

        blocks = _persisted(inputs.get("persisted"))

        if blocks:
            inject(context, blocks)

        return context

    async def _preview(
            self,
            inputs: dict[str, Any],
            *,
            retrieve: Mapping[str, Any],
            agent_id: str,
            user_id: str,
    ) -> Any | None:
        """
        跑一轮预览。

        ⚠️ 这里**不兜异常** —— 取不到服务就让它抛
        ------------------------------------
        降级由 `spec.optional=True` 负责：流水线会把这一阶段记成 FAILED +
        一条 warning，然后**照常跑完**这一轮（规格 §二十八：记忆是旁路
        信息）。在这里自己 catch 掉会丢掉那两条记录，于是"记忆整段不可用"
        在快照里变得看不见 —— 只剩一行日志，而快照是排查时真正看的地方。

        （`agent_builder_node` 里 `get_service_factory()` 那个 RuntimeError
        是被 catch 的，但那里不同：分类不是 optional 阶段，抛出去会中断
        整轮对话。）

        **层**内的失败不经过这里：`retrieval_preview` 自己把它们收进
        `degraded`（某一层挂了，其余照常）。所以能抛到这一层的只有
        "连服务都取不到"，那正是该被记成 FAILED 的事。
        """

        query = message_text(inputs.get("query"))

        if not query or not query.strip():
            # 没有查询串就构造不出请求（契约里 `input` 是 min_length=1），
            # 而"没有可问的东西"本来也不该去查记忆。**这不是失败** ——
            # 所以它返回 None（= 什么都不注入），而不是抛。
            return None

        request = MemoryRetrievalRequest(
            target=RetrievalTarget(agent_id=agent_id, user_id=user_id),
            input=query,
            retrieve=MemoryPreviewLayers(
                l0=bool(retrieve.get("l0")),
                l1=bool(retrieve.get("l1")),
                # ⚠️ L2/L3 在**另一个处理器**（profile.py 的
                # MemoryProfileProcessor）里取，而且那边每轮现读。这里显式
                # 关掉，而不是"靠默认值"：那两个开关默认是 true，不写就会
                # 白读一遍画像与场景清单 —— 读出来还没人用（`blocks_from`
                # 只渲染响应里有的段，多出来的段会被丢掉），症状是一次
                # 白花的查询，不报错。
                l2=False,
                l3=False,
            ),
            limits=MemoryPreviewLimits(l0=_LIMIT, l1=_LIMIT),
            # ⚠️ **不传 session_id** —— L0 与 L1 都跨会话
            # --------------------------------------------
            # 传了它，L0 的召回就被收窄到**当前这一个会话**（见 schema 的
            # `session_id` 说明）。而分类提示词里判 l0 的线索是「我之前说过」
            # 「我告诉过你」「上次」「你还记不记得」「我们聊过」——**这些多数
            # 发生在别的会话里**。收窄之后，那些线索判出来的 l0=true 会去搜
            # 一个注定搜不到答案的范围。
            #
            # 实测症状：用户在**新会话**里问「你上次推荐我练什么曲子」，
            # 预召回搜本会话（只有这一句问话）-> 空 -> 上下文里没有 L0 段
            # -> 主模型只好去调记忆检索**工具**，而那个工具走的是跨会话
            # （`scope=user`）—— 同一个问题，两条路给出不同的范围。
            #
            # 省略即跨会话，这是 schema 里写明的**默认值**；相关性排序与
            # `_LIMIT` 负责把结果收在合理条数内。
            #
            # ⚠️ 只影响**召回**。记忆的**提炼**（L1/L2/L3 的生成）仍然按会话
            # 进行 —— 那是 `memory_atom_cursor` / `memory_scene_cursor` 的
            # 事，与本处理器无关，不要因为这一行去动那条链路。
        )

        factory = get_service_factory()

        async with factory.scope(
                MemoryQueryService, begin=False
        ) as service:

            return await service.retrieval_preview(request)


# ============================================================
# internals
# ============================================================


def _degraded(preview: Any) -> list[dict[str, str]]:
    """
    哪几层没查成（进保活 payload，供排查）。

    它是 `retrieval_preview` 的**分层**降级结果 —— 与 `optional=True` 那个
    处理器级降级不是一回事：这里是"层"挂了，那层没有内容、其余照常。
    所以它既不抛异常，也不该被当成"这一轮没有记忆"。
    """

    return [
        {"layer": str(item.layer), "reason": str(item.reason)}
        for item in (getattr(preview, "degraded", None) or ())
    ]


def _persisted(raw: Any) -> list[dict[str, Any]]:
    """
    `state["memory"]` -> 上一次 INITIAL 存下来的分节。

    **形状守卫，不做内容校验**：payload 过 checkpoint，也可能残留自一个
    旧版本的形状。对不上就当"没有" —— 那这一轮的 CONTINUATION 少几段记忆，
    而不是整轮对话失败。与 `retrieve` 那格同一条取舍。
    """

    if not isinstance(raw, Mapping):
        return []

    blocks = raw.get("blocks")

    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return []

    return [
        {
            "label": str(block["label"]),
            "text": str(block["text"]),
            "source_ids": [
                str(item) for item in (block.get("source_ids") or ())
            ],
        }
        for block in blocks
        if isinstance(block, Mapping)
        and isinstance(block.get("label"), str)
        and isinstance(block.get("text"), str)
    ]


