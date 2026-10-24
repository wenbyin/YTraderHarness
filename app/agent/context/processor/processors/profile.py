"""
阶段 2：PROFILE_LOADER —— 注入用户画像（规格 §二十九 第 6 条）。

数据从哪来（唯一规则）
------------------
    配了 ProfileSource -> **源说了算**
    没配              -> 用 state 里绑定的 `user_context.profile`

⚠️ `optional=True`
----------------
画像拿不到不该让整轮对话失败（规格 §二十八 把"某个非关键 Profile 字段
不可用"列为 Optional）。但 `user_id` 本身是**必需**输入：一个"不知道是
谁"的画像查询没有意义，而那种缺失通常意味着上游漏配（应当 FAILED）。

没有 user_id 的部署怎么办：把 `context_policy.include_user_profile`
设成 false（那是显式决定），或者给一个空串 `user_id: ""`
（空串是"有值但为空"，于是本处理器静默跳过 —— 见下）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.common.context import StageId
from app.common.factories import get_service_factory
from app.services.memory.application import MemoryQueryService
from app.services.memory.application.profile_cache import (
    get_cached_profile_blocks,
    put_cached_profile_blocks,
)
from app.services.memory.domain.schemas.query import (
    MemoryPreviewLayers,
    MemoryPreviewLimits,
    MemoryRetrievalRequest,
    RetrievalTarget,
)

from ..base import ContextProcessor
from ..context import ContextBuildContext
from ..dependencies import dependencies_of
from ..spec import ContextProcessorSpec, StateBinding
from ._blocks import blocks_from, inject, mapping

__all__ = [
    "MemoryProfileProcessor",
    "UserProfileProcessor",
]

#: 这一段在 system 里的标题。
#:
#: ⚠️ **不叫"用户画像"** —— 那个名字归记忆模块的 L3 长期画像
#: （app/agent/prompt/prompts/memory_retrieval.py 的 `PERSONA_TITLE`）。
#: 两段同时注入时，两个一模一样的 `## 用户画像` 会让模型（与读快照的人）
#: 分不清哪段是哪来的。
#:
#: 改这一侧而不是那一侧：`PERSONA_TITLE` 会进 `POST /memory/retrieval`
#: 响应的 `sections[].title`（对外契约的一部分），而本常量只落在这个
#: 处理器自己的分节上。
#:
#: 两者的数据也不同源：这一段来自 ProfileSource（租户 / 偏好 / 权限
#: 摘要），L3 那段是从对话里抽象出来的长期画像。
LABEL = "用户档案"


class UserProfileProcessor(ContextProcessor):
    """把用户画像注入成一段 system 片段。"""

    spec = ContextProcessorSpec(
        name="user_profile",
        order=20,
        stage=StageId.PROFILE_LOADER,
        when="context_policy.include_user_profile",
        inputs={
            # ⚠️ `user_id` 是**可选**的
            # ----------------------
            # "这次没有具体用户"（匿名 / 单用户部署 / 系统调用）是合法状态，
            # 不是接线错误。把必需性交给**策略开关**
            # （`context_policy.include_user_profile`）表达 —— 那是一个
            # 显式的、进快照的决定。
            #
            # 真需要"必须有画像"的部署怎么办：让 ProfileSource 在拿不到
            # 画像时抛错（源自己声明 required），而不是在这里把 user_id
            # 变必需 —— 后者会让每一条匿名请求都以 ContextStateError 失败。
            "user_id": StateBinding("user_id", required=False),
            "profile": StateBinding("user_context.profile", required=False),
        },
        optional=True,
        description="注入用户画像（租户 / 偏好 / 约束）。",
    )

    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:

        user_id = str(inputs.get("user_id") or "").strip()

        if not user_id:
            # 有值但为空 = "这次没有具体用户"（匿名 / 系统调用）。
            # 这不是错误，因此不 warn（否则每条匿名请求都会刷日志）。
            return context

        deps = dependencies_of(context)

        profile: Mapping[str, Any] | None

        if deps.profile_source is not None:

            profile = await deps.profile_source.load_profile(user_id=user_id)

        else:

            raw = inputs.get("profile")

            profile = raw if isinstance(raw, Mapping) else None

        if not profile:
            return context

        text = _render(profile)

        if not text:
            return context

        counter = deps.resolved_token_counter()

        context.add_section(
            name="profile",
            label=LABEL,
            text=text,
            source_ids=(f"user:{user_id}",),
            token_estimate=counter.count_text(text),
            # 画像值来自租户 / 用户配置，按引用数据定界。
            untrusted=True,
        )

        return context


#: L2 场景索引最多注入几篇（= 契约里 `limits.l2` 的默认值）。
#:
#: 与 `memory.py` 的 `_LIMIT`（L0/L1 各 5 条）是同一种取舍，但数不同：那条
#: 限制的是**召回**的条数（一条一段正文，贵），这条限制的是清单的行数（一行
#: 是"文件名｜热度｜时间：摘要"，十几行本来就由 L2 自己的 `DEFAULT_MAX_SCENES`
#: 把着）。显式写出来是为了让"注入几篇"在一个地方看得见 —— 靠契约默认值
#: 的话，改那边会静默改这里。
_L2_LIMIT = 12


class MemoryProfileProcessor(ContextProcessor):
    """把 L3 长期画像与 L2 场景索引注入成两段 system 片段。"""

    spec = ContextProcessorSpec(
        name="memory_profile",
        # ⚠️ 45：与 `memory`(50) **同一个阶段**，排在它前面
        # ------------------------------------------------
        # 于是 system 里四段的顺序（画像 / 场景 / 原话 / 记忆）与拆分之前
        # **逐字节相同** —— 这次拆的是"哪段代码在哪取数"，不是"它们排在哪"。
        #
        # 换阶段（比如挪到 PROFILE_LOADER 与 `user_profile` 作伴）会改变那
        # 两段在 system 里的位置：它们会从"历史之后"挪到"摘要之前"。那是
        # 一个产品决定，不是这次拆分的副产品 —— 真要挪，改 stage/order 两
        # 个值即可（pipeline 只要求 order 全局唯一且与阶段顺序一致）。
        order=45,
        stage=StageId.MEMORY_RETRIEVER,
        # 记忆的开关。**不是** `include_user_profile`：那一个管的是
        # ProfileSource 给的租户档案（见本文件的 UserProfileProcessor），
        # 与这两层记忆内容无关。
        when="context_policy.include_memory",
        inputs={
            # `agent_id`（= duty_key）在 retrieve 里，与 memory.py 同一格。
            # 缺了它就没法定位 scope（那两层按 (agent_id, user|org) 存）。
            "retrieve": StateBinding(
                "context_policy.memory_retrieve", required=False
            ),
            "user_id": StateBinding("user_id", required=False),
        },
        optional=True,
        description=(
            "注入 L3 长期画像与 L2 场景索引"
            "（Redis 缓存 + 未命中时查库；两层分别缓存/失效）。"
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
            # 身份不全 = 定位不出 scope。画像与场景都是**按 scope** 存的
            # （agent + user|org），缺一半就查不到该查的那一份。
            #
            # 与 memory.py 同一处理，但**不含**它那一层额外含义：那里
            # agent_id 为空还代表"上游判这次不召回"，这里只代表"定位不
            # 了" —— 画像是每轮都带的东西，与"要不要召回"无关。
            return context

        target = RetrievalTarget(agent_id=agent_id, user_id=user_id)
        scope = target.to_scope()

        # L2（场景索引）与 L3（画像）**分开缓存**：两者的更新时机不同
        # （L2 每次整合、L3 只在 L2 举手或首次创建时），合成一个键会让
        # L2 的一次变化白丢 L3 的缓存。
        #
        # 注入顺序保持 persona -> scenes，与 render_blocks 的排版一致。
        l3_hit, l3_blocks = await get_cached_profile_blocks(scope, "l3")

        if not l3_hit:
            preview = await self._preview(
                target=target,
                l2=False,
                l3=True,
            )
            l3_blocks = blocks_from(preview)
            await put_cached_profile_blocks(scope, "l3", l3_blocks)

        l2_hit, l2_blocks = await get_cached_profile_blocks(scope, "l2")

        if not l2_hit:
            preview = await self._preview(
                target=target,
                l2=True,
                l3=False,
            )
            l2_blocks = blocks_from(preview)
            await put_cached_profile_blocks(scope, "l2", l2_blocks)

        # 空列表也缓存：这个 scope 还没有画像 / 场景是有效结果，
        # 建出来时对应的写路径会把它删掉。
        if l3_blocks:
            inject(context, l3_blocks)

        if l2_blocks:
            inject(context, l2_blocks)

        return context

    # --------------------------------------------------------

    @staticmethod
    async def _preview(
            *,
            target: RetrievalTarget,
            l2: bool,
            l3: bool,
    ) -> Any:
        """
        取这一份 scope 的**指定层**（缓存未命中时才会走到这里）。

        调用方按层分别调用：L2 与 L3 有各自的缓存键与失效时机。

        ⚠️ 走 `retrieval_preview`，与 `memory.py` / 管理端**同一条实现**
        --------------------------------------------------------
        `retrieve` 显式关掉 L0/L1（那两层归 memory.py），`input` **不传** ——
        L2/L3 不需要查询串，契约里那条"要召回才必填"就是为这里留的口子。

        因此本处理器**不看 `run.phase`**：这两层每轮都该在，而它们便宜到
        不必保活（见模块 docstring 与 react/state.py 的 `memory` 通道说明）。

        ⚠️ 与 memory.py 的 `_preview` 一样**不兜异常**：连服务都取不到时
        让它抛，由 `spec.optional=True` 记 FAILED + warning 并跑完这一轮。
        在这里 catch 掉会让"这两层整个不可用"在快照里看不见。
        """

        request = MemoryRetrievalRequest(
            target=target,
            retrieve=MemoryPreviewLayers(
                l0=False,
                l1=False,
                l2=l2,
                l3=l3,
            ),
            limits=MemoryPreviewLimits(l2=_L2_LIMIT),
        )

        factory = get_service_factory()

        async with factory.scope(
                MemoryQueryService, begin=False
        ) as service:

            return await service.retrieval_preview(request)


# ============================================================
# internals
# ============================================================


def _render(profile: Mapping[str, Any]) -> str:
    """
    把画像渲染成几行 `- key: value`。

    刻意**不**用 JSON：system 里给模型看的东西用 JSON 反而更难读，
    而画像的键值大多是短句。嵌套值直接 `str()`（画像是给人看的摘要，
    不是需要机器解析的结构）。
    """

    lines: list[str] = []

    for key, value in profile.items():

        if value is None or value == "":
            continue

        if isinstance(value, (list, tuple, set)):

            rendered = "、".join(
                str(item) for item in value if str(item).strip()
            )

        elif isinstance(value, Mapping):

            rendered = "; ".join(
                f"{inner_key}={inner_value}"
                for inner_key, inner_value in value.items()
            )

        else:
            rendered = str(value)

        if rendered.strip():
            lines.append(f"- {key}: {rendered}")

    return "\n".join(lines)
