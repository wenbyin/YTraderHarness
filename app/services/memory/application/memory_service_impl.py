"""
MemoryServiceImpl —— 检索与索引的实现。

两个入口形状完全不同，理由也完全不同：

    search    请求路径。身份由调用方给（HTTP 从 token、工具从运行时），
              与请求体里的"搜什么"拼成过滤条件
    extract   后台路径。由 memory.l0.extract 任务驱动，按 run_id 取消息、
              写两张索引表。**没有调用者身份**，它索引的就是那一轮的全部
              历史（用户消息也在内）

⚠️ 融合按**排名**做（RRF），不按分数相加
------------------------------------
两路的分数不同量纲：关键词是 `ts_rank`（本实例实测量级 0.06 上下），
向量是余弦相似度（0~1）。直接相加的话，谁的量级大谁说了算 —— 而那是
实现细节决定的，不是相关性。RRF 只用"在第几名"，天然免疫量纲：

    score(d) = Σ_routes  1 / (K + rank_in_route(d))

K 取 60 是 RRF 那篇论文的常用值：它决定"第 1 名与第 2 名的差距"有多陡。
K 小则头部龙头的权重更大，K 大则两路的名次更平等。60 是既不极端、
也不需要按语料调的中间值。

⚠️ 某一路挂掉不让整轮失败
--------------------
与 document 的 `degraded_routes` 同一取舍：另一路召回到的内容是**真实
可检索**的。但静默跳过会让调用方分不清"没召回到"和"后端挂了"，
所以降级写进响应的同时打一条 warning 日志。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any
from uuid import UUID, uuid4

from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext
from app.common.tool_caller import current_tool_caller
from app.services.document.embedder.base import EmbeddingOutput
from app.services.document.pipeline.registry import PipelineRegistry
from app.services.session.domain.entities import AgentMessage
from app.services.session.domain.schemas.enum import MessageKind
from app.services.session.repository.agent_session import (
    AgentSessionRepository,
)
from app.services.session.repository.message import MessageRepository

from ..domain.entities import (
    IndexedMessage,
    MemoryQueryScope,
    MemoryRecall,
)
from ..domain.schemas import (
    DegradedRoute,
    MemoryRecallRoute,
    MemorySearchHit,
    MemorySearchMode,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySearchScope,
)
from ..domain.tasks import MemoryExtractParams
from ..fusion import RRF_K, fuse_by_rank
from ..repository.memory import MemoryIndexRepository
from ..tokenization import index_text, tsquery_text
from ._assembly import build_degraded, message_text, to_fts_row, to_hit, to_vec_row
from .memory_service import MemoryService

__all__ = ["MemoryServiceImpl"]

from ...tool.registry import tool

logger = logging.getLogger(__name__)


#: RRF 的常数。见模块 docstring —— 它只影响排序的陡峭程度，不影响召回集合。
#:
#: ⚠️ 它**不在本模块定义**，是从 `..fusion` 引进来的（L0 的检索与 L1 去重
#: 的候选召回共用一份）。两处各定义一个 60，迟早会变成 60 与 45。
#: （上面那行 import 就是它的定义处，这里不再重复赋值。）


#: 召回深度 = 最终条数的倍数。
#:
#: ⚠️ 为什么不直接取 top limit
#: -----------------------
#: 只看每路前 N 条来融合，等于假定两路的前 N 条高度重叠。真实情况正相反：
#: 向量路擅长召回"换了个说法"的消息（关键词搜不到），关键词路擅长召回
#: 精确的专名与数字（向量会被近义淹没）。两路各取多一点再融合，才是
#: hybrid 的意义所在。
#:
#: 4 倍 + 下限 20：limit=10 时召回 40 条，limit=3 时召回 20 条 ——
#: 小 limit 下也要有足够的候选可供排序，否则"取前 3"退化成"取关键词前 3"。
RECALL_MULTIPLIER = 4
MIN_RECALL_LIMIT = 20


#: 向量路的**相似度下限**。低于它的候选不进入融合。
#:
#: ⚠️ 为什么必须有这个下限
#: -------------------
#: 关键词路自己就是选择性的（`tsv @@ tsquery` 是个过滤条件），命中的都
#: 是真匹配。向量路**不是** —— 它是"按相似度排序"，于是永远能返回
#: recall_limit 条，哪怕全库跟查询毫无关系（相似度 0 的那条也会被排到
#: 第 20 名）。而 RRF 是按**名次**给分的，第 2 名的无关消息拿到
#: 1/(60+1)，足够挤进结果里。
#:
#: 端到端跑出来的症状（冒烟脚本第 [3] 段实测）：查询"股东户数"，
#: 一条讲天气的消息照样出现在结果里。对模型来说那是纯噪声 —— 它会把
#: "搜到了 10 条"当成"这个话题有 10 条相关记忆"。
#:
#: ⚠️ 0.35 是一个**待校准的判断**，不是实测值
#: -------------------------------------
#: 向量已 L2 归一化，所以分数就是余弦相似度。bge-small-zh 上"语义相关"
#: 通常落在 0.5 以上、完全无关落在 0.3 以下，因此 0.35 是这个模型上
#: 一个合理的起点。但它**没有**在本仓库的语料上测过 —— 换模型、换语种、
#: 换业务领域都要重新看这个数。
#:
#: 调它的两个方向各有一个可观察的症状：
#:     调太高    该召回的被挡掉（表现为"明明说过却搜不到"），且不报错
#:     调太低    无关消息混进结果（表现为"搜什么都有一堆结果"）
#:
#: 宁可偏高：**少召回**由关键词路与"多说一句"补，**假召回过**会直接
#: 污染模型的上下文，而它没有办法从结果里看出哪条是噪声。
VECTOR_MIN_SIMILARITY = 0.35


class MemoryServiceImpl(MemoryService):

    def __init__(
            self,
            *,
            index: MemoryIndexRepository,
            messages: MessageRepository,
            sessions: AgentSessionRepository,
            pipeline_registry: PipelineRegistry,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._index = index
        self._messages = messages
        self._sessions = sessions

        # 与 document 模块**共用**同一个注册表与同一个 embedder 实例：
        # 模型是进程级预热的（几 GB 权重），在这里再建一个等于把内存翻倍，
        # 更糟的是两份权重可能加载成不同的东西。
        self._pipeline_registry = pipeline_registry

    # ========================================================
    # 工具入口（暴露给主模型的那一个）
    # ========================================================

    @tool(key="memory.search")
    async def search_memory(
            self,
            request: MemorySearchRequest,
    ) -> MemorySearchResponse:
        """
        L0级别记忆，对话原文查询。
        FTS & VECTOR双路检索。参数：query 查询文本；scope 取样范围
        （session=当前会话，user=该用户的所有会话）；mode 检索方式
        （hybrid=关键词+向量，keyword、vector 只跑一路）；limit 条数。
        返回原文片段与定位信息，按相关度降序。

        ⚠️ 身份由运行时给，**不在入参里**
        ------------------------------
        这个包装方法存在的唯一理由就是"只收一个模型参数"：
        工具基建只认这一种形状（见 tool/registry/schema.py 的
        `resolve_request_model`），多一个参数就会退化成"扁平参数"那条路，
        而那条路不把 dict 反序列化成模型（就是线上那 6 种报错的成因）。

        `user_id` / `session_id` 从**进程内的调用者上下文**取
        （app/common/tool_caller.py）—— 它们绝不能进 schema：模型能填一个
        别人的 user_id，那就是跨租户读。

        ⚠️ 拿不到身份就拒绝
        ----------------
        拿不到只有两种可能：不来自 agent 那条链，或者那条链没接上。
        两种情况下"是谁"都不知道，而记忆检索**必须**知道以谁的身份搜 ——
        放行等于把全库摊开。所以这里直接报错，不回落到"不过滤"。
        """

        caller = current_tool_caller()

        if caller is None or caller.user_id is None:
            raise ValueError(
                "拿不到调用者身份，无法检索记忆 —— 拒绝而不是放行。"
            )

        # ⚠️ scope=user 时**不能**带 session_id，scope=session 时**必须**带
        # ------------------------------------------------------------------
        # 服务侧对这两种组合都做了校验（见 `_resolve_scope`）。这里按调用
        # 发生的会话填，于是"上一轮我们说过什么"这种问法默认就能用。
        if request.scope == MemorySearchScope.USER:
            return await self.search(request, user_id=caller.user_id)

        if caller.session_id is None:
            raise ValueError(
                "这次调用不挂在任何会话上，没法按 scope=session 检索；"
                "要跨会话检索请传 scope=user。"
            )

        return await self.search(
            request,
            session_id=caller.session_id,
            user_id=caller.user_id,
        )

    # ========================================================
    # 检索
    # ========================================================

    async def search(
            self,
            request: MemorySearchRequest,
            *,
            session_id: UUID | None = None,
            user_id: str | None = None,
    ) -> MemorySearchResponse:
        """
        一轮混合召回。

        **不是工具** —— 它是 HTTP 路径（`POST /memory/search`）与上面那个
        工具包装共用的实现。身份是显式入参，因为两条入口确认身份的方式
        不同：HTTP 从 token，工具从运行时上下文。
        """

        scope = await self._resolve_scope(
            request, session_id=session_id, user_id=user_id
        )

        recall_limit = max(
            request.limit * RECALL_MULTIPLIER,
            MIN_RECALL_LIMIT,
        )

        rankings: dict[MemoryRecallRoute, list[MemoryRecall]] = {}
        degraded: list[DegradedRoute] = []

        if request.mode in (
                MemorySearchMode.HYBRID, MemorySearchMode.KEYWORD
        ):
            recall = await self._recall_keyword(
                request.query, scope, recall_limit, degraded
            )
            if recall is not None:
                rankings[MemoryRecallRoute.KEYWORD] = recall

        if request.mode in (
                MemorySearchMode.HYBRID, MemorySearchMode.VECTOR
        ):
            recall = await self._recall_vector(
                request.query, scope, recall_limit, degraded
            )
            if recall is not None:
                rankings[MemoryRecallRoute.VECTOR] = recall

        fused = _fuse(rankings)

        # 截断在**融合之后**：recall 阶段刻意取多了，响应只给 limit 条。
        ordered = fused[: request.limit]

        indexed = await self._index.load(
            [recall.message_id for recall, _, _ in ordered]
        )

        results: list[MemorySearchHit] = []

        for recall, routes, score in ordered:

            row = indexed.get(recall.message_id)

            if row is None:
                # 索引行不见了（另一路写进来又被单独清理过）。丢掉而不是
                # 报错：拿不到原文的命中本来也呈现不出来，而"少一条"
                # 比"整轮失败"轻得多。
                continue

            results.append(
                to_hit(
                    indexed=row,
                    score=score,
                    matched_by=sorted(routes, key=lambda r: r.value),
                )
            )

        return MemorySearchResponse(
            search_id=uuid4(),
            query=request.query,
            mode=request.mode,
            scope=request.scope,
            results=results,
            degraded_routes=degraded,
        )

    # ========================================================
    # 索引（后台任务）
    # ========================================================

    async def message_extract(
            self,
            params: MemoryExtractParams,
            ctx: Any,
    ) -> None:

        messages = await self._messages.list_by_run(params.run_id)

        if not messages:
            # 一次运行完全没产出消息是常见的（图一进来就失败、排队期间被
            # 取消）。没有数据就是没有数据，不是失败 —— 抛出去会让它被
            # 重试三次，每次拿到同一个空列表。
            logger.debug(
                "[Memory] 运行 %s 没有消息，跳过索引。", params.run_id,
            )
            return

        # user_id / org_id 在**会话**上（agent_message 没有这两列），
        # 取一次给这一轮的所有行共用。
        session = await self._sessions.get_any(params.session_id)

        if session is None:
            # 会话没了（将来的删除端点）而消息还在。索引行不带归属的话
            # 它会对**所有**用户可见（NULL 的存量语义），那是一条越权路径 ——
            # 所以宁可整轮不索引，并让它作为一次失败被看见。
            raise NotFoundError(
                f"会话 {params.session_id} 不存在，无法确定索引行的归属。"
            )

        payload = _indexable(messages)

        if not payload:
            logger.debug(
                "[Memory] 运行 %s 的 %d 条消息都没有可索引正文，跳过。",
                params.run_id,
                len(messages),
            )
            return

        # ⚠️ 向量化失败**不吞**：让它抛出去、由任务重试接手。
        # 吞掉的后果是"关键词索引有、向量索引没有"，而这个缺口除了
        # 搜不到之外没有任何表现 —— 而重试是廉价且幂等的（写回走
        # ON CONFLICT DO UPDATE）。
        embeddings = await self._embed_documents(
            [text for _, text, _ in payload]
        )

        fts_rows = [
            to_fts_row(
                message,
                tokens=tokens,
                user_id=session.user_id,
                org_id=session.org_id,
            )
            for message, _, tokens in payload
        ]

        vec_rows = [
            to_vec_row(
                message,
                embedding=list(embedding),
                user_id=session.user_id,
                org_id=session.org_id,
            )
            for (message, _, _), embedding in zip(payload, embeddings)
        ]

        written_fts = await self._index.upsert_fts(fts_rows)
        written_vec = await self._index.upsert_vec(vec_rows)

        logger.info(
            "[Memory] 运行 %s 索引完成：关键词 %d 条、向量 %d 条"
            "（该轮共 %d 条消息，跳过 %d 条 —— 工具活动 / 无正文 / 切不出词元）。",
            params.run_id,
            written_fts,
            written_vec,
            len(messages),
            len(messages) - len(payload),
        )

    # ========================================================
    # internals
    # ========================================================

    async def _resolve_scope(
            self,
            request: MemorySearchRequest,
            *,
            session_id: UUID | None,
            user_id: str | None,
    ) -> MemoryQueryScope:
        """
        请求 + 身份 -> 过滤条件。**唯一**同时看到两边的地方。

        ⚠️ 两种 scope 与 session_id 的组合都要报错
        --------------------------------------
            scope=session 而没有 session_id    -> 不知道搜哪个会话
            scope=user    而给了 session_id    -> 两个意图冲突，猜哪个都是错

        宁可报错也不猜：猜错的那个方向是"以为在搜全部、其实只搜了一个
        会话"，而用户没有任何办法从结果里看出来。

        ⚠️ 有 session_id 就**必须**校验归属，且没有身份时拒绝
        ------------------------------------------------
        校验走 `AgentSessionRepository.get(session_id, user_id=...)` ——
        那个方法与 session 模块的接口共用同一个可见性规则（`= 调用者 OR
        IS NULL`）。没有身份（user_id is None）时不校验而直接搜，就等于
        "谁都能读任意会话" —— 所以那样直接拒绝，与 auth 拦截器
        "拿不到身份时默认拒绝"是同一条纪律。
        """

        if request.scope == MemorySearchScope.SESSION:

            if session_id is None:
                raise ValueError(
                    "scope=session 需要 session_id：没有它不知道搜哪个会话。"
                )

            if user_id is None:
                raise ValueError(
                    "没有调用者身份，无法确认这个会话是否属于它 —— "
                    "拒绝而不是放行。"
                )

            session = await self._sessions.get(session_id, user_id=user_id)

            if session is None:
                # "不存在"与"不属于你"共用同一个响应，不泄露 id 是否存在。
                raise NotFoundError(f"会话 {session_id} 不存在。")

            return MemoryQueryScope(
                session_id=session_id,
                user_id=user_id,
                roles=_role_values(request),
                time_from=request.time_from,
                time_to=request.time_to,
            )

        if session_id is not None:
            raise ValueError(
                "scope=user 不接受 session_id：要限定单个会话请用 "
                "scope=session。"
            )

        # ⚠️ 两种 scope 都要求身份
        # ---------------------
        # scope=user 更要紧：它是**唯一**一条放宽到会话之外的路径。身份
        # 缺失时若不在这里断掉，过滤就只剩"kind / 角色 / 时间窗"那几条 ——
        # 等于把全库所有人的消息都摊在结果里。
        #
        # 仓储侧还有一道 `user_id IS NULL` 的兜底（见 memory_impl 的
        # `_conditions`），所以这里即使漏判也**不会**越权，只会安静地少
        # 召回。两道都要有：这一道给出清晰的错误，那一道保证不越权。
        if user_id is None:
            raise ValueError(
                "没有调用者身份，无法确定搜索范围 —— 拒绝而不是放行。"
            )

        return MemoryQueryScope(
            session_id=None,
            user_id=user_id,
            roles=_role_values(request),
            time_from=request.time_from,
            time_to=request.time_to,
        )

    async def _recall_keyword(
            self,
            query: str,
            scope: MemoryQueryScope,
            recall_limit: int,
            degraded: list[DegradedRoute],
    ) -> list[MemoryRecall] | None:
        """
        关键词路。没跑成时往 `degraded` 追加一条并返回 None。

        ⚠️ 返回 None 与返回空列表是**两件事**
        --------------------------------
            空列表   跑了，没命中
            None     没跑（降级）

        两者都让调用方"没拿到这条路的排序"，但只有后者要进 degraded_routes。
        混成一个的话，响应里会出现"关键词后端有问题"的假告警（每次没命中
        都报一次），而真出问题时又不显眼。
        """

        tsquery = tsquery_text(query)

        if tsquery is None:
            # 切不出词元（纯标点 / emoji / 空白）。这是**输入**的性质，
            # 不是后端故障，但对这条路来说结论一样：它没法参与。
            degraded.append(
                build_degraded(
                    MemoryRecallRoute.KEYWORD,
                    "查询里没有可切分的词元，关键词路无法参与"
                    "（向量路不受影响）。",
                )
            )
            return None

        try:
            return await self._index.search_keyword(
                tsquery=tsquery,
                scope=scope,
                recall_limit=recall_limit,
            )

        except Exception as exc:  # noqa: BLE001 —— 降级，见模块 docstring
            logger.warning(
                "[Memory] 关键词路召回失败，降级为向量路：%s", exc,
            )
            degraded.append(
                build_degraded(
                    MemoryRecallRoute.KEYWORD, f"关键词路召回失败：{exc}",
                )
            )
            return None

    async def _recall_vector(
            self,
            query: str,
            scope: MemoryQueryScope,
            recall_limit: int,
            degraded: list[DegradedRoute],
    ) -> list[MemoryRecall] | None:

        try:
            embedding = await self._embed_query(query)

        except Exception as exc:  # noqa: BLE001 —— 降级
            logger.warning(
                "[Memory] 查询向量化失败，降级为关键词路：%s", exc,
            )
            degraded.append(
                build_degraded(
                    MemoryRecallRoute.VECTOR,
                    f"查询向量化失败（embedder 不可用？）：{exc}",
                )
            )
            return None

        try:
            recalls = await self._index.search_vector(
                embedding=embedding,
                scope=scope,
                recall_limit=recall_limit,
            )

        except Exception as exc:  # noqa: BLE001 —— 降级
            logger.warning(
                "[Memory] 向量路召回失败，降级为关键词路：%s", exc,
            )
            degraded.append(
                build_degraded(
                    MemoryRecallRoute.VECTOR, f"向量路召回失败：{exc}",
                )
            )
            return None

        # ⚠️ 截断在这里而不是在 SQL 里
        # -------------------------
        # SQL 的 `ORDER BY <=> LIMIT n` 是**索引可用**的形状（HNSW 要求
        # 按距离排序加 LIMIT 才走索引），而相似度下限是一个在 Python 侧
        # 过滤的阈值 —— 塞进 SQL（`WHERE embedding <=> :q < :t`）会让
        # 优化器认不出索引可用的形状，退化成顺序扫描：结果一样、只是慢。
        #
        # 代价是"多取了一些再丢掉"。recall_limit 不大（几十条），这部分
        # 开销远小于一次全表扫描。
        #
        # 返回**空列表**是正当结果：库里确实没有相似的东西。它不是降级
        # （这一路跑了），所以不进 degraded_routes。
        return [
            recall for recall in recalls
            if recall.score >= VECTOR_MIN_SIMILARITY
        ]

    # ---- 向量化 ----------------------------------------------------

    def _dense_embedder(self):
        """
        取稠密 embedder。

        ⚠️ 取不到时**抛**，由调用方决定降级还是失败
        -------------------------------------
        这条路在检索侧是可选的（降级进 degraded_routes），在索引侧不是
        （写在 extract 里，失败就该重试）。抛在这里、让两边各自决定，
        比在这里替它们决定好。
        """

        return self._pipeline_registry.embedder_for(EmbeddingOutput.DENSE)

    async def _embed_query(self, query: str) -> list[float]:
        """
        查询向量 —— 走 `embed_query`（带指令前缀），不是 `embed`。

        ⚠️ 两侧的调用方式不同是**模型要求**的，不是笔误
        ----------------------------------------
        bge-small-zh 的检索质量依赖查询侧那句指令（"为这个句子生成表示
        以用于检索相关文章："）。文档侧（`embed`）不加。搞反的表现是
        召回质量下降 —— **不报错**，所以这条区分必须留在代码里而不是
        靠约定。
        """

        embedder = self._dense_embedder()

        if embedder is None:
            raise RuntimeError("稠密 embedder 没有注册，向量路不可用。")

        return list(await embedder.embed_query(query, config={}))

    async def _embed_documents(
            self,
            texts: Sequence[str],
    ) -> list[list[float]]:
        """批量向量化消息正文。写索引侧用 `embed`（不加指令前缀）。"""

        embedder = self._dense_embedder()

        if embedder is None:
            raise RuntimeError("稠密 embedder 没有注册，无法建立向量索引。")

        vectors = await embedder.embed(list(texts), config={})

        return [list(vector) for vector in vectors]


# ============================================================
# 模块级纯函数：融合与过滤
# ============================================================


def _role_values(
        request: MemorySearchRequest,
) -> list[str] | None:
    """请求里的角色白名单 -> 字符串（并集去重，顺序稳定）。"""

    if request.roles is None:
        return None

    return sorted({role.value for role in request.roles})


def _indexable(
        messages: Sequence[AgentMessage],
) -> list[tuple[AgentMessage, str, str]]:
    """
    挑出该进索引的消息，返回 `(消息, 正文, 分词串)`。

    ⚠️ 只收 `kind == chat`，工具相关的一律不进索引
    -------------------------------------------
    `kind` 由 message_recorder 判定（见它的 `_kind`）：**带 tool_calls 的
    assistant 消息、以及 ToolMessage 的结果**都是 `tool_activity`，剩下的是
    `chat`。于是这一条过滤同时挡掉两类东西：

        带 tool_call 块的 assistant 消息    "我来查一下……"这种**动手前的
                                          旁白**。它是给工具调用做铺垫的
                                          过程话，不是结论 —— 而它带的那
                                          4 个调用块本身也不该进倒排
        工具结果（ToolMessage）             一坨 JSON。搜"股东户数"时它
                                          会把真正回答了问题的那条挤下去

    端到端实测的症状（就是促使改成这样的那个）：一条 5 条消息的运行里，
    索引里进的是 seq=2 那句"我来查一下平台当前实际注册的工具和能力清单。"
    —— 而真正回答了用户问题的 seq=5 没进去。

    ⚠️ 与 memory_l0.md 的规则**不同**，这是刻意的
    -----------------------------------------
    那份规格写的是"assistant 中**有自然语言文本的**保留"（也就是 seq=2
    那种要留下）。改成"只留 chat"是后来的决定：L0 要做的是**对话**的可
    检索记忆，过程旁白与工具产物属于 trace，不属于对话。规格那段按新口径
    理解即可。

    ⚠️ 顺带删掉的东西：`include_tool_activity` 与索引表的 kind / status
    -----------------------------------------------------------------
    索引侧收紧到只收 chat 之后，检索侧那个"要不要包含工具活动"的开关
    就失去了意义 —— 索引里没有这类行了，打开它只会得到空结果，而
    "调用方以为能搜到、实际永远搜不到"正是最该避免的那种谎。所以它连同
    `MemorySearchHit.kind` 一起从契约里删掉了。

    两张索引表的 `kind` / `status` 两列同样删了，判据是"有没有人读"：
    kind 按构造恒为 chat，status 从来没有消费者。**要读它们回
    `agent_message` 读** —— message_id 就在索引表的主键上，而索引表存的
    是派生数据，能回源表拿到的东西不该在这儿放第二份。详见
    models/sqlalchemy/_scope.py 里那段说明。

    ⚠️ 另两条过滤
    ----------
        正文为空（message_text 返回 None）
            模型返回空串的消息。规格里"content 为空则丢"落在这里。
        切不出词元
            纯标点、emoji、纯空白。`tokens` 列有 `<> ''` 的 CHECK ——
            放过去会在写库那一刻炸，而那时离原因已经很远。

    两条路索引的必须是**同一批消息**：向量路本来可以索引纯 emoji 的消息，
    但那样两路候选集就不一样了，而"某条只有向量路找得到"会变成一个
    无法解释的现象。
    """

    indexable: list[tuple[AgentMessage, str, str]] = []

    for message in messages:

        if message.kind != MessageKind.CHAT.value:
            continue

        text = message_text(message)

        if text is None:
            continue

        tokens = index_text(text)

        if not tokens:
            continue

        indexable.append((message, text, tokens))

    return indexable


def _fuse(
        rankings: dict[MemoryRecallRoute, list[MemoryRecall]],
) -> list[tuple[MemoryRecall, set[MemoryRecallRoute], float]]:
    """
    RRF 融合，返回按融合分降序的
    `(任一路的一条召回, 它被哪几路召回, 融合分)`。

    ⚠️ 实现搬去了 `..fusion`（L0 的检索与 L1 去重的候选召回共用一份）
    ---------------------------------------------------------
    这里只是把身份（`message_id`）喂进去。整个函数保留是因为它是本模块
    检索路径的**命名**：调用方问的是"融合这几路"，不是"用 RRF 融合"。
    算法、它的常数与那三条不能动的理由（分数原样带出、对象留哪个都行、
    排序必须确定）都写在 fusion 的 docstring 里。
    """

    return fuse_by_rank(
        rankings,
        key=lambda recall: recall.message_id,
    )
