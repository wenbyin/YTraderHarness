"""
MemoryAtomServiceImpl —— L1 抽取 + 去重（一次模型调用各一次）。

一次抽取的形状
------------
    读水位 -> 按水位取增量消息 -> [模型①：抽取] -> 组装实体
      -> [召回候选池] -> [模型②：冲突检测] -> 应用判定（写 / 改 / 删）
      -> 推水位

⚠️ 为什么抽取与去重是**同一个 handler 里的两步**
-----------------------------------------
两步之间传递的是一个**只在内存里存在**的上下文："刚刚抽出来的是哪几条、
它们的 record_id 是什么“。拆成两条任务就得把它塞进 params（一批十几条
UUID，撞 String(255) 的幂等键长）或让第二步去反推（按时间、按 version
—— 一条**猜**的规则，猜错的表现是漏掉几条不做去重且不报错）。

⚠️ 写入发生在**两步之后**，不是抽取之后立刻写
--------------------------------------
这是与”先写再改“相比更省事的一处：冲突判定的动作里有 `skip`（这条不要），
先写再 skip 就得再删一次。所以顺序是**先判定、后落库**。

代价：模型②失败会让模型①白跑一次（重试时重抽）。可接受 —— 两次都是
LLM 调用，而”先写后删“多出来的是**破坏性**操作。

⚠️ 模型调用走 ModelRuntime，不是自己拼客户端
--------------------------------------
超时、重试、降级、用量观测都在那一层，而且那里重试的是 **provider
attempt**，不是重跑本方法。本方法因此不为”模型抽风“写任何重试。

⚠️ 解析失败要抛，不要吞
------------------
"这一轮什么都没抽到”与“模型输出坏了”在数据上长得一样，但一个是常态、
一个是 bug。吞掉等于让后者永远查不出来（见 domain/schemas/atom.py）。
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole, get_model_registry
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.memory_atom_dedup import (
    render_user_message as render_dedup_message,
)
from app.agent.prompt.prompts.memory_atom_extract import (
    render_user_message as render_extract_message,
)
from app.common.exceptions import NotFoundError
from app.common.service.base import ServiceContext
from app.common.tool_caller import current_tool_caller
from app.services.document.embedder.base import EmbeddingOutput
from app.services.document.pipeline.registry import PipelineRegistry
from app.services.session.domain.entities import AgentMessage, AgentSession
from app.services.session.repository.agent_session import (
    AgentSessionRepository,
)
from app.services.session.repository.message import MessageRepository
from app.services.tool.registry import tool

from ..domain.entities.atom import (
    AtomFtsRow,
    AtomRecall,
    AtomVecRow,
    MemoryAtom,
    MemoryAtomQueryScope,
)
from ..domain.schemas.atom import (
    ConflictDecision,
    ExtractedScene,
    parse_conflicts,
    parse_extraction,
)
from ..domain.schemas.memory_atom import (
    DegradedAtomRoute,
    MemoryAtomRecallRoute,
    MemoryAtomSearchHit,
    MemoryAtomSearchMode,
    MemoryAtomSearchRequest,
    MemoryAtomSearchResponse,
    MemoryAtomSearchScope,
)
from ..domain.tasks import MemoryAtomExtractParams
from ..fusion import fuse_by_rank
from ..repository.atom import MemoryAtomRepository
from ..tokenization import index_text, tsquery_text
from .memory_atom_service import MemoryAtomService
from .trigger import submit_scene_extract, task_id_of

__all__ = [
    "AGENT_TYPE",
    "AGENT_TYPE_DEDUP",
    "DEFAULT_CONFLICT_TOP_K",
    "MemoryAtomServiceImpl",
]

logger = logging.getLogger(__name__)


#: 模型绑定用的 agent_type（见 model.yaml 的 bindings）。
#:
#: ⚠️ 两个 agent_type，不是“一个服务一个"
#: -----------------------------------
#: 抽取与去重是两次**取舍不同**的调用（抽取要读得多、写得全；去重要判得准），
#: 各自要能单独换模型。所以键按调用分，不按服务分。
AGENT_TYPE = "memory_atom_extract"
AGENT_TYPE_DEDUP = "memory_atom_dedup"

#: 每条**新记忆各自**留多少个候选（原规格的 `conflictRecallTopK`，默认 5）。
#:
#: 它是"给模型看多少个可能冲突的已有记忆"，不是召回质量的上限：池子是
#: **并集**（多条新记忆各自的候选合在一起），所以池子最多
#: `top_k × 新记忆条数` 条（去重后更少）。调大更不容易漏合并，代价是
#: 提示词变长。
#:
#: ⚠️ 这个上限**曾经是坏的**（2026-10-11 修）
#: ------------------------------------
#: 它按候选的 id 分组计数（`len(bucket)` 恒为 1），所以截断从没生效 ——
#: 池子实际是"所有新记忆召回到的全部候选的并集"。修好之后真正的行为与这
#: 段注释一致了。要改回大池子请调这个数，别把 `_candidate_pool` 里的按
#: 新记忆分组改回去。
DEFAULT_CONFLICT_TOP_K = 5

#: 冲突判定的动作取值域。**不在其中**的一律按 store 处理（见 `_apply`）。
_CONFLICT_ACTIONS = frozenset({"store", "skip", "update", "merge"})


#: 召回深度 = 最终条数的倍数（检索用，与 L0 同一条取舍）。
#:
#: ⚠️ 为什么不直接取 top limit
#: -----------------------
#: 只看每路前 N 条来融合，等于假定两路的前 N 条高度重叠。真实情况正相反：
#: 向量路擅长召回“换了个说法”的记忆（关键词搜不到），关键词路擅长精确的
#: 专名与数字（向量会被近义淹没）。两路各取多一点再融合，才是 hybrid 的
#: 意义所在。
RECALL_MULTIPLIER = 4

#: 召回条数的下限：小 limit 下也要有足够的候选可供排序，
#: 否则“取前 3"会退化成”取关键词前 3"。
MIN_RECALL_LIMIT = 20

#: 向量路的**相似度下限**。低于它的候选不进入融合。
#:
#: ⚠️ 与 L0 的 `VECTOR_MIN_SIMILARITY` 是**两个独立的值**，只是暂时同数
#: --------------------------------------------------------------
#: 它们要挡的是同一件事（向量路永远能返回 top_k 条，哪怕全库毫无关系），
#: 但语料不同：L0 是原始对话，L1 是提炼过的陈述（语义更密、相似度普遍偏高）。
#: 现在沿用 0.35，**没有**在 L1 的语料上校准过 —— 换 embedder 或换业务
#: 领域都要重新看这个数。
VECTOR_MIN_SIMILARITY = 0.35


class MemoryAtomServiceImpl(MemoryAtomService):

    def __init__(
            self,
            *,
            atoms: MemoryAtomRepository,
            messages: MessageRepository,
            sessions: AgentSessionRepository,
            pipeline_registry: PipelineRegistry,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._atoms = atoms
        self._messages = messages
        self._sessions = sessions

        # 与 document / L0 共用同一个 embedder 实例（进程级预热、几 GB），
        # 见 memory_service_impl 里同一处的说明。
        self._pipeline_registry = pipeline_registry

    # ========================================================
    # Handler 入口
    # ========================================================

    async def memory_atom_extract(
            self,
            params: MemoryAtomExtractParams,
            ctx: Any,
    ) -> None:

        # ---- 1. 会话与水位 ----------------------------------------
        #
        # 会话先取：user_id / org_id / thread_id 三列要从它带进每一条记忆。
        # 取不到就**抛** —— 记忆行不带归属的话，它对所有用户可见（NULL 的
        # 存量语义），那是一条越权路径。与 L0 的 extract 同一取舍。
        session = await self._sessions.get_any(params.session_id)

        if session is None:
            raise NotFoundError(
                f"会话 {params.session_id} 不存在，无法确定记忆的归属。"
            )

        cursor = await self._atoms.get_cursor(params.session_id)

        messages = await self._messages.list_chat_after(
            params.session_id, cursor.last_seq,
        )

        if not messages:
            # 增量为空是**常态**，不是失败：任务可能排晚了一步（前一条
            # 已经抽走了这一段）、或者那一轮没有 chat 消息。抛出去会让它
            # 白重试三次，每次拿到同一个空列表。与 L0 同一条纪律。
            logger.debug(
                "[Memory] 会话 %s 自 seq=%d 起没有新的 chat 消息，跳过 L1 抽取。",
                params.session_id, cursor.last_seq,
            )
            return

        last_seq = max(message.seq for message in messages)

        # ---- 2. 抽取（模型①）--------------------------------------
        scenes = await self._extract(messages=messages, session=session, cursor=cursor)

        # 归属的主 agent 职责：取本批消息上**最近的那个** duty_key，
        # 退回会话的 agent_id（存量消息可能没有 duty 信息）。
        duty_key = _duty_key(messages) or session.agent_id

        atoms = self._to_atoms(
            scenes,
            session=session,
            messages=messages,
            duty_key=duty_key,
        )

        last_scene_name = self._next_scene(scenes, fallback=cursor.last_scene_name)

        # ---- 3. 去重并落库（模型② + 三张表）-----------------------
        report = await self._dedup_and_write(
            atoms,
            session_id=params.session_id,
            user_id=session.user_id,
        )

        # ---- 4. 推水位 --------------------------------------------
        #
        # ⚠️ 必须在上面全部做完之后、且与它们同一个事务里（调用方的作用域）。
        # 提前推 = 那一段消息永久跳过；分事务 = 记录没写而水位动了，症状
        # 同样是永久跳过。契约里 advance_cursor 那一节写的就是这件事。
        #
        # ⚠️ 模型一条记忆都没抽到时**照推**：否则下一轮会拿同一段再问一遍
        # 模型，而它每次都答”没有值得记的“，永远出不去。
        await self._atoms.advance_cursor(
            params.session_id,
            last_seq=last_seq,
            last_scene_name=last_scene_name,
            user_id=session.user_id,
        )

        logger.info(
            "[Memory] 会话 %s 的 L1 完成：seq %d..%d 共 %d 条消息 -> "
            "新 %d / 改 %d / 合并 %d / 跳过 %d / 删 %d（情境 %r）。",
            params.session_id,
            cursor.last_seq + 1,
            last_seq,
            len(messages),
            report.stored,
            report.updated,
            report.merged,
            report.skipped,
            report.deleted,
            last_scene_name,
        )

        # ⚠️ 同一件事也打到控制台
        # ---------------------
        # 本应用**没有配置 logging**（main.py 里没有 basicConfig），所以上面
        # 那句 `logger.info` 在控制台上**一个字都不会出现**。
        #
        # 而"这一轮到底抽出来没有"是最常见的问题（"为什么这个会话没有
        # memory_atom？"）—— 答案就在这行里：
        #
        #     全是 0  ->  跑了，模型判"没有值得记的"（不是没跑）
        #     有计数   ->  抽到了，`skipped` 非 0 则是去重把它跳过了
        #
        # 水位读出来是空的（那一段已经被前一条抽走）在**上面**就 return 了，
        # 不会走到这里 —— 所以看到这行就一定意味着"模型这次真的答了"。
        print(
            f"[Memory] L1 完成：session={params.session_id} "
            f"seq {cursor.last_seq + 1}..{last_seq}（{len(messages)} 条消息）"
            f" -> 新 {report.stored} / 改 {report.updated}"
            f" / 合并 {report.merged} / 跳过 {report.skipped}"
            f" / 删 {report.deleted}，候选池 {report.pool_size} 条"
            f"（{'未调去重' if report.pool_size == 0 else '已调去重'}）"
            f"，情境={last_scene_name!r}",
            flush=True,
        )

        # ---- 5. 排下一次 L2 整合 ----------------------------------
        #
        # ⚠️ 位置在这里（L1 handler 的末尾）是**刻意的**，不是"顺手"：
        # 只有这一刻才同时成立两件事 ——
        #
        #     这一批真的抽出来了      上面几行已经把记忆落库、水位也推了
        #     触发者知道"完成了"      放进 params 或另挂钩子都得再传一次事实
        #
        # 挂到运行收尾那条路（L0/L1 的排法）不行：收尾时 L1 的任务**还没跑**
        # （排在 60s 后或立刻投递），而且相当一部分会空转 —— 那会让 L2 在
        # 没有新记忆的时候被反复排出来。判定与理由全在
        # `trigger.submit_scene_extract` 里，这里只负责"在这一刻问它"。
        #
        # ⚠️ 失败不抛（`submit_scene_extract` 自己吞掉并留 warning）：整合是
        # 旁路，它失败不该让一次成功的 L1 抽取变成失败的任务。而它会自愈 ——
        # 下一次 L1 完成会把同一个会话重新带上（增量还在，水位没动）。
        #
        # ⚠️ 已知代价：这一刻上面的写入**还没提交**（提交发生在 handler
        # 返回之后，由调用方的作用域做）。若随后回滚，L2 已经排在库里了，
        # 届时它读到空增量、空转一次。见 `submit_scene_extract` 的说明。
        await submit_scene_extract(
            session_id=params.session_id,
            causation_id=task_id_of(ctx),
        )

    # ========================================================
    # 工具入口（暴露给主模型的那一个）
    # ========================================================

    @tool(key="memory_atom.search")
    async def search_atom(
            self,
            request: MemoryAtomSearchRequest,
    ) -> MemoryAtomSearchResponse:
        """
        L1级别记忆，提炼过的结构化事实查询。FTS & VECTOR双路检索。
        参数：query 查询文本；scope 取样范围（session=当前会话，
        user=该用户的所有会话——记忆事实的主场）；mode 检索方式
        （hybrid=关键词+向量，keyword、vector 只跑一路）；limit 条数。
        返回记忆正文（完整、不截断）与类型、重要性、情境，按相关度降序。

        ⚠️ 什么时候该用它
        --------------
        "用户之前说过什么偏好 / 我们之前定过什么" —— 这类问题问的是
        **提炼过的记忆**，比翻原始对话划算：一条记忆已经是完整陈述，而
        原文要读好几轮才能拼出来。要原话、要证据链时才回去查原始对话。

        （这里刻意**不写另一个工具的名字**：模型看到的是函数名，
        而工具名是我们随时会改的标识，写死在描述里改错了不报错。）

        ⚠️ 身份由运行时给，**不在入参里**
        ------------------------------
        这个包装方法存在的唯一理由就是“只收一个模型参数”：工具基建只认
        这一种形状（见 tool/registry/schema.py 的 `resolve_request_model`），
        多一个参数就会退化成“扁平参数”那条路（就是线上那 6 种报错的成因）。

        `user_id` / `session_id` 从**进程内的调用者上下文**取
        （app/common/tool_caller.py）—— 它们绝不能进 schema：模型能填一个
        别人的 user_id，那就是跨租户读。

        ⚠️ 拿不到身份就拒绝
        ----------------
        拿不到只有两种可能：不来自 agent 那条链，或者那条链没接上。两种
        情况下“是谁”都不知道，而记忆检索**必须**知道以谁的身份搜 —— 放行
        等于把全库摊开。与 L0 的 `memory.search` 同一条纪律。
        """

        caller = current_tool_caller()

        if caller is None or caller.user_id is None:
            raise ValueError(
                "拿不到调用者身份，无法检索记忆 —— 拒绝而不是放行。"
            )

        # ⚠️ scope=user 时**不能**带 session_id，scope=session 时**必须**带
        # —— 两种组合服务侧都做了校验（见 `_resolve_scope`）。
        if request.scope == MemoryAtomSearchScope.USER:
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
    # 检索（HTTP 路径与工具共用的实现）
    # ========================================================

    async def search(
            self,
            request: MemoryAtomSearchRequest,
            *,
            session_id: UUID | None = None,
            user_id: str | None = None,
    ) -> MemoryAtomSearchResponse:
        """
        一轮混合召回。**不是工具** —— 它是 HTTP 路径与上面那个工具包装
        共用的实现，身份是显式入参。
        """

        scope = await self._resolve_scope(
            request, session_id=session_id, user_id=user_id
        )

        recall_limit = max(
            request.limit * RECALL_MULTIPLIER,
            MIN_RECALL_LIMIT,
        )

        rankings: dict[str, list[AtomRecall]] = {}
        degraded: list[DegradedAtomRoute] = []

        if request.mode in (
                MemoryAtomSearchMode.HYBRID, MemoryAtomSearchMode.KEYWORD
        ):
            recalls = await self._search_keyword(
                request.query, scope, recall_limit, degraded
            )
            if recalls is not None:
                rankings[MemoryAtomRecallRoute.KEYWORD.value] = recalls

        if request.mode in (
                MemoryAtomSearchMode.HYBRID, MemoryAtomSearchMode.VECTOR
        ):
            recalls = await self._search_vector(
                request.query, scope, recall_limit, degraded
            )
            if recalls is not None:
                rankings[MemoryAtomRecallRoute.VECTOR.value] = recalls

        fused = fuse_by_rank(rankings, key=lambda recall: recall.record_id)

        # 截断在**融合之后**：recall 阶段刻意取多了，响应只给 limit 条。
        ordered = fused[: request.limit]

        indexed = await self._atoms.load(
            [recall.record_id for recall, _, _ in ordered]
        )

        results: list[MemoryAtomSearchHit] = []

        for recall, routes, score in ordered:

            row = indexed.get(recall.record_id)

            if row is None:
                # 索引行没了（另一条任务刚好把它合并删掉）。丢掉而不是
                # 报错：拿不到正文的命中本来也呈现不出来，而“少一条”比
                # "整轮失败“轻得多 —— 与 L0 的 `search` 同一条取舍。
                continue

            results.append(
                MemoryAtomSearchHit(
                    record_id=row.record_id,
                    content=row.content,
                    type=row.type,
                    priority=row.priority,
                    scene_name=row.scene_name,
                    score=score,
                    matched_by=sorted(
                        (MemoryAtomRecallRoute(route) for route in routes),
                        key=lambda route: route.value,
                    ),
                    # `row` 是 `_atoms.load()` 读**主表**回来的实体，两个
                    # 时间列本来就在手上 —— 只是此前没往上传。
                    created_time=row.created_time,
                    updated_time=row.updated_time,
                )
            )

        return MemoryAtomSearchResponse(
            search_id=uuid4(),
            query=request.query,
            mode=request.mode,
            scope=request.scope,
            results=results,
            degraded_routes=degraded,
        )

    async def _resolve_scope(
            self,
            request: MemoryAtomSearchRequest,
            *,
            session_id: UUID | None,
            user_id: str | None,
    ) -> MemoryAtomQueryScope:
        """
        请求 + 身份 -> 过滤条件。**唯一**同时看到两边的地方。

        与 L0 的 `_resolve_scope` 同一套规则（两种 scope 与 session_id 的
        非法组合都报错、有 session_id 就必须校验归属、没有身份就拒绝），
        少一个时间窗。
        """

        if request.scope == MemoryAtomSearchScope.SESSION:

            if session_id is None:
                raise ValueError(
                    "scope=session 需要 session_id：没有它不知道搜哪个会话。"
                )

            if user_id is None:
                raise ValueError(
                    "没有调用者身份，无法确认这个会话是否属于它 —— 拒绝而不是放行。"
                )

            session = await self._sessions.get(session_id, user_id=user_id)

            if session is None:
                # "不存在”与“不属于你”共用同一个响应，不泄露 id 是否存在。
                raise NotFoundError(f"会话 {session_id} 不存在。")

            return MemoryAtomQueryScope(
                session_id=session_id,
                user_id=user_id,
            )

        if session_id is not None:
            raise ValueError(
                "scope=user 不接受 session_id：要限定单个会话请用 "
                "scope=session。"
            )

        # scope=user 是**唯一**一条放宽到会话之外的路径，身份缺失时若不在这里
        # 断掉，过滤就只剩仓储那道兜底 —— 它会退化成“只看得到无归属的行”，
        # 也就是安静地少召回，而不是报错。
        if user_id is None:
            raise ValueError(
                "没有调用者身份，无法确定搜索范围 —— 拒绝而不是放行。"
            )

        return MemoryAtomQueryScope(session_id=None, user_id=user_id)

    async def _search_keyword(
            self,
            query: str,
            scope: MemoryAtomQueryScope,
            recall_limit: int,
            degraded: list[DegradedAtomRoute],
    ) -> list[AtomRecall] | None:
        """
        关键词路。没跑成时往 `degraded` 追加一条并返回 None。

        ⚠️ 返回 None 与返回空列表是**两件事**：空列表 = 跑了没命中；
        None = 没跑（降级）。混成一个会让响应里出现假告警（每次没命中都
        报一次），而真出问题时又不显眼。
        """

        tsquery = tsquery_text(query)

        if tsquery is None:
            # 切不出词元（纯标点 / emoji / 空白）—— 这是**输入**的性质，
            # 不是后端故障，但对这条路来说结论一样：它没法参与。
            degraded.append(
                DegradedAtomRoute(
                    route=MemoryAtomRecallRoute.KEYWORD,
                    reason=(
                        "查询里没有可切分的词元，关键词路无法参与"
                        "（向量路不受影响）。"
                    ),
                )
            )
            return None

        try:
            return await self._atoms.recall_keyword(
                scope, tsquery=tsquery, limit=recall_limit,
            )

        except Exception as exc:  # noqa: BLE001 —— 降级，见方法 docstring
            logger.warning("[Memory] 记忆关键词路召回失败，降级为向量路：%s", exc)
            degraded.append(
                DegradedAtomRoute(
                    route=MemoryAtomRecallRoute.KEYWORD,
                    reason=f"关键词路召回失败：{exc}",
                )
            )
            return None

    async def _search_vector(
            self,
            query: str,
            scope: MemoryAtomQueryScope,
            recall_limit: int,
            degraded: list[DegradedAtomRoute],
    ) -> list[AtomRecall] | None:

        try:
            embedding = await self._embed_query(query)

        except Exception as exc:  # noqa: BLE001 —— 降级
            logger.warning("[Memory] 记忆查询向量化失败，降级为关键词路：%s", exc)
            degraded.append(
                DegradedAtomRoute(
                    route=MemoryAtomRecallRoute.VECTOR,
                    reason=f"查询向量化失败（embedder 不可用？）：{exc}",
                )
            )
            return None

        try:
            recalls = await self._atoms.recall_vector(
                scope, embedding=embedding, limit=recall_limit,
            )

        except Exception as exc:  # noqa: BLE001 —— 降级
            logger.warning("[Memory] 记忆向量路召回失败，降级为关键词路：%s", exc)
            degraded.append(
                DegradedAtomRoute(
                    route=MemoryAtomRecallRoute.VECTOR,
                    reason=f"向量路召回失败：{exc}",
                )
            )
            return None

        # ⚠️ 相似度下限在 Python 侧过滤，不塞进 SQL
        # -------------------------------------
        # 与 L0 同一条取舍：`WHERE embedding <=> :q < :t` 会让优化器认不出
        # HNSW 可用的形状（`ORDER BY <=> LIMIT n`），退化成顺序扫描 ——
        # 结果一样、只是慢。代价是“多取了一些再丢掉”，而 recall_limit 很小。
        #
        # 没有这个下限的话，向量路**永远**返回 recall_limit 条（相似度 0 的
        # 也会被排到第 20 名），而 RRF 按名次给分，第 2 名的无关记忆足够
        # 挤进结果里 —— 对模型来说那是纯噪声。
        #
        # 返回**空列表**是正当结果（这一路跑了，库里确实没有相似的），
        # 不进 degraded_routes。
        return [
            recall for recall in recalls
            if recall.score >= VECTOR_MIN_SIMILARITY
        ]

    # ========================================================
    # 模型①：抽取
    # ========================================================

    async def _extract(
            self,
            *,
            messages: list[AgentMessage],
            session: AgentSession,
            cursor: Any,
    ) -> list[ExtractedScene]:

        runtime = get_model_registry().get_runtime(
            agent_type=AGENT_TYPE,
            model_role=ModelRole.MEMORY_ATOM_EXTRACT,
        )

        response = await runtime.ainvoke(
            self._build_extract_request(
                messages=messages,
                previous_scene=cursor.last_scene_name,
            )
        )

        return parse_extraction(llm_message_text(response.message))

    def _build_extract_request(
            self,
            *,
            messages: list[AgentMessage],
            previous_scene: str,
    ) -> ModelRequest:
        """
        提示词 + 这次要分析的消息 -> 一次模型请求。

        单独成方法是为了**可测**：断言”喂进去的长什么样“不必真的调模型
        （与 compaction 的 build_request 同一条理由）。

        ⚠️ 这里的职责只有"取"
        -----------------
        系统提示词按 model_role 查（那是配置驱动的，所以走 provider）；
        用户消息交给 `memory_atom_extract.render_user_message` —— 那一半
        的**排版**是提示词的一部分（表头那两句限定就是提示词的正文，
        只是位置在用户消息里），归提示词模块。见 prompts/__init__.py 的
        约定。

        本方法因此只回答两件事：这次要分析**哪些**消息、**上一个情境**
        是什么。至于"【背景对话】刻意没接"那件事的来龙去脉，写在
        `memory_atom_extract.py` 里 `render_user_message` 的上方 ——
        因为它要接的话，落点在那里（多一个 `background` 参数）。
        """

        system_prompt = get_prompt_provider().node_prompt(
            ModelRole.MEMORY_ATOM_EXTRACT
        )

        # 用户消息**怎么排版**归提示词模块（见 prompts/__init__.py 的约定：
        # 静态文本与动态块都是提示词的一部分）。这里只负责**取**这一批消息
        # 与上一个情境 —— 那两样是服务知道的（它按水位读的库）。
        human = render_extract_message(
            messages=messages,
            previous_scene=previous_scene,
        )

        return ModelRequest(
            messages=(
                SystemMessage(content=system_prompt),
                HumanMessage(content=human),
            ),
            # ⚠️ 这里**曾经**关过思考模式（`disable_thinking=True`），
            # 2026-10-10 撤回 —— 留着这条记录，别再加回来
            # ------------------------------------------------------------
            # 加它的理由是**截断**：`max_tokens` 与思考共用，输出写到一半被
            # 切（`Unterminated string`）。但同一轮还做了另一件事 ——
            # 提示词的输出里**去掉了情境级的 `message_ids`**，而那一大串
            # UUID 正是输出的大头。**C 已经把截断的根因拿掉了**，于是 A 的
            # 收益不再需要，而它的代价开始显形：
            #
            # 关掉思考之后，模型在"这次到底有没有值得记的东西"上判得更粗 ——
            # 实测连续两轮（3d53e96a / 59ffe189）明明有可记的内容（AI 转述了
            # 用户的事实、而这按"提取主体＝用户或 AI"是**该抽**的）却抽了 0 条。
            # 而**漏抽是静默的**（库里少一条，没有任何报错），比"偶尔截断一次、
            # 重试救回来"坏得多。
            #
            # 所以要再开这个开关，先确认它换来的东西大于它压掉的抽取质量。
            metadata={
                "purpose": "memory_atom_extract",
                "agent_type": AGENT_TYPE,
                "model_role": ModelRole.MEMORY_ATOM_EXTRACT,
                "message_count": len(messages),
                "previous_scene": previous_scene,
            },
        )

    # ========================================================
    # 模型②：冲突检测 + 落库
    # ========================================================

    async def _dedup_and_write(
            self,
            atoms: list[MemoryAtom],
            *,
            session_id: UUID,
            user_id: str | None,
    ) -> "_Report":

        if not atoms:
            return _Report()

        # ---- 1. 候选池 --------------------------------------------
        #
        # ⚠️ 按**会话**收窄 —— 这是刻意的，别改成用户级
        # --------------------------------------------
        # 原规格（腾讯那版）就是这么做的：冲突检测只在**一段对话内部**比较。
        #
        # ⚠️ 代价说清楚：跨会话的同一事实**不会**被合并
        # --------------------------------------
        # 同一个用户在 5 个会话里各说一次"今晚去爬大蜀山"，就会留下 5 条
        # 记忆（2026-10-10 实测到的就是这 5 条）。而检索那条路是
        # `scope=user`（L1 事实本来就要跨会话找得到），于是一次召回 5 条
        # 重复 —— 主模型会如实汇报"这条记录在多处重复出现"。
        #
        # 也就是说"重复"是这个取舍的**必然结果**，不是 bug。真要合并跨会话
        # 的重复，那是另一条改动（池子放到用户级），但那就不是原规格的行为了。
        pool, related = await self._candidate_pool(
            atoms, session_id=session_id, user_id=user_id,
        )

        report = _Report()
        report.pool_size = len(pool)

        # ---- 2. 判定（没有候选池时不调模型）-------------------------
        #
        # 池子空 = 这个会话还没有任何已有记忆 —— 没有可比的东西，每条都
        # 只能是 store。此时调模型是纯浪费（而且提示词里那句”批量比较
        # 【新记忆】与【候选池】"会拿到一个空池）。
        decisions = (
            await self._conflict(atoms, pool, related) if pool else {}
        )

        # ---- 3. 应用判定 -> 要写的、要删的 --------------------------
        to_write, to_delete, applied = self._apply(atoms, decisions, pool)

        # `_apply` 自己造一份账；池子大小是这一步之外的信息，带过去。
        applied.pool_size = report.pool_size
        report = applied

        # ---- 4. 落库 ----------------------------------------------
        #
        # ⚠️ 先删后写：删的是“被吃掉的那几条”，写的是“存活 + 新增”。
        # 两者按构造不相交（`_apply` 把存活 id 从删除集里摘掉了），所以
        # 顺序不影响结果 —— 挑这个顺序只是与数据层级一致（先清旧、
        # 再立新）。同一个事务，中间失败全回滚。
        if to_delete:
            report.deleted = await self._atoms.delete_batch(
                sorted(to_delete), session_id=session_id,
            )

        if to_write:

            # 分词与向量化都在**最终内容**上做 —— 合并改写过的内容必须
            # 重新过一遍，否则索引里留的是旧正文（症状：搜到的还是旧内容，
            # 不报错）。切不出词元的整条丢掉：content_tokens 有 `<> ''`
            # 的 CHECK，放过去会在写库那一刻炸，离原因很远。
            indexable = [
                (atom, index_text(atom.content))
                for atom in to_write
            ]
            indexable = [
                (atom, tokens) for atom, tokens in indexable if tokens
            ]

            dropped = len(to_write) - len(indexable)

            if dropped:
                logger.warning(
                    "[Memory] 丢弃 %d 条切不出词元的记忆（%s）。",
                    dropped, session_id,
                )

            if indexable:

                embeddings = await self._embed_documents(
                    [atom.content for atom, _ in indexable]
                )

                await self._atoms.upsert_atoms(
                    [atom for atom, _ in indexable]
                )

                await self._atoms.upsert_fts([
                    AtomFtsRow(
                        record_id=atom.record_id,
                        session_id=atom.session_id,
                        business_agent_id=atom.business_agent_id,
                        user_id=atom.user_id,
                        org_id=atom.org_id,
                        tokens=tokens,
                        original=atom.content,
                    )
                    for atom, tokens in indexable
                ])

                await self._atoms.upsert_vec([
                    AtomVecRow(
                        record_id=atom.record_id,
                        session_id=atom.session_id,
                        business_agent_id=atom.business_agent_id,
                        user_id=atom.user_id,
                        org_id=atom.org_id,
                        embedding=list(embedding),
                    )
                    for (atom, _), embedding in zip(indexable, embeddings)
                ])

        return report

    async def _candidate_pool(
            self,
            atoms: list[MemoryAtom],
            *,
            session_id: UUID,
            user_id: str | None,
    ) -> tuple[dict[UUID, MemoryAtom], dict[UUID, list[UUID]]]:
        """
        为新记忆召回候选，返回 `(并集池, 每条新记忆各自的候选 id)`。

        ⚠️ 两样都要返回，不能只给并集
        ------------------------
        提示词给模型的输入里有**每条新记忆的"关联候选 ID"**（原项目
        `formatBatchConflictPrompt` 的 `perMemoryCandidateIds`）。只给并集
        时模型得自己猜"这一池子里哪几条跟这条新记忆有关"，而那个对应关系
        在这里**本来就算出来了** —— 丢掉它等于让模型拿并集当搜索空间。

        ⚠️ 池子是**并集**，不是"每条新记忆一个池"
        ------------------------------------
        提示词要的是"多条新记忆 + 统一候选池"，因为两条新记忆之间也可能
        重复，而分别看池子就发现不了。`related` 是**附加信息**，不替代并集。

        ⚠️ top_k 是**每条新记忆各自**的上限（这一处曾经是坏的）
        ---------------------------------------------
        融合分降序，所以一条新记忆取到前 `top_k` 个就可以停 —— 后面的都是
        更不相关的。原先的实现把它写成了

            bucket = scores.setdefault(recall.record_id, set())
            if len(bucket) >= DEFAULT_CONFLICT_TOP_K: continue

        `scores` 按**候选**的 id 建键、值是 `{"seen"}`，所以 `len(bucket)`
        恒为 1，**那个上限一次都没生效过** —— 池子实际是"所有新记忆召回
        到的全部候选的并集"，与注释里说的相反。现在改成按新记忆分组计数，
        截断才是真的。

        代价说清楚：这会让池子**变小**（每条最多 top_k 个），也就是"更不
        容易漏合并"与"提示词更短"之间的取舍变了 —— 往"提示词短"那侧挪。
        要换回大池子就调 `DEFAULT_CONFLICT_TOP_K`，别把这段改回去。

        ⚠️ 召回时多取（top_k + 本批条数）
        -----------------------------
        多取是为了给"本批自己"占掉的名额留出空间 —— 新记忆的向量/词元与
        已有记忆接近时，会把候选里本该给真正冲突项的位置占掉。下面还有
        一次"排除本批 id"的过滤，两步合起来才是完整的那句。
        """

        new_ids = {atom.record_id for atom in atoms}

        # 会话级（原规格）。跨会话的重复**不合并** —— 见 _dedup_and_write
        # 里那段关于代价的说明。
        #
        # ⚠️ `user_id` **必须一起带上**，哪怕粒度是会话
        # ---------------------------------------
        # `_conditions` 在 `user_id` 为 None 时追加的是 `user_id IS NULL`
        # （那是**读**路径的防御：没有身份只看得到无归属的存量行）。而这里
        # 的身份是已知的 —— 不带就等于把池子过滤成"只含无归属行"，而真实
        # 原子都带 user_id，于是一条都召不回来、池子恒空、每条都判 store。
        #
        # 这个坑实际发生过：召回改成走 `_conditions` 之后池子一直是空的，
        # 而症状是"**没有报错，只是永远不合并**"。会话粒度与用户身份是两件
        # 事，两件都要给。
        scope = MemoryAtomQueryScope(
            session_id=session_id, user_id=user_id,
        )

        recall_top_k = DEFAULT_CONFLICT_TOP_K + len(atoms)

        related: dict[UUID, list[UUID]] = {}

        union: list[UUID] = []
        union_seen: set[UUID] = set()

        for atom in atoms:

            rankings: dict[str, list[AtomRecall]] = {}

            tsquery = tsquery_text(atom.content)

            if tsquery is not None:
                rankings["keyword"] = await self._atoms.recall_keyword(
                    scope, tsquery=tsquery, limit=recall_top_k,
                )

            try:
                embedding = await self._embed_query(atom.content)

            except Exception as exc:  # noqa: BLE001 —— 降级，见下
                # 向量路缺席不致命：关键词路召回到的仍是**真实可检索**的
                # 候选。但静默跳过会让“为什么没合并”永远查不出来，所以
                # 留一条 warning（与 L0 的 degraded_routes 同一立场）。
                logger.warning(
                    "[Memory] 去重候选的查询向量化失败，降级为关键词路：%s",
                    exc,
                )
                embedding = None

            if embedding is not None:
                rankings["vector"] = await self._atoms.recall_vector(
                    scope, embedding=embedding, limit=recall_top_k,
                )

            picked: list[UUID] = []

            for recall, routes, _ in fuse_by_rank(
                rankings, key=lambda item: item.record_id,
            ):

                if recall.record_id in new_ids:
                    # 本批自己：它们不在库里（还没写），也不该成为自己的
                    # 候选。
                    continue

                if len(picked) >= DEFAULT_CONFLICT_TOP_K:
                    # 这一条已经够名额了 —— 融合分降序，后面的只会更不相关。
                    break

                picked.append(recall.record_id)

                if recall.record_id not in union_seen:
                    union_seen.add(recall.record_id)
                    union.append(recall.record_id)

            related[atom.record_id] = picked

        if not union:
            return {}, {}

        # 回主表取正文 —— 提示词要看到内容才能判“是不是同一件事”。
        pool = await self._atoms.load(union)

        # ⚠️ `related` 要按**池子里真有的**过滤
        # ---------------------------------
        # `load` 取不到的那些是"召回之后、读正文之前被别的任务删掉/合并掉"
        # 的（并发下会发生）。留着它们会让提示词给出一个池子里根本没有的
        # id —— 模型于是要么找不到它（判不了），要么把它写进 target_ids
        # （服务侧会按"不在候选池里"丢掉，白给一次幻觉的机会）。
        related = {
            record_id: [value for value in values if value in pool]
            for record_id, values in related.items()
        }

        return pool, related

    async def _conflict(
            self,
            atoms: list[MemoryAtom],
            pool: dict[UUID, MemoryAtom],
            related: dict[UUID, list[UUID]],
    ) -> dict[str, ConflictDecision]:
        """
        一次批量冲突检测 -> `新记忆 record_id(字符串) -> 判定`。

        ⚠️ 返回的键是**字符串**（模型回显的形式），不是 UUID —— 服务侧不
        假设模型回显的一定是合法 UUID，匹配不上就当“这条没有判定”（按
        store 处理）。这与 `source_message_ids` 同一条纪律：模型给的 id
        只能当线索，不能当身份。
        """

        runtime = get_model_registry().get_runtime(
            agent_type=AGENT_TYPE_DEDUP,
            model_role=ModelRole.MEMORY_ATOM_DEDUP,
        )

        response = await runtime.ainvoke(
            self._build_conflict_request(atoms, pool, related)
        )

        decisions = parse_conflicts(llm_message_text(response.message))

        return {
            decision.record_id: decision
            for decision in decisions
            if decision.record_id
        }

    def _build_conflict_request(
            self,
            atoms: list[MemoryAtom],
            pool: dict[UUID, MemoryAtom],
            related: dict[UUID, list[UUID]],
    ) -> ModelRequest:
        """新记忆 + 候选池 -> 冲突检测的请求。与抽取那份同分工（可变部分走 HumanMessage）。"""

        system_prompt = get_prompt_provider().node_prompt(
            ModelRole.MEMORY_ATOM_DEDUP
        )

        # 与 `_build_extract_request` 同分工：排版归提示词模块，这里只给数据。
        human = render_dedup_message(
            atoms=atoms,
            pool=list(pool.values()),
            related=related,
        )

        return ModelRequest(
            messages=(
                SystemMessage(content=system_prompt),
                HumanMessage(content=human),
            ),
            # 与 `_build_extract_request` 一样**不关思考** —— 这一次比抽取更
            # 需要判断（决定 store / skip / update / merge），关掉的代价更大。
            metadata={
                "purpose": "memory_atom_dedup",
                "agent_type": AGENT_TYPE_DEDUP,
                "model_role": ModelRole.MEMORY_ATOM_DEDUP,
                "new_count": len(atoms),
                "pool_count": len(pool),
            },
        )

    # ========================================================
    # 应用判定
    # ========================================================

    def _apply(
            self,
            atoms: list[MemoryAtom],
            decisions: dict[str, ConflictDecision],
            pool: dict[UUID, MemoryAtom],
    ) -> tuple[list[MemoryAtom], set[UUID], "_Report"]:
        """
        判定 -> `(要写的实体, 要删的 id, 统计)`。

        ⚠️ update 与 merge 是**同一段机械操作**
        --------------------------------
        两者在提示词里是两种意图（"新的更好，覆盖“与”信息互补，合并“），
        但在原地化语义下落到代码上是同一件事：**取 target_ids 里的第一条
        当存活者，把合并后的内容写回它，其余删掉**。意图的差别只影响模型
        给出的 merged_content，不影响这里怎么落库。

        ⚠️ 看不懂的动作按 store 处理
        ------------------------
        宁可多留一条记忆，也不要在看不懂的动作上**删东西**。模型偶尔会
        发明一个动作（"replace"、"ignore"…），把它当 update 会删掉
        target，当 store 只是多留一条 —— 后者可逆。
        """

        report = _Report()

        written: list[MemoryAtom] = []
        survivors: dict[UUID, MemoryAtom] = {}
        doomed: set[UUID] = set()

        for atom in atoms:

            decision = decisions.get(str(atom.record_id))

            action = (decision.action if decision else "store").strip().lower()

            if action not in _CONFLICT_ACTIONS:
                logger.warning(
                    "[Memory] 模型给了一个看不懂的动作 %r，按 store 处理"
                    "（多留一条比误删好）。", action,
                )
                action = "store"

            if action == "skip":
                report.skipped += 1
                continue

            targets = self._resolve_targets(decision, pool)

            if action in ("update", "merge") and not targets:
                # 要合并但一个有效 target 都没有（模型幻觉了 id，或那条
                # 已经被别的任务删了）。退化成 store —— 与”看不懂的动作"
                # 同一条取舍：不删。
                logger.warning(
                    "[Memory] 判定为 %s 但没有有效的 target_ids，退化为 store。",
                    action,
                )
                action = "store"

            if action == "store":
                written.append(atom)
                report.stored += 1
                continue

            survivor_id = targets[0].record_id

            merged = self._merge(
                incoming=atom,
                decision=decision,
                targets=targets,
            )

            if merged is None:
                # 合并后的内容切不出词元：**整条判定作废**（不写、也不删）。
                # 删而无所立是最坏的结果 —— 那几条旧记忆没了，新的又写不
                # 进去。
                logger.warning(
                    "[Memory] 合并后的内容不可索引，放弃这次 %s（不写也不删）。",
                    action,
                )
                report.skipped += 1
                continue

            if survivor_id in survivors:
                # 同一批里两条新记忆都要更新同一条旧记忆。理论上不该发生
                # （模型看得到整批），真发生了就**留着先到的那个**：后来的
                # 那条已经基于同一份旧内容算过合并，再叠一次会把前一次的
                # 结果冲掉。
                logger.warning(
                    "[Memory] 同一批里有两条记忆要更新同一条 %s，保留先到的那次。",
                    survivor_id,
                )
                report.skipped += 1
                continue

            survivors[survivor_id] = merged
            doomed.update(target.record_id for target in targets[1:])

            if action == "merge":
                report.merged += 1
            else:
                report.updated += 1

        # 存活者从删除集里摘掉 —— 它被“更新”了，不是被删了。
        doomed -= set(survivors)
        # 新写入的 id 也不可能在删除集里（targets 只来自库里的候选池），
        # 但显式减一次，免得将来有人放宽 targets 的来源。
        doomed -= {atom.record_id for atom in written}
        doomed -= {atom.record_id for atom in survivors.values()}

        return written + list(survivors.values()), doomed, report

    @staticmethod
    def _resolve_targets(
            decision: ConflictDecision | None,
            pool: dict[UUID, MemoryAtom],
    ) -> list[MemoryAtom]:
        """
        `target_ids`（模型给的字符串）-> 池子里**真实存在的那几条实体**，
        按模型给的顺序。

        只认**候选池里有**的 id：模型可能给出一个它记错的、或别的会话的
        id。这里挡掉之后，`delete_batch` 的归属校验是第二道 —— 两道都要
        （这一道给出“少删一条”，那一道保证“绝不跨会话删”）。

        返回实体而不是 id：合并要用到它们的 version / 时间线 / 溯源，
        而那几样只有池子里的实体有（再查一次库就多一次往返，还要处理
        "查的时候已经被别人删了“）。
        """

        if decision is None:
            return []

        resolved: list[MemoryAtom] = []
        seen: set[UUID] = set()

        for raw in decision.target_ids:

            try:
                record_id = UUID(raw)

            except (ValueError, AttributeError, TypeError):
                logger.warning(
                    "[Memory] 模型给的 target_id %r 不是 UUID，忽略。", raw,
                )
                continue

            atom = pool.get(record_id)

            if atom is None:
                logger.warning(
                    "[Memory] 模型给的 target_id %s 不在候选池里，忽略。",
                    record_id,
                )
                continue

            if record_id in seen:
                continue

            seen.add(record_id)
            resolved.append(atom)

        return resolved

    @staticmethod
    def _merge(
            *,
            incoming: MemoryAtom,
            decision: ConflictDecision | None,
            targets: list[MemoryAtom],
    ) -> MemoryAtom | None:
        """
        `targets` 里的第一条当存活者，把合并结果写回它。内容不可索引时
        返回 None（整条判定作废）。

        ⚠️ 原地更新下，这些字段**必须显式决定**（不决定就是静默保留旧值）
        ---------------------------------------------------------
            content / type / priority   判定给的（缺省则用新记忆的）
            version                     `被合并目标里最大的 version + 1`
                                        —— 取**全部** target 的最大值，不是
                                        存活者自己的：合并吃掉的那几条也参与
                                        过这份记忆的历史
            timestamps                  提示词给的并集；落进 meta['timestamps']
                                        并算 start/end
            source_message_ids          新旧**并集**（合并后的记忆来自两处）
            scene_name                  **新记忆的** —— 合并发生在当下的情境里，
                                        而冲突检测的输出里根本没有 scene_name
                                        （不显式取新记忆的话这里会静默保留旧值）
            meta                        旧 + 新（新的覆盖同名字段）
        """

        survivor = targets[0]

        content = (
            decision.merged_content if decision else ""
        ).strip() or incoming.content

        if not index_text(content):
            return None

        # 提示词要求 merged_timestamps 是所有相关记忆时间戳的并集；我们另外
        # 把”存下来的那几个时间列“也并进去，缺了它们会让时间线在几轮合并
        # 之后慢慢丢掉最早的那一段（提示词只看得到它被告知的那几个）。
        trail = _timestamp_union(
            decision.merged_timestamps if decision else [],
            targets,
            incoming,
        )

        meta = {**survivor.meta, **incoming.meta}

        if trail:
            meta["timestamps"] = trail

        return survivor.model_copy(update={
            "content": content,
            "type": (decision.merged_type if decision else "").strip() or incoming.type,
            "priority": (
                decision.merged_priority
                if decision is not None and decision.merged_priority is not None
                else incoming.priority
            ),
            "scene_name": incoming.scene_name,
            "version": max(
                target.version for target in targets
            ) + 1,
            "timestamp_str": (trail[0] if trail else survivor.timestamp_str),
            "timestamp_start": (trail[0] if trail else survivor.timestamp_start),
            "timestamp_end": (trail[-1] if trail else survivor.timestamp_end),
            "source_message_ids": sorted(
                set(survivor.source_message_ids)
                | set(incoming.source_message_ids)
            ),
            "meta": meta,
            # 归属跟着新记忆走（同一次抽取，通常一模一样；真不一样时以
            # 新的为准 —— 记忆刚刚在这一次抽取里被重申）。
            "business_agent_id": incoming.business_agent_id,
            "user_id": incoming.user_id,
            "org_id": incoming.org_id,
        })

    # ========================================================
    # 产物组装
    # ========================================================

    def _to_atoms(
            self,
            scenes: list[ExtractedScene],
            *,
            session: AgentSession,
            messages: list[AgentMessage],
            duty_key: str | None,
    ) -> list[MemoryAtom]:
        """
        情境 + 记忆 -> 待落库的实体（**还没过词元与向量**，那在落库前做）。

        ⚠️ 模型给的 id 是**字符串**，可能是它自己编的
        ------------------------------------
        解析不出 UUID、或不在本批消息里的，直接丢掉 —— 留着一条指向不存在
        消息的溯源比没有更糟（它看起来像有据可查）。
        """

        known = {str(message.message_id) for message in messages}

        atoms: list[MemoryAtom] = []

        for scene in scenes:

            for memory in scene.memories:

                text = memory.content.strip()

                if not text:
                    continue

                meta = dict(memory.metadata)

                # ⚠️ 活动时间只从 metadata 里取，**不从时间戳列反推**
                # ------------------------------------------------
                # 那三个列存的是”这条记忆讲的事发生在什么时候“，由提示词
                # 从对话里推出来（可能是"2018 年”这种不能解析的半成品），
                # 所以它们是 TEXT。合并时的时间线并集在 `_merge` 里做。
                start = _as_text(meta.get("activity_start_time"))
                end = _as_text(meta.get("activity_end_time"))

                atoms.append(
                    MemoryAtom(
                        record_id=uuid4(),
                        content=text,
                        type=memory.type.strip(),
                        priority=memory.priority,
                        scene_name=scene.scene_name,
                        session_key=session.thread_id or "",
                        version=0,
                        timestamp_str=start or end,
                        timestamp_start=start,
                        timestamp_end=end,
                        source_message_ids=sorted({
                            mid
                            for mid in memory.source_message_ids
                            if mid in known and _is_uuid(mid)
                        }),
                        meta=meta,
                        session_id=session.session_id,
                        # 主 agent 的**职责**（duty_key）—— 这条记忆是
                        # 为那个职责而记的。取值见 `_duty_key` 的说明
                        # （取本批消息上最近的那个，退回会话的 agent_id）。
                        business_agent_id=duty_key,
                        user_id=session.user_id,
                        org_id=session.org_id,
                    )
                )

        return atoms

    @staticmethod
    def _next_scene(
            scenes: list[ExtractedScene],
            *,
            fallback: str,
    ) -> str:
        """
        下一次抽取要继承的情境。

        取**最后一个**情境 —— 这批消息讲完之后处在哪个情境，就是下一轮
        的”上一个情境“。模型一条情境都没给时保持原值（不让水位把情境
        清空）。
        """

        if not scenes:
            return fallback

        return scenes[-1].scene_name or fallback

    # ========================================================
    # internals
    # ========================================================

    def _dense_embedder(self):
        """
        取稠密 embedder。取不到就抛，由调用方决定降级还是失败
        （与 L0 的 `_dense_embedder` 同一取舍）。
        """

        return self._pipeline_registry.embedder_for(EmbeddingOutput.DENSE)

    async def _embed_query(self, text: str) -> list[float]:
        """
        查询向量 —— 走 `embed_query`（带指令前缀），不是 `embed`。

        ⚠️ 去重的候选召回是**检索**，所以用查询侧；写入索引用文档侧。
        搞反的表现是召回质量下降，**不报错**（与 L0 里那段说明同一条）。
        """

        embedder = self._dense_embedder()

        if embedder is None:
            raise RuntimeError("稠密 embedder 没有注册，向量路不可用。")

        return list(await embedder.embed_query(text, config={}))

    async def _embed_documents(
            self,
            texts: list[str],
    ) -> list[list[float]]:
        """批量向量化记忆正文（写入侧，不带指令前缀）。"""

        embedder = self._dense_embedder()

        if embedder is None:
            raise RuntimeError("稠密 embedder 没有注册，无法建立记忆向量索引。")

        vectors = await embedder.embed(texts, config={})

        return [list(vector) for vector in vectors]


# ============================================================
# 纯函数与中间结构
# ============================================================


class _Report:
    """一次抽取的账 —— 落日志用，也方便冒烟断言。"""

    __slots__ = ("deleted", "merged", "pool_size", "skipped", "stored", "updated")

    def __init__(self) -> None:
        self.stored = 0
        self.updated = 0
        self.merged = 0
        self.skipped = 0
        self.deleted = 0

        #: 去重的候选池有多大。**0 表示这一步没调模型**（池子空时按设计
        #: 跳过，见 `_dedup_and_write`）。
        #:
        #: 单独记它是因为"为什么没合并"最常见的答案是"池子是空的"，而那个
        #: 从"新 N / 合并 0"这两个数上看不出来 —— 2026-10-11 就是为了回答
        #: 这个问题加的。
        self.pool_size = 0


def _timestamp_union(
        merged: list[str],
        targets: list[MemoryAtom],
        incoming: MemoryAtom,
) -> list[str]:
    """
    时间线并集：提示词给的那一串 + 被合并各条存着的时间列 + 新记忆的，去重排序。

    ⚠️ 是**字符串排序**，不是时间排序
    ---------------------------
    提示词约定输出 ISO 8601，那个格式的字典序与时间序一致。混进别的格式
    （"2018 年”）时排序位置没有意义 —— 但它是**溯源与展示**用的一个串，
    不是排序键（检索从不按它排），所以这个代价可以接受；换成真时间类型
    反而会在写入那一刻因为无法解析而炸。
    """

    values = {
        value.strip()
        for value in [
            *merged,
            *[
                value
                for target in targets
                for value in (
                    target.timestamp_str,
                    target.timestamp_start,
                    target.timestamp_end,
                )
            ],
            incoming.timestamp_str,
            incoming.timestamp_start,
            incoming.timestamp_end,
        ]
        if value and value.strip()
    }

    return sorted(values)


def _duty_key(messages: list[AgentMessage]) -> str | None:
    """
    这批消息里**最近的那个** duty_key —— 记忆归属的主 agent 职责。

    ⚠️ 为什么取自消息，而不是会话或任务参数
    --------------------------------
    一条记忆是“主 agent 在履行某个职责时记住的”，而职责是**逐运行**的：
    `agent_run.duty_key` 在建运行那一刻解析一次，写消息时冗余到
    `agent_message.business_agent_id`。一批增量可能跨好几个运行，取最近的
    那个 = "这批消息讲完时在履职的那份职责“。

    会话上的 `agent_id` 是更粗的角色标识（本仓库里还没接线），所以只作
    兜底 —— 它在调用点做（`_duty_key(messages) or session.agent_id`）。

    ⚠️ 返回 None 是**允许**的状态，不是错误
    ---------------------------------
    存量消息的 `business_agent_id` 是 NULL（那一列加进来之前写下的行）。
    此时返回 None 让调用方去兜底，而不是在这里编一个默认职责 —— 那会让
    "这一列为什么全是 general"变成一件查不出来的事。
    """

    for message in reversed(messages):

        if message.business_agent_id:
            return message.business_agent_id

    return None


def _as_text(value: Any) -> str:
    """metadata 里的值 -> 字符串。None / 空 -> 空串。"""

    if value is None:
        return ""

    return str(value).strip()


def _is_uuid(value: str) -> bool:
    """这个字符串是不是一个能解析的 UUID。"""

    try:
        UUID(value)

    except (ValueError, AttributeError, TypeError):
        return False

    return True


