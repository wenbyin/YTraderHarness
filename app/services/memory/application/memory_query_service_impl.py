"""
MemoryQueryServiceImpl —— 管理端的只读查询：三张水位 + 一组浏览 + 一条溯源边。

五个仓储存的都是**别的服务的表**，各自归各自的契约：

    atoms      memory_atom / memory_atom_cursor      （L1 的归属）
    scenes     memory_scene / memory_scene_cursor    （L2 的归属）
    personas   memory_persona / memory_persona_cursor（L3 的归属）
    messages   agent_message                          （session 模块的归属）
    sessions   agent_session                          （同上）

⚠️ 一个写入都没有，也不自己拼 SQL 去 join 别的模块的表
-----------------------------------------------
水位的读法只有各自的仓储说了算（"没有行"折成什么默认值就是它们定义的），
消息的取法只有 session 模块说了算。本服务把它们拼起来，**不重述**它们的
规则 —— 重述一份的代价是两张口径慢慢分叉，而分叉不报错。

⚠️ 归属校验：走会话的可见性口径，不自己写一条
--------------------------------------
`_require_session` 用的是 `AgentSessionRepository.get(session_id,
user_id=...)` —— 同一个方法、同一条规则（会话的 user_id 等于它，或为
NULL 的存量语义）也服务着 L1 检索那条路。

代价说清楚：**存量会话（user_id 为空的那批）对任何 target 都通过校验**。
这是那条规则本身的样子，不是这里放松的 —— 要收紧就得改 session 模块的
可见性口径（那会让所有调用点一起变），而不是在管理端加一条平行的判断。

⚠️ 分页的游标只在这一层编解码
---------------------
仓储收的是**解出来的排序键**（`after=(值, 唯一键)`），返回的是 `has_more`；
"怎么变成那个不透明串"住在这里（`_pagination.py`）。于是仓储不必知道
契约长什么样，而契约也不必知道某个 order 的 SQL 怎么写。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.agent.prompt.prompts.memory_retrieval import (
    CONVERSATIONS_TITLE,
    FACTS_TITLE,
    PERSONA_TITLE,
    SCENES_TITLE,
    SYSTEM_PROMPT,
    render_user_message,
)
from app.agent.prompt.prompts.memory_scene_consolidate import parse_meta
from app.common.exceptions import NotFoundError
from app.common.factories import get_service_factory
from app.common.service.base import ServiceContext
from app.common.tool_caller import current_tool_caller

from ..domain.entities.persona import PersonaRow
from ..domain.entities.scene import PERSONA_FILENAME, SceneRow, SceneScope
from ..domain.schemas.memory import (
    MemorySearchRequest,
    MemorySearchScope,
)
from ..domain.schemas.memory_scene import MemorySceneGetRequest
from ..domain.schemas.memory_atom import (
    MemoryAtomSearchRequest,
    MemoryAtomSearchScope,
)
from ..domain.schemas.query import (
    ConversationHit,
    ConversationHitSection,
    ConversationRecord,
    DegradedHitRoute,
    DegradedLayer,
    FactHit,
    FactHitSection,
    FactRecord,
    L0AtomsRequest,
    L0AtomsResponse,
    L0QueryRequest,
    L0QueryResponse,
    L1CursorRecord,
    L1ExtractionRequest,
    L1ExtractionResponse,
    L1QueryRequest,
    L1QueryResponse,
    L1SourcesRequest,
    L1SourcesResponse,
    L2CursorRecord,
    L2DetailRequest,
    L2DetailResponse,
    L2ExtractionRequest,
    L2ExtractionResponse,
    L2QueryRequest,
    L2QueryResponse,
    L3CursorRecord,
    L3ExtractionRequest,
    L3ExtractionResponse,
    L3QueryResponse,
    MemoryRetrievalPrompt,
    MemoryRetrievalPromptText,
    MemoryRetrievalRequest,
    Pagination,
    PersonaRecord,
    PersonaSection,
    SceneDetail,
    SceneSection,
    SceneSummary,
    TargetRequest,
)
from ..repository.atom import MemoryAtomRepository
from ..repository.persona import MemoryPersonaRepository
from ..repository.scene import MemorySceneRepository
from app.services.session.repository.agent_session import (
    AgentSessionRepository,
)
from app.services.session.repository.message import MessageRepository
from app.services.tool.registry import tool

from ._assembly import message_text
from ._pagination import decode_cursor, encode_cursor
from .memory_atom_service import MemoryAtomService
from .memory_query_service import MemoryQueryService
from .memory_service import MemoryService

__all__ = ["MemoryQueryServiceImpl"]

logger = logging.getLogger(__name__)


#: 团队 scope 下 L0/L1 开不起来的原因（进 `degraded`，不是静默少两段）。
#:
#: 与 `RetrievalTarget.require_personal_scope` 那条 400 说的是**同一件事**，
#: 只是响应方式不同：那里的整个请求就是那一层（没有"部分成功"可言）；
#: 这里是汇总端点，其余几层查得到，所以是**降级 + 说明**。
_TEAM_RECALL_REASON = (
    "L0/L1 的召回过滤维度只有 session / user（agent_message 没有 org 列、"
    "memory_atom 的 org_id 不参与过滤），团队 scope 下查不了。"
    "要按团队看这两层得先落一套团队级过滤。"
)


#: `created_at_*` 那两个 order 的游标键类型：(created_at, message_id)。
_CREATED_AT_KEY = ("datetime", "uuid")

#: `session_seq_*` 那两个 order 的游标键类型：(session_id, seq)。
_SESSION_SEQ_KEY = ("uuid", "int")


class MemoryQueryServiceImpl(MemoryQueryService):

    def __init__(
            self,
            *,
            atoms: MemoryAtomRepository,
            scenes: MemorySceneRepository,
            personas: MemoryPersonaRepository,
            messages: MessageRepository,
            sessions: AgentSessionRepository,
            l0_recall: MemoryService | None = None,
            l1_recall: MemoryAtomService | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._atoms = atoms
        self._scenes = scenes
        self._personas = personas
        self._messages = messages
        self._sessions = sessions

        # ⚠️ 两个召回服务**默认为空**，用到时才现取
        # ---------------------------------------------
        # 它们各自要注入 embedder（向量路那个进程级预热、几 GB 的东西），
        # 而本服务绝大多数方法（水位、浏览、溯源）根本不召回。做成构造参数
        # 就等于让每个管理端查询都拖上模型栈 —— 与本服务"纯 SQL、不注入
        # embedder"那条设计冲突（见契约的模块 docstring）。
        #
        # 默认 None 时按 `factory.scope(..., begin=False)` 现取（用完即还）
        # —— 与 trigger.py 取 TaskService 同一条手法。**显式传进来**是给
        # 测试与将来别处的装配用的（见 `_recall_service`）。
        self._l0_recall = l0_recall
        self._l1_recall = l1_recall

    # ========================================================
    # 上下文预览
    # ========================================================

    async def retrieval_preview(
            self,
            request: MemoryRetrievalRequest,
    ) -> MemoryRetrievalPrompt:

        scope = request.target.to_scope()

        sections: list[PersonaSection | SceneSection] = []

        degraded: list[DegradedLayer] = []

        # 渲染用的原始材料。与 `sections` 由**同一次读取**产出 —— 分两次
        # 查会在两次之间有人写入时给出"结构里有、文本里没有"的预览，而
        # 没人会去核。
        persona_row: PersonaRow | None = None
        scene_rows: list[SceneRow] = []
        scenes_total = 0
        scene_items: list[SceneSummary] = []

        # ---- L3（persona -> scenes 的顺序就是 prompt 的顺序）---------
        if request.retrieve.l3:

            try:
                persona_row = await self._personas.get_persona(scope)

            except Exception as exc:  # noqa: BLE001 —— 部分降级，见契约
                degraded.append(DegradedLayer(layer="l3", reason=str(exc)))
                logger.warning(
                    "[Memory] 上下文预览：L3 读取失败（scope=%s）：%s",
                    scope.key, exc,
                )

            else:
                items = (
                    []
                    if persona_row is None
                    else [
                        PersonaRecord(
                            content=persona_row.content,
                            filename=PERSONA_FILENAME,
                            version=persona_row.version,
                            updated_at=persona_row.updated_at,
                        )
                    ]
                )

                # ⚠️ 没有画像时**照样给一个空 section**（不是省掉）
                # -----------------------------------------
                # `degraded` 与"空"必须分得开：省掉的话，"L3 查过但没有
                # 画像"与"L3 没查"在响应里长得一样。
                sections.append(
                    PersonaSection(
                        block="persona",
                        layer="l3",
                        title=PERSONA_TITLE,
                        items=items,
                        returned=len(items),
                        truncated=False,
                    )
                )

        # ---- L2（只用索引字段，不给正文）----------------------------
        if request.retrieve.l2:

            try:
                scene_rows = await self._scenes.list_scenes(scope)

            except Exception as exc:  # noqa: BLE001 —— 部分降级，见契约
                degraded.append(DegradedLayer(layer="l2", reason=str(exc)))
                logger.warning(
                    "[Memory] 上下文预览：L2 读取失败（scope=%s）：%s",
                    scope.key, exc,
                )

            else:
                # `list_scenes` 已按 filename 升序（那是给提示词缓存用的稳定
                # 顺序），这里只做截断 —— **不重排**。
                scenes_total = len(scene_rows)
                scene_items = [
                    self._scene_summary(row)
                    for row in scene_rows[:request.limits.l2]
                ]

                sections.append(
                    SceneSection(
                        block="scenes",
                        layer="l2",
                        title=SCENES_TITLE,
                        items=scene_items,
                        returned=len(scene_items),
                        # 总数与实际给的条数**同源**（同一个 scene_rows），
                        # 见上面那段说明：分两次读会让提示词里的计数与清单
                        # 条数对不上，而没人会去核。
                        scenes_total=scenes_total,
                        truncated=scenes_total > len(scene_items),
                    )
                )

        # ---- L0 / L1（**真召回**，只在开了的时候跑）------------------
        #
        # 放在 persona / scenes **之后**：契约里 sections 的顺序就是 prompt
        # 的顺序，而那两段是每轮都带的（默认开），这两段是按需的。
        l0_items: list[ConversationHit] = []
        l1_items: list[FactHit] = []

        if request.retrieve.l0 or request.retrieve.l1:

            # ⚠️ 团队 scope：**不整单失败**，两层各自进 degraded
            # --------------------------------------------------
            # 这两层没有团队维度（见 `_TEAM_RECALL_REASON`）。汇总端点里
            # 其余几层是查得到的，所以不能像单层端点那样 400 —— 但也不能
            # 静默少两段。
            if request.target.is_team:

                for layer, wanted in (
                    ("l0", request.retrieve.l0),
                    ("l1", request.retrieve.l1),
                ):
                    if wanted:
                        degraded.append(
                            DegradedLayer(
                                layer=layer,
                                reason=_TEAM_RECALL_REASON,
                            )
                        )

            else:
                if request.retrieve.l0:
                    l0_items = await self._recall_l0(
                        request,
                        user_id=scope.user_id or "",
                        degraded=degraded,
                        sections=sections,
                    )

                if request.retrieve.l1:
                    l1_items = await self._recall_l1(
                        request,
                        user_id=scope.user_id or "",
                        degraded=degraded,
                        sections=sections,
                    )

        return MemoryRetrievalPrompt(
            retrieval_id=uuid4(),
            input=request.input,
            target=request.target,
            scope_kind=request.target.scope_kind,
            sections=sections,
            degraded=degraded,
            prompt=MemoryRetrievalPromptText(
                # 空串 -> null：提示词模块里那份常量是**占位**（正文还没写），
                # 契约说"尚未实现时为 null"。空串留着会看起来像"实现了但
                # 内容为空"。
                system=SYSTEM_PROMPT or None,
                user=render_user_message(
                    persona=persona_row,
                    scenes=scene_items,
                    scenes_total=scenes_total,
                    conversations=l0_items,
                    facts=l1_items,
                ),
            ),
            generated_at=datetime.now(timezone.utc),
        )

    # ========================================================
    # 上下文预览：两层的真召回
    # ========================================================

    async def _recall_l0(
            self,
            request: MemoryRetrievalRequest,
            *,
            user_id: str,
            degraded: list[DegradedLayer],
            sections: list[Any],
    ) -> list[ConversationHit]:
        """
        跑一轮 L0 混合召回，并把结果挂成一段。失败只进 `degraded`（返回空）。

        ⚠️ 身份口径与工具那条**逐字一致**
        ------------------------------
        走 `MemoryService.search`（HTTP 与工具共用的那个方法），身份是显式
        入参的 `user_id`。**不另写一条检索**：两条实现一旦分叉，预览就
        变成"给你看一个 Agent 其实拿不到的结果"。

        ⚠️ scope 由 `request.session_id` 决定
        -----------------------------------
        给了就收窄到那一个会话（`scope=session`），没给就跨会话
        （`scope=user`）—— 这是**同一个开关的两个取值**，不是两条实现。
        服务侧对这两种组合都有校验（`_resolve_scope`），因此这里传的
        `session_id` 必须与 `scope` 一致：给 None 时不能是 session、
        给值时不能是 user。
        """

        session_id = request.session_id

        try:
            async with self._recall_service(
                    MemoryService, self._l0_recall,
            ) as service:

                response = await service.search(
                    MemorySearchRequest(
                        query=request.input,
                        scope=(
                            MemorySearchScope.SESSION
                            if session_id is not None
                            else MemorySearchScope.USER
                        ),
                        limit=request.limits.l0,
                    ),
                    session_id=session_id,
                    user_id=user_id,
                )

        except Exception as exc:  # noqa: BLE001 —— 部分降级，见契约
            degraded.append(DegradedLayer(layer="l0", reason=str(exc)))
            logger.warning(
                "[Memory] 上下文预览：L0 召回失败（user=%s）：%s",
                user_id, exc,
            )
            return []

        items = [
            ConversationHit(
                message_id=hit.message_id,
                session_id=hit.session_id,
                seq=hit.seq,
                role=hit.role,
                text=hit.text,
                created_at=hit.created_at,
                score=hit.score,
                matched_by=[route.value for route in hit.matched_by],
            )
            for hit in response.results
        ]

        sections.append(
            ConversationHitSection(
                block="conversations",
                layer="l0",
                title=CONVERSATIONS_TITLE,
                items=items,
                returned=len(items),
                # 层内某一路挂了（比如向量后端不可用）**照样给另一路的结果**
                # —— 与顶层 degraded（整层没查成）分开，见 DegradedHitRoute。
                degraded_routes=[
                    DegradedHitRoute(
                        route=route.route.value,
                        reason=route.reason,
                    )
                    for route in response.degraded_routes
                ],
            )
        )

        return items

    async def _recall_l1(
            self,
            request: MemoryRetrievalRequest,
            *,
            user_id: str,
            degraded: list[DegradedLayer],
            sections: list[Any],
    ) -> list[FactHit]:
        """跑一轮 L1 混合召回。与 `_recall_l0` 同形，见那里的说明。"""

        try:
            async with self._recall_service(
                    MemoryAtomService, self._l1_recall,
            ) as service:

                response = await service.search(
                    MemoryAtomSearchRequest(
                        query=request.input,
                        scope=MemoryAtomSearchScope.USER,
                        limit=request.limits.l1,
                    ),
                    user_id=user_id,
                )

        except Exception as exc:  # noqa: BLE001 —— 部分降级，见契约
            degraded.append(DegradedLayer(layer="l1", reason=str(exc)))
            logger.warning(
                "[Memory] 上下文预览：L1 召回失败（user=%s）：%s",
                user_id, exc,
            )
            return []

        items = [
            FactHit(
                record_id=hit.record_id,
                content=hit.content,
                type=hit.type,
                priority=hit.priority,
                scene_name=hit.scene_name,
                score=hit.score,
                matched_by=[route.value for route in hit.matched_by],
                # ⚠️ 这里就是那个**改名点**：源自 `memory_atom.created_time`
                # / `updated_time`，本段统一成 `*_at`（理由见 FactHit 的
                # 字段注释）。改名只有这一处，改动它要连着看那两处。
                created_at=hit.created_time,
                updated_at=hit.updated_time,
            )
            for hit in response.results
        ]

        sections.append(
            FactHitSection(
                block="facts",
                layer="l1",
                title=FACTS_TITLE,
                items=items,
                returned=len(items),
                degraded_routes=[
                    DegradedHitRoute(
                        route=route.route.value,
                        reason=route.reason,
                    )
                    for route in response.degraded_routes
                ],
            )
        )

        return items

    @asynccontextmanager
    async def _recall_service(
            self,
            contract: type,
            override: Any | None,
    ) -> Any:
        """
        取一个召回服务：显式传进来的优先，否则**现取现还**。

        ⚠️ `begin=False`：这是一次只读预览
        ------------------------------
        不显式开事务（与只读的仓储作用域同一条口径）。召回只是几条 SELECT。

        ⚠️ 它会**另开一段 session**
        -----------------------
        本方法跑在请求自己那段作用域里（`Service[MemoryQueryService]` 开的），
        而 `factory.scope` 再开一段。两段都是只读快照，没有写入需要跨它们
        保持一致 —— 代价是多占一个连接，换来的是"召回逻辑只有一份"
        （不必把 embedder 拖进本服务，也不必抄一遍融合）。
        """

        if override is not None:
            yield override
            return

        factory = get_service_factory()

        async with factory.scope(contract, begin=False) as service:
            yield service

    # ========================================================
    # 提取进度水位
    # ========================================================

    async def l1_extraction(
            self,
            request: L1ExtractionRequest,
    ) -> L1ExtractionResponse:

        scope = request.target.require_personal_scope()

        await self._require_session(request.session_id, scope)

        cursor = await self._atoms.get_cursor(request.session_id)

        return L1ExtractionResponse(
            scope_kind=request.target.scope_kind,
            session_id=request.session_id,
            cursor=L1CursorRecord(
                last_seq=cursor.last_seq,
                last_scene_name=cursor.last_scene_name,
            ),
        )

    async def l2_extraction(
            self,
            request: L2ExtractionRequest,
    ) -> L2ExtractionResponse:

        scope = request.target.require_personal_scope()

        await self._require_session(request.session_id, scope)

        cursor = await self._scenes.get_l2_cursor(request.session_id)

        return L2ExtractionResponse(
            scope_kind=request.target.scope_kind,
            session_id=request.session_id,
            cursor=L2CursorRecord(
                last_atom_updated_time=cursor.last_atom_updated_time,
                last_atom_ids=list(cursor.last_atom_ids),
                last_l2_at=cursor.last_l2_at,
            ),
        )

    async def l3_extraction(
            self,
            request: L3ExtractionRequest,
    ) -> L3ExtractionResponse:

        # ⚠️ 这里**不做**会话校验（也没得做）：L3 的水位是 scope 级的，
        # 请求里没有会话。target 本身就是完整的定位条件 —— 与 L1/L2 不同，
        # 那两条要先证明"这个会话属于这个 target"。
        scope = request.target.to_scope()

        cursor = await self._personas.get_l3_cursor(scope)

        return L3ExtractionResponse(
            scope_kind=request.target.scope_kind,
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            team_id=scope.org_id,
            cursor=L3CursorRecord(
                last_scene_updated_time=cursor.last_scene_updated_time,
                last_scene_ids=list(cursor.last_scene_ids),
                last_l3_at=cursor.last_l3_at,
            ),
        )

    # ========================================================
    # 浏览
    # ========================================================

    async def l0_records(
            self,
            request: L0QueryRequest,
    ) -> L0QueryResponse:

        scope = request.target.require_personal_scope()

        # 给了会话就先证明它属于这个 target（拿不到 -> 404）。放在查询**之前**：
        # 否则一次跨 target 的调用会先读到别人的消息，再在别处才发现不该读。
        if request.session_id is not None:
            await self._require_session(request.session_id, scope)

        by_created_at = request.order.startswith("created_at")

        after = (
            decode_cursor(
                request.cursor,
                order=request.order,
                kinds=_CREATED_AT_KEY if by_created_at else _SESSION_SEQ_KEY,
            )
            if request.cursor
            else None
        )

        rows, has_more = await self._messages.browse(
            user_id=scope.user_id or "",
            session_id=request.session_id,
            # 枚举 -> 值：`MessageRole` 是 str 的子类，直接传也能比对，
            # 但显式取 value 让"传给 SQL 的是什么类型"不必靠继承关系推断。
            roles=[role.value for role in request.roles] if request.roles else None,
            created_from=request.time_from,
            created_to=request.time_to,
            text_contains=request.text_contains,
            order=request.order,
            limit=request.limit,
            after=after,
        )

        return L0QueryResponse(
            scope_kind=request.target.scope_kind,
            items=[
                ConversationRecord(
                    message_id=message.message_id,
                    session_id=message.session_id,
                    seq=message.seq,
                    role=message.role,
                    # 与 L0 索引、与溯源端点同一个口径（只拼 text 块）。
                    # 纯工具调用的消息没有 text 块 -> 空串（契约里 text 必填）。
                    text=message_text(message) or "",
                    created_at=message.created_at,
                )
                for message in rows
            ],
            page=self._page(
                request.limit,
                has_more,
                order=request.order,
                last=rows[-1] if rows and has_more else None,
                key=lambda message: (
                    (message.created_at, message.message_id)
                    if by_created_at
                    else (message.session_id, message.seq)
                ),
            ),
        )

    async def l1_records(
            self,
            request: L1QueryRequest,
    ) -> L1QueryResponse:

        scope = request.target.require_personal_scope()

        by_updated_at = request.order.startswith("updated_at")

        after = (
            decode_cursor(
                request.cursor,
                order=request.order,
                kinds=("datetime", "uuid") if by_updated_at else ("int", "uuid"),
            )
            if request.cursor
            else None
        )

        rows, has_more = await self._atoms.list_page(
            user_id=scope.user_id or "",
            record_type=request.type,
            priority=request.priority,
            scene_name=request.scene_name,
            updated_from=request.updated_from,
            updated_to=request.updated_to,
            text_contains=request.text_contains,
            order=request.order,
            limit=request.limit,
            after=after,
        )

        return L1QueryResponse(
            scope_kind=request.target.scope_kind,
            items=[self._fact(atom) for atom in rows],
            page=self._page(
                request.limit,
                has_more,
                order=request.order,
                last=rows[-1] if rows and has_more else None,
                key=lambda atom: (
                    (atom.updated_time, atom.record_id)
                    if by_updated_at
                    else (atom.priority, atom.record_id)
                ),
            ),
        )

    async def l2_scenes(
            self,
            request: L2QueryRequest,
    ) -> L2QueryResponse:

        scope = request.target.to_scope()

        # ⚠️ 全量读进来再在内存里排序/分页 —— 这是**刻意**的
        # --------------------------------------------------
        # 两个理由，都不是"图省事"：
        #
        #   1. 摘要与热度是**从正文解析出来的**（`parse_meta`）—— 不给正文
        #      就给不出 `SceneSummary`，所以这一页本来就得把正文读出来。
        #   2. 语料有上限：一个 scope 十几篇（L2 的 `DEFAULT_MAX_SCENES`
        #      把着，而整段拼进提示词本来就是这条链的设计）。
        #
        # 于是"排序口径"只有一处（这里），不必再造一个仓储方法去重复
        # filename/updated_at 两种排序的 SQL。
        rows = await self._scenes.list_scenes(scope)

        if request.filename_contains:
            rows = [
                row for row in rows
                if request.filename_contains in row.filename
            ]

        ordered = self._order_scenes(rows, request.order)

        after = (
            decode_cursor(
                request.cursor,
                order=request.order,
                kinds=(
                    ("str",)
                    if request.order == "filename_asc"
                    else ("datetime", "str")
                ),
            )
            if request.cursor
            else None
        )

        if after is not None:
            ordered = [
                row for row in ordered
                if self._scene_after(row, after, request.order)
            ]

        page = ordered[:request.limit]

        has_more = len(ordered) > request.limit

        return L2QueryResponse(
            scope_kind=request.target.scope_kind,
            items=[self._scene_summary(row) for row in page],
            page=self._page(
                request.limit,
                has_more,
                order=request.order,
                last=page[-1] if page and has_more else None,
                key=lambda row: self._scene_key(row, request.order),
            ),
        )

    async def l2_scene_detail(
            self,
            request: L2DetailRequest,
    ) -> L2DetailResponse:

        scope = request.target.to_scope()

        return L2DetailResponse(
            scope_kind=request.target.scope_kind,
            scene=await self._scene_detail(scope, request.filename),
        )

    async def l3_persona(
            self,
            request: TargetRequest,
    ) -> L3QueryResponse:

        scope = request.target.to_scope()

        row = await self._personas.get_persona(scope)

        return L3QueryResponse(
            scope_kind=request.target.scope_kind,
            persona=(
                None
                if row is None
                else PersonaRecord(
                    content=row.content,
                    # ⚠️ 用常量而不是 `row.filename`：契约把这一列钉成
                    # `persona.md`（一个 scope 至多一篇），而这一行的定位
                    # 条件本来就是 scope —— 库里那一列真有别的值时，让它
                    # 在这里变成一次 400 是**错的响应**（请求本身没毛病）。
                    filename=PERSONA_FILENAME,
                    version=row.version,
                    updated_at=row.updated_at,
                )
            ),
        )

    # ========================================================
    # 跨层溯源
    # ========================================================

    async def l1_sources(
            self,
            request: L1SourcesRequest,
    ) -> L1SourcesResponse:

        scope = request.target.require_personal_scope()

        # `load` 取不到就**不在返回里**（不是错误）—— 那边是给去重候选池
        # 用的语义（两条并发任务之间可能有记录被删）。这里是"点名要这一条"，
        # 所以取不到必须抛。
        atom = (await self._atoms.load([request.record_id])).get(
            request.record_id
        )

        if atom is None:
            raise NotFoundError(f"记忆不存在：{request.record_id}")

        # 归属：承载它的会话必须对这个 target 可见。放在取消息**之前** ——
        # 否则一次跨 target 的调用会先读到别人的消息，再在下一步才发现
        # 不该读（虽然不会返回出去，但那次读已经发生了）。
        await self._require_session(atom.session_id, scope)

        # ⚠️ 落库的值本该都是能解析成 UUID 的字符串（写侧过滤过，见
        # `domain/schemas/atom.py` 那个字段的说明），但这是**从库里读回来**
        # 的数据，可能来自更早的版本。解析不出来的直接跳过 —— 它自然落进
        # 下面的 missing，不必单开一份记录：溯源的价值恰恰在于把异常看
        # 清楚，那也包括"有一条来源我认不出来"。
        wanted: list[UUID] = []

        for raw in atom.source_message_ids:

            try:
                wanted.append(UUID(str(raw)))

            except (ValueError, AttributeError, TypeError):
                continue

        messages = await self._messages.list_by_ids(
            wanted,
            session_id=atom.session_id,
        )

        # ⚠️ 差集按 **atom 上的原始串**给，不是按解析后的 UUID
        # --------------------------------------------------
        # 调用方拿着它去对照 `source_message_ids`（模型输出里的那串文本），
        # 重排成 UUID 的写法会让人对不上号。
        found = {str(message.message_id) for message in messages}

        missing = [
            str(raw)
            for raw in atom.source_message_ids
            if str(raw) not in found
        ]

        return L1SourcesResponse(
            scope_kind=request.target.scope_kind,
            record_id=atom.record_id,
            session_id=atom.session_id,
            content=atom.content,
            version=atom.version,
            sources=[
                ConversationRecord(
                    message_id=message.message_id,
                    session_id=message.session_id,
                    seq=message.seq,
                    role=message.role,
                    text=message_text(message) or "",
                    created_at=message.created_at,
                )
                for message in messages
            ],
            missing_message_ids=missing,
        )

    async def l0_atoms(
            self,
            request: L0AtomsRequest,
    ) -> L0AtomsResponse:

        scope = request.target.require_personal_scope()

        await self._require_session(request.session_id, scope)

        # ⚠️ 消息必须先**确实存在**，才能回答"它进了哪些记忆"
        # ---------------------------------------------
        # 直接返回空列表会把"你给错消息了"和"这条消息没产生记忆"混成
        # 同一个响应，而后者是常态（抽不出东西的轮次不产生记忆）。
        # 这一步顺带把归属校验也做了一遍 —— 消息按 session_id 取。
        existing = await self._messages.list_by_ids(
            [request.message_id],
            session_id=request.session_id,
        )

        if not existing:
            raise NotFoundError(
                f"消息不存在或不属于会话 {request.session_id}："
                f"{request.message_id}"
            )

        atoms = await self._atoms.list_by_source_message(
            request.message_id,
            session_id=request.session_id,
            limit=request.limit,
        )

        return L0AtomsResponse(
            scope_kind=request.target.scope_kind,
            session_id=request.session_id,
            message_id=request.message_id,
            atoms=[self._fact(atom) for atom in atoms],
        )

    # ========================================================
    # 工具入口（暴露给主模型的那一个）
    # ========================================================

    @tool(key="memory.scene.get")
    async def get_scene(
            self,
            request: MemorySceneGetRequest,
    ) -> L2DetailResponse:
        """
        L2场景全文，按文件名读取。参数：filename 场景文件名（逐字取自上下
        文里场景索引那一行的开头）。返回该场景的完整 markdown（含 META 块）。

        ⚠️ 什么时候该用它
        --------------
        上下文里给的是场景**索引**（文件名 + 摘要 + 热度 + 时间）—— 那够判断
        "这件事记在哪一篇"；要细节、要当时的原话结论时才取全文。要的是原话
        证据而不是提炼过的结论时，去查原始对话那一层。

        （这里刻意**不写另一个工具的名字**：模型看到的是函数名，而工具名是
        我们随时会改的标识，写死在描述里改错了不报错。）

        ⚠️ 身份与 scope 由运行时给，**不在入参里**
        ------------------------------
        这个包装方法存在的唯一理由就是"只收一个模型参数"：工具基建只认这一
        种形状（见 tool/registry/schema.py 的 `resolve_request_model`）。

        `user_id` / `agent_id` / `session_id` 从**进程内的调用者上下文**取
        （app/common/tool_caller.py）—— 它们绝不能进 schema：模型能填一个别人
        的 id，那就是跨租户读。

        ⚠️ `agent_id` 必须是**与注入场景索引时同一个**
        ----------------------------------------
        记忆那两层的 scope 是 (agent_id, user|org)，而 agent 那一半是这一轮
        的 `duty_key`（处理器注入时用的就是它）。从会话行猜一个 agent_id 会
        在"一个会话跑多份职责"时漂开，那时模型会拿着刚看到的文件名被告知
        "场景不存在" —— 不报错、只是答不上来。见 ToolCaller 的说明。

        ⚠️ 拿不到身份就拒绝
        ----------------
        拿不到只有两种可能：不来自 agent 那条链，或者那条链没接上。两种情况
        下"是谁 / 在哪份职责下"都不知道，而定位一篇场景**必须**知道 scope
        —— 放行等于把别人的场景摊开。与 `memory.search` / `memory_atom.search`
        同一条纪律。
        """

        caller = current_tool_caller()

        if caller is None or not caller.user_id or not caller.agent_id:
            raise ValueError(
                "拿不到调用者身份（user_id）或这一轮的职责（agent_id），"
                "无法定位记忆的 scope —— 拒绝而不是放行。"
            )

        # 团队会话的 org 在**会话行**上（调用者上下文里只有 user/session）。取
        # 不到会话时按个人 scope 处理：那种情况下 org 本来也无从谈起。
        org_id: str | None = None

        if caller.session_id is not None:

            session = await self._sessions.get_any(caller.session_id)

            org_id = None if session is None else session.org_id

        scope = SceneScope.resolve(
            agent_id=caller.agent_id,
            user_id=caller.user_id,
            org_id=org_id,
        )

        return L2DetailResponse(
            scope_kind="team" if scope.org_id is not None else "user",
            scene=await self._scene_detail(scope, request.filename),
        )

    # ========================================================
    # internals
    # ========================================================

    async def _scene_detail(
            self,
            scope: SceneScope,
            filename: str,
    ) -> SceneDetail:
        """
        按 scope + 文件名取一篇场景全文。

        ⚠️ 抽成一处是因为它有**两个**入口：管理端的 `/memory/query/l2/detail`
        与给模型的 `memory.scene.get`。两条路各写一遍的话，"文件名怎么匹配"、
        "摘要怎么解析"会慢慢分叉 —— 而症状是同一个文件名在一边取得到、在
        另一边取不到。
        """

        rows = await self._scenes.list_scenes(scope)

        row = next(
            (item for item in rows if item.filename == filename),
            None,
        )

        if row is None:
            raise NotFoundError(f"场景不存在：{filename}")

        meta = parse_meta(row.content)

        return SceneDetail(
            filename=row.filename,
            summary=meta.summary,
            heat=meta.heat,
            version=row.version,
            # 两个时间都带上：`created_at` 是"这份场景第一次建立"的时刻，
            # 与 `updated_at` 一起进提示词（只在两者不同时才打后半段，
            # 见 prompts/memory_retrieval.py 的 `_render_scene`）。
            created_at=row.created_at,
            updated_at=row.updated_at,
            content=row.content,
        )

    @staticmethod
    def _fact(atom: Any) -> FactRecord:
        """
        `MemoryAtom` -> 契约的 `FactRecord`（浏览与反查共用）。

        ⚠️ `updated_time` 在这条路上**必定有值**：拿到的都是仓储从库里读
        出来的行（列是 NOT NULL DEFAULT now()）。实体上那个 `| None` 是留给
        "还没落库的记忆"的（L1 抽取时刚组装出来那批），这里碰不到。
        """

        return FactRecord(
            record_id=atom.record_id,
            content=atom.content,
            type=atom.type,
            priority=atom.priority,
            scene_name=atom.scene_name,
            version=atom.version,
            timestamp_str=atom.timestamp_str,
            updated_at=atom.updated_time,
        )

    @staticmethod
    def _page(
            limit: int,
            has_more: bool,
            *,
            order: str,
            last: Any | None,
            key: Any,
    ) -> Pagination:
        """
        `(items, has_more)` -> 契约的 `Pagination`。

        ⚠️ `has_more` 为 False 时游标一定是 null —— 两个字段说的是同一件
        事，而"没有下一页却给一个游标"会让客户端多翻一次空页。
        """

        return Pagination(
            limit=limit,
            next_cursor=(
                encode_cursor(order, key(last))
                if has_more and last is not None
                else None
            ),
            has_more=has_more,
        )

    @staticmethod
    def _scene_summary(row: Any) -> SceneSummary:

        meta = parse_meta(row.content)

        return SceneSummary(
            filename=row.filename,
            summary=meta.summary,
            heat=meta.heat,
            version=row.version,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _scene_key(row: Any, order: str) -> tuple[Any, ...]:
        """场景在某个 order 下的排序键（游标装的就是它）。"""

        if order == "filename_asc":
            return (row.filename,)

        return (row.updated_at, row.filename)

    @staticmethod
    def _order_scenes(rows: list[Any], order: str) -> list[Any]:
        """
        按 order 排序（内存里排，理由见 `l2_scenes`）。

        `filename` 在两类键里都在 —— 它是场景的**身份**，所以它当
        tie-breaker 是稳的（`updated_at` 可能撞，一次整合写的几篇共享
        同一个 `now()`）。
        """

        if order == "filename_asc":
            return sorted(rows, key=lambda row: row.filename)

        return sorted(
            rows,
            key=lambda row: (row.updated_at, row.filename),
            reverse=order == "updated_at_desc",
        )

    @staticmethod
    def _scene_after(row: Any, after: list[Any], order: str) -> bool:
        """这一行是否排在游标边界**之后**（内存版的 keyset 条件）。"""

        if order == "filename_asc":
            return row.filename > after[0]

        boundary = (after[0], after[1])

        if order == "updated_at_desc":
            return (row.updated_at, row.filename) < boundary

        return (row.updated_at, row.filename) > boundary

    async def _require_session(
            self,
            session_id: UUID,
            scope: SceneScope,
    ) -> None:
        """
        会话必须对这个 target 可见，否则 404（**不区分**"不存在"与"不属于"）。

        ⚠️ 只用得上 `scope.user_id`
        ------------------------
        调用方都已经过 `require_personal_scope`，所以 user_id 必定有值、
        org_id 必定为空。传团队 target 在到达这里之前就已经 400 了。
        """

        session = await self._sessions.get(
            session_id,
            user_id=scope.user_id or "",
        )

        if session is None:
            raise NotFoundError(f"会话不存在：{session_id}")
