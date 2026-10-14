"""
MemoryQueryService —— 管理端记忆查询。

对应 memory_api_v2.yaml 里**已实现**的两组端点：

    Memory Extraction    提取进度水位（L1 / L2 / L3 各一条）
    Memory Query         跨层溯源（L1 <-> L0，`source_message_ids` 那条真边）

⚠️ 与另外四个 Memory 服务的关系
-------------------------
那四个各自负责**一层**的读写（L0 索引与召回、L1 抽取与召回、L2 整合、
L3 生成），本服务**不写任何东西**，只读 —— 而且刻意跨层：水位按会话的
归 L1/L2、按 scope 的归 L3，硬塞进任何单个服务都会让那个服务多一个它
执行路径上用不到的仓储。

⚠️ 它**不注入** embedder，也不调模型
----------------------------
水位、浏览、溯源都是纯 SQL。`retrieval_preview` 打开 L0/L1 时会**去取**
那两个召回服务（它们才带 embedder），但那是现取现还（`factory.scope(...,
begin=False)`）、不进本服务的构造函数 —— 所以本服务构造时不带
`pipeline_registry`，大多数管理端查询不碰模型栈，起得来也不预热模型。

代价说清楚：**开了 L0/L1 的那一次预览**会付一次 embedder 的加载/使用
（进程级预热的实例，与 document 共用），那是"真召回"的应有成本。

⚠️ 这是**管理端**入口，身份在请求体里
-------------------------------
与 `MemoryService` / `MemoryAtomService` 相反：那两个的请求体同时是暴露给
模型的工具入参 schema，身份（user_id / session_id）绝不能进去。本服务的
调用者只有 HTTP 的管理端视图，它的语义就是"显式指定目标 scope"，
所以 target 必须能带到服务层。

安全边界不在这份契约上，而在两处：

    角色    端点要求 Y-Trader-Admin（见 app/api/v1/router.py 的注册那一行）
    归属    target 里的 user_id 仍要过会话可见性校验（见 impl 的
            `_require_session`）—— 管理端不跳过归属，只跳过"只能看自己"
"""

from __future__ import annotations

from abc import abstractmethod

from app.common.service.base import BaseService

from ..domain.schemas.memory_scene import MemorySceneGetRequest
from ..domain.schemas.query import (
    L0AtomsRequest,
    L0AtomsResponse,
    L0QueryRequest,
    L0QueryResponse,
    L1ExtractionRequest,
    L1ExtractionResponse,
    L1QueryRequest,
    L1QueryResponse,
    L1SourcesRequest,
    L1SourcesResponse,
    L2DetailRequest,
    L2DetailResponse,
    L2ExtractionRequest,
    L2ExtractionResponse,
    L2QueryRequest,
    L2QueryResponse,
    L3ExtractionRequest,
    L3ExtractionResponse,
    L3QueryResponse,
    MemoryRetrievalPrompt,
    MemoryRetrievalRequest,
    TargetRequest,
)

__all__ = ["MemoryQueryService"]


class MemoryQueryService(BaseService):

    # ========================================================
    # 上下文预览
    # ========================================================

    @abstractmethod
    async def retrieval_preview(
            self,
            request: MemoryRetrievalRequest,
    ) -> MemoryRetrievalPrompt:
        """
        预览这一轮会拼出的记忆上下文（**不调模型**，但会真的去召回）。

        默认（`retrieve` 不传）只组装 L3 画像与 L2 场景索引 —— 那两样是
        "每轮都带"的东西，成本是几句话的量级。

        ⚠️ `retrieve.l0` / `retrieve.l1` 为 true 时**真跑一轮混合召回**
        -------------------------------------------------------
        以 `input` 为查询串，走的就是 Agent 调 `memory.search` /
        `memory_atom.search` 那条实现。所以 `input` 是必填的（契约里就是
        这么定的）—— 它是一个"模拟这一轮"的端点，不是纯回显。

        L0 的 scope 由 `session_id` 决定：传了就收窄到那一个会话，不传
        则跨会话。L1 **恒跨会话**（它是跨会话累积的用户事实，按会话过滤
        与那一层的定义冲突）。

        那两层是**按需**的：一次真召回要付两路检索 + 融合，而绝大多数轮次
        根本不需要（Agent 那边由它自己决定调不调工具）。

        ⚠️ 单层失败**不整单失败**
        --------------------
        某一层查不出来时，那一层进 `degraded`、其余照常返回（契约里
        "单层失败时可能部分降级"说的就是这个）。注意与"该层是空的"分得开：
        空 -> 有 section 但 `items` 为空；失败 -> 进 `degraded`、没有 section。

        层内**某一路**召回挂了是另一回事：那一层照样给结果，route 级的
        原因在 section 自己的 `degraded_routes` 里。

        L0/L1 在**团队 target** 下开不起来（那两层没有团队维度），此时它们
        不进 sections 而是进 `degraded`。L2/L3 两种 scope 都支持。
        """

        raise NotImplementedError

    # ========================================================
    # 提取进度水位
    # ========================================================

    @abstractmethod
    async def l1_extraction(
            self,
            request: L1ExtractionRequest,
    ) -> L1ExtractionResponse:
        """
        这个会话的 L1 抽取到哪条消息了（`memory_atom_cursor`）。

        读的是**权威水位本身**，不做任何派生 —— "还剩几轮没抽"这类问题
        要按仓库唯一那条"一轮"的口径算，不能在这里另发明一个（见
        `domain/schemas/query.py` 的模块 docstring）。

        会话不存在、或不属于 target 指向的 user -> `NotFoundError`（404）。
        这两件事**共用同一个响应**：管理端也不该靠"这个 id 存不存在"来
        探测别人的会话。

        传团队 target（`team_id`）-> `ValueError`（400）：水位表的键是
        session_id、没有 org 维度。理由见 `RetrievalTarget.
        require_personal_scope`。
        """

        raise NotImplementedError

    @abstractmethod
    async def l2_extraction(
            self,
            request: L2ExtractionRequest,
    ) -> L2ExtractionResponse:
        """
        这个会话的 L2 整合到哪条记忆了（`memory_scene_cursor`）。

        三个字段一起看才有意义：`last_atom_updated_time` 是进度，
        `last_l2_at` 是"上次跑完是什么时候"（含空转那一次），
        `last_atom_ids` 是边界那一批 —— 见 `L2CursorRecord` 的说明。

        错误语义与 `l1_extraction` 同。
        """

        raise NotImplementedError

    @abstractmethod
    async def l3_extraction(
            self,
            request: L3ExtractionRequest,
    ) -> L3ExtractionResponse:
        """
        这个 scope 的画像抽象到哪一版场景了（`memory_persona_cursor`）。

        与上面两条**粒度不同**：L3 的水位按 scope 记（画像就是跨会话聚合的
        产物，`memory_scene` 上没有 session_id），所以请求里没有会话。

        ⚠️ 这个 scope 从没生成过画像**不是错误**
        -------------------------------------
        返回的是一组默认值（两个水位 null、空 id 集）。与
        `memory.query.l3` 那边"画像尚未生成时返回 `persona: null`"是同一条
        取舍：管理员问的是"到哪了"，"还没开始"是一个正当答案。
        """

        raise NotImplementedError

    # ========================================================
    # 浏览
    # ========================================================

    @abstractmethod
    async def l0_records(
            self,
            request: L0QueryRequest,
    ) -> L0QueryResponse:
        """
        浏览 L0 原始消息记录（**不执行**召回）。

        过滤只有四种：会话、角色、时间范围、字面文本包含；翻页只有游标。
        没有向量、没有关键词打分、没有融合 —— 那些在 `memory.yaml` 的
        `/memory/search`（同时是给主模型的工具）。

        `session_id` 给了但会话对 target 不可见 -> 404（与"不存在"共用
        同一个响应）。团队 target -> 400：`agent_message` 没有 org 列，
        归属在会话上。
        """

        raise NotImplementedError

    @abstractmethod
    async def l1_records(
            self,
            request: L1QueryRequest,
    ) -> L1QueryResponse:
        """
        浏览 L1 原子记忆记录（**不执行**召回）。

        过滤：类型 / 优先级 / 情境（都是等值）、更新时间范围、正文包含。
        `text_contains` 是 `ILIKE` 整串匹配，**不是**语义召回 —— 后者在
        `memory_atom.yaml`。

        团队 target -> 400（与 L0 同一条边界：`memory_atom` 的可见性维度
        只有 `user_id`，`org_id` 是元数据、不参与过滤）。
        """

        raise NotImplementedError

    @abstractmethod
    async def l2_scenes(
            self,
            request: L2QueryRequest,
    ) -> L2QueryResponse:
        """
        列出 L2 场景索引（**不含**正文）。

        ⚠️ 这一层是"列举"，不是"检索"
        ------------------------
        `memory_scene` 没有挂载表（没有 FTS、没有向量），一个 scope 也就
        十几篇 —— 建索引买不到任何东西。所以这里把该 scope 的场景**全读
        出来**再在内存里排序分页：摘要与热度本来就要从正文解析
        （`parse_meta`），不读正文也给不出 `SceneSummary`。

        两种 scope 都支持（场景就是按 scope 聚合的）。
        """

        raise NotImplementedError

    @abstractmethod
    async def l2_scene_detail(
            self,
            request: L2DetailRequest,
    ) -> L2DetailResponse:
        """
        取单篇场景全文（精确读取，不执行语义检索）。

        文件名不存在 -> `NotFoundError`（404）—— 与列表那边"没有场景"
        返回空列表不同：这里是**点名要一篇**，取不到就是取不到。
        """

        raise NotImplementedError

    @abstractmethod
    async def l3_persona(
            self,
            request: TargetRequest,
    ) -> L3QueryResponse:
        """
        读取该 scope 的画像；尚未生成 -> `persona: null`（**不是 404**）。

        ⚠️ 为什么不是 404
        -------------
        scope 不是一个"可以不存在"的实体（它由 target 算出来），而"还没
        生成过画像"是一个正常状态 —— 判据是"库里有没有那一行"。用 404
        表达它会让调用方把"这个用户还没画像"与"你把 agent_id 传错了"
        混成同一个响应。
        """

        raise NotImplementedError

    # ========================================================
    # 工具入口
    # ========================================================

    @abstractmethod
    async def get_scene(
            self,
            request: "MemorySceneGetRequest",
    ) -> L2DetailResponse:
        """
        暴露给主模型的工具 `memory.scene.get`。

        取一篇场景的**全文**（L2 进上下文的是索引，全文按需取）。与
        `/memory/query/l2/detail` 是**同一条读**（共用 `_scene_detail`），
        差别只在身份从哪来：那个的 target 在请求体里，这个从进程内的调用者
        上下文取（`app/common/tool_caller.py`）。

        ⚠️ 它必须用**与注入索引时同一个 scope**
        --------------------------------
        记忆那两层的 scope 是 (agent_id, user|org)，agent 那一半是这一轮的
        `duty_key`。工具要的那份同样从调用者上下文取（`ToolCaller.agent_id`）
        —— 换成"从会话行猜一个"会在一个会话跑多份职责时漂开，那时模型会拿着
        刚在索引里看到的文件名被告知"场景不存在"。

        Raises:
            ValueError: 拿不到调用者身份或职责。定位一篇场景必须知道 scope，
                        所以这里**拒绝**而不是放行（与 L0/L1 的工具同一条
                        纪律）。
            NotFoundError: 该 scope 下没有这个文件名的场景（404 的语义；
                        对模型来说表现为"这篇取不到"）。
        """

        raise NotImplementedError

    # ========================================================
    # 跨层溯源
    # ========================================================

    @abstractmethod
    async def l1_sources(
            self,
            request: L1SourcesRequest,
    ) -> L1SourcesResponse:
        """
        这条 L1 记忆是从哪几句原话来的（`memory_atom.source_message_ids`）。

        层间唯一一条真落在库里的边。返回里同时给 `missing_message_ids`：
        那些 id 记得有、但已经取不到了（没有外键，会话可能被删）—— "记得有
        5 条、只找到 3 条"必须是看得见的事实。

        记忆不存在 -> `NotFoundError`（404）。承载它的会话不属于 target
        -> 同样 404（不区分）。

        团队 target -> `ValueError`（400），与 v2 规格里 L0/L1 Query 的
        边界一致（本仓库的 L1 记录是会话级的，没有"按团队浏览 L1"这条路径）。
        """

        raise NotImplementedError

    @abstractmethod
    async def l0_atoms(
            self,
            request: L0AtomsRequest,
    ) -> L0AtomsResponse:
        """
        反查：这句原话进了哪些记忆（同一列的反向）。

        ⚠️ `session_id` 必填，且消息必须**确实在这个会话里**
        --------------------------------------------
        包含查询没有索引可用，靠会话把候选收窄。因此消息不存在（或不在这个
        会话里）-> `NotFoundError`（404），而不是返回空列表 —— 后者读起来
        是"这条消息没有产生任何记忆"，与"你给错消息了"是两件事。

        消息存在但没有任何记忆引用它 -> **空列表**（这是常态：抽不出东西
        的那些轮次本来就不产生记忆）。

        团队 target -> `ValueError`（400），同上。
        """

        raise NotImplementedError
