"""
管理端记忆查询的**对外契约**（memory_api_v2.yaml 的全部端点）。

四组东西，问的问题不同：

    上下文预览   "每轮会往主模型的上下文里塞什么" —— 拼一份 prompt 预览
                 （只含 L3 画像与 L2 场景索引，不调模型）
    提取进度     "这个会话 / 这个 scope 的记忆抽到哪了" —— 读三张 cursor 表
    浏览         "已经存下来的记录长什么样" —— L0-L3 各一个列表 + 两个详情
    跨层溯源     "这条记忆是从哪几句原话来的 / 这句话进了哪些记忆" —— 读
                 memory_atom.source_message_ids

⚠️ 浏览这一组**没有** `scope_kind: team` 之外的东西，也**不做召回**
--------------------------------------------------------
过滤只有等值 / 范围 / 字面包含三种，分页只有游标一种。语义召回在
`memory.yaml` 与 `memory_atom.yaml`（那两个同时是暴露给模型的工具）。

⚠️ 身份在**请求体里**，与 memory.yaml / memory_atom.yaml 相反
------------------------------------------------------
那两份契约同时是暴露给模型的工具入参 schema，身份进去就是一条越权路径。
这一份**只给 HTTP 用**：管理端视图的语义就是"显式指定目标 scope"（见
v2 规格的 Target 与权限），所以 target 必须在这里。安全边界落在两处，
都不在这份模型上：

    角色      端点要求 Y-Trader-Admin（见 app/api/v1/router.py 的注册那一行）
    归属      target 里的 user_id 仍要走会话可见性校验（见 impl）

⚠️ 水位按原样返回，**不做派生**
---------------------------
最想加的两个派生字段是"距上次抽取攒了多少轮""还有几条没抽"，这里刻意
都不给：

    口径只有一处   "一轮 = 一条 role=user 且 kind=chat 的消息"这条定义
                   住在本仓库唯一一处（context builder 的 split_turns 与
                   session 模块的 count_user_turns）。在这里按"消息条数"
                   另算一个近似值，会让同一个问题出现两个答案，而**两者
                   不一致不会报错**（L3 那处轮次判定的注释里写着同一件事）
    不猜           L1 的水位是 `agent_message.seq`，L2 的是
                   `memory_atom.updated_time`，L3 的是
                   `memory_scene.updated_at` —— 三个"进度"根本不是同一种
                   单位，"还差多少"要各算各的，那就是三次口径决定，
                   不是一次加字段

所以这里给的是**权威水位本身**，刻度由调用方自己对着数据看。

⚠️ 没有的东西，以及为什么（跨层）
---------------------------
层间的溯源边只有**一条**是真的落在库里的：

    L1 -> L0     memory_atom.source_message_ids（真列，本文件的两个端点）
    L2 -> L1     没有。场景文件名是模型自己写的路径
                 （memory_scene_service_impl 的 normalize_filename），与
                 memory_atom.scene_name（L1 抽出的**情境名**）不是同一个
                 东西，join 不上
    L3 -> L2     没有。memory_persona_cursor 上只剩**边界那一批**
                 （last_scene_ids），上一次生成到底看过哪些场景没有落库

后两条**不提供近似接口** —— 用 scene_name 匹配文件名、或用边界批冒充
全量，都会给出一份看起来完整、实际缺一半的答案。要它们就得先落一条
映射（写入侧的改动），不是加个端点的事。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.session.domain.schemas.enum import MessageRole

from ..entities.scene import SceneScope

__all__ = [
    "ConversationHit",
    "ConversationHitSection",
    "ConversationRecord",
    "DegradedHitRoute",
    "DegradedLayer",
    "FactHit",
    "FactHitSection",
    "FactRecord",
    "L0AtomsRequest",
    "L0AtomsResponse",
    "L0QueryOrder",
    "L0QueryRequest",
    "L0QueryResponse",
    "L1CursorRecord",
    "L1ExtractionRequest",
    "L1ExtractionResponse",
    "L1QueryOrder",
    "L1QueryRequest",
    "L1QueryResponse",
    "L1SourcesRequest",
    "L1SourcesResponse",
    "L2CursorRecord",
    "L2DetailRequest",
    "L2DetailResponse",
    "L2ExtractionRequest",
    "L2ExtractionResponse",
    "L2QueryOrder",
    "L2QueryRequest",
    "L2QueryResponse",
    "L3CursorRecord",
    "L3ExtractionRequest",
    "L3ExtractionResponse",
    "L3QueryResponse",
    "MemoryPreviewLayers",
    "MemoryPreviewLimits",
    "MemoryRetrievalPrompt",
    "MemoryRetrievalPromptText",
    "MemoryRetrievalRequest",
    "Pagination",
    "PersonaRecord",
    "PersonaSection",
    "PromptSection",
    "RecallRoute",
    "RetrievalTarget",
    "SceneDetail",
    "SceneSection",
    "SceneSummary",
    "ScopeKind",
    "TargetRequest",
]


#: 由 target 解析得到的 scope 类型。取值与 v2 规格的 ScopeKind 一致。
ScopeKind = Literal["user", "team"]


class RetrievalTarget(BaseModel):
    """
    查询 scope：`user_id` 与 `team_id` **恰好提供一个**，`agent_id` 必填。

    与 v2 规格的 RetrievalTarget 逐字段对应。`team_id` 在库里叫 `org_id`
    （契约里用 team 这个词，因为调用方嘴里的"团队"就是那次聚合的单位），
    折成 SceneScope 时映射过去。

    ⚠️ 校验放在这里，而且**同时**在 SceneScope 上
    ------------------------------------------
    两层不是重复：这一层给的是"哪个字段写错了"的错误信息（贴着出问题的
    那个值），SceneScope 那层守的是"绝不构造出一个两列都空的 scope"这条
    不变量 —— 后者的失效症状是查不到东西且不报错。
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        min_length=1,
        description=(
            "归属的业务 agent。数据库里对应 business_agent_id / duty_key。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        min_length=1,
        description="个人记忆的归属用户（Casdoor sub）。",
    )

    team_id: str | None = Field(
        default=None,
        min_length=1,
        description="团队记忆的归属组织。数据库列名为 org_id。",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "RetrievalTarget":

        if (self.user_id is None) == (self.team_id is None):
            raise ValueError(
                "target 必须恰好有 user_id 或 team_id 之一："
                "有 team_id 的是团队记忆，没有的是个人记忆。"
                "两个都空会让这次查询谁都匹配不上（不报错）。"
            )

        return self

    @property
    def is_team(self) -> bool:
        """团队 scope（team + agent）还是个人 scope（user + agent）。"""

        return self.team_id is not None

    @property
    def scope_kind(self) -> ScopeKind:
        """"user" / "team" —— 响应里回显的那个值。"""

        return "team" if self.is_team else "user"

    def to_scope(self) -> SceneScope:
        """
        折成本模块内部唯一的 scope 表示（`SceneScope`）。

        走 `SceneScope.resolve` 而不是直接构造：它同时实现了两条本仓库
        反复强调的规则 —— 有 org 就**丢掉** user_id（不是两个都填），
        以及空串按 NULL 处理（历史数据里 `agent_session.org_id` 出现过
        空串，当成独立 scope 会让同一个人的画像莫名其妙分成两批）。
        """

        return SceneScope.resolve(
            agent_id=self.agent_id,
            user_id=self.user_id,
            org_id=self.team_id,
        )

    def require_personal_scope(self) -> SceneScope:
        """
        只接受个人 scope，团队 scope 抛错（HTTP 400）。

        四个调用点，同一条理由：L1/L2 的**提取水位**（那两张水位表），
        L0 的**浏览**，L1 的**浏览**，以及两个**跨层溯源**端点。

        ⚠️ 这里**不是**"还没做"，是数据模型上没有那个维度
        --------------------------------------------
        这三类东西都按**会话**或**用户**记：

            memory_atom_cursor / memory_scene_cursor   键是 session_id，
                                                       只带一列 user_id
            agent_message                              只有 session_id，
                                                       归属在**会话**上
            memory_atom                                可见性维度是
                                                       user_id（org_id 是
                                                       元数据，不参与过滤）

        而"一个团队的 L1 抽到哪了"本身也不是一句话：水位按会话记
        （`agent_message.seq` 只在会话内单调），团队下 N 个会话就有 N 条
        水位，没有一条能代表团队。

        所以传团队 target 是**参数不合法**（400），不是"查不到"（那会返回
        空，读起来像"这个团队没有记忆"）—— 与 v2 规格里 L0/L1 Query 的
        同一条边界。

        支持团队 scope 的只有 **L2 的场景**与 **L3 的画像**：那两样本来就是
        按 (agent_id + user_id|org_id) 聚合的（它们走 `to_scope`）。
        """

        if self.is_team:
            raise ValueError(
                "团队 scope 不支持这一个端点：L0/L1 的浏览、跨层溯源与 "
                "L1/L2 的提取水位都按**会话/用户**记录（消息表没有 org 列，"
                "水位表也没有），没有团队维度。请传 user_id 而不是 team_id。"
            )

        return self.to_scope()


class TargetRequest(BaseModel):
    """
    只要一个 `target` 的请求。

    目前只有 `/memory/query/l3` 用它：画像的定位条件就是 scope 本身
    （`agent_id` + `user_id`|`team_id`），没有第二个参数 —— 一篇画像一个
    scope，不需要指名"要哪一篇"。
    """

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget


# ============================================================
# 上下文预览（/memory/retrieval）
# ============================================================
#
# ⚠️ 这一组**不调模型**，但要拼出一份 prompt 预览
# -----------------------------------------
# 它回答的是"勾上 L2/L3 之后，会往主模型的上下文里塞什么"。所以响应里既有
# **结构**（`sections`），也有一份**拼好的文本**（`prompt.user`）。
#
# ⚠️ 排版归提示词模块，不在这份契约里
# -----------------------------
# `prompt.user` 由 `app/agent/prompt/prompts/memory_retrieval.py` 的渲染
# 函数拼出来（见提示词目录的约定：静态文本与动态块都住在提示词文件里）。
# 契约只规定结构与字段，**不规定**小节标题怎么写、用什么分隔符 ——
# 否则"提示词长什么样"就有两个答案。
#
# 同理，`prompt.system` 是那份文件的 `SYSTEM_PROMPT` 常量；它现在是**空**的
# （正文还没写），于是这里返回 `null` —— 契约里那句"尚未实现时为 null"
# 说的就是这个状态。


class MemoryPreviewLayers(BaseModel):
    """
    预览里展示哪几层，**以及要不要真的去召回**。

    ⚠️ 默认只开 L2/L3，L0/L1 要手动开
    -----------------------------
    这不是"省事"，是成本的分界（原规格把这条写得很清楚）：

        L2 + L3    画像 + 场景索引 ≈ 几句话的量级，**每一轮都负担得起**
                   —— 它们本来就是每轮都进上下文的
        L0 + L1    一次**真正的检索**（两路召回 + 融合 + 条数）—— 所以是
                   **按需**的动作：Agent 那边由它自己决定调不调工具，
                   管理端这里由调用方显式勾

    ⚠️ 开了就是**真跑**，不是假装
    ----------------------
    `l0` / `l1` 为 true 时，服务端会用 `input` 作为查询串真的跑一轮混合
    召回（与 Agent 调 `memory.search` / `memory_atom.search` 同一条实现），
    并把命中放进对应的 section。所以 `input` 是**必填**的 —— 没有查询串
    就没有召回可言。

    ⚠️ 团队 target 下 L0/L1 开不起来
    -------------------------
    那两层没有团队维度（`agent_message` 没有 org 列、`memory_atom` 的
    org_id 不参与过滤），所以团队 scope 下开了它们不会静默少两段，
    而是进 `degraded` 并说明原因。
    """

    model_config = ConfigDict(extra="forbid")

    l0: bool = Field(
        default=False,
        description="是否真的跑一轮 L0 原始消息召回（要以 `input` 为查询串）。默认关闭。",
    )
    l1: bool = Field(
        default=False,
        description="是否真的跑一轮 L1 原子记忆召回（要以 `input` 为查询串）。默认关闭。",
    )
    l2: bool = Field(default=True, description="是否包含 L2 场景索引，默认开启。")
    l3: bool = Field(default=True, description="是否包含 L3 画像，默认开启。")


class MemoryPreviewLimits(BaseModel):
    """
    预览中各层的条数上限。

    ⚠️ L0/L1 的默认值与取值范围**跟着召回端点**（`DEFAULT_LIMIT` = 10、
    `MAX_LIMIT` = 50）—— 同一次召回的条数在预览与工具里必须是同一套口径，
    否则"预览里看到 5 条"与"Agent 拿到 10 条"会被当成两个答案。
    """

    model_config = ConfigDict(extra="forbid")

    l0: int = Field(default=10, ge=1, le=50, description="L0 召回最多返回几条（融合之后截断）。")
    l1: int = Field(default=10, ge=1, le=50, description="L1 召回最多返回几条（融合之后截断）。")
    l2: int = Field(default=12, ge=1, le=100, description="最多返回的场景索引数量。")


class MemoryRetrievalRequest(BaseModel):
    """
    预览请求。

    ⚠️ `input` 是**条件必填**：要召回才必填
    --------------------------------
    这个端点的语义是"**模拟这一轮**：用户说了这句话，会拼出什么上下文"。
    L0/L1 打开时用它真的去召回（见 `MemoryPreviewLayers`），所以那时它不能
    缺：缺了只剩两种坏选择 —— 静默不召回（返回一个假的空段），或者报一个
    与原因无关的错。

    但 L2/L3 **不需要**查询串（画像是这一篇、场景是这一份清单，与"这一轮
    问了什么"无关），而取它们的那条路（上下文流水线的 profile 处理器）
    在一个没有用户消息的轮次上根本没有查询串。所以规则精确成一条：
    **开了 `l0` 或 `l1` 就必须给 `input`**，只取 L2/L3 时可以不给自己。

    它同时**原样回显**在响应里（`input`），于是读到结果的人不必往上翻
    就知道这一轮问的是什么。

    ⚠️ `session_id` 只作用于 L0，**不作用于 L1**
    -----------------------------------------
    默认（不传）两层都是**跨会话**的。传了它，L0 的召回收窄到这一个会话
    —— "上一轮我们说过什么"问的就是本会话里的原话。

    L1 **刻意不收窄**：它是跨会话累积的用户事实（"用户不允许自动下单"
    这类），按会话过滤会把它削成本轮对话的记忆，与那一层的定义冲突。

    L2/L3 也不受影响 —— 那两层的粒度是 scope，本来就没有会话维度。

    ⚠️ 这一层**不做归属校验**（与 `MemoryService.search` 那条路不同）
    --------------------------------------------------------------
    这里传进来的 id 只是过滤条件。谁有资格看这个会话的记忆，由调用方
    那道门槛决定（管理端是端点上的角色依赖；agent 侧的身份来自 state）。
    管理端端点因此仍是管理语义：它能看谁由角色决定，而 target 里的
    `user_id` 照常过 `_require_session` 的可见性口径。
    """

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    input: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "本轮的用户输入。**开了 L0/L1 时必填** —— 混合召回以它为查询串；"
            "只取 L2/L3 时可以不给自己（那两层与「这一轮问了什么」无关）。"
        ),
    )
    retrieve: MemoryPreviewLayers = Field(default_factory=MemoryPreviewLayers)
    limits: MemoryPreviewLimits = Field(default_factory=MemoryPreviewLimits)
    session_id: UUID | None = Field(
        default=None,
        description=(
            "限定 L0 的召回范围到单个会话；省略则跨会话。"
            "对 L1 / L2 / L3 无效。"
        ),
    )

    @model_validator(mode="after")
    def _input_required_for_recall(self) -> "MemoryRetrievalRequest":
        """
        要召回就必须有查询串。

        ⚠️ 判在这里，而不是让服务在运行时兜
        -------------------------------
        它是**请求本身**的性质（缺了就没法执行），而调用点有两个：管理端
        端点与上下文流水线的 profile / memory 两个处理器。搁在服务里判就要
        三处各记得处理，而漏掉任何一处的表现都是"召回被静默跳过"。

        反过来，L2/L3 那两层**不能**要求它：profile 处理器取画像与场景
        清单时，一个没有用户消息的轮次（CONTINUATION、系统触发）手上根本
        没有查询串 —— 要求它等于把那两段也一起丢掉。
        """

        if (self.retrieve.l0 or self.retrieve.l1) and not self.input:
            raise ValueError(
                "开了 L0/L1 就必须给 input —— 混合召回以它为查询串，"
                "没有查询串就没有召回可言。只取 L2/L3 时不需要它。"
            )

        return self


class PersonaSection(BaseModel):
    """
    L3 那一段。

    `items` 最多一项（一个 scope 至多一篇画像），`returned` 是实际给出的
    条数（0 或 1），`truncated` 恒为 False —— 画像不截断（截断一整篇
    文档会毁掉它，而它本来就有字数约束在提示词侧）。
    """

    model_config = ConfigDict(extra="forbid")

    block: Literal["persona"]
    layer: Literal["l3"]
    title: str
    items: list[PersonaRecord] = Field(max_length=1)
    returned: int
    truncated: bool


class SceneSection(BaseModel):
    """
    L2 那一段（**只有索引字段**，没有正文）。

    `truncated` 为 True 表示该 scope 的场景比 `limits.l2` 多 —— 它必须
    看得见：模型看不到被截掉的部分时，会以为清单就是全部。

    ⚠️ `scenes_total` 必须单独给，不能只靠 `truncated`
    -------------------------------------------------
    渲染那一段的标题要写"共 N 篇，以下 M 篇"（见
    app/agent/prompt/prompts/memory_retrieval.py）。`truncated` 只说得清
    "被截过"，说不出**总共多少篇** —— 而"12 篇里给了 8 篇"与"80 篇里给了
    8 篇"对读它的人是完全不同的两件事。

    它同时也让 `sections` 成为这段内容的**完整**结构化表示：只拿
    `sections`（不拿 `prompt.user` 那个串）的调用方也能还原出同样的文字。
    """

    model_config = ConfigDict(extra="forbid")

    block: Literal["scenes"]
    layer: Literal["l2"]
    title: str
    items: list[SceneSummary]
    returned: int
    scenes_total: int
    truncated: bool


#: 一路召回的名字（两层的取值域相同，所以是**一个**类型）。
#:
#: 它不是"两个同值枚举"的偷懒：`keyword` / `vector` 说的就是那两张挂载表，
#: 而 L0 与 L1 各有自己的一对 —— 名字一样是因为表一样，不是因为可以合并。
RecallRoute = Literal["keyword", "vector"]


class ConversationHit(BaseModel):
    """
    L0 召回的一条命中。

    ⚠️ `text` 是**片段**，与 `/memory/query/l0` 的 `ConversationRecord.text`
    不是一回事
    ------------------------------------------------------------------
    浏览那边给**整条**消息；召回这边给截断后的片段（原规格的
    `SNIPPET_CHAR_LIMIT`，400 字符）—— 因为它的消费者是模型，"一次往
    上下文里倒五条长消息"比"倒五句相关的"更糟。所以这里**不**复用
    `ConversationRecord`：同一个字段名装两种东西，是这条契约里最容易
    被读错的地方。
    """

    model_config = ConfigDict(extra="forbid")

    message_id: UUID
    session_id: UUID
    seq: int
    role: MessageRole
    text: str = Field(description="命中的**片段**（截断过），不是整条消息。")
    created_at: datetime
    score: float = Field(
        description="融合分，越大越相关。**只在本次响应内可比**（关键词分来自 ts_rank、向量分来自余弦距离）。",
    )
    matched_by: list[RecallRoute] = Field(description="这条被哪几路召回到。")


class FactHit(BaseModel):
    """
    L1 召回的一条命中。

    ⚠️ `content` **不截断**（与 L0 的片段相反）
    ------------------------------------
    这边已经是提炼过的陈述，截断可能把「用户喜欢 X，但**不能** Y」切掉
    后半句 —— 那比不给还糟。
    """

    model_config = ConfigDict(extra="forbid")

    record_id: UUID = Field(description="记忆身份。合并会改写它保活的那条、删掉被吃掉的。")
    content: str = Field(description="记忆正文，**原样返回、不截断**。")
    type: str
    priority: int
    scene_name: str
    score: float = Field(description="融合分。**只在本次响应内可比**。")
    matched_by: list[RecallRoute] = Field(description="这条被哪几路召回到。")

    # ⚠️ 两个时间来自 `memory_atom.created_time` / `updated_time`
    # ------------------------------------------------------
    # 源列就叫这两个名字，**这里刻意归一成 `*_at`** —— 本段与本文件里
    # 同族的 `ConversationHit.created_at` / `PersonaRecord.updated_at`
    # 用的是 `*_at`，四段一起进同一段提示词，混着两种后缀只会让人以为
    # 它们来自不同的地方。改名只发生在这一层，映射点写在
    # memory_query_service_impl 的 `_recall_l1` 里。
    #
    # 两个都给：合并会改写 updated_time、created_time 不动 ——
    # 「早就有的事实」与「刚被合并改写过的事实」是两件事。
    #
    # ⚠️ **可空**，跟的是源头（`MemoryAtom.created_time / updated_time`
    # 在实体上就是 `| None`，见 entities/atom.py）。被召回的记忆按定义
    # 都落过库、两个值必有；留成可空是为了"上游给了个 None"时这里不
    # 抛 —— 一条时间缺失不该让整段记忆注入失败（渲染端会省掉那一段）。
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DegradedHitRoute(BaseModel):
    """
    这一轮里**层内某一路**召回没跑成（关键词路或向量路）。

    ⚠️ 与顶层 `degraded` 不是一回事
    -------------------------
        顶层 degraded       **整层**没查成（读库失败、团队 scope 不支持…）
        这个（section 内）  这一层查成了，但其中**一路**召回挂了

    分开是刻意的：前者没有 section，后者有（带着另一路的结果）。合成
    一个的话，"向量路挂了但还是给了关键词结果"会看起来像"整层没跑成"。
    """

    model_config = ConfigDict(extra="forbid")

    route: RecallRoute
    reason: str


class ConversationHitSection(BaseModel):
    """
    L0 那一段（一次**真召回**的结果）。

    ⚠️ 没有 `truncated`
    ----------------
    L2 那两段有，这里没有，因为召回是 **top-K**、不是分页：融合之后
    "还有没有更多"没有定义（分数是相对的，`limit` 就是答案的边界）。
    要更多就调大 `limits.l0`。硬塞一个布尔量进去只会是一个猜出来的值。
    """

    model_config = ConfigDict(extra="forbid")

    block: Literal["conversations"]
    layer: Literal["l0"]
    title: str
    items: list[ConversationHit] = Field(description="按融合分降序。")
    returned: int
    degraded_routes: list[DegradedHitRoute]


class FactHitSection(BaseModel):
    """L1 那一段（一次**真召回**的结果）。字段与 L0 那段同形。"""

    model_config = ConfigDict(extra="forbid")

    block: Literal["facts"]
    layer: Literal["l1"]
    title: str
    items: list[FactHit] = Field(description="按融合分降序。")
    returned: int
    degraded_routes: list[DegradedHitRoute]


#: 按 `block` 判别的联合 —— 与契约里的 discriminator 逐字对应。
#:
#: 顺序就是**默认组装顺序**：画像 -> 场景 -> L0 召回 -> L1 召回。前两段是
#: 每轮都带的（默认开），后两段是按需的（默认关），所以它们排在后面。
PromptSection = Annotated[
    PersonaSection | SceneSection | ConversationHitSection | FactHitSection,
    Field(discriminator="block"),
]


class DegradedLayer(BaseModel):
    """
    **整层**没查成的原因（部分降级）。

    ⚠️ 它是"这一层没查成"，不是"这一层是空的"
    -----------------------------------
    两者在响应里必须分得开：空 -> 该层有 section 但 `items` 为空；
    失败 -> 该层进 `degraded`、**没有** section。合成一个的话，"画像
    是空的"与"画像查不出来"就长得一样了，而它们的处理方式完全不同。

    ⚠️ 团队 target 下开了 L0/L1 也走这里
    ------------------------------
    那两层没有团队维度（见 `RetrievalTarget.require_personal_scope`），
    而汇总端点不能整单 400（其余几层是查得到的）—— 所以那两段不出现、
    在这里说明原因，**不静默少两段**。
    """

    model_config = ConfigDict(extra="forbid")

    layer: Literal["l0", "l1", "l2", "l3"]
    reason: str = Field(description="层查询失败的原因；服务端同时应记录 warning 日志。")


class MemoryRetrievalPromptText(BaseModel):
    """拼好的 prompt 预览。"""

    model_config = ConfigDict(extra="forbid")

    system: str | None = Field(
        description=(
            "memory_retrieval 的系统提示词常量；**尚未实现时为 null**"
            "（见 app/agent/prompt/prompts/memory_retrieval.py 的 SYSTEM_PROMPT）。"
        ),
    )
    user: str = Field(description="根据 sections 渲染得到的用户消息预览。")


class MemoryRetrievalPrompt(BaseModel):
    """
    上下文预览的完整结果。

    ⚠️ `sections` 的顺序就是 prompt 的顺序（persona -> scenes）
    ------------------------------------------------
    它是**给前端渲染**的结构；`prompt.user` 是同一份数据的文本形态。
    两者由同一次读取产生 —— 不要分两次查（两次之间有人写入时，结构里
    有而文本里没有，没人会去核）。
    """

    model_config = ConfigDict(extra="forbid")

    retrieval_id: UUID = Field(description="本次预览的 id（每次都新生成，不落库）。")
    input: str | None = Field(
        description=(
            "本轮输入回显。请求里 `input` 是**条件必填**（要召回才必填），"
            "所以这里可以是 null —— 只取 L2/L3 的那次预览就会是 null。"
            "键恒在（没有输入时值为 null），与 L3 水位那两个 nullable 字段"
            "同一条口径：**键必须在，值可以是 null**。"
        ),
    )
    target: RetrievalTarget
    scope_kind: ScopeKind
    sections: list[PromptSection] = Field(
        description=(
            "按 prompt 顺序返回：画像 -> 场景 -> L0 命中 -> L1 命中"
            "（后两段只在 `retrieve.l0` / `retrieve.l1` 为 true 时出现）。"
        ),
    )
    degraded: list[DegradedLayer]
    prompt: MemoryRetrievalPromptText
    generated_at: datetime


# ============================================================
# 提取进度：L1（memory_atom_cursor）
# ============================================================


class L1CursorRecord(BaseModel):
    """
    一个会话的 L1 抽取水位（`memory_atom_cursor` 的读模型）。

    ⚠️ 没有行与"有行但没抽过"在这里是**同一件事**
    ------------------------------------------
    仓储把"没有行"折成 `last_seq=0` / 空情境名（见 `AtomCursorState`），
    本响应沿用那个折法：`last_seq=0` 就是"一条都还没抽过"。水位行是惰性
    创建的，所以这个折法不丢信息 —— 它只丢"这个会话到底抽过没有"这个
    可以用 `last_seq=0` 表达的区别。

    ⚠️ 没有 `updated_at`
    ----------------
    库里那一列是有的，但 `AtomCursorState` 不读它。加上去要动实体与仓储
    实现，而它的用途（"水位什么时候动过"）在这次的目标之外 —— 与其给一个
    只有 L1 才有的字段，不如三层都不给。
    """

    model_config = ConfigDict(extra="forbid")

    last_seq: int = Field(
        description=(
            "已抽过的最大 `agent_message.seq`。下一次抽取读 `seq > 它` —— "
            "这就是「抽到哪了」的全部答案。"
        ),
    )

    last_scene_name: str = Field(
        description=(
            "上一次抽取结束时处在哪个情境。它是提示词任务一「结合上一个"
            "情境」的输入，不是 L2 的场景名 —— 两者不是同一个东西"
            "（见本模块 docstring 的跨层那一段）。空串 = 还没有过情境。"
        ),
    )


class L1ExtractionRequest(BaseModel):
    """L1 提取进度：target + 会话。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget

    session_id: UUID = Field(
        description=(
            "要查的会话。水位是**会话级**的（`memory_atom_cursor` 的主键），"
            "所以它是必填。"
        ),
    )


class L1ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    session_id: UUID
    cursor: L1CursorRecord


# ============================================================
# 提取进度：L2（memory_scene_cursor）
# ============================================================


class L2CursorRecord(BaseModel):
    """
    一个会话的整合水位（`memory_scene_cursor` 的读模型）。

    ⚠️ 两个水位 + 一个时刻，三样都要看
    -----------------------------
    "整合到哪了"（第一个字段）与"上次整合是什么时候"（第三个）是**两件
    事**：后者含"跑了但没有新记忆可整合"那一次（空转也戳），所以它是触发
    侧的成本闸门（`lastL2 + 900s`）的基准，不是进度。只看第一个字段会把
    "跑过但空转"读成"从没跑过"。

    ⚠️ `last_atom_ids` 不是冗余，也别拿它当"上次整合了哪些记忆"
    ------------------------------------------------
    它只装**恰好落在水位那个时间戳上**的那一批（Postgres 的 `now()` 取的是
    事务开始时刻，一次 L1 写下的所有记忆共享同一个 `updated_time`）。它的
    作用是在取增量时排除边界重复，**不是**一份消费台账 —— 上一次整合真正
    看过哪些记忆，库里没有记（见本模块 docstring 的跨层那一段）。
    """

    model_config = ConfigDict(extra="forbid")

    last_atom_updated_time: datetime | None = Field(
        description=(
            "已经整合过的 L1 记忆的最大 `updated_time`。下次按 `> 它`"
            "（并列上下面那批 id 的排除）取增量。null = 还没有任何 L1"
            "进过画像。"
        ),
    )

    last_atom_ids: list[str] = Field(
        description=(
            "恰好落在上面那个时间戳上、已经消费过的记忆 id。"
            "**只是边界那一批**，见类 docstring。"
        ),
    )

    last_l2_at: datetime | None = Field(
        description=(
            "上一次 L2 跑完的时间（含空转那一次）。null = 这个会话从没跑过"
            "整合。"
        ),
    )


class L2ExtractionRequest(BaseModel):
    """L2 提取进度：target + 会话。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    session_id: UUID


class L2ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    session_id: UUID
    cursor: L2CursorRecord


# ============================================================
# 提取进度：L3（memory_persona_cursor）
# ============================================================


class L3CursorRecord(BaseModel):
    """
    一个 scope 的画像生成水位（`memory_persona_cursor` 的读模型）。

    ⚠️ 键是 scope，不是会话 —— 三层里唯一一个
    ------------------------------------
    L1/L2 的水位按会话记（增量是会话里的消息 / 记忆），L3 的输入
    `memory_scene` 上**没有 session_id**（画像就是跨会话聚合的产物），
    所以这条链上根本凑不出会话这个维度。请求里因此没有 session_id。
    """

    model_config = ConfigDict(extra="forbid")

    last_scene_updated_time: datetime | None = Field(
        description=(
            "已经抽象过的场景的最大 `updated_at`。null = 还没有任何场景"
            "进过画像。"
        ),
    )

    last_scene_ids: list[str] = Field(
        description=(
            "恰好落在上面那个时间戳上、已经抽象过的场景 id。与 L2 的"
            " `last_atom_ids` 同一个成因，同样**只是边界那一批**。"
        ),
    )

    last_l3_at: datetime | None = Field(
        description=(
            "上一次画像生成的时间（含空转那一次）。触发条件 2"
            "（距上次生成 >= N 轮）拿它当基准。null = 从没生成过。"
        ),
    )


class L3ExtractionRequest(BaseModel):
    """L3 提取进度：只要 target（scope 就是它）。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget


class L3ExtractionResponse(BaseModel):
    """
    L3 水位。

    把 scope 的两半**回显**出来（而不是只回 scope_kind）：L3 是三层里唯一
    按 scope 定位的，回显能让调用方确认"查的确实是这个 (agent, user|team)"，
    而 `agent_id` 恰好是最容易传错的那一半（同一个 user 在不同 duty 下是
    两份画像）。

    ⚠️ 两半都是**必填**（其中一个恒为 null），不给默认值
    --------------------------------------------
    给默认值等于允许"只回显一半" —— 而调用方恰恰要靠"另一半是 null"来确认
    scope 解对了（团队 scope 的 user_id 必须是 null，见 `SceneScope.resolve`
    那条"有 org 就丢掉 user_id"的规则）。契约（yaml）那边这两个字段也在
    `required` 里，两边说的是同一件事：**键必须在，值可以是 null**。
    """

    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    agent_id: str
    user_id: str | None
    team_id: str | None
    cursor: L3CursorRecord


# ============================================================
# 浏览（L0-L3 的记录）
# ============================================================
#
# 这一组是"看已经存下来的东西"，与上面的提取进度（看水位）是两件事：
# 水位回答"抽到哪了"，这一组回答"抽出来的长什么样"。
#
# ⚠️ 四个浏览端点都**不做召回**
# ------------------------
# 没有向量、没有关键词打分、没有融合 —— 只有等值 / 范围 / 字面包含过滤 +
# 游标分页。要语义召回走 `memory.yaml` 与 `memory_atom.yaml` 那两个工具
# 端点（它们同时是暴露给主模型的工具）。
#
# ⚠️ 分页是**不透明游标**，不是 offset
# ------------------------------
# 每个列表接口的 `cursor` 是上一页返回的 `next_cursor`，服务端解出来的是
# "上一页最后一条的排序键 + 唯一键"，据此构造 keyset 条件（见
# application/_pagination.py）。offset 分页在这里不可用：排序键都是
# **可变或非唯一**的（`updated_time` 会被合并改写、`priority` 大量重复），
# 翻页期间有人写一条就会漏掉或重复一整屏。


class Pagination(BaseModel):
    """
    一页的元信息。

    ⚠️ `limit` 回显的是**这一页实际请求的条数**，不是返回的条数
    ------------------------------------------------
    最后一页可能不足 limit。要不要据 `has_more` 收手是调用方的事 ——
    而 `has_more` 为 False 时 `next_cursor` 一定是 null，两个字段说的是
    同一件事（客户端据任一即可）。
    """

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(ge=1)
    next_cursor: str | None = Field(
        description="不透明的下一页游标；没有下一页时为 null。**不要自己构造它** —— 它与 `order` 绑定，服务端会拒绝对不上的游标。",
    )
    has_more: bool


class ConversationRecord(BaseModel):
    """
    一条已经保存的 L0 原始消息记录（不是搜索命中的片段）。

    ⚠️ 与召回端点（memory.yaml）返回的 `hits` 不是一回事
    ----------------------------------------------
    那边给的是**片段**（截断到 400 字符，见 `_assembly.SNIPPET_CHAR_LIMIT`）
    加召回分数；这里是**整条**记录、没有分数。浏览不截断：截断属于"给模型
    看的一小段"，而浏览是给人看的。

    ⚠️ 没有 `kind`
    -----------
    契约里只有 `role`（user / assistant）。因此一条 `kind=tool_activity`
    或 `notice` 的消息在这里与普通对话**长得一样**，它的 `text` 可能是空串
    （没有 text 块）。要区分得先把 `kind` 加进契约 —— 那是契约变更，不在
    实现里顺手加。
    """

    model_config = ConfigDict(extra="forbid")

    message_id: UUID
    session_id: UUID
    seq: int = Field(description="会话内消息序号，允许存在空洞。")
    role: MessageRole
    text: str = Field(
        description="完整消息正文（只拼 text 块；纯工具调用的消息是空串）。",
    )
    created_at: datetime


class FactRecord(BaseModel):
    """
    一条已经保存的 L1 原子记忆（不包含召回分数）。

    ⚠️ `timestamp_str` 是 LLM 推出来的**文本**，不可信
    --------------------------------------------
    它可能是"2018 年"这类半成品（见 `memory_atom` 那三列的注释），所以
    **没有任何端点拿它做时间范围过滤** —— 时间范围走 `updated_from` /
    `updated_to`（那是系统写的 `updated_time`）。这里只把它当展示字段带出来。

    ⚠️ 合并之后旧 id 不保证长期稳定
    ------------------------
    `update` / `merge` 是原地改写 + 真删（存活者保留 id、被吃掉的整行消失），
    所以 `record_id` 会随一次合并**消失**。它是"当前存活记录的 id"，不是
    "这条记忆的永久身份"。
    """

    model_config = ConfigDict(extra="forbid")

    record_id: UUID = Field(description="当前存活记录的 ID；合并后不保证长期稳定。")
    content: str = Field(description="原子记忆陈述，整条返回，不截断。")
    type: str = Field(description="记忆类型，例如 persona、episodic、instruction；词表可扩展。")
    priority: int = Field(ge=-1, le=100, description="-1 表示全局指令；普通优先级为 0-100。")
    scene_name: str | None = Field(description="所属情境名；可能为空。")
    version: int
    timestamp_str: str | None = Field(
        description="从提示词提取的时间线文本，**不可用作可靠的时间范围过滤**。",
    )
    updated_at: datetime


class SceneSummary(BaseModel):
    """
    一个场景文件的**索引**（不含正文）。

    ⚠️ `summary` 与 `heat` 是**按需解析** META 块得来的，不是表上的列
    ----------------------------------------------------
    这两个值由模型写在 `content` 里（见 `memory_scene_consolidate` 的
    `parse_meta`）。所以列表接口必须把正文读出来才能给出摘要 —— 它不是
    "为了省一次查询"，而是这两个字段的来源就是正文本身。

    解析不出来时**不报错**：`summary` 给空串、`heat` 给 null，内容本身
    还在（那才是要保的东西）。
    """

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(description="相对 scene_blocks/ 的场景文件名，是该场景的标识。")
    summary: str = Field(description="META 块里的摘要；空字符串表示没有摘要。")
    heat: int | None = Field(description="场景热度；无法解析时为 null。")
    version: int
    created_at: datetime
    updated_at: datetime


class SceneDetail(SceneSummary):
    """
    场景全文（含 META 块）。

    继承 `SceneSummary` 而不是重复列一遍 —— 契约那边是 `allOf`，两边说的是
    同一件事：详情就是索引加正文。
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="场景全文（含 META 块）。")


class PersonaRecord(BaseModel):
    """
    一篇画像（`memory_persona` 的一行）。

    ⚠️ `filename` 恒为 `persona.md`
    ---------------------------
    一个 scope 至多一篇画像。团队与个人的区分靠 `agent_id + (user_id|team_id)`，
    **不靠文件名**（见 `PersonaRow` 的说明）。
    """

    model_config = ConfigDict(extra="forbid")

    content: str = Field(description="整篇 persona.md（含 META 块），不截断。")
    filename: Literal["persona.md"]
    version: int
    updated_at: datetime


#: L0 浏览的排序键。`session_seq_*` 只在给了 `session_id` 时有意义
#: （`seq` 是会话内的序号，跨会话排序没有可比性）。
L0QueryOrder = Literal[
    "created_at_desc",
    "created_at_asc",
    "session_seq_asc",
    "session_seq_desc",
]

#: L1 浏览的排序键。`priority_*` 下由服务端追加唯一键做 tie-breaker
#: （优先级大量重复，没有 tie-breaker 就分不了页）。
L1QueryOrder = Literal[
    "updated_at_desc",
    "updated_at_asc",
    "priority_desc",
    "priority_asc",
]

#: L2 浏览的排序键。没有 `updated_at` 之外的时间维度（场景只有这一对时间列）。
L2QueryOrder = Literal[
    "updated_at_desc",
    "updated_at_asc",
    "filename_asc",
]


class L0QueryRequest(BaseModel):
    """
    浏览 L0 原始消息记录。

    ⚠️ 不传 `session_id` 时是**跨该个人 scope 的会话**查
    --------------------------------------------
    `agent_message` 上没有 `user_id` 列 —— 归属在**会话**上。所以跨会话
    浏览靠 join `agent_session` 取可见性，规则与单会话那条一致
    （`user_id = 目标 OR user_id IS NULL`：后者是加这一列之前的存量语义）。
    """

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    session_id: UUID | None = Field(
        default=None,
        description="可选；限定单个会话。不传则跨该个人 scope 的会话查询。传了但会话不可见 -> 404。",
    )
    roles: list[MessageRole] | None = Field(
        default=None,
        description="限定消息角色；不传表示两种角色都返回。",
    )
    time_from: datetime | None = Field(default=None, description="created_at 下界（含）。")
    time_to: datetime | None = Field(default=None, description="created_at 上界（含）。")
    text_contains: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "字面文本包含过滤（整串匹配，不执行语义或向量召回）。"
            "⚠️ 它匹配的是**消息记录里的文本块**，不含工具调用参数。"
        ),
    )
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, description="上一页返回的 next_cursor；第一页不传。")
    order: L0QueryOrder = "created_at_desc"


class L0QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    items: list[ConversationRecord]
    page: Pagination


class L1QueryRequest(BaseModel):
    """浏览 L1 原子记忆记录（等值 / 范围 / 字面包含过滤，不召回）。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    type: str | None = Field(default=None, description="精确匹配记忆类型；不传表示全部类型。")
    priority: int | None = Field(default=None, ge=-1, le=100, description="精确匹配 priority；-1 表示全局指令。")
    scene_name: str | None = Field(default=None, description="精确匹配所属情境名。")
    updated_from: datetime | None = Field(default=None, description="updated_at 下界（含）。")
    updated_to: datetime | None = Field(default=None, description="updated_at 上界（含）。")
    text_contains: str | None = Field(
        default=None,
        min_length=1,
        description="记忆正文字面包含过滤（整串匹配，不执行语义或向量召回）。",
    )
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = None
    order: L1QueryOrder = "updated_at_desc"


class L1QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    items: list[FactRecord]
    page: Pagination


class L2QueryRequest(BaseModel):
    """
    列出 L2 场景索引。

    ⚠️ 这是**列举**，不是检索
    --------------------
    `memory_scene` 没有挂载表（既没有 FTS 也没有向量索引），所以 `filename_contains`
    只做文件名的字面包含筛选 —— 而场景名是模型写的路径，按它筛的实际价值有限。
    """

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    filename_contains: str | None = Field(default=None, min_length=1, description="文件名字面包含过滤。")
    limit: int = Field(default=50, ge=1, le=100)
    cursor: str | None = None
    order: L2QueryOrder = "updated_at_desc"


class L2QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    items: list[SceneSummary]
    page: Pagination


class L2DetailRequest(BaseModel):
    """取单篇场景全文（精确读取，不执行语义检索）。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    filename: str = Field(min_length=1, description="精确场景文件名，来自 L2 列表返回的 filename。")


class L2DetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    scene: SceneDetail


class L3QueryResponse(BaseModel):
    """
    L3 画像。

    ⚠️ 没有对应画像行**不是错误**
    ------------------------
    `persona: null` 是一个有意义的答案（这个 scope 还没生成过画像），
    与"这个 scope 不存在"在库里的区别只是"有没有行" —— 而 scope 本身是
    由 target 算出来的，不是一个可以"不存在"的实体。所以这里不返回 404。
    """

    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    persona: PersonaRecord | None = Field(
        description="当前画像；null 表示此 scope 尚未生成画像。",
    )


# ============================================================
# 跨层：L1 -> L0（source_message_ids）
# ============================================================


# ⚠️ 来源消息（sources）直接用 `ConversationRecord`，不为它单开一个模型
# ------------------------------------------------------------------
# 两者要表达的字节完全一样（message_id / session_id / seq / role / text /
# created_at），而"浏览看到的消息"与"某条记忆引用的消息"在契约上是同一个
# 东西：都是 `agent_message` 的一行。曾经分成两个模型（理由猜的是两者会
# 各自演化），真动手实现时发现那只是两份同样的字段要同步改 —— 合并。


class L1SourcesRequest(BaseModel):
    """跨层溯源：这条 L1 记忆是从哪几句原话来的。"""

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    record_id: UUID = Field(description="L1 记忆 id（`memory_atom.record_id`）。")


class L1SourcesResponse(BaseModel):
    """
    一条 L1 记忆的来源消息。

    ⚠️ 三个字段一起给，缺一个都读不出结论
    --------------------------------
        content              这条记忆现在长什么样（**会被合并改写**，见下）
        sources              它的 source_message_ids 里**还取得到**的那些消息
        missing_message_ids  取不到的那些

    最后那个字段是这个端点的重点：`source_message_ids` 上没有外键、也不保证
    消息还在（见那个列的注释），而 L1 的合并会把两条记忆的 id 并起来
    （`memory_atom_service_impl._dedup` 那句并集）。所以"记得有 5 条来源、
    只找到 3 条"必须是**看得见**的事实 —— 只返回找到的那些，读起来就是
    "这条记忆只有 3 条来源"。

    ⚠️ 来源是**当下的**，不是"抽它那一刻的"
    ---------------------------------
    合并会改写 content 与 version，而 source_message_ids 是并集 —— 于是
    一条被合并过的记忆，它的来源列表覆盖的是**所有**被合并进来的那几条。
    这正是溯源想要的（"这条结论一共站在哪些原话上"），但它不是"最初那次
    抽取读了什么"。
    """

    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    record_id: UUID
    session_id: UUID
    content: str
    version: int
    sources: list[ConversationRecord]
    missing_message_ids: list[str]


# ============================================================
# 跨层：L0 -> L1（同一列的反向）
# ============================================================


# ⚠️ 反查出来的记忆也用 `FactRecord` —— 与上面 sources 合并到
# `ConversationRecord` 是同一条理由：反查的产物就是"一条记忆记录"，
# 与浏览列表里的那条没有区别，没有召回分数这一条两者也一致。


class L0AtomsRequest(BaseModel):
    """
    跨层溯源（反向）：这句原话进了哪些记忆。

    ⚠️ `session_id` 是**必填**，不是可选收窄
    --------------------------------
    `source_message_ids` 上没有 GIN 索引，包含查询（`@>`）本身会全表扫。
    带上会话之后，计划器先用 `idx_memory_atom_session` 把候选收窄到这一个
    会话的记忆（几条到几十条），包含判断只在那一小撮上做。

    代价说清楚：**跨会话反查不支持** —— 那要一张 GIN 索引，是写入侧的
    决定（每次落库都多维护一棵索引），不在这里顺手做掉。
    """

    model_config = ConfigDict(extra="forbid")

    target: RetrievalTarget
    session_id: UUID = Field(
        description="消息所属会话。同时用来把包含查询收窄到该会话的记忆。",
    )
    message_id: UUID = Field(
        description="L0 消息 id（`agent_message.message_id`）。",
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description=(
            "最多返回几条记忆。一条消息被很多条记忆引用是可能的"
            "（同一句话可以支撑好几条不同的结论）。"
        ),
    )


class L0AtomsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_kind: ScopeKind
    session_id: UUID
    message_id: UUID
    atoms: list[FactRecord]
