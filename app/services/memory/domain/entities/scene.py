"""
L2（memory_scene）与它的水位的领域实体。

三种东西，对应两张表 + 一行状态：

    SceneScope         业务键 —— 一份画像属于谁（`agent_id` + 用户 / 组织）
    SceneRow             一个场景文件（memory_scene 的写/读模型）
    SceneCursorState  会话级的整合水位（memory_scene_cursor 的读模型）

⚠️ 为什么 `SceneScope` 值得单独成一类
---------------------------------
"这份画像属于谁"这个判断在**三个地方**都要用，而三处对它的期望不同：

    写入      拼业务键（agent_id + user_id / org_id）
    读取      拼 WHERE（按 scope 列出全部场景）
    建 id     算 sha256 的输入（`scene:v1:...`）

散成三个地方的后果是**它们会漂**：某一次改了 WHERE 里的口径而没改建 id
的口径，症状是"写进去的行读不出来"（或反过来，同一份画像出现两行），
两边都不报错。所以口径收进这一个类：`key` 是它的唯一字符串投影，
`scene_id` 只吃它。

⚠️ `org_id` 有值就**只**填 org
--------------------------
这是原规格的业务规则（见 memory_l2.md）：一个会话如果带组织，它聚合出的
是**团队画像**（org + agent）；不带组织，聚合出的是**个人画像**
（user + agent）。两列**恰有一个**为 NULL，不变量由 DB 的 CHECK 兜底
（`ck_memory_scene_scope_exclusive`）—— 这里再挡一次是为了让错误发生在
构造那一刻（带上调用栈），而不是在 flush 时。

⚠️ 为什么有 `DEFAULT_AGENT_ID` 这种兜底
---------------------------------
`agent_id` 在库里是 NOT NULL（业务键的一半，见 _scope.MemoryFileColumns），
而它在链路上可能一路为空：会话的 `agent_id` 这一列本仓库还没接线
（见 domain/entities/atom.py 里那段说明），L1 记忆的 `business_agent_id`
也可能是 NULL（存量行）。取值沿用原规格的列默认值 `'default'` ——
**不要**在这里编一个新的词（比如 `"general"`）：那会让"为什么这一列
全是 general"变成一件查不出来的事。
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "DEFAULT_AGENT_ID",
    "PERSONA_FILENAME",
    "PERSONA_ID_PREFIX",
    "SCENE_ID_PREFIX",
    "SceneCursorState",
    "SceneScope",
    "SceneRow",
    "persona_id",
    "scene_id",
]


#: `agent_id` 拿不到任何值时的兜底。见模块 docstring 最后一段。
DEFAULT_AGENT_ID = "default"

#: 两种"记忆文件"行的主键前缀。带版本号是为了将来换编码时**一眼看得出**
#: 旧行是哪一代 —— 换编码意味着全量重建，而"哪些行还是 v1"是那件事的第一步。
#:
#: ⚠️ 两张表各一段，所以主键**不会撞**（它们共用同一套建 id 的算法）。
#: 这也是为什么 id 里不需要再带一个"这是哪张表"的段落：前缀就是它。
SCENE_ID_PREFIX = "scene:v1:"
PERSONA_ID_PREFIX = "persona:v1:"

#: L3 那一行的文件名**恒定**是这个值（一个 scope 只有一篇画像）。
#:
#: 与 L2 不同，画像不是"一个话题一行"而是**一整篇**，所以它没有"文件名
#: 由模型给"这一步 —— 建 id 时用这个常量，写库时也用它。
PERSONA_FILENAME = "persona.md"


class SceneScope(BaseModel):
    """
    一份画像的业务键。

    构造走 `resolve`（它实现"有 org 就是团队画像"那条规则）；直接构造也
    可以，但三个字段的取值域由 validator 把守。
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        min_length=1,
        description=(
            "画像归属的 agent。**取 L1 记忆上的 `business_agent_id`（duty_key）**"
            "——整批增量取最近的那个，一条都没有时回退会话的 `agent_id`，"
            "再没有就是 `DEFAULT_AGENT_ID`。规则与 L1 的 `_duty_key` 同一份口径。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description="个人画像的归属用户。与 `org_id` 恰有一个非空。",
    )

    org_id: str | None = Field(
        default=None,
        description="团队画像的归属组织。与 `user_id` 恰有一个非空。",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "SceneScope":

        if (self.user_id is None) == (self.org_id is None):
            raise ValueError(
                "画像的 scope 必须恰好有 user_id 或 org_id 之一："
                "有 org 的是团队画像（org + agent），没有的是个人画像"
                "（user + agent）。两个都空会让这一行谁都查不出来（不报错）。"
            )

        return self

    @classmethod
    def resolve(
            cls,
            *,
            agent_id: str | None,
            user_id: str | None,
            org_id: str | None,
    ) -> "SceneScope":
        """
        会话（或那一批记忆）的三个身份 -> scope。**唯一**实现这条规则的地方。

        ⚠️ org 有值就丢掉 user_id，不是"两个都填"
        ------------------------------------
        见模块 docstring。丢掉而不是保留成元数据，是因为两列同时有值会让
        `ck_memory_scene_scope_exclusive` 直接拒绝写入 —— 那是一个在
        flush 时才炸的错误，离"这次会话带了 org"这个原因很远。

        ⚠️ 空串按 NULL 处理
        ----------------
        `agent_session.org_id` 是 JWT 的 claim，历史数据里出现过空串
        （与 NULL 是两种不同的"没有"）。当成 NULL 处理，否则空串会造出
        一个**与 NULL 不同的** scope（`org:`），症状是同一个人的画像莫名其妙
        分成两批。
        """

        return cls(
            agent_id=(agent_id or "").strip() or DEFAULT_AGENT_ID,
            # 有 org 就是团队画像：user_id 丢掉（不是"两个都填"，见上）。
            user_id=_clean(user_id) if _blank(org_id) else None,
            org_id=_clean(org_id),
        )

    @property
    def key(self) -> str:
        """
        scope 的**唯一字符串投影** —— 建 id 用的就是它。

        带 `user:` / `org:` 前缀：两种命名空间共用一张表，不加前缀的话
        某个 user_id 恰好等于某个 org_id 时两份画像会算出同一个 id
        （静默互相覆盖）。
        """

        if self.org_id is not None:
            return f"org:{self.org_id}"

        return f"user:{self.user_id}"

    @property
    def is_team(self) -> bool:
        """团队画像（org + agent）还是个人画像（user + agent）。"""

        return self.org_id is not None


def scene_id(
        scope: SceneScope,
        *,
        filename: str,
) -> str:
    """
    场景行的主键：`scene:v1:` + sha256(scope \\0 文件名) 的 hex。

    ⚠️ 为什么是**派生**的而不是随机的
    ---------------------------
    "一个 scope 下一个文件名只有一行"这条不变量靠它成立 —— 复合唯一约束
    做不到（scope 那两列必有一列为 NULL，而 Postgres 的唯一约束不认 NULL，
    见 _scope.MemoryFileColumns）。派生键让 upsert 与乐观锁都能一次命中。

    ⚠️ `\\0` 分隔而不是拼接
    -------------------
    直接拼 `scope_key + filename` 会让 ("ab","c") 与 ("a","bc") 撞成同一个
    id。`\\0` 在文件名里不可能出现（归一化会把它去掉），所以它做分隔符是
    安全的。
    """

    return SCENE_ID_PREFIX + _digest(scope, filename)


def persona_id(scope: SceneScope) -> str:
    """
    L3 画像行的主键：`persona:v1:` + sha256(scope \\0 "persona.md") 的 hex。

    与 `scene_id` 同一个算法、不同的前缀，所以两张表的主键不会撞。文件名
    用常量 `PERSONA_FILENAME` —— 画像是一整篇，没有"模型给文件名"这一步，
    所以这里不接参数（写库时用的也是同一个常量）。
    """

    return PERSONA_ID_PREFIX + _digest(scope, PERSONA_FILENAME)


def _digest(scope: SceneScope, filename: str) -> str:
    """两处建 id 唯一的落点。改编码（加字段、换分隔符）只改这里。"""

    return hashlib.sha256(
        f"{scope.key}\0{filename}".encode("utf-8")
    ).hexdigest()


class SceneRow(BaseModel):
    """
    一个场景文件（memory_scene 的一行）。

    写回时**不带** `version` 判断用哪条语句 —— 那是服务层的事（它有库里
    读出来的那一份）。这里只负责把行搬来搬去。
    """

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="派生主键，见 `scene_id`。")
    filename: str
    content: str
    content_md5: str

    agent_id: str
    user_id: str | None = None
    org_id: str | None = None

    version: int = Field(
        default=0,
        description="乐观锁基线。写回时带上它做 `WHERE version = ?`。",
    )

    created_at: datetime | None = None
    updated_at: datetime | None = None


class SceneCursorState(BaseModel):
    """
    一个会话的整合水位（memory_scene_cursor 的读模型）。

    ⚠️ 没有对应行时返回的是**默认值**（水位 None、空 id 集），而不是 None
    ------------------------------------------------
    与 `AtomCursorState` 同一条取舍：让调用方每次判 None 等于把"还没有
    行 = 还没整合过"这条规则抄到每个调用点。仓储直接把它折成默认值。

    ⚠️ 水位为 None 与"有行但水位为 None"含义不同
    -------------------------------------
    前者 = 这个会话从没跑过 L2；后者 = 跑过、但当时没有任何 L1 进画像
    （`last_l2_at` 因此有值）。触发 A 要的 `lastL2` 是后者那个 —— 所以
    两者都存在这一行里，读的时候别只看水位。
    """

    model_config = ConfigDict(extra="forbid")

    last_atom_updated_time: datetime | None = Field(
        default=None,
        description=(
            "已经整合过的 L1 记忆的最大 `updated_time`。下次按 `> 它`"
            "（并列上 `last_atom_ids` 的排除）取增量。None = 还没有任何"
            " L1 进过画像。"
        ),
    )

    last_atom_ids: list[str] = Field(
        default_factory=list,
        description=(
            "恰好落在 `last_atom_updated_time` 上、**已经消费过**的 atom id。"
            "存在的原因是同一微秒的并发提交会被 `>` 静默跳过 —— 见 "
            "models/sqlalchemy/memory_scene_cursor.py 的模块 docstring。"
        ),
    )

    last_l2_at: datetime | None = Field(
        default=None,
        description=(
            "上一次 L2 跑完的时间（含空转那一次）。触发 A 用它算最早的"
            "下次时间：`T = max(now + 90s, lastL2 + 900s)`。"
        ),
    )


# ============================================================
# internals
# ============================================================


def _clean(value: str | None) -> str | None:
    """空串 -> None，其余去空白。"""

    if value is None:
        return None

    return value.strip() or None


def _blank(value: str | None) -> bool:
    """None 或全空白。用于判"这个 claim 到底有没有值"。"""

    return value is None or not value.strip()
