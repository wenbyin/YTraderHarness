"""
L3（memory_persona）与它的水位的领域实体。

两种东西，对应两张表（水位的表在 `memory_persona_cursor`）：

    PersonaRow          一篇画像（memory_persona 的写/读模型）
    PersonaCursorState  scope 级的生成水位（memory_persona_cursor 的读模型）

⚠️ `SceneScope` **直接复用**，不另造一个
--------------------------------
"这份画像属于谁"在两层的口径是**逐字相同的**：`agent_id` + （`user_id`
或 `org_id`，恰有一个）。L2 与 L3 是同一份 scope 的两种粒度 —— L2 把记忆
收成场景、L3 把场景收成画像，两者的业务键不可能不一样。

另造一个类会立刻产生一个分叉点：某天 L2 改了 `resolve` 的回退顺序（比如
`agent_id` 取不到时的兜底值），而 L3 那个没跟着改，症状是**同一个人有两份
画像**（一份按旧口径、一份按新口径），两边都不报错。所以这里 import 走
`scene.py` 的那一个。

⚠️ `persona_id` 也复用（来自 scene.py）
--------------------------------
主键是 `persona:v1:` + sha256(scope \\0 "persona.md")，与 L2 的 `scene_id`
同一套算法、不同前缀。建 id 这件事**只有一个落点**（`scene._digest`）。

⚠️ 水位为什么与 L2 的水位长得像但取值域不同
------------------------------------
    memory_scene_cursor.last_atom_updated_time    读 memory_atom.updated_time
    memory_persona_cursor.last_scene_updated_time 读 memory_scene.updated_at

形状一样（时间戳 + 边界 id 集 + 上次跑完的时间）是因为它们解决的是**同一
类问题**（"按写入时间取增量，且同一微秒的那批不能被 `>` 漏掉"），而这条
形状是被 Postgres 的 `now()` 语义逼出来的，不是巧合。别因为像就合并 ——
两者的键分别是会话与 scope，见 memory_persona_cursor.py 的模块 docstring。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PersonaCursorState",
    "PersonaRow",
]


class PersonaRow(BaseModel):
    """
    一篇画像（memory_persona 的一行）。

    与 `SceneRow` 同形（两张表共用 `_scope.MemoryFileColumns`），只多一个
    `filename` 恒为 `persona.md` 这条约定 —— 它仍然是个字段而不是常量，
    因为它是**表上真实存在的列**，读出来就得有个地方装。

    写回时**不带** version 判断用哪条语句 —— 那是服务层的事（它有库里读
    出来的那一份）。这里只负责把行搬来搬去。
    """

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="派生主键，见 scene.py 的 `persona_id`。")

    filename: str = Field(
        default="persona.md",
        description=(
            "恒为 `persona.md`（一个 scope 只有一篇画像）。团队与个人的"
            "区分**不靠文件名**，靠 org_id / user_id 两列 —— 这一条是"
            "2026-10-11 定稿的（腾讯那版按文件区分，本仓库不跟）。"
        ),
    )

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


class PersonaCursorState(BaseModel):
    """
    一个 scope 的生成水位（memory_persona_cursor 的读模型）。

    ⚠️ 没有对应行时返回的是**默认值**，而不是 None
    ----------------------------------------
    与 `SceneCursorState` / `AtomCursorState` 同一条取舍：让调用方每次判
    None 等于把"还没有行 = 还没生成过"这条规则抄到每个调用点。仓储直接
    把它折成默认值。

    ⚠️ `last_scene_updated_time` 为 None 与"有行但为 None"含义不同
    ----------------------------------------------------
    前者 = 这个 scope 从没跑过 L3；后者 = 跑过、但当时没有任何场景进画像
    （`last_l3_at` 因此有值）。触发条件 2 要的"上次生成时刻"是后者那个
    —— 所以两者都存在这一行里，读的时候别只看水位。

    ⚠️ **没有** `has_persona` 这种字段
    -----------------------------
    "有没有画像行"是 `memory_persona` 那件事，不是水位这件事。把它塞进
    这里会让一行状态同时回答两个问题，而它们的真相来自两张表 —— 那正是
    "两份数据会漂"的经典形状。触发逻辑自己去读 persona 行（一次主键查询）。
    """

    model_config = ConfigDict(extra="forbid")

    last_scene_updated_time: datetime | None = Field(
        default=None,
        description=(
            "已经抽象过的场景的**最大** `memory_scene.updated_at`。下次按"
            " `> 它`（并列上 `last_scene_ids` 的排除）取变化场景。None = "
            "还没有任何场景进过画像。"
        ),
    )

    last_scene_ids: list[str] = Field(
        default_factory=list,
        description=(
            "恰好落在 `last_scene_updated_time` 上、**已经抽象过**的场景 id。"
            "存在的原因是同一微秒的并发提交会被 `>` 静默跳过 —— 见 "
            "models/sqlalchemy/memory_persona_cursor.py 的模块 docstring。"
        ),
    )

    last_l3_at: datetime | None = Field(
        default=None,
        description=(
            "上一次 L3 **提取真正跑完**的时间。None = 这个 scope 从没生成过"
            "画像。\n\n"
            "⚠️ 2026-10-13：语义从「上次跑完（含空转）」收窄到「上次真提取」\n"
            "----------------------------------------------------\n"
            "原来跳过（判据不成立、什么都没做）也会推它。那件事有两个问题：\n\n"
            "  语义上  `memory_persona_cursor` 整行回答的是「这一层的提取"
            "**消费**到哪了」（另两列是 last_scene_updated_time / ids）。"
            "空转没有消费任何东西 —— 场景水位没动、没调模型、没写画像。"
            "往一张进度表里记一笔没发生的事，是错的定义。\n"
            "  功能上  task 跑 ≠ 提取跑：触发侧挂在每一次 L2 完成上、无条件"
            "排队，所以「排了任务」不代表这一层执行过。而「满 N 轮次」那条"
            "触发的基准就是它 —— 空转推它等于每跳过一次就把窗口归零，那条"
            "判据于是永远到不了。\n\n"
            "现在只有真正生成（含模型回 `no_change` —— 那也是跑完了一次提取）"
            "才推它。\n\n"
            "⚠️ 「上次**检查**是什么时候」不归本字段回答，去 task 表看\n"
            "----------------------------------------------------\n"
            "每条 L3 任务都有 created_at / finished_at —— 那才是「跑过几次、"
            "上次什么时候跑的」。一个字段兼这两个语义，正是上面那个 bug 的"
            "来源。"
        ),
    )
