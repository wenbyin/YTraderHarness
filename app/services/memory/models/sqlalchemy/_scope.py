"""
挂载表共用的 **scope 列**：L0 的消息索引一对、L1 的记忆索引一对，
外加 L2/L3 画像两张表共用的一组。

    MemoryScopeColumns        挂 agent_message 的两张（关键词 / 向量）
    MemoryAtomScopeColumns  挂 memory_atom 的三张（本体 + 关键词 / 向量）
    MemoryFileColumns          L2 memory_scene / L3 memory_persona 两张

分开写而不是合并成一个：两组的列集与过滤口径都不同。理由逐条写在各自
的类 docstring 里。

为什么提出来
----------
`agent_message_fts` 与 `agent_message_vec` 各自是独立的一张表、由不同的写入
时机填（一个要分词、一个要过 embedding 模型），但**它们过滤时用的那组列
完全相同**。补一份复制粘贴的话，"scope 到底由哪几列构成“就有了两个答案，
而这类分叉的症状是检索少召回几条、且不报错。

因此列定义在这里写一次，两个模型各自继承。

⚠️ org_id 存下来但**不参与过滤**
-----------------------------
`agent_session.org_id` 的 docstring 写得很明确：

    "用途只有一个：**团队 / 组织归属的记录**。它不参与鉴权（那是 groups
     的事），也不参与可见性过滤（过滤键仍然是 user_id）。"

所以本模块把它当**元数据**带上（将来真做团队协作时不用回表），但检索时
**不要**拿它做 WHERE 条件 —— 那样会漏掉 org_id 为 NULL 的行，而 NULL 是
"这次登录的 token 里没有组织信息”，是**允许的**状态，不是漏写。

⚠️ user_id 的过滤规则必须与 session 模块逐字一致
--------------------------------------------
`agent_session.user_id` 的过滤条件不是 `= 当前用户`，而是：

    user_id = 当前用户 OR user_id IS NULL

NULL 是刻意的存量语义（加那一列之前的会话对谁都可见）。记忆检索如果写成
严格的等值，同一个用户会看到“消息列表里有、记忆检索搜不到”——两边对同一
份数据给出不同答案，而用户只会认为搜索坏了。规则跟着那边走。

⚠️ 这些列是**冗余**的，不是权威
------------------------
权威始终是 `agent_session`（user_id / org_id / thread_id）与
`agent_message`（role / created_at 等）。这里存一份是为了
让一轮召回在单表上做完：否则每次检索都要先 join 会话表，把“两个索引扫描"
变成”三表连接“，而收益只是省掉一次写入时的回表。

代价说清楚：会话归属变了（今天的接口不允许改，将来若允许）就要同步刷
这两张表 —— 写入方是 memory.l0.extract 任务，它按 run_id 重跑即可重建。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

__all__ = [
    "MemoryAtomScopeColumns",
    "MemoryScopeColumns",
    "MemoryFileColumns",
]


class MemoryScopeColumns:
    """
    挂在 `agent_message` 上的索引表共用的那一组列。

    继承它的模型**必须**自己声明 `message_id` 为主键 —— 这里不声明，
    因为两张表的键虽然同名，理由不同（见各自模块的说明）。
    """

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "所属会话。等价于 agent_message.session_id，也是 scope=session "
            "的过滤键。"
        ),
    )

    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "产出这条消息的运行（agent_message.run_id）。可空 —— "
            "存量行与“直接喂进库”的测试行都没有它。\n\n"
            "它同时也是**重建索引的抓手**：任务按 run_id 重跑就能把这一轮"
            "的行全部重写。"
        ),
    )

    seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment=(
            "会话内序号（agent_message.seq）。排序与“取邻居”都按它，"
            "不按时间戳 —— 同一事务里写的多条消息时间戳可能完全相同。\n\n"
            "允许有空洞。"
        ),
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="user / assistant。取值域与 agent_message.role 的 CHECK 一致。",
    )

    # ⚠️ 这里**没有** kind 与 status —— 它们是刻意删掉的
    # -------------------------------------------------
    # 两张表原先都带这两列，后来去掉了，判据是”有没有人读“：
    #
    #     kind     过滤已经挪到写入侧（`_indexable` 只放 kind=chat 进来），
    #              于是这一列**按构造恒定**，一个只有一个取值的列
    #     status   从来没有消费者。而它记的是”那一轮的结局“，源表
    #              `agent_message.status` 是同一个值
    #
    # 要读它们就回 `agent_message` 读 —— `message_id` 就在本表的主键上，
    # 而索引表存的是**派生**数据，能回源表拿到的东西不该在这里放第二份
    # （第二份会漂，且漂了没人知道）。
    #
    # ⚠️ 但下面那几个 scope 列**必须**留着，别按同一条判据顺手删掉：
    #     session_id / user_id / role / created_at 是**过滤键**。
    #     它们在这里的意义是”让过滤和 ANN 扫描在同一张表上做" ——
    #     挪到源表去过滤，就得每次召回都 join 出去，而带过滤的向量搜索
    #     是个两难：预过滤会切碎 HNSW 的候选集（漏召回），后过滤要多取
    #     好几倍才能凑够 limit。kind/status 从来没参与过滤，所以它们不
    #     属于这一类。

    business_agent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "这一轮的职责提示词 key（duty_key），来自 agent_message 的同名列。"
            "列名是历史命名。NULL = 那一轮没有职责信息（存量行）。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话归属用户 —— Casdoor 的 sub，与 agent_session.user_id 同口径"
            "（String(255)）。\n\n"
            "⚠️ 过滤条件是 `user_id = 当前用户 OR user_id IS NULL`，不是等值"
            " —— NULL 是刻意的存量语义，见模块 docstring。"
        ),
    )

    org_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话所属组织（JWT 的 org claim），与 agent_session.org_id 同口径。\n\n"
            "⚠️ **元数据，不参与过滤** —— 理由见模块 docstring。存它是为了"
            "将来做团队协作时不必回表重建索引。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment=(
            "消息被记录的时间（agent_message.created_at）。time_from / "
            "time_to 两个界都作用在它上面。\n\n"
            "⚠️ 是**记录时间**，不是正文里提到的时间 —— 今天写的一段回忆录，"
            "落在这个窗口里。"
        ),
    )


class MemoryAtomScopeColumns:
    """
    L1 的三张表（`memory_atom` 本体 + `memory_atom_fts` / `memory_atom_vec` 两张挂载表）共用的
    scope 列。

    ⚠️ 与 MemoryScopeColumns 不是同一组，别合并
    ----------------------------------------
    上面那组挂在**消息**上，带 seq / role / run_id / created_at —— 那些是
    "一条消息在会话里的位置“。L1 的行挂的是**提炼出来的记忆**：没有序号、
    没有角色，而且召回条件也不一样（去重按会话收窄，**没有时间窗**）。

    两边重合的四列（session_id / user_id / org_id / business_agent_id）
    是同一套口径，逐条理由见上面那组，这里不重复。

    多出来的一列是 `task_id` —— L1 独有、**当前空置**的预留位，见它的列
    注释。

    ⚠️ 过滤只按 session_id，其余是元数据
    ---------------------------------
    dedup 的候选池是**会话级**的（原规格：五个维度全收窄 → 等价于会话级）
    —— org / agent 两个维度由会话隐含。把它们也写进 WHERE 只会多一个"该列
    为 NULL 就少召回一条"的机会，而 NULL 在这些列上是**允许**的状态
    （存量行），不是漏写。

    ⚠️ 代价：跨会话的同一事实**不会**合并
    -------------------------------
    同一个用户在多个会话里各说一次同一件事，就会留下多份（2026-10-10 实测：
    5 个会话 5 份），而检索那条路是 `scope=user`，于是一次召回多份重复 ——
    主模型会如实汇报"这条记录在多处重复出现"。

    这是**刻意的取舍**（原规格就是这么做的），不是 bug。要跨会话合并是
    另一条改动（池子放到用户级），但那就不是原规格的行为了。

    所以本组里只有 session_id 是**过滤键**，另外三列是带上来的元数据
    （与 MemoryScopeColumns 对 org_id 的立场一致）。

    ⚠️ user_id 仍然是可空的，过滤规则跟着 session 模块走
    ------------------------------------------------
    规则是 `user_id = 当前用户 OR user_id IS NULL`，不是等值 —— 见上面
    那组的注释。L1 的检索如果写成严格等值，同一个用户会出现”消息列表
    里有、记忆搜不到“，而用户只会认为搜索坏了。
    """

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "这条记忆是从哪个会话的对话里提炼出来的。**唯一参与过滤的"
            " scope 列**（去重的候选池按它收窄，删除也按它放行）。"
        ),
    )

    business_agent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "主 agent 的职责（duty_key）—— 这条记忆是为那份职责而记的。\n\n"
            "⚠️ 与 L0 索引表同名列**同一个口径**（都装 duty_key），但粒度"
            "不同：那边一列一消息；这里一条记忆跨很多轮，取的是这批增量"
            "讲完时在履职的那份职责。\n\n"
            "⚠️ 元数据，不参与过滤 —— 会话已经隐含了它。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "记忆归属用户（Casdoor 的 sub）。\n\n"
            "⚠️ 过滤条件是 `user_id = 当前用户 OR user_id IS NULL`，"
            "不是等值 —— 见类 docstring。"
        ),
    )

    org_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "会话所属组织（JWT 的 org claim）。\n\n"
            "⚠️ **元数据，不参与过滤** —— 理由见类 docstring。"
        ),
    )

    task_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "⚠️ **当前恒为空**，是留给 work_* 四类记忆（work_fact / "
            "work_task / work_method / work_artifact）的预留位。\n\n"
            "那一套要靠「工作任务」的标识判断 work 状态，而本仓库目前没有"
            "这个概念（用户 2026-10-10 定：保留列、空置即可）。\n\n"
            "⚠️ 在它被真正填上之前**绝不能进 WHERE** —— 现存行全是 NULL，"
            "加过滤等于一条都搜不到，而且不报错。\n\n"
            "⚠️ 别把它与 `tasks.task_id`（后台任务主键）混起来：同名不同物。"
        ),
    )


class MemoryFileColumns:
    """
    L2 `memory_scene` 与 L3 `memory_persona` 共用的那一组列。

    ⚠️ 与上面两组不是一回事，别合并
    ---------------------------
    上面两组挂在**会话里的东西**上（消息 / 原子记忆），过滤键是 session_id
    或 user_id。这两张表挂的是**画像文件**：一行 = 一个 `.md` 文件，
    归属由 `(agent_id, user_id|org_id)` 决定，**与具体会话无关**（跨会话
    聚合出来的）。所以这里既没有 session_id，也没有 seq / role / task_id。

    ⚠️ 没有虚表，也没有召回路径
    ----------------------
    L0/L1 都挂了关键词 + 向量两张派生表，因为它们是”被检索的“。L2/L3 不是：
    它们是被**拼进提示词**的（场景清单 + 全文），当前没有任何检索入口。
    真要做（按场景名/内容搜画像）再加挂载表 —— 那是派生数据，随时能从
    本表重建。

    ⚠️ 主键是**内容派生**的 hash，不是复合唯一约束
    -------------------------------------------
    自然的业务键是 `(agent_id, user_id|org_id, filename)`，但**不能**写成
    `UniqueConstraint(agent_id, user_id, org_id, filename)`：scope 那两列
    **恰有一列恒为 NULL**（个人 / 团队二选一），而 Postgres 的唯一约束把
    NULL 视为互不相同 —— 那个约束实际上一条冲突都挡不住，却看起来像是
    在挡。症状是同一个 scope 下同一个 filename 悄悄出现两行。

    所以键由应用侧算成 `scene:v1:{sha256(scope_key \\0 filename)}`（L3 的
    画像行用 `persona:v1:`，同一套算法、不同前缀），
    用主键的唯一性来保证”一个 scope 下一个文件名只有一行“（原规格也是
    这么做的，见 memory_l2.md 的 `ProfileRecord.id`）。代价说清楚：
    改文件名 = 换一行（旧行要显式删，MERGE 那条路径就是这么做的），
    而不是原地改名。

    ⚠️ 没有外键、没有 CHECK 在 ORM 上的例外
    ---------------------------------
    与 memory_atom 同一取舍（见那个模块的 docstring）：agent_id / user_id
    都只是”指向谁“的记录，存亡由写入方负责。CHECK 与索引**在 ORM 与
    models.sql 上各有一份**，两处必须逐字一致。

    ⚠️ 没有 summary / heat 列
    ----------------------
    它们由模型写在 `content` 的 `-----META-START-----` 块里。提到列上会
    造出**第二个**”这个场景讲什么“的答案，而要它俩同步就得每次写入都解析
    一遍。读场景清单时按需解析 content（见提示词模块
    `prompts/memory_scene_consolidate.py` 的 `parse_meta` —— 它定义在
    那份模板旁边）—— 一行只有一个真相。
    """

    filename: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "场景文件的名字。L2 是相对 `scene_blocks/` 的路径（本仓库里"
            "就是纯文件名，不带目录）；L3 恒为 `persona.md`。\n\n"
            "⚠️ 它是**给模型看的身份**：模型的输入用文件名 + 全文，输出用"
            "文件名指认要改写哪一个。所以工程侧必须先归一化（见 "
            "application/memory_scene_service_impl.py 的 "
            "`_normalize_filename`），否则模型写个带空格的路径就落到新行上。"
        ),
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "整篇 markdown 内联。这就是”没有表只有文件“那个印象的来源："
            "在文件平面它是一个文件，在行平面它是行里的一整段字符串。\n\n"
            "⚠️ 语言跟随用户消息；META 块（created/updated/summary/heat）"
            "也在里面 —— 见类 docstring 最后一段。"
        ),
    )

    content_md5: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment=(
            "content 的 md5（hex）。**只用来短路写回**：与库里一致就跳过"
            "写入、也不推 version。\n\n"
            "没有它的话，模型每一轮把同一个场景原样吐回来都会把 version "
            "推一格，于是”版本号“变成”跑过几轮“的同义词，乐观锁的基线"
            "失去意义（冲突永远发生，或永远不发生，取决于谁先读）。"
        ),
    )

    agent_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment=(
            "画像归属的 agent —— 业务键的一半。取**L1 记忆上的 "
            "business_agent_id（duty_key）**：整批增量取最近的那个，一条"
            "都没有时回退 `agent_session.agent_id`（与 L1 的 `_duty_key` "
            "同一规则）。\n\n"
            "⚠️ NOT NULL 是刻意的（与 memory_atom 的 business_agent_id 可空"
            "不同）：这里是**业务键的一部分**，NULL 会让”这个 scope 是谁的“"
            "变成要看另一张表才知道，而键必须在写入那一刻就定死。"
        ),
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "个人画像的归属用户（Casdoor 的 sub）。\n\n"
            "⚠️ 与 org_id **恰有一个为 NULL** —— 那个判据是"
            "「提取它的会话有没有 org」：没有 org 聚合出的是个人画像"
            "（user + agent），有 org 聚合出的是团队画像（org + agent）。"
            "不变量由 CHECK 兜底，见各自模型的 `__table_args__`。\n\n"
            "⚠️ 与 memory_atom.user_id 的过滤语义**不同**：那边是"
            "`= 当前用户 OR IS NULL` 的可见性口径（NULL 是存量语义），"
            "这边 NULL 是”这是团队画像“的**业务含义**。别照搬那套 WHERE。"
        ),
    )

    org_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "团队画像的归属组织（JWT 的 org claim）—— 与 agent_session.org_id "
            "同口径。\n\n"
            "⚠️ 与 memory_atom.org_id 的立场**相反**：那边它是”不参与过滤的"
            "元数据“，这边它是**业务键的一半**（团队画像就按它聚合）。"
            "同一个名字在两张表上是两种角色，改动时别顺手统一。"
        ),
    )

    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default="0",
        comment=(
            "**乐观锁基线**：每次成功写回 +1。\n\n"
            "写入方读出行时拿到它，写回时带上 `WHERE version = 读到的值`；"
            "影响行数为 0 就是”这中间有人改过“，交给任务重入（见 "
            "application/memory_scene_service_impl.py 的 `_apply_writes`）。\n\n"
            "⚠️ 它与 memory_atom.version 的语义不同：那边是”这条记忆被改写"
            "了几次“的生命史，这边只是**并发控制的令牌**。所以 content 没变"
            "时不推它（见 content_md5）。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="这一行第一次写入的时间。后续重写不动它。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "最后一次写回的时间。\n\n"
            "⚠️ `onupdate=func.now()` 只对 ORM 的 UPDATE 生效；core 的 "
            "`ON CONFLICT DO UPDATE` 要显式写（见 scene_impl）。"
        ),
    )
