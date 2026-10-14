from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


class AgentMessage(Base):
    """
    一条完整的会话消息。

    ⚠️ updated_at 是**后加的列**，且它与 created_at 目前恒等
    --------------------------------------------------
    消息在业务上仍然不可变：写下去之后没有任何代码改它（见
    domain/entities/agent_message.py 里那段说明）。这一列是为了**与另外
    两张表对齐、并让契约有一个统一的"最后有效时间"**而加的；两个写入点
    都把它填成 created_at 的值，所以今天 `updated_at == created_at` 恒
    成立。等真出现"编辑消息"的能力（契约里还没有这条端点）它才会分叉。

    已有的库必须**先补这一列、再上新代码**：

        ALTER TABLE agent_message ADD COLUMN IF NOT EXISTS
            updated_at TIMESTAMPTZ;
        UPDATE agent_message SET updated_at = created_at
            WHERE updated_at IS NULL;
        ALTER TABLE agent_message ALTER COLUMN updated_at SET NOT NULL;
        ALTER TABLE agent_message ALTER COLUMN updated_at SET DEFAULT NOW();

    create_all 只建缺失的表、不改已存在的表（见 infra/datasource/
    initializer.py）。中间那句回填不能省：存量行读出来是 NULL，而实体上
    updated_at 是非可选的 datetime —— 症状是**读历史直接 500**，且看起来
    像序列化的问题，离原因很远。

    ⚠️ reasoning_content 也是**后加的列**，但它**不需要回填**
    -----------------------------------------------------
    模型的思维链。已有的库要补一条（与 updated_at 那段同理，先跑再上新代码）：

        ALTER TABLE agent_message ADD COLUMN IF NOT EXISTS
            reasoning_content TEXT;

    与 updated_at 的区别值得记一笔：**这里没有回填句，也不该有。**
    可空列没有"存量行读出来是 NULL 而实体上是非可选"那个问题，所以少了
    最容易漏的那一句。更要紧的是**不能顺手补**：那张 UPDATE 会碰到
    `updated_at` 的 `onupdate=func.now()`，于是"`updated_at == created_at`
    恒成立"这句话就成假话了（agent_message_fts 的 docstring 正是拿这条
    当理由另建了一张表）。

    历史的思维链也没有丢：它仍在 checkpoint 的 blob 里（原始 AIMessage
    随图 state 序列化，见 graph/react/agent_node.py）。刻意**不搬**过来
    —— 那件事与本次改动是两码事，且搬它就得付上面那份 UPDATE 的代价。

    ⚠️ 列名是 reasoning_content 而不是 reasoning
    -------------------------------------------
    `reasoning` 在本仓库已经被**主循环那个 model_role** 占住了
    （`ModelRole.REASONING`，事件流里 node.started 的 model_role 与
    message.* 的 role 都是它）。同一个词在这张表上装"思维链"，读事件流
    与读库的人会分不清说的是"谁在说话"还是"它想了什么"。

    reasoning_content 还有一个好处：与 langchain / DeepSeek 侧的字段名
    **逐字相同**（`additional_kwargs["reasoning_content"]`），提取那一处
    不必做一次改名映射。

    ⚠️ 表名带 agent_ 前缀（与另外两张表一致）
    -------------------------------------
    agent_session / agent_run 带前缀是因为 `session` 这个词在本仓库里已经被
    数据库会话（AsyncSession / session_factory）占满了，不加前缀会处处歧义。
    本表跟着加前缀是另一条理由：**同属一个模块的三张表用同一套命名**，
    夹在两张 agent_* 之间的裸 `messages` 读起来像另一套东西。

    这是一次**刻意的改名**，不是照规格抄的：设计原文与 messages.yaml 里的
    组件都叫 Message / messages。改的只是库表名与 Python 类名，对外契约
    一个字没动 —— 生成的文档里组件仍是 `Message`，靠 DTO 的 title 与
    app/common/openapi.py 的改名表钉住（见 domain/schemas/agent_message.py）。

    ⚠️ 这一张表**没有任何外键**
    -------------------------
    session_id / run_id / reply_to_message_id 三列都只是"指向谁"的记录，
    不是数据库约束。取舍说清楚：

        代价    可能悬空：会话（将来有删除端点时）不会连带清掉它的消息，
                run_id 可能指向一行已经不存在的运行。
        理由    规格原文写的是 `REFERENCES sessions(id)` / `runs(id)`，而
                那两张表在本仓库根本不存在（实际是 agent_session /
                agent_run）。与其自己挑一套引用来补，不如按用户的决定
                全部去掉 —— 消息表因此不依赖另外两张表的存亡。

    今天不会有悬空行：没有删除会话或运行的端点，三列都只在
    "刚刚写过那一行"的上下文里被填。等真的出现删除路径时，这条要重新审。

    ⚠️ 属性名 `meta` 而不是 `metadata`，`message_id` 而不是列名 `id`
    --------------------------------------------------------------
    前者与另外两张表同一条理由（SQLAlchemy 的声明式基类把 `metadata` 这个
    **属性名**留给了 MetaData 对象，同名直接 InvalidRequestError）。
    后者是本仓库既有的另一种写法：列名保持规格里的 `id`，属性名用
    `message_id` 与契约对齐 —— 与 capability 表完全一样
    （app/services/capability/models/sqlalchemy/capability.py 的
    `capability_id` -> 列 `"id"`）。

    ⚠️ CHECK 与部分索引**必须写在这里**，不能只写在 models.sql
    --------------------------------------------------------
    建表走的是 `Base.metadata.create_all`（见 infra/datasource/initializer.py），
    它只认 ORM 上声明的东西，**不读 models.sql**。只在那个文件里写约束，
    真实建出来的表就没有它们 —— 而 models.sql 读起来像是有的，最坏的一种
    不一致。两处因此必须同步；models.sql 是给人读的说明 + 手工建库时的脚本。
    """

    __tablename__ = "agent_message"

    message_id: Mapped[uuid.UUID] = mapped_column(
        "id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="消息 ID。列名按规格叫 id，属性名与契约对齐叫 message_id。",
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="所属会话。无外键，见类 docstring。",
    )

    run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "产出这条消息的运行。用户消息在建运行时写下，因此也有值。"
            "无外键，见类 docstring。"
        ),
    )

    seq: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment=(
            "会话内单调递增的序号。允许有空洞（一次写入失败会留洞），"
            "翻页按它做 keyset，不按时间戳。"
        ),
    )

    role: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        comment="user / assistant，见 domain/schemas/enum.py。",
    )

    kind: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="chat",
        comment="chat / interaction / tool_activity / notice。",
    )

    business_agent_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "产出这条消息的**职责**——也就是这一轮运行用的那份职责提示词的"
            " key（duty_key，如 `stock_research`）。\n\n"
            "取值来自 `agent_run.duty_key`（建运行那一刻定下的），因此同一轮"
            "的 user / assistant / 取消公告各条消息上**是同一个值**。\n\n"
            "NULL 表示那一轮没有职责信息 —— 存量行（这一列开始写入之前建的）"
            "就是这种。**不要再把它读成\"没有 Agent 目录\"**：列名是历史命名，"
            "当初预留的是业务 Agent 的位置，现在装的是职责提示词的 key。"
        ),
    )

    content: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="内容块数组。纯文本是 [{'type': 'text', 'text': ...}]。",
    )

    # ⚠️ 后加的列，已有库要手动补一条 ALTER（可空，**不回填**），
    # 理由与列名取舍见类 docstring。
    reasoning_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "模型的思维链（CoT）。NULL = 这次调用没有思维链 —— 非推理模型、"
            "用户消息、取消公告都是 NULL。\n\n"
            "只装**可见正文之外**的那一段：content 里不会再出现推理块"
            "（见 application/message_recorder.py 的 _content_blocks），"
            "两个容器不重叠，前端因此只渲染一处。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="completed",
        comment=(
            "completed / interrupted / failed / cancelled。"
            "只表示最终结果，没有 streaming 这一档。"
        ),
    )

    client_message_id: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "客户端幂等 ID。当前永远是 NULL —— CreateRunRequest 里没有这个"
            "字段，要等将来的 POST /sessions/{id}/messages。"
        ),
    )

    reply_to_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="被回复的消息。assistant 指向本轮 user 消息，user 为 NULL。",
    )

    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="扩展元数据。列名不用 metadata，理由见类 docstring。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。写完就不再变。",
    )

    # ⚠️ 后加的列，已有库要手动补，见类 docstring 那四句。
    #
    # 为什么也带 onupdate：消息今天不可变，这条规则对**这一列**其实永远
    # 不会触发。留着它是为了将来真出现编辑路径时不必再改一次表 ——
    # 与另外两张表同一套写法，读到这里的人不必猜"为什么只有它没有"。
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment=(
            "最后更新时间。当前恒等于 created_at —— 消息写一次就不再改，"
            "两个写入点填的都是同一个时刻。见类 docstring。"
        ),
    )

    __table_args__ = (
        # ---- 取值域 ----------------------------------------------------
        # 这几条与规格里的 CHECK 一一对应。它们不是"多余的兜底"：写库的
        # 两条路径（建运行的请求事务、派发 handler）都会自己算取值，算错的
        # 表现是前端拿到一个它不认识的状态字符串。
        CheckConstraint(
            "seq > 0",
            name="ck_agent_message_seq_positive",
        ),
        CheckConstraint(
            "role IN ('user', 'assistant')",
            name="ck_agent_message_role",
        ),
        CheckConstraint(
            "kind IN ('chat', 'interaction', 'tool_activity', 'notice')",
            name="ck_agent_message_kind",
        ),
        CheckConstraint(
            "status IN ('completed', 'interrupted', 'failed', 'cancelled')",
            name="ck_agent_message_status",
        ),
        # 形状退化成字符串/对象时要在这里就断掉：契约那边是数组，前端按数组
        # 解析。放过去的话报错点会跑到前端，而那时已经离原因很远了。
        CheckConstraint(
            "jsonb_typeof(content) = 'array'",
            name="ck_agent_message_content_array",
        ),
        CheckConstraint(
            "jsonb_typeof(meta) = 'object'",
            name="ck_agent_message_meta_object",
        ),
        # ---- 唯一性 ----------------------------------------------------
        # seq 的真正护栏。分配走 COALESCE(MAX(seq), 0) + 1，而它靠 thread 锁
        # 串行化（见 application/agent_run_service_impl.py 的写入点）。锁是
        # 进程内的，装配漏配 dispatcher 时根本没有锁 —— 那时唯一约束是唯一
        # 拦住并发重号的东西。
        UniqueConstraint(
            "session_id",
            "seq",
            name="uq_agent_message_session_seq",
        ),
        # ---- 索引 ------------------------------------------------------
        # 会话历史的唯一查询形状：按会话取、按 seq 倒序翻最新一页。
        # DESC 与查询方向一致，因此不需要额外排序。
        Index(
            "idx_agent_message_session_seq",
            "session_id",
            text("seq DESC"),
        ),
        # 按运行查它产出了哪些消息（回填 reply_to / 排查某轮说了什么）。
        # 部分索引：大多数消息的 run_id 有值，但列可空，条件索引省掉无谓项。
        Index(
            "idx_agent_message_run_seq",
            "run_id",
            "seq",
            postgresql_where=text("run_id IS NOT NULL"),
        ),
        # 客户端幂等：同一个会话里一个 client_message_id 只认一条。
        # 可空列上的唯一索引必须是部分的 —— NULL 不参与唯一性，但不加
        # WHERE 会把这层意思写得含糊。
        Index(
            "uq_agent_message_client_message",
            "session_id",
            "client_message_id",
            unique=True,
            postgresql_where=text("client_message_id IS NOT NULL"),
        ),
    )
