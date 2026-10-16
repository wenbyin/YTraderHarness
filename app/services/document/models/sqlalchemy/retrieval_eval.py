"""
第一层检索评测（Retrieval Eval）的全部表。

权威定义在 `rag_eval_design/retrieval_eval_schema.sql` —— 本文件是它的
SQLAlchemy 表达。表名、列名、约束名、索引名**逐一对应**，便于两边对照。

与那份 SQL 的四处刻意偏差
------------------------
1. **状态列用 `String(32)` 而不是 `VARCHAR(24)`。** 取 32 是跟 document
   模块既有表（`document_type_definition.status` 等）一致；24 也装得下
   最长取值（`partial_failed`，14 字符），纯属口径统一。
2. **没有把 SQL 里的 `CHECK (status IN (...))` 搬过来。** 仓库既有表
   一律不写状态 CHECK，取值约束落在枚举类型上（`EvalBundleStatus` 等）
   —— 两处都写就多了一处会漂移的真相。
3. **保留了三条真正表达不变量的 CHECK**（答案偏移先后、候选至少有
   一个排名）。它们不是枚举取值，是行内字段之间的关系，数据库是唯一
   能拦住它们的地方。
4. **`metadata` 列的 Python 属性名是 `metadata_`**（列名仍是
   `metadata`）。`metadata` 是 SQLAlchemy Declarative 的保留属性名，
   与 `document_type_definition` 同一口径。

⚠️ **建表没有 Alembic。** 表由启动时的
`app/infra/datasource/initializer.py` 走 `Base.metadata.create_all` 建，
而它**只建缺失的表、不加列、不改约束**。所以这九张表要么一次定准，
要么后续改动只能手写 ALTER —— 那份 SQL 文件在这种情况下是给人执行用的
那份，不是历史记录。

这一层测什么
------------
**只测召回。** LLM 只在离线造题时出现（把源 chunk 变成"问题 + 逐字
答案"），跑评测时是纯 ID 比对 —— 没有生成、没有 LLM 判分。所以这里
没有"生成答案""判相关性"相关的列。

九张表分四组
------------
    数据集侧（离线、LLM 参与）
        retrieval_eval_bundle            一份不可变的数据集
        retrieval_eval_generation_run    一次可复现的抽样 + 造题尝试
        retrieval_eval_generation_window 一次受上下文窗口限制的 LLM 调用
        retrieval_eval_case              一条校验通过的 QA
        retrieval_eval_case_gold_chunk   该问题**全部**的金标 chunk

    评测侧（在线、无 LLM）
        retrieval_eval_run               在固定 bundle 上跑一轮
        retrieval_eval_case_result       单条 QA 的命中结果
        retrieval_eval_stage_observation 单条 QA 的逐阶段金标进出（宏观统计的来源）
        retrieval_eval_stage_candidate   逐阶段 top-N 候选与分数（诊断单个 case 用）

三个贯穿的设计取舍
------------------
**一、跨模块 id 全是快照，不建外键。** `document_type_definition_id` /
`document_version_id` / `source_chunk_id` / `chunk_id` 都不指向 document
模块的表。数据集是对**当时那份语料**的断言，源文档之后被 reprocess、
逻辑删除、甚至物理清掉，都不该改变这份断言的内容，更不该把它级联
删掉。这与 `document_chunk` 自己"归属靠推导而不是外键"是同一条纪律，
只是这里更硬：快照是**故意**不去追源的。

**二、窗口存输入快照。** `retrieval_eval_generation_window.input_chunks`
里放的是**正文本身**，不是 chunk id 列表。重试一个失败的窗口时，源
chunk 可能已经被重新切分（id 变了、内容变了），只存 id 的话重试要么
取不到、要么悄悄用了不同的正文，而产出会进同一份数据集 —— 那种不一致
事后无法分辨。存快照让"重试产出与首轮产出"可比较。

**三、候选带排名与分数，且**不要跨阶段比分数**。**

⚠️ 这条初稿写错过，实测口径记在这里：本管线暴露出来的分数**始终是
归一化到 [0,1] 的**（召回后按路归一化、融合后归一化、精排后归一化），
所以"BM25 是几十、余弦是 0..1"说的是**归一化之前的原始量纲**，不是这里
存下来的值。

即便如此仍不能跨阶段比：归一化是"除以本阶段最大值"，两个阶段的 0.9
基数不同。名字与方向两列保留着是为了将来记录下推到索引层的原始分
（`min_score` 那类过滤）—— 目前它们几乎不携带信息，见
`retrieval_eval_stage_candidate` 的详细说明。
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base


# ============================================================
# 1. Bundle —— 一份不可变的数据集
# ============================================================

class RetrievalEvalBundle(Base):
    """
    一份 QA 数据集。

    bundle 是**不可变**的：`ready` 之后改抽样量、DTD 范围、M、L 或者
    出题模型，都只能新建一份，不能改这一份。理由不是洁癖 —— 评测结果
    要能与"当时那份数据集"对上，而同一份 bundle 内容变过之后，历史
    结果就没有分母了。

    `dataset_sha256` 是导出内容的指纹，冻结在同一行上：它让"两次评测
    用的是不是同一份数据"变成一个可比较的值，而不是靠人去比对
    case_count 和日期。
    """

    __tablename__ = "retrieval_eval_bundle"

    bundle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Bundle 主键。",
    )

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "租户 ID。可空 —— 与 document 模块一样，"
            "单租户部署时这里是 NULL。"
        ),
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="数据集名称。",
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="数据集说明（给人在选择器里看的）。",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="draft",
        comment=(
            "生命周期状态，见 EvalBundleStatus。"
            "**只有 ready 可以拿去跑评测** ——"
            "partial_failed 跑出来的偏低召回率来自数据集不完整，"
            "不是检索的问题，两者混在一起无法归因。"
        ),
    )

    schema_version: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="1.0",
        comment=(
            "导出格式版本。导出物会被外部工具（ragas 等）读走，"
            "格式变过之后必须能从这一列分辨。"
        ),
    )

    selection_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "抽样配置快照：选了哪些 DTD、抽多少个 chunk、"
            "sampling_strategy、seed。"
        ),
    )

    generation_config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "造题配置快照：questions_per_chunk、chunks_per_window、"
            "max_input_tokens、model_role 等。"
        ),
    )

    dataset_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "导出内容的 SHA-256。ready 时算出并冻结。"
            "作用见类说明 —— 它让'两次评测是不是同一份数据'"
            "变成一个可比较的值。"
        ),
    )

    source_chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="实际抽中的源 chunk 数（分母，不是期望值）。",
    )

    case_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="校验通过的 QA 条数。",
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="创建人。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="最后更新时间。",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="进入终态（ready / partial_failed / failed）的时间。",
    )

    __table_args__ = (
        Index(
            "ix_retrieval_eval_bundle_tenant_status",
            "tenant_id",
            "status",
            "created_at",
        ),
    )


# ============================================================
# 2. Generation Run —— 一次可复现的抽样 + 造题尝试
# ============================================================

class RetrievalEvalGenerationRun(Base):
    """
    一次造题尝试。

    与 bundle 是一对多：同一份 bundle 可以有多次尝试（比如第一次
    全部窗口都撞了限流，重来一次）。这样"这次为什么没造出来"可以
    按运行分辨，而不是只能看到 bundle 上那个累积的计数。

    `sample_seed` 单独成列而不是塞在 config_snapshot 里：它是**复现
    的钥匙**，固定种子 + 不变的源语料应当抽出同一批 chunk —— 这是
    可验证的性质，值得是一个能被索引、能被断言的一等字段。
    """

    __tablename__ = "retrieval_eval_generation_run"

    generation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="造题运行主键。",
    )

    bundle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_bundle.bundle_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment=(
            "所属 bundle。RESTRICT 而不是 CASCADE："
            "数据集是历史断言，不该被顺手删掉。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        comment="执行状态，见 EvalJobStatus。",
    )

    sample_seed: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        comment=(
            "抽样种子。调用方没给时由服务端生成并存下来 ——"
            "存下来才有'同种子同结果'这个可验证的性质。"
        ),
    )

    config_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="本次运行的全部配置（含默认值展开后的结果）。",
    )

    model_role: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="eval.qa_generation",
        comment=(
            "出题用的模型角色，不是模型名 —— 换模型不该改代码，"
            "所以这里记角色，实际解析到的模型名记在 model_name。"
        ),
    )

    model_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "本次实际解析到的模型名。"
            "与 model_role 分开：换提示词/换模型之后要重建数据集，"
            "而'这批是哪次模型产出的'只能从这里读。"
        ),
    )

    total_chunks: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="本次抽中的源 chunk 数。",
    )

    total_windows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="切出的窗口数。",
    )

    completed_windows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="成功产出的窗口数。",
    )

    failed_windows: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="失败的窗口数。",
    )

    generated_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="累计产出的有效 case 数。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="批次级失败原因（单窗口的失败记在窗口行上）。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始时间。",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="结束时间。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        Index(
            "ix_retrieval_eval_generation_bundle",
            "bundle_id",
            "created_at",
        ),
    )


# ============================================================
# 3. Generation Window —— 一次 LLM 调用
# ============================================================

class RetrievalEvalGenerationWindow(Base):
    """
    一个上下文窗口 = 一次 LLM 调用。

    窗口是**重试的单位**。单窗口失败必须只让那一个窗口作废，而不是
    让整批陪葬（LLM 会超时、会限流、会返回不合格式的东西）；反过来，
    重试也不能把已经校验通过的 case 重造一遍、造出重复。

    因此窗口的两个字段是重试的支点：

        input_chunks     输入快照（含正文），重试时**不再回源查 chunk**
        request_sha256   请求指纹，用来判断"这次重试与上次是不是同一个请求"

    为什么输入要存正文
    ------------------
    源 chunk 可能已经被重新切分（id 与内容都变了）。只存 id 的话，
    重试要么取不到、要么悄悄换了正文，而产出会进同一份数据集 ——
    那种不一致事后无法分辨。
    """

    __tablename__ = "retrieval_eval_generation_window"

    generation_window_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="窗口主键。",
    )

    generation_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_generation_run.generation_run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属造题运行。随运行级联删除 —— 窗口没有独立价值。",
    )

    window_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment=(
            "窗口在本次运行内的序号（从 0 起）。"
            "**顺序是确定性的**，重试靠它定位，所以不能靠自增 id。"
        ),
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        comment="执行状态，见 EvalJobStatus。",
    )

    input_chunks: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "输入快照，元素形如 "
            "{chunk_id, document_type_definition_id, document_version_id, "
            "content, content_hash, token_count}。"
            "**含正文** —— 理由见类说明。"
        ),
    )

    input_token_estimate: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "输入的 token 估算值。"
            "记下来是为了让'为什么这个窗口只放了 2 个 chunk'"
            "能直接看出来，而不是去重算一遍。"
        ),
    )

    request_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment=(
            "请求指纹（系统提示 + 用户提示 + 输入）。"
            "重试时与上次比对：一致说明重试的是同一个请求，"
            "不一致说明配置/输入变过，产出不可与首轮混为一谈。"
        ),
    )

    raw_response: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "模型原始返回。**保留原始文本而不是只留解析结果** ——"
            "校验把一条题判掉时，判断'是模型的问题还是解析的问题'"
            "只能靠它。"
        ),
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="已尝试次数。重试幂等要靠它与 case 侧的唯一约束共同保证。",
    )

    valid_case_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="本窗口最终产出的有效 case 数。",
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="失败分类码（机器判断是否可重试）。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败原因（给人看的）。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始时间。",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="结束时间。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        UniqueConstraint(
            "generation_run_id",
            "window_index",
            name="uq_retrieval_eval_generation_window",
        ),
        Index(
            "ix_retrieval_eval_window_status",
            "generation_run_id",
            "status",
            "window_index",
        ),
    )


# ============================================================
# 4. Case —— 一条校验通过的 QA
# ============================================================

class RetrievalEvalCase(Base):
    """
    一条 QA。数据集的最小单位，也是评测的最小单位。

    `answer_original` 是**源 chunk 里的逐字片段**，不是模型写的答案。
    这不是风格问题：如果答案是模型生成的（"1998年" vs 原文
    "成立于1998年"），那 ground truth 就带上了模型的措辞，而检索侧
    返回的是原文 —— 比对时到底是"没召回到"还是"措辞对不上"分不清。
    存原文片段 + 偏移，比对的是位置，不是字符串。

    `source_chunk_id` 与 `answer_start_offset/end_offset` 的关系：
    前者是"这道题从哪个 chunk 出的"，后者是答案在**那个 chunk 正文
    里**的字符区间。有了区间才能验证"答案确实来自这一块"。

    ⚠️ 没有 FK 到 document_chunk，是刻意的 —— 见模块说明第一条。
    """

    __tablename__ = "retrieval_eval_case"

    eval_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Case 主键。",
    )

    bundle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_bundle.bundle_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment="所属 bundle。",
    )

    generation_window_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_generation_window.generation_window_id",
            ondelete="SET NULL",
        ),
        nullable=True,
        comment=(
            "产出它的窗口。SET NULL 而不是 CASCADE："
            "窗口是可重跑的中间记录，case 是结果 ——"
            "清理窗口不该顺手删掉数据集。"
        ),
    )

    document_type_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "**唯一**的 DTD。一条 case 只打这一个 DTD 的召回 ——"
            "不指定 DTD 的场景由 agentic_rag 自己路由，不进这一层。"
        ),
    )

    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="源 chunk 所属版本（溯源用）。",
    )

    document_retrieval_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment=(
            "评测时解析出的 retrieval 定义（**推导出来的**，不是关联 ——"
            "chunk 与 retrieval 之间没有外键，归属靠 chunker 身份比对）。"
            "可空：解析不出时不该拦住这道题。"
        ),
    )

    # ---- 源文档的身份（快照） ----
    #
    # ⚠️ 这三列不是"顺手多存点东西"，是这类题**能不能复核**的前提。
    #
    # 本语料是"一家公司一份文档"（`document.title` = 公司名，
    # `document_key` = 证券代码）。而事件正文里公司名常常不出现 ——
    # 它写的是「本公司」（实测 1477/39697 个 chunk 含这三个字）。
    #
    # 于是出题时必须把主体从**文档**取来写进问题，否则：
    #
    #     问题「本公司经审计的净资产是多少？」→ 100 家公司出 100 条
    #     逐字相同的题；而且「本年」「本公司」在索引里毫无区分度，
    #     检索会把各家公司的同类事件全捞回来。
    #
    # 但一旦问题里写了「特锐德」，**答案的正确性就只能靠"这块 chunk 属于
    # 特锐德"来保证** —— 正文里的那个数字本身不带主体。所以不记文档身份，
    # 这类题事后无法复核：看不出它该不该命中，也看不出金标对不对。
    #
    # 没有 FK：与 document_type_definition_id / source_chunk_id 同一口径
    # —— 数据集是对**当时那份语料**的断言，源文档之后被 reprocess、
    # 逻辑删除、甚至清掉，都不该改变这份断言。
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="源 chunk 所属 Document（快照）。服务层总是填它。",
    )

    source_document_title: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "源文档标题，即**主体名称**（本语料里=公司名，如「特锐德」）。"
            "可空：Document.title 本身就是可空的。"
        ),
    )

    source_document_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment=(
            "源文档的业务标识（本语料里=证券代码，如「300001.SZ」）。"
            "出题时与标题一起给模型，它才知道该把主体写成什么。"
        ),
    )

    source_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment=(
            "出题用的源 chunk（**不可变的金标 ID**）。"
            "没有 FK，见模块说明第一条。"
        ),
    )

    source_content_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment=(
            "源 chunk 的内容指纹。与 offset 一起构成'答案确实来自"
            "这一块'的验证依据。"
        ),
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="问题（将被当成用户查询词丢进检索）。",
    )

    answer_original: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment=(
            "**源 chunk 里的逐字片段**，不是模型生成的摘要 ——"
            "理由见类说明。"
        ),
    )

    answer_start_offset: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="答案在源 chunk 正文里的起始字符偏移。可空：定位不到时仍可留题。",
    )

    answer_end_offset: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="答案在源 chunk 正文里的结束字符偏移（左闭右开）。",
    )

    question_order: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        comment=(
            "同一源 chunk 内这道题的序号。"
            "它进唯一约束，让重试**不能**为同一个 chunk 造出第二条"
            "同序号的问题 —— 幂等靠这条约束兜底，不靠应用层判断。"
        ),
    )

    question_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment=(
            "问题文本指纹。让'同一 chunk 出了两道一样的问题'"
            "在库层面被拒绝，而不是靠人去翻。"
        ),
    )

    validation_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="validated",
        comment=(
            "服务端校验结论，见 EvalCaseValidationStatus。"
            "**只信模型返回的合法 JSON 是不够的** ——"
            "未知 chunk_id、答案不是逐字子串、问题重复都要在这里落下来。"
        ),
    )

    validation_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="被拒的具体原因。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        UniqueConstraint(
            "bundle_id",
            "source_chunk_id",
            "question_order",
            name="uq_retrieval_eval_case_order",
        ),
        UniqueConstraint(
            "bundle_id",
            "source_chunk_id",
            "question_sha256",
            name="uq_retrieval_eval_case_question",
        ),
        CheckConstraint(
            "answer_start_offset IS NULL"
            " OR answer_end_offset IS NULL"
            " OR answer_end_offset >= answer_start_offset",
            name="ck_retrieval_eval_answer_offsets",
        ),
        Index(
            "ix_retrieval_eval_case_bundle_dtd",
            "bundle_id",
            "document_type_definition_id",
        ),
        Index(
            "ix_retrieval_eval_case_source_chunk",
            "bundle_id",
            "source_chunk_id",
        ),
    )


# ============================================================
# 5. Gold Chunk —— 一个问题全部的金标
# ============================================================

class RetrievalEvalCaseGoldChunk(Base):
    """
    一个 chunk 被认定为某道题的**正确答案所在**。

    为什么单独一张表而不是 case 上的一个数组列
    ------------------------------------------
    这是本次设计里最关键的一处放松：逆向造的题（从 chunk 出题）可能
    在**别的 chunk** 里也有正确答案 —— 而造题时那个 chunk 未必在同一
    个窗口里（受上下文窗口所限，一次只能丢进去 N 个 chunk）。

    于是"目标只属于出题的那一块"是错的：如果检索返回了另一块也能答
    这个问题的 chunk，判成 miss 是冤枉它。这张表就是修这个的。

    但**不是所有相关 chunk 都算金标** —— 只有"独立地含有足以回答这个
    确切问题"的才算。仅仅主题相近、或被提到同一个实体都不算。所以每
    一行都带 `validation_method` 与 `evidence_excerpt`：金标有多可信
    是可分辨的。

    `source_chunk_id` 所在那一行**必然存在**（`is_source_chunk=true`）。
    """

    __tablename__ = "retrieval_eval_case_gold_chunk"

    eval_case_gold_chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="主键。",
    )

    eval_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_case.eval_case_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属 case。随 case 级联删除 —— 金标离开题目没有意义。",
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="能回答该问题的 chunk。没有 FK，见模块说明第一条。",
    )

    is_source_chunk: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment=(
            "是否就是出题用的那一块。"
            "让'这条题的命中来自源 chunk 还是来自别的 chunk'"
            "可以分开统计 —— 后者才是这次放松真正救回来的那些。"
        ),
    )

    validation_method: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="source_chunk",
        comment="凭什么认定，见 EvalGoldValidationMethod。",
    )

    evidence_excerpt: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "该 chunk 里支撑这个结论的原文片段（逐字）。"
            "给人复核用，也是'它真的能答'的证据。"
        ),
    )

    source_content_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="该 chunk 当时的内容指纹。",
    )

    judgment: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="裁定过程的原始记录（模型返回等）。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        UniqueConstraint(
            "eval_case_id",
            "chunk_id",
            name="uq_retrieval_eval_case_gold_chunk",
        ),
        Index(
            "ix_retrieval_eval_case_gold_chunk_case",
            "eval_case_id",
        ),
        Index(
            "ix_retrieval_eval_case_gold_chunk_id",
            "chunk_id",
        ),
    )


# ============================================================
# 6. Eval Run —— 在固定 bundle 上跑一轮
# ============================================================

class RetrievalEvalRun(Base):
    """
    一轮评测。

    `effective_config_snapshot` 不是冗余：DTD 的融合/精排配置是**可改
    的**，而结果是"在当时那套配置下"得出的。不存快照的话，改完 DTD
    再去读历史结果，会以为那些数字是当前配置的表现。

    `query_processor_override` 与 `effective_config_snapshot` 分开存：
    前者是**调用方要求了什么**，后者是**实际生效了什么**（含 DTD 配置
    与类默认值的合并结果）。两者相等不代表没问题，但不相等时差异就是
    答案所在。
    """

    __tablename__ = "retrieval_eval_run"

    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="评测运行主键。",
    )

    bundle_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_bundle.bundle_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment="被评测的 bundle。",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        comment="执行状态，见 EvalJobStatus。",
    )

    max_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        comment=(
            "命中判定的截断位置。"
            "**必须与最终返回条数口径一致** —— 见 cutter.top_k 的说明。"
        ),
    )

    query_processor_override: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "调用方显式传入的四段覆盖（只含**显式传过**的字段 ——"
            "pydantic 会把没传的填上默认值，直接 model_copy 会把"
            "'继承 DTD 配置'变成'重置成类默认值'）。"
        ),
    )

    effective_config_snapshot: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="实际生效的检索配置快照（按 DTD）。",
    )

    include_stage_trace: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment=(
            "是否采集逐阶段痕迹。关掉只剩最终命中结果，"
            "宏观丢因统计会退化成整体命中率。"
        ),
    )

    total_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="本轮要跑的 case 数。",
    )

    completed_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="跑完的 case 数。",
    )

    failed_cases: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment=(
            "**执行失败**的 case 数。与'没命中'是两回事 ——"
            "混在一起会让检索故障看起来像召回质量差。"
        ),
    )

    metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "聚合指标的缓存。"
            "**真值仍在 result / observation / candidate 三张表里** ——"
            "这一列只是为了让 UI 一次加载不用重算。"
        ),
    )

    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="发起人。",
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="开始时间。",
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="结束时间。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        Index(
            "ix_retrieval_eval_run_bundle",
            "bundle_id",
            "created_at",
        ),
    )


# ============================================================
# 7. Case Result —— 单条 QA 的命中结果
# ============================================================

class RetrievalEvalCaseResult(Base):
    """
    一条 case 在一轮评测里的结果。

    `gold_hit` 用 NULL 表示**没跑到**（不是"没命中"）—— 三值而不是
    两值，因为 failed 与 miss 必须分得开。把它们并成 false 的话，
    "检索挂了"会表现为"召回率下降"，而那是最容易误判成检索质量问题
    的一种假象。
    """

    __tablename__ = "retrieval_eval_case_result"

    eval_case_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="主键。",
    )

    #: 第几轮（从 1 起）。
    #:
    #: 评测是**逐轮放宽**的：第 N 轮只跑第 N-1 轮没命中的 case，并且
    #: 每轮用不同的召回/融合/裁剪参数（第 3 轮才开精排）。所以同一条
    #: case 在一轮评测里会有**多行**，每轮一行。
    #:
    #: ⚠️ 没有这一列的话，第 2 轮会被 `(eval_run_id, eval_case_id)` 的
    #: 唯一约束挡住 —— 而 `add_case_results` 是 ON CONFLICT DO NOTHING，
    #: 于是**第 2 轮一行都写不进去且不报错**，看起来像"第 2 轮没救回任何
    #: case"。
    attempt_no: Mapped[int] = mapped_column(
        SmallInteger,
        nullable=False,
        default=1,
        comment="第几轮（从 1 起）。每轮一行。",
    )

    eval_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_run.eval_run_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属评测运行。",
    )

    eval_case_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_case.eval_case_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
        comment="被评测的 case。RESTRICT：结果存在时不许删 case。",
    )

    query_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        comment="检索侧本次调用的标识（与日志/链路追踪对齐）。",
    )

    query_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "本轮**实际发给检索**的文本（改写之后）。"
            "\n\n"
            "⚠️ 它不等于 case 的 `question`：逐轮评测里每一轮可以用不同的"
            "改写器（第 1 轮日期归一化、第 2 轮大模型改写、第 3 轮不改）"
            "—— 而 LLM 改写**不可复现**。不存这一列的话，事后问「这条命中"
            "的是哪句话」只能靠翻日志，而日志会轮转。"
        ),
    )

    gold_hit: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment=(
            "**任意**一个金标 chunk 出现在返回结果里即为 true。"
            "NULL = 这一条没跑成（见类说明）。"
        ),
    )

    gold_rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment=(
            "**最好**的那个金标 chunk 的 1-based 排名。"
            "MRR 用它；多个金标时取最好的那个，不是第一个碰到的。"
        ),
    )

    retrieved_chunk_ids: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "最终结果的有序候选，元素形如 "
            "{chunk_id, rank, score, score_name, source_key, metadata}。"
            "**带上分数**：逆向问题可能别的 chunk 也答得上，"
            "只看 id 集合丢掉的信息无法事后补回。"
        ),
    )

    degraded_routes: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment=(
            "本轮**没能参与**的召回路线（元素形如 "
            "{retrieval_type, retrieval_definition_id, reason}）。"
            "\n\n"
            "⚠️ 没有这一列的话，一次**降级跑的**评测与一次正常跑的评测在"
            "数据上完全一样：某个后端超时、那条路没贡献候选，于是这条 case "
            "判成 miss —— 而 miss 会被读成「检索质量差」，实际是「那一路"
            "当时不可用」。检索的响应里本来就有 degraded_routes（README 写着"
            "「调用方应当检查它」），是评测这一层把它丢掉了。"
        ),
    )

    latency_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="本条检索耗时。",
    )

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="completed",
        comment="执行结果，见 EvalCaseResultStatus。",
    )

    error_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="失败分类码。",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="失败原因。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        # 三元组：一条 case 在一轮评测里**每轮一行**。
        UniqueConstraint(
            "eval_run_id",
            "eval_case_id",
            "attempt_no",
            name="uq_retrieval_eval_case_result",
        ),
        Index(
            "ix_retrieval_eval_case_result_run_status",
            "eval_run_id",
            "status",
        ),
    )


# ============================================================
# 8. Stage Observation —— 逐阶段的金标进出
# ============================================================

class RetrievalEvalStageObservation(Base):
    """
    金标 chunk 在**一个阶段**前后的位置。

    "宏观统计 chunk 怎么被丢的"就建在这张表上：按 (stage_key,
    source_key, drop_reason) 聚合即可得到"哪一阶段丢得最多、因为什么"。
    它刻意只记**金标**的进出，不记全部候选 —— 全量候选在
    stage_candidate 里，但那是给单个 case 诊断用的，不是聚合的来源。

    归因规则（**这张表的语义核心**）
    ------------------------------
    一个阶段只有在"进阶段前金标**还在**、出阶段后**没了**"时才能被
    判为丢弃了它。若它在更早的阶段就已经不在了，后面的阶段一律记
    `drop_reason=not_applicable`。

    不这么做的后果：金标在 dense 那一路根本没召回到，而 filter、rerank、
    cutter 每一段都"出阶段后不在"—— 于是四个阶段各记一次丢弃，宏观
    统计里 filter 会背上一大堆根本不是它造成的锅。

    并行分支同理：召回是**多路并行**的，dense 没召回到只是那一支没中，
    不能算最终失败 —— 另一支（keyword）可能命中了。
    """

    __tablename__ = "retrieval_eval_stage_observation"

    stage_observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="主键。",
    )

    eval_case_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_case_result.eval_case_result_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属 case 结果。",
    )

    stage_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment=(
            "阶段标识。用**真实存在的**阶段，不要另造一套："
            "召回按 retrieval_type 分路（recall.dense_vector、"
            "recall.keyword……），随后是 fusioner / filter / "
            "reranker / cutter。"
        ),
    )

    source_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="",
        comment=(
            "并行分支标识（哪一路召回）。"
            "非分支阶段用空串而不是 NULL —— 空串能直接进唯一约束，"
            "NULL 在唯一约束里不相等，会让同一阶段插出多行。"
        ),
    )

    executed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment=(
            "这一阶段本次是否真的跑了。"
            "没跑（未配置/未接通/后端挂了）与'跑了但没召回'是两回事，"
            "后者是检索质量，前者是能力缺口。"
        ),
    )

    input_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="进阶段的候选数。",
    )

    output_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="出阶段的候选数。",
    )

    gold_present_before: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="进阶段时金标是否在候选里。NULL = 这一阶段拿不到该信息。",
    )

    gold_present_after: Mapped[bool | None] = mapped_column(
        Boolean,
        nullable=True,
        comment="出阶段时金标是否在候选里。",
    )

    gold_rank_before: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="进阶段时金标的排名（有多个金标时取最好的）。",
    )

    gold_rank_after: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="出阶段时金标的排名。",
    )

    gold_score_before: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="进阶段时金标的分数。**只在同一阶段内可比**。",
    )

    gold_score_after: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="出阶段时金标的分数。",
    )

    gold_score_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "分数名，例如 cosine_similarity / bm25 / rrf_score / "
            "reranker_score。**没有它，分数就不可解释** ——"
            "同一个 0.7 在不同阶段含义完全不同。"
        ),
    )

    gold_score_direction: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="分数方向，见 EvalScoreDirection。",
    )

    drop_reason: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "受控词表，见 EvalDropReason。"
            "NULL = 这一阶段没有丢（金标仍在或本就不适用）。"
        ),
    )

    details: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="阶段特有的补充信息。",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        UniqueConstraint(
            "eval_case_result_id",
            "stage_key",
            "source_key",
            name="uq_retrieval_eval_stage_observation",
        ),
        Index(
            "ix_retrieval_eval_stage_observation_stage",
            "stage_key",
            "source_key",
            "drop_reason",
        ),
        Index(
            "ix_retrieval_eval_stage_observation_result",
            "eval_case_result_id",
        ),
    )


# ============================================================
# 9. Stage Candidate —— 逐阶段的 top-N 与分数
# ============================================================

class RetrievalEvalStageCandidate(Base):
    """
    一个阶段进出的 **top-N 候选**（默认 5），带分数。

    与 stage_observation 的分工
    --------------------------
        stage_observation   只记金标的进出 —— 少、可聚合，宏观统计靠它
        stage_candidate     记 top-N 全部候选 —— 多、只用于诊断单个 case

    为什么两个都要
    --------------
    只记"金标没了"解释不了"为什么没了"。看到 top5 的实际分数与排名，
    才能判断是阈值定高了（0.29 卡在 0.3 上）还是排序本身错了。
    这是"逆向提问可能别的 chunk 也有正确答案"这条需求的落点：top5
    里有别的能答题的 chunk、而它不在 gold 集合里，正是需要人回头看
    的场景。

    为什么记 before/after 两套排名与分数
    -----------------------------------
    阈值的丢弃只能靠对比发现：进阶段 rank=3、score=0.29，出阶段没了，
    而阈值是 0.3 —— 一眼看得出。只留一套的话，"被 threshold 砍掉"
    与"被 top_k 截掉"无法区分。

    ⚠️ 关于分数的**实测口径**（写下来是因为我一开始搞错了）
    --------------------------------------------------
    本管线里流动的 `RetrieveCandidate.score` **始终是归一化到 [0, 1]
    的分数**，且一律 higher-is-better —— 召回后按路归一化、融合后归一化、
    精排后归一化，三处都调 `normalize_scores`（见 retrieve/base.py 的
    "分数的不变量"那一段与它的三处调用点）。

    所以 `score_name_*` 与 `score_direction_*` 这两列**目前几乎不携带
    信息**：方向恒为 higher_is_better，名字恒指"该阶段的归一化分"。

    它们**保留着是刻意的**，理由有两条：
        - 原始量纲确实不同（BM25 可到几十、余弦 [-1,1]、RRF 约 0.016），
          而 `retrieval_config.min_score` 那一类**下推到索引层**的过滤
          用的就是原始分。将来若要记"被 min_score 砍掉的那个候选的原始
          分"，这两列就是它的位置。
        - 列可空，不存在"必须填一个假值"的问题。

    ⚠️ 但**不要**因此以为可以跨阶段比分数：它们现在同为 [0,1] 只是因为
    都被归一化了，而"按各自最大值归一化"意味着两个阶段的 0.9 基数不同。
    """

    __tablename__ = "retrieval_eval_stage_candidate"

    stage_candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="主键。",
    )

    eval_case_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "retrieval_eval_case_result.eval_case_result_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        comment="所属 case 结果。",
    )

    stage_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="阶段标识，与 stage_observation 同口径。",
    )

    source_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="",
        comment="并行分支标识；非分支阶段用空串。",
    )

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        comment="候选 chunk。没有 FK，见模块说明第一条。",
    )

    rank_before: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="进阶段时的 1-based 排名。",
    )

    rank_after: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="出阶段时的 1-based 排名。NULL = 出阶段时已被丢。",
    )

    score_before: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="进阶段时的分数（配 score_name_before 才有意义）。",
    )

    score_after: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="出阶段时的分数。",
    )

    score_name_before: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="进阶段时的分数名，例如 cosine_similarity / bm25。",
    )

    score_name_after: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="出阶段时的分数名。",
    )

    score_direction_before: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="进阶段时的分数方向，见 EvalScoreDirection。",
    )

    score_direction_after: Mapped[str | None] = mapped_column(
        String(16),
        nullable=True,
        comment="出阶段时的分数方向。",
    )

    selected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="是否被这一阶段选中/保留。",
    )

    drop_reason: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="被丢的原因，见 EvalDropReason。",
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        comment=(
            "候选的补充信息。"
            "注意 Python 属性名用 metadata_，列名仍是 metadata ——"
            "与 DTD 表同一口径（metadata 是 SQLAlchemy 的保留属性名）。"
        ),
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="创建时间。",
    )

    __table_args__ = (
        CheckConstraint(
            "rank_before IS NOT NULL OR rank_after IS NOT NULL",
            name="ck_retrieval_eval_stage_candidate_has_rank",
        ),
        UniqueConstraint(
            "eval_case_result_id",
            "stage_key",
            "source_key",
            "chunk_id",
            name="uq_retrieval_eval_stage_candidate_chunk",
        ),
        Index(
            "ix_retrieval_eval_stage_candidate_stage_rank",
            "eval_case_result_id",
            "stage_key",
            "source_key",
            "rank_after",
            "rank_before",
        ),
        Index(
            "ix_retrieval_eval_stage_candidate_chunk",
            "eval_case_result_id",
            "chunk_id",
        ),
    )
