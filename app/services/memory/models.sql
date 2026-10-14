-- Memory 模块的两张挂载表 —— agent_message 的关键词 / 向量索引。
--
-- ⚠️ 建表实际走的是 ORM，不是本文件
-- ---------------------------------
-- `initialize_database` 用的是 `Base.metadata.create_all`
-- （见 infra/datasource/initializer.py），它只认 ORM 上声明的东西，
-- **不读**本文件。因此下面两张表的 CHECK 与索引在 ORM 上**也有一份**
-- （models/sqlalchemy/agent_message_fts.py 与 agent_message_vec.py），
-- 两处必须同步。
--
-- 只改这里 = 真实建出来的表没有那些约束，而本文件读起来像是有的 ——
-- 最坏的一种不一致。
--
-- ⚠️ 本文件里的列顺序与 ORM **不一致**（实测：ORM 先把类自己声明的列
--    排在前面 —— message_id / 正文 / 派生列，然后才是 mixin 的 scope 列）。
--    列顺序不影响语义，这里按"可读性"排：先 key，再 scope，再正文，
--    最后派生列。**约束与索引的名字必须逐字一致** —— 那才是有语义的部分。
--
-- ⚠️ agent_message_vec 依赖 pgvector 扩展，装的顺序是死的
-- -------------------------------------------------------
--    ① 装服务端扩展文件（vector.dll → lib，vector.control + *.sql →
--       share/extension），然后**重启 PG**（pg_available_extensions
--       要重启才会刷新）
--    ② CREATE EXTENSION vector;（在 ytrader_agent 这个库里，扩展是
--       per-database 的）
--    ③ 才把 app.services.memory.models.sqlalchemy 加进 initializer.py
--       的 import_all_models 列表
--
-- 反过来做拿到的是一个起不来的应用，报错是
--     type "vector" does not exist
-- 而它指向模型定义，离真正的原因（扩展没装）很远。
--
-- ⚠️ 为什么没有 agent_message_vec / agent_message_fts 的外键
-- ----------------------------------------------------
-- 与 agent_message 自己同样的取舍（见它类 docstring）：三列
-- (message_id / session_id / run_id) 都只是"指向谁"的记录。索引行的
-- 存亡由写入方负责 —— memory.extract 任务按 run_id 重写。


-- ============================================================
-- 关键词索引
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_message_fts (
    message_id        UUID PRIMARY KEY,

    -- ---- scope 列（见 models/sqlalchemy/_scope.py）----------------
    -- 只有**参与过滤/排序**的列在这里。kind / status 曾经也有，
    -- 后来删了：kind 按构造恒为 chat（过滤在写入侧做），status 没有
    -- 消费者 —— 要读它们回 agent_message 读，message_id 就在主键上。
    session_id        UUID         NOT NULL,
    run_id            UUID,
    seq               BIGINT       NOT NULL,
    role              VARCHAR(16)  NOT NULL,
    business_agent_id VARCHAR(128),
    -- NULL 是刻意的存量语义（加这一列之前的会话对谁都可见），
    -- 过滤条件因此是 `user_id = 当前用户 OR user_id IS NULL`，
    -- **不是**等值 —— 规则与 agent_session 逐字一致。
    user_id           VARCHAR(255),
    -- ⚠️ 元数据，**不参与过滤**（agent_session.org_id 的 docstring：
    -- "不参与鉴权…也不参与可见性过滤（过滤键仍然是 user_id）"）。
    org_id            VARCHAR(255),
    created_at        TIMESTAMPTZ  NOT NULL,

    -- ---- 正文与派生列 ---------------------------------------------
    -- 分词后的词元串（空格连接，由应用侧分词器产出）。不是给人读的。
    tokens            TEXT         NOT NULL,
    -- 原文。展示与高亮从这一列出，与 tokens 由同一次调用产出。
    original          TEXT         NOT NULL,
    -- 生成列：to_tsvector 的**两参**形式是 IMMUTABLE 的，所以能这么写。
    -- 一参形式依赖 default_text_search_config，是 STABLE，用在生成列里
    -- 会直接报错。
    tsv               TSVECTOR     GENERATED ALWAYS AS
                                   (to_tsvector('simple'::regconfig, tokens))
                                   STORED,

    CONSTRAINT ck_agent_message_fts_seq_positive
        CHECK (seq > 0),
    CONSTRAINT ck_agent_message_fts_role
        CHECK (role IN ('user', 'assistant')),
    CONSTRAINT ck_agent_message_fts_tokens_not_blank
        CHECK (tokens <> '')
);

-- 召回主索引。
CREATE INDEX IF NOT EXISTS idx_agent_message_fts_tsv
    ON agent_message_fts USING GIN (tsv);

-- scope=session 的形状。
CREATE INDEX IF NOT EXISTS idx_agent_message_fts_session_seq
    ON agent_message_fts (session_id, seq);

-- scope=user + 时间窗的形状。
CREATE INDEX IF NOT EXISTS idx_agent_message_fts_user_created
    ON agent_message_fts (user_id, created_at DESC);


-- ============================================================
-- 向量索引（需要 pgvector）
-- ============================================================

-- CREATE EXTENSION IF NOT EXISTS vector;   -- ← ② ：建表之前先跑这句

CREATE TABLE IF NOT EXISTS agent_message_vec (
    message_id        UUID PRIMARY KEY,

    -- ---- scope 列（与 fts 表同一组，理由见 _scope.py）--------------
    session_id        UUID         NOT NULL,
    run_id            UUID,
    seq               BIGINT       NOT NULL,
    role              VARCHAR(16)  NOT NULL,
    business_agent_id VARCHAR(128),
    user_id           VARCHAR(255),
    org_id            VARCHAR(255),
    created_at        TIMESTAMPTZ  NOT NULL,

    -- ---- 正文 -----------------------------------------------------
    -- bge-small-zh-v1.5，512 维，已归一化。与 fts 表的 original
    -- **必须是同一段文本**，否则两路召回在回答不同的问题。
    embedding         vector(512)  NOT NULL,

    CONSTRAINT ck_agent_message_vec_seq_positive
        CHECK (seq > 0),
    CONSTRAINT ck_agent_message_vec_role
        CHECK (role IN ('user', 'assistant'))
);

-- 召回主索引。vector_cosine_ops 必须与查询里的 `<=>` 一致：
-- 写成 l2_ops 而查询用 `<=>` 时 Postgres 会**忽略**这个索引改用顺序
-- 扫描 —— 结果正确、只是慢，所以这个坑不会报错。
CREATE INDEX IF NOT EXISTS idx_agent_message_vec_embedding_hnsw
    ON agent_message_vec USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_agent_message_vec_session_seq
    ON agent_message_vec (session_id, seq);

CREATE INDEX IF NOT EXISTS idx_agent_message_vec_user_created
    ON agent_message_vec (user_id, created_at DESC);


-- ============================================================
-- 运维备注
-- ============================================================

-- ⚠️ 改了分词器（app/services/memory/tokenization.py）就要重建 fts 表
-- ------------------------------------------------------------------
-- 倒排只认写入时的切法。分词规则一改，新写入的 query 与旧索引里的词元
-- 互不可见 —— 症状是"明明有这条却搜不到"，**不报错**。
--
--     DELETE FROM agent_message_fts;
--
-- 然后重跑索引任务。
--
-- ⚠️ 全量重跑目前**没有入口**：memory.extract 是按 run_id 一轮一轮索引
--    的，没有"把库里的消息全部重索引一遍"的命令。真要重建时，按
--    agent_message 逐行喂 index_text + embedder 写回即可 —— 这张表是
--    纯派生的，删干净再从消息重建是安全的。写在这里，是为了不让读到
--    这句的人以为已经有一条现成的命令。
--
-- 向量那侧同理：换 embedding 模型（维度可能不变！）也要重建，
-- 否则新旧向量落在同一个索引里，排序没有意义且不报错。


-- ============================================================
-- L1：从对话里提炼出来的记忆
-- ============================================================
--
-- 四张表，三种角色：
--
--     memory_atom   记忆**本体** —— 权威数据（不是派生表，见下）
--     memory_atom_fts       它的关键词索引（派生）
--     memory_atom_vec       它的向量索引（派生）
--     memory_atom_cursor    每个会话的增量水位 —— 也是权威状态
--
-- ⚠️ "删干净重建是安全的"这句话**不适用于 memory_atom**
-- ---------------------------------------------------
-- 上面 L0 那两张表是纯派生的（从 agent_message 重建即可），所以本文件
-- 末尾那段"重建脚本"的口径是对的。L1 不同：memory_atom 装的是 LLM 的
-- 结论，从同一批消息重跑一次会得到**不同**的记忆。能重建的只有它的两张
-- 挂载表（源是本表）。别照搬。
--
-- ⚠️ 挂载表**只放召回要用的列**
-- ------------------------
-- 原规格（见 memory_l1.md）把主表的每一列都冗到 memory_atom_fts 上并标 UNINDEXED。
-- 那是 SQLite 的 necessity（FTS5 虚表只能自带列）。本仓库的做法与 L0
-- 一致：只留过滤键 + 正文，其余按 record_id 回 memory_atom 读（主键查）。
-- 所以 type / priority / scene_name / version / 三个时间戳 / meta 都不在
-- 挂载表上 —— 尤其是 priority，它明确不参与排序与过滤，冗余进来早晚会
-- 有人拿它做 ORDER BY。
--
-- ⚠️ 同样**没有外键**，同样两处必须同步
-- --------------------------------
-- 建表实际走 ORM 的 Base.metadata.create_all，它**不读**本文件（见文件
-- 开头那段警告）。下面每一条约束与索引的名字都与 ORM 上逐字一致。
--
-- ⚠️ onupdate=func.now() 在本文件里**没有对应物**
-- -------------------------------------------
-- 它是 SQLAlchemy 在发 UPDATE 时带上 `updated_time = now()`，不是数据库
-- 触发器 —— 手工建库之后用裸 SQL 改数据不会推这个值。写在这是为了让读
-- 到的人不必去翻 ORM 才知道为什么这里少一句。


CREATE TABLE IF NOT EXISTS memory_atom (
    record_id          UUID PRIMARY KEY,

    -- ---- 正文与分类 ------------------------------------------------
    content            TEXT         NOT NULL,
    type               VARCHAR(32)  NOT NULL DEFAULT '',
    -- ⚠️ 不参与排序、不参与过滤（只给 LLM 当阈值 + 工具检索里的展示串）。
    -- 没有 CHECK：词表会长（work_* 四类还没做），见 ORM 的 MEMORY_TYPES。
    priority           INTEGER      NOT NULL DEFAULT 50,
    scene_name         TEXT         NOT NULL DEFAULT '',

    -- ---- 身份与版本 ------------------------------------------------
    session_key        VARCHAR(255) NOT NULL DEFAULT '',
    version            INTEGER      NOT NULL DEFAULT 0,

    -- ---- 时间线三个（**LLM 产出**，可能是 "2018 年" 这类半成品）--------
    timestamp_str      TEXT         NOT NULL DEFAULT '',
    timestamp_start    TEXT         NOT NULL DEFAULT '',
    timestamp_end      TEXT         NOT NULL DEFAULT '',

    -- ---- 系统生成的两个（真时间类型）--------------------------------
    created_time       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_time       TIMESTAMPTZ  NOT NULL DEFAULT now(),

    -- ---- 溯源与自由字段 --------------------------------------------
    source_message_ids JSONB        NOT NULL DEFAULT '[]',
    -- 契约与提示词里叫 metadata；SQLAlchemy 的 Base.metadata 占了这个名字。
    meta               JSONB        NOT NULL DEFAULT '{}',

    -- ---- scope 列（见 models/sqlalchemy/_scope.py）-----------------
    -- 只有 session_id 参与过滤；其余几列是元数据（该列为 NULL 是**允许**
    -- 的状态，拿它们做 WHERE 只会少召回）。
    session_id         UUID         NOT NULL,
    business_agent_id  VARCHAR(128),
    user_id            VARCHAR(255),
    org_id             VARCHAR(255),
    -- ⚠️ 当前恒为空：留给 work_* 四类记忆的预留位（本仓库还没有「工作任务」
    -- 这个概念）。**在它被真正填上之前绝不能进 WHERE** —— 现存行全是 NULL，
    -- 加过滤等于一条都搜不到，而且不报错。
    task_id            VARCHAR(255),

    CONSTRAINT ck_memory_atom_version_non_negative
        CHECK (version >= 0),
    CONSTRAINT ck_memory_atom_content_not_blank
        CHECK (content <> '')
);

-- 增量对账的形状（按 updated_time 取增量）。
CREATE INDEX IF NOT EXISTS idx_memory_atom_updated
    ON memory_atom (updated_time);

-- 去重候选池的形状：按会话收窄。这是它**唯一**的过滤键。
CREATE INDEX IF NOT EXISTS idx_memory_atom_session
    ON memory_atom (session_id);

-- scope=user 的形状（跨会话召回）。
CREATE INDEX IF NOT EXISTS idx_memory_atom_user_updated
    ON memory_atom (user_id, updated_time);


-- ============================================================
-- L1 的关键词索引（派生）
-- ============================================================

CREATE TABLE IF NOT EXISTS memory_atom_fts (
    record_id          UUID PRIMARY KEY,

    -- 分词后的词元串（空格连接，应用侧产出）。不是给人读的。
    content_tokens     TEXT         NOT NULL,
    -- 原文。展示与高亮从这一列出 —— 且必须与 memory_atom.content 逐字相同。
    content_original   TEXT         NOT NULL,
    -- 生成列：两参形式的 to_tsvector 是 IMMUTABLE，一参形式不能用在生成列里。
    tsv                TSVECTOR     GENERATED ALWAYS AS
                                    (to_tsvector('simple'::regconfig,
                                                 content_tokens)) STORED,

    -- ---- scope 列（与 memory_atom 同一组，只有 session_id 参与过滤）----
    session_id         UUID         NOT NULL,
    business_agent_id  VARCHAR(128),
    user_id            VARCHAR(255),
    org_id             VARCHAR(255),
    task_id            VARCHAR(255),

    CONSTRAINT ck_memory_atom_fts_tokens_not_blank
        CHECK (content_tokens <> '')
);

CREATE INDEX IF NOT EXISTS idx_memory_atom_fts_tsv
    ON memory_atom_fts USING GIN (tsv);

CREATE INDEX IF NOT EXISTS idx_memory_atom_fts_session
    ON memory_atom_fts (session_id);

CREATE INDEX IF NOT EXISTS idx_memory_atom_fts_user
    ON memory_atom_fts (user_id);


-- ============================================================
-- L1 的向量索引（派生，需要 pgvector）
-- ============================================================

CREATE TABLE IF NOT EXISTS memory_atom_vec (
    record_id          UUID PRIMARY KEY,

    -- bge-small-zh-v1.5，512 维，已归一化。与 agent_message_vec 同模型、
    -- 同一个进程级 embedder 实例。
    --
    -- ⚠️ 已知代价：content 的语言跟随用户消息，可能是英文 —— 中文模型对
    -- 英文的语义召回会退化，且不报错。缓解在关键词那一路。
    embedding          vector(512)  NOT NULL,

    -- ---- scope 列（同上）-------------------------------------------
    session_id         UUID         NOT NULL,
    business_agent_id  VARCHAR(128),
    user_id            VARCHAR(255),
    org_id             VARCHAR(255),
    task_id            VARCHAR(255)
);

-- vector_cosine_ops 必须与查询里的 `<=>` 一致，否则 Postgres **忽略**这个
-- 索引改用顺序扫描：结果正确、只是慢（不报错的性能坑）。
CREATE INDEX IF NOT EXISTS idx_memory_atom_vec_embedding_hnsw
    ON memory_atom_vec USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memory_atom_vec_session
    ON memory_atom_vec (session_id);

CREATE INDEX IF NOT EXISTS idx_memory_atom_vec_user
    ON memory_atom_vec (user_id);


-- ============================================================
-- L1 的增量水位（**权威状态**，一个会话一行）
-- ============================================================

-- ⚠️ 为什么水位不能从 memory_atom.updated_time 反推
-- --------------------------------------------
-- 原规格是那么写的，但 L1 的 update / merge 是**破坏性**的（多对一合并
-- 要删掉被吃掉的记录）：被删的恰好是最新那条时水位会倒退；被删的行根本
-- 不在表里，"还活着的最大时间戳"不等于"处理到哪了"。
--
-- ⚠️ 推进时机：只在抽取成功并提交**之后**
-- ---------------------------------
-- 提前推进 => 那批消息永久跳过（延时任务读的就是 seq > last_seq）。
--
-- ⚠️ 行是惰性创建的：没有行等价于 last_seq = 0。

CREATE TABLE IF NOT EXISTS memory_atom_cursor (
    -- 一个会话一行。键是 session_id 而不是 user_id —— agent_message.seq
    -- 是**会话内**单调递增的，跨会话没有可比性。
    session_id         UUID PRIMARY KEY,

    -- 已抽过的最大 agent_message.seq。
    last_seq           BIGINT       NOT NULL DEFAULT 0,

    -- 上一个情境（提示词任务一的输入）。放在这里而不是从 memory_atom 读：
    -- 承载它的那条记录可能已经被一次合并删掉了。
    last_scene_name    TEXT         NOT NULL DEFAULT '',

    user_id            VARCHAR(255),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_memory_atom_cursor_last_seq_non_negative
        CHECK (last_seq >= 0)
);

CREATE INDEX IF NOT EXISTS idx_memory_atom_cursor_user
    ON memory_atom_cursor (user_id);


-- ============================================================
-- L2：场景文件（**权威数据**，一行 = 一个 .md）
-- ============================================================
--
-- ⚠️ 没有挂载表，与 L1 的三张不一样
-- -----------------------------
-- L0/L1 都挂了关键词 + 向量两张派生表，因为它们的正文是要被**检索**的。
-- L2/L3 不是：场景是整段拼进提示词的（场景清单 + 全文），当前没有任何
-- 召回入口。要做再加，源就是本表。
--
-- ⚠️ 主键是**内容派生**的 hash，不是复合唯一约束
-- -----------------------------------------
-- 自然的业务键是 (agent_id, user_id|org_id, filename)，但**不能**写成
-- 唯一约束：scope 那两列**恰有一列恒为 NULL**（个人 / 团队二选一），而
-- Postgres 的唯一约束把 NULL 视为互不相同 —— 那个约束一条冲突都挡不住，
-- 却看起来像是在挡。所以键由应用侧算成
--     "scene:v1:" + sha256(scope_key \0 filename)
-- 用主键的唯一性来保证"一个 scope 下一个文件名只有一行"。
--
-- ⚠️ onupdate=func.now() 在本文件里**没有对应物**（同 memory_atom）
-- -------------------------------------------------------------
-- 它是 SQLAlchemy 在发 UPDATE 时带上 `updated_at = now()`，不是数据库
-- 触发器 —— 手工建库之后用裸 SQL 改数据不会推这个值。
--
-- ⚠️ 没有 summary / heat 列
-- ----------------------
-- 它们由模型写在 content 的 -----META-START----- 块里。提到列上会造出
-- 第二个"这个场景讲什么"的答案，而要它俩同步就得每次写入都解析一遍。
-- 场景清单按需解析 content（见 prompts/memory_scene_consolidate.py 的
-- parse_meta —— 它定义在那份模板旁边）。

CREATE TABLE IF NOT EXISTS memory_scene (
    -- 内容派生的稳定身份，见上面两段。75 字符（11 + 64 hex），留到 80。
    id                 VARCHAR(80)  PRIMARY KEY,

    -- ---- 文件本体（整篇 markdown 内联）--------------------------------
    filename           TEXT         NOT NULL,
    content            TEXT         NOT NULL,
    -- 只用来短路写回：与库里一致就跳过写入、也不推 version。
    content_md5        VARCHAR(32)  NOT NULL,

    -- ---- 业务键：agent_id + （user_id 或 org_id，恰好一个）-------------
    -- 取 L1 记忆上的 business_agent_id（duty_key）。NOT NULL 是刻意的
    -- （与 memory_atom 的同名列可空不同）：这里是业务键的一部分。
    agent_id           VARCHAR(128) NOT NULL,
    -- ⚠️ 这两个的 NULL 语义与 memory_atom 上同名的那两列**不同**：
    -- 那边 user_id 是"= 当前用户 OR IS NULL"的可见性口径、org_id 是
    -- 不参与过滤的元数据；这边它们是**业务键的一半**（个人画像按
    -- user 聚合、团队画像按 org 聚合），NULL 表示"这一行属于另一种"。
    -- 别照搬那套 WHERE。
    user_id            VARCHAR(255),
    org_id             VARCHAR(255),

    -- ---- 并发控制 ----------------------------------------------------
    version            INTEGER      NOT NULL DEFAULT 0,

    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_memory_scene_content_not_blank
        CHECK (content <> ''),
    CONSTRAINT ck_memory_scene_version_non_negative
        CHECK (version >= 0),
    -- 个人 / 团队二选一，不许多也不许空。挡的是"scope 解析写错了"这类
    -- bug —— 那种行（比如两列都为 NULL）永远查不出来，且不报错。
    CONSTRAINT ck_memory_scene_scope_exclusive
        CHECK ((user_id IS NULL) <> (org_id IS NULL))
);

-- 两个方向各自一个部分索引：复合索引 (agent_id, user_id, org_id) 对
-- `WHERE agent_id=? AND org_id=?` 用不上（中间那列被跳过），而团队画像
-- 恰恰只按 org 查。
CREATE INDEX IF NOT EXISTS idx_memory_scene_user
    ON memory_scene (agent_id, user_id)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_scene_org
    ON memory_scene (agent_id, org_id)
    WHERE org_id IS NOT NULL;


-- ============================================================
-- L3：用户画像（**权威数据**，一个 scope 一行）
-- ============================================================
--
-- 写入方是 `memory.l3.extract` 那条后台任务（由 L2 完成触发，见
-- application/trigger.py 的 submit_persona_extract），读取方是主模型的
-- 上下文装配（待接）。它的增量水位在同名的 _cursor 表上（见下面 L3 水位段）。
--
-- ⚠️ 列集刻意与 memory_scene **一模一样**（除了 filename 的取值域）
-- ---------------------------------------------------------
-- 两张表装的是同一类东西（一行 = 一篇 markdown），共用
-- `_scope.MemoryFileColumns`。差别只有 filename 与主键前缀（scene:v1: /
-- persona:v1:），于是"将来真要合表"不必重算 id。
--
-- filename 恒为 "persona.md"，主键是
--     "persona:v1:" + sha256(scope_key \0 "persona.md")
-- 与 memory_scene 同一个编码、同一个 kind 段分隔 —— 两张表的主键不会撞。

CREATE TABLE IF NOT EXISTS memory_persona (
    id                 VARCHAR(80)  PRIMARY KEY,

    filename           TEXT         NOT NULL,
    content            TEXT         NOT NULL,
    content_md5        VARCHAR(32)  NOT NULL,

    agent_id           VARCHAR(128) NOT NULL,
    user_id            VARCHAR(255),
    org_id             VARCHAR(255),

    version            INTEGER      NOT NULL DEFAULT 0,

    created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),

    CONSTRAINT ck_memory_persona_content_not_blank
        CHECK (content <> ''),
    CONSTRAINT ck_memory_persona_version_non_negative
        CHECK (version >= 0),
    CONSTRAINT ck_memory_persona_scope_exclusive
        CHECK ((user_id IS NULL) <> (org_id IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_memory_persona_user
    ON memory_persona (agent_id, user_id)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_persona_org
    ON memory_persona (agent_id, org_id)
    WHERE org_id IS NOT NULL;


-- ============================================================
-- L2 的增量水位（**权威状态**，一个会话一行）
-- ============================================================
--
-- ⚠️ 为什么不复用 memory_atom_cursor
-- -------------------------
-- 那张表的键也是 session_id、也是"一个会话一行"，看着像能合并。不能：
--     memory_atom_cursor.last_seq         读的是 **agent_message.seq**（消息序号）
--     memory_scene_cursor.last_atom_*   读的是 **memory_atom.updated_time**（记忆写入时间）
-- 两个水位推进的时机与粒度都不同（L1 一次运行写的若干条 atom 在同一个
-- 事务里、now() 相同），而且一个由 L1 推进、一个由 L2 推进。
--
-- ⚠️ 水位按 updated_time 而不是 record_id
-- -----------------------------------
-- L1 的 update / merge 是**破坏性**的：多对一合并里存活那条保留 id、
-- 被吃掉的几条整行消失。按 id 记水位会随合并失真。updated_time 天然覆盖
-- 这件事 —— 存活者一定被改写、一定前移，于是重新进入下一批增量（内容变
-- 了，画像该重看）。代价：被删掉的记忆不会再出现，画像里由它衍生的内容
-- 只能靠模型这一次整合覆盖掉。
--
-- ⚠️ last_atom_ids 不是冗余
-- ----------------------
-- Postgres 的 now() 取事务开始时刻，L1 一次运行写的所有 atom 因此共享
-- 同一个 updated_time。L2 按 `updated_time > 水位` 读，水位推过去之后，
-- 恰好落在同一微秒的另一批（并发提交）会被 `>` **静默跳过** —— 漏一条
-- 记忆、不报错、也补不回来。所以查询是
--     updated_time > :t OR (updated_time = :t AND record_id NOT IN :ids)
-- 边界上已消费的那批 id 记在这里。
--
-- ⚠️ last_l2_at 为什么在这张表上
-- --------------------------
-- 触发 A 要算 T = max(now + 90s, lastL2 + 900s)，而 profile 行上**没有
-- 会话维度**（一个 scope 一行、被该 scope 下所有会话共享），所以 lastL2
-- 只能按会话记 —— 就记在这一行，与水位同事务更新。
--
-- ⚠️ 推进时机：只在整合成功并提交**之后**
-- ---------------------------------
-- 提前推 => 那批 L1 永远进不了画像（下次按 `> 水位` 取增量），且不报错。
--
-- ⚠️ 行是惰性创建的：没有行等价于"这个会话还没整合过"。

CREATE TABLE IF NOT EXISTS memory_scene_cursor (
    session_id              UUID         PRIMARY KEY,

    -- 已经整合过的 L1 记忆的最大 memory_atom.updated_time。NULL = 还没有
    -- 任何 L1 进过画像（有行但为 NULL 是"L2 跑过、但没有新记忆可整合"）。
    last_atom_updated_time  TIMESTAMPTZ,

    -- 恰好落在上面那个时间戳上、已经消费过的 atom id（字符串形式）。
    last_atom_ids           JSONB        NOT NULL DEFAULT '[]',

    -- 上一次 L2 跑完的时间（含"跑了但没东西可整合"那一次）—— 触发 A 的
    -- lastL2。空转也要戳，否则反复空转的会话会每 90 秒白跑一次。
    last_l2_at              TIMESTAMPTZ,

    user_id                 VARCHAR(255),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_scene_cursor_user
    ON memory_scene_cursor (user_id);


-- ============================================================
-- L3 的增量水位（**权威状态**，一个 scope 一行）
-- ============================================================
--
-- ⚠️ 为什么不复用 memory_scene_cursor
-- -----------------------------
-- 那张表的键是 session_id、这张的键是 scope。两者不是粒度不同，是**维度
-- 不同**：
--     memory_scene_cursor     "这个会话的 L1 记忆整合到哪一条了" —— L2 的增量
--     memory_persona_cursor   "这个 scope 的场景抽象到哪一版了"   —— L3 的增量
-- L2 的产物按 scope 落行、水位按会话记（一个 scope 被 N 个会话共同书写）；
-- 而 L3 的输入 memory_scene **没有 session_id**（见 _scope.MemoryFileColumns）
-- —— 这条链上根本凑不出会话这个维度。
--
-- ⚠️ 主键是 (agent_id, scope_key) 而**不是**三列
-- ----------------------------------------
-- 业务键是 agent_id + (user_id | org_id)，但后两列恰有一个恒为 NULL，而主键
-- 的每一列都必须 NOT NULL。写成 UNIQUE (agent_id, user_id, org_id) 更糟：
-- Postgres 把 NULL 视为互不相同，那个约束一条冲突都挡不住却看起来很严。
-- 所以折成非空的 scope_key（`user:123` / `org:456`，即 SceneScope.key）。
--
-- ⚠️ 场景水位按 updated_at 而不是场景 id
-- ---------------------------------
-- L2 的写入是原地改写 + 真删（[DELETED] 那些被删掉），按 id 记水位会随删除
-- 失真。updated_at 天然覆盖：被改写的场景一定前移（update_scene 里显式写
-- updated_at=now()），于是重新进入下一批增量 —— 内容变了，画像该重看。
-- 代价：**删除不进增量**，画像里由它衍生的内容只能靠模型这一次生成覆盖掉
-- （它看得见现有整篇 persona）。
--
-- ⚠️ last_scene_ids 不是冗余
-- ----------------------
-- 与 memory_scene_cursor.last_atom_ids 同一个成因：一次 L2 整合写的所有场景
-- 共享一个 now()。水位推过去之后，恰好落在同一微秒的另一批会被 `>`
-- 静默跳过。所以查询是
--     updated_at > :t OR (updated_at = :t AND id NOT IN :ids)
--
-- ⚠️ 推进时机：只在画像成功落库并提交**之后**
-- ------------------------------------
-- 提前推 => 那批场景永远进不了画像（下次按 `> 水位` 取增量），且不报错。
-- 与 L1 / L2 那条最硬的约束同源。
--
-- ⚠️ 行是惰性创建的：没有行等价于"这个 scope 还没生成过画像"。

CREATE TABLE IF NOT EXISTS memory_persona_cursor (
    -- 业务键的两半。agent_id 必须是主键的一部分：同一个 user 在不同 duty 下
    -- 是**两份**画像（L2 就是按 (agent, user) 聚合的）。
    agent_id                VARCHAR(128) NOT NULL,
    -- `user:{user_id}` / `org:{org_id}` —— 前缀是必须的，否则某个 user_id
    -- 恰好等于某个 org_id 时两份画像算出同一个键、静默互相覆盖。
    scope_key               VARCHAR(300) NOT NULL,

    -- 两个真实列：能按它们查（下面的部分索引），也能一眼看出这行是团队
    -- 还是个人。与 scope_key 说的是同一件事，由 CHECK 兜底。
    user_id                 VARCHAR(255),
    org_id                  VARCHAR(255),

    -- 已经抽象过的场景的最大 memory_scene.updated_at。NULL = 还没有任何场景
    -- 进过画像（有行但为 NULL 是"L3 跑过、但当时没有变化的场景"）。
    last_scene_updated_time TIMESTAMPTZ,

    -- 恰好落在上面那个时间戳上、已经抽象过的场景 id。
    last_scene_ids          JSONB        NOT NULL DEFAULT '[]',

    -- 上一次 L3 跑完的时间（含"跑了但没东西可更新"那一次）。触发条件 2
    -- （距上次生成 >= 50 轮次）用它当基准 —— 空转也要戳。
    last_l3_at              TIMESTAMPTZ,

    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),

    PRIMARY KEY (agent_id, scope_key),

    CONSTRAINT ck_memory_persona_cursor_scope_exclusive
        CHECK ((user_id IS NULL) <> (org_id IS NULL)),
    CONSTRAINT ck_memory_persona_cursor_scope_key_not_blank
        CHECK (scope_key <> '')
);

-- 两个方向各自一个部分索引，理由与 memory_scene / memory_persona 上那两个
-- 逐字相同：复合索引对 "WHERE agent_id=? AND org_id=?" 用不上。
CREATE INDEX IF NOT EXISTS idx_memory_persona_cursor_user
    ON memory_persona_cursor (agent_id, user_id)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_persona_cursor_org
    ON memory_persona_cursor (agent_id, org_id)
    WHERE org_id IS NOT NULL;


-- ============================================================
-- 改名记录（2026-10-11）：表名/实体类/服务名里不再出现 l1 / l2
-- ============================================================
--
-- ⚠️ 已建好的库要**手工**执行下面这几句；create_all **不会**帮你改
-- ---------------------------------------------------------
-- `create_all` 只做一件事：缺表就建。表改名之后它看到的是"目标名不存在"，
-- 于是**建一张空的新表**，而数据还躺在旧表里 —— 症状是"记忆突然全丢了"，
-- 且不报错。所以顺序是：先手工改名，再启动应用。
--
--     ALTER TABLE memory_fts        RENAME TO memory_atom_fts;
--     ALTER TABLE memory_cursor     RENAME TO memory_atom_cursor;
--     ALTER TABLE memory_profile    RENAME TO memory_scene;
--     ALTER TABLE memory_l2_cursor  RENAME TO memory_scene_cursor;
--
-- 索引与约束名**不跟着表名走**（Postgres 只改表），所以要逐个改，否则
-- 真实库里的名字与 ORM 上声明的不一致 —— 那会让"两处必须逐字一致"这条
-- 纪律失效（.sql 读起来是一套、库里是另一套）。
--
--     ALTER INDEX  memory_fts_pkey        RENAME TO memory_atom_fts_pkey;
--     ALTER INDEX  idx_memory_fts_tsv     RENAME TO idx_memory_atom_fts_tsv;
--     ALTER INDEX  idx_memory_fts_session RENAME TO idx_memory_atom_fts_session;
--     ALTER INDEX  idx_memory_fts_user    RENAME TO idx_memory_atom_fts_user;
--     ALTER TABLE  memory_atom_fts
--         RENAME CONSTRAINT ck_memory_fts_tokens_not_blank
--         TO ck_memory_atom_fts_tokens_not_blank;
--     -- （auto 命名的 *_not_null 约束同理，可留可改：它们不参与语义）
--     ALTER INDEX  memory_cursor_pkey     RENAME TO memory_atom_cursor_pkey;
--     ALTER INDEX  idx_memory_cursor_user RENAME TO idx_memory_atom_cursor_user;
--     ALTER TABLE  memory_atom_cursor
--         RENAME CONSTRAINT ck_memory_cursor_last_seq_non_negative
--         TO ck_memory_atom_cursor_last_seq_non_negative;
--     ALTER INDEX  memory_profile_pkey     RENAME TO memory_scene_pkey;
--     ALTER INDEX  idx_memory_profile_user RENAME TO idx_memory_scene_user;
--     ALTER INDEX  idx_memory_profile_org  RENAME TO idx_memory_scene_org;
--     ALTER TABLE  memory_scene
--         RENAME CONSTRAINT ck_memory_profile_content_not_blank
--         TO ck_memory_scene_content_not_blank;
--     -- （version_non_negative / scope_exclusive 同理）
--     ALTER INDEX  memory_l2_cursor_pkey     RENAME TO memory_scene_cursor_pkey;
--     ALTER INDEX  idx_memory_l2_cursor_user RENAME TO idx_memory_scene_cursor_user;
--
-- ⚠️ 主键编码也换了
-- ------------
-- L2/L3 那两行的 `id` 从 `profile:v1:sha256(scope \0 kind \0 文件名)` 改成
--     scene:v1:sha256(scope \0 文件名)          （L2）
--     persona:v1:sha256(scope \0 "persona.md")  （L3）
-- 本地这两张表当时是空的，所以没有数据要迁。**别处若已有行，必须重算 id**
-- （或用 UPDATE 把前缀替换掉 —— 但后缀的摘要也是旧的，所以本质上要重算）。
--
-- ⚠️ 没有动的两处：handler 名与 model_role
-- ----------------------------------
-- `memory.l0.extract` / `memory.l1.extract` / `memory.l2.extract` 这三个
-- **handler 字符串**保持不变（它们是**持久化数据**，存在 tasks 表里；
-- 改名 = 让库里已有的任务行解析失败，见 app/services/memory/domain/tasks.py
-- 的说明）。model_role **不在库里**，所以跟着新命名走了：
--     memory.atom.extract / memory.atom.dedup   （L1，原 memory.l1.*）
--     memory.scene.consolidate                  （L2）
