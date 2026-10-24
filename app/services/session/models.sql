-- Agent 运行时的会话与运行。
--
-- 与 document 模块的对应关系：
--     agent_session  ~  document_version（长期存在的主体）
--     agent_run      ~  document_pipeline_run（一次执行，追加型历史）
--
-- ⚠️ 表名带 agent_ 前缀，不叫裸的 session —— 那个词在本仓库里已经被
-- 数据库会话（AsyncSession / session_factory）占满了，见 __init__.py。
-- 消息表也跟着这个前缀（agent_message，见下面第 6 条）。

CREATE TABLE agent_session (
    session_id      UUID PRIMARY KEY,

    -- 这个会话属于哪种 Agent。取值由 Agent 目录决定（当前还没有那个
    -- 目录，见 domain/schemas/enum.py 的说明），因此这里刻意不加 CHECK：
    -- 一个把取值钉死在数据库里的 CHECK，会在新增 Agent 时要求一次迁移。
    agent_id        VARCHAR(255) NOT NULL,

    -- active / archived
    status          VARCHAR(16) NOT NULL,

    -- ⚠️ 列名叫 meta 而不是 metadata：SQLAlchemy 的声明式基类把
    -- `metadata` 这个**属性名**留给了 MetaData 对象，同名属性会直接
    -- 报 InvalidRequestError。对外契约里仍然是 metadata（见 session.yaml）。
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- LangGraph checkpointer 里的线程键，全库唯一。
    -- 首轮运行时生成（形如 dev:react:<uuid>，见 infra/graph/ids.py 的
    -- new_session_thread_id），此后不再变 —— 同一会话的多轮历史靠它累积。
    -- 可空是为了不要求回填已有会话：第一次发运行时才生成。
    thread_id       VARCHAR(255),

    -- 会话归属的用户 ID —— Casdoor 的 sub（app/common/auth），字符串。
    -- 两种过滤口径：
    --   按 id 读      `user_id = 当前用户 OR user_id IS NULL`
    --   列表 / 删除   `user_id = 当前用户`（严格）
    -- NULL 是刻意保留的存量语义（加这列之前的会话对谁都可见），
    -- 但无主会话不出现在任何人的列表里、也不能被删除。
    -- 不代表可以省略 —— 新建会话必须写入。
    user_id         VARCHAR(255),

    -- 会话所属的**组织** —— JWT 的 org claim（claim 名可配，见
    -- app/common/auth 的 org_claim）。
    --
    -- 它只是"团队 / 组织归属"的**记录**：既不参与鉴权（那是 groups 的
    -- 事），也不参与可见性过滤（过滤键仍然是上面的 user_id）。
    -- NULL 表示这次登录的 token 里没有组织信息 —— 这是**允许**的结果
    -- （Casdoor 各版本与各应用的 Token fields 配置不一样），不是漏写。
    org_id          VARCHAR(255),

    -- 预留位：这个会话是为哪一件**工作任务**服务的。
    -- ⚠️ 当前恒为空（没有任何地方写它），见 ORM 上那一列的注释。
    -- ⚠️ 别与 tasks.task_id（后台任务主键）混：同名不同物。
    task_id         VARCHAR(255),

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_agent_session_agent
    ON agent_session(agent_id);

CREATE INDEX idx_agent_session_status
    ON agent_session(status);

-- 唯一：thread_id 是 checkpointer 的全局键，两个会话共用一条会互相续跑
-- 对方的对话。NULL 不参与唯一性（Postgres 语义），因此未生成的行不受影响。
CREATE UNIQUE INDEX uq_agent_session_thread
    ON agent_session(thread_id);


CREATE TABLE agent_run (
    run_id          UUID PRIMARY KEY,

    -- 主体。删会话就删它的全部运行 —— 运行没有独立于会话的存在理由。
    session_id      UUID NOT NULL
                    REFERENCES agent_session(session_id)
                    ON DELETE CASCADE,

    -- queued / running / waiting_input / completed / failed / cancelled
    -- 这一组是**对外契约**的取值（见 session.yaml 的 Run.status）。
    -- app/infra/graph/recorder.py 的 RunStatus 是**观测层**的另一套
    -- （pending / running / completed / failed / partial / cancelled），
    -- 两者的映射在接入派发时做，见 domain/schemas/enum.py。
    status          VARCHAR(16) NOT NULL,

    -- 用户这次的输入载荷（CreateRunRequest.input）。
    -- 形状由规格定死：{"text": "..."}，见 session.yaml 的 RunInput 段。
    input           JSONB NOT NULL,

    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 失败时的原因。规格的 Run **不回**这两个字段（只有 status），
    -- 因此排查一次失败的运行只能从这里看 —— 这就是它们落库的理由。
    -- 未结束或成功时是 NULL。
    error_type      VARCHAR(128),
    error_message   TEXT,

    -- 发起这次运行的用户 ID（Casdoor 的 sub）。注意是**发起人**，不是所属
    -- 会话的 user_id —— 存量公共会话上仍然能记录真实执行者。派发时它会被
    -- 写进 ReAct 图初始 state 的 user_id（见 agent_run_dispatch._initial_state）。
    user_id         VARCHAR(255),

    -- 发起运行时调用者的**身份快照**（{"user_id","groups","roles"}）。
    --
    -- 为什么要在这一刻存下来：agent 运行是异步派发的（请求提交后才跑），
    -- 跑的时候手里没有 token，而 Casdoor 是用户的唯一真相、本地没有用户表。
    -- 不存这一份，工具级鉴权（checkauth 拦截器）在派发时就没有依据。
    -- 形状见 app/agent/tools/processor/context.py 的 build_caller_context。
    --
    -- ⚠️ 内部字段，**不进对外契约** —— 装配 Run DTO 时与 input / meta
    -- 一样丢掉它。
    caller_identity JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 这次运行用哪份**职责提示词**（duty_key）。
    --
    -- 建运行那一步解析一次（见 agent_run_service_impl.create_run 与
    -- app/services/prompt 的 default_duty_key），此后这一轮里不再变：
    -- 派发时它进图 state（builder 节点据此取提示词），写消息时它填进
    -- agent_message.business_agent_id。
    --
    -- 之所以存下来而不是各处理解一次：同一轮运行里"用的哪份职责"必须
    -- 只有一个答案，否则运行行、提示词、消息三处可能对不上。
    duty_key        VARCHAR(128),

    -- 客户端传入的 Trace 聚合键。NULL = 这次运行未开启 Trace。
    -- trace 事件的 Redis 键是 trace:<trace_id>:events；事件里另带 run_id，
    -- 因为 trace 可以跨 run / session。
    trace_id        VARCHAR(255),

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

-- ⚠️ 仓库没有迁移工具：initialize_database 用的是 create_all，
-- 它只建缺失的表、不改已存在的表。下面这些列是后加的，**已有的库**
-- 必须手动补：
--
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         error_type VARCHAR(128);
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         error_message TEXT;
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         user_id VARCHAR(255);
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         caller_identity JSONB NOT NULL DEFAULT '{}'::jsonb;
--     ALTER TABLE agent_session ADD COLUMN IF NOT EXISTS
--         user_id VARCHAR(255);
--     ALTER TABLE agent_session ADD COLUMN IF NOT EXISTS
--         org_id VARCHAR(255);
--     ALTER TABLE agent_session ADD COLUMN IF NOT EXISTS
--         task_id VARCHAR(255);
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         duty_key VARCHAR(128);
--     ALTER TABLE agent_run ADD COLUMN IF NOT EXISTS
--         trace_id VARCHAR(255);
--
-- ⚠️⚠️ 上面这几条（org_id / task_id / duty_key）也必须在**部署新代码之前**跑
-- --------------------------------------------------------------------
-- 理由与下面 thread_id 那条完全相同：ORM 一旦声明了字段，查询就会把它
-- 带进 SELECT / INSERT。列不存在时不是"新功能不可用"，而是**整个会话与
-- 运行接口一起 500**（列不存在 -> UndefinedColumn）。顺序只能是：
-- 先跑 ALTER，再上新代码。
--
-- 不需要回填：几列都可空，且都没有"历史值"可言 —— org_id 是登录时才有
-- 的组织信息，task_id 目前**没有任何地方写它**（预留位），duty_key 是
-- 那一轮运行的职责（存量运行没有它，读出来是 NULL，主循环会回落到
-- 默认职责）。
--
-- ⚠️ file.uploaded_by 是**换类型**（UUID -> VARCHAR(255)），不是加列。
-- 本文件不管 file 表，记在这里是因为它属于同一批改动（file 模块没有
-- models.sql，DDL 也写在它的 ORM 类 docstring 里）：
--
--     ALTER TABLE file
--         ALTER COLUMN uploaded_by TYPE VARCHAR(255)
--         USING uploaded_by::text;   -- 现存值全为 NULL，空操作
--
-- ⚠️⚠️ agent_session.thread_id 必须在**部署新代码之前**先加
-- ------------------------------------------------------------------
-- ORM 一旦声明了这个字段，每一次会话查询（含 GET /sessions/{id}、
-- create_run 开头的存在性校验）都会把它带进 SELECT。列不存在时
-- 那不是"新功能不可用"，而是**所有会话接口一起 500**。
-- 所以顺序只能是：先跑下面两句，再上新代码。
--
--     ALTER TABLE agent_session ADD COLUMN IF NOT EXISTS
--         thread_id VARCHAR(255);
--     CREATE UNIQUE INDEX IF NOT EXISTS uq_agent_session_thread
--         ON agent_session(thread_id);
--
-- 不需要回填：可空，第一轮运行时才生成（见 agent_run_service_impl.py）。
-- user_id 同样不回填 —— NULL 就是「存量对谁都可见」那个语义。

-- 按会话列运行历史（最常用的查询：某个会话发生过什么，新的在前）。
CREATE INDEX idx_agent_run_session
    ON agent_run(session_id, created_at DESC);




-- 会话消息。写完即不可变（正文不再改）：两个时间戳因此恒等。
--
-- ⚠️ 建表实际走的是 ORM，不是本文件
-- ---------------------------------
-- `initialize_database` 用的是 `Base.metadata.create_all`
-- （见 infra/datasource/initializer.py），它只认 ORM 上声明的东西。
-- 因此下面的 CHECK 与索引在 ORM 上**也有一份**
-- （models/sqlalchemy/agent_message.py 的 __table_args__），两处必须同步。
-- 只改这里 = 真实建出来的表没有那些约束，而本文件读起来像是有的。
--
-- ⚠️ 本表是新的，建表不需要手工 ALTER —— 但 updated_at 这一列需要
-- ---------------------------------------------------------------
-- create_all 会直接把它建出来（与上面 agent_run 后加的列不同）。
--
-- updated_at 是**后来加的列**，已有库必须先补、再上新代码：
--
--     ALTER TABLE agent_message ADD COLUMN IF NOT EXISTS
--         updated_at TIMESTAMPTZ;
--     UPDATE agent_message SET updated_at = created_at
--         WHERE updated_at IS NULL;
--     ALTER TABLE agent_message ALTER COLUMN updated_at SET NOT NULL;
--     ALTER TABLE agent_message ALTER COLUMN updated_at SET DEFAULT NOW();
--
-- 回填那句不能省：存量行读出来是 NULL，而实体上 updated_at 是非可选的
-- datetime —— 症状是读历史直接 500。最后那句 DEFAULT 让不写这一列的
-- 裸 SQL 插入（如 message_smoke 里那几条）不被 NOT NULL 拒掉 ——
-- 否则它们会因为**错误的理由**通过"约束拦住了坏数据"那些用例。
-- reasoning_content 也是后加的列，但**不需要回填**（可空列没有
-- "存量行是 NULL 而实体非可选"那个问题），因此只有一条：
--
--     ALTER TABLE agent_message ADD COLUMN IF NOT EXISTS
--         reasoning_content TEXT;
--
-- ⚠️ 不要顺手"补历史"：那句 UPDATE 会碰到 updated_at 的 onupdate，
-- 于是上面"两个时间戳恒等"变成假话。历史那段思维链仍在 checkpoint 的
-- blob 里（原始 AIMessage 随图 state 序列化），刻意不搬 —— 见
-- models/sqlalchemy/agent_message.py 的类 docstring。
--
-- 但若库里已经存在一版 messages（旧表名）或**带外键的**同名表，
-- create_all 不会改它、也不会改名为 agent_message —— 它只看表存不存在。
-- 那时会多出一张没人读的空表 agent_message，而数据还留在 messages 里；
-- 有数据要保的话必须手工 `ALTER TABLE messages RENAME TO agent_message`
-- （约束与索引名也一并改，见下），否则先 DROP 再让应用重建。
--
-- 相对设计原文的五处偏离（都是落库时才发现必须处理的）：
--
--   1. 外键全部去掉。原文的 `REFERENCES sessions(id)` / `runs(id)` 指向的
--      表在本仓库不存在（实际是 agent_session / agent_run，本文件开头自己
--      解释了为什么带 agent_ 前缀）—— 照抄根本建不起来。代价：没有引用
--      完整性，删会话不会连带清掉消息（今天没有删除端点），run_id /
--      reply_to_message_id 可能出现悬空值，容忍。
--
--   2. metadata -> meta（列名与 ORM 属性都改）。SQLAlchemy 的声明式基类把
--      `metadata` 这个**属性名**留给了 MetaData 对象，同名直接
--      InvalidRequestError。本模块另外两张表已经是这个叫法。对外契约里
--      仍然叫 metadata（见 messages.yaml），改名只发生在契约边界。
--
--   3. 主键列名保留 id，ORM 属性名用 message_id。与 capability 表一致
--      （列 "id"、属性 capability_id），同时对齐契约字段。
--
--   4. seq 允许有空洞。一次写入失败会留洞，因此"递增"不等于"连续"——
--      翻页按 keyset 做，不按 COUNT 推。
--
--   5. client_message_id 今天**永远为 NULL**：要等将来的
--      POST /sessions/{id}/messages。
--      business_agent_id **已经不是**这种状态了：它装的是产出这条消息的
--      职责（这一轮运行用的 duty_key，见 agent_run.duty_key）。列名是当初
--      预留业务 Agent 时取的，没跟着改 —— 改名要动对外契约
--      （messages.yaml 里它已经是响应字段）。
--      列与索引先建着，是因为它们是表设计的一部分，删了将来要补一次迁移。
--
--   6. 表名由 messages 改成 agent_message（约束与索引名同步带前缀）。
--      设计原文与 messages.yaml 里的组件都叫 Message / messages，这是一次
--      **刻意的改名**：同属一个模块的三张表用同一套命名。改的只是库表与
--      Python 类名，**对外契约一个字没动** —— 生成的 OpenAPI 组件名由
--      DTO 的 title 与 app/common/openapi.py 的改名表钉回 Message
--      （见 domain/schemas/agent_message.py）。

CREATE TABLE agent_message (
    id UUID PRIMARY KEY,

    -- 无外键，见上面第 1 条。这三列都只是"指向谁"的记录。
    session_id UUID NOT NULL,
    run_id UUID NULL,

    -- 会话内单调递增的消息序号
    seq BIGINT NOT NULL CHECK (seq > 0),

    role VARCHAR(16) NOT NULL
        CHECK (role IN ('user', 'assistant')),

    kind VARCHAR(24) NOT NULL DEFAULT 'chat'
        CHECK (kind IN (
            'chat',
            'interaction',
            'tool_activity',
            'notice',
            'task_instruction',
            'task_result'
        )),

    business_agent_id VARCHAR(128) NULL,

    -- 完整消息内容，一次性写入
    content JSONB NOT NULL
        CHECK (jsonb_typeof(content) = 'array'),

    -- 模型的思维链。后加的列，可空、不回填，见上面那段。
    -- 只装可见正文之外的那一段：content 里不会再出现推理块。
    reasoning_content TEXT NULL,

    -- 仅表示消息最终结果，不存在 streaming 状态
    status VARCHAR(24) NOT NULL DEFAULT 'completed'
        CHECK (status IN (
            'completed',
            'interrupted',
            'failed',
            'cancelled'
        )),

    -- 用户提交消息的幂等 ID。当前永远是 NULL，见上面第 5 条。
    client_message_id VARCHAR(128) NULL,

    -- 回复关联。无外键，见上面第 1 条。
    reply_to_message_id UUID NULL,

    -- 扩展元数据。列名 meta 而不是 metadata，见上面第 2 条。
    meta JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(meta) = 'object'),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- 后加的列，已有库要手动补 —— 见下面那段。
    -- 当前恒等于 created_at：消息写一次就不再改。
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_agent_message_session_seq
        UNIQUE (session_id, seq)
);

CREATE INDEX idx_agent_message_session_seq
    ON agent_message (session_id, seq DESC);

CREATE INDEX idx_agent_message_run_seq
    ON agent_message (run_id, seq)
    WHERE run_id IS NOT NULL;

CREATE UNIQUE INDEX uq_agent_message_client_message
    ON agent_message (session_id, client_message_id)
    WHERE client_message_id IS NOT NULL;