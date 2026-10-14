CREATE TABLE tool_definition (
    id                  UUID PRIMARY KEY,

    -- Tool 的稳定业务标识
    tool_key            VARCHAR(255) NOT NULL UNIQUE,

    -- 代码注册得到
    name                VARCHAR(255) NOT NULL,

    -- local_method / mcp / openapi / ...
    source_type         VARCHAR(32) NOT NULL,

    -- 来源对象，例如 service/module/mcp-server/openapi spec
    source_ref          VARCHAR(500),

    -- 实际方法入口，例如 python.module:Class.method
    entrypoint          VARCHAR(500),

    -- 代码产生的输入 / 输出 schema
    input_schema        JSONB NOT NULL,
    output_schema       JSONB,

    -- 代码 docstring 的原始/解析结果
    source_docstring    JSONB,

    -- 用于判断代码定义是否发生变化
    manifest_hash       VARCHAR(64) NOT NULL,

    -- ACTIVE / MISSING
    source_state        VARCHAR(16) NOT NULL DEFAULT 'ACTIVE',

    discovered_at       TIMESTAMPTZ NOT NULL,
    last_synced_at      TIMESTAMPTZ NOT NULL,

    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_tool_definition_source_type
    ON tool_definition(source_type);

CREATE INDEX idx_tool_definition_source_state
    ON tool_definition(source_state);


CREATE TABLE tool_documentation (
    tool_id                     UUID PRIMARY KEY
                                REFERENCES tool_definition(id)
                                ON DELETE CASCADE,

    -- Tool 整体描述
    description_override        TEXT,

    -- 返回值描述
    return_description_override TEXT,

    updated_by                  VARCHAR(255),
    updated_at                  TIMESTAMPTZ NOT NULL
);


CREATE TABLE tool_parameter_documentation (
    id                  UUID PRIMARY KEY,

    tool_id             UUID NOT NULL
                        REFERENCES tool_definition(id)
                        ON DELETE CASCADE,

    -- 建议使用 JSON Pointer
    -- 例如：
    -- /query
    -- /top_k
    -- /filters/date_from
    parameter_path      VARCHAR(500) NOT NULL,

    description_override TEXT,

    updated_by          VARCHAR(255),
    updated_at          TIMESTAMPTZ NOT NULL,

    UNIQUE(tool_id, parameter_path)
);

CREATE INDEX idx_tool_parameter_doc_tool
    ON tool_parameter_documentation(tool_id);

-- 能力树（capability 表）不在这里
-- --------------------------------
-- 它原本叫 tool_classification、与本文件同处一个模块，现已改名
-- capability 并拆到独立模块：见 app/services/capability/models.sql。
--
-- 下面这张关联表**留在本模块**：它的主语是工具，删工具就该顺带
-- 解除关联（ON DELETE CASCADE）。列名 classification_id 也保持
-- 不变 —— 改名的是被引用的表，不是这条关联。

CREATE TABLE tool_definition_classification (
    tool_id             UUID NOT NULL
                        REFERENCES tool_definition(id)
                        ON DELETE CASCADE,

    classification_id   UUID NOT NULL
                        REFERENCES capability(id)
                        ON DELETE CASCADE,

    PRIMARY KEY(tool_id, classification_id)
);

CREATE INDEX idx_tool_classification_relation_classification
    ON tool_definition_classification(classification_id);


-- ============================================================
-- 工具执行记录：幂等判重与审计
--
-- ⚠️ create_all 不读本文件，约束必须同时写在 ORM 的 __table_args__
--    （见 models/sqlalchemy/tool_execution.py）。本文件是给人看的
--    契约 + 手工建库脚本，两处必须同步。
--
-- ⚠️ 刻意**没有** run_id 的外键：本表要在 agent_run 的生命周期之外
--    活着。级联删除会删掉「副作用发生过」的证据，NO ACTION 会让一条
--    审计记录把 run 的清理卡住 —— 两者对执行账本都是坏的。
-- ============================================================

CREATE TABLE tool_execution (
    id                  UUID PRIMARY KEY,

    -- 哪条会话、哪一次运行
    -- ⚠️ session_id 是 agent_session.session_id（会话行主键），
    --    不是 langgraph 的 thread_id —— 后者是 langgraph 内部键，
    --    而会话详情页 / 消息 / 运行列表全按前者关联。
    session_id          UUID NOT NULL,
    run_id              UUID NOT NULL,

    -- 调了什么，参数是什么（sha256 摘要，不是原文）
    tool_key            VARCHAR(255) NOT NULL,
    param_hash          VARCHAR(64) NOT NULL,

    -- PENDING / RUNNING / SUCCESS / FAILED / UNKNOWN /
    -- CANCELLED / REJECTED / INTERRUPT
    -- 每个取值的「能不能重跑」见 domain/schemas/enum.py
    status              VARCHAR(16) NOT NULL DEFAULT 'PENDING',

    -- 返回值（规范化后的 JSON 形态）。⚠️ 是**脱敏之后**的那一份：
    -- 写库发生在处理器链跑完之后，privacy_transformer 已生效。
    -- 可空列必须带 none_as_null —— 见 ORM 的说明。
    result              JSONB,

    -- {"type": 异常类名, "message": 给模型看的文案}
    -- 完整堆栈只进日志，不入库。
    error               JSONB,

    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,

    CONSTRAINT ck_tool_execution_status
        CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED',
                          'UNKNOWN', 'CANCELLED', 'REJECTED',
                          'INTERRUPT')),

    CONSTRAINT ck_tool_execution_error_object
        CHECK (error IS NULL OR jsonb_typeof(error) = 'object')
);

-- 判重与抢锁都落在这条上：并发同键靠 INSERT ... ON CONFLICT 串行化，
-- 不是先查再插（那样两个并发分支会都执行）。
CREATE UNIQUE INDEX uq_tool_execution_identity
    ON tool_execution(session_id, run_id, tool_key, param_hash);

-- 运行详情页：这一次运行调了哪些工具
CREATE INDEX idx_tool_execution_run_created
    ON tool_execution(run_id, created_at);

-- 按工具查历史执行
CREATE INDEX idx_tool_execution_tool_key
    ON tool_execution(tool_key, created_at);

-- 只覆盖「没有走到终态的行」：崩溃遗留的、以及将来重试扫描要看的
CREATE INDEX idx_tool_execution_unsettled
    ON tool_execution(created_at)
    WHERE status IN ('PENDING', 'RUNNING');
