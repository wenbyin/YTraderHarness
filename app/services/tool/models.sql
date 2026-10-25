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


-- ============================================================
-- 运行时开关：有行 = 显式禁用，无行 = 启用
--
-- ⚠️ **没有** enabled 列：插入即禁用，删除即恢复。"只过滤显式
--    声明禁用"因此是表结构本身保证的。
--
-- ⚠️ **没有**外键：remove_missing=true 会物理删除 tool_definition
--    的行；ON DELETE CASCADE 会让一次同步悄悄清掉人工设的禁用。
--    键是稳定的 tool_key，同名工具回来时仍保持禁用（fail-closed）。
--    代价是可能留下指向 MISSING / 已删工具的孤儿行。
-- ============================================================

CREATE TABLE tool_definition_state (
    tool_key       VARCHAR(255) PRIMARY KEY,
    disabled_by    VARCHAR(255),
    disabled_at    TIMESTAMPTZ NOT NULL
);


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
-- 工具执行尝试：守卫计数与审计（一次执行一行，append-only）
--
-- ⚠️ create_all 不读本文件，约束必须同时写在 ORM 的 __table_args__
--    （见 models/sqlalchemy/tool_execution.py）。本文件是给人看的
--    契约 + 手工建库脚本，两处必须同步。
--
-- ⚠️ 沿用 tool_execution 表名，在原表上扩展：
--    删掉旧的 UNIQUE(session_id, run_id, tool_key, param_hash)，
--    增加 scope / execution / counted / tool_call_id 等列，用
--    UNIQUE(run_id, tool_call_id) 做 replay/resume 去重。
--    迁移脚本见 app/services/tool/tool_execution_migration.sql，
--    需**手工执行** —— create_all 只建缺失的表，不改已存在的表。
--
-- ⚠️ 刻意**没有** run_id 的外键：本表要在 agent_run 的生命周期之外
--    活着。级联删除会删掉「副作用发生过」的证据，NO ACTION 会让一条
--    审计记录把 run 的清理卡住 —— 两者对执行账本都是坏的。
-- ============================================================

CREATE TABLE tool_execution (
    id                  UUID PRIMARY KEY,

    -- 调了什么
    tool_key            VARCHAR(255) NOT NULL,
    tool_id             UUID,

    -- 哪条会话、哪一次运行、谁发起的
    -- ⚠️ session_id 是 agent_session.session_id（会话行主键），
    --    不是 langgraph 的 thread_id。
    session_id          UUID,
    run_id              UUID,
    user_id             VARCHAR(255),

    -- 守卫的维度：scope + execution + 参数摘要
    scope_type          VARCHAR(16) NOT NULL,
    scope_key           VARCHAR(255) NOT NULL,
    execution_type      VARCHAR(16) NOT NULL,
    param_hash          VARCHAR(64) NOT NULL DEFAULT '',
    match_key           VARCHAR(64) NOT NULL,

    -- PENDING / RUNNING / SUCCESS / FAILED / UNKNOWN /
    -- BLOCKED / CANCELLED / REJECTED / INTERRUPT
    status              VARCHAR(16) NOT NULL DEFAULT 'PENDING',

    -- 是否占用额度。拒绝 / 中断 / 取消 / 守卫拦下为 FALSE。
    counted             BOOLEAN NOT NULL DEFAULT TRUE,

    -- 模型给的调用 id；与 run_id 一起做 replay/resume 去重。
    tool_call_id        VARCHAR(255),

    -- 返回值（规范化后的 JSON 形态）。⚠️ 是**脱敏之后**的那一份：
    -- 写库发生在处理器链跑完之后，privacy_transformer 已生效。
    result              JSONB,

    -- {"type": 异常类名, "message": 给模型看的文案}
    -- 完整堆栈只进日志，不入库。
    error               JSONB,

    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,

    CONSTRAINT ck_tool_execution_status
        CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED',
                          'UNKNOWN', 'BLOCKED', 'CANCELLED', 'REJECTED',
                          'INTERRUPT')),

    CONSTRAINT ck_tool_execution_scope
        CHECK (scope_type IN ('SESSION', 'RUN', 'USER', 'GLOBAL')),

    CONSTRAINT ck_tool_execution_execution
        CHECK (execution_type IN ('METHOD', 'INVOKE')),

    CONSTRAINT ck_tool_execution_error_object
        CHECK (error IS NULL OR jsonb_typeof(error) = 'object')
);

-- replay / resume 去重：同一次模型调用在图中重跑时复用同一行。
-- 部分索引：没有 tool_call_id 的图外直调不参与去重。
CREATE UNIQUE INDEX uq_tool_execution_call_id
    ON tool_execution(run_id, tool_call_id)
    WHERE tool_call_id IS NOT NULL AND run_id IS NOT NULL;

-- 守卫计数：按维度 + 时间窗口。只覆盖 counted 的行。
CREATE INDEX idx_tool_execution_match
    ON tool_execution(match_key, created_at)
    WHERE counted;

-- 运行详情页：这一次运行调了哪些工具
CREATE INDEX idx_tool_execution_run_created
    ON tool_execution(run_id, created_at);

-- 按工具查历史执行
CREATE INDEX idx_tool_execution_tool_key
    ON tool_execution(tool_key, created_at);

-- 只覆盖「没有走到终态的行」：崩溃遗留的、以及将来 reaper 要看的
CREATE INDEX idx_tool_execution_unsettled
    ON tool_execution(created_at)
    WHERE status IN ('PENDING', 'RUNNING');
