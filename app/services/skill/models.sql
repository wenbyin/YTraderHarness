-- Skill 管理与能力绑定。
--
-- 实际建表走 SQLAlchemy 的 Base.metadata.create_all()（见
-- app/infra/datasource/initializer.py）；本文件是结构的事实记录，
-- 也是手工建库 / 审阅时用的等价 DDL。
--
-- 工具可见性不直接挂在 skill 上，而是沿能力树推导：
--
--     skill_capability.capability_id
--       -> capability.code
--       -> tool_definition_classification
--       -> tool_definition.tool_key（ACTIVE）
--
-- 明确不含：版本表、分类绑定、token、A2A Agent Card、鉴权。

CREATE TABLE skill (
    id              UUID PRIMARY KEY,

    -- 稳定业务标识，创建后不可改。后续表按它或 id 引用本行。
    skill_key       VARCHAR(120) NOT NULL UNIQUE,

    name            VARCHAR(200) NOT NULL,

    -- 给主 agent 选择 skill 时读。
    description     TEXT NOT NULL,

    -- A2A AgentSkill 元数据。只在发布 AgentCard 时使用；
    -- 内部 subagent 运行不读它们。instructions 永远不进卡片。
    --
    -- 单条 skill 只有这两个 A2A 字段：tags 是**必填**（允许空数组），
    -- examples 可选。inputModes / outputModes / securityRequirements
    -- 属于 AgentCard / agent 运行时，不是 skill 的列。
    tags            TEXT[] NOT NULL DEFAULT '{}',
    examples        JSONB NOT NULL DEFAULT '[]'::jsonb,

    -- 注入子 agent 节点的行为提示词 / SOP。内部字段。
    instructions    TEXT NOT NULL,

    status          VARCHAR(20) NOT NULL DEFAULT 'DRAFT',

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,

    CONSTRAINT ck_skill_key_format
        CHECK (skill_key ~ '^[a-z][a-z0-9_]{1,119}$'),
    CONSTRAINT ck_skill_status
        CHECK (status IN ('DRAFT', 'ACTIVE', 'DISABLED')),
    CONSTRAINT ck_skill_examples_array
        CHECK (jsonb_typeof(examples) = 'array')
);

CREATE INDEX idx_skill_status
    ON skill(status);


-- 绑定只决定工具可见性，不是授权。
--
-- 删除语义不对称：
--   skill 侧       CASCADE —— 删 skill 顺带解除绑定
--   capability 侧  RESTRICT —— 防止 force 删能力时无声抹掉绑定
CREATE TABLE skill_capability (
    skill_id        UUID NOT NULL
                    REFERENCES skill(id) ON DELETE CASCADE,

    capability_id   UUID NOT NULL
                    REFERENCES capability(id) ON DELETE RESTRICT,

    PRIMARY KEY (skill_id, capability_id)
);

CREATE INDEX idx_skill_capability_capability
    ON skill_capability(capability_id);


-- 绑定是**精确匹配**：ToolQueryService.tool_keys_for_capabilities
-- 按 capability.code IN (...) 查，不沿 parent_id 递归。
-- 要覆盖子树就显式绑定多个 capability。
--
-- updated_at 由 ORM 的 onupdate 维护（见 models/sqlalchemy/skill.py），
-- 本仓不给它建库级 trigger —— 与 capability / tool 模块一致。
-- 绕开 ORM 直接改库时，需要自己写 updated_at。
