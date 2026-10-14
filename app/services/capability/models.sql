-- 业务能力树（taxonomy）。
--
-- 历史上这张表叫 tool_classification，与工具的定义表同处
-- app/services/tool/models.sql。改名 capability 的同时它也被拆成了
-- 独立模块：它是一棵独立的树，只是**被**工具引用。
--
-- 工具 ↔ 能力的关联表不在这里，仍然是 tool 模块的
-- tool_definition_classification —— 那条记录的主语是工具，
-- 删工具就该顺带解除关联。

CREATE TABLE capability (
    id              UUID PRIMARY KEY,

    code            VARCHAR(255) NOT NULL UNIQUE,

    name            VARCHAR(255) NOT NULL,

    description     TEXT,

    parent_id       UUID
                    REFERENCES capability(id),

    sort_order      INTEGER NOT NULL DEFAULT 0,

    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_capability_parent
    ON capability(parent_id);

CREATE INDEX idx_capability_active
    ON capability(is_active);


-- ============================================================
-- 从 tool_classification 改名而来时，必须手动执行下面这条
-- ============================================================
--
-- initializer 用的是 Base.metadata.create_all()，它是**幂等但不会
-- 改名**的：表存在就跳过，不存在就新建。所以对已有库直接启动，
-- 结果是新建一张空的 capability，而数据仍留在 tool_classification
-- 里 —— 接口看起来正常，只是全空。
--
--   ALTER TABLE tool_classification RENAME TO capability;
--
-- 外键不用跟着改：PostgreSQL 的 RENAME 会自动更新引用它的约束，
-- 关联表 tool_definition_classification 的 classification_id 仍然
-- 指向这张表（列名保持 classification_id 不变 —— 关联表这次不动）。
--
-- 两个索引名（idx_tool_classification_parent / _active）会跟着表走
-- 但名字还是旧的。想对齐上面的新名字，另跑：
--
--   ALTER INDEX idx_tool_classification_parent RENAME TO idx_capability_parent;
--   ALTER INDEX idx_tool_classification_active RENAME TO idx_capability_active;
