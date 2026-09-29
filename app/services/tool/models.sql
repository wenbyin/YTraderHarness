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

CREATE TABLE tool_classification (
    id              UUID PRIMARY KEY,

    code            VARCHAR(255) NOT NULL UNIQUE,

    name            VARCHAR(255) NOT NULL,

    description     TEXT,

    parent_id       UUID
                    REFERENCES tool_classification(id),

    sort_order      INTEGER NOT NULL DEFAULT 0,

    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_tool_classification_parent
    ON tool_classification(parent_id);

CREATE INDEX idx_tool_classification_active
    ON tool_classification(is_active);

CREATE TABLE tool_definition_classification (
    tool_id             UUID NOT NULL
                        REFERENCES tool_definition(id)
                        ON DELETE CASCADE,

    classification_id   UUID NOT NULL
                        REFERENCES tool_classification(id)
                        ON DELETE CASCADE,

    PRIMARY KEY(tool_id, classification_id)
);

CREATE INDEX idx_tool_classification_relation_classification
    ON tool_definition_classification(classification_id);
