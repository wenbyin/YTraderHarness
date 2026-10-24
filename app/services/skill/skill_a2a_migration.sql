-- 把已存在的 skill 表补齐到 A2A AgentSkill 的字段。
--
-- 本仓没有 Alembic，create_all() 也不会修改已存在的表：
-- 对于在补字段之前就建好的库，必须手工跑这一份。
-- 新建的库由 SQLAlchemy 模型直接建出同样的列，不需要本文件。
--
-- 单条 skill 的 A2A 字段只有两个：
--
--   tags      必填（可为空数组）  TEXT[]
--   examples  可选                JSONB（字符串数组）
--
-- inputModes / outputModes / securityRequirements 不属于 skill：
-- 它们是 AgentCard / agent 运行时的字段。
--
-- instructions 也不在其中：它是内部提示词，永远不进 AgentCard。

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS tags
        TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS examples
        JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE skill
    DROP CONSTRAINT IF EXISTS ck_skill_examples_array;

ALTER TABLE skill
    ADD CONSTRAINT ck_skill_examples_array
        CHECK (jsonb_typeof(examples) = 'array');

-- 如果之前误加过 agent 级字段，顺手清掉（幂等）。
ALTER TABLE skill
    DROP COLUMN IF EXISTS input_modes;

ALTER TABLE skill
    DROP COLUMN IF EXISTS output_modes;
